"""Dynamic toolkit: LLM-generated diagnostic tools as run-scoped specs.

This module provides:

1. **ToolOutputCollector** — the I/O harness injected into the ``exec()``
   sandbox.  Generated tool code calls ``show_plt``, ``show_df``, and
   ``show_json`` to produce output; the collector captures everything for
   the pipeline to route to the VLM.

2. **DynamicToolSpec(Typed)** — a frozen data class describing one
   runtime-generated diagnostic tool (name + description + code + an
   ``execute()`` method).  Unlike the static tool classes
   (``DistributionFittingTool``, ``TimeSeriesStaticTool``) which are
   Morphic ``Registry`` subclasses, ``DynamicToolSpec`` is *data*: every
   ``run()`` call owns its own ``Dict[str, DynamicToolSpec]`` (see
   ``experiment_step_state.RunDeps.dynamic_tools``) so dynamic tools
   never leak across datasets or runs.  When
   ``config.toolkit.accumulate_tools`` is true, the run-level dict is
   seeded from ``tool_registry.json`` at the start of ``run()`` and
   written back at the end; otherwise the dict is always fresh-empty.

3. **execute_dynamic_tool** — the sandbox that runs generated code with
   ``data``, ``map_estimate``, ``family_name``, ``np``, ``plt``, ``pd``,
   ``stats`` in scope, plus the three ``show_*`` harness functions.

4. **GENERATE_NEW_TOOL_SCHEMA** — the OpenAI tool schema the VLM uses
   to request generation of a new diagnostic tool.
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from morphic import Typed, validate
from morphic.string import format_exception_msg
from scipy import stats

from api_repair_diagnostics import build_api_discovery_report
from domains import (
    CodeGenerationAttempt,
    CodeGenerationFailure,
    DiagnosticArtifact,
    DiagnosticToolResult,
    FitState,
)
from experiment_enums import Domain, ToolkitMode
from logging_utils import format_log_block
from processing_utils import unescape_broken_code_if_syntax_error
from sandbox_namespaces import get_tool_runtime_namespace

logger: logging.Logger = logging.getLogger("dynamic_toolkit")


# ===========================================================================
# 1. ToolOutputCollector
# ===========================================================================


class ToolOutputCollector:
    """Captures output from dynamically generated tool code.

    An instance is created per tool execution and its bound methods
    (``show_plt``, ``show_df``, ``show_json``) are injected into the
    ``exec()`` namespace.  After execution, the pipeline reads the
    captured outputs to build the next VLM prompt.

    Args:
        fit_path: File path where the primary plot should be saved.
    """

    def __init__(self, *, fit_path: str) -> None:
        self.fit_path: str = fit_path
        self._plot_counter: int = 0
        self.plots: List[Tuple[str, str]] = []
        self.dataframes: List[Tuple[str, str]] = []
        self.jsons: List[Tuple[str, str]] = []

    def _make_plot_path(self) -> str:
        """Generate a unique save path for each show_plt call.

        First call saves to fit_path directly (the primary image the VLM sees).
        Subsequent calls append a sub-index: fit_path_1.png, fit_path_2.png, etc.
        """
        if self._plot_counter == 0:
            return self.fit_path
        base: str = self.fit_path.rsplit(".", 1)[0]
        ext: str = self.fit_path.rsplit(".", 1)[1] if "." in self.fit_path else "png"
        return f"{base}_{self._plot_counter}.{ext}"

    def show_plt(self, fig: plt.Figure, desc: str) -> None:
        """Save a matplotlib figure to a unique path and record it.

        Each call gets its own file. The first call saves to ``fit_path``
        (the primary image shown to the VLM). Additional calls save to
        ``fit_path`` with a sub-index suffix (e.g., ``_1.png``, ``_2.png``).

        Args:
            fig: The matplotlib Figure object to save.
            desc: A human-readable description of what the plot shows.
        """
        save_path: str = self._make_plot_path()
        self._plot_counter += 1
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        self.plots.append((save_path, desc))

    def show_df(self, df: pd.DataFrame, desc: str) -> None:
        """Render a DataFrame as a markdown table and record it.

        Args:
            df: The DataFrame to render.
            desc: A human-readable description.
        """
        md: str = df.to_markdown(index=False)
        self.dataframes.append((md, desc))

    def show_json(self, x: Any, desc: str) -> None:
        """JSON-serialize an object and record it.

        Args:
            x: Any JSON-serializable object (dict, list, number, etc.).
            desc: A human-readable description.
        """
        json_str: str = json.dumps(x, indent=2, default=str)
        self.jsons.append((json_str, desc))

    def has_output(self) -> bool:
        """Return True if any ``show_*`` method was called."""
        return len(self.plots) > 0 or len(self.dataframes) > 0 or len(self.jsons) > 0

    @property
    def has_plots(self) -> bool:
        """Return True if at least one ``show_plt`` call was made."""
        return len(self.plots) > 0

    def build_artifacts(self) -> List[DiagnosticArtifact]:
        """Build structured artifacts from all captured outputs."""
        artifacts: List[DiagnosticArtifact] = []
        for path, desc in self.plots:
            artifacts.append(
                DiagnosticArtifact(
                    artifact_type="image",
                    description=desc,
                    inline_content=None,
                    attachment_path=path,
                    truncated=False,
                )
            )
        for md, desc in self.dataframes:
            artifacts.append(
                DiagnosticArtifact(
                    artifact_type="table",
                    description=desc,
                    inline_content=md,
                    attachment_path=None,
                    truncated=False,
                )
            )
        for json_str, desc in self.jsons:
            artifacts.append(
                DiagnosticArtifact(
                    artifact_type="json",
                    description=desc,
                    inline_content=json_str,
                    attachment_path=None,
                    truncated=False,
                )
            )
        return artifacts


# ===========================================================================
# 2. DynamicToolSpec — run-scoped data (NOT a Registry)
# ===========================================================================


class DynamicToolSpec(Typed):
    """One LLM-generated diagnostic tool, stored as data rather than a class.

    Architecture (see ``domains/__init__.py`` module docstring):

    - Static tools (``DistributionFittingTool``, ``TimeSeriesStaticTool``)
      are ``Tool + Registry`` subclasses: compile-time-known, one class
      per tool, auto-registered by Morphic's metaclass.
    - Dynamic tools are **not** Registry entries.  They are frozen
      ``Typed`` instances (name + description + code + ``execute()``)
      owned by a single run via ``RunDeps.dynamic_tools: Dict[str,
      DynamicToolSpec]``.

    Why data, not classes:

    - **Run scoping.**  A ``Registry`` lives on a class object, which is
      process-global.  In multi-dataset runs within one process,
      Registry entries leak across datasets.  A per-run dict cannot
      leak — the reference dies when ``run()`` returns.
    - **No ``exec()`` to build a class.**  The old
      ``register_dynamic_tool`` built a class-definition string and
      ``exec()``-ed it just to get Morphic to auto-register the
      resulting subclass.  With a data class, we just construct an
      instance: ``DynamicToolSpec(name=..., description=..., code=...)``.
    - **Thread-safe by construction.**  With
      ``--parallel.nthread > 0``, concurrent datasets in one process
      each get their own dict; no cross-thread shared state.

    Persistence:
        ``save_dynamic_tools(path=..., dynamic_tools=...)`` dumps the
        dict to JSON; ``load_dynamic_tools(path=...)`` returns a fresh
        ``Dict[str, DynamicToolSpec]``.  Only used when
        ``config.toolkit.accumulate_tools`` is true.

    Not a ``Tool`` subclass:
        ``Tool`` is an ABC with ``ClassVar`` schema fields designed for
        compile-time-known tool classes.  ``DynamicToolSpec`` carries
        the same information as *instance* fields so it can be built
        at runtime; the two interfaces happen to overlap but do not
        share a hierarchy.  The pipeline dispatches on them separately
        in ``_run_diagnostic_rounds`` (dynamic tool in ``deps.dynamic_tools``
        → ``spec.execute(...)`` directly; otherwise → domain
        ``execute_tool`` for the static path).

    Reproducibility fields:
        ``domain``, ``code_gen_model``, and ``code_gen_prompt`` together
        capture *how* the tool was generated: which domain requested it,
        which code-generation LLM produced the code, and the exact
        prompt that was sent.  A reviewer reading a persisted registry
        file can re-issue the same prompt to the same model to reproduce
        the code, and the ``domain`` field prevents cross-domain leakage
        at the persistence layer (a time-series registry file loaded
        into a distribution-fitting run will surface obvious mismatches).
    """

    name: str
    description: str
    code: str
    domain: Domain
    code_gen_model: str
    code_gen_prompt: str

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return the OpenAI function-calling schema dict for this tool.

        Dynamic tools take no parameters at call time — the code is
        wired at registration, the sandbox inputs come from the
        pipeline (data, map_estimate, family_name).  So the schema
        always has an empty ``properties`` and ``required`` list.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def execute(
        self,
        *,
        data: Union[np.ndarray, pd.Series],
        fit_state: Optional[FitState],
        fit_path: str,
    ) -> DiagnosticToolResult:
        """Run the tool's code against the current ``fit_state`` and return artifacts.

        Raises:
            RuntimeError: if ``fit_state`` is None.  Dynamic tools are
                generated in a fitted-model context (they typically
                reference ``map_estimate`` / ``family_name``), so calling
                them at step 0 — where no model exists yet — is a bug at
                the dispatcher level, not a valid state.  The pipeline
                already avoids this by not offering dyn tools at step 0
                (the run-level dict is empty or contains only
                accumulate_tools-seeded entries from prior runs).
        """
        if fit_state is None:
            raise RuntimeError(
                f"DynamicToolSpec {self.name!r} cannot execute without a fitted model "
                f"(fit_state is None). Dynamic tools are generated from a fitted model's "
                f"context and require map_estimate + family_name to run. "
                f"Tool description: {self.description}"
            )
        artifacts: List[DiagnosticArtifact] = execute_dynamic_tool(
            code=self.code,
            data=data,
            map_estimate=fit_state.map_estimate,
            family_name=list(fit_state.family_name),
            fit_path=fit_path,
        )
        return DiagnosticToolResult(
            tool_name=self.name,
            tool_description=self.description,
            artifacts=artifacts,
        )


class GeneratedToolExecutionResult(Typed):
    """Structured result from generate-new-tool orchestration.

    Carries the freshly-constructed ``DynamicToolSpec`` plus the
    artifacts produced by its first execution.  The caller is
    responsible for inserting ``spec`` into its run-level
    ``dynamic_tools`` dict — we keep the function pure (no caller
    mutation) so it works cleanly through ``@validate``'s
    argument-copy boundary.
    """

    spec: DynamicToolSpec
    artifacts: List[DiagnosticArtifact]
    attempts: List[CodeGenerationAttempt]

    @property
    def registered_tool_name(self) -> str:
        """Name of the registered tool (delegates to ``self.spec.name``)."""
        return self.spec.name

    @property
    def tool_description(self) -> str:
        """Description passed to the VLM (delegates to ``self.spec.description``)."""
        return self.spec.description


@validate
def save_dynamic_tools(
    *,
    path: str,
    dynamic_tools: Dict[str, DynamicToolSpec],
) -> None:
    """Persist a run's ``dynamic_tools`` dict to JSON.

    Each entry is serialized as ``{"name": ..., "description": ...,
    "code": ...}``.  Load with ``load_dynamic_tools()``.  A no-op when
    the dict is empty.
    """
    entries: List[Dict[str, Any]] = [spec.model_dump() for spec in dynamic_tools.values()]
    with open(path, "w") as file_handle:
        json.dump(entries, file_handle, indent=2)


@validate
def load_dynamic_tools(*, path: str) -> Dict[str, DynamicToolSpec]:
    """Load dynamic tool specs from a JSON file produced by ``save_dynamic_tools``.

    Returns an empty dict if ``path`` does not exist (so callers can
    unconditionally attempt a seed on every run).  Each entry is
    validated via Pydantic construction of ``DynamicToolSpec``.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as file_handle:
        entries: List[Dict[str, Any]] = json.load(file_handle)
    return {entry["name"]: DynamicToolSpec(**entry) for entry in entries}


# ===========================================================================
# 4. execute_dynamic_tool
# ===========================================================================


class _SilentPlt:
    """Proxy around ``matplotlib.pyplot`` that makes ``show()`` and
    ``savefig()`` no-ops.  Generated tool code that ignores the prompt
    and calls ``plt.show()`` will silently continue instead of hanging
    on a non-interactive backend."""

    def __init__(self, real_plt: Any) -> None:
        self._plt: Any = real_plt

    def show(self, *args: Any, **kwargs: Any) -> None:
        pass

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._plt, name)


@validate
def execute_dynamic_tool(
    *,
    code: str,
    data: Union[np.ndarray, pd.Series],
    map_estimate: Dict[str, Any],
    family_name: List[str],
    fit_path: str,
) -> List[DiagnosticArtifact]:
    """Run LLM-generated tool code in a controlled sandbox.

    The sandbox exposes:
        - ``data``, ``map_estimate``, ``family_name`` — the current state
        - ``np``, ``plt``, ``pd``, ``stats`` — standard scientific libraries
        - ``show_plt(fig, desc)``, ``show_df(df, desc)``,
          ``show_json(x, desc)`` — the three output harness functions

    Args:
        code: Python source code to execute.
        data: The 1-D dataset being fitted.
        map_estimate: MAP parameter estimates from the current best model.
        family_name: Canonical list of component family names, matching
            ``FitState.family_name`` (e.g. ``["gaussian"]``,
            ``["student_t"]``, or ``["gaussian", "cauchy"]``). Generated
            tool code should iterate over this list to handle mixtures
            uniformly; do **not** string-join and re-split.
        fit_path: Path where the primary plot should be saved.

    Returns:
        Structured artifacts emitted by ``show_plt``, ``show_df``, and
        ``show_json`` in emission order.
    """
    collector: ToolOutputCollector = ToolOutputCollector(fit_path=fit_path)

    logger.debug(
        format_log_block(
            title=f"Executing dynamic tool code ({len(code)} chars)",
            body=code,
        )
    )

    code = unescape_broken_code_if_syntax_error(code)

    namespace: Dict[str, Any] = {
        **get_tool_runtime_namespace(plt_wrapper=_SilentPlt(plt)),
        "data": data,
        "map_estimate": map_estimate,
        "family_name": family_name,
        "show_plt": collector.show_plt,
        "show_df": collector.show_df,
        "show_json": collector.show_json,
    }

    try:
        exec(code, namespace)
    except Exception as exc:
        # Broad catch is intentional: exec'd LLM-generated code can raise ANY exception
        # type (SyntaxError, NameError, IndexError, ZeroDivisionError, etc.).
        # We cannot enumerate all possible failures from arbitrary generated code.
        error_msg: str = f"Generated tool code crashed during dynamic execution: {format_exception_msg(exc)}."
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc
    finally:
        plt.close("all")

    if collector.has_output() is False:
        error_msg = "Generated tool code ran but produced no show_plt/show_df/show_json output."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if collector.has_plots is False:
        logger.debug(
            f"Dynamic tool produced text/data output but no plot. "
            f"fit_path ({fit_path}) will not have an image — callers must handle this."
        )

    return collector.build_artifacts()


# ===========================================================================
# 5. GENERATE_NEW_TOOL_SCHEMA
# ===========================================================================

GENERATE_NEW_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_new_tool",
        "description": (
            "Generate a NEW diagnostic visualization or analysis tool that "
            "does not already exist in the available toolkit. Describe in "
            "detail what the tool should compute or plot. The tool code "
            "will be generated by a separate code-generation model and "
            "added to the toolkit for this and future iterations. Use this "
            "when no existing tool can answer your current diagnostic "
            "question (e.g., you need a zoomed tail plot, a specific "
            "statistical test, a residual analysis, or a custom transform "
            "that the static toolkit does not provide)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_description": {
                    "type": "string",
                    "description": (
                        "A detailed description of what the new tool should "
                        "do: what it computes, what it plots, and what "
                        "diagnostic question it answers. This description "
                        "is passed to a code-generation model, so be "
                        "specific about axes, data transformations, and "
                        "what the visual or numeric output should show."
                    ),
                },
            },
            "required": ["tool_description"],
        },
    },
}


# ===========================================================================
# 6. TOOL_GENERATION_PROMPT
# ===========================================================================

TOOL_GENERATION_PROMPT: str = """\
You are an expert Python data scientist. Write a short Python code snippet \
that performs the following diagnostic analysis:

{{tool_description}}

{domain_context}

OUTPUT FUNCTIONS — use ONLY these to produce visible output:
  show_plt(fig, desc)   — the ONLY way to display a matplotlib Figure
  show_df(df, desc)     — the ONLY way to display a pandas DataFrame
  show_json(obj, desc)  — the ONLY way to display a JSON-serializable object

CRITICAL RULES:
  - To display a plot: create a Figure with fig, ax = plt.subplots(), draw on ax,
    then call show_plt(fig, "description of what this plot shows").
  - NEVER call plt.show() — it is a no-op and will NOT display anything.
  - NEVER call plt.savefig() — show_plt handles saving.
  - Call at least one show_* function. Code that produces no output is discarded.
  - Do NOT import anything. All libraries are already in scope.
  - Keep the code under 30 lines. Focus on one diagnostic question.
  - The description string passed to show_* must be plain-language and specific.
  - Do NOT prefix descriptions with [Visual], [Table], [Data], or similar tags.

KNOWN ANTI-PATTERNS — these are BANNED because they crash at runtime:
  1. plt.stem(…, use_line_collection=…) — The `use_line_collection` parameter
     was removed from matplotlib. NEVER pass `use_line_collection` to stem().
     Just call `ax.stem(x, y)` without that keyword.
  2. df.fillna(method='ffill') or df.fillna(method='bfill') — The `method`
     parameter was removed from pandas fillna(). Use `df.ffill()` or
     `df.bfill()` instead.
  3. scipy.stats.diagnostic — This module DOES NOT EXIST in any version of
     SciPy. Do not import or reference it. Statistical tests like
     `shapiro`, `normaltest`, `kstest`, `anderson`, etc. live directly in
     `scipy.stats` (available as `stats` in the sandbox).

{domain_example}

Return ONLY Python code inside a ```python code fence. No explanation outside the fence.\
"""


_DOMAIN_CONTEXT_DISTRIBUTION_FITTING: str = """\
SANDBOX CONTRACT — your code runs in an exec() sandbox with these variables:
  data          : numpy 1-D array of observed values
  map_estimate  : dict of MAP parameter estimates (e.g. {{"gaussian_mu_0": 1.2}})
  family_name   : List[str], the current fitted distribution family
                  (e.g. ["gaussian"], ["student_t"], or ["gaussian", "cauchy"]).
                  Always a list, even for single distributions. For mixtures,
                  iterate over the list; do NOT string-join-and-split.
  np            : numpy
  plt           : matplotlib.pyplot (show() and savefig() are disabled)
  pd            : pandas
  stats         : scipy.stats"""


_DOMAIN_CONTEXT_TIME_SERIES: str = """\
SANDBOX CONTRACT — your code runs in an exec() sandbox with these variables:
  data          : pandas Series — the observed time series.
                  data.index is the time axis (numeric or datetime-like).
                  data.values is the signal as a numpy array.
  map_estimate  : dict of MAP parameter estimates from the current GP model
                  (e.g. {{"periodic_period": 7.0, "rbf_lengthscale": 20.0}}).
                  Empty dict at step 0 when no model has been fitted yet.
  family_name   : List[str], the GP kernel name(s) used in the current model
                  (e.g. ["periodic"], ["rbf", "linear"], ["warped_periodic"]).
                  Empty list at step 0. Always a list, even for single kernels.
  np            : numpy  (np.fft is available for spectral analysis)
  plt           : matplotlib.pyplot (show() and savefig() are disabled)
  pd            : pandas
  stats         : scipy.stats  (stats.diagnostic does NOT exist —
                  use stats.shapiro, stats.normaltest, stats.kstest, etc.
                  directly from stats)

DOMAIN NOTES — this is a time-series Gaussian Process modeling task:
  - The goal is to identify periodicity, trend, and noise structure.
  - Useful diagnostics: autocorrelation (np.correlate), power spectrum
    (np.fft.fft / np.abs), residual analysis, zero-crossing spacing,
    peak-to-peak interval measurement, rolling statistics.
  - Always plot against data.index (the time axis), not integer indices.
  - If map_estimate is empty, focus on raw signal exploration."""


_DOMAIN_EXAMPLE_DISTRIBUTION_FITTING: str = """\
EXAMPLE — this is the kind of code you should produce:

```python
fig, ax = plt.subplots(figsize=(8, 5))
sorted_data = np.sort(data)
ccdf = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax.loglog(sorted_data[sorted_data > 0], ccdf[sorted_data > 0], 'b.', alpha=0.5)
ax.set_xlabel('Value (log scale)')
ax.set_ylabel('P(X > x) (log scale)')
ax.set_title('Log-Log CCDF — Power Law Tail Check')
show_plt(fig, "Log-log CCDF plot: a straight line indicates power-law tails")

tail_idx = int(0.9 * len(sorted_data))
tail_data = sorted_data[tail_idx:]
slope, intercept, r, p, se = stats.linregress(
    np.log(tail_data[tail_data > 0]),
    np.log(ccdf[tail_idx:][tail_data > 0]),
)
show_json(
    {{"tail_slope": round(slope, 3), "r_squared": round(r**2, 3)}},
    "Tail regression: slope is the tail index estimate",
)
```"""


_DOMAIN_EXAMPLE_TIME_SERIES: str = """\
EXAMPLE — this is the kind of code you should produce:

```python
values = data.values
n = len(values)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# Autocorrelation to find dominant period
mean_val = np.mean(values)
centered = values - mean_val
acf_full = np.correlate(centered, centered, mode='full')
acf = acf_full[n - 1:] / acf_full[n - 1]
max_lag = min(n // 2, 200)
lags = np.arange(max_lag)
ax1.plot(lags, acf[:max_lag])
ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Lag')
ax1.set_ylabel('Autocorrelation')
ax1.set_title('ACF — Dominant Period Detection')

# Power spectrum via FFT
spectrum = np.abs(np.fft.rfft(centered)) ** 2
freqs = np.fft.rfftfreq(n)
ax2.plot(freqs[1:], spectrum[1:])
ax2.set_xlabel('Frequency (cycles per sample)')
ax2.set_ylabel('Power')
ax2.set_title('Power Spectral Density')
fig.tight_layout()
show_plt(fig, "ACF and power spectrum for periodicity detection")

peak_freq = freqs[1:][np.argmax(spectrum[1:])]
dominant_period = round(1.0 / peak_freq, 2) if peak_freq > 0 else None
show_json(
    {{"dominant_period_samples": dominant_period, "n_points": n}},
    "Estimated dominant period in number of samples from FFT peak",
)
```"""


_DOMAIN_CONTEXTS: Dict[Domain, str] = {
    Domain.distribution_fitting: _DOMAIN_CONTEXT_DISTRIBUTION_FITTING,
    Domain.time_series: _DOMAIN_CONTEXT_TIME_SERIES,
}

_DOMAIN_EXAMPLES: Dict[Domain, str] = {
    Domain.distribution_fitting: _DOMAIN_EXAMPLE_DISTRIBUTION_FITTING,
    Domain.time_series: _DOMAIN_EXAMPLE_TIME_SERIES,
}


def _build_tool_generation_prompt(*, domain: Domain, tool_description: str) -> str:
    """Build the tool generation prompt with domain-specific context."""
    domain_context: str = _DOMAIN_CONTEXTS[domain]
    domain_example: str = _DOMAIN_EXAMPLES[domain]
    template: str = TOOL_GENERATION_PROMPT.format(
        domain_context=domain_context,
        domain_example=domain_example,
    )
    return template.format(tool_description=tool_description)


# ===========================================================================
# 7. build_tools_list
# ===========================================================================


@validate
def build_tools_list(
    *,
    toolkit_mode: ToolkitMode,
    static_tools: List[Dict[str, Any]],
    dynamic_tools: Dict[str, DynamicToolSpec],
) -> List[Dict[str, Any]]:
    """Build the ``tools`` list passed to ``backend.predict()``.

    Args:
        toolkit_mode: ``ToolkitMode`` enum member. String inputs like
            ``"generate_only"`` or ``"generate-only"`` are auto-coerced
            by ``@validate`` via AutoEnum fuzzy matching.
        static_tools: The domain's static tool schemas (from
            ``DomainToolkit.get_static_tools()``).
        dynamic_tools: The run's dyn-tool dict (from
            ``RunDeps.dynamic_tools``).  Empty at step 0 of a fresh run
            with ``accumulate_tools=False``.  Schemas are emitted only
            for entries present in this dict — so tools from other
            datasets or earlier runs that leaked into a process-wide
            store (there is none any more) cannot reach the VLM.

    Returns:
        List of OpenAI function-calling tool schemas.
    """
    if toolkit_mode is ToolkitMode.none:
        return []
    elif toolkit_mode is ToolkitMode.static:
        return list(static_tools)
    elif toolkit_mode is ToolkitMode.generate_only:
        tools: List[Dict[str, Any]] = [spec.to_openai_schema() for spec in dynamic_tools.values()]
        tools.append(GENERATE_NEW_TOOL_SCHEMA)
        return tools
    elif toolkit_mode is ToolkitMode.accumulated_only:
        return [spec.to_openai_schema() for spec in dynamic_tools.values()]
    elif toolkit_mode is ToolkitMode.dynamic:
        static_names: Set[str] = {t["function"]["name"] for t in static_tools}
        tools = list(static_tools)

        for spec in dynamic_tools.values():
            if spec.name not in static_names:
                tools.append(spec.to_openai_schema())

        tools.append(GENERATE_NEW_TOOL_SCHEMA)
        return tools
    else:
        raise ValueError(
            f"Unknown toolkit_mode={toolkit_mode!r}. Must be one of {[m.value for m in ToolkitMode]}."
        )


# ===========================================================================
# 8. handle_generate_new_tool
# ===========================================================================


def _build_tool_repair_prompt(
    *, base_prompt: str, previous_code: str, error_message: str, repair_context: str
) -> str:
    """Build a repair prompt for failed generated diagnostic tool code.

    Args:
        base_prompt: The original tool-generation prompt.
        previous_code: The code that failed during sandbox execution.
        error_message: The exception message from execution.
        repair_context: A runtime-grounded API discovery report built by
            ``build_api_discovery_report``. May be empty string if the error
            type is not classified, in which case the block is omitted.
    """
    context_section: str = (
        f"{'═' * 60}\n"
        f"RUNTIME API DISCOVERY (from the live Python environment)\n"
        f"{'═' * 60}\n"
        f"{repair_context}\n"
        f"{'═' * 60}\n\n"
        if len(repair_context) > 0
        else ""
    )
    return (
        f"{base_prompt}\n\n"
        f"The previous code or code-fenced response failed. The full traceback is included below. "
        f"Fix the code and return ONLY corrected Python code inside a ```python code fence.\n\n"
        f"{context_section}"
        f"Previous code:\n```python\n{previous_code}\n```\n\n"
        f"Error (full traceback):\n{error_message}"
    )


def _extract_code_from_response(raw: str) -> str:
    """Extract Python code from an LLM response.

    Tries three strategies in order:
    1. Find a ```python ... ``` fenced code block (most reliable).
    2. Find a ``` ... ``` fenced code block (no language tag).
    3. If no fences found, treat the entire response as code after
       stripping leading/trailing whitespace.
    """
    fenced_match: Optional[re.Match] = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if fenced_match is not None:
        return fenced_match.group(1).strip()

    generic_match: Optional[re.Match] = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
    if generic_match is not None:
        return generic_match.group(1).strip()

    return raw.strip()


_BOILERPLATE_PREFIXES: List[str] = [
    "create a diagnostic tool that ",
    "create a diagnostic ",
    "create a tool that ",
    "create a tool to ",
    "generate a tool that ",
    "generate a tool to ",
    "build a tool that ",
    "build a tool to ",
    "compute and return ",
    "compute and plot ",
    "create a ",
    "generate a ",
    "build a ",
    "plot the ",
    "compute the ",
    "analyze the ",
    "diagnostic ",
    "tool that ",
    "tool to ",
]


def _slugify_tool_name(description: str) -> str:
    """Derive a concise, meaningful Python identifier from a tool description.

    Strips common boilerplate prefixes (e.g. 'Create a diagnostic tool that'),
    then takes the first few meaningful words + a short hash for uniqueness.
    """
    cleaned: str = description.lower().strip()
    for prefix in _BOILERPLATE_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    slug: str = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    filler_words: Set[str] = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "with",
        "by",
        "from",
        "using",
        "via",
    }
    words: List[str] = [w for w in slug.split("_") if w not in filler_words and len(w) > 0]
    slug = "_".join(words)[:40].rstrip("_")
    short_hash: str = hashlib.md5(description.encode()).hexdigest()[:6]
    name: str = f"dyn_{slug}_{short_hash}"
    if not name.isidentifier():
        name = f"dyn_tool_{short_hash}"
    return name


@validate
def handle_generate_new_tool(
    *,
    tool_description: str,
    tool_gen_backend: Any,  # VLMBackend; typed as Any for @validate compatibility with mocks/proxies
    data: Union[np.ndarray, pd.Series],
    map_estimate: Dict[str, Any],
    family_name: List[str],
    fit_path: str,
    verbosity: int,
    domain: Domain,
    tool_gen_model: str,
    max_tool_generation_attempts: int,
) -> GeneratedToolExecutionResult:
    """Orchestrate dynamic tool generation: code-gen → sandbox execute → return spec.

    This is the high-level function called when the VLM selects
    ``generate_new_tool``.  It:

    1. Sends the tool description to the tool-generation LLM via ``tool_gen_backend``.
    2. Runs the returned code in the sandbox via ``execute_dynamic_tool``.
    3. If the code produces valid output, builds a ``DynamicToolSpec``
       (stamped with ``domain``, ``code_gen_model``, ``code_gen_prompt``
       for reproducibility) and returns it inside the
       ``GeneratedToolExecutionResult``.

    The caller is responsible for registering the returned spec in the
    run's ``dynamic_tools`` dict.  This keeps ``handle_generate_new_tool``
    pure w.r.t. caller state, which is required because ``@validate``
    copies dict arguments at the wrapper boundary — so a
    ``dynamic_tools: Dict[...]`` parameter could not be mutated in place.

    Args:
        tool_description: What the VLM wants the new tool to do.
        tool_gen_backend: A ``VLMBackend`` instance used for dynamic-tool code generation.
            Called with ``response_type=None`` to get raw text.
        data: The 1-D dataset being fitted.
        map_estimate: MAP parameter estimates from the current best model
            (empty dict at step 0, where no fit exists yet).
        family_name: Canonical list of component family names
            (``List[str]``). Empty list at step 0 (no fit yet).
        fit_path: Path where the tool's primary plot should be saved.
        verbosity: Logging verbosity level.
        domain: The problem domain that requested the tool; stamped on
            the returned ``DynamicToolSpec`` so a persisted registry
            file cannot accidentally leak into a different domain.
        tool_gen_model: The LiteLLM model identifier that produced the
            dynamic-tool code (e.g. ``"azure/gpt-5.4-mini"``); stamped on the
            returned spec so a reviewer can re-issue the same prompt
            against the same model to reproduce the generation.
        max_tool_generation_attempts: Maximum generate-execute-repair attempts before
            surfacing the last execution failure.

    Returns:
        ``GeneratedToolExecutionResult`` containing the built
        ``DynamicToolSpec`` plus the artifacts produced by its
        first execution.
    """
    logger.debug("Calling tool generation LLM for dynamic tool request.")
    logger.debug(
        format_log_block(
            title="Calling tool generation LLM",
            body=f"Tool request: {tool_description}",
        )
    )

    base_prompt: str = _build_tool_generation_prompt(
        domain=domain,
        tool_description=tool_description,
    )
    attempts: List[CodeGenerationAttempt] = []
    last_error: Optional[str] = None
    last_exception: Optional[Exception] = None
    prompt: str = base_prompt

    for attempt_number in range(1, max_tool_generation_attempts + 1):
        attempt_kind: str = "initial" if attempt_number == 1 else "repair"
        if attempt_kind == "repair":
            if last_error is None:
                raise RuntimeError(
                    f"Internal error: tool repair attempt {attempt_number} has no prior error."
                )
            logger.info(
                f"Retrying dynamic tool generation for {tool_description!r}: "
                f"attempt {attempt_number}/{max_tool_generation_attempts} will use a repair prompt. "
                f"Previous error: {last_error}"
            )
            tool_repair_context: str = build_api_discovery_report(
                generated_code=attempts[-1].code,
                error_message=last_error,
                runtime_namespace=get_tool_runtime_namespace(),
            )
            prompt = _build_tool_repair_prompt(
                base_prompt=base_prompt,
                previous_code=attempts[-1].code,
                error_message=last_error,
                repair_context=tool_repair_context,
            )

        logger.debug(
            format_log_block(
                title=(
                    f"Tool generation {attempt_kind} prompt attempt "
                    f"{attempt_number}/{max_tool_generation_attempts} ({len(prompt)} chars)"
                ),
                body=prompt,
            )
        )

        raw_response: str = ""
        code: str = ""

        try:
            raw_response = tool_gen_backend.call(
                prompt=prompt,
                response_type=None,
                verbosity=verbosity,
            )
        except Exception as exc:
            error_message: str = format_exception_msg(exc)
            logger.error(
                f"Tool generation backend call failed for {tool_description!r} "
                f"attempt {attempt_number}/{max_tool_generation_attempts}: {error_message}"
            )
            raise CodeGenerationFailure(
                message=(
                    f"Tool generation backend call failed for {tool_description!r} "
                    f"attempt {attempt_number}/{max_tool_generation_attempts}. Full traceback:\n{error_message}"
                ),
                attempts=attempts,
            ) from exc

        logger.debug(
            format_log_block(
                title=(
                    f"Tool generation {attempt_kind} raw response attempt "
                    f"{attempt_number}/{max_tool_generation_attempts} ({len(raw_response)} chars)"
                ),
                body=raw_response,
            )
        )

        try:
            code = _extract_code_from_response(raw_response)
            if len(code) == 0:
                raise RuntimeError(
                    "Code-generation LLM returned no extractable Python code for tool "
                    f"{tool_description!r}. Raw response:\n{raw_response}"
                )
        except Exception as exc:
            last_error = format_exception_msg(exc)
            last_exception = exc
            attempts.append(
                CodeGenerationAttempt(
                    stage="dynamic_tool",
                    target=tool_description,
                    attempt_number=attempt_number,
                    max_attempts=max_tool_generation_attempts,
                    attempt_kind=attempt_kind,
                    failure_stage="parse",
                    prompt=prompt,
                    raw_response=raw_response,
                    code=code,
                    success=False,
                    error=last_error,
                )
            )
            logger.error(
                f"Tool generation parse attempt {attempt_number}/{max_tool_generation_attempts} failed for "
                f"{tool_description!r}: {last_error}"
            )
            continue

        logger.debug(
            format_log_block(
                title=(
                    f"Generated tool code attempt {attempt_number}/{max_tool_generation_attempts} "
                    f"({len(code)} chars)"
                ),
                body=code,
            )
        )

        try:
            artifacts: List[DiagnosticArtifact] = execute_dynamic_tool(
                code=code,
                data=data,
                map_estimate=map_estimate,
                family_name=family_name,
                fit_path=fit_path,
            )
        except Exception as exc:
            last_error = format_exception_msg(exc)
            last_exception = exc
            attempts.append(
                CodeGenerationAttempt(
                    stage="dynamic_tool",
                    target=tool_description,
                    attempt_number=attempt_number,
                    max_attempts=max_tool_generation_attempts,
                    attempt_kind=attempt_kind,
                    failure_stage="execute",
                    prompt=prompt,
                    raw_response=raw_response,
                    code=code,
                    success=False,
                    error=last_error,
                )
            )
            logger.error(
                f"Tool generation execution attempt {attempt_number}/{max_tool_generation_attempts} failed for "
                f"{tool_description!r}: {last_error}"
            )
            continue

        attempts.append(
            CodeGenerationAttempt(
                stage="dynamic_tool",
                target=tool_description,
                attempt_number=attempt_number,
                max_attempts=max_tool_generation_attempts,
                attempt_kind=attempt_kind,
                failure_stage="none",
                prompt=prompt,
                raw_response=raw_response,
                code=code,
                success=True,
                error=None,
            )
        )
        tool_name: str = _slugify_tool_name(tool_description)
        spec: DynamicToolSpec = DynamicToolSpec(
            name=tool_name,
            description=tool_description,
            code=code,
            domain=domain,
            code_gen_model=tool_gen_model,
            code_gen_prompt=base_prompt,
        )
        logger.debug(f"Tool built: name={tool_name} (caller will insert into run dynamic_tools dict)")
        return GeneratedToolExecutionResult(
            spec=spec,
            artifacts=artifacts,
            attempts=attempts,
        )

    if last_exception is None:
        logger.error(
            f"Dynamic tool generation FAILED after all {max_tool_generation_attempts} attempt(s) for "
            f"{tool_description!r} (no terminal exception captured)."
        )
        raise CodeGenerationFailure(
            message=(
                f"Dynamic tool generation failed without an exception after "
                f"{max_tool_generation_attempts} attempt(s)."
            ),
            attempts=attempts,
        )
    logger.error(
        f"Dynamic tool generation FAILED after all {max_tool_generation_attempts} attempt(s) for "
        f"{tool_description!r}. Last error: {last_error}"
    )
    raise CodeGenerationFailure(
        message=(
            f"Dynamic tool generation failed after {max_tool_generation_attempts} attempt(s) for "
            f"{tool_description!r}. Last error: {last_error}"
        ),
        attempts=attempts,
    ) from last_exception


# ===========================================================================
# 9. Diagnostic phase prompt (used by experiments._run_diagnostic_rounds)
# ===========================================================================

DIAGNOSTIC_PHASE_PROMPT: str = """\
You are evaluating a statistical model fit. The {carried_label} fit so far is shown in the attached plot.

{carried_label_capitalized} model summary: {current_model_summary}
{carried_label_capitalized} AIC: {current_aic}
Step: {step_num} of {max_steps}
Previously tested families/kernels: {tested_model_structures}

If you need diagnostic information before proposing a revised model, call one of the available \
diagnostic tools. If you have enough information to propose a model directly, respond without \
calling any tool."""


# Appended to DIAGNOSTIC_PHASE_PROMPT only when the toolkit mode allows
# the VLM to synthesize new tools (i.e. ``ToolkitMode.generate_only`` or
# ``ToolkitMode.dynamic``).  It must NOT be shown under ``none`` (no
# tools offered at all) or ``static`` (``generate_new_tool`` is not in
# the toolkit), because under those modes the instruction would either
# be meaningless or refer to a tool the VLM cannot call.  Without this
# nudge the VLM tends to reuse the first tool it ever generated instead
# of growing a diverse library across heterogeneous datasets.
GENERATE_NEW_TOOL_NUDGE: str = """

If none of the available diagnostic tools directly address your current \
uncertainty about this specific data, prefer calling ``generate_new_tool`` \
to build a targeted diagnostic rather than reusing an imperfect existing tool. \
Reuse an existing tool only when it is a genuinely good fit for the question \
you need answered about this dataset."""
