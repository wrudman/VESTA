"""Domain registry base classes for the VESTA pipeline.

Four ABC base classes decompose domain-specific behavior so that the
``run()`` function in ``experiments.py`` is domain-agnostic:

- ``DomainPrompts``  — prompt rendering, VLM response parsing, data shapes
- ``DomainToolkit``  — tool execution dispatch (delegates to Tool subclasses)
- ``DomainPlotting`` — visualization + fit state extraction
- ``Tool``           — base class for diagnostic tools (see below)

Tool System Architecture
========================
The tool system uses a two-level Registry pattern:

1. ``Tool`` (this module) is a ``Typed + ABC`` base class — NOT a Registry.
   It defines the interface every diagnostic tool must implement: ClassVars
   for schema (``tool_description``, ``output_type``, ``parameters_schema``)
   and an abstract ``execute()`` method.

2. Each **domain** creates its own Registry subclass of ``Tool``:
   - ``DistributionFittingTool(Tool, Registry, ABC)`` in
     ``domains/distribution_fitting/toolkit.py``
   - ``TimeSeriesExpertTool(Tool, Registry, ABC)`` in
     ``domains/time_series/toolkit.py``

   Concrete tool implementations (e.g., ``QQPlot``, ``FitVsActuals``)
   subclass the domain-specific Registry.  Morphic auto-registers them,
   so ``DistributionFittingTool.of("qq_plot")`` resolves ``QQPlot``.

3. **LLM-generated tools are deliberately NOT a Registry.**
   ``DynamicToolSpec`` in ``dynamic_toolkit.py`` is a frozen ``Typed``
   data class (name + description + code + ``execute()``).  Each
   ``run()`` call owns its own ``Dict[str, DynamicToolSpec]`` via
   ``RunDeps.dynamic_tools``, so dynamic tools cannot leak across
   datasets in the same process.  When
   ``config.toolkit.accumulate_tools`` is true the dict is seeded from
   ``config.toolkit.tool_registry_filename`` at run start and written
   back at run end; otherwise it is always fresh-empty.

   Why not a Registry?  A ``Registry`` stores subclasses on the class
   object, which is process-global.  Runtime-generated tools that
   depend on a fitted model's context (``map_estimate``,
   ``family_name``) must not outlive the run that produced them.  The
   per-run dict makes that invariant structural rather than relying
   on a defensive clear at the top of each run.

WHY ``Tool`` is NOT a Registry:
    If ``Tool`` were a Registry, all tools from all domains would share one
    flat namespace.  ``Tool.of("qq_plot")`` would succeed even in the
    time-series domain where QQ plots don't exist.  Domain-scoped registries
    prevent this: each domain's Registry contains only its own tools.

Adding a new expert tool:
    1. Define a class in the domain's ``toolkit.py`` that subclasses the
       domain's Tool Registry (e.g., ``class MyTool(DistributionFittingTool)``).
    2. Set ``tool_description`` and ``output_type`` as ClassVars.
    3. Implement ``execute()``.
    That's it.  The tool auto-registers under its snake_case class name,
    auto-generates its OpenAI schema, and is included in
    ``get_expert_tools()`` via ``DistributionFittingTool.subclasses()``.
"""

import inspect
import re
import threading
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

from morphic import Registry, Typed, classproperty

_MATPLOTLIB_LOCK: threading.Lock = threading.Lock()


class CodeGenResponse(Typed):
    """Response from a code-generation LLM call.

    Used across all domains — the VLM returns JSON with a ``"code"`` key
    containing PyMC model code.  Pydantic validates at construction time;
    if ``code`` is missing or not a string, construction raises
    ``ValueError`` which triggers SlowBurn retry.
    """

    code: str

    def __str__(self) -> str:
        return f"  code ({len(self.code)} chars):\n{'─' * 50}\n{self.code}\n{'─' * 50}"


class CodeGenerationAttempt(Typed):
    """One generate → execute validation attempt for LLM-written code.

    The pipeline uses this for both runtime-generated diagnostic tools and
    PyMC model code.  Each record stores the exact prompt, raw response,
    extracted code, and execution outcome so run logs contain the complete
    repair trajectory rather than only the final successful code.
    """

    stage: str
    target: str
    attempt_number: int
    max_attempts: int
    attempt_kind: str
    failure_stage: str
    prompt: str
    raw_response: str
    code: str
    success: bool
    error: Optional[str] = None

    def __str__(self) -> str:
        status: str = "success" if self.success else "failed"
        lines: List[str] = [
            f"stage: {self.stage}",
            f"target: {self.target}",
            f"attempt: {self.attempt_number}/{self.max_attempts}",
            f"attempt kind: {self.attempt_kind}",
            f"failure stage: {self.failure_stage}",
            f"status: {status}",
            "prompt:",
            "─" * 50,
            self.prompt,
            "─" * 50,
            "raw response:",
            "─" * 50,
            self.raw_response,
            "─" * 50,
            "code:",
            "─" * 50,
            self.code,
            "─" * 50,
        ]
        if self.error is not None:
            lines.extend(["error:", self.error])
        return "\n".join(lines)


class CodeGenerationFailure(RuntimeError):
    """Raised when generated code fails all repair attempts."""

    def __init__(self, *, message: str, attempts: List[CodeGenerationAttempt]) -> None:
        super().__init__(message)
        self.attempts: List[CodeGenerationAttempt] = attempts


class DiagnosticArtifact(Typed):
    """One diagnostic artifact emitted by a tool execution."""

    artifact_type: str
    description: str
    inline_content: Optional[str] = None
    attachment_path: Optional[str] = None
    truncated: bool = False


class DiagnosticToolResult(Typed):
    """Structured diagnostic output for one invoked tool."""

    tool_name: str
    tool_description: str
    artifacts: List[DiagnosticArtifact]


# ══════════════════════════════════════════════════════════════════════════════
#  Tool base class
# ══════════════════════════════════════════════════════════════════════════════


class Tool(Typed, ABC):
    """Base class for all diagnostic tools across all domains.

    NOT a Registry itself — domain-specific registries extend this.
    See the module docstring for the full architecture.

    Each concrete tool subclass co-locates its schema (ClassVars) with its
    implementation (``execute()``), so adding a tool is a single-class change
    with zero wiring elsewhere.

    Required ClassVars on every concrete subclass:

    - ``tool_description: ClassVar[str]`` — VLM-facing description of what
      the tool does.  Also used as the ``description`` field in the OpenAI
      function-calling schema.
    - ``output_type: ClassVar[str]`` — ``"visualization"`` or ``"numeric"``.
    - ``parameters_schema: ClassVar[Dict[str, Any]]`` — JSON Schema for
      tool parameters.  Empty dict ``{}`` for tools that take no arguments.

    Auto-derived ClassVars (override only if the auto-derived value is wrong):

    - ``tool_name`` — snake_case of the class name, e.g. ``QQPlot`` →
      ``"qq_plot"``.  Used as the function name in OpenAI schemas AND as
      the Morphic Registry lookup key (via ``_registry_keys()``).

    LLM Agent Instructions:
        When creating a new tool, subclass the domain's Tool Registry
        (e.g., ``DistributionFittingTool``), NOT this class directly.
        Set ``tool_description``, ``output_type``, ``parameters_schema``
        as ClassVars, then implement ``execute()``.  Do NOT write aliases,
        do NOT write ``to_openai_schema()``, do NOT add the tool to any
        list or constant — all of that is automatic.
    """

    tool_description: ClassVar[str]
    output_type: ClassVar[str]
    parameters_schema: ClassVar[Dict[str, Any]]

    @classproperty
    def tool_name(cls) -> str:
        """Auto-derive snake_case name from PascalCase class name.

        Examples:
            ``QQPlot`` → ``"qq_plot"``
            ``CalculateMoments`` → ``"calculate_moments"``
            ``FitVsActualsWithResidualsDistribution`` →
                ``"fit_vs_actuals_with_residuals_distribution"``

        Subclasses can override by declaring ``tool_name`` as an explicit
        ``ClassVar[str]``.  This value is used for three purposes:

        1. The ``"name"`` field in the OpenAI function-calling schema.
        2. The Morphic Registry lookup key (``DomainTool.of("qq_plot")``).
        3. Human-readable identification in logs and error messages.
        """
        name: str = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", cls.__name__)
        name: str = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        return name.lower()

    @classmethod
    def _registry_keys(cls) -> List[str]:
        """Auto-register under the snake_case ``tool_name`` AND validate ClassVars.

        Called by Morphic during subclass registration.  This is the
        definition-time validation hook: if a concrete Tool subclass is
        missing ``tool_description``, ``output_type``, or
        ``parameters_schema``, registration fails with ``TypeError``.

        DO NOT override ``__init_subclass__`` on Tool — it conflicts with
        Morphic's Typed/Registry metaclass machinery.  Use this method
        for definition-time checks instead.
        """
        if not inspect.isabstract(cls):
            for attr in ("tool_description", "output_type", "parameters_schema"):
                if not hasattr(cls, attr):
                    raise TypeError(
                        f"{cls.__name__} must define ClassVar '{attr}'. "
                        f"All concrete Tool subclasses require tool_description, "
                        f"output_type, and parameters_schema."
                    )
            return [cls.tool_name]
        return []

    @classmethod
    def to_openai_schema(cls) -> Dict[str, Any]:
        """Generate the OpenAI function-calling tool schema from ClassVars.

        Auto-generated — no hand-written JSON needed.  The schema uses
        ``tool_name`` as the function name, ``tool_description`` as the
        description, and ``parameters_schema`` as the parameters spec.

        All properties in ``parameters_schema`` are treated as required
        (consistent with OpenAI's function-calling convention where all
        declared parameters are expected).
        """
        properties: Dict[str, Any] = {}
        for k, v in cls.parameters_schema.items():
            prop: Dict[str, Any] = dict(v) if isinstance(v, dict) else {"type": "string"}
            prop.pop("required", None)
            properties[k] = prop
        required_params: List[str] = list(cls.parameters_schema.keys())
        return {
            "type": "function",
            "function": {
                "name": cls.tool_name,
                "description": cls.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

    @abstractmethod
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional["FitState"],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        """Execute the diagnostic tool.

        Args:
            data: The dataset (numpy array for dist-fitting, pandas Series
                for time-series).
            fit_state: Domain-specific fit state, or ``None`` at step 0
                when no model has been fitted yet.
            best_idx: Index of the best model proposal.
            fit_path: File path where the tool should save its plot.
            selected_tool_args: Optional arguments the VLM passed to this
                tool via the function-calling API.  Most tools ignore this
                (no parameters).  Tools with ``parameters_schema != {}``
                (e.g., ``SegmentDistributionsAndCalculateMoments``) read
                their arguments from here.

        Returns:
            Structured output describing the invoked tool and all emitted
            artifacts (image, json, table, text, error).
        """
        ...


class FitState(Typed, Registry):
    """Immutable snapshot of the best model's fit results after a pipeline step.

    This is the base Registry class with fields common to all domains.
    Each domain defines a concrete subclass that adds domain-specific fields:

    - ``DistFittingFitState`` adds ``ans`` (the VLM response accumulator).
    - ``TimeSeriesFitState`` adds ``trend`` and ``kernels`` (best model only).

    At step 0 (before any model is fitted), callers pass ``fit_state=None``
    instead of constructing a FitState.  All consumers check
    ``if fit_state is None`` to handle the no-model case.

    Domain-specific tools receive their domain's concrete subclass at runtime.
    They access subclass fields directly without isinstance checks — if the
    wrong subclass is passed, that is a bug in the caller (AttributeError).
    """

    map_estimate: Dict[str, Any]
    family_name: List[str]


class DomainPrompts(Typed, Registry, ABC):
    """Domain identity: prompt rendering, VLM response parsing, data shapes.

    Each domain registers a concrete subclass under shared aliases so that
    ``DomainPrompts.of("distribution-fitting")`` resolves correctly.
    """

    task_string: ClassVar[str]

    # ── Response type ────────────────────────────────────────────────────

    @abstractmethod
    def get_response_type(self) -> Type[Typed]:
        """Return the Typed class that VLM proposal responses should parse into.

        The returned class (e.g. ``DistFittingVLMResponse``) is passed to
        ``backend.call(response_type=...)`` which constructs a SlowBurn
        validator that:

        1. Parses JSON from the raw VLM text.
        2. Constructs the Typed class via Pydantic coercion.
        3. Raises ``ValueError`` on schema mismatch → SlowBurn retries.
        4. Returns the validated Typed instance on success.

        This replaces hand-written validator functions with Pydantic's
        built-in field validation (``@field_validator``, type coercion,
        required-field enforcement).
        """
        ...

    @abstractmethod
    def render_proposal_prompt(self, *, num_proposals: int) -> str: ...

    @abstractmethod
    def render_code_gen_prompt(self, *, entity_value: Any, priors: Dict[str, str]) -> str: ...

    @abstractmethod
    def render_code_repair_prompt(
        self,
        *,
        base_prompt: str,
        previous_code: str,
        error_message: str,
        repair_context: str,
    ) -> str: ...

    @abstractmethod
    def get_feedback_prompt_template(self, *, num_proposals: int, max_steps: int) -> str: ...

    @abstractmethod
    def render_initial_summary(
        self,
        *,
        entity_value: Any,
        pymc_code: str,
        description: str,
        aic_score: float,
        plot_description: str,
    ) -> str: ...

    @abstractmethod
    def render_feedback_summary(
        self,
        *,
        entity_value: Any,
        pymc_code: str,
        description: str,
        aic_score: float,
        plot_description: str,
        tool_name: str,
        tool_output_type: str,
        tool_output_summary: str,
    ) -> str: ...

    @abstractmethod
    def get_entity_key(self) -> str: ...

    @abstractmethod
    def build_ans_dict(self, *, description: str) -> Dict[str, Any]: ...

    @abstractmethod
    def extract_proposal_fields(
        self,
        *,
        proposal_config: Dict[str, Any],
        ans: Dict[str, Any],
        ix: str,
    ) -> Tuple[Any, Dict[str, str]]: ...

    @abstractmethod
    def extract_dataset_fields(self, dataset: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def build_step_record_extras(
        self,
        *,
        ans: Dict[str, Any],
        fit_state: FitState,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def build_result_extras(
        self,
        *,
        dataset: Dict[str, Any],
        fit_state: FitState,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def should_log_map_estimate(self) -> bool: ...

    @abstractmethod
    def get_plot_type_descriptions(self) -> Dict[str, str]: ...


class DomainToolkit(Typed, Registry, ABC):
    """Expert tool schemas + tool execution dispatch.

    Tool implementation functions remain as module-level functions in each
    domain's ``toolkit.py``. This class calls them but does not contain them.
    """

    @abstractmethod
    def get_expert_tools(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def execute_tool(
        self,
        *,
        selected_tool: Optional[str],
        selected_tool_args: Dict[str, Any],
        data: Any,
        fit_state: Optional[FitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        plot_type_descriptions: Dict[str, str],
    ) -> DiagnosticToolResult: ...

    def supports_dynamic_generation(self) -> bool:
        """Whether this domain supports generate_only / dynamic toolkit modes."""
        return False


class DomainPlotting(Typed, Registry, ABC):
    """Domain-specific data visualization and fit state extraction.

    Owns ``extract_fit_state()`` because plotting must know the fit state
    shape to render overlays.
    """

    @abstractmethod
    def plot_initial_data(self, data: Any, *, save_path: str) -> None: ...

    @abstractmethod
    def plot_fit_overlay(
        self,
        *,
        data: Any,
        fit_state: FitState,
        path: str,
        best_idx: Union[str, int],
    ) -> None: ...

    @abstractmethod
    def get_default_plot_description(self) -> str: ...

    @abstractmethod
    def extract_fit_state(
        self,
        *,
        fit_results: Dict[str, Any],
        best_idx: Union[str, int],
        ans: Dict[str, Any],
    ) -> FitState: ...
