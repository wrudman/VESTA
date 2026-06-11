"""ExperimentConfig — validated, CLI-enabled experiment configuration.

Replaces the 150-line argparse block and bare ``Dict[str, Any]`` model configs
with a single ``pydantic-settings`` ``BaseSettings`` class.  Constructed from
CLI args in ``__main__``, or programmatically in notebooks/tests.

All enum-valued fields use Morphic ``AutoEnum`` (defined in
``experiment_enums.py``) rather than ``Literal`` tuples.  AutoEnum gives us
fuzzy-matched string inputs (so ``"distribution-fitting"`` and
``"distribution_fitting"`` both resolve to the same member), JSON-safe
serialization, and static-type-checker support.

CLI usage (auto-generated from field definitions)::

    python experiments.py \\
        --model.id azure/gpt-5-mini \\
        --data-pkl data_single.pkl \\
        --max-steps 3 \\
        --toolkit.mode expert \\
        --output.expt baseline

Notebook usage::

    from vesta import ExperimentConfig
    from vesta.core.experiment_config import ModelConfig, ToolkitConfig, OutputConfig

    config = ExperimentConfig(
        model=ModelConfig(litellm_model="azure/gpt-5-mini"),
        toolkit=ToolkitConfig(mode="expert"),
        output=OutputConfig(expt="my_experiment"),
        data_pkl="data_single.pkl",
        max_steps=3,
    )
"""

import logging
import os
from typing import Any, Dict, Optional

from morphic import Typed, validate
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from vesta.core.experiment_enums import (
    CarryForwardStrategy,
    Domain,
    OutputFormat,
    PyTensorMode,
    ReasoningEffort,
    ToolkitMode,
)

logger: logging.Logger = logging.getLogger(__name__)


class ModelConfig(Typed):
    """VLM backend configuration.  CLI prefix: ``--model.*``."""

    litellm_model: str = Field(
        description="LiteLLM model string (e.g. azure/gpt-5-mini)",
        validation_alias=AliasChoices("litellm_model", "id"),
    )
    max_tokens: int = Field(default=16000, ge=1, description="Max output tokens per LLM call")
    temperature: float = Field(default=0.7, ge=0.0, description="Sampling temperature")
    backend: str = Field(default="api", description="VLM backend Registry key")
    reasoning_effort: ReasoningEffort = Field(default=ReasoningEffort.low, description="LLM reasoning effort")
    num_retries: int = Field(default=5, ge=0, description="Retries on transient VLM/API failures")
    retry_wait: float = Field(default=1.0, ge=0.0, description="Base wait (seconds) between retry attempts")
    retry_algorithm: str = Field(
        default="Linear",
        description="Retry backoff strategy: 'Linear', 'Exponential', or 'Fibonacci'",
    )
    call_timeout: float = Field(default=300, gt=0.0, description="Per-call timeout in seconds")
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Optional OpenAI-compatible API base URL for LiteLLM calls, such as "
            "http://localhost:20128/v1 for 9router. When set, calls are routed "
            "through LiteLLM's OpenAI-compatible provider path."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional API key for the custom endpoint. For standard providers "
            "(azure, bedrock, etc.), LiteLLM reads keys from environment variables. "
            "For custom endpoints like OpenCode Go, pass the key explicitly."
        ),
    )
    litellm_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Extra provider-specific LiteLLM params forwarded verbatim to the "
            "backend. These take precedence over the reasoning_effort/api_base "
            "params computed from the other fields: any key passed here "
            "overrides the computed value. Pass from a Harbor config, the CLI, "
            "or a notebook to override or disable any provider behavior."
        ),
    )

    @classmethod
    def pre_initialize(cls, data: Dict[str, Any]) -> None:
        """Normalize API keys before Pydantic validates the config."""
        raw_key: Optional[Any] = data.get("api_key")
        if isinstance(raw_key, str) and len(raw_key) > 0:
            data["api_key"] = cls._strip_surrounding_quotes(raw_key)

        litellm_model: str = cls._resolve_raw_litellm_model(data=data)
        if litellm_model.startswith("openrouter/"):
            current_key: Optional[Any] = data.get("api_key")
            if current_key is None or (isinstance(current_key, str) and len(current_key) == 0):
                if "OPENROUTER_API_KEY" in os.environ:
                    env_key: str = os.environ["OPENROUTER_API_KEY"]
                    if len(env_key) > 0:
                        data["api_key"] = cls._strip_surrounding_quotes(env_key)

    @classmethod
    def _strip_surrounding_quotes(cls, value: str) -> str:
        """Remove one layer of matching shell-style quotes from an API key."""
        stripped: str = value.strip()
        if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
            return stripped[1:-1]
        elif len(stripped) >= 2 and stripped[0] == "'" and stripped[-1] == "'":
            return stripped[1:-1]
        else:
            return stripped

    @classmethod
    def _resolve_raw_litellm_model(cls, *, data: Dict[str, Any]) -> str:
        """Resolve the pre-validation LiteLLM model from field name or CLI alias."""
        if "litellm_model" in data and isinstance(data["litellm_model"], str):
            return data["litellm_model"]
        elif "id" in data and isinstance(data["id"], str):
            return data["id"]
        else:
            return ""

    def post_initialize(self) -> None:
        """Log the effective OpenRouter API key in masked form."""
        if not self._is_openrouter_model():
            return

        if self.api_key is not None and len(self.api_key) > 0:
            masked: str = self.api_key[:8] + "***" + self.api_key[-4:]
            logger.info(f"OpenRouter API key for {self.litellm_model}: {masked}")
        else:
            logger.warning(
                f"No OpenRouter API key for {self.litellm_model}. "
                f"Set OPENROUTER_API_KEY or pass --model.api-key."
            )

    def _is_anthropic_model(self) -> bool:
        """Check whether the model targets Anthropic's API natively.

        Does NOT match OpenRouter-prefixed Anthropic models
        (``openrouter/anthropic/...``) — those are routed through
        the OpenRouter provider path instead.
        """
        return (
            self.litellm_model.startswith("anthropic/")
            or self.litellm_model.startswith("bedrock/anthropic")
            or self.litellm_model.startswith("vertex_ai/claude")
        )

    def _is_openrouter_model(self) -> bool:
        """Check whether the model is routed through OpenRouter."""
        return self.litellm_model.startswith("openrouter/")

    def _is_azure_model(self) -> bool:
        """Check whether the model targets Azure OpenAI."""
        return self.litellm_model.startswith("azure/")

    def _is_gpt_5_4_model(self) -> bool:
        """Check whether the model is a GPT-5.4 variant."""
        return "gpt-5.4" in self.litellm_model

    def _is_openrouter_sonnet_4_6(self) -> bool:
        """Check whether this is Sonnet 4.6 routed through OpenRouter."""
        return self._is_openrouter_model() and "claude-sonnet-4.6" in self.litellm_model

    def _to_litellm_params(self) -> Dict[str, Any]:
        """Build provider-specific LiteLLM params for backend construction.

        Reasoning-effort mapping is provider-aware:
        - OpenRouter Sonnet 4.6: ``extra_body.reasoning`` with ``max_tokens``
          budget (token-capped thinking). ``low`` → 1024 tokens; ``none`` →
          disabled.  ``medium``/``high`` raise — only budget-controlled thinking
          is supported for this model.
        - Other OpenRouter: ``extra_body.reasoning`` with ``effort`` string.
        - Azure GPT-5.4: dict ``{"effort": ..., "summary": "detailed"}``
          (works with both Responses API and Chat Completions via
          LiteLLM's GPT-5 normalization).
        - Native Anthropic: ``output_config`` (compatible with forced
          tool use).
        - Everything else: plain ``reasoning_effort`` string.

        User-supplied ``litellm_params`` always take precedence: the
        provider-specific reasoning/api_base params are computed first, then
        any keys present in ``self.litellm_params`` overwrite them. This lets a
        caller (Harbor config, CLI, or notebook) override or disable any
        computed param by passing the same key explicitly.
        """
        merged_litellm_params: Dict[str, Any] = {}

        if self._is_openrouter_sonnet_4_6():
            self._add_openrouter_sonnet_reasoning(merged_litellm_params=merged_litellm_params)
        elif self.reasoning_effort is ReasoningEffort.none:
            self._add_disabled_reasoning(merged_litellm_params=merged_litellm_params)
        elif self._is_openrouter_model():
            self._add_openrouter_effort_reasoning(merged_litellm_params=merged_litellm_params)
        elif self._is_azure_model() and self._is_gpt_5_4_model():
            # Azure GPT-5.4: dict format works with Responses API and
            # gets normalized to string for Chat Completions by
            # LiteLLM's GPT-5 transformation.
            merged_litellm_params["reasoning_effort"] = {
                "effort": self.reasoning_effort.value,
                "summary": "detailed",
            }
        elif self._is_anthropic_model():
            # Anthropic Claude 4.6: pass output_config directly.
            # Using reasoning_effort triggers BOTH thinking (extended
            # thinking) AND output_config.  Anthropic rejects thinking
            # when tool_choice forces tool use, so we pass output_config
            # to set effort without enabling extended thinking.
            merged_litellm_params["output_config"] = {"effort": self.reasoning_effort.value}
        else:
            merged_litellm_params["reasoning_effort"] = self.reasoning_effort.value

        if self.api_base is not None:
            merged_litellm_params["api_base"] = self.api_base
            merged_litellm_params["custom_llm_provider"] = "openai"

        # User-supplied litellm_params always win: apply them last so any key
        # the caller passes (e.g. reasoning_effort, extra_body, api_base)
        # overrides the provider-specific value computed above.
        if self.litellm_params is not None:
            merged_litellm_params.update(self.litellm_params)

        return merged_litellm_params

    def _add_openrouter_sonnet_reasoning(
        self,
        *,
        merged_litellm_params: Dict[str, Any],
    ) -> None:
        """Configure OpenRouter Sonnet 4.6 token-budgeted reasoning."""
        if self.reasoning_effort is ReasoningEffort.none:
            extra_body: Dict[str, Any] = self._get_extra_body(merged_litellm_params=merged_litellm_params)
            extra_body["reasoning"] = {"enabled": False}
        elif self.reasoning_effort is ReasoningEffort.low:
            extra_body: Dict[str, Any] = self._get_extra_body(merged_litellm_params=merged_litellm_params)
            extra_body["reasoning"] = {
                "max_tokens": 1024,
                "exclude": False,
            }
        else:
            raise ValueError(
                f"Sonnet 4.6 via OpenRouter only supports reasoning_effort "
                f"'low' or 'none'; got {self.reasoning_effort.value!r}. "
                f"This model uses token-budgeted thinking (max_tokens), "
                f"not effort levels."
            )

    def _add_disabled_reasoning(
        self,
        *,
        merged_litellm_params: Dict[str, Any],
    ) -> None:
        """Configure provider-specific disabled reasoning when needed."""
        if self._is_openrouter_model():
            # Explicitly request none for reproducibility — omitting
            # the reasoning object altogether would defer to OpenRouter's
            # default, which may vary by model.
            extra_body: Dict[str, Any] = self._get_extra_body(merged_litellm_params=merged_litellm_params)
            extra_body["reasoning"] = {
                "effort": "none",
                "exclude": False,
            }

    def _add_openrouter_effort_reasoning(
        self,
        *,
        merged_litellm_params: Dict[str, Any],
    ) -> None:
        """Configure OpenRouter effort-based reasoning."""
        extra_body: Dict[str, Any] = self._get_extra_body(merged_litellm_params=merged_litellm_params)
        extra_body["reasoning"] = {
            "effort": self.reasoning_effort.value,
            "exclude": False,
        }

    def _get_extra_body(
        self,
        *,
        merged_litellm_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return the mutable LiteLLM extra_body dict, creating it if absent."""
        if "extra_body" not in merged_litellm_params:
            merged_litellm_params["extra_body"] = {}
        extra_body: Dict[str, Any] = merged_litellm_params["extra_body"]
        return extra_body

    @validate
    def to_backend_kwargs(self, *, verbosity: int, max_rpm: int) -> Dict[str, Any]:
        """Assemble the dict that ``VLMBackend.of()`` expects.

        Merges ``reasoning_effort`` and custom endpoint routing into
        ``litellm_params``. Includes ``max_rpm`` from the parallelism config.
        """
        backend_kwargs: Dict[str, Any] = {
            "backend": self.backend,
            "litellm_model": self.litellm_model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "verbosity": verbosity,
            "num_retries": self.num_retries,
            "retry_wait": self.retry_wait,
            "retry_algorithm": self.retry_algorithm,
            "call_timeout": self.call_timeout,
            "max_rpm": max_rpm,
            "litellm_params": self._to_litellm_params(),
        }
        if self.api_key is not None:
            backend_kwargs["api_key"] = self.api_key
        return backend_kwargs


class ToolkitConfig(Typed):
    """Toolkit configuration.  CLI prefix: ``--toolkit.*``."""

    mode: ToolkitMode = Field(description="Toolkit mode")
    code_gen_model: Optional[str] = Field(
        default=None,
        description=(
            "Optional LiteLLM model override for PyMC proposal code generation. "
            "When omitted, proposal code generation reuses the main model."
        ),
    )
    tool_gen_model: Optional[str] = Field(
        default=None,
        description=(
            "Optional LiteLLM model override for dynamic diagnostic-tool generation. "
            "When omitted, tool generation reuses the main model."
        ),
    )
    code_gen_temperature: float = Field(default=0.3, description="Code-gen sampling temperature")
    max_code_generation_attempts: int = Field(
        default=1,
        ge=1,
        description="Maximum generate-execute-repair attempts for each PyMC model proposal.",
    )
    max_tool_generation_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum generate-execute-repair attempts for each generated dynamic tool.",
    )
    force_tool_call: CliImplicitFlag[bool] = Field(default=False, description="Demand tool call each step")
    max_tool_calls_per_step: int = Field(
        default=1,
        ge=0,
        description="Max diagnostic tool executions per step (one per diagnostic round)",
    )
    accumulate_tools: CliImplicitFlag[bool] = Field(default=False, description="Load/save dynamic tools")
    tools_max_size: int = Field(default=-1, description="Registry cap (-1=unlimited)")
    tool_registry_filename: str = Field(
        default="tool_registry.json",
        description=(
            "Filename (relative to the vesta package directory) "
            "for persisting dynamic tools across runs. Only consulted when "
            "accumulate_tools is set; each run seeds its dynamic_tools dict "
            "from this file at start and writes it back at end."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_accumulated_only_explicit_registry(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Require an explicit persisted registry filename for accumulated-only evaluation."""
        if "mode" not in data:
            return data
        is_accumulated_only: bool = (
            data["mode"] == "accumulated_only" or data["mode"] is ToolkitMode.accumulated_only
        )
        if is_accumulated_only and "tool_registry_filename" not in data:
            raise ValueError(
                "toolkit.mode='accumulated_only' requires explicitly passing "
                "--toolkit.tool-registry-filename. This mode evaluates only persisted "
                "dynamic tools from that registry file; it does not generate tools and "
                "does not use expert tools."
            )
        return data

    @model_validator(mode="after")
    def _validate_accumulated_only_requirements(self) -> "ToolkitConfig":
        """Require read-only accumulated-only evaluation with a non-empty registry filename."""
        if self.mode is ToolkitMode.accumulated_only:
            if self.accumulate_tools:
                raise ValueError(
                    "toolkit.mode='accumulated_only' reads an existing dynamic-tool registry "
                    "read-only and does not accept --toolkit.accumulate-tools. Use "
                    "toolkit.mode='generate_only' with --toolkit.accumulate-tools for the "
                    "sequential train/build phase, then toolkit.mode='accumulated_only' with "
                    "--toolkit.tool-registry-filename for the held-out evaluation phase."
                )
            if len(self.tool_registry_filename) == 0:
                raise ValueError(
                    "toolkit.mode='accumulated_only' requires a non-empty "
                    "--toolkit.tool-registry-filename so the persisted dynamic-tool registry is explicit."
                )
        return self

    @validate
    def to_code_backend_kwargs(
        self,
        *,
        model: "ModelConfig",
        model_override: Optional[str],
        verbosity: int,
        code_gen_max_rpm: int,
    ) -> Optional[Dict[str, Any]]:
        """Derive backend kwargs for a stage-specific code-generation model.

        Returns ``None`` when no stage-specific override is configured so the
        caller can reuse the main backend.
        """
        if model_override is None:
            return None

        code_backend_kwargs: Dict[str, Any] = model.to_backend_kwargs(
            verbosity=verbosity,
            max_rpm=code_gen_max_rpm,
        )
        code_backend_kwargs["litellm_model"] = model_override
        code_backend_kwargs["temperature"] = self.code_gen_temperature
        return code_backend_kwargs


class ParallelismConfig(Typed):
    """Parallelism configuration.  CLI prefix: ``--parallel.*``."""

    nproc: int = Field(default=0, ge=0, description="Outer workers (0=sync)")
    nthread: int = Field(default=0, ge=0, description="Inner threads per outer worker (0=sync)")
    compute_threads: int = Field(
        default=1,
        ge=-1,
        description=(
            "Max BLAS/OpenMP threads per process (OPENBLAS_NUM_THREADS, "
            "OMP_NUM_THREADS, MKL_NUM_THREADS, VECLIB_MAXIMUM_THREADS, "
            "NUMEXPR_NUM_THREADS). Default 1 caps all BLAS backends to a "
            "single thread, which is optimal for n<~1000 PyMC GP workloads "
            "and avoids openblas-pthreads x libgomp oversubscription on "
            "Linux (OpenBLAS#3187, pymc#6640). Set -1 to leave env vars "
            "untouched (opt-out sentinel); set N>1 to cap at N threads. "
            "Applied via _thread_caps.py at import time — see that module "
            "for the full rationale."
        ),
    )
    max_rpm: int = Field(
        default=120,
        ge=0,
        description="TOTAL RPM budget for the main proposal/summary LLM. Auto-divided by concurrency.",
    )
    code_gen_max_rpm: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "TOTAL RPM budget for the code-generation LLM. Auto-divided by concurrency. "
            "If omitted, reuses max_rpm."
        ),
    )

    @property
    def total_concurrency(self) -> int:
        """Number of concurrent dataset workers driving LLM calls."""
        return max(1, self.nproc) * max(1, self.nthread)

    @property
    def per_llm_rpm(self) -> int:
        """Per-worker RPM budget for the main LLM after dividing by concurrency."""
        return self._per_worker_rpm(total_rpm=self.max_rpm)

    @property
    def resolved_code_gen_max_rpm(self) -> int:
        """Total code-generation RPM budget, falling back to the main budget when unset."""
        if self.code_gen_max_rpm is None:
            return self.max_rpm
        return self.code_gen_max_rpm

    @property
    def per_code_gen_llm_rpm(self) -> int:
        """Per-worker RPM budget for the code-generation LLM after dividing by concurrency."""
        return self._per_worker_rpm(total_rpm=self.resolved_code_gen_max_rpm)

    @validate
    def _per_worker_rpm(self, *, total_rpm: int) -> int:
        """Divide a total RPM budget across concurrent dataset workers."""
        per_worker_rpm: int = total_rpm // self.total_concurrency
        if total_rpm > 0 and per_worker_rpm == 0:
            return 1
        return per_worker_rpm


class OutputConfig(Typed):
    """Output configuration.  CLI prefix: ``--output.*``."""

    expt: str = Field(description="Experiment name. Becomes the top-level dir under outputs/.")
    base_dir: str = Field(default="outputs", description="Root outputs directory")
    output_format: OutputFormat = Field(default=OutputFormat.parquet, description="File format for results")


class ExperimentConfig(BaseSettings):
    """Top-level experiment configuration.

    Constructed from CLI via ``ExperimentConfig(_cli_parse_args=True)``
    or programmatically via ``ExperimentConfig(model=ModelConfig(...), ...)``.

    The ``run_verbosity`` property provides the effective verbosity for
    ``run()`` calls: reduced by 1 for multi-dataset runs (>1 dataset)
    so progress bars don't drown in per-step logs.  This lives here
    because ExperimentConfig is the source of truth for all config
    — the reduction logic should not be scattered in ``run_all()``.
    """

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_enforce_required=True,
        cli_prog_name="experiments.py",
        env_prefix="PYMC_",
    )

    domain: Domain = Field(default=Domain.distribution_fitting, description="Problem domain")
    data_pkl: str = Field(description="Path to pkl file")
    dataset_idx: Optional[str] = Field(default=None, description="Index spec: single, comma, slice")
    max_steps: int = Field(ge=0, description="Feedback iterations after step 0")
    proposals_per_step: int = Field(default=3, ge=1, description="Model proposals per VLM step")
    carry_forward: CarryForwardStrategy = Field(
        default=CarryForwardStrategy.best,
        description=(
            "Which fitted model is passed to the next step's diagnostic + "
            "proposal phases: 'best' (lowest-AIC-so-far, default) or 'latest' "
            "(most-recent fit). Affects the fit_state/image/AIC/code "
            "shown to the VLM and the prompt wording. Both 'current_*' "
            "and 'best_*' views are always recorded in the final run log."
        ),
    )
    verbosity: int = Field(
        default=2,
        ge=0,
        le=3,
        description="0=silent, 1=progress, 2=detailed, 3=debug+litellm",
    )
    pytensor_mode: PyTensorMode = Field(
        default=PyTensorMode.FAST_RUN,
        description="PyTensor compilation mode",
    )

    model: ModelConfig = Field(description="VLM backend config")
    toolkit: ToolkitConfig = Field(description="Toolkit config")
    parallel: ParallelismConfig = Field(default_factory=ParallelismConfig, description="Parallelism config")
    output: OutputConfig = Field(description="Output config")

    @validate
    def get_run_verbosity(self, *, num_datasets: int) -> int:
        """Effective verbosity for run() calls.

        For multi-dataset runs, verbosity 1/2 are reduced by one level so
        progress bars are not drowned in per-step detail logs. Verbosity 3
        is preserved and propagated unchanged to backend layers for deep
        transport-level debugging.
        """
        if num_datasets <= 1:
            effective_verbosity: int = self.verbosity
        elif self.verbosity == 0:
            effective_verbosity = 0
        elif self.verbosity == 1:
            effective_verbosity = 0
        elif self.verbosity == 2:
            effective_verbosity = 1
        elif self.verbosity == 3:
            effective_verbosity = 3
        else:
            raise ValueError(f"Unsupported verbosity={self.verbosity!r}. Expected one of 0, 1, 2, or 3.")
        return effective_verbosity

    @model_validator(mode="after")
    def _validate_cross_field(self) -> "ExperimentConfig":
        """Cross-field validation rules."""
        return self
