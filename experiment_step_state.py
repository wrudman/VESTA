"""Immutable per-step state and observation models for the VLM fitting pipeline.

Naming hierarchy (used consistently across this module and
``experiments.py``):

    run   →  steps (step 0, 1, ..., max_steps)
    step  →  phases (diagnostic, proposal, codegen, fit, plot, summary)
    phase →  rounds (only the diagnostic phase iterates rounds of tool
              calls; see ``_run_diagnostic_rounds``)

A step's lifecycle is tracked by ``StepStatus`` (running / complete /
error) — deliberately a "status" rather than a "phase", because
"phase" is reserved for the six phases that run within a step.

The ``run()`` function in ``experiments.py`` is structured as a
three-part cycle (Functional Core / Imperative Shell + per-field
reducer):

1. ``_execute_step`` (impure SHELL): runs the six phases for one step
   and produces a ``StepObservation`` describing what happened.
2. ``_reduce_step`` (pure CORE): takes ``(state, observation)`` and
   returns a new ``StepState``. No I/O, no mutation. This is the single
   location where state evolves — adding a new trajectory field is a
   one-line change here.
3. Outer loop: threads the state forward, persists step records to
   disk, updates the progress bar, and terminates when
   ``state.status != StepStatus.running``.

The state and observation models are frozen ``Typed`` (Pydantic) classes
with ``Tuple[...]`` collections so that mutation is impossible after
construction. Updates use ``validated_copy(state=..., update={...})``
(defined below) to produce a new, re-validated snapshot for the next
iteration. We deliberately avoid ``state.model_copy(update=...)`` because
Pydantic v2 does not re-validate on copy, so a typo in the update dict
would silently attach a stray attribute while leaving the real field
unchanged.

The ``summary_trajectory`` field unifies step 0's initial summary with
the per-step feedback summaries into a single append-only tuple:
``trajectory[0]`` is step 0's initial summary, ``trajectory[i]`` is the
summary produced at step ``i``. Every step appends its own summary here
and the feedback prompt builder reads from this single source of truth.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from morphic import Typed, validate
from pydantic import ConfigDict

from domains import (
    CodeGenerationAttempt,
    DiagnosticToolResult,
    DomainPlotting,
    DomainPrompts,
    DomainToolkit,
    FitState,
)
from dynamic_toolkit import DynamicToolSpec, GeneratedToolExecutionResult
from experiment_enums import (
    CarryForwardStrategy,
    Domain,
    ObservationKind,
    PhaseName,
    StepStatus,
    ToolkitMode,
)

# ═══════════════════════════════════════════════════════════════════════════
# RunDeps — frozen container of run-scope invariants
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class RunDeps:
    """Run-scope invariants passed to every phase + reducer.

    These values are constructed once at the top of ``run()`` and do not
    change across iterations. Bundling them into a single frozen container
    keeps phase signatures compact while making it obvious that none of the
    fields are per-step mutable state.

    ``backend``, optional code-generation backends, ``prompts_reg``, ``toolkit_reg``,
    and ``plotting_reg`` are fully-constructed Typed/Registry instances
    (not proxies), but we use a plain frozen dataclass here — not a
    Typed — because ``ProgressBar`` is a concurry object that does not
    play well with Pydantic's strict field validation.

    ``dynamic_tools`` is the **one intentionally-mutable field** in
    RunDeps: a ``Dict[str, DynamicToolSpec]`` owned by this run.
    ``RunDeps`` is frozen so the reference cannot be reassigned, but
    the dict's contents may be mutated (``dynamic_tools[name] = spec``)
    by ``_run_diagnostic_rounds`` after a successful
    ``handle_generate_new_tool`` call.  This scopes LLM-generated
    tools per-run: each ``run()`` call owns its own dict, so dyn
    tools registered by dataset N never leak into dataset N+1 in the
    same process.  Seeded from ``config.toolkit.tool_registry_filename``
    when ``accumulate_tools`` is true, else starts empty.
    """

    backend: Any  # VLMBackend — validated at construction in run()
    code_gen_backend: Any  # VLMBackend used for PyMC proposal code generation
    code_gen_model: str  # the LiteLLM model id used for PyMC proposal code generation
    tool_gen_backend: Any  # VLMBackend used for dynamic diagnostic-tool generation
    tool_gen_model: str  # the LiteLLM model id used for dynamic diagnostic-tool generation
    prompts_reg: DomainPrompts
    toolkit_reg: DomainToolkit
    plotting_reg: DomainPlotting
    pbar: Any  # concurry ProgressBar — not a Typed/Pydantic type
    data: Any  # numpy array or pandas Series
    dataset_fields: Dict[str, Any]
    out_dir: str
    verbosity: int
    max_steps: int
    domain: Domain
    toolkit_mode: ToolkitMode
    max_tool_calls_per_step: int
    max_code_generation_attempts: int
    max_tool_generation_attempts: int
    force_tool_call: bool
    accumulate_tools: bool
    carry_forward: CarryForwardStrategy
    response_type: Type[Typed]
    entity_key: str
    plot_type_descriptions: Dict[str, str]
    model_spec_proposal_prompt_template: str
    model_spec_feedback_prompt_template: str
    dynamic_tools: Dict[str, DynamicToolSpec]


# ═══════════════════════════════════════════════════════════════════════════
# StepState — immutable per-step snapshot. The ONLY place state evolves is
# inside _reduce_step. Every field that needs to be remembered across steps
# lives here; adding a new field is a one-line change here plus a one-line
# append in _reduce_step.
# ═══════════════════════════════════════════════════════════════════════════


class StepState(Typed):
    """Immutable per-step snapshot threaded through the refinement loop.

    Two design invariants:

    1. **Every collection field is a ``Tuple[...]`` (not ``List[...]``)** so
       that accidentally calling ``.append()`` raises ``AttributeError``.
       Typed's ``frozen=True`` only blocks attribute assignment on the
       model itself; tuple contents stay immutable.
    2. **``step_num = -1`` is the sentinel pre-step-0 state.** The outer
       loop's first call to ``_execute_step`` increments it to 0. This
       eliminates the peeled "Step 0 block" — there is one unified loop
       body, and ``state.step_num == 0`` simply means "we are about to
       run step 0".

    ``summary_trajectory`` unifies step 0's initial summary with the
    per-step feedback summaries. At the end of step ``i``, the reducer
    appends ``observation.summary`` so that:

    - ``trajectory[0]`` is step 0's initial summary
    - ``trajectory[i]`` is step ``i``'s feedback summary

    The feedback-prompt builder reads this single source of truth — it
    fills the ``{initial_summary}`` slot from ``trajectory[0]`` and each
    ``{feedback_summary_step_i}`` slot from ``trajectory[i]``.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=False,
    )

    # ── Lifecycle ──────────────────────────────────────────────────────
    step_num: int = -1
    status: StepStatus = StepStatus.running
    error: Optional[str] = None
    error_phase: Optional[PhaseName] = None

    # ── Current view (inputs to the next step) ─────────────────────────
    # At step_num == -1 (pre-step-0), current_* are None; ``fit_path`` is
    # the initial histogram so it can be shown to the VLM at step 0.
    current_model: Optional[str] = None
    current_model_structure: Optional[Union[str, List[str]]] = None
    current_map_estimate: Optional[Dict[str, Any]] = None
    current_metrics: Optional[Dict[str, Any]] = None
    current_fit_state: Optional[FitState] = None
    fit_path: str

    # ── Monotone global best ───────────────────────────────────────────
    # ``best_*`` fields track the lowest-AIC model seen so far. They are
    # only updated on AIC improvement (see _reduce_step). They exist in
    # addition to ``current_*`` so that the final run record can always
    # report both views and so that the ``carry_forward='best'`` strategy
    # has a coherent payload to hand to the next step's prompts/tools.
    best_aic: float = float("inf")
    best_model_structure: Optional[Union[str, List[str]]] = None
    best_model_code: Optional[str] = None
    best_map_estimate: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_fit_state: Optional[FitState] = None
    best_fit_path: Optional[str] = None

    # ── Trajectories (append-only) ─────────────────────────────────────
    tested_model_structures: Tuple[Union[str, List[str]], ...] = ()
    selected_tool_history: Tuple[str, ...] = ()
    summary_trajectory: Tuple[str, ...] = ()
    step_records: Tuple[Dict[str, Any], ...] = ()

    @property
    def is_terminal(self) -> bool:
        """True when the loop should stop after reducing this state."""
        return self.status is not StepStatus.running


class CarriedModel(Typed):
    """Read-only view of the fitted model being carried forward into the
    next step's diagnostic + proposal phases.

    Every field mirrors the corresponding slot on ``StepState`` but the
    specific source (``current_*`` vs ``best_*``) is chosen by
    ``carried_model()`` according to the run's
    ``CarryForwardStrategy``. Using this tiny struct as the interface for
    phase builders (instead of reading ``state.current_*`` / ``state.best_*``
    directly) guarantees that the model, structure, AIC, code, and fit
    overlay image shown to the VLM always agree with each other — the
    prior bug where ``current_model_structure`` was paired with
    ``state.best_aic`` cannot recur once all consumers route through this
    struct.

    ``fit_path`` is never None even before step 0 completes: at
    ``step_num == -1`` it points at the initial raw-data histogram written
    by ``_write_initial_plot``. ``model_structure`` / ``model_code`` /
    ``map_estimate`` / ``metrics`` / ``fit_state`` are ``None`` at the
    pre-step-0 point and become populated once the first step fits a
    model.

    ``label`` is the short human-readable tag ("latest" or "best") that
    prompt builders can drop into user-facing strings without needing to
    branch on the strategy themselves.
    """

    model_structure: Optional[Union[str, List[str]]]
    model_code: Optional[str]
    map_estimate: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, Any]]
    fit_state: Optional[FitState]
    fit_path: str
    label: str


@validate
def carried_model(*, state: StepState, strategy: CarryForwardStrategy) -> CarriedModel:
    """Select the fitted model that should be carried into the next step.

    Both strategies read from the same ``StepState`` — ``latest`` uses the
    ``current_*`` slice (updated every step unconditionally), ``best``
    uses the ``best_*`` slice (updated only on AIC improvement). When
    ``strategy is best`` and no model has improved yet (e.g. at step 0
    before any fit or on a run where every step failed), we fall back to
    the ``current_*`` slice — otherwise the VLM would be shown "no model
    yet" forever despite step 0 having produced a fit. This matches what
    a human would expect: "best model so far, and if there isn't one use
    whatever we have."

    The returned ``CarriedModel`` is a frozen dataclass, so accidental
    mutation is impossible and passing it through multiple phase
    builders is safe.
    """
    if strategy is CarryForwardStrategy.best and state.best_fit_state is not None:
        assert state.best_fit_path is not None, (
            "best_fit_state is set but best_fit_path is None — "
            "reducer invariant violated"
        )
        return CarriedModel(
            model_structure=state.best_model_structure,
            model_code=state.best_model_code,
            map_estimate=state.best_map_estimate,
            metrics=state.best_metrics,
            fit_state=state.best_fit_state,
            fit_path=state.best_fit_path,
            label="best",
        )
    return CarriedModel(
        model_structure=state.current_model_structure,
        model_code=state.current_model,
        map_estimate=state.current_map_estimate,
        metrics=state.current_metrics,
        fit_state=state.current_fit_state,
        fit_path=state.fit_path,
        label="latest",
    )


@validate
def validated_copy(*, state: StepState, update: Dict[str, Any]) -> StepState:
    """Re-validating replacement for ``state.model_copy(update=...)``.

    ``BaseModel.model_copy(update=...)`` silently accepts update keys that
    are not declared fields on the model (Pydantic v2 does not re-validate
    on copy), which means a typo like ``update={"phase": StepStatus.complete}``
    gets attached as a stray attribute while ``state.status`` remains
    ``running`` and the outer loop runs forever. This helper forces a full
    re-validation so unknown keys and type mismatches fail loudly.

    Reference semantics (verified): values whose annotated type is an
    arbitrary class (numpy arrays, PyMC models, nested frozen ``Typed``
    instances that are accepted via the default
    ``revalidate_instances='never'``) pass through by reference — no
    deep copy. Only outer collection wrappers (``Tuple[...]``, ``Dict[...]``,
    ``List[...]``) get rebuilt shallowly as part of normal Pydantic
    validation; their contents remain the same refs.

    Args:
        state: The current immutable snapshot. Its type is used to drive
            validation so subclasses (if any) retain their identity.
        update: Mapping of field-name → new value. Must only contain keys
            that are declared fields on ``type(state)``; a ``ValidationError``
            is raised otherwise.

    Returns:
        A new instance of ``type(state)`` with ``update`` applied and all
        fields re-validated.

    Raises:
        pydantic.ValidationError: if ``update`` contains keys not declared
            on ``type(state)``, or if any field value fails type validation.
    """
    data: Dict[str, Any] = {k: getattr(state, k) for k in type(state).model_fields}
    data.update(update)
    return type(state).model_validate(data)


# ═══════════════════════════════════════════════════════════════════════════
# StepObservation — immutable record of what one call to _execute_step
# produced. Phases populate their fields; the reducer consumes it.
# ═══════════════════════════════════════════════════════════════════════════


class StepObservation(Typed):
    """Immutable description of what happened during a single call to
    ``_execute_step``.

    ``kind`` discriminates three shapes:

    - ``ok``: all phases completed; ``new_*`` fields are populated and
      the reducer advances state.
    - ``complete``: the VLM returned COMPLETE during the proposal phase
      (step > 0 only); no codegen/fit/summary were run. Reducer records
      this and sets ``status = complete``.
    - ``error``: one phase raised; ``error`` and ``error_phase`` pinpoint
      where. Reducer records the partial result and sets
      ``status = error``.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_default=True,
        validate_assignment=False,
    )

    # ── Discriminator ──────────────────────────────────────────────────
    kind: ObservationKind
    step_num: int
    error: Optional[str] = None
    error_phase: Optional[PhaseName] = None

    # ── Diagnostic phase outputs ───────────────────────────────────────
    diagnostic_prompt: Optional[str] = None
    diagnostic_tools_offered: Tuple[Dict[str, Any], ...] = ()
    tool_results: Tuple[DiagnosticToolResult, ...] = ()
    generated_tool_results: Tuple[GeneratedToolExecutionResult, ...] = ()
    selected_tool_names: str = "none"

    # ── Proposal phase outputs ─────────────────────────────────────────
    model_spec_prompt: Optional[str] = None
    model_spec_images: Tuple[str, ...] = ()
    model_spec_response_dict: Optional[Dict[str, Any]] = None
    phase2_call_time_s: float = 0.0

    # ── Codegen phase outputs ──────────────────────────────────────────
    model_code_generation_prompts: Tuple[str, ...] = ()
    model_code_generation_responses: Tuple[str, ...] = ()
    model_code_generation_attempts: Tuple[CodeGenerationAttempt, ...] = ()

    # ── Fit phase outputs ──────────────────────────────────────────────
    model_spec_state: Optional[Dict[str, Any]] = None
    fit_results: Optional[Dict[str, Any]] = None
    best_idx: Optional[Union[str, int]] = None
    new_model_code: Optional[str] = None
    new_model_structure: Optional[Union[str, List[str]]] = None
    new_map_estimate: Optional[Dict[str, Any]] = None
    new_metrics: Optional[Dict[str, Any]] = None
    new_fit_state: Optional[FitState] = None

    # ── Plot phase output ──────────────────────────────────────────────
    new_fit_path: Optional[str] = None

    # ── Summary phase output (appended to state.summary_trajectory) ────
    summary_prompt: Optional[str] = None
    summary: Optional[str] = None
