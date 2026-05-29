"""AutoEnum types used across the pipeline.

This module is intentionally a leaf: it imports only from ``morphic`` so that
every other module — config, state, dynamic_toolkit, experiments — can safely
import these enums without risking circular dependencies.

Every closed set of string values in the codebase must live here as an
``AutoEnum`` rather than a ``Literal`` tuple. ``AutoEnum`` members behave
as strings (subclass ``str``), support fuzzy-matched construction
(``ToolkitMode("generate-only")`` → ``ToolkitMode.generate_only``), are
type-checker friendly, and serialize cleanly via Pydantic's ``model_dump``.

Identity-based comparison is the convention in this codebase:
``mode is ToolkitMode.static`` — not ``mode == "static"``, because
``AutoEnum.__eq__`` is defined as identity on the singleton instance.
"""

from morphic import AutoEnum, auto


class StepStatus(AutoEnum):
    """Lifecycle state of the refinement loop.

    ``running`` means the loop is still iterating; ``complete`` means the
    VLM returned ``COMPLETE`` for the ``description`` field; ``error``
    means a phase raised and we terminated early.

    Named ``StepStatus`` (not ``StepPhase``) to preserve the project's
    naming hierarchy: a run contains *steps*, each step contains
    *phases* (diagnostic / proposal / codegen / fit / plot / summary),
    and the diagnostic phase contains *rounds*. The lifecycle of a step
    is therefore a status, not a phase.
    """

    running = auto()
    complete = auto()
    error = auto()


class ObservationKind(AutoEnum):
    """Discriminator on ``StepObservation`` — what happened during this step."""

    ok = auto()
    complete = auto()
    error = auto()


class PhaseName(AutoEnum):
    """Identifies which of the six pipeline phases within a step is running / failed.

    Used on ``StepObservation.error_phase`` to localise failures and in
    log messages so readers know exactly where the pipeline broke. Each
    step runs the phases in the order declared here; the diagnostic
    phase internally iterates *rounds* of tool calls (see
    ``_run_diagnostic_rounds``).
    """

    diagnostic = auto()
    proposal = auto()
    codegen = auto()
    fit = auto()
    plot = auto()
    summary = auto()


class ToolkitMode(AutoEnum):
    """Which toolkit the pipeline uses for diagnostic tool selection."""

    none = auto()
    static = auto()
    generate_only = auto()
    accumulated_only = auto()
    dynamic = auto()


class Domain(AutoEnum):
    """Problem domain. Kebab-case CLI values are accepted via AutoEnum fuzzy
    matching (``Domain('distribution-fitting')`` → ``Domain.distribution_fitting``)."""

    distribution_fitting = auto()
    time_series = auto()


class ReasoningEffort(AutoEnum):
    """LLM reasoning-effort level. Matches provider-accepted values."""

    none = auto()
    low = auto()
    medium = auto()
    high = auto()


class PyTensorMode(AutoEnum):
    """PyTensor compilation mode. Values match the exact strings PyTensor accepts."""

    NUMBA = auto()
    FAST_RUN = auto()
    FAST_COMPILE = auto()


class OutputFormat(AutoEnum):
    """Format for on-disk result files."""

    parquet = auto()


class RunStatus(AutoEnum):
    """Status field on the final run result dict."""

    ok = auto()
    error = auto()


class CarryForwardStrategy(AutoEnum):
    """Which fitted model is passed forward to the next step's diagnostic
    + proposal phases.

    The choice controls the fit_state / fit overlay image / AIC label /
    model structure label / PyMC code snippet shown to the VLM at the
    start of the next step. It does NOT change the reducer's bookkeeping
    of ``current_*`` (latest) and ``best_*`` (monotone minimum) fields
    — both continue to be tracked so the final run record always contains
    both views.

    ``latest``: carry the most recently fitted model forward, regardless
    of whether it improved on the best-so-far AIC. The VLM sees its own
    previous proposal's result (even if worse than an earlier step's),
    which lets it course-correct based on the latest mistake.

    ``best``: carry the lowest-AIC model seen so far forward. The VLM
    always builds on the best model; a regression at step N does not
    propagate into step N+1's context. Useful when the search is expected
    to make exploratory detours that shouldn't reset the baseline.
    """

    latest = auto()
    best = auto()
