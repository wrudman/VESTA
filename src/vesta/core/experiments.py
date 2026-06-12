"""Full end-to-end VLM-guided PyMC fitting pipeline.

Architecture
============

``run()`` is structured as a three-part cycle (Functional Core /
Imperative Shell + per-field reducer):

1. **SHELL** — ``_execute_step`` runs the six pipeline phases
   (diagnostic → proposal → codegen → fit) plus a summary generation
   step for one iteration and emits a frozen ``StepObservation``
   describing what happened.
2. **CORE** — ``_reduce_step`` is a pure function ``(state, obs) → new_state``.
   No I/O, no mutation. This is the *only* place state evolves — adding
   a new trajectory field is a one-line change here.
3. **Outer loop** threads the state forward, persists the step records
   to ``run_log.parquet`` after every iteration, updates the progress
   bar, and terminates when ``state.status != StepStatus.running``.

Step 0 is not peeled out of the loop. It runs as the first iteration of
the unified loop with ``state.step_num == 0``. The only per-phase
difference between step 0 and step N lives inside two phase helpers
(proposal + summary) and is localised to a one-line ``if step_num == 0``
branch in each — never polluting the main loop body.

``summary_trajectory`` is a single tuple that unifies step 0's initial
summary (position 0) with the per-step feedback summaries
(positions 1..N). Every step appends its own summary to this trajectory,
and the feedback-prompt builder reads from this single source of truth.

Usage from notebook::

    from vesta import ExperimentConfig, run_all
    from vesta.core.experiment_config import ModelConfig, ToolkitConfig, OutputConfig

    config = ExperimentConfig(
        model=ModelConfig(litellm_model="anthropic/claude-sonnet-4.6"),
        toolkit=ToolkitConfig(mode="expert"),
        output=OutputConfig(expt="my_experiment"),
        data_pkl="data_single.pkl",
        max_steps=3,
    )
    results = run_all(config=config)

Usage from CLI::

    python experiments.py \\
        --model.id azure/gpt-5-mini \\
        --data-pkl data_single.pkl \\
        --max-steps 3 \\
        --toolkit.mode expert \\
        --output.expt baseline
"""

# _thread_caps and _pytensor_compiledir MUST be the very first imports of this
# module.  They mutate os.environ (OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, ...,
# PYTENSOR_FLAGS) before numpy / scipy / pymc / pytensor load.  Those libraries
# latch their thread pools and compile directory at dlopen / import time, so
# moving either import below `import numpy` silently disables them.  See each
# module's docstring for the full rationale and the PYMC_PARALLEL__COMPUTE_THREADS
# and PYMC_PYTENSOR_COMPILEDIR_ROOT configuration variables they honour.
import vesta.runtime._thread_caps  # noqa: F401,I001
import vesta.runtime._pytensor_compiledir  # noqa: F401,I001

import datetime  # noqa: I001
import gc
import json
import logging
import os
import pickle
import re
import textwrap
import time
import sys
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH: str = os.path.join(_THIS_DIR, "..", "..", "..", ".env")
load_dotenv(_ENV_PATH)

if sys.platform == "darwin":
    os.environ.setdefault("no_proxy", "*")

# _SLOWBURN_SRC: str = os.path.abspath(os.path.join(_THIS_DIR, "..", "slowburn", "src"))
# if _SLOWBURN_SRC not in sys.path:
#     sys.path.insert(0, _SLOWBURN_SRC)

import numpy as np
import pandas as pd
from concurry import gather
from concurry.utils.progress import ProgressBar
from morphic import validate
from morphic.string import format_exception_msg

import vesta.domains.distribution_fitting  # noqa: F401
import vesta.domains.time_series  # noqa: F401
from vesta.core.api_repair_diagnostics import build_api_discovery_report
from vesta.domains import (
    CodeGenerationAttempt,
    CodeGenerationFailure,
    DiagnosticArtifact,
    DiagnosticToolResult,
    DomainPlotting,
    DomainPrompts,
    DomainToolkit,
    FitState,
)
from vesta.core.dynamic_toolkit import (
    DIAGNOSTIC_PHASE_PROMPT,
    GENERATE_NEW_TOOL_NUDGE,
    DynamicToolSpec,
    GeneratedToolExecutionResult,
    build_tools_list,
    handle_generate_new_tool,
    load_dynamic_tools,
    save_dynamic_tools,
)
from vesta.core.experiment_config import ExperimentConfig
from vesta.core.experiment_enums import (
    Domain,
    ObservationKind,
    PhaseName,
    RunStatus,
    StepStatus,
    ToolkitMode,
)
from vesta.core.experiment_step_state import (
    CarriedModel,
    RunDeps,
    StepObservation,
    StepState,
    carried_model,
    validated_copy,
)
from vesta.core.experiment_workers import DatasetRunnerProcess
from vesta.core.logging_utils import BLOCK_LIGHT_SEP, format_log_block
from vesta.core.processing_utils import clean_params, fit_single_model, select_best_fit_result
from vesta.core.sandbox_namespaces import get_pymc_namespace
from vesta.vlm_backends import ToolCallResponse, ToolCallResult, VLMBackend
from vesta.vlm_backends.parsing import parse_json_from_text

logger: logging.Logger = logging.getLogger("experiments")


# ══════════════════════════════════════════════════════════════════════════
# Prompt construction helpers
# ══════════════════════════════════════════════════════════════════════════


def _build_model_spec_feedback_prompt(
    model_spec_feedback_prompt_template: str,
    *,
    plot_type_description: str,
    current_model: str,
    model_structure: Union[str, List[str]],
    tested_model_structures: Tuple[Union[str, List[str]], ...],
    selected_tool_history: Tuple[str, ...],
    summary_trajectory: Tuple[str, ...],
    diagnostic_results: str,
    max_steps: int,
) -> str:
    """Format the Phase-2 feedback prompt from the unified summary trajectory.

    ``summary_trajectory`` is the full list of summaries produced so far.
    ``trajectory[0]`` is step 0's initial summary; ``trajectory[i]`` is
    step ``i``'s feedback summary. The template has a single
    ``{initial_summary}`` slot and ``{feedback_summary_step_i}`` slots
    for each feedback step; this helper fills them directly from the
    trajectory, so callers never build or pass those values separately.

    Raises:
        ValueError: If ``summary_trajectory`` is empty. Feedback prompts
            are built only at step ``>= 1``, by which point step 0's
            summary must already be in the trajectory.
    """
    if len(summary_trajectory) == 0:
        raise ValueError(
            "summary_trajectory must contain at least step 0's initial summary "
            "before a feedback prompt is built."
        )

    num_feedback_slots: int = max(max_steps - 1, 1)
    initial_summary: str = summary_trajectory[0]
    feedback_kwargs: Dict[str, str] = {}
    for i in range(1, num_feedback_slots + 1):
        if i < len(summary_trajectory):
            feedback_kwargs[f"feedback_summary_step_{i}"] = summary_trajectory[i]
        else:
            feedback_kwargs[f"feedback_summary_step_{i}"] = "Not triggered yet!"

    indented_current_model: str = textwrap.indent(current_model, "    ")
    return model_spec_feedback_prompt_template.format(
        plot_type_description=plot_type_description,
        current_model=indented_current_model,
        model_structure=model_structure,
        tested_model_structures=list(tested_model_structures),
        selected_tool_history=list(selected_tool_history),
        initial_summary=initial_summary,
        diagnostic_results=diagnostic_results,
        **feedback_kwargs,
    )


def _inject_initial_diagnostic_results(
    *,
    proposal_prompt: str,
    diagnostic_results: str,
) -> str:
    """Insert step-level diagnostic results into the initial proposal prompt.

    Initial (step-0) proposal prompts should mirror feedback prompts by placing
    ``DIAGNOSTIC RESULTS`` above ``YOUR TASK`` and above the output JSON
    structure section. This helper injects the rendered diagnostic block at the
    correct location and fails loudly if no insertion marker exists.
    """
    diagnostic_block: str = f"DIAGNOSTIC RESULTS (from this step's analysis):\n{diagnostic_results}\n\n"
    your_task_marker: str = "YOUR TASK:\n"
    if your_task_marker in proposal_prompt:
        return proposal_prompt.replace(your_task_marker, diagnostic_block + your_task_marker, 1)

    return_marker: str = "Return ONLY a valid JSON object"
    if return_marker in proposal_prompt:
        return proposal_prompt.replace(return_marker, diagnostic_block + return_marker, 1)

    raise ValueError(
        "Initial proposal prompt template is missing both 'YOUR TASK:' and "
        "'Return ONLY a valid JSON object' insertion markers."
    )


def _is_complete_feedback_description(*, description: str) -> bool:
    """Return True if feedback ``description`` starts with COMPLETE.

    Phase 2 feedback responses are schema-validated JSON objects with a
    free-form ``description`` string. Some models ignore the "EXACTLY to
    COMPLETE" instruction and emit variants like:

    - ``COMPLETE.``
    - ``complete``
    - ``COMPLETE The fit is satisfactory ...``

    These should all terminate refinement. Matching is prefix-based and
    case-insensitive; non-prefix mentions (e.g. "almost complete") do not
    terminate.
    """
    normalized_description: str = description.strip()
    if len(normalized_description) == 0:
        return False
    return re.match(r"^complete\b", normalized_description, flags=re.IGNORECASE) is not None


def _collect_image_artifact_paths(
    *,
    tool_results: Tuple[DiagnosticToolResult, ...],
) -> List[str]:
    """Collect all image attachment paths from diagnostic artifacts."""
    image_paths: List[str] = []
    for tool_result in tool_results:
        for artifact in tool_result.artifacts:
            if artifact.artifact_type == "image":
                if artifact.attachment_path is None:
                    raise ValueError(
                        f"Image artifact for tool {tool_result.tool_name!r} has no attachment_path."
                    )
                image_paths.append(artifact.attachment_path)
    return image_paths


def _derive_tool_summary_fields(
    *,
    tool_results: Tuple[DiagnosticToolResult, ...],
) -> Tuple[str, str]:
    """Derive legacy summary fields from artifacts for summary prompts."""
    if len(tool_results) == 0:
        return "none", "N/A"

    first_tool_result: DiagnosticToolResult = tool_results[0]
    first_non_image_artifact: Optional[DiagnosticArtifact] = None
    for artifact in first_tool_result.artifacts:
        if artifact.artifact_type != "image":
            first_non_image_artifact = artifact
            break

    if first_non_image_artifact is None:
        return "visualization", "N/A"

    if first_non_image_artifact.artifact_type == "error":
        if (
            first_non_image_artifact.inline_content is not None
            and len(first_non_image_artifact.inline_content) > 0
        ):
            return "none", first_non_image_artifact.inline_content
        return "none", first_non_image_artifact.description

    if (
        first_non_image_artifact.inline_content is not None
        and len(first_non_image_artifact.inline_content) > 0
    ):
        return "numeric", first_non_image_artifact.inline_content
    return "numeric", first_non_image_artifact.description


def _serialize_tool_results(
    *,
    tool_results: Tuple[DiagnosticToolResult, ...],
) -> List[Dict[str, Any]]:
    """Serialize typed tool results for step logging/parquet."""
    return [tool_result.model_dump() for tool_result in tool_results]


def _serialize_generated_tools(
    *,
    generated_tools: Tuple[GeneratedToolExecutionResult, ...],
) -> List[Dict[str, Any]]:
    """Serialize generated-tool execution results for step logging/parquet."""
    return [generated_tool.model_dump() for generated_tool in generated_tools]


def _release_fit_model_references(*, fit_results: Dict[str, Dict[str, Any]]) -> None:
    """Drop heavy PyMC model object references once no longer needed."""
    for result in fit_results.values():
        if "model" in result:
            result["model"] = None


def _format_diagnostic_results(
    tool_results: Tuple[DiagnosticToolResult, ...],
    *,
    base_image_description: Optional[str] = None,
) -> str:
    """Format Phase 1 diagnostic results as a numbered list for the Phase 2 prompt.

    When ``base_image_description`` is provided, item ``1) Context image``
    documents the base visual context the Phase-2 VLM call sees (initial
    plot at step 0, or current fit overlay at step N≥1). Each invoked
    diagnostic tool becomes a subsequent numbered item describing its own
    artifacts. This cleanly separates the always-attached context image from
    tool-produced artifacts, so numeric-only tools (e.g. ``calculate_moments``)
    do not have the fit overlay spuriously attributed to them as an output.

    When no base image description is given and no tools were invoked, the
    function returns the sentinel sentence previously expected by callers.
    """
    body_indent: str = "   "
    inline_indent: str = "      "

    items: List[str] = []
    item_index: int = 1

    if base_image_description is not None:
        items.append(
            f"{item_index}) Context image\n"
            + textwrap.indent(
                f"- [image] {base_image_description} (see attached)",
                body_indent,
            )
        )
        item_index += 1

    for tool_result in tool_results:
        artifact_lines: List[str] = []
        for artifact in tool_result.artifacts:
            if artifact.artifact_type == "image":
                artifact_lines.append(f"- [image] {artifact.description} (see attached diagnostic image)")
                continue
            if artifact.artifact_type not in ("json", "table", "text", "error"):
                raise ValueError(
                    f"Unknown artifact_type={artifact.artifact_type!r} for tool {tool_result.tool_name!r}."
                )
            artifact_lines.append(f"- [{artifact.artifact_type}] {artifact.description}")
            if artifact.inline_content is not None:
                artifact_lines.append(textwrap.indent(artifact.inline_content, inline_indent))

        body: str = f"What this tool does: {tool_result.tool_description}\nOutputs:\n" + "\n".join(
            artifact_lines
        )
        items.append(f"{item_index}) Tool '{tool_result.tool_name}'\n" + textwrap.indent(body, body_indent))
        item_index += 1

    if len(items) == 0:
        return "No diagnostic tools were used this step."

    return "\n\n".join(items)


def _synthesize_tool_error_result(
    *,
    tool_name: str,
    tool_description: Optional[str],
    exc: Exception,
    step_num: int,
) -> DiagnosticToolResult:
    """Build an error ``DiagnosticToolResult`` when a tool crashes mid-round.

    The VLM needs to see *something* for every tool it picked, otherwise
    Phase 2's DIAGNOSTIC RESULTS block silently elides the attempt and the
    next step's feedback summary cannot mention it either. Rendering the
    failure as an ``artifact_type="error"`` artifact routes it through the
    existing ``_format_diagnostic_results`` + ``_derive_tool_summary_fields``
    pipeline without any special-casing.
    """
    error_msg: str = f"{type(exc).__name__}: {format_exception_msg(exc)}"
    description: str = (
        tool_description
        if tool_description is not None and len(tool_description) > 0
        else f"Tool {tool_name!r} (failed to execute at step {step_num})"
    )
    return DiagnosticToolResult(
        tool_name=tool_name,
        tool_description=description,
        artifacts=[
            DiagnosticArtifact(
                artifact_type="error",
                description=(
                    f"Tool {tool_name!r} raised {type(exc).__name__} during execution "
                    f"at step {step_num}. The tool produced NO diagnostic output; "
                    f"do not rely on any past-tense description of its result."
                ),
                inline_content=error_msg,
                attachment_path=None,
                truncated=False,
            )
        ],
    )


def _run_diagnostic_rounds(
    *,
    backend: VLMBackend,
    diagnostic_prompt: str,
    images: List[str],
    tools_for_vlm: List[Dict[str, Any]],
    max_tool_calls: int,
    force_tool_call: bool,
    toolkit_reg: DomainToolkit,
    data: Any,
    fit_state: Optional[FitState],
    best_idx: Optional[Union[str, int]],
    out_dir: str,
    step_num: int,
    plot_type_descriptions: Dict[str, str],
    tool_gen_backend: VLMBackend,
    tool_gen_model: str,
    domain: Domain,
    verbosity: int,
    dynamic_tools: Dict[str, DynamicToolSpec],
    generated_tool_results: List[GeneratedToolExecutionResult],
    code_generation_attempts: List[CodeGenerationAttempt],
    max_tool_generation_attempts: int,
) -> List[DiagnosticToolResult]:
    """Iterate up to ``max_tool_calls`` *rounds* of diagnostic tool calls.

    This is the inner round loop of Phase 1 (the diagnostic phase).
    ``_phase_diagnostic`` is the Phase-1 orchestrator that builds the
    diagnostic prompt and delegates the actual round iteration here.

    The VLM is offered diagnostic tools via native function-calling.
    Each round makes one ``call_for_tool`` request. At most one tool
    is executed per round; if a model returns multiple ``tool_calls``
    in a single response, only the first is executed and the rest are
    ignored.

    This makes ``max_tool_calls`` a strict cap on executed tools per
    step. If ``force_tool_call`` is True, each round uses
    ``tool_choice="required"``; a required round returning no tool
    calls raises ``ValueError``. In auto mode, if the VLM declines to
    call a tool, the round loop ends early.

    Returns:
        Structured diagnostic outputs for all invoked tools. Any new
        ``GeneratedToolExecutionResult`` instances are appended in-place
        to ``generated_tool_results`` (caller-provided accumulator, per
        the explicit accumulator pattern).
    """
    available_tools: List[Dict[str, Any]] = list(tools_for_vlm)
    tool_results: List[DiagnosticToolResult] = []

    tool_names: List[str] = [t["function"]["name"] for t in tools_for_vlm]
    logger.debug(
        format_log_block(
            title=f"[step {step_num}] PHASE 1 DIAGNOSTIC PROMPT",
            body=(f"{diagnostic_prompt}\n{BLOCK_LIGHT_SEP}\nAvailable tools: {tool_names}"),
        )
    )

    for call_idx in range(max_tool_calls):
        tool_choice: str = "required" if force_tool_call else "auto"

        if len(available_tools) == 0:
            logger.debug(f"[step {step_num}] Phase 1: no tools left to offer, ending diagnostic phase")
            break

        available_tool_names: List[str] = [t["function"]["name"] for t in available_tools]
        logger.debug(
            f"[step {step_num}] Phase 1 diagnostic round {call_idx + 1}/{max_tool_calls}: "
            f"tool_choice={tool_choice!r}, offering {len(available_tools)} tool(s): "
            f"{available_tool_names}"
        )

        response: ToolCallResponse = backend.call_for_tool(
            prompt=diagnostic_prompt,
            images=images,
            tools=available_tools,
            tool_choice=tool_choice,
            verbosity=verbosity,
        )

        if response.has_tool_calls is False:
            declined_content: str = (
                response.content if response.content is not None else "(no assistant content)"
            )
            if tool_choice == "required" and len(available_tools) > 0:
                # VLM declined despite tool_choice='required'.  Instead of
                # aborting, auto-select a sensible default diagnostic tool
                # so the pipeline continues with the same flow.  The
                # preferred default depends on the domain:
                #   - time_series:            fit_vs_actuals
                #   - distribution_fitting:   histogram
                # Fall back to the first available tool if the preferred
                # one isn't in the offered set.
                available_tool_names_local: List[str] = [t["function"]["name"] for t in available_tools]
                _DOMAIN_DEFAULT_TOOLS: Dict[Domain, str] = {
                    Domain.time_series: "fit_vs_actuals",
                    Domain.distribution_fitting: "qq_plot",
                }
                preferred: str = _DOMAIN_DEFAULT_TOOLS.get(domain, available_tool_names_local[0])
                default_name: str = (
                    preferred if preferred in available_tool_names_local else available_tool_names_local[0]
                )
                logger.warning(
                    f"[step {step_num}] Phase 1 diagnostic round {call_idx + 1}/{max_tool_calls}: "
                    f"tool_choice='required' but VLM declined — auto-selecting "
                    f"default tool '{default_name}'. "
                    f"Assistant content:\n{textwrap.indent(declined_content, '   ')}"
                )
                response = ToolCallResponse(
                    content=declined_content,
                    tool_calls=[
                        ToolCallResult(
                            id=f"auto_{step_num}_{call_idx}",
                            name=default_name,
                            arguments={},
                        )
                    ],
                )
                # Fall through to the normal tool-execution path below.
            else:
                logger.debug(
                    f"[step {step_num}] Phase 1 diagnostic round {call_idx + 1}/{max_tool_calls}: "
                    f"VLM declined to call any tool. Assistant content:\n"
                    f"{textwrap.indent(declined_content, '   ')}"
                )
                break

        requested_tool_names: List[str] = [tool_call.name for tool_call in response.tool_calls]
        logger.debug(
            f"[step {step_num}] Phase 1 diagnostic round {call_idx + 1}/{max_tool_calls}: "
            f"VLM requested {len(response.tool_calls)} tool call(s): "
            f"{requested_tool_names}"
        )

        if len(response.tool_calls) > 1:
            ignored_tool_names: List[str] = [tool_call.name for tool_call in response.tool_calls[1:]]
            logger.warning(
                f"[step {step_num}] Phase 1 diagnostic round {call_idx + 1}/{max_tool_calls}: "
                f"received {len(response.tool_calls)} tool calls; executing only first call "
                f"{response.tool_calls[0].name!r} and ignoring {ignored_tool_names}"
            )

        selected_tool_call: ToolCallResult = response.tool_calls[0]
        round_failed: bool = False
        selected_tool_result: Optional[DiagnosticToolResult] = None
        tool_fit_path: str = os.path.join(
            out_dir,
            f"step_{step_num:03d}-phase_1-diagnostic-{call_idx}-0-{selected_tool_call.name}.png",
        )

        logger.debug(
            f"[step {step_num}] Phase 1 — Tool Call {call_idx + 1}.1: "
            f"{selected_tool_call.name}  (args: {selected_tool_call.arguments})"
        )

        if selected_tool_call.name == "generate_new_tool":
            map_estimate_for_generation: Dict[str, Any] = {}
            model_structure_for_generation: List[str] = []
            if fit_state is not None:
                map_estimate_for_generation = fit_state.map_estimate
                model_structure_for_generation = list(fit_state.family_name)
            try:
                generated_result: GeneratedToolExecutionResult = handle_generate_new_tool(
                    tool_description=selected_tool_call.arguments["tool_description"],
                    tool_gen_backend=tool_gen_backend,
                    data=data,
                    map_estimate=map_estimate_for_generation,
                    family_name=model_structure_for_generation,
                    fit_path=tool_fit_path,
                    verbosity=verbosity,
                    domain=domain,
                    tool_gen_model=tool_gen_model,
                    max_tool_generation_attempts=max_tool_generation_attempts,
                )
            except CodeGenerationFailure as exc:
                if len(exc.attempts) > 0:
                    code_generation_attempts.extend(exc.attempts)
                logger.error(
                    f"[step {step_num}] Phase 1 — Tool {selected_tool_call.name} failed: "
                    f"{format_exception_msg(exc)}"
                )
                error_result: DiagnosticToolResult = _synthesize_tool_error_result(
                    tool_name=selected_tool_call.name,
                    tool_description=selected_tool_call.arguments["tool_description"],
                    exc=exc,
                    step_num=step_num,
                )
                selected_tool_result = error_result
                tool_results.append(error_result)
                round_failed = True
            except Exception as exc:
                logger.error(
                    f"[step {step_num}] Phase 1 — Tool {selected_tool_call.name} failed: "
                    f"{format_exception_msg(exc)}"
                )
                error_result = _synthesize_tool_error_result(
                    tool_name=selected_tool_call.name,
                    tool_description=selected_tool_call.arguments["tool_description"],
                    exc=exc,
                    step_num=step_num,
                )
                selected_tool_result = error_result
                tool_results.append(error_result)
                round_failed = True
            else:
                # Register the freshly-built spec into the run-level dict
                # so the VLM can reuse the tool on subsequent steps.
                dynamic_tools[generated_result.spec.name] = generated_result.spec
                tool_result: DiagnosticToolResult = DiagnosticToolResult(
                    tool_name=generated_result.registered_tool_name,
                    tool_description=generated_result.tool_description,
                    artifacts=generated_result.artifacts,
                )
                selected_tool_result = tool_result
                tool_results.append(tool_result)
                generated_tool_results.append(generated_result)
                if len(generated_result.attempts) > 0:
                    code_generation_attempts.extend(generated_result.attempts)
                logger.debug(
                    f"[step {step_num}] Phase 1 — Registered dynamic tool "
                    f"{generated_result.registered_tool_name} "
                    f"(run dynamic_tools now has {len(dynamic_tools)} tool(s)); "
                    f"executed with {len(generated_result.artifacts)} artifact(s)."
                )
        elif selected_tool_call.name in dynamic_tools:
            # Previously-registered dynamic tool: dispatch via the run-level
            # dict directly, bypassing the domain's expert-tool table.
            try:
                spec: DynamicToolSpec = dynamic_tools[selected_tool_call.name]
                tool_result: DiagnosticToolResult = spec.execute(
                    data=data,
                    fit_state=fit_state,
                    fit_path=tool_fit_path,
                )
            except Exception as exc:
                logger.error(
                    f"[step {step_num}] Phase 1 — Tool {selected_tool_call.name} failed: "
                    f"{format_exception_msg(exc)}"
                )
                spec_description: Optional[str] = (
                    dynamic_tools[selected_tool_call.name].description
                    if selected_tool_call.name in dynamic_tools
                    else None
                )
                error_result: DiagnosticToolResult = _synthesize_tool_error_result(
                    tool_name=selected_tool_call.name,
                    tool_description=spec_description,
                    exc=exc,
                    step_num=step_num,
                )
                selected_tool_result = error_result
                tool_results.append(error_result)
                round_failed = True
            else:
                selected_tool_result = tool_result
                tool_results.append(tool_result)
                logger.debug(
                    f"[step {step_num}] Phase 1 — Dynamic tool {tool_result.tool_name} "
                    f"returned {len(tool_result.artifacts)} artifact(s)."
                )
        else:
            try:
                tool_result: DiagnosticToolResult = toolkit_reg.execute_tool(
                    selected_tool=selected_tool_call.name,
                    selected_tool_args=selected_tool_call.arguments,
                    data=data,
                    fit_state=fit_state,
                    best_idx=best_idx,
                    fit_path=tool_fit_path,
                    plot_type_descriptions=plot_type_descriptions,
                )
                selected_tool_result = tool_result
                tool_results.append(tool_result)
                logger.debug(
                    f"[step {step_num}] Phase 1 — Tool {tool_result.tool_name} "
                    f"returned {len(tool_result.artifacts)} artifact(s)."
                )
            except Exception as exc:
                logger.error(
                    f"[step {step_num}] Phase 1 — Tool {selected_tool_call.name} failed: "
                    f"{format_exception_msg(exc)}"
                )
                error_result: DiagnosticToolResult = _synthesize_tool_error_result(
                    tool_name=selected_tool_call.name,
                    tool_description=plot_type_descriptions.get(selected_tool_call.name),
                    exc=exc,
                    step_num=step_num,
                )
                selected_tool_result = error_result
                tool_results.append(error_result)
                round_failed = True

        available_tools = [
            tool_dict
            for tool_dict in available_tools
            if tool_dict["function"]["name"] != selected_tool_call.name
        ]

        if selected_tool_result is not None:
            latest_image_paths: List[str] = _collect_image_artifact_paths(
                tool_results=(selected_tool_result,),
            )
            if len(latest_image_paths) > 0:
                images = images + latest_image_paths

        if round_failed:
            break

    return tool_results


# ══════════════════════════════════════════════════════════════════════════
# Phase helpers — one per pipeline phase. Each reads ``state`` + ``deps``
# and performs its side effects, returning the raw outputs needed to build
# the final ``StepObservation``. Phases never write to state directly.
# ══════════════════════════════════════════════════════════════════════════


@validate
def _carried_model_summary(*, carried: CarriedModel) -> str:
    """One-line description of the model being carried forward, or the pre-0 placeholder.

    Always pairs the structure with ``carried.model_code``'s length from
    the same payload so the label and the code snippet in the feedback
    prompt cannot drift out of sync.
    """
    if carried.model_code is None:
        return "(no model yet — initial plot)"
    return f"{carried.model_structure} (code: {len(carried.model_code)} chars)"


@validate
def _carried_aic_label(*, carried: CarriedModel) -> str:
    """AIC label for the carried model, or ``N/A`` before the first fit.

    Reads from the SAME payload as the model summary so VLM never sees
    an AIC from a different step than the family/code it's evaluating.
    """
    if carried.metrics is None or "aic" not in carried.metrics:
        return "N/A"
    return f"{round(float(carried.metrics['aic']), 1)}"


def _phase_diagnostic(
    *,
    state: StepState,
    deps: RunDeps,
    step_num: int,
) -> Tuple[
    Tuple[DiagnosticToolResult, ...],
    Tuple[GeneratedToolExecutionResult, ...],
    Tuple[CodeGenerationAttempt, ...],
    Tuple[Dict[str, Any], ...],
    Optional[str],
]:
    """Run the Phase-1 diagnostic phase for ``step_num``.

    Returns:
        ``(tool_results, generated_tool_results, code_generation_attempts,
        tools_offered, diagnostic_prompt)``. When ``toolkit_mode is none``
        or the toolkit produced no schemas, these tuple fields are empty
        and ``diagnostic_prompt`` is ``None``.
    """
    logger.info(f"[step {step_num}] Phase 1: diagnostic tool selection")

    tools_for_vlm: List[Dict[str, Any]] = build_tools_list(
        toolkit_mode=deps.toolkit_mode,
        expert_tools=deps.toolkit_reg.get_expert_tools(),
        dynamic_tools=deps.dynamic_tools,
    )

    if deps.toolkit_mode is ToolkitMode.none or len(tools_for_vlm) == 0:
        logger.info(f"[step {step_num}] Phase 1: skipped (no toolkit tools enabled)")
        return (), (), (), (), None

    carried: CarriedModel = carried_model(state=state, strategy=deps.carry_forward)

    # Accumulated dynamic tools require a fitted model (fit_state).
    # At step 0 (and on any error-step where no model was fitted),
    # fit_state is None — skip the phase rather than offering tools
    # that cannot execute.
    if deps.toolkit_mode is ToolkitMode.accumulated_only and carried.fit_state is None:
        logger.info(
            f"[step {step_num}] Phase 1: skipped "
            f"(accumulated_only tools require a fitted model; fit_state is None)"
        )
        return (), (), (), (), None

    tested_structures_label: Any = (
        "(none yet)" if len(state.tested_model_structures) == 0 else list(state.tested_model_structures)
    )
    diagnostic_prompt: str = DIAGNOSTIC_PHASE_PROMPT.format(
        carried_label=carried.label,
        carried_label_capitalized=carried.label.capitalize(),
        current_model_summary=_carried_model_summary(carried=carried),
        current_aic=_carried_aic_label(carried=carried),
        step_num=step_num,
        max_steps=deps.max_steps,
        tested_model_structures=tested_structures_label,
    )
    # Encourage the VLM to call ``generate_new_tool`` when existing tools
    # don't fit this dataset — but only when the toolkit mode actually
    # offers that tool.  Under ``none`` no tools are offered at all, and
    # under ``expert`` the toolkit is the fixed domain set without
    # ``generate_new_tool``; in both cases the nudge would be either
    # meaningless or misleading.
    if deps.toolkit_mode is ToolkitMode.generate_only or deps.toolkit_mode is ToolkitMode.dynamic:
        diagnostic_prompt = diagnostic_prompt + GENERATE_NEW_TOOL_NUDGE

    generated_accumulator: List[GeneratedToolExecutionResult] = []
    code_generation_attempts: List[CodeGenerationAttempt] = []
    tool_results: List[DiagnosticToolResult] = _run_diagnostic_rounds(
        backend=deps.backend,
        diagnostic_prompt=diagnostic_prompt,
        images=[carried.fit_path],
        tools_for_vlm=tools_for_vlm,
        max_tool_calls=deps.max_tool_calls_per_step,
        force_tool_call=deps.force_tool_call,
        toolkit_reg=deps.toolkit_reg,
        data=deps.data,
        fit_state=carried.fit_state,
        best_idx=None,
        out_dir=deps.out_dir,
        step_num=step_num,
        plot_type_descriptions=deps.plot_type_descriptions,
        tool_gen_backend=deps.tool_gen_backend,
        tool_gen_model=deps.tool_gen_model,
        domain=deps.domain,
        verbosity=deps.verbosity,
        dynamic_tools=deps.dynamic_tools,
        generated_tool_results=generated_accumulator,
        code_generation_attempts=code_generation_attempts,
        max_tool_generation_attempts=deps.max_tool_generation_attempts,
    )
    logger.info(f"[step {step_num}] Phase 1 complete: executed {len(tool_results)} diagnostic tool(s).")
    return (
        tuple(tool_results),
        tuple(generated_accumulator),
        tuple(code_generation_attempts),
        tuple(tools_for_vlm),
        diagnostic_prompt,
    )


def _phase_proposal(
    *,
    state: StepState,
    deps: RunDeps,
    step_num: int,
    tool_results: Tuple[DiagnosticToolResult, ...],
) -> Tuple[str, Tuple[str, ...], Dict[str, Any], float]:
    """Run the Phase-2 proposal phase for ``step_num``.

    At step 0 the prompt is built from ``model_spec_proposal_prompt_template``
    with the diagnostic results injected. At step > 0 the prompt is built
    from ``model_spec_feedback_prompt_template`` with the running
    ``summary_trajectory``.

    Returns:
        ``(proposal_prompt, images_sent_to_vlm, response_dict, api_time_s)``.
    """
    diagnostic_images: List[str] = _collect_image_artifact_paths(tool_results=tool_results)
    carried: CarriedModel = carried_model(state=state, strategy=deps.carry_forward)
    if step_num == 0:
        base_image_description: str = "Initial plot of the observed data (no fitted model yet)"
    else:
        if carried.metrics is None or "aic" not in carried.metrics:
            raise RuntimeError(
                f"[step {step_num}] carried model has no AIC metric; "
                f"expected a fitted model to be available at step > 0."
            )
        base_image_description = (
            f"Current {carried.label}-fit overlay "
            f"(family={carried.model_structure!r}, AIC={carried.metrics['aic']:.1f})"
        )
    diagnostic_text: str = _format_diagnostic_results(
        tool_results,
        base_image_description=base_image_description,
    )

    if step_num == 0:
        proposal_prompt: str = _inject_initial_diagnostic_results(
            proposal_prompt=deps.model_spec_proposal_prompt_template,
            diagnostic_results=diagnostic_text,
        )
    else:
        if carried.model_code is None or carried.model_structure is None:
            raise RuntimeError(
                f"[step {step_num}] Cannot build feedback prompt: carried model "
                f"has no code/structure. This should only happen on the very first "
                f"iteration (step_num == 0)."
            )
        proposal_prompt = _build_model_spec_feedback_prompt(
            deps.model_spec_feedback_prompt_template,
            plot_type_description=_latest_plot_description(tool_results=tool_results, deps=deps),
            current_model=carried.model_code,
            model_structure=carried.model_structure,
            tested_model_structures=state.tested_model_structures,
            selected_tool_history=state.selected_tool_history,
            summary_trajectory=state.summary_trajectory,
            diagnostic_results=diagnostic_text,
            max_steps=deps.max_steps,
        )

    images_sent: Tuple[str, ...] = (carried.fit_path,) + tuple(diagnostic_images)
    logger.info(f"[step {step_num}] Phase 2: proposal generation")
    logger.debug(f"[step {step_num}] Phase 2 VLM images: {list(images_sent)}")
    logger.debug(
        format_log_block(
            title=f"[step {step_num}] PROPOSAL PROMPT ({len(proposal_prompt)} chars)",
            body=proposal_prompt,
        )
    )

    call_start: float = time.time()
    vlm_response: Any = deps.backend.call(
        prompt=proposal_prompt,
        images=list(images_sent),
        tools=None,
        response_type=deps.response_type,
        verbosity=deps.verbosity,
    )
    api_time_s: float = time.time() - call_start
    response_dict: Dict[str, Any] = vlm_response.model_dump()

    description_text: str = response_dict["description"]
    is_complete: bool = step_num > 0 and _is_complete_feedback_description(description=description_text)
    if is_complete:
        logger.debug(
            format_log_block(
                title=f"[step {step_num}] PROPOSAL RESPONSE: COMPLETE",
                body=(
                    f"  description: {description_text}\n"
                    "  Placeholder proposals returned alongside COMPLETE are "
                    "discarded (pipeline terminating)."
                ),
            )
        )
        logger.info(f"[step {step_num}] Phase 2 complete: model marked COMPLETE.")
    else:
        logger.debug(
            format_log_block(
                title=f"[step {step_num}] PROPOSAL RESPONSE",
                body=str(vlm_response),
            )
        )
        logger.info(
            f"[step {step_num}] Phase 2 complete: received {len(response_dict['proposals'])} proposal(s)."
        )
    return proposal_prompt, images_sent, response_dict, api_time_s


def _latest_plot_description(
    *,
    tool_results: Tuple[DiagnosticToolResult, ...],
    deps: RunDeps,
) -> str:
    """Pick the plot description to weave into the feedback prompt.

    If the diagnostic phase ran at least one tool, its description becomes
    the context for the VLM (so it can reason about the tool's output).
    Otherwise we fall back to the domain default.
    """
    if len(tool_results) > 0:
        return tool_results[-1].tool_description
    return deps.plot_type_descriptions[deps.plotting_reg.get_default_plot_description()]


def _parse_code_generation_response(*, raw_response: str, target: str) -> str:
    """Extract the ``code`` string from a raw code-generation response."""
    parsed_response: Dict[str, Any] = parse_json_from_text(raw_response)
    if "code" not in parsed_response:
        raise ValueError(
            f"Code-generation response for {target} did not contain required key 'code'. "
            f"Available keys: {list(parsed_response.keys())}. Raw response:\n{raw_response}"
        )
    if not isinstance(parsed_response["code"], str):
        raise TypeError(
            f"Code-generation response for {target} contained non-string code value: "
            f"{parsed_response['code']!r}. Raw response:\n{raw_response}"
        )
    return parsed_response["code"]


def _failed_codegen_fit_result(
    *,
    ans: Dict[str, Any],
    model_idx: str,
    task: str,
    error_message: str,
) -> Dict[str, Any]:
    """Build a failed fit-result entry for code-generation failures before fitting."""
    failed_metrics: Dict[str, Any] = {"aic": np.inf, "bic": np.inf, "n_params": None}
    if task == "fitting":
        return {
            "model": None,
            "map_estimate": None,
            "metrics": failed_metrics,
            "distribution_family": ans["distribution_family"][model_idx],
            "is_mixture": ans["is_mixture"][model_idx],
            "error": error_message,
        }
    elif task == "time_series":
        return {
            "model": None,
            "map_estimate": None,
            "metrics": failed_metrics,
            "kernels": ", ".join(ans["kernels"][model_idx]),
            "trend": None,
            "error": error_message,
        }
    else:
        raise ValueError(f"Unexpected task={task!r}. Must be 'fitting' or 'time_series'.")


def _generate_and_fit_model_code(
    *,
    deps: RunDeps,
    ans: Dict[str, Any],
    model_idx: str,
    base_prompt: str,
    step_num: int,
) -> Tuple[
    Optional[str],
    Dict[str, Any],
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[CodeGenerationAttempt, ...],
]:
    """Run code-generation repair attempts for one proposal and fit each candidate."""
    prompts: List[str] = []
    raw_responses: List[str] = []
    attempts: List[CodeGenerationAttempt] = []
    max_attempts: int = deps.max_code_generation_attempts
    fit_result: Dict[str, Any] = _failed_codegen_fit_result(
        ans=ans,
        model_idx=model_idx,
        task=deps.prompts_reg.task_string,
        error_message="No code-generation attempt was executed.",
    )
    last_error: Optional[str] = None
    prompt: str = base_prompt

    for attempt_number in range(1, max_attempts + 1):
        attempt_kind: str = "initial" if attempt_number == 1 else "repair"
        if attempt_kind == "repair":
            if last_error is None:
                raise RuntimeError(
                    f"Internal error: model repair attempt {attempt_number} for proposal "
                    f"{model_idx} has no prior error."
                )
            logger.info(
                f"[step {step_num}] Retrying model code generation for proposal {model_idx}: "
                f"attempt {attempt_number}/{max_attempts} will use a repair prompt. "
                f"Previous error: {last_error}"
            )
            model_repair_context: str = build_api_discovery_report(
                generated_code=attempts[-1].code,
                error_message=last_error,
                runtime_namespace=get_pymc_namespace(),
            )
            prompt = deps.prompts_reg.render_code_repair_prompt(
                base_prompt=base_prompt,
                previous_code=attempts[-1].code,
                error_message=last_error,
                repair_context=model_repair_context,
            )

        prompts.append(prompt)
        logger.debug(
            format_log_block(
                title=(
                    f"[step {step_num}] Model code-generation {attempt_kind} prompt "
                    f"for proposal {model_idx} attempt "
                    f"{attempt_number}/{max_attempts} ({len(prompt)} chars)"
                ),
                body=prompt,
            )
        )

        raw_response: str = ""
        model_code: str = ""
        try:
            raw_response = str(
                deps.code_gen_backend.call(
                    prompt=prompt,
                    response_type=None,
                    verbosity=deps.verbosity,
                )
            )
        except Exception as exc:
            error_message: str = format_exception_msg(exc)
            logger.error(
                f"[step {step_num}] Model code-generation backend call failed for proposal "
                f"{model_idx} attempt {attempt_number}/{max_attempts}: {error_message}"
            )
            raise RuntimeError(
                f"[step {step_num}] Model code-generation backend call failed for proposal "
                f"{model_idx} attempt {attempt_number}/{max_attempts}. Full traceback:\n{error_message}"
            ) from exc

        raw_responses.append(raw_response)
        logger.debug(
            format_log_block(
                title=(
                    f"[step {step_num}] Model code-generation {attempt_kind} raw response "
                    f"for proposal {model_idx} attempt "
                    f"{attempt_number}/{max_attempts} "
                    f"({len(raw_response)} chars)"
                ),
                body=raw_response,
            )
        )

        try:
            model_code = _parse_code_generation_response(
                raw_response=raw_response,
                target=f"proposal {model_idx}",
            )
            if len(model_code) == 0:
                raise RuntimeError(
                    f"Code-generation LLM returned an empty code string for proposal {model_idx}. "
                    f"Raw response:\n{raw_response}"
                )
        except Exception as exc:
            last_error = format_exception_msg(exc)
            fit_result = _failed_codegen_fit_result(
                ans=ans,
                model_idx=model_idx,
                task=deps.prompts_reg.task_string,
                error_message=last_error,
            )
            attempts.append(
                CodeGenerationAttempt(
                    stage="pymc_model",
                    target=f"proposal {model_idx}",
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    attempt_kind=attempt_kind,
                    failure_stage="parse",
                    prompt=prompt,
                    raw_response=raw_response,
                    code=model_code,
                    success=False,
                    error=last_error,
                )
            )
            logger.error(
                f"[step {step_num}] Model code-generation parse attempt "
                f"{attempt_number}/{max_attempts} failed for proposal "
                f"{model_idx}: {last_error}"
            )
            continue

        logger.debug(
            format_log_block(
                title=(
                    f"[step {step_num}] Generated model code for proposal {model_idx} "
                    f"attempt {attempt_number}/{max_attempts} "
                    f"({len(model_code)} chars)"
                ),
                body=model_code,
            )
        )

        try:
            fit_result = fit_single_model(
                data=deps.data,
                ans=ans,
                model_idx=model_idx,
                model_code=model_code,
                task=deps.prompts_reg.task_string,
            )
            if fit_result["model"] is None or fit_result["map_estimate"] is None:
                if "error" in fit_result:
                    raise RuntimeError(str(fit_result["error"]))
                raise RuntimeError(
                    f"Generated model code for proposal {model_idx} did not produce a fitted model. "
                    f"Fit result keys: {list(fit_result.keys())}."
                )
        except Exception as exc:
            last_error = format_exception_msg(exc)
            fit_result = _failed_codegen_fit_result(
                ans=ans,
                model_idx=model_idx,
                task=deps.prompts_reg.task_string,
                error_message=last_error,
            )
            attempts.append(
                CodeGenerationAttempt(
                    stage="pymc_model",
                    target=f"proposal {model_idx}",
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                    attempt_kind=attempt_kind,
                    failure_stage="execute",
                    prompt=prompt,
                    raw_response=raw_response,
                    code=model_code,
                    success=False,
                    error=last_error,
                )
            )
            logger.error(
                f"[step {step_num}] Model code-generation attempt "
                f"{attempt_number}/{max_attempts} failed for proposal "
                f"{model_idx}: {last_error}"
            )
            continue

        attempts.append(
            CodeGenerationAttempt(
                stage="pymc_model",
                target=f"proposal {model_idx}",
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                attempt_kind=attempt_kind,
                failure_stage="none",
                prompt=prompt,
                raw_response=raw_response,
                code=model_code,
                success=True,
                error=None,
            )
        )
        return model_code, fit_result, tuple(prompts), tuple(raw_responses), tuple(attempts)

    logger.error(
        f"[step {step_num}] Model code-generation FAILED after all "
        f"{max_attempts} attempt(s) for proposal {model_idx}."
    )
    return None, fit_result, tuple(prompts), tuple(raw_responses), tuple(attempts)


def _phase_codegen(
    *,
    deps: RunDeps,
    response_dict: Dict[str, Any],
    step_num: int,
) -> Tuple[
    Dict[str, Any],
    Tuple[str, ...],
    Tuple[str, ...],
    Tuple[CodeGenerationAttempt, ...],
    Dict[str, Dict[str, Any]],
]:
    """Run Phase 3 model code generation with per-proposal execute-repair loops."""
    ans: Dict[str, Any] = deps.prompts_reg.build_ans_dict(description=response_dict["description"])
    codegen_prompts: List[str] = []
    codegen_responses: List[str] = []
    code_generation_attempts: List[CodeGenerationAttempt] = []
    fit_results: Dict[str, Dict[str, Any]] = {}

    logger.info(
        f"[step {step_num}] Phase 3: code generation for {len(response_dict['proposals'])} proposal(s)."
    )
    for ix, proposal_config in response_dict["proposals"].items():
        entity_value: Any
        priors: Dict[str, str]
        entity_value, priors = deps.prompts_reg.extract_proposal_fields(
            proposal_config=proposal_config,
            ans=ans,
            ix=ix,
        )
        coder_prompt: str = deps.prompts_reg.render_code_gen_prompt(
            entity_value=entity_value,
            priors=priors,
        )
        logger.debug(
            format_log_block(
                title=f"[step {step_num}] Base code-gen prompt for proposal {ix}",
                body=coder_prompt,
            )
        )

        model_code: Optional[str]
        model_fit_result: Dict[str, Any]
        proposal_prompts: Tuple[str, ...]
        proposal_responses: Tuple[str, ...]
        proposal_attempts: Tuple[CodeGenerationAttempt, ...]
        model_code, model_fit_result, proposal_prompts, proposal_responses, proposal_attempts = (
            _generate_and_fit_model_code(
                deps=deps,
                ans=ans,
                model_idx=ix,
                base_prompt=coder_prompt,
                step_num=step_num,
            )
        )
        codegen_prompts.extend(proposal_prompts)
        codegen_responses.extend(proposal_responses)
        code_generation_attempts.extend(proposal_attempts)
        fit_results[ix]: Dict[str, Any] = model_fit_result
        if model_code is not None:
            ans["pymc_models"][ix]: str = model_code

    successful_models: int = len(ans["pymc_models"])
    proposed_models: int = len(response_dict["proposals"])
    logger.debug(
        f"[step {step_num}] Code gen complete: "
        f"{successful_models}/{proposed_models} proposals fit successfully"
    )
    logger.info(
        f"[step {step_num}] Phase 3 complete: "
        f"{successful_models}/{proposed_models} proposal(s) generated executable code."
    )
    return ans, tuple(codegen_prompts), tuple(codegen_responses), tuple(code_generation_attempts), fit_results


def _phase_fit(
    *,
    deps: RunDeps,
    ans: Dict[str, Any],
    fit_results: Dict[str, Dict[str, Any]],
    step_num: int,
) -> Tuple[
    Union[str, int],
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    FitState,
    str,
    Union[str, List[str]],
]:
    """Run Phase 4 selection from the Phase-3 reusable fit results.

    Phase 3 already executes and fits each generated candidate while
    validating the code-generation attempts. Phase 4 therefore only selects
    the best successful fit by AIC and extracts the domain-specific fit state;
    it does not re-run PyMC code.
    """
    logger.info(f"[step {step_num}] Phase 4: selecting best generated model fit.")
    best_idx: str
    model: Any
    map_estimate: Dict[str, Any]
    metrics: Dict[str, Any]
    best_idx, model, map_estimate, metrics = select_best_fit_result(fit_results=fit_results)
    logger.debug(f"[step {step_num}] All model fit results:")
    for model_idx, model_result in fit_results.items():
        aic_val: float = model_result["metrics"]["aic"]
        status_label: str = "pass" if model_result["model"] is not None else "FAILED"
        logger.debug(f"  [{model_idx}] AIC={aic_val:.1f}  {status_label}")

    if model is None or map_estimate is None:
        raise RuntimeError(
            f"[step {step_num}] All PyMC models failed to fit (model or map_estimate is None)."
        )

    model_structure: Union[str, List[str]] = ans[deps.entity_key][best_idx]
    model_code: str = ans["pymc_models"][best_idx]
    new_fit_state: FitState = deps.plotting_reg.extract_fit_state(
        fit_results=fit_results,
        best_idx=best_idx,
        ans=ans,
    )
    logger.debug(
        f"[step {step_num}] Best model: idx={best_idx}  family={model_structure}  AIC={metrics['aic']:.1f}"
    )
    logger.debug(f"[step {step_num}] Best model code:\n{model_code}")
    if deps.prompts_reg.should_log_map_estimate():
        logger.debug(f"[step {step_num}] MAP estimate: {clean_params(map_estimate)}")
    logger.info(
        f"[step {step_num}] Phase 4 complete: best family={model_structure}  AIC={metrics['aic']:.1f}"
    )
    model = None
    return best_idx, map_estimate, metrics, fit_results, new_fit_state, model_code, model_structure


def _phase_plot_overlay(
    *,
    state: StepState,
    deps: RunDeps,
    fit_state: FitState,
    model_structure: Union[str, List[str]],
    best_idx: Union[str, int],
    step_num: int,
) -> str:
    """Render the fit overlay for this step and return its path.

    If plotting raises, we fall back to ``state.fit_path`` — which is the
    last successful fit overlay path (or the initial plot at step 0).
    """
    model_structure_slug: str = (
        "_".join(model_structure) if isinstance(model_structure, list) else model_structure
    )
    new_fit_path: str = os.path.join(
        deps.out_dir,
        f"step_{step_num:03d}-phase_2-fit_overlay-{model_structure_slug}.png",
    )
    try:
        deps.plotting_reg.plot_fit_overlay(
            data=deps.data,
            fit_state=fit_state,
            path=new_fit_path,
            best_idx=best_idx,
        )
        return new_fit_path
    except Exception as exc:
        logger.exception(
            f"Plot failed at step {step_num} — falling back to previous fit path. "
            f"Exception: {format_exception_msg(exc)}"
        )
        return state.fit_path


def _phase_summary(
    *,
    state: StepState,
    deps: RunDeps,
    step_num: int,
    tool_results: Tuple[DiagnosticToolResult, ...],
    model_structure: Union[str, List[str]],
    model_code: str,
    ans: Dict[str, Any],
    metrics: Dict[str, Any],
    selected_tool_names: str,
) -> Tuple[str, str]:
    """Generate this step's summary and return ``(prompt, summary_text)``.

    At step 0 the initial-summary template is used; at step > 0 the
    feedback-summary template is used. Both return a plain string which
    the reducer appends to ``state.summary_trajectory``.
    """
    if step_num == 0:
        plot_type_description: str = deps.plot_type_descriptions[
            deps.plotting_reg.get_default_plot_description()
        ]
        summary_prompt: str = deps.prompts_reg.render_initial_summary(
            entity_value=model_structure,
            pymc_code=model_code,
            description=ans["description"],
            aic_score=round(float(metrics["aic"]), 1),
            plot_description=plot_type_description,
        )
    else:
        plot_type_description = _latest_plot_description(tool_results=tool_results, deps=deps)
        tool_output_type: str
        tool_output_summary: str
        tool_output_type, tool_output_summary = _derive_tool_summary_fields(tool_results=tool_results)
        summary_prompt = deps.prompts_reg.render_feedback_summary(
            entity_value=model_structure,
            pymc_code=model_code,
            description=ans["description"],
            aic_score=round(float(metrics["aic"]), 1),
            plot_description=plot_type_description,
            tool_name=selected_tool_names,
            tool_output_type=tool_output_type,
            tool_output_summary=tool_output_summary if tool_output_type == "numeric" else "N/A",
        )

    summary_title: str = "Initial summary prompt" if step_num == 0 else "Feedback summary prompt"
    logger.debug(
        format_log_block(
            title=f"[step {step_num}] {summary_title}",
            body=summary_prompt,
        )
    )
    summary: str = deps.backend.call(prompt=summary_prompt, verbosity=deps.verbosity)
    summary_response_title: str = "Initial summary response" if step_num == 0 else "Feedback summary response"
    logger.debug(
        format_log_block(
            title=f"[step {step_num}] {summary_response_title}",
            body=summary,
        )
    )
    return summary_prompt, summary


# ══════════════════════════════════════════════════════════════════════════
# Orchestration: _execute_step (SHELL) + _reduce_step (CORE) + helpers
# ══════════════════════════════════════════════════════════════════════════


def _selected_tool_names(*, tool_results: Tuple[DiagnosticToolResult, ...]) -> str:
    """Human-readable comma-joined list of invoked tool names (or ``"none"``)."""
    if len(tool_results) == 0:
        return "none"
    return ", ".join(tool_result.tool_name for tool_result in tool_results)


def _execute_step(*, state: StepState, deps: RunDeps) -> StepObservation:
    """Execute one iteration of the refinement pipeline.

    Runs the six phases in order (diagnostic → proposal → codegen → fit)
    followed by plot overlay generation and summary generation. Every
    phase can raise; the first phase to raise short-circuits the rest and
    produces an ``error``-kind observation pinpointing exactly where the
    failure occurred.

    At step > 0, if the proposal phase returns ``COMPLETE``, we skip
    codegen/fit/summary and return a ``complete``-kind observation that
    the reducer turns into ``StepStatus.complete``.
    """
    step_num: int = state.step_num + 1
    deps.pbar.set_description("Step 0: proposal" if step_num == 0 else f"step {step_num}: feedback")
    gc.collect()

    # ── Phase 1: diagnostic ────────────────────────────────────────────
    try:
        (
            tool_results,
            generated_tool_results,
            diagnostic_code_generation_attempts,
            tools_offered,
            diagnostic_prompt,
        ) = _phase_diagnostic(
            state=state,
            deps=deps,
            step_num=step_num,
        )
    except Exception as exc:
        return StepObservation(
            kind=ObservationKind.error,
            step_num=step_num,
            error=f"diagnostic phase failed: {format_exception_msg(exc)}",
            error_phase=PhaseName.diagnostic,
        )
    selected_tool_names: str = _selected_tool_names(tool_results=tool_results)

    # ── Phase 2: proposal ──────────────────────────────────────────────
    try:
        proposal_prompt, proposal_images, response_dict, phase2_call_time_s = _phase_proposal(
            state=state,
            deps=deps,
            step_num=step_num,
            tool_results=tool_results,
        )
    except Exception as exc:
        return StepObservation(
            kind=ObservationKind.error,
            step_num=step_num,
            error=f"proposal phase failed: {format_exception_msg(exc)}",
            error_phase=PhaseName.proposal,
            diagnostic_prompt=diagnostic_prompt,
            diagnostic_tools_offered=tools_offered,
            tool_results=tool_results,
            generated_tool_results=generated_tool_results,
            selected_tool_names=selected_tool_names,
        )

    is_complete: bool = step_num > 0 and _is_complete_feedback_description(
        description=response_dict["description"]
    )
    if is_complete:
        return StepObservation(
            kind=ObservationKind.complete,
            step_num=step_num,
            diagnostic_prompt=diagnostic_prompt,
            diagnostic_tools_offered=tools_offered,
            tool_results=tool_results,
            generated_tool_results=generated_tool_results,
            selected_tool_names=selected_tool_names,
            model_spec_prompt=proposal_prompt,
            model_spec_images=proposal_images,
            model_spec_response_dict=response_dict,
            phase2_call_time_s=phase2_call_time_s,
        )

    # ── Phase 3: codegen ───────────────────────────────────────────────
    try:
        (
            ans,
            codegen_prompts,
            codegen_response_codes,
            model_code_generation_attempts,
            fit_results,
        ) = _phase_codegen(
            deps=deps,
            response_dict=response_dict,
            step_num=step_num,
        )
    except Exception as exc:
        return StepObservation(
            kind=ObservationKind.error,
            step_num=step_num,
            error=f"codegen phase failed: {format_exception_msg(exc)}",
            error_phase=PhaseName.codegen,
            diagnostic_prompt=diagnostic_prompt,
            diagnostic_tools_offered=tools_offered,
            tool_results=tool_results,
            generated_tool_results=generated_tool_results,
            selected_tool_names=selected_tool_names,
            model_spec_prompt=proposal_prompt,
            model_spec_images=proposal_images,
            model_spec_response_dict=response_dict,
            phase2_call_time_s=phase2_call_time_s,
        )

    if len(ans["pymc_models"]) == 0 or len(ans[deps.entity_key]) == 0:
        error_detail: str = (
            f"VLM proposed no usable models at step {step_num}. "
            f"pymc_models={len(ans['pymc_models'])}, "
            f"{deps.entity_key}={len(ans[deps.entity_key])}."
        )
        logger.error(error_detail)
        return StepObservation(
            kind=ObservationKind.error,
            step_num=step_num,
            error=error_detail,
            error_phase=PhaseName.codegen,
            diagnostic_prompt=diagnostic_prompt,
            diagnostic_tools_offered=tools_offered,
            tool_results=tool_results,
            generated_tool_results=generated_tool_results,
            selected_tool_names=selected_tool_names,
            model_spec_prompt=proposal_prompt,
            model_spec_images=proposal_images,
            model_spec_response_dict=response_dict,
            model_code_generation_prompts=codegen_prompts,
            model_code_generation_responses=codegen_response_codes,
            model_code_generation_attempts=model_code_generation_attempts,
            model_spec_state=ans,
            fit_results=fit_results,
            phase2_call_time_s=phase2_call_time_s,
        )

    # ── Phase 4: fit ───────────────────────────────────────────────────
    try:
        fit_best_idx, fit_map, fit_metrics, fit_results, new_fit_state, fit_code, fit_structure = _phase_fit(
            deps=deps,
            ans=ans,
            fit_results=fit_results,
            step_num=step_num,
        )
    except Exception as exc:
        return StepObservation(
            kind=ObservationKind.error,
            step_num=step_num,
            error=f"fit phase failed: {format_exception_msg(exc)}",
            error_phase=PhaseName.fit,
            diagnostic_prompt=diagnostic_prompt,
            diagnostic_tools_offered=tools_offered,
            tool_results=tool_results,
            generated_tool_results=generated_tool_results,
            selected_tool_names=selected_tool_names,
            model_spec_prompt=proposal_prompt,
            model_spec_images=proposal_images,
            model_spec_response_dict=response_dict,
            model_code_generation_prompts=codegen_prompts,
            model_code_generation_responses=codegen_response_codes,
            model_code_generation_attempts=model_code_generation_attempts,
            model_spec_state=ans,
            fit_results=fit_results,
            phase2_call_time_s=phase2_call_time_s,
        )

    # ── Plot overlay + summary (both read the successful fit result) ──
    new_fit_path: str = _phase_plot_overlay(
        state=state,
        deps=deps,
        fit_state=new_fit_state,
        model_structure=fit_structure,
        best_idx=fit_best_idx,
        step_num=step_num,
    )
    try:
        summary_prompt, summary = _phase_summary(
            state=state,
            deps=deps,
            step_num=step_num,
            tool_results=tool_results,
            model_structure=fit_structure,
            model_code=fit_code,
            ans=ans,
            metrics=fit_metrics,
            selected_tool_names=selected_tool_names,
        )
    except Exception as exc:
        return StepObservation(
            kind=ObservationKind.error,
            step_num=step_num,
            error=f"summary phase failed: {format_exception_msg(exc)}",
            error_phase=PhaseName.summary,
            diagnostic_prompt=diagnostic_prompt,
            diagnostic_tools_offered=tools_offered,
            tool_results=tool_results,
            generated_tool_results=generated_tool_results,
            selected_tool_names=selected_tool_names,
            model_spec_prompt=proposal_prompt,
            model_spec_images=proposal_images,
            model_spec_response_dict=response_dict,
            model_code_generation_prompts=codegen_prompts,
            model_code_generation_responses=codegen_response_codes,
            model_code_generation_attempts=model_code_generation_attempts,
            model_spec_state=ans,
            fit_results=fit_results,
            best_idx=fit_best_idx,
            new_model_code=fit_code,
            new_model_structure=fit_structure,
            new_map_estimate=fit_map,
            new_metrics=fit_metrics,
            new_fit_state=new_fit_state,
            new_fit_path=new_fit_path,
            phase2_call_time_s=phase2_call_time_s,
        )

    logger.debug(
        f"step {step_num}: family={fit_structure}  AIC={fit_metrics['aic']:.1f}  "
        f"tools={selected_tool_names}  ({phase2_call_time_s:.1f}s)"
    )
    logger.info(
        f"[step {step_num}] Complete: family={fit_structure}  "
        f"AIC={fit_metrics['aic']:.1f}  tools={selected_tool_names}"
    )

    return StepObservation(
        kind=ObservationKind.ok,
        step_num=step_num,
        diagnostic_prompt=diagnostic_prompt,
        diagnostic_tools_offered=tools_offered,
        tool_results=tool_results,
        generated_tool_results=generated_tool_results,
        selected_tool_names=selected_tool_names,
        model_spec_prompt=proposal_prompt,
        model_spec_images=proposal_images,
        model_spec_response_dict=response_dict,
        phase2_call_time_s=phase2_call_time_s,
        model_code_generation_prompts=codegen_prompts,
        model_code_generation_responses=codegen_response_codes,
        model_code_generation_attempts=model_code_generation_attempts,
        model_spec_state=ans,
        fit_results=fit_results,
        best_idx=fit_best_idx,
        new_model_code=fit_code,
        new_model_structure=fit_structure,
        new_map_estimate=fit_map,
        new_metrics=fit_metrics,
        new_fit_state=new_fit_state,
        new_fit_path=new_fit_path,
        summary_prompt=summary_prompt,
        summary=summary,
    )


def _build_step_record(
    *,
    state: StepState,
    obs: StepObservation,
    deps: RunDeps,
) -> Dict[str, Any]:
    """Build the parquet-friendly dict capturing everything about this step.

    The shape differs by ``obs.kind``:

    - ``error`` → includes ``error`` + partial data captured before the failure
    - ``complete`` → includes the proposal response with ``description`` set
      to ``"COMPLETE"`` and no codegen/fit fields (they were not run)
    - ``ok`` → full fit result, summary, codegen, tool history, etc.
    """
    diagnostic_images_list: List[str] = _collect_image_artifact_paths(
        tool_results=obs.tool_results,
    )
    images_sent: List[str] = [state.fit_path] + diagnostic_images_list

    record: Dict[str, Any] = {
        "step": obs.step_num,
        "fit_overlay_path": state.fit_path,
        "model_spec_prompt": obs.model_spec_prompt,
        "diagnostic_tools_offered": (
            list(obs.diagnostic_tools_offered) if len(obs.diagnostic_tools_offered) > 0 else None
        ),
        "diagnostic_tool_results": _serialize_tool_results(tool_results=obs.tool_results),
        "generated_tools": _serialize_generated_tools(generated_tools=obs.generated_tool_results),
        "model_spec_response": obs.model_spec_response_dict,
        "selected_diagnostic_tools": obs.selected_tool_names,
        "phase2_call_time_s": obs.phase2_call_time_s,
        "diagnosis_prompt": obs.diagnostic_prompt,
        "model_spec_images": images_sent,
        "model_code_generation_prompts": list(obs.model_code_generation_prompts),
        "model_code_generation_responses": list(obs.model_code_generation_responses),
        "model_code_generation_attempts": [
            code_generation_attempt.model_dump()
            for code_generation_attempt in obs.model_code_generation_attempts
        ],
    }

    if obs.kind is ObservationKind.error:
        record["error"] = obs.error
        record["error_phase"] = obs.error_phase.value if obs.error_phase is not None else None
        if obs.model_spec_state is not None:
            record["pymc_models"] = obs.model_spec_state["pymc_models"]
        if obs.fit_results is not None:
            record["fit_results_all"] = _serialize_fit_results(obs.fit_results)
        # Preserve the running bests from prior successful steps (if any),
        # so an error mid-run does not lose the schema columns. Step-level
        # fields are None because this step produced no winning proposal.
        record["step_best_proposed_model_idx"] = None
        record["step_best_proposed_model_structure"] = None
        record["step_best_proposed_map_estimate"] = None
        record["step_best_proposed_model_aic"] = None
        record["step_best_proposed_model_code"] = None
        record["run_best_model_structure"] = state.best_model_structure
        record["run_best_model_aic"] = state.best_aic if state.best_aic != float("inf") else None
        record["run_best_map_estimate"] = (
            clean_params(state.best_map_estimate) if state.best_map_estimate is not None else None
        )
        record["run_best_model_code"] = state.best_model_code
        return record

    if obs.kind is ObservationKind.complete:
        record["fit_overlay_path"] = state.fit_path
        record["model_spec_description"] = "COMPLETE"
        step_aic_complete: Optional[float] = (
            state.current_metrics["aic"]
            if state.current_metrics is not None and "aic" in state.current_metrics
            else None
        )
        record["step_best_proposed_model_structure"] = state.current_model_structure
        record["step_best_proposed_map_estimate"] = (
            clean_params(state.current_map_estimate) if state.current_map_estimate is not None else {}
        )
        record["step_best_proposed_model_aic"] = step_aic_complete
        record["step_best_proposed_model_code"] = state.current_model
        record["step_best_proposed_model_idx"] = None
        record["metrics"] = state.current_metrics
        record["run_best_model_structure"] = state.best_model_structure
        record["run_best_model_aic"] = state.best_aic if state.best_aic != float("inf") else None
        record["run_best_map_estimate"] = (
            clean_params(state.best_map_estimate) if state.best_map_estimate is not None else None
        )
        record["run_best_model_code"] = state.best_model_code
        return record

    # ObservationKind.ok — the full-fat record
    if obs.new_fit_path is None or obs.new_fit_state is None or obs.new_metrics is None:
        raise RuntimeError(
            f"[step {obs.step_num}] ok observation missing required fit fields "
            f"(new_fit_path / new_fit_state / new_metrics). This is a bug in _execute_step."
        )
    record["fit_overlay_path"] = obs.new_fit_path
    record["model_spec_description"] = obs.model_spec_state["description"]
    record["pymc_models"] = obs.model_spec_state["pymc_models"]

    # Per-step (this step's winning proposal among the K candidates)
    step_aic_ok: Optional[float] = (
        obs.new_metrics["aic"] if obs.new_metrics is not None and "aic" in obs.new_metrics else None
    )
    record["step_best_proposed_model_idx"] = obs.best_idx
    record["step_best_proposed_model_structure"] = obs.new_model_structure
    record["step_best_proposed_map_estimate"] = clean_params(obs.new_map_estimate)
    record["step_best_proposed_model_aic"] = step_aic_ok
    record["step_best_proposed_model_code"] = obs.new_model_code
    record["metrics"] = obs.new_metrics
    record["fit_results_all"] = _serialize_fit_results(obs.fit_results)

    # Run-level bests (cumulative up-to-and-including this step; based on AIC).
    # The reducer has already folded the ok observation into state.best_* via
    # _reduce_step, but this record is built *before* the reducer runs, so we
    # compute the post-fold values ourselves: compare this step's AIC against
    # the prior running best and pick the winner. This keeps the run_best_*
    # columns semantically identical to what the reducer will store.
    new_aic: Optional[float] = obs.new_metrics.get("aic") if obs.new_metrics is not None else None
    improved_run_best: bool = new_aic is not None and new_aic < state.best_aic
    run_best_aic_value: Optional[float]
    if improved_run_best:
        run_best_aic_value = new_aic
        run_best_structure_value: Optional[Any] = obs.new_model_structure
        run_best_map_value: Optional[Dict[str, Any]] = clean_params(obs.new_map_estimate)
        run_best_code_value: Optional[str] = obs.new_model_code
    else:
        run_best_aic_value = state.best_aic if state.best_aic != float("inf") else None
        run_best_structure_value = state.best_model_structure
        run_best_map_value = (
            clean_params(state.best_map_estimate) if state.best_map_estimate is not None else None
        )
        run_best_code_value = state.best_model_code
    record["run_best_model_structure"] = run_best_structure_value
    record["run_best_model_aic"] = run_best_aic_value
    record["run_best_map_estimate"] = run_best_map_value
    record["run_best_model_code"] = run_best_code_value

    if obs.step_num == 0:
        record["initial_plot_path"] = state.fit_path  # state.fit_path is the plot path at step 0
        record["initial_summary"] = obs.summary
        record["initial_summary_prompt"] = obs.summary_prompt
    else:
        record["feedback_summary"] = obs.summary
        record["feedback_summary_prompt"] = obs.summary_prompt
    record["summary"] = obs.summary

    record.update(
        deps.prompts_reg.build_step_record_extras(
            ans=obs.model_spec_state,
            fit_state=obs.new_fit_state,
        )
    )
    return record


def _reduce_step(
    *,
    state: StepState,
    obs: StepObservation,
    deps: RunDeps,
) -> StepState:
    """Pure ``(state, observation) → new_state`` transition.

    This is the single source of truth for state evolution: to remember
    a new piece of cross-step context, add the field to ``StepState`` and
    one line here. Phases do not write to state; the reducer is the
    only writer.
    """
    record: Dict[str, Any] = _build_step_record(state=state, obs=obs, deps=deps)
    new_records: Tuple[Dict[str, Any], ...] = state.step_records + (record,)

    if obs.kind is ObservationKind.error:
        return validated_copy(
            state=state,
            update={
                "step_num": obs.step_num,
                "status": StepStatus.error,
                "error": obs.error,
                "error_phase": obs.error_phase,
                "step_records": new_records,
            },
        )

    if obs.kind is ObservationKind.complete:
        return validated_copy(
            state=state,
            update={
                "step_num": obs.step_num,
                "status": StepStatus.complete,
                "step_records": new_records,
            },
        )

    # ObservationKind.ok — advance current + update monotone global best
    if (
        obs.new_model_code is None
        or obs.new_model_structure is None
        or obs.new_map_estimate is None
        or obs.new_metrics is None
        or obs.new_fit_state is None
        or obs.new_fit_path is None
        or obs.summary is None
    ):
        raise RuntimeError(
            f"[step {obs.step_num}] ok observation missing fields required for state advance. "
            f"This is a bug in _execute_step."
        )
    improved: bool = obs.new_metrics["aic"] < state.best_aic
    return validated_copy(
        state=state,
        update={
            "step_num": obs.step_num,
            "current_model": obs.new_model_code,
            "current_model_structure": obs.new_model_structure,
            "current_map_estimate": obs.new_map_estimate,
            "current_metrics": obs.new_metrics,
            "current_fit_state": obs.new_fit_state,
            "fit_path": obs.new_fit_path,
            "best_aic": obs.new_metrics["aic"] if improved else state.best_aic,
            "best_model_structure": obs.new_model_structure if improved else state.best_model_structure,
            "best_model_code": obs.new_model_code if improved else state.best_model_code,
            "best_map_estimate": obs.new_map_estimate if improved else state.best_map_estimate,
            "best_metrics": obs.new_metrics if improved else state.best_metrics,
            "best_fit_state": obs.new_fit_state if improved else state.best_fit_state,
            "best_fit_path": obs.new_fit_path if improved else state.best_fit_path,
            "tested_model_structures": state.tested_model_structures + (obs.new_model_structure,),
            "selected_tool_history": state.selected_tool_history + (obs.selected_tool_names,),
            "summary_trajectory": state.summary_trajectory + (obs.summary,),
            "step_records": new_records,
        },
    )


def _build_run_deps(
    *,
    config: ExperimentConfig,
    dataset: Dict[str, Any],
    out_dir: str,
    verbosity: int,
    backend: VLMBackend,
    code_gen_backend: VLMBackend,
    tool_gen_backend: VLMBackend,
    prompts_reg: DomainPrompts,
    toolkit_reg: DomainToolkit,
    plotting_reg: DomainPlotting,
    pbar: ProgressBar,
    dynamic_tools: Dict[str, DynamicToolSpec],
) -> RunDeps:
    """Assemble the immutable ``RunDeps`` bag passed to every phase + the reducer.

    ``dynamic_tools`` is the run-scoped dict; empty when dynamic tools are
    disabled, seeded from the explicit registry when accumulating or running
    ``accumulated_only``. The caller constructs it once per ``run()`` call so
    dyn-tool entries never leak between datasets.
    """
    data: Any = dataset["data"]
    dataset_fields: Dict[str, Any] = prompts_reg.extract_dataset_fields(dataset)
    return RunDeps(
        backend=backend,
        code_gen_backend=code_gen_backend,
        code_gen_model=config.toolkit.code_gen_model or config.model.litellm_model,
        tool_gen_backend=tool_gen_backend,
        tool_gen_model=config.toolkit.tool_gen_model or config.model.litellm_model,
        prompts_reg=prompts_reg,
        toolkit_reg=toolkit_reg,
        plotting_reg=plotting_reg,
        pbar=pbar,
        data=data,
        dataset_fields=dataset_fields,
        out_dir=out_dir,
        verbosity=verbosity,
        max_steps=config.max_steps,
        domain=config.domain,
        toolkit_mode=config.toolkit.mode,
        max_tool_calls_per_step=config.toolkit.max_tool_calls_per_step,
        max_code_generation_attempts=config.toolkit.max_code_generation_attempts,
        max_tool_generation_attempts=config.toolkit.max_tool_generation_attempts,
        force_tool_call=config.toolkit.force_tool_call,
        accumulate_tools=config.toolkit.accumulate_tools,
        carry_forward=config.carry_forward,
        response_type=prompts_reg.get_response_type(),
        entity_key=prompts_reg.get_entity_key(),
        plot_type_descriptions=prompts_reg.get_plot_type_descriptions(),
        model_spec_proposal_prompt_template=prompts_reg.render_proposal_prompt(
            num_proposals=config.proposals_per_step
        ),
        model_spec_feedback_prompt_template=prompts_reg.get_feedback_prompt_template(
            num_proposals=config.proposals_per_step,
            max_steps=config.max_steps,
        ),
        dynamic_tools=dynamic_tools,
    )


def _stop_backends(
    *,
    backend: VLMBackend,
    code_gen_backend: VLMBackend,
    tool_gen_backend: VLMBackend,
) -> None:
    """Stop backend workers and force a GC cycle so resources are released."""
    backends_to_stop: List[Tuple[str, VLMBackend]] = [("primary", backend)]
    if code_gen_backend is not backend:
        backends_to_stop.append(("code-gen", code_gen_backend))
    if tool_gen_backend is not backend and tool_gen_backend is not code_gen_backend:
        backends_to_stop.append(("tool-gen", tool_gen_backend))

    for backend_label, backend_to_stop in backends_to_stop:
        try:
            backend_to_stop.stop()
        except Exception as exc:
            logger.debug(f"{backend_label} backend stop failed: {format_exception_msg(exc)}")
    gc.collect()


def _release_observation_fit_refs(*, obs: StepObservation) -> None:
    """Drop PyMC model handles on the observation's fit_results.

    Called immediately after the reducer commits the step so that PyMC
    ``model`` objects do not pile up in memory across iterations.
    """
    if obs.fit_results is not None:
        _release_fit_model_references(fit_results=obs.fit_results)


# ══════════════════════════════════════════════════════════════════════════
# run() — thin orchestrator over the shell/core cycle
# ══════════════════════════════════════════════════════════════════════════


@validate
def run(
    *,
    config: ExperimentConfig,
    dataset: Dict[str, Any],
    out_dir: str,
    verbosity: int,
) -> Dict[str, Any]:
    """Run the full VLM-guided fitting pipeline for one dataset.

    Loop shape::

        state = StepState(fit_path=<initial plot>)
        for _ in range(max_steps + 1):
            obs   = _execute_step(state, deps)     # SHELL
            state = _reduce_step(state, obs, deps) # CORE
            persist step records; update pbar
            if state.is_terminal: break

    Args:
        config: Validated experiment configuration.
        dataset: One row from the pkl file.
        out_dir: Per-dataset output directory for plots and artifacts.
        verbosity: Effective verbosity (may be reduced for multi-dataset runs).
    """
    # ── Domain registries ────────────────────────────────────────────────
    prompts_reg: DomainPrompts = DomainPrompts.of(config.domain)
    toolkit_reg: DomainToolkit = DomainToolkit.of(config.domain)
    plotting_reg: DomainPlotting = DomainPlotting.of(config.domain)

    requires_dynamic_generation: bool = config.toolkit.mode in (
        ToolkitMode.dynamic,
        ToolkitMode.generate_only,
    )
    if requires_dynamic_generation and not toolkit_reg.supports_dynamic_generation():
        raise ValueError(
            f"toolkit_mode={config.toolkit.mode.value!r} is not supported for "
            f"domain={config.domain.value!r}. "
            f"This domain does not support dynamic tool generation. "
            f"Use toolkit_mode='none' or toolkit_mode='expert'."
        )

    # ── Dataset extraction + progress bar ────────────────────────────────
    dataset_fields: Dict[str, Any] = prompts_reg.extract_dataset_fields(dataset)
    dataset_idx: int = dataset_fields["dataset_idx"]
    dataset_label: str = dataset_fields["dist_label"]

    # ── Backends ─────────────────────────────────────────────────────────
    backend_kwargs: Dict[str, Any] = config.model.to_backend_kwargs(
        verbosity=verbosity,
        max_rpm=config.parallel.per_llm_rpm,
    )
    backend_name: str = backend_kwargs.pop("backend")
    backend: VLMBackend = VLMBackend.of(
        backend_name,
        **backend_kwargs,
        dataset_prefix=f"{dataset_idx:03d}_{dataset_label[:10]}",
    )
    logger.debug(f"Backend: {type(backend).__name__} / {config.model.litellm_model}")

    code_gen_backend: VLMBackend = backend
    code_gen_kwargs: Optional[Dict[str, Any]] = config.toolkit.to_code_backend_kwargs(
        model=config.model,
        model_override=config.toolkit.code_gen_model,
        verbosity=verbosity,
        code_gen_max_rpm=config.parallel.per_code_gen_llm_rpm,
    )
    if code_gen_kwargs is not None:
        code_gen_backend_name: str = code_gen_kwargs.pop("backend")
        code_gen_backend = VLMBackend.of(
            code_gen_backend_name,
            **code_gen_kwargs,
            dataset_prefix=f"{dataset_idx:03d}_{dataset_label[:10]}",
        )
        logger.info(
            f"Code-gen backend: {type(code_gen_backend).__name__} / {config.toolkit.code_gen_model} "
            f"(per-worker RPM={config.parallel.per_code_gen_llm_rpm})"
        )
    else:
        logger.info(
            f"Code-gen backend: reusing main backend / {config.model.litellm_model} "
            f"(per-worker RPM={config.parallel.per_llm_rpm})"
        )

    tool_gen_backend: VLMBackend = backend
    tool_gen_kwargs: Optional[Dict[str, Any]] = config.toolkit.to_code_backend_kwargs(
        model=config.model,
        model_override=config.toolkit.tool_gen_model,
        verbosity=verbosity,
        code_gen_max_rpm=config.parallel.per_code_gen_llm_rpm,
    )
    if tool_gen_kwargs is not None:
        tool_gen_backend_name: str = tool_gen_kwargs.pop("backend")
        tool_gen_backend = VLMBackend.of(
            tool_gen_backend_name,
            **tool_gen_kwargs,
            dataset_prefix=f"{dataset_idx:03d}_{dataset_label[:10]}",
        )
        logger.info(
            f"Tool-gen backend: {type(tool_gen_backend).__name__} / {config.toolkit.tool_gen_model} "
            f"(per-worker RPM={config.parallel.per_code_gen_llm_rpm})"
        )
    else:
        logger.info(
            f"Tool-gen backend: reusing main backend / {config.model.litellm_model} "
            f"(per-worker RPM={config.parallel.per_llm_rpm})"
        )

    # ── Dynamic-tools dict (run-scoped; seeded from disk when accumulating) ─
    # Registry lives in the domain's own folder so time-series tools can
    # never be loaded into a distribution-fitting run (and vice versa),
    # even if the filename is shared between domains.
    registry_path: str = os.path.join(
        _THIS_DIR,
        "domains",
        config.domain.value,
        config.toolkit.tool_registry_filename,
    )

    if config.toolkit.mode is ToolkitMode.accumulated_only:
        if not os.path.exists(registry_path):
            raise FileNotFoundError(
                f"toolkit.mode='accumulated_only' requires a populated dynamic-tool registry file, "
                f"but no registry exists at {registry_path!r}. Run the sequential train/build phase first, "
                f"or pass --toolkit.tool-registry-filename pointing at an existing registry."
            )
        dynamic_tools: Dict[str, DynamicToolSpec] = load_dynamic_tools(path=registry_path)
        if len(dynamic_tools) == 0:
            raise ValueError(
                f"toolkit.mode='accumulated_only' requires a non-empty dynamic-tool registry file at "
                f"{registry_path!r}, but it loaded zero tools. Run the sequential train/build phase first, "
                f"or pass --toolkit.tool-registry-filename pointing at a populated registry."
            )
        logger.info(f"Loaded {len(dynamic_tools)} dynamic tool(s) from {registry_path}")
    elif config.toolkit.accumulate_tools:
        dynamic_tools = load_dynamic_tools(path=registry_path)
        if len(dynamic_tools) > 0:
            logger.info(f"Loaded {len(dynamic_tools)} dynamic tool(s) from {registry_path}")
    else:
        dynamic_tools = {}
    # Snapshot the pre-run size so the save-time observability log can
    # report grew-vs-unchanged without re-reading disk.
    initial_tool_count: int = len(dynamic_tools)
    initial_tool_names: Tuple[str, ...] = tuple(dynamic_tools.keys())

    os.makedirs(out_dir, exist_ok=True)
    pbar: ProgressBar = ProgressBar(
        total=1 + config.max_steps,
        desc=f"{dataset_label}[{dataset_idx}] → {config.model.litellm_model}",
        unit="step",
        disable=(verbosity == 0),
    )

    # ── Initial plot (side effect at the shell boundary) ────────────
    plot_path: str = os.path.join(out_dir, "step_000-initial_plot.png")
    plotting_reg.plot_initial_data(dataset["data"], save_path=plot_path)

    # ── Assemble deps + initial state (pre-step-0 sentinel) ──────────────
    deps: RunDeps = _build_run_deps(
        config=config,
        dataset=dataset,
        out_dir=out_dir,
        verbosity=verbosity,
        backend=backend,
        code_gen_backend=code_gen_backend,
        tool_gen_backend=tool_gen_backend,
        prompts_reg=prompts_reg,
        toolkit_reg=toolkit_reg,
        plotting_reg=plotting_reg,
        pbar=pbar,
        dynamic_tools=dynamic_tools,
    )
    state: StepState = StepState(fit_path=plot_path)

    # ══════════════════════════════════════════════════════════════════
    # UNIFIED LOOP: step 0 and step 1..max_steps share one body.
    # ══════════════════════════════════════════════════════════════════
    for _iteration in range(config.max_steps + 1):
        obs: StepObservation = _execute_step(state=state, deps=deps)
        if obs.kind is ObservationKind.error:
            error_phase_name: str = obs.error_phase.value if obs.error_phase is not None else "unknown"
            logger.error(
                f"[step {obs.step_num}] {error_phase_name} phase errored — terminating run.\n{obs.error}"
            )
        state = _reduce_step(state=state, obs=obs, deps=deps)

        # Persist every step to disk immediately — a crash mid-run still
        # leaves the completed steps recoverable.
        _write_dataset_run_log(
            steps=list(state.step_records),
            out_dir=out_dir,
            dataset_idx=dataset_idx,
            dataset_label=dataset_label,
        )
        _release_observation_fit_refs(obs=obs)
        pbar.update(1)
        gc.collect()
        if state.is_terminal:
            break

    # ── Build final result dict ──────────────────────────────────────────
    if state.status is StepStatus.error:
        pbar.close()
        result_dict: Dict[str, Any] = {
            "status": RunStatus.error.value,
            "dataset_idx": dataset_idx,
            "dataset_label": dataset_label,
            "error": state.error,
            "data": dataset["data"],
            "steps": list(state.step_records),
        }
        _flush_final_summary_columns(
            state=state,
            result=result_dict,
            out_dir=out_dir,
            dataset_idx=dataset_idx,
            dataset_label=dataset_label,
        )
        _stop_backends(
            backend=backend,
            code_gen_backend=code_gen_backend,
            tool_gen_backend=tool_gen_backend,
        )
        return result_dict

    # Success: state.status is either `complete` (early termination on COMPLETE)
    # or `running` (ran all max_steps without COMPLETE).
    # The progress bar always reports the monotone-best pair (family + AIC
    # from the same step) so users see the winning model of the run,
    # independent of the carry-forward strategy used to feed the VLM.
    if state.best_model_structure is not None and state.best_aic != float("inf"):
        pbar.success(f"done — {state.best_model_structure} (best AIC={state.best_aic:.1f})")
    elif state.current_model_structure is not None and state.current_metrics is not None:
        pbar.success(f"done — {state.current_model_structure} (AIC={state.current_metrics['aic']:.1f})")
    else:
        pbar.success("done — (no model fitted)")
    gc.collect()

    if config.toolkit.accumulate_tools and len(deps.dynamic_tools) > 0:
        save_dynamic_tools(path=registry_path, dynamic_tools=deps.dynamic_tools)
        final_tool_count: int = len(deps.dynamic_tools)
        added_count: int = final_tool_count - initial_tool_count
        if added_count > 0:
            newly_added_names: List[str] = [
                name for name in deps.dynamic_tools.keys() if name not in initial_tool_names
            ]
            logger.debug(
                f"Dynamic tool registry GREW: {initial_tool_count} -> {final_tool_count} "
                f"(+{added_count} new: {newly_added_names}) at {registry_path}"
            )
        else:
            logger.debug(
                f"Dynamic tool registry UNCHANGED at {final_tool_count} tool(s): {registry_path} "
                f"(VLM reused existing tool(s) this run without generating new ones)"
            )
        logger.info(f"Saved {final_tool_count} dynamic tool(s) to {registry_path}")

    run_best_map_estimate_cleaned: Optional[Dict[str, Any]] = (
        clean_params(state.best_map_estimate) if state.best_map_estimate is not None else None
    )
    run_best_aic_value: Optional[float] = state.best_aic if state.best_aic != float("inf") else None
    result_dict = {
        "status": RunStatus.ok.value,
        "dataset_idx": dataset_idx,
        "dataset_label": dataset_label,
        "run_best_model_structure": state.best_model_structure,
        "run_best_model_aic": run_best_aic_value,
        "run_best_map_estimate": run_best_map_estimate_cleaned,
        "run_best_model_code": state.best_model_code,
        "data": dataset["data"],
        "steps": list(state.step_records),
    }
    # Preserve domain-specific final-result extras that the old pipeline
    # appended after the loop (e.g. final kernels for time-series).
    if state.current_fit_state is not None:
        result_dict.update(
            prompts_reg.build_result_extras(dataset=dataset, fit_state=state.current_fit_state)
        )
    _flush_final_summary_columns(
        state=state,
        result=result_dict,
        out_dir=out_dir,
        dataset_idx=dataset_idx,
        dataset_label=dataset_label,
    )
    _stop_backends(
        backend=backend,
        code_gen_backend=code_gen_backend,
        tool_gen_backend=tool_gen_backend,
    )
    return result_dict


def _flush_final_summary_columns(
    *,
    state: StepState,
    result: Dict[str, Any],
    out_dir: str,
    dataset_idx: int,
    dataset_label: str,
) -> None:
    """Rewrite ``run_log.parquet`` with per-row run-level summary columns.

    Each row of ``run_log.parquet`` represents one step. The run-level
    summary columns (``status``, ``num_steps``, ``error``,
    ``total_time_s``) are identical across all rows of the same dataset
    — they reflect the *run-level* outcome. We write them at the end
    once the outcome is known, rather than leaving them as nulls from
    the per-step flushes.

    The per-step and cumulative run-best fields (``step_best_proposed_*``
    and ``run_best_*``) live on the per-row ``record`` built by
    ``_build_step_record`` — NOT here — because their values differ
    across rows (they grow monotonically with steps).
    """
    is_error: bool = result["status"] == RunStatus.error.value
    run_summary_fields: Dict[str, Any] = {
        "status": result["status"],
        "num_steps": len(result["steps"]),
        "error": result.get("error") if is_error else None,
        "total_time_s": sum(s.get("phase2_call_time_s", 0.0) for s in result["steps"]),
    }
    _write_dataset_run_log(
        steps=list(state.step_records),
        out_dir=out_dir,
        dataset_idx=dataset_idx,
        dataset_label=dataset_label,
        run_summary_fields=run_summary_fields,
    )


# ---------------------------------------------------------------------------
#  Parquet serialization helpers
# ---------------------------------------------------------------------------


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects to serializable equivalents."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    return obj


def _serialize_fit_results(fit_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Serialize fit_results for parquet storage.

    Drops non-serializable objects (PyMC Model, PyTensor variables).
    Converts numpy arrays to Python lists/floats.
    """
    serializable: Dict[str, Dict[str, Any]] = {}
    for model_idx, result in fit_results.items():
        entry: Dict[str, Any] = {}
        for key, value in result.items():
            if key == "model":
                entry["model_fitted"]: bool = value is not None
            elif key == "trend":
                entry["trend_fitted"]: bool = value is not None
            elif key == "map_estimate":
                entry["map_estimate"] = clean_params(value) if value is not None else None
            elif key == "metrics":
                entry["metrics"] = _clean_for_json(value)
            else:
                entry[key] = _clean_for_json(value)
        serializable[str(model_idx)] = entry
    return serializable


def _serialize_for_parquet(step_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a step record for parquet storage.

    Rules:
    - Dicts and lists -> JSON string via json.dumps (after numpy cleaning)
    - Numpy scalars -> Python native (int, float)
    - Numpy arrays -> lists (then JSON-serialized)
    - Strings, ints, floats -> kept as-is
    - None -> kept as None (parquet stores as null)
    """
    flat: Dict[str, Any] = {}
    for key, value in step_dict.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(_clean_for_json(value))
        elif isinstance(value, np.ndarray):
            flat[key] = json.dumps(value.tolist())
        elif isinstance(value, pd.Series):
            flat[key] = json.dumps(value.tolist())
        elif isinstance(value, (np.integer, np.floating)):
            flat[key] = value.item()
        else:
            flat[key] = value
    return flat


def _write_dataset_run_log(
    *,
    steps: List[Dict[str, Any]],
    out_dir: str,
    dataset_idx: int,
    dataset_label: str,
    run_summary_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Write per-dataset run_log.parquet with all accumulated steps so far.

    Called after EVERY reducer step in run(). Rewrites the full file
    each time (1-10 rows, microseconds). This ensures the parquet is on
    disk the moment a step completes — no waiting for the entire run.

    ``run_summary_fields`` carries run-level values that mirror
    ``summary.parquet`` columns. When provided (typically at final flush),
    those fields are written onto each step row so summary-style recovery
    is possible directly from ``run_log.parquet``.
    """
    resolved_summary_fields: Dict[str, Any] = {
        "status": None,
        "num_steps": None,
        "error": None,
        "total_time_s": None,
    }
    if run_summary_fields is not None:
        resolved_summary_fields.update(run_summary_fields)

    step_records: List[Dict[str, Any]] = []
    for step_dict in steps:
        serialized_step: Dict[str, Any] = _serialize_for_parquet(step_dict)
        step_record: Dict[str, Any] = {
            "dataset_idx": dataset_idx,
            "dataset_label": dataset_label,
            **serialized_step,
            **resolved_summary_fields,
        }
        if "error" in serialized_step and serialized_step["error"] is not None:
            step_record["step_error"] = serialized_step["error"]
        step_records.append(step_record)
    if len(step_records) == 0:
        return
    df: pd.DataFrame = pd.DataFrame(step_records)
    parquet_path: str = os.path.join(out_dir, "run_log.parquet")
    df.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
#  Dataset index parsing (supports single, comma-separated, and slice syntax)
# ---------------------------------------------------------------------------


def _parse_dataset_indices(
    raw: Optional[str],
    *,
    num_datasets: int,
) -> Optional[List[int]]:
    """Parse ``--dataset-idx`` into a list of integer indices.

    Supports three formats:
        - ``None`` → all datasets (returns ``None``)
        - ``"5"`` → ``[5]``
        - ``"0,1,8"`` → ``[0, 1, 8]``
        - ``"0:9"`` → ``list(range(0, 9))`` → ``[0, 1, ..., 8]``
        - ``"0:9:2"`` → ``list(range(0, 9, 2))`` → ``[0, 2, 4, 6, 8]``

    Raises:
        ValueError: If any index is out of range or the format is invalid.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if len(raw) == 0:
        return None

    indices: List[int]
    if ":" in raw:
        parts: List[str] = raw.split(":")
        if len(parts) == 2:
            start: int = int(parts[0]) if len(parts[0]) > 0 else 0
            stop: int = int(parts[1]) if len(parts[1]) > 0 else num_datasets
            indices = list(range(start, stop))
        elif len(parts) == 3:
            start = int(parts[0]) if len(parts[0]) > 0 else 0
            stop = int(parts[1]) if len(parts[1]) > 0 else num_datasets
            step: int = int(parts[2]) if len(parts[2]) > 0 else 1
            indices = list(range(start, stop, step))
        else:
            raise ValueError(f"Invalid slice format: {raw!r}. Expected 'start:stop' or 'start:stop:step'.")
    elif "," in raw:
        indices = [int(x.strip()) for x in raw.split(",")]
    else:
        indices = [int(raw)]

    for idx in indices:
        if idx < 0 or idx >= num_datasets:
            raise ValueError(
                f"Dataset index {idx} out of range. File has {num_datasets} datasets (0..{num_datasets - 1})."
            )

    return indices


# ---------------------------------------------------------------------------
#  Two-level parallel dataset runner (Concurry: outer pool × inner thread pool)
# ---------------------------------------------------------------------------


@validate
def _run_datasets_parallel(
    *,
    datasets: List[Dict[str, Any]],
    config: ExperimentConfig,
    run_dir: str,
    run_verbosity: int,
) -> List[Dict[str, Any]]:
    """Run datasets with two-level parallelism: outer pool × inner threads.

    - nproc=0, nthread=0: fully sequential (sync × sync)
    - nproc=0, nthread=4: single process, 4 threads per chunk
    - nproc=2, nthread=0: 2 outer workers, each sequential
    - nproc=2, nthread=3: 2 outer workers × 3 threads = 6 concurrent datasets

    API rate limiting is handled at the LLM layer: each ``run()`` call
    creates SlowBurn LLM workers with per-worker ``max_rpm`` budgets computed
    from ``config.parallel.per_llm_rpm`` and
    ``config.parallel.per_code_gen_llm_rpm``.
    """
    nproc: int = config.parallel.nproc
    nthread: int = config.parallel.nthread
    num_datasets: int = len(datasets)
    if num_datasets == 0:
        return []

    outer_mode: str = "sync" if nproc == 0 else "process"
    outer_workers: int = max(nproc, 1)
    effective_outer: int = min(outer_workers, num_datasets)

    logger.info(
        f"Running {num_datasets} dataset(s): "
        f"outer={outer_mode}(workers={effective_outer}), "
        f"inner={'sync' if nthread == 0 else 'thread'}(workers={max(nthread, 1)})"
    )

    runner: DatasetRunnerProcess = DatasetRunnerProcess.options(
        mode=outer_mode,
        max_workers=effective_outer,
    ).init(config=config, run_dir=run_dir, run_verbosity=run_verbosity, nthread=nthread)

    chunk_size: int = max(nthread, 1)
    chunks: List[List[Dict[str, Any]]] = [
        datasets[i : i + chunk_size] for i in range(0, num_datasets, chunk_size)
    ]

    futures: List[Any] = [runner.run_chunk(chunk) for chunk in chunks]  # Concurry futures
    show_progress: bool = num_datasets > 1
    chunk_results: List[Union[List[Dict[str, Any]], Exception]] = gather(
        futures,
        return_exceptions=True,
        progress=show_progress,
    )
    runner.stop()

    all_results: List[Dict[str, Any]] = []
    for i, chunk_res in enumerate(chunk_results):
        if isinstance(chunk_res, Exception):
            chunk_error_msg: str = format_exception_msg(chunk_res)
            logger.error(f"Chunk {i} raised: {chunk_error_msg}")
            for ds in chunks[i]:
                ds_fields: Dict[str, Any] = DomainPrompts.of(config.domain).extract_dataset_fields(ds)
                all_results.append(
                    {
                        "status": RunStatus.error.value,
                        "dataset_idx": ds_fields["dataset_idx"],
                        "dataset_label": ds_fields["dist_label"],
                        "steps": [],
                        "error": f"chunk {i} error: {chunk_error_msg}",
                    }
                )
        else:
            all_results.extend(chunk_res)

    return all_results


class _AsyncioTaskDestroyedFilter(logging.Filter):
    """Drop the LiteLLM/asyncio teardown-race ``Task was destroyed`` messages.

    Python's asyncio event loop emits ``Task was destroyed but it is pending!``
    at ``ERROR`` level whenever a still-pending task is garbage-collected
    without having been awaited. In this codebase this only happens during
    ``SlowBurn`` / LiteLLM teardown (LiteLLM's ``LoggingWorker._worker_loop``
    leaves an in-flight task when its parent thread exits). The warning is
    harmless — the response we care about has already been returned — but
    without this filter every code-gen run contributes spurious ``ERROR``
    lines that inflate error counts during log review.

    We filter only this exact message pattern so that genuine asyncio errors
    (cancellations, broken invariants, etc.) still surface.
    """

    _PATTERNS: ClassVar[Tuple[str, ...]] = (
        "Task was destroyed but it is pending",
        "Exception in callback",  # often follows the same teardown race
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message: str = record.getMessage()
        return not any(pattern in message for pattern in self._PATTERNS)


def _configure_logging(*, verbosity: int) -> None:
    """Set up logging: basicConfig + suppress noisy third-party loggers.

    The root logger is set to DEBUG so that per-dataset FileHandlers (added
    in experiment_workers) capture all output.  The StreamHandler's level is
    controlled by the user's ``--verbosity`` flag:
    - 0 → WARNING (errors only on stdout)
    - 1 → INFO (milestones on stdout)
    - 2 → DEBUG (everything on stdout)
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=r".*PyTensor could not link to a BLAS installation.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*pytensor\.config\.cxx.*identifiable `g\+\+` compiler.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Task was destroyed but it is pending.*",
    )

    if verbosity >= 2:
        stream_level: int = logging.DEBUG
    elif verbosity >= 1:
        stream_level = logging.INFO
    else:
        stream_level = logging.WARNING

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d [%(name)-20s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d::%H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(stream_level)
    _NOISY_LOGGER_NAMES: Tuple[str, ...] = (
        "matplotlib",
        "PIL",
        "urllib3",
        "httpcore",
        "httpx",
        "openai",
        "litellm",
        "LiteLLM",
        "numba",
        "asyncio",
        "arviz",
        "pymc",
        "pytensor",
        "filelock",
    )
    for noisy_logger_name in _NOISY_LOGGER_NAMES:
        logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

    # Attach the asyncio teardown-race filter at the root so the pattern is
    # suppressed regardless of which child logger emits it (``asyncio``,
    # ``asyncio.tasks``, or LiteLLM's ``LoggingWorker``).
    _ASYNCIO_FILTER: logging.Filter = _AsyncioTaskDestroyedFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(_ASYNCIO_FILTER)


@validate
def run_all(*, config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Run the full experiment: load data, create output dirs, dispatch, save results.

    This is the main entry point for both CLI and notebook usage.
    Handles dataset loading, directory creation, parallel dispatch,
    result printing, and result saving.
    """
    with open(config.data_pkl, "rb") as f:
        all_datasets: List[Dict[str, Any]] = pickle.load(f)
    logger.info(f"Loaded {len(all_datasets)} datasets from {config.data_pkl}")

    parsed_indices: Optional[List[int]] = _parse_dataset_indices(
        config.dataset_idx,
        num_datasets=len(all_datasets),
    )

    logger.info(f"Parsed dataset indices: {parsed_indices}")

    if parsed_indices is not None:
        datasets_to_run: List[Dict[str, Any]] = [all_datasets[i] for i in parsed_indices]
    else:
        datasets_to_run = all_datasets

    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name: str = os.path.splitext(os.path.basename(config.data_pkl))[0]
    run_dir: str = os.path.join(config.output.base_dir, config.output.expt, timestamp, data_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(
        format_log_block(
            title="MULTI-DATASET EXPERIMENT START",
            body=(
                f"Data file:          {config.data_pkl}\n"
                f"Datasets selected:  {len(datasets_to_run)}\n"
                f"Domain:             {config.domain}\n"
                f"Max steps:          {config.max_steps}\n"
                f"Toolkit mode:       {config.toolkit.mode}\n"
                f"Main model:         {config.model.litellm_model}\n"
                f"Timestamp:          {timestamp}\n"
                f"Run directory:      {run_dir}\n"
                f"Parallelism:        nproc={config.parallel.nproc}, nthread={config.parallel.nthread}\n"
                f"Verbosity (per-ds): {config.get_run_verbosity(num_datasets=len(datasets_to_run))}"
            ),
        )
    )

    config_json_str: str = json.dumps(config.model_dump(mode="json"), indent=4, default=str)
    logger.info(format_log_block(title="EXPERIMENT CONFIG", body=config_json_str))

    run_verbosity: int = config.get_run_verbosity(num_datasets=len(datasets_to_run))
    if run_verbosity != config.verbosity:
        logger.info(
            f"Per-dataset verbosity reduced from {config.verbosity} to {run_verbosity} for multi-dataset run."
        )

    if config.parallel.total_concurrency > 1 and config.parallel.max_rpm > 0:
        logger.info(
            f"Main LLM rate limit: {config.parallel.max_rpm} RPM total / "
            f"{config.parallel.total_concurrency} concurrent dataset workers "
            f"= {config.parallel.per_llm_rpm} RPM per worker"
        )
    if config.parallel.total_concurrency > 1 and config.parallel.per_code_gen_llm_rpm > 0:
        logger.info(
            f"Code-gen LLM rate limit: {config.parallel.resolved_code_gen_max_rpm} RPM total / "
            f"{config.parallel.total_concurrency} concurrent dataset workers "
            f"= {config.parallel.per_code_gen_llm_rpm} RPM per worker"
        )

    results: List[Dict[str, Any]] = _run_datasets_parallel(
        datasets=datasets_to_run,
        config=config,
        run_dir=run_dir,
        run_verbosity=run_verbosity,
    )

    for cli_result in results:
        dataset_idx_val: int = cli_result["dataset_idx"]
        result_status: str = cli_result["status"]
        if config.verbosity >= 1:
            print(f"\n[Dataset {dataset_idx_val}] Status: {result_status}")
            print(f"  True: {cli_result['dataset_label']}")
            if result_status == RunStatus.error.value:
                print("  Run-best family: N/A")
                print("  Run-best AIC: N/A")
            else:
                print(f"  Run-best family: {cli_result['run_best_model_structure']}")
                run_best_aic_val: Optional[float] = cli_result["run_best_model_aic"]
                if run_best_aic_val is None:
                    print("  Run-best AIC: N/A")
                else:
                    print(f"  Run-best AIC: {run_best_aic_val:.1f}")
            print(f"  Steps: {len(cli_result['steps'])}")

    return results


if __name__ == "__main__":
    config: ExperimentConfig = ExperimentConfig(_cli_parse_args=True)
    _configure_logging(verbosity=config.verbosity)
    run_all(config=config)
