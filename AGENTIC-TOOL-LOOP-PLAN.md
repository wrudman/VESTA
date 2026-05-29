# Agentic Tool Loop: Implementation Plan

## Table of Contents

1. [Codebase Analysis](#1-codebase-analysis)
2. [Problem Statement](#2-problem-statement)
3. [Desired Behavior](#3-desired-behavior)
4. [Current Architecture (What to Change)](#4-current-architecture-what-to-change)
5. [Implementation Details](#5-implementation-details)
6. [Testing Strategy](#6-testing-strategy)
7. [Migration Checklist](#7-migration-checklist)

---

## 1. Codebase Analysis

### 1.1 Overall Architecture

The `pymc_model_selection` codebase implements a VLM-guided iterative model fitting pipeline. A Vision Language Model (VLM) views a data visualization (histogram or time-series plot), proposes statistical models as PyMC code, and receives diagnostic feedback to refine those models over multiple steps. The pipeline supports two domains — distribution fitting (1-D data → distribution families like gaussian, student_t, lognormal, mixtures) and time-series fitting (temporal data → Gaussian Process kernels like linear, periodic, RBF, Matern). The architecture is domain-agnostic: `experiments.py` contains the main `run()` function which orchestrates the pipeline, and all domain-specific behavior is delegated to three Morphic Registry hierarchies: `DomainPrompts` (prompt rendering and VLM response parsing), `DomainToolkit` (diagnostic tool dispatch), and `DomainPlotting` (visualization and fit-state extraction).

### 1.2 The `run()` Function in `experiments.py`

The `run()` function (lines 153–758) is the canonical experiment runner. It takes a domain string, a dataset dict, model config, and pipeline parameters (max_steps, toolkit_mode, force_tool_call, etc.). It resolves domain registries via `DomainPrompts.of(domain)`, `DomainToolkit.of(domain)`, and `DomainPlotting.of(domain)`, then executes two phases: **Step 0** (initial prediction: VLM sees histogram → proposes 5 models → code-gen → fit → select best by AIC) and **Steps 1..max_steps** (feedback loop: VLM sees fit overlay → proposes 1 refined model → code-gen → fit → execute diagnostic tool → generate next plot). The function tracks state across steps via mutable variables: `family_name`, `current_model`, `map_estimate`, `metrics`, `fit_state`, `best_aic`, `tested_families`, `tested_models`, `selected_tool_history`, and `step_feedbacks`.

### 1.3 The VLM Backend Layer (`vlm_backends/slowburn_api.py`)

`SlowBurnAPIBackend` is a pure transport layer that wraps SlowBurn (which wraps LiteLLM). Its `call()` method accepts `prompt`, `images`, `tools`, `response_type`, and `verbosity`. When `response_type` is provided, it builds a SlowBurn `validator` closure that parses JSON from the VLM's text response and constructs a Typed instance via Pydantic coercion. When `tools` is provided, it passes them to `self._llm.call_llm(tools=tools, tool_choice="auto")`. Critically, this layer has **zero domain knowledge** — it does not inspect response keys, does not know about proposals or tool calls, and delegates all schema enforcement to the `response_type` Typed class's Pydantic validators. The SlowBurn library handles retries automatically when the validator raises `ValueError`.

### 1.4 The Current Tool Selection Mechanism (The Core Problem)

The current codebase has **two redundant, conflicting mechanisms** for tool selection. First, the **text-prompt mechanism**: each domain's feedback prompt (e.g., `FIT_FEEDBACK_PROMPT` in distribution_fitting/prompts.py) includes a `"toolkit"` field in the expected JSON schema (line 317: `"toolkit": "name_of_toolkit or None"`) and a TOOLKIT OPTIONS section listing available tools in the prompt text. The VLM returns `"toolkit": "calculate_moments"` in its JSON body alongside `"proposals"`. Second, the **native function-calling mechanism**: `experiments.py` passes `tools=resolved_tools_for_vlm` to `backend.call()` (line 463–466), which sets `tool_choice="auto"` in the SlowBurn call. When the VLM decides to use native function calling, it returns a `tool_calls` array with `content: null`, and the proposals JSON disappears entirely because OpenAI's Chat Completions API treats `content` and `tool_calls` as alternatives (content is optional when tool_calls is present). This dual mechanism is the root cause of the "tool-call-only response" bug where the VLM returns no proposals.

### 1.5 How Tool Results Currently Flow

In the current pipeline, tool results flow **forward to the next step**, not to the current step. Within a single feedback step, the flow is: (1) VLM receives the previous step's diagnostic plot + feedback context → (2) VLM returns proposals + toolkit selection simultaneously in one JSON response → (3) pipeline fits the proposed models, selects best by AIC → (4) pipeline executes the selected toolkit → (5) toolkit generates a diagnostic plot for the NEXT step. This means the VLM's model proposals are made **without seeing the diagnostic tool's output**. The tool selection and proposal are simultaneous decisions about different time horizons: proposals act on the current visualization, while the tool selection determines what the VLM will see next. Step 0 proposals are always "blind" — no diagnostic tool has been run yet.

### 1.6 The Prompt Templates

Each domain has four prompt constants: a predict prompt (Step 0, multi-proposal), a code-gen prompt (per-proposal PyMC code generation), a feedback prompt template (Steps 1+, single refined proposal), and summary prompts (initial + per-step). The feedback prompt template is the one that changes most in this refactoring. For distribution fitting, `FIT_FEEDBACK_PROMPT` (lines 244–363 of distribution_fitting/prompts.py) includes inline TOOLKIT OPTIONS listing all 5 tools by name and description, a `"toolkit"` field in the JSON schema, and anti-patterns. The `_build_step_feedback_prompt` function in `experiments.py` (lines 63–138) appends additional toolkit sections (`_TOOL_SECTION_TEMPLATE`, `_GENERATE_TOOL_SECTION`, `_GENERATE_ONLY_SECTION`, `_FORCE_TOOL_CALL_SECTION`) from `dynamic_toolkit.py` based on the `toolkit_mode`. All of these text-based tool instructions become redundant when tools are passed via the native `tools=` parameter.

### 1.7 The Tool System (Registry-Based)

The tool system uses a two-level Registry pattern. `Tool` (in `domains/__init__.py`) is a `Typed + ABC` base class that defines the interface: ClassVars for schema (`tool_description`, `output_type`, `parameters_schema`), a `to_openai_schema()` classmethod that auto-generates OpenAI function-calling JSON, and an abstract `execute()` method. Each domain creates its own Registry subclass: `DistributionFittingTool(Tool, Registry, ABC)` with 5 concrete tools (QQPlot, CalculateMoments, SegmentDistributionsAndCalculateMoments, PlotTailsTransform, ProbabilityPlot), and `TimeSeriesStaticTool(Tool, Registry, ABC)` with 5 tools (GetDominantPeriod, FitVsActuals, FitVsActualsWithResidualsDistribution, ResidualsAutoCorrelationPlot, ResidualsAutoCorrelationScore). `DomainToolkit.execute_tool()` dispatches via `ToolRegistry.of(selected_tool).execute(...)`. There is also `DynamicTool(Tool, Registry)` for LLM-generated tools.

### 1.8 The Response Models (Typed Classes)

`DistFittingVLMResponse` (distribution_fitting/prompts.py, lines 460–488) and `TimeSeriesVLMResponse` (time_series/prompts.py, lines 342–365) are Morphic Typed classes that the SlowBurn validator parses VLM responses into. Currently both have: `description: str = ""`, `toolkit: Optional[str] = None`, `tool_calls: Optional[List[Dict]] = None`, and `proposals: Optional[Dict[str, Proposal]] = None`. The `toolkit` and `tool_calls` fields were added as workarounds for the dual-mechanism problem — `toolkit` captures the text-prompt tool selection, `tool_calls` captures native function calling. Both `proposals` and `description` were made optional to handle tool-call-only responses. This is backwards: proposals should be mandatory (every step must propose models), and `toolkit`/`tool_calls` should not exist on the proposal response model at all.

### 1.9 The `extract_tool_from_response` Method

Both `DistributionFittingPrompts` and `TimeSeriesPrompts` implement `extract_tool_from_response()` which reads `"selected_tool"` or `"toolkit"` from the proposal dict (the model_dump of the VLM response). This method is a remnant of the text-prompt tool selection mechanism. It returns `(selected_tool, selected_tool_args)` where `selected_tool_args` is always `{}` because the text-prompt mechanism has no way to pass arguments (unlike native function calling which includes structured `arguments`). With the agentic loop, tool selection comes from `response.tool_calls[0].function.name` and arguments from `response.tool_calls[0].function.arguments`, so `extract_tool_from_response` is deleted.

### 1.10 LLM Backend Compatibility Constraints

Research during the previous chat established that different LLM backends handle native tool calling differently. **Qwen 3.5 and GPT-5** (via Chat Completions API) return `content: null` when `tool_calls` is present — they cannot produce both text content and tool calls in a single response. **GPT 5.4** can produce "preambles" (brief text before tool calls) via the new Responses API, but not via Chat Completions. **Claude models** (via Anthropic API) return a `content` array where text blocks and `tool_use` blocks coexist — they CAN produce both. Since the pipeline must support all three model families, the architecture cannot rely on getting both proposals and tool calls in a single VLM response. This is the fundamental constraint that drives the two-phase agentic loop design: Phase 1 (diagnostic tool calls, where `content` may be null) is separated from Phase 2 (model proposals, where no tools are offered so the VLM must produce `content`).

---

## 2. Problem Statement

### 2.1 The Dual-Mechanism Conflict

The pipeline currently passes diagnostic tools to the VLM via **two redundant channels simultaneously**:

| Channel | Where it lives | What the VLM sees | How the VLM responds |
|---|---|---|---|
| **Native function-calling** | `backend.call(tools=tools_for_vlm)` in `experiments.py` line 463 | `tools` parameter in the API request | Returns `tool_calls` array; `content` is often `null` |
| **Text-prompt instructions** | `FIT_FEEDBACK_PROMPT` lines 301–308 + `_TOOL_SECTION_TEMPLATE` appended by `_build_step_feedback_prompt` | Tool names and descriptions embedded in the prompt text, plus a `"toolkit"` field in the expected JSON schema | Returns `"toolkit": "calculate_moments"` inside the JSON body alongside `"proposals"` |

When both are active, the VLM sees the same tools described twice. It then picks ONE way to respond (usually native `tool_calls` since the API biases toward it). When it picks native function calling, `content` becomes `null`, the proposals JSON disappears, and the pipeline gets a response with no model proposals.

### 2.2 The `content: null` Problem Across Model Families

The OpenAI Chat Completions API spec says `content` is *"Required unless `tool_calls` is specified"*. In practice:

- **Qwen 3.5** (via Together AI / vLLM): follows OpenAI convention. `content` is `null` when `tool_calls` present. Has known issues with dumping tool-call JSON into `content` instead of `tool_calls`.
- **GPT-5 / GPT-5-mini** (via Azure, Chat Completions API): `content` is almost always `null` when `tool_calls` present.
- **GPT 5.4** (via Responses API): can produce "preambles" (brief text before tool calls). But this requires the new Responses API, not Chat Completions. Through Chat Completions, same `content: null` behavior.
- **Claude 4.5 Haiku / 4.6 Sonnet** (via Bedrock): uses a `content` array where `text` blocks and `tool_use` blocks coexist as siblings. CAN return both. However, even Claude is unreliable for **parallel** tool calls (multiple tools in one response).

**Since the pipeline must support Qwen, GPT, and Claude, it cannot rely on getting both content and tool_calls in a single response.** This is the hard constraint.

### 2.3 The Consequence: Lost Proposals

When `force_tool_call` is active (or the VLM spontaneously decides to call a tool), the current pipeline:
1. Passes `tools=tools_for_vlm` to `backend.call()` with `response_type=DistFittingVLMResponse`
2. The VLM returns `tool_calls` with `content: null`
3. SlowBurn's validator receives `{"tool_calls": [...]}` (no proposals, no description)
4. `DistFittingVLMResponse` is constructed with `proposals=None`, `description=""`
5. The pipeline hits the `proposals is None` guard (lines 506–533) and **reuses the previous step's model** — a silent fallback that wastes an entire pipeline step

This is a critical bug: each step MUST propose fresh models. Reusing the previous best model is never correct behavior.

### 2.4 What Needs to Change (Summary)

The fix requires **separating tool calling from model proposal** into two distinct VLM turns within each pipeline step:
- **Phase 1 (Diagnostic):** VLM is offered tools via native `tools=` parameter. It either calls a tool or declines. If it calls a tool, the result is fed back into the conversation context.
- **Phase 2 (Proposal):** VLM is called with `tools=None`. It MUST produce a JSON response with proposals. No escape hatch.

Additionally:
- Remove the text-prompt tool selection mechanism (`"toolkit"` JSON field, `TOOLKIT OPTIONS` section in prompts, `_TOOL_SECTION_TEMPLATE`, `_FORCE_TOOL_CALL_SECTION`)
- Remove `extract_tool_from_response()` from both domain prompts classes
- Remove `toolkit` and `tool_calls` fields from the Typed response models
- Make `proposals` mandatory again on the response models
- Add a new `backend.call_for_tool()` method (or mode) that returns raw tool_calls instead of parsing through the Typed validator
- Add a `max_tool_calls_per_step` parameter to control how many diagnostic tools the VLM can chain before being forced to propose

---

## 3. Desired Behavior

### 3.1 The New Per-Step Flow

The previous flow was: `[Proposal + ToolSelection] → [Proposal + ToolSelection] → ...`

The new flow is: `[ToolCall(s) → Proposal] → [ToolCall(s) → Proposal] → ...`

Within each step of the feedback loop (steps 1 through max_steps), the pipeline executes:

```
STEP N:
  Context: previous step's best model code + AIC + fit overlay plot

  ── Phase 1: Diagnostic (0 to max_tool_calls_per_step tool calls) ──
  available_tools = {all domain tools}   # e.g. qq_plot, calculate_moments, ...
  tool_results = []                      # accumulates tool outputs for Phase 2 context
  conversation_history = [system_context, image, feedback_context]

  for i in 1, 2, ..., max_tool_calls_per_step:
      tool_choice = "required" if (i == 1 AND force_tool_call) else "auto"

      response = backend.call_for_tool(
          messages=conversation_history,
          tools=available_tools,
          tool_choice=tool_choice,
      )

      if response has NO tool_calls:
          break   # VLM is satisfied, move to Phase 2

      tool_call = response.tool_calls[0]    # one tool per turn
      tool_instance = DomainTool.of(tool_call.function.name)
      plot_desc, output_type, summary = tool_instance.execute(
          data=data, fit_state=fit_state, best_idx=best_idx,
          fit_path=..., selected_tool_args=json.loads(tool_call.function.arguments),
      )

      # Append tool call + result to conversation history (OpenAI multi-turn format)
      conversation_history.append(assistant_message_with_tool_call)
      conversation_history.append(tool_result_message(tool_call.id, plot_desc + summary))
      tool_results.append({name, plot_desc, output_type, summary, image_path})

      # Remove used tool so VLM can't call it again
      available_tools = [t for t in available_tools if t.function.name != tool_call.function.name]

  ── Phase 2: Proposal (exactly 1 VLM call, no tools offered) ──
  proposal_response = backend.call(
      prompt=proposal_prompt,        # feedback prompt WITHOUT toolkit sections
      images=[fit_path] + tool_result_images,
      tools=None,                    # forces VLM to produce content
      response_type=VLMResponseType,
  )

  # proposal_response.proposals is MANDATORY (non-Optional)
  # Proceed with code-gen → fit → select best → update state
```

### 3.2 Step 0 Also Gets the Diagnostic Phase

Step 0 currently has no diagnostic phase — the VLM sees only the raw histogram and proposes 5 models blind. With the agentic loop, Step 0 also gets Phase 1:

```
STEP 0:
  ── Phase 1: Diagnostic (optional) ──
  VLM sees histogram + available tools
  If it calls calculate_moments or segment_distributions, those results
  are added to context before proposal

  ── Phase 2: Proposal ──
  VLM proposes 5 models (informed by any diagnostic results)
  Code-gen → fit → select best by AIC
```

This means even the first proposal is data-driven rather than blind.

### 3.3 New Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_tool_calls_per_step` | `int` | `1` | Max diagnostic tool calls per step before forcing proposal. With N=1, each step has at most 2 VLM calls (1 tool + 1 proposal). With N=3, the VLM can chain up to 3 diagnostics. |
| `force_tool_call` | `bool` | `False` | (existing) When True, the first diagnostic turn uses `tool_choice="required"`. Subsequent turns use `tool_choice="auto"`. |
| `toolkit_mode` | `str` | (existing) | `"none"` skips Phase 1 entirely. `"static"`, `"generate_only"`, `"dynamic"` behave as before for which tools are available. |

### 3.4 What the VLM Sees in Each Phase

**Phase 1 (Diagnostic Turn):**
- System/user message with: the current fit overlay image, the current best model code, AIC, tested families history, and a short instruction like "You are evaluating a model fit. Decide if you need a diagnostic tool before proposing a revision. If satisfied, respond with no tool call."
- `tools=[...]` — the domain's static tools in OpenAI schema format
- `tool_choice="required"` or `"auto"` depending on `force_tool_call` and iteration number

**Phase 2 (Proposal Turn):**
- The full feedback prompt (same as current `FIT_FEEDBACK_PROMPT` but with the TOOLKIT OPTIONS section and `"toolkit"` field REMOVED)
- Images: the fit overlay + any diagnostic images from Phase 1
- `tools=None` — no tools offered, VLM must produce JSON content
- `response_type=DistFittingVLMResponse` (with mandatory `proposals`)

### 3.5 Handling `generate_new_tool` in the Agentic Loop

When `toolkit_mode` is `"dynamic"` or `"generate_only"`, `generate_new_tool` is included in the `available_tools` list for Phase 1. If the VLM calls `generate_new_tool`, the pipeline:
1. Extracts `tool_description` from the tool call arguments
2. Calls `handle_generate_new_tool()` (existing function in `dynamic_toolkit.py`)
3. The generated tool code runs in the sandbox, produces output
4. The output is appended to the conversation history as a tool result
5. The generated tool is registered in the `DynamicTool` Registry for future steps

This works identically to today, except the tool call comes from native `tool_calls` instead of the JSON body.

---

## 4. Current Architecture — What to Change (File by File)

### 4.1 `vlm_backends/base.py` — Add `call_for_tool()` Abstract Method

**Current:** Only has `call()` which returns `Union[str, Typed]`. When `response_type` is set, it parses the VLM's text content into a Typed model. There is no way to get raw `tool_calls` from the response.

**Change:** Add a new abstract method `call_for_tool()` that:
- Accepts `messages` (list of chat messages, not a single prompt string), `tools`, `tool_choice`, `images`
- Returns a structured result containing both `content` (Optional[str]) and `tool_calls` (Optional[List[Dict]])
- Does NOT use a validator or `response_type` — this is a raw LLM call
- The return type should be a simple Typed class, e.g. `ToolCallResponse(content=..., tool_calls=...)`

**Why a new method instead of modifying `call()`:** The existing `call()` contract is "prompt in → validated Typed out". The diagnostic phase needs "messages in → raw tool_calls out". These are fundamentally different operations. Mixing them into one method with mode flags would violate single responsibility.

### 4.2 `vlm_backends/slowburn_api.py` — Implement `call_for_tool()`

**Current:** `call()` (lines 188–281) builds `call_kwargs` with `prompt`, `images`, `tools`, `tool_choice`, `validator`, and calls `self._llm.call_llm(**call_kwargs).result()`. When a validator is present, SlowBurn parses the response text. When tools are present, SlowBurn sends them but the response handling still goes through the text validator.

**Change:** Implement `call_for_tool()` which:
1. Calls `self._llm.call_llm(...)` with `return_messages=True` (so SlowBurn returns the full message list instead of just the text content)
2. Extracts `tool_calls` from the assistant message in the returned message list
3. Also extracts `content` if present (for Claude models that return both)
4. Returns a `ToolCallResponse` Typed instance

**Key SlowBurn detail:** When `return_messages=True`, SlowBurn returns the full list of messages (including the assistant's response message). The assistant message has `.tool_calls` (a list of tool call objects) and `.content` (the text, or None). The `call_for_tool()` method must parse these from the raw message list.

**No validator needed:** Phase 1 does not need JSON parsing or Typed validation. The VLM either returns tool_calls or it doesn't. The pipeline reads `tool_calls[0].function.name` and `tool_calls[0].function.arguments` directly.

### 4.3 `experiments.py` — Restructure the Feedback Loop

**Current feedback loop** (lines 430–737):
```
for step_num in range(1, max_steps + 1):
    tools_for_vlm = build_tools_list(...)
    feedback_prompt = _build_step_feedback_prompt(...)   # includes toolkit sections
    vlm_response = backend.call(prompt, images, tools, response_type)  # ONE call
    proposal_dict = vlm_response.model_dump()
    selected_tool, args = prompts_reg.extract_tool_from_response(proposal_dict)
    # ... code-gen, fit, execute tool, summary ...
```

**New feedback loop:**
```
for step_num in range(1, max_steps + 1):
    tools_for_vlm = build_tools_list(...)

    # ── Phase 1: Diagnostic ──────────────────────────────────
    tool_results = []
    if toolkit_mode != "none" and len(tools_for_vlm) > 0:
        tool_results = _run_diagnostic_phase(
            backend=backend,
            tools_for_vlm=tools_for_vlm,
            images=[fit_path],
            diagnostic_context=...,     # current model, AIC, history
            max_tool_calls=max_tool_calls_per_step,
            force_first_call=force_tool_call,
            toolkit_reg=toolkit_reg,
            data=data, fit_state=fit_state, best_idx=best_idx,
        )

    # ── Phase 2: Proposal ────────────────────────────────────
    feedback_prompt = _build_step_feedback_prompt(...)   # NO toolkit sections
    extra_images = [tr["image_path"] for tr in tool_results if tr.get("image_path")]
    vlm_response = backend.call(
        prompt=feedback_prompt,
        images=[fit_path] + extra_images,
        tools=None,                          # NO tools — forces content
        response_type=response_type,
    )
    proposal_dict = vlm_response.model_dump()
    # proposals is now MANDATORY — no None guard needed
    # ... code-gen, fit, summary (no tool execution here) ...
```

**New helper function `_run_diagnostic_phase()`:** This encapsulates the Phase 1 loop. It returns a list of tool result dicts, each containing `name`, `plot_description`, `output_type`, `summary`, `image_path`. This function:
- Builds the diagnostic conversation history (system message + image + context)
- Loops up to `max_tool_calls_per_step` times
- Each iteration calls `backend.call_for_tool(messages, tools, tool_choice)`
- If tool_calls present: execute tool, append result to history, remove tool from available list
- If no tool_calls: break
- Returns accumulated tool results

### 4.4 `experiments.py` — `_build_step_feedback_prompt()` Simplification

**Current** (lines 63–138): Appends toolkit sections (`_TOOL_SECTION_TEMPLATE`, `_GENERATE_TOOL_SECTION`, `_GENERATE_ONLY_SECTION`, `_FORCE_TOOL_CALL_SECTION`) based on `toolkit_mode` and `force_tool_call`.

**Change:** Remove ALL toolkit section appending. The function should no longer accept `toolkit_mode`, `tools_for_vlm`, or `force_tool_call` parameters. It formats only the base feedback prompt template. The toolkit-related context (tool results) is provided via images and a brief "Diagnostic Results" section added to the prompt text describing what tools were called and what they found.

### 4.5 `domains/distribution_fitting/prompts.py` — Prompt and Response Model Changes

**FIT_FEEDBACK_PROMPT changes:**
- Remove lines 301–308 (the `TOOLKIT OPTIONS` section listing tool names and descriptions)
- Remove `"toolkit": "name_of_toolkit or None"` from the JSON schema example (line 317)
- Remove `"toolkit": "None"` from the MIXTURE EXAMPLE (line 336)
- Add a new `DIAGNOSTIC CONTEXT` placeholder where Phase 1 tool results are inserted:
  ```
  DIAGNOSTIC RESULTS (from this step's analysis):
  {diagnostic_results}
  ```
  When no tools were called, this says "No diagnostic tools were used this step."

**DistFittingVLMResponse changes:**
- Remove `toolkit: Optional[str] = None`
- Remove `tool_calls: Optional[List[Dict[str, Any]]] = None`
- Change `proposals` back to mandatory: `proposals: Dict[str, DistFittingProposal]`
- Change `description` back to mandatory: `description: str`
- Restore the `proposals_must_be_nonempty` validator to reject empty dicts
- Add a `description_must_not_be_empty` validator

**Delete `extract_tool_from_response()`** — tool selection now comes from Phase 1's native tool_calls, not the proposal JSON body.

### 4.6 `domains/time_series/prompts.py` — Same Changes as Distribution Fitting

Mirror all changes from 4.5:
- Remove TOOLKIT section from `TS_FEEDBACK_PROMPT` (lines 176–183)
- Remove `"toolkit"` from JSON examples
- Update `TimeSeriesVLMResponse` (remove toolkit, tool_calls; make proposals mandatory)
- Delete `extract_tool_from_response()`

### 4.7 `domains/__init__.py` — Remove `extract_tool_from_response` from ABC

**Current:** `DomainPrompts` has an abstract method `extract_tool_from_response()` (lines 307–312).

**Change:** Delete this abstract method. It no longer exists in the interface. Tool extraction is handled by `_run_diagnostic_phase()` in `experiments.py` which reads from native `tool_calls`.

### 4.8 `dynamic_toolkit.py` — Remove Text-Prompt Tool Section Templates

**Current:** Exports `_TOOL_SECTION_TEMPLATE`, `_GENERATE_TOOL_SECTION`, `_GENERATE_ONLY_SECTION`, `_FORCE_TOOL_CALL_SECTION` (lines 858–892).

**Change:** Delete all four template constants. They are no longer imported or used. The imports in `experiments.py` (line 41–44) should be updated to remove them.

### 4.9 `experiments.py` — Step 0 Gets Diagnostic Phase

**Current Step 0** (lines 253–419): VLM sees histogram → proposes 5 models → code-gen → fit → summary. No diagnostic phase.

**Change:** Insert Phase 1 before the proposal call at Step 0:
```
# Step 0 Phase 1: optional diagnostic
if toolkit_mode != "none":
    step0_tool_results = _run_diagnostic_phase(...)

# Step 0 Phase 2: proposal (existing code, but with tool results in context)
vlm_response = backend.call(prompt=dist_proposal_prompt_with_diagnostics, ...)
```

### 4.10 `run()` Signature Change

Add `max_tool_calls_per_step: int = 1` to the `run()` function signature and the CLI parser. Wire it through to `_run_diagnostic_phase()`.

---

## 5. Implementation Details

### 5.1 The `ToolCallResponse` Typed Class

Create in `vlm_backends/base.py`:

```python
class ToolCallResult(Typed):
    """One tool call extracted from a VLM response."""
    id: str
    name: str
    arguments: Dict[str, Any]

class ToolCallResponse(Typed):
    """Structured response from a VLM call that may contain tool calls."""
    content: Optional[str] = None
    tool_calls: List[ToolCallResult] = []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
```

### 5.2 The `call_for_tool()` Method on SlowBurnAPIBackend

```python
@validate
def call_for_tool(
    self,
    *,
    prompt: str,
    images: Optional[List[Union[str, Path]]] = None,
    tools: List[Dict[str, Any]],
    tool_choice: str,       # "auto" or "required"
    verbosity: int,
) -> ToolCallResponse:
    """Send a tool-calling request and return raw tool_calls + content.

    Unlike call(), this method does NOT use a validator or response_type.
    It returns the raw VLM response decomposed into content and tool_calls.
    Used for Phase 1 (diagnostic) of the agentic tool loop.
    """
    # Build call_kwargs with return_messages=True
    # Call self._llm.call_llm(...).result(timeout=self.call_timeout)
    # Parse the returned message list to extract tool_calls and content
    # Return ToolCallResponse(content=..., tool_calls=[...])
```

**Critical implementation detail:** SlowBurn's `call_llm()` with `return_messages=True` returns a list of message dicts. The assistant message (last in list with `role="assistant"`) contains `tool_calls` (list of dicts with `id`, `type`, `function.name`, `function.arguments`) and `content` (str or None). The `call_for_tool()` method must:
1. Find the assistant message in the returned list
2. Extract `tool_calls` if present, parsing `function.arguments` from JSON string to dict
3. Extract `content` if present
4. Wrap in `ToolCallResponse`

### 5.3 The `_run_diagnostic_phase()` Helper

This is a new function in `experiments.py`:

```python
def _run_diagnostic_phase(
    *,
    backend: VLMBackend,
    diagnostic_prompt: str,      # brief context for the VLM
    images: List[str],           # fit overlay + histogram
    tools_for_vlm: List[Dict[str, Any]],
    max_tool_calls: int,
    force_first_call: bool,
    toolkit_reg: DomainToolkit,
    data: Any,
    fit_state: Dict[str, Any],
    best_idx: Any,
    out_dir: str,
    step_num: int,
    verbosity: int,
) -> List[Dict[str, Any]]:
    """Run Phase 1 of the agentic tool loop: diagnostic tool calls.

    Returns a list of tool result dicts, each with keys:
        - "name": str (tool name)
        - "plot_description": str
        - "output_type": str ("visualization" or "numeric")
        - "summary": str
        - "image_path": Optional[str] (path to diagnostic plot, if any)
    """
    available_tools: List[Dict[str, Any]] = list(tools_for_vlm)
    tool_results: List[Dict[str, Any]] = []

    for call_idx in range(max_tool_calls):
        tool_choice: str = (
            "required" if (call_idx == 0 and force_first_call) else "auto"
        )

        if len(available_tools) == 0:
            break  # no tools left to offer

        response: ToolCallResponse = backend.call_for_tool(
            prompt=diagnostic_prompt,
            images=images,
            tools=available_tools,
            tool_choice=tool_choice,
            verbosity=verbosity,
        )

        if not response.has_tool_calls:
            break  # VLM declined to call a tool

        tool_call: ToolCallResult = response.tool_calls[0]

        # Execute the tool
        tool_fit_path: str = os.path.join(
            out_dir, f"diagnostic_{step_num}_{call_idx}_{tool_call.name}.png"
        )
        plot_desc, output_type, summary = toolkit_reg.execute_tool(
            selected_tool=tool_call.name,
            selected_tool_args=tool_call.arguments,
            data=data,
            fit_state=fit_state,
            best_idx=best_idx,
            fit_path=tool_fit_path,
            plot_type_descriptions=...,
        )

        tool_results.append({
            "name": tool_call.name,
            "plot_description": plot_desc,
            "output_type": output_type,
            "summary": summary,
            "image_path": tool_fit_path if output_type == "visualization" else None,
        })

        # Remove used tool from available list
        available_tools = [
            t for t in available_tools
            if t["function"]["name"] != tool_call.name
        ]

        # Add tool result images to the images list for subsequent turns
        if output_type == "visualization":
            images = images + [tool_fit_path]

    return tool_results
```

### 5.4 The Diagnostic Prompt (Phase 1)

Phase 1 needs a short prompt that gives the VLM context about the current fit and asks it to decide whether it needs a diagnostic tool. This is NOT the full feedback prompt — it's a concise instruction:

```python
DIAGNOSTIC_PHASE_PROMPT: str = """\
You are evaluating a statistical model fit. The current best model is shown in the attached plot.

Current model: {current_model_summary}
Current AIC: {current_aic}
Step: {step_num} of {max_steps}
Previously tested: {tested_families}

If you need diagnostic information before proposing a revised model, call one of the available \
diagnostic tools. If you have enough information to propose a model directly, respond without \
calling any tool."""
```

This prompt is domain-agnostic. It does NOT contain tool descriptions (those come from the `tools=` parameter natively).

### 5.5 Integrating Diagnostic Results into Phase 2

After Phase 1 completes, the tool results need to be injected into the Phase 2 feedback prompt. Add a `{diagnostic_results}` placeholder to both `FIT_FEEDBACK_PROMPT` and `TS_FEEDBACK_PROMPT`. The `_build_step_feedback_prompt()` function formats this:

```python
if len(tool_results) == 0:
    diagnostic_text = "No diagnostic tools were used this step."
else:
    parts: List[str] = []
    for tr in tool_results:
        if tr["output_type"] == "numeric":
            parts.append(f"Tool '{tr['name']}': {tr['summary']}")
        else:
            parts.append(f"Tool '{tr['name']}': [see attached diagnostic image]")
    diagnostic_text = "\n".join(parts)
```

The diagnostic images are passed as additional `images` in the Phase 2 `backend.call()`.

### 5.6 Changes to `DomainToolkit.execute_tool()`

**Current signature** includes `backend` and `code_gen_fn` parameters (for `generate_new_tool` and dynamic toolkit). These are only used when `selected_tool == "generate_new_tool"`.

**Change:** The `_run_diagnostic_phase()` function handles `generate_new_tool` as a special case BEFORE calling `toolkit_reg.execute_tool()`. When the VLM calls `generate_new_tool`, `_run_diagnostic_phase()` calls `handle_generate_new_tool()` directly (it already exists in `dynamic_toolkit.py`). The `execute_tool()` method on `DomainToolkit` only handles domain-specific static tools. This simplifies `execute_tool()` — remove `backend` and `code_gen_fn` from its signature.

**Updated `DomainToolkit.execute_tool()` signature (on the ABC):**
```python
@abstractmethod
def execute_tool(
    self,
    *,
    selected_tool: str,              # tool name from tool_calls
    selected_tool_args: Dict[str, Any],  # parsed arguments from tool_calls
    data: Any,
    fit_state: Dict[str, Any],
    best_idx: Any,
    fit_path: str,
    plot_type_descriptions: Dict[str, str],
) -> Tuple[str, str, str]:
    ...
```

Note: `backend` and `code_gen_fn` are REMOVED from this signature. They were only needed for `generate_new_tool` which is now handled in `_run_diagnostic_phase()`.

---

## 6. Testing Strategy

### 6.1 Unit Tests to Update

**Existing tests that will break and need updating:**

1. **Tests for `DistFittingVLMResponse` / `TimeSeriesVLMResponse`:** These test that `proposals=None` is accepted (tool-call-only response). After the change, `proposals` is mandatory. Update tests to verify that missing proposals raises `ValidationError`.

2. **Tests for `extract_tool_from_response`:** Delete these tests entirely — the method is deleted.

3. **Tests for `_build_step_feedback_prompt`:** Update to verify that no toolkit sections are appended. The function no longer accepts `toolkit_mode`, `tools_for_vlm`, or `force_tool_call`.

4. **Tests that construct mock VLM responses with `toolkit` field:** Remove `toolkit` from the mock data.

### 6.2 New Unit Tests to Write

1. **`test_call_for_tool_returns_tool_calls`:** Mock SlowBurn to return a message list with tool_calls. Verify `call_for_tool()` returns a `ToolCallResponse` with the correct tool name and arguments.

2. **`test_call_for_tool_no_tool_calls`:** Mock SlowBurn to return a message with content only (no tool_calls). Verify `call_for_tool()` returns `ToolCallResponse(content="...", tool_calls=[])`.

3. **`test_run_diagnostic_phase_single_tool`:** Mock backend.call_for_tool to return one tool call on first iteration, then no tool call on second. Verify that execute_tool is called once and the result list has one entry.

4. **`test_run_diagnostic_phase_force_first`:** Verify that when `force_first_call=True`, the first call uses `tool_choice="required"` and subsequent calls use `"auto"`.

5. **`test_run_diagnostic_phase_max_calls`:** Set `max_tool_calls=3`. Mock backend to always return a tool call. Verify exactly 3 tools are called (loop stops at max).

6. **`test_run_diagnostic_phase_tool_removal`:** Verify that after a tool is called, it's removed from available_tools for subsequent iterations.

7. **`test_run_diagnostic_phase_no_tools_mode_none`:** When `toolkit_mode="none"`, verify Phase 1 is skipped entirely.

8. **`test_proposals_mandatory_on_response`:** Verify that `DistFittingVLMResponse(description="test")` (no proposals) raises `ValidationError`.

### 6.3 E2E Test Commands

After implementation, run these commands to verify:

```bash
# Test 1: toolkit=none (Phase 1 skipped, should work as before)
cd /Users/adivekar/workplace/pymc_model_selection && \
$(conda info --base)/envs/pymc/bin/python experiments.py \
    --domain distribution-fitting \
    --model "azure/gpt-5-mini" \
    --data-pkl data_single.pkl \
    --dataset-idx 0 \
    --max-steps 2 \
    --toolkit none \
    --verbosity 2

# Test 2: toolkit=static, no force (Phase 1 with auto tool_choice)
$(conda info --base)/envs/pymc/bin/python experiments.py \
    --domain distribution-fitting \
    --model "azure/gpt-5-mini" \
    --data-pkl data_single.pkl \
    --dataset-idx 0 \
    --max-steps 2 \
    --toolkit static \
    --verbosity 2

# Test 3: toolkit=static with force (Phase 1 uses tool_choice=required)
$(conda info --base)/envs/pymc/bin/python experiments.py \
    --domain distribution-fitting \
    --model "azure/gpt-5-mini" \
    --data-pkl data_single.pkl \
    --dataset-idx 0 \
    --max-steps 2 \
    --toolkit static \
    --force-tool-call \
    --verbosity 2

# Test 4: max_tool_calls_per_step=3 (VLM can chain diagnostics)
$(conda info --base)/envs/pymc/bin/python experiments.py \
    --domain distribution-fitting \
    --model "azure/gpt-5-mini" \
    --data-pkl data_single.pkl \
    --dataset-idx 0 \
    --max-steps 2 \
    --toolkit static \
    --force-tool-call \
    --max-tool-calls-per-step 3 \
    --verbosity 2
```

**What to verify in the output:**
- Phase 1 diagnostic calls appear in the log BEFORE the proposal
- Diagnostic images are generated (check `stat_plots/diagnostic_*.png`)
- The proposal VLM call receives diagnostic images (logged in verbose mode)
- Every step produces fresh proposals (no "Reusing previous best model" warnings)
- The `"toolkit"` field does NOT appear in VLM responses
- AIC and model selection are reasonable

### 6.4 Run Existing Test Suite

```bash
cd /Users/adivekar/workplace/pymc_model_selection && \
$(conda info --base)/envs/pymc/bin/python -m pytest --tb=short -rf tests/
```

All 105 existing tests must pass (with updates from 6.1).

---

## 7. Migration Checklist

### Phase A: Backend Layer (no behavior change yet)
- [ ] Create `ToolCallResult` and `ToolCallResponse` Typed classes in `vlm_backends/base.py`
- [ ] Add `call_for_tool()` abstract method to `VLMBackend`
- [ ] Implement `call_for_tool()` in `SlowBurnAPIBackend`
- [ ] Write unit tests for `call_for_tool()` (mock SlowBurn)
- [ ] Verify all existing tests still pass

### Phase B: Response Model Cleanup
- [ ] Remove `toolkit`, `tool_calls` fields from `DistFittingVLMResponse`
- [ ] Remove `toolkit`, `tool_calls` fields from `TimeSeriesVLMResponse`
- [ ] Make `proposals` mandatory (non-Optional) on both response models
- [ ] Make `description` mandatory (non-Optional, remove default) on both
- [ ] Restore strict validators
- [ ] Update unit tests for response models
- [ ] Delete `extract_tool_from_response()` from both DomainPrompts implementations
- [ ] Delete `extract_tool_from_response()` from `DomainPrompts` ABC in `domains/__init__.py`

### Phase C: Prompt Changes
- [ ] Remove TOOLKIT OPTIONS section from `FIT_FEEDBACK_PROMPT`
- [ ] Remove `"toolkit"` from JSON schema examples in `FIT_FEEDBACK_PROMPT`
- [ ] Remove TOOLKIT section from `TS_FEEDBACK_PROMPT`
- [ ] Remove `"toolkit"` from JSON schema examples in `TS_FEEDBACK_PROMPT`
- [ ] Add `{diagnostic_results}` placeholder to both feedback prompts
- [ ] Create `DIAGNOSTIC_PHASE_PROMPT` constant (domain-agnostic)
- [ ] Delete `_TOOL_SECTION_TEMPLATE`, `_GENERATE_TOOL_SECTION`, `_GENERATE_ONLY_SECTION`, `_FORCE_TOOL_CALL_SECTION` from `dynamic_toolkit.py`
- [ ] Update imports in `experiments.py` to remove deleted template constants

### Phase D: Experiments.py Restructure
- [ ] Write `_run_diagnostic_phase()` helper function
- [ ] Simplify `_build_step_feedback_prompt()` — remove toolkit params, add diagnostic_results
- [ ] Restructure feedback loop (Steps 1..max_steps) into Phase 1 + Phase 2
- [ ] Add Phase 1 to Step 0
- [ ] Add `max_tool_calls_per_step` parameter to `run()` and CLI parser
- [ ] Remove the `proposals is None` reuse guard (proposals are now mandatory)
- [ ] Handle `generate_new_tool` calls in `_run_diagnostic_phase()`
- [ ] Remove `backend` and `code_gen_fn` from `DomainToolkit.execute_tool()` signature
- [ ] Update both `DomainToolkit` implementations to match new signature

### Phase E: Logging & Verbosity
- [ ] Log Phase 1 tool calls with separators (tool name, arguments, result summary)
- [ ] Log diagnostic images being passed to Phase 2
- [ ] Log full `model_dump()` of VLM responses at verbosity >= 2
- [ ] Log the generated PyMC code for each proposal at verbosity >= 2
- [ ] Log the feedback prompt text at verbosity >= 2

### Phase F: Testing
- [ ] All existing unit tests pass (with Phase B updates)
- [ ] New unit tests for `call_for_tool()`, `_run_diagnostic_phase()`, response models
- [ ] E2E test: toolkit=none
- [ ] E2E test: toolkit=static (auto tool_choice)
- [ ] E2E test: toolkit=static + force_tool_call
- [ ] E2E test: max_tool_calls_per_step=3
- [ ] E2E test: time-series domain with toolkit=static
