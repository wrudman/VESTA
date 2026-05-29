# Option B Implementation Plan: Time-Series Box Loop Baseline

## Goal

Make `baselines/box_loop` runnable as a production Time Series baseline comparable to the Time Series blocks in `runs_easy.sh`, `runs_medium.sh`, and `runs_hard.sh`, while keeping the existing per-dataset core loop (`run_box_loop_for_array`) essentially unchanged.

This plan implements the pragmatic Option B scope:

1. Fix environment variable naming and backend routing.
2. Fix output/resume persistence so the file type is consistent.
3. Default to 5 total Box Loop rounds.
4. Add Kimi K2.5, Claude Sonnet 4.6, and GPT 5.4 mini routing.
5. Add `--dataset-idx` chunking.
6. Add simple one-level `concurry` process parallelism over single dataset entries.

## Important clarification: what does `--rounds 5` mean?

In the current core loop, `num_rounds` is the total number of LLM calls per dataset, not “initial proposal plus N improvement rounds.” The loop is:

```python
for round_num in range(1, num_rounds + 1):
    if round_num == 1:
        user_msg = _build_round1_prompt_ts(env)
    else:
        user_msg = _build_followup_prompt_ts(...)
```

Therefore:

- `--rounds 5` means **5 total LLM calls**.
- That is **1 initial proposal round + 4 improvement rounds**.
- If we wanted **1 proposal + 5 improvement rounds**, the argument would need to be `--rounds 6`.

For equivalence with `--max-steps 5` in the main experiment scripts, this plan sets the Box Loop default to `5` total rounds.

## Current issues to fix

### 1. Accidental non-TS data slicing

`run.py::load_data` still contains the old debug slice:

```python
print("RUNNING ON A SAMPLE OF 1")
raw = raw[:1]
```

Even though the Time Series path uses `load_data_ts`, this must be removed so the baseline is not dangerous if the task is changed later.

### 2. CSV/pickle mismatch

`run_all_arrays` currently tries to resume from `save_path` as pickle:

```python
with open(save_path, "rb") as f:
    prev = pickle.load(f)
```

But it checkpoints only a CSV:

```python
path = save_path.split('.')[0] + '.csv'
df.to_csv(path)
```

This breaks resume. It also means `--output results.pkl` does not actually create `results.pkl`.

### 3. Backend routing is inconsistent with the main codebase

`agent.py` currently routes:

- GPT via `openai.AzureOpenAI` and lowercase env vars: `azure_api_key`, `azure_api_version`, `azure_endpoint`.
- Claude via Bedrock with lowercase AWS env vars.
- Qwen via Together using `TOGETHER_API_KEY`.

The main codebase uses LiteLLM-style model strings and env vars:

- `azure/gpt-5.4-mini` via Azure using `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`.
- `openrouter/anthropic/claude-sonnet-4.6` via OpenRouter using `OPENROUTER_API_KEY`.
- `openrouter/moonshotai/kimi-k2.5` via OpenRouter using `OPENROUTER_API_KEY`.

The Box Loop baseline should match those model strings.

### 4. No `--dataset-idx`

The current runner always loads all Time Series entries. It cannot run chunks like:

- `0:50`
- `50:100`
- `100:110`
- `0,3,7`
- `42`

### 5. Serial only

`run_all_arrays` loops serially over all entries. We need `--nproc` where:

- `--nproc 0`: run in the main process, sequentially.
- `--nproc >= 1`: use a simple `concurry` process worker pool.

The worker should run exactly one dataset entry per call. No process pool executor, no nested processes, no nested threads, no chunk worker inside another worker.

## Proposed file changes

### File 1: `baselines/box_loop/run.py`

#### CLI changes

Add:

- `--rounds`, default `5`.
- `--dataset-idx`, optional string.
- `--nproc`, default `0`.
- Possibly `--temperature`, default `0.7` if we choose to thread it to `LMExperimenter` later. This is optional for Option B because the core loop currently hardcodes `temperature=0.7`.
- Possibly `--max-tokens`, default `3000`. Also optional because the core loop currently hardcodes `max_tokens=3000`.

Recommended minimal signature:

```python
p.add_argument("--rounds", type=int, default=5, help="Total LLM calls per series: 1 proposal + rounds-1 improvements")
p.add_argument("--dataset-idx", default=None, help="Dataset indices: '5', '0,1,8', '0:50', or '0:50:2'")
p.add_argument("--nproc", type=int, default=0, help="0 = main process; >=1 = concurry process workers")
```

#### Data loading changes

Remove the debug slice from `load_data`.

For `load_data_ts`, keep returning `{item["series_id"]: item for item in series_list}` unless we find that `series_id` is not stable integer-like. The main Time Series datasets appear designed around `series_id`.

#### Dataset index parsing

Add a local `_parse_dataset_indices` helper to `run.py`, modeled on `experiments.py::_parse_dataset_indices`, supporting:

- `None` or empty string -> all datasets.
- Single integer -> one dataset.
- Comma-separated integers -> explicit list.
- Python-style slices `start:stop` and `start:stop:step`.

Important implementation detail: parse indices against the ordered list of loaded keys, not by assuming keys are contiguous. For Time Series, `arrays_dict` is keyed by `series_id`; the selected integer index should refer to list position in the loaded dataset order, matching `experiments.py` behavior.

Plan:

```python
array_items: List[Tuple[int, Any]] = list(arrays_dict.items())
parsed_indices = _parse_dataset_indices(args.dataset_idx, num_datasets=len(array_items))
if parsed_indices is not None:
    arrays_dict = {array_items[i][0]: array_items[i][1] for i in parsed_indices}
```

This means `--dataset-idx 0:50` selects the first 50 entries in the pickle, while preserving their original `series_id` keys for output.

#### Call into batch runner

Pass the new `nproc` argument:

```python
results = run_all_arrays(
    arrays_dict=arrays_dict,
    model_name=args.model,
    num_rounds=args.rounds,
    save_path=args.output,
    resume=not args.no_resume,
    task=args.task,
    nproc=args.nproc,
)
```

### File 2: `baselines/box_loop/simple_box_loop_adapter.py`

#### Defaults

Change defaults from 3 to 5 where relevant:

- `run_box_loop_for_array(..., num_rounds: int = 5, ...)`
- `run_all_arrays(..., num_rounds: int = 5, ...)`
- Module usage docstring should say `num_rounds=5`.

This does not change the meaning of the core loop; it just changes the default total LLM calls.

#### Persistence fix

Replace CSV-only checkpointing with pickle checkpointing to the exact `save_path`.

Recommended behavior:

- Store sanitized, serializable results at `save_path` after every completed dataset.
- Optionally also write a sibling CSV summary for convenience, but never as the only checkpoint.
- Resume reads the same pickle file it writes.

Recommended minimal data contract:

```python
_SANITIZED_KEYS = {"array_id", "best_code", "best_loo", "best_waic", "success", "all_rounds"}
```

`all_rounds` should be sanitized too, because successful rounds currently include `trace` and `model`, which can be large and may not serialize reliably across processes. Recommended sanitized round keys:

- `round`
- `code`
- `loo`
- `waic`
- `error`

This preserves enough information to debug generated code and scores while avoiding PyMC model/trace transport through process boundaries.

Implementation shape:

```python
def _sanitize_round(round_result: Dict[str, Any]) -> Dict[str, Any]:
    keep: Set[str] = {"round", "code", "loo", "waic", "error"}
    return {key: round_result[key] for key in keep if key in round_result}


def _sanitize(result: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {key: result[key] for key in keep if key in result}
    if "all_rounds" in result:
        sanitized["all_rounds"] = [_sanitize_round(round_result) for round_result in result["all_rounds"]]
    return sanitized
```

Use `pickle.dump(cleaned_results, f)` to write `save_path`.

For CSV summary, use `Path(save_path).with_suffix(".csv")` rather than `split('.')`.

#### Parallelism argument

Extend `run_all_arrays`:

```python
def run_all_arrays(..., nproc: int = 0) -> List[Dict[str, Any]]:
```

Behavior:

- Load `completed` from the pickle checkpoint if `resume=True`.
- Build `pending_items` from `arrays_dict.items()` excluding completed array IDs.
- If `nproc == 0`, run pending entries inline exactly as today.
- If `nproc > 0`, run pending entries through a `concurry` process worker.
- Persist after each resolved result in both modes.

Important: do not mutate `run_box_loop_for_array` for parallelism. The worker should call it.

### File 3: `baselines/box_loop/box_loop_workers.py` (new)

Create an importable worker module so `concurry` process mode can serialize by module path, matching the rationale in `experiment_workers.py`.

Proposed worker:

```python
from typing import Any, Dict

from concurry import Worker
from morphic import Typed, validate


class BoxLoopDatasetWorker(Typed, Worker):
    model_name: str
    num_rounds: int
    task: str

    @validate
    def run_dataset(self, *, array_id: int, observed_array: Any) -> Dict[str, Any]:
        # Inline import required: Worker method may execute in process remote context.
        from simple_box_loop_adapter import run_box_loop_for_array, _sanitize

        result: Dict[str, Any] = run_box_loop_for_array(
            observed_array=observed_array,
            array_id=array_id,
            model_name=self.model_name,
            num_rounds=self.num_rounds,
            task=self.task,
        )
        return _sanitize(result)
```

Notes:

- `observed_array` must be `Any` here because it can be a numpy array, a dict containing a pandas Series, or a pandas Series depending on task.
- The worker returns sanitized results only, so the parent process does not have to receive PyMC traces/models from child processes.
- This keeps the core loop unchanged.

### File 4: `baselines/box_loop/agent.py`

#### Replace ad-hoc provider routing with explicit provider branches

Keep `LMExperimenter` as the Box Loop API, but add routing for the main model strings.

Recommended model strings:

- `azure/gpt-5.4-mini`
- `openrouter/anthropic/claude-sonnet-4.6`
- `openrouter/moonshotai/kimi-k2.5`

#### Azure GPT 5.4 mini

For `model_name.startswith("azure/")`, use Azure env vars matching LiteLLM and `slowburn_api.py` docs:

- `AZURE_API_KEY`
- `AZURE_API_BASE`
- `AZURE_API_VERSION`

When calling Azure directly with `openai.AzureOpenAI.responses.create`, strip the `azure/` prefix for the Azure deployment name:

```python
self._azure_deployment = model_name.removeprefix("azure/")
```

Call:

```python
response = self.llm.responses.create(
    model=self._azure_deployment,
    input=self.messages,
    max_output_tokens=self.max_tokens,
    reasoning={"effort": "low"},
)
```

This follows the current GPT path but uses standard env vars and supports `azure/gpt-5.4-mini`.

#### OpenRouter Claude Sonnet 4.6 and Kimi K2.5

For `model_name.startswith("openrouter/")`, use OpenAI-compatible chat completions:

- Client: `OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")`
- Messages: normal Chat Completions format.
- Model: pass the full OpenRouter model string or strip the prefix depending on OpenRouter/OpenAI SDK expectations.

The existing main codebase and LiteLLM use `openrouter/...` because LiteLLM strips/routes it. Direct OpenRouter Chat Completions usually expects the provider path without the LiteLLM prefix, e.g.:

- `anthropic/claude-sonnet-4.6`
- `moonshotai/kimi-k2.5`

So `LMExperimenter` should derive:

```python
self._openrouter_model = model_name.removeprefix("openrouter/")
```

Call shape:

```python
response = self.llm.chat.completions.create(
    model=self._openrouter_model,
    messages=self.messages,
    max_tokens=self.max_tokens,
    temperature=self.temperature,
    extra_body={"reasoning": {"effort": "low", "exclude": False}},
)
```

For Sonnet 4.6, mirror `experiment_config.py` by using token-budgeted reasoning rather than effort reasoning:

```python
extra_body={"reasoning": {"max_tokens": 1024, "exclude": False}}
```

For Kimi K2.5, use effort reasoning:

```python
extra_body={"reasoning": {"effort": "low", "exclude": False}}
```

If this direct OpenRouter model-name behavior differs in practice, the fallback is to pass the full `openrouter/...` string only for OpenRouter calls. The first implementation should use stripped provider paths because that is OpenRouter-native.

#### Message format cleanup

Current `set_system_message` only handles GPT and Claude, and `add_message` only handles GPT, Claude, or Qwen. Add a generic OpenAI-chat message format for OpenRouter:

```python
{"role": role, "content": message}
```

Use separate internal provider tags rather than repeated string checks:

- `self._provider = "azure"`
- `self._provider = "openrouter"`
- optionally retain `"together"` for old Qwen behavior if needed.

Do not use `.get()` fallbacks for required env vars. Use `os.environ[...]` or explicit check-and-raise with clear messages.

### File 5: `baselines/box_loop/run.sh`

Update the example script so it is safe and aligned with the new CLI:

- Do not hardcode empty secrets.
- Document required env vars instead of assigning empty strings.
- Use `ROUNDS=5`.
- Add `DATASET_IDX="0:50"`.
- Add `NPROC=0` by default.
- Use one of the supported model strings.

Recommended example:

```bash
#!/bin/bash
set -euo pipefail

DATA="../../dataset_time_series/dataset_ts_easy_50.pkl"
OUTPUT="outputs/box_loop_ts_easy_claude_sonnet46.pkl"
MODEL="openrouter/anthropic/claude-sonnet-4.6"
ROUNDS=5
DATASET_IDX="0:50"
NPROC=0

python run.py \
    --data "$DATA" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --rounds "$ROUNDS" \
    --dataset-idx "$DATASET_IDX" \
    --nproc "$NPROC" \
    --task "box_loop_ts"
```

Required env vars by model:

- Azure GPT: `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`.
- OpenRouter Claude/Kimi: `OPENROUTER_API_KEY`.

## Parallel execution design

### `nproc=0`: main-process sequential path

This is intentionally not a `concurry` sync worker. It runs exactly like today:

```python
for array_id, observed_array in pending_items:
    result = run_box_loop_for_array(...)
    result = _sanitize(result)
    checkpoint(results)
```

This satisfies “default with `nproc=0`, use the main process, similar to `experiments.py`.”

### `nproc>=1`: one-level `concurry` process path

Create one `BoxLoopDatasetWorker` proxy:

```python
worker = BoxLoopDatasetWorker.options(
    mode="process",
    max_workers=min(nproc, len(pending_items)),
).init(model_name=model_name, num_rounds=num_rounds, task=task)
```

Submit one future per dataset entry:

```python
futures = [
    worker.run_dataset(array_id=array_id, observed_array=observed_array)
    for array_id, observed_array in pending_items
]
```

Resolve futures with `concurry.gather`:

```python
resolved_results = gather(futures, return_exceptions=True, progress=len(futures) > 1)
```

Then iterate over resolved results, convert exceptions into error result dicts, append them, and checkpoint. Finally call `worker.stop()` in a `finally` block.

This is a single process-worker layer. There is no inner thread worker, no nested pool, no chunk worker.

### Future handling requirement

A `concurry` worker method returns a future. The parent must not treat the return value as an immediate dict. The correct handling is:

- For one future: `future.result()`.
- For many futures: `concurry.gather(futures, ...)`.

This plan uses `gather` because the baseline run will usually submit many datasets.

## Output contract

After this change, a run with:

```bash
python run.py --output outputs/box_loop_ts_easy_kimi25.pkl ...
```

will write:

- `outputs/box_loop_ts_easy_kimi25.pkl`: primary checkpoint/resume file.
- `outputs/box_loop_ts_easy_kimi25.csv`: optional summary file for quick inspection.

The pickle contains a list of sanitized result dicts:

```python
{
    "array_id": int,
    "best_code": Optional[str],
    "best_loo": float,
    "best_waic": float,
    "success": bool,
    "all_rounds": [
        {"round": int, "code": str, "loo": float, "waic": float, "error": Optional[str]},
        ...
    ],
}
```

This is intentionally smaller and more robust than preserving PyMC model and trace objects in the cross-process checkpoint.

## Time Series command matrix for baseline equivalence

The baseline only needs the `none`-toolkit analog because Box Loop is its own agentic loop and does not implement the project toolkit modes.

### Easy: `dataset_ts_easy_50.pkl`

```bash
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_easy_50.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "openrouter/anthropic/claude-sonnet-4.6" --output outputs/box_loop_ts_easy_claude_sonnet46.pkl
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_easy_50.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "openrouter/moonshotai/kimi-k2.5" --output outputs/box_loop_ts_easy_kimi25.pkl
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_easy_50.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "azure/gpt-5.4-mini" --output outputs/box_loop_ts_easy_gpt54_mini.pkl
```

### Medium: `dataset_ts_medium_110.pkl`

Run in chunks matching `runs_medium.sh`:

```bash
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_medium_110.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "openrouter/anthropic/claude-sonnet-4.6" --output outputs/box_loop_ts_medium_0to50_claude_sonnet46.pkl
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_medium_110.pkl --dataset-idx "50:100" --rounds 5 --nproc 0 --model "openrouter/anthropic/claude-sonnet-4.6" --output outputs/box_loop_ts_medium_50to100_claude_sonnet46.pkl
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_medium_110.pkl --dataset-idx "100:110" --rounds 5 --nproc 0 --model "openrouter/anthropic/claude-sonnet-4.6" --output outputs/box_loop_ts_medium_100to110_claude_sonnet46.pkl
```

Repeat the same chunks for:

- `openrouter/moonshotai/kimi-k2.5`
- `azure/gpt-5.4-mini`

### Hard: `dataset_ts_gravitational_chirp_50.pkl`

```bash
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_gravitational_chirp_50.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "openrouter/anthropic/claude-sonnet-4.6" --output outputs/box_loop_ts_gravitational_chirp_claude_sonnet46.pkl
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_gravitational_chirp_50.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "openrouter/moonshotai/kimi-k2.5" --output outputs/box_loop_ts_gravitational_chirp_kimi25.pkl
python baselines/box_loop/run.py --task box_loop_ts --data dataset_time_series/dataset_ts_gravitational_chirp_50.pkl --dataset-idx "0:50" --rounds 5 --nproc 0 --model "azure/gpt-5.4-mini" --output outputs/box_loop_ts_gravitational_chirp_gpt54_mini.pkl
```

Once smoke tests pass with `--nproc 0`, increase `--nproc` carefully based on API limits, for example:

- Claude/OpenRouter: `--nproc 2` to start.
- Kimi/OpenRouter: `--nproc 2` to start, then raise if rate limits permit.
- Azure GPT 5.4 mini: `--nproc 2` to start.

## Validation plan

### Static smoke checks

1. Import `run.py` and `simple_box_loop_adapter.py`.
2. Parse CLI defaults and verify `rounds == 5`, `nproc == 0`.
3. Test `_parse_dataset_indices` for:
   - `None`
   - `"5"`
   - `"0,1,8"`
   - `"0:9"`
   - `"0:9:2"`
   - out-of-range index raises `ValueError`.

### Persistence smoke test without an LLM call

Mock or temporarily replace `run_box_loop_for_array` in a small script so it returns deterministic results. Verify:

1. First run writes `results.pkl`.
2. First run also writes `results.csv` if CSV summary is enabled.
3. Second run with `resume=True` skips completed array IDs.
4. `--no-resume` reruns all selected entries.

### Single real LLM smoke test

Run one easy Time Series dataset:

```bash
python baselines/box_loop/run.py \
    --task box_loop_ts \
    --data dataset_time_series/dataset_ts_easy_50.pkl \
    --dataset-idx "0" \
    --rounds 1 \
    --nproc 0 \
    --model "openrouter/moonshotai/kimi-k2.5" \
    --output outputs/smoke_box_loop_ts_easy_kimi25.pkl \
    --no-resume
```

This validates OpenRouter routing and the Time Series prompt path cheaply.

Then repeat for:

- `openrouter/anthropic/claude-sonnet-4.6`
- `azure/gpt-5.4-mini`

### Process-mode smoke test

Use two easy datasets and one LLM round:

```bash
python baselines/box_loop/run.py \
    --task box_loop_ts \
    --data dataset_time_series/dataset_ts_easy_50.pkl \
    --dataset-idx "0:2" \
    --rounds 1 \
    --nproc 2 \
    --model "openrouter/moonshotai/kimi-k2.5" \
    --output outputs/smoke_box_loop_ts_easy_kimi25_nproc2.pkl \
    --no-resume
```

This validates:

- worker serialization,
- future resolution via `gather`,
- child-process imports,
- sanitized cross-process results,
- checkpoint writing in parent process.

## Implementation order

1. Update `run.py` CLI defaults and add `--dataset-idx`/`--nproc`.
2. Add `_parse_dataset_indices` and selection logic in `run.py`.
3. Remove `load_data` debug slice.
4. Update `simple_box_loop_adapter.py` defaults to 5.
5. Implement sanitized pickle checkpointing in `simple_box_loop_adapter.py`.
6. Add `box_loop_workers.py` with `BoxLoopDatasetWorker`.
7. Add `nproc` orchestration in `run_all_arrays`.
8. Update `agent.py` provider routing for Azure/OpenRouter model strings.
9. Update `run.sh` to safe env expectations and new defaults.
10. Run static smoke checks.
11. Run one-dataset `--rounds 1 --nproc 0` smoke tests per provider.
12. Run two-dataset `--rounds 1 --nproc 2` smoke test.

## Risks and mitigations

### Risk: OpenRouter direct client model name may require full `openrouter/...` string

Mitigation: implement the OpenRouter model name normalization in one place (`self._api_model_name`). If stripped provider names fail, switch that one assignment to use the full string.

### Risk: PyMC trace/model objects are not safe to return across processes

Mitigation: workers return sanitized result dicts only. The core loop can still keep trace/model internally for scoring; parent checkpoint does not depend on serializing them.

### Risk: parallel runs exceed provider RPM limits

Mitigation: default remains `--nproc 0`. Start smoke tests with `--nproc 2`. Do not implement internal thread pools or nested concurrency.

### Risk: Box Loop Time Series prompt is not toolkit-equivalent

Mitigation: this is expected. Box Loop is its own baseline; compare it to the main pipeline’s Time Series dataset splits and model set, not to toolkit modes one-for-one.

### Risk: `load_data_ts` keying by `series_id` conflicts with position-based chunking

Mitigation: chunking selects by loaded order positions, then preserves original keys. Output `array_id` remains `series_id`.

## Acceptance criteria

- `python baselines/box_loop/run.py --help` shows `--rounds` default 5, `--dataset-idx`, and `--nproc`.
- `load_data` no longer slices to one item.
- `--output something.pkl` creates `something.pkl` and resume reads that same file.
- Optional CSV summary, if present, is a secondary artifact only.
- `--model openrouter/moonshotai/kimi-k2.5` routes through OpenRouter.
- `--model openrouter/anthropic/claude-sonnet-4.6` routes through OpenRouter, not Bedrock Sonnet 4.
- `--model azure/gpt-5.4-mini` uses Azure env vars with uppercase names.
- `--dataset-idx` supports single indices, comma lists, and slices.
- `--nproc 0` runs sequentially in the main process.
- `--nproc 2` uses one `concurry` process worker layer and resolves futures correctly.
- `run_box_loop_for_array` has no parallelism-specific edits.
