"""
run.py
======
Run Box's Apprentice on distribution or time-series datasets.

Usage:
    python run.py --data path/to/time_series.pkl --rounds 5 --model openrouter/moonshotai/kimi-k2.5
"""

import argparse
import datetime
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_BOX_LOOP_DIR: Path = Path(__file__).resolve().parent
_PROJECT_ROOT: Path = _BOX_LOOP_DIR.parents[1]
for path in (str(_PROJECT_ROOT), str(_BOX_LOOP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# These must run before numpy / scipy / pymc / pytensor import in this process.
import _thread_caps  # noqa: E402,F401,I001
import _pytensor_compiledir  # noqa: E402,F401,I001

from dotenv import load_dotenv  # noqa: E402

_ENV_PATH: Path = _PROJECT_ROOT / ".env"
_DOTENV_LOADED: bool = load_dotenv(_ENV_PATH)
_OUTPUT_DIR: Path = _BOX_LOOP_DIR / "outputs"

if sys.platform == "darwin":
    os.environ.setdefault("no_proxy", "*")

import numpy as np  # noqa: E402,I001

from simple_box_loop_adapter import run_all_arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="Path to your .pkl file.",
    )
    parser.add_argument(
        "--output",
        default="results",
        help=(
            "Output run name, with optional .pkl extension. Files are always saved under outputs/ as "
            "<name>_YYYYMMDD_HHMMSS.pkl; --resume loads the latest matching timestamped .pkl file."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Total LLM calls per series: 1 proposal + rounds-1 improvements (default: 5).",
    )
    parser.add_argument(
        "--dataset-idx",
        default=None,
        help="Dataset positions: '5', '0,1,8', '0:50', or '0:50:2'.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=0,
        help="0 = main process; >=1 = one Concurry process-worker layer.",
    )
    parser.add_argument(
        "--max-rpm",
        "--parallel.max-rpm",
        dest="max_rpm",
        type=int,
        default=30,
        help="Maximum LLM requests per minute (default: 30, spread evenly across workers).",
    )
    parser.add_argument(
        "--model",
        default="azure/gpt-5.4-mini",
        help=(
            "LLM model string, e.g. azure/gpt-5.4-mini, "
            "openrouter/anthropic/claude-sonnet-4.6, or openrouter/moonshotai/kimi-k2.5."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3000,
        help="Maximum output tokens per LLM call (default: 3000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the latest timestamped checkpoint for --output. "
            "Default: create a new timestamped run."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["box_loop", "box_loop_ts"],
        default="box_loop_ts",
        help="Which task to run: original (box_loop) or time-series GP (box_loop_ts).",
    )
    return parser.parse_args()


def load_data(path: str) -> Dict[int, np.ndarray]:
    """Load the pkl file and return {array_id: 1-D numpy array}."""
    with open(path, "rb") as input_file:
        raw: Any = pickle.load(input_file)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of dicts, got {type(raw)}.")
    if len(raw) == 0:
        raise ValueError(f"Expected at least one item in {path}, got an empty list.")
    if "data" not in raw[0]:
        raise ValueError(f"Each dict must have a 'data' key. Keys found: {list(raw[0].keys())}.")

    arrays_dict: Dict[int, np.ndarray] = {}
    for item_idx, item in enumerate(raw):
        array: np.ndarray = np.asarray(item["data"], dtype=float)
        if array.ndim != 1:
            raise ValueError(
                f"Item {item_idx}: 'data' must be 1-D, got shape {array.shape}. "
                f"Flatten it before saving, or update this loader."
            )
        arrays_dict[item_idx] = array

    print(f"Loaded {len(arrays_dict)} arrays from {path}")
    for item_idx in range(min(3, len(arrays_dict))):
        array = arrays_dict[item_idx]
        print(f"  [{item_idx}] n={len(array)}  mean={array.mean():.3f}  std={array.std():.3f}")
    if len(arrays_dict) > 3:
        print(f"  ... and {len(arrays_dict) - 3} more")

    return arrays_dict


def load_data_ts(path: str) -> Dict[Union[int, str], Dict[str, Any]]:
    """
    Load a list-of-dicts time series dataset.
    Each element has keys: series_id, unique_id, category, name,
                           anomaly_info, source_path, data (pd.Series).
    Returns {series_id: series_dict} — same shape as arrays_dict for box_loop.
    """
    with open(path, "rb") as input_file:
        series_list: Any = pickle.load(input_file)

    if not isinstance(series_list, list):
        raise ValueError(f"Expected a list of time-series dicts, got {type(series_list)}.")
    if len(series_list) == 0:
        raise ValueError(f"Expected at least one time series in {path}, got an empty list.")
    return {item["series_id"]: item for item in series_list}


def _parse_dataset_idx(*, dataset_idx: Optional[str], total: int) -> List[int]:
    if dataset_idx is None:
        return list(range(total))

    dataset_idx = dataset_idx.strip()
    if len(dataset_idx) == 0:
        return list(range(total))

    positions: List[int]
    if ":" in dataset_idx:
        parts: List[str] = dataset_idx.split(":")
        if len(parts) == 2:
            start: int = int(parts[0]) if len(parts[0]) > 0 else 0
            stop: int = int(parts[1]) if len(parts[1]) > 0 else total
            positions = list(range(start, stop))
        elif len(parts) == 3:
            start = int(parts[0]) if len(parts[0]) > 0 else 0
            stop = int(parts[1]) if len(parts[1]) > 0 else total
            step: int = int(parts[2]) if len(parts[2]) > 0 else 1
            positions = list(range(start, stop, step))
        else:
            raise ValueError(
                f"Invalid dataset_idx slice {dataset_idx!r}. Expected 'start:stop' or 'start:stop:step'."
            )
    elif "," in dataset_idx:
        positions = [int(part.strip()) for part in dataset_idx.split(",")]
    else:
        positions = [int(dataset_idx)]

    for position in positions:
        if position < 0 or position >= total:
            raise ValueError(
                f"dataset_idx position {position} out of range. File has {total} datasets (0..{total - 1})."
            )

    return positions


def _select_dataset_idx(
    *,
    arrays_dict: Dict[Union[int, str], Any],
    dataset_idx: Optional[str],
) -> Dict[Union[int, str], Any]:
    positions: List[int] = _parse_dataset_idx(dataset_idx=dataset_idx, total=len(arrays_dict))
    items: List[Tuple[Union[int, str], Any]] = list(arrays_dict.items())
    selected_items: List[Tuple[Union[int, str], Any]] = [items[position] for position in positions]
    return dict(selected_items)


def _normalize_output_base_path(*, output_path: Path) -> Path:
    if output_path.parent != Path("."):
        raise ValueError(
            f"--output must be a run name without a folder because outputs always go in {_OUTPUT_DIR}, "
            f"got {output_path}."
        )
    if len(output_path.suffix) == 0:
        return _OUTPUT_DIR / output_path.name
    if output_path.suffix == ".pkl":
        return _OUTPUT_DIR / output_path.with_suffix("").name
    raise ValueError(
        f"--output must be a run name with no extension or a .pkl extension, got {output_path}."
    )


def _timestamped_output_path(*, output_path: Path, timestamp: str) -> Path:
    output_base_path: Path = _normalize_output_base_path(output_path=output_path)
    return output_base_path.with_name(f"{output_base_path.name}_{timestamp}.pkl")


def _is_valid_output_timestamp(*, timestamp: str) -> bool:
    return (
        len(timestamp) == 15
        and timestamp[8] == "_"
        and timestamp[:8].isdigit()
        and timestamp[9:].isdigit()
    )


def _is_timestamped_output_for_base(*, candidate_path: Path, output_path: Path) -> bool:
    output_base_path: Path = _normalize_output_base_path(output_path=output_path)
    output_prefix: str = f"{output_base_path.name}_"
    if candidate_path.suffix != ".pkl":
        return False
    if not candidate_path.stem.startswith(output_prefix):
        return False
    timestamp: str = candidate_path.stem[len(output_prefix) :]
    return _is_valid_output_timestamp(timestamp=timestamp)


def _latest_timestamped_output_path(*, output_path: Path) -> Path:
    output_base_path: Path = _normalize_output_base_path(output_path=output_path)
    matching_paths: List[Path] = [
        candidate_path
        for candidate_path in output_base_path.parent.glob(f"{output_base_path.name}_*.pkl")
        if candidate_path.is_file()
        and _is_timestamped_output_for_base(
            candidate_path=candidate_path,
            output_path=output_path,
        )
    ]
    if len(matching_paths) == 0:
        example_output_path: Path = _timestamped_output_path(
            output_path=output_path,
            timestamp="YYYYMMDD_HHMMSS",
        )
        raise FileNotFoundError(
            f"--resume was passed, but no timestamped checkpoint matched base output {output_path}. "
            f"Expected files like {example_output_path}."
        )
    matching_paths.sort()
    return matching_paths[-1]


def _resolve_save_path(*, output_path: Path, resume: bool) -> Path:
    output_base_path: Path = _normalize_output_base_path(output_path=output_path)
    output_base_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        return _latest_timestamped_output_path(output_path=output_path)
    timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _timestamped_output_path(output_path=output_path, timestamp=timestamp)


def _required_env_names_for_model(*, model_name: str) -> List[str]:
    if model_name.startswith("azure/"):
        return ["AZURE_API_KEY", "AZURE_API_VERSION", "AZURE_API_BASE"]
    elif model_name.startswith("openrouter/"):
        return ["OPENROUTER_API_KEY"]
    elif "Qwen" in model_name or "together" in model_name:
        return ["TOGETHERAI_API_KEY"]
    else:
        raise ValueError(f"Unsupported model {model_name!r}.")


def _print_runner_configuration(
    *,
    args: argparse.Namespace,
    save_path: Path,
    per_worker_rpm: int,
) -> None:
    required_env_names: List[str] = _required_env_names_for_model(model_name=args.model)
    missing_env_names: List[str] = [name for name in required_env_names if name not in os.environ]
    empty_env_names: List[str] = [
        name for name in required_env_names if name in os.environ and len(os.environ[name]) == 0
    ]
    print(
        "Runner configuration:\n"
        f"  project_root={_PROJECT_ROOT}\n"
        f"  box_loop_dir={_BOX_LOOP_DIR}\n"
        f"  dotenv_path={_ENV_PATH}\n"
        f"  dotenv_loaded={_DOTENV_LOADED}\n"
        f"  output_dir={_OUTPUT_DIR}\n"
        f"  pytensor_flags={os.environ.get('PYTENSOR_FLAGS', '<unset>')}\n"
        f"  openblas_num_threads={os.environ.get('OPENBLAS_NUM_THREADS', '<unset>')}\n"
        f"  omp_num_threads={os.environ.get('OMP_NUM_THREADS', '<unset>')}\n"
        f"  model={args.model}\n"
        f"  temperature={args.temperature}\n"
        f"  max_tokens={args.max_tokens}\n"
        f"  resume={args.resume}\n"
        f"  base_output={_normalize_output_base_path(output_path=Path(args.output))}\n"
        f"  checkpoint_output={save_path}\n"
        f"  required_env_names={required_env_names}\n"
        f"  missing_env_names={missing_env_names}\n"
        f"  empty_env_names={empty_env_names}\n"
        f"  max_rpm={args.max_rpm}\n"
        f"  per_worker_rpm={per_worker_rpm}"
    )


def main() -> None:
    args: argparse.Namespace = parse_args()
    output_path: Path = Path(args.output)
    save_path: Path = _resolve_save_path(output_path=output_path, resume=args.resume)
    per_worker_rpm: int = args.max_rpm // max(1, args.nproc)
    if args.max_rpm > 0 and per_worker_rpm == 0:
        per_worker_rpm = 1

    _print_runner_configuration(
        args=args,
        save_path=save_path,
        per_worker_rpm=per_worker_rpm,
    )

    # ── load data ──────────────────────────────────────────────────────────
    if args.task == "box_loop_ts":
        arrays_dict: Dict[Union[int, str], Any] = load_data_ts(args.data)
    else:
        arrays_dict = load_data(args.data)

    arrays_dict = _select_dataset_idx(
        arrays_dict=arrays_dict,
        dataset_idx=args.dataset_idx,
    )
    print(
        f"Selected {len(arrays_dict)} datasets"
        f" with dataset_idx={args.dataset_idx!r}, nproc={args.nproc}, "
        f"max_rpm={args.max_rpm}, per_worker_rpm={per_worker_rpm}."
    )

    # ── run ───────────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = run_all_arrays(
        arrays_dict=arrays_dict,
        model_name=args.model,
        num_rounds=args.rounds,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_path=str(save_path),
        resume=args.resume,
        task=args.task,
        nproc=args.nproc,
        per_worker_rpm=per_worker_rpm,
    )

    # ── summary ───────────────────────────────────────────────────────────
    n_ok: int = sum(result["success"] for result in results)
    print(f"\n{'=' * 50}")
    print(f"Done.  {n_ok}/{len(results)} arrays succeeded.")
    print(f"Results saved to: {save_path}")
    print(f"{'=' * 50}\n")

    print("LOO scores (best first):")
    for result in sorted(results, key=lambda item: item["best_loo"], reverse=True):
        status: str = "✓" if result["success"] else "✗"
        print(f"  [{status}] array {str(result['array_id']):>12}  LOO={result['best_loo']:.2f}")


if __name__ == "__main__":
    main()
