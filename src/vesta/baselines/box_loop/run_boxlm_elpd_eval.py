#!/usr/bin/env python
"""Run one BoxLM Time Series ELPD evaluation by model and dataset name."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

BOX_LOOP_DIR: Path = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR: Path = BOX_LOOP_DIR / "outputs"
DEFAULT_OUTPUT_BASE: Path = BOX_LOOP_DIR / "evals"
DEFAULT_DATASET_PATHS: Dict[str, Path] = {
    "easy_50": BOX_LOOP_DIR / "dataset_ts_easy_50.pkl",
    "medium_110": BOX_LOOP_DIR / "dataset_ts_medium_110.pkl",
    "gravitational_chirp_50": BOX_LOOP_DIR / "dataset_ts_gravitational_chirp_50.pkl",
}
DEFAULT_RUN_PREFIXES: Dict[str, Dict[str, List[str]]] = {
    "easy_50": {
        "claude": ["box_loop_ts_easy_claude_sonnet46"],
        "gpt": ["box_loop_ts_easy_gpt54_mini"],
        "kimi": ["box_loop_ts_easy_kimi25"],
    },
    "medium_110": {
        "claude": [
            "box_loop_ts_medium_claude_sonnet46",
            "box_loop_ts_medium_claude_sonnet46_50to100",
            "box_loop_ts_medium_claude_sonnet46_100to110",
        ],
        "gpt": [
            "box_loop_ts_medium_gpt54_mini",
            "box_loop_ts_medium_gpt54_mini_50to100",
            "box_loop_ts_medium_gpt54_mini_100to110",
        ],
        "kimi": [
            "box_loop_ts_medium_kimi25",
            "box_loop_ts_medium_kimi25_50to100",
            "box_loop_ts_medium_kimi25_100to110",
        ],
    },
    "gravitational_chirp_50": {
        "claude": ["box_loop_ts_gravitational_chirp_claude_sonnet46"],
        "gpt": ["box_loop_ts_gravitational_chirp_gpt54_mini"],
        "kimi": ["box_loop_ts_gravitational_chirp_kimi25"],
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Resolve one BoxLM run by --model and --dataset, then invoke "
            "evaluate_boxlm_elpd.py with the matching CSV shard(s)."
        )
    )
    parser.add_argument(
        "--model",
        choices=["claude", "gpt", "kimi"],
        required=True,
        help="Model alias to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        choices=["easy_50", "medium_110", "gravitational_chirp_50"],
        required=True,
        help="Dataset alias to evaluate.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing timestamped BoxLM CSV outputs.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory where per-run ELPD output directories are created.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to invoke evaluate_boxlm_elpd.py.",
    )
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument("--max-obs", type=int, default=150)
    parser.add_argument("--n-subsample-reps", type=int, default=3)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Forwarded to evaluate_boxlm_elpd.py for debugging small runs.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Forwarded to evaluate_boxlm_elpd.py as comma-separated sample indices.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def _is_timestamped_csv_for_prefix(*, candidate_path: Path, prefix: str) -> bool:
    if candidate_path.suffix != ".csv":
        return False
    if not candidate_path.stem.startswith(f"{prefix}_"):
        return False
    suffix: str = candidate_path.stem[len(prefix) + 1 :]
    return len(suffix) == 15 and suffix[8] == "_" and suffix[:8].isdigit() and suffix[9:].isdigit()


def _latest_csv_for_prefix(*, runs_dir: Path, prefix: str) -> Path:
    matching_paths: List[Path] = [
        candidate_path
        for candidate_path in runs_dir.glob(f"{prefix}_*.csv")
        if candidate_path.is_file()
        and _is_timestamped_csv_for_prefix(
            candidate_path=candidate_path,
            prefix=prefix,
        )
    ]
    if len(matching_paths) == 0:
        raise FileNotFoundError(
            f"No timestamped BoxLM CSV found for prefix {prefix!r} in {runs_dir}. "
            f"Expected a file like {runs_dir / f'{prefix}_YYYYMMDD_HHMMSS.csv'}."
        )
    matching_paths.sort()
    return matching_paths[-1]


def _resolve_csv_paths(*, model: str, dataset: str, runs_dir: Path) -> List[Path]:
    prefixes: List[str] = DEFAULT_RUN_PREFIXES[dataset][model]
    return [_latest_csv_for_prefix(runs_dir=runs_dir, prefix=prefix) for prefix in prefixes]


def _resolve_dataset_path(*, dataset: str) -> Path:
    dataset_path: Path = DEFAULT_DATASET_PATHS[dataset]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_path}")
    return dataset_path


def _build_command(
    *,
    args: argparse.Namespace,
    csv_paths: List[Path],
    dataset_path: Path,
    output_dir: Path,
) -> List[str]:
    command: List[str] = [
        str(args.python),
        str(BOX_LOOP_DIR / "evaluate_boxlm_elpd.py"),
    ]
    command.extend(str(csv_path) for csv_path in csv_paths)
    command.extend(
        [
            "--dataset-pkl",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--draws",
            str(args.draws),
            "--tune",
            str(args.tune),
            "--chains",
            str(args.chains),
            "--cores",
            str(args.cores),
            "--max-obs",
            str(args.max_obs),
            "--n-subsample-reps",
            str(args.n_subsample_reps),
            "--target-accept",
            str(args.target_accept),
        ]
    )
    if args.max_samples is not None:
        command.extend(["--max-samples", str(args.max_samples)])
    if args.sample_indices is not None:
        command.extend(["--sample-indices", args.sample_indices])
    return command


def _print_command(*, command: List[str]) -> None:
    print("Resolved command:")
    print(" ".join(command))


def main() -> None:
    """Resolve the selected run and invoke the lower-level evaluator."""
    args: argparse.Namespace = parse_args()
    runs_dir: Path = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    csv_paths: List[Path] = _resolve_csv_paths(
        model=args.model,
        dataset=args.dataset,
        runs_dir=runs_dir,
    )
    dataset_path: Path = _resolve_dataset_path(dataset=args.dataset)
    output_dir: Path = (args.output_base / f"boxlm_{args.model}_{args.dataset}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"BoxLM ELPD selection: model={args.model}, dataset={args.dataset}")
    print(f"Runs dir          : {runs_dir}")
    print(f"CSV shard count   : {len(csv_paths)}")
    for csv_path in csv_paths:
        print(f"  - {csv_path}")
    print(f"Dataset pkl       : {dataset_path}")
    print(f"Output dir        : {output_dir}")

    command: List[str] = _build_command(
        args=args,
        csv_paths=csv_paths,
        dataset_path=dataset_path,
        output_dir=output_dir,
    )
    _print_command(command=command)
    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
