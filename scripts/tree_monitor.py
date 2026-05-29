#!/usr/bin/env python3
"""Tree-style monitoring of all experiment runs under outputs/.

Usage::

    watch -n 60 scripts/tree_monitor.py                  # tree only
    python scripts/tree_monitor.py --errors               # tree + error summaries
    python scripts/tree_monitor.py --latest 3             # latest 3 runs per expt
    python scripts/tree_monitor.py --latest 3 --errors    # latest 3 + errors

When ``--errors`` is passed, the script also inspects every ERROR
dataset's ``run_log.parquet`` and prints the last error line (error
class + final sentence) for triage.

When ``--latest N`` is passed, only the last N runs (by timestamp)
are shown per experiment, and the experiment header shows
``(latest: N)``.

The script scans ``outputs/`` recursively, groups runs by experiment,
then by timestamp/data_name, and prints a concise tree showing
how many datasets are PENDING, IN_PROGRESS, SUCCEEDED, or ERROR.

Derived from ``run_log.parquet`` (source of truth) and ``config.json``
(TOTAL count from ``dataset_idx`` range).
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_OUTPUTS_DIR: str
if os.path.isdir(os.path.join(_SCRIPT_DIR, "outputs")):
    _OUTPUTS_DIR = os.path.join(_SCRIPT_DIR, "outputs")
else:
    _OUTPUTS_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "outputs")
print(f"Outputs dir: {_OUTPUTS_DIR}")

# ---------------------------------------------------------------------------
#  Error classification keywords
# ---------------------------------------------------------------------------
# Classification uses two signals:
#   1. The exception type in the first line (before any Traceback)
#   2. Known pipeline-level error messages (no standard exception type)
#
# Genuine PyMC/PyTensor failures observed across 140+ parquet error entries:
#   - MissingGXX: PyTensor C compiler not available
#   - "cannot access local variable": PyTensor UnboundLocalError bug
#   - "VLM proposed no usable models": Pipeline PyMC failure
#   - RuntimeError "All N models failed to fit": PyMC sampling failure
#
# Everything else (TimeoutError, AuthenticationError, BadRequestError,
# RateLimitError, KeyError, ValueError for schema mismatch, etc.) is
# OTHER_ERROR — API issues, code bugs, or VLM response validation failures.
_PYMC_ERROR_PATTERNS: Tuple[str, ...] = (
    "missinggxx",
    "cannot access local variable",
    "vlm proposed no usable models",
    "all \\d+ models failed to fit",
)


def _classify_error(error_str: str) -> str:
    """Return a coarse error-class label from an error string.

    Returns one of ``PYMC_ERROR`` or ``OTHER_ERROR``.

    Classification uses the summary section before the first
    ``Traceback:`` to avoid false matches from file paths.
    """
    summary: str = error_str.split("\nTraceback:", 1)[0]
    summary_lower: str = summary.lower()
    if any(re.search(pat, summary_lower) for pat in _PYMC_ERROR_PATTERNS):
        return "PYMC_ERROR"
    return "OTHER_ERROR"


def _shorten_error(error_str: str, max_chars: int = 200) -> str:
    """Truncate an error string to a reader-friendly length.

    Tries to keep the first sentence (error class) and the last meaningful
    sentence. Falls back to a simple truncation.
    """
    if error_str is None or len(error_str) == 0:
        return ""
    if len(error_str) <= max_chars:
        return error_str

    lines: List[str] = error_str.replace("\\n", "\n").split("\n")
    # Find the last non-empty, non-stacktrace line
    meaningful_lines: List[str] = [
        l
        for l in lines
        if len(l.strip()) > 0
        and not l.strip().startswith(
            ("  ", "Traceback", "Apply node", "Inputs types", "HINT", "Toposort", "/home", "/work2")
        )
    ]
    first_bit: str = meaningful_lines[0][:150] if len(meaningful_lines) > 0 else error_str[:150]
    if len(error_str) > max_chars:
        last_bit: str = meaningful_lines[-1][:150] if len(meaningful_lines) > 0 else ""
        if len(first_bit) + len(last_bit) + 10 < max_chars * 2:
            return f"{first_bit} | ... | {last_bit}"
    return error_str[:max_chars]


# ---------------------------------------------------------------------------
#  Parsing helpers
# ---------------------------------------------------------------------------


def _parse_dataset_idx_range(value: str) -> int:
    """Parse a dataset_idx string like ``"0:50"`` or ``"0,1,2"`` into count.

    Args:
        value: The ``dataset_idx`` field from config.json.

    Returns:
        Number of datasets.
    """
    if value is None or len(value) == 0:
        return 0
    colon_match: Optional[re.Match] = re.match(r"^(\d+)\s*:\s*(\d+)$", value)
    if colon_match is not None:
        return int(colon_match.group(2)) - int(colon_match.group(1))
    comma_parts: List[str] = [p.strip() for p in value.split(",") if len(p.strip()) > 0]
    return len(comma_parts)


def _get_expected_total(*, run_dir: Path) -> int:
    """Read config.json to determine how many datasets were launched.

    Falls back to counting existing dataset subdirectories if config.json
    is missing.

    Args:
        run_dir: Path like ``outputs/<expt>/<timestamp>/<data_name>``.

    Returns:
        Expected total number of datasets.
    """
    config_path: Path = run_dir / "config.json"
    if config_path.exists() is True:
        with open(config_path, "r") as f:
            config_data: Dict[str, Any] = json.load(f)
        dataset_idx_raw: str = config_data.get("dataset_idx", "")
        count: int = _parse_dataset_idx_range(dataset_idx_raw)
        if count > 0:
            return count
    # Fallback: count existing subdirectories.
    return len([d for d in run_dir.iterdir() if d.is_dir()])


def _classify_from_parquet(*, parquet_path: Path, dataset_dir: Path) -> Tuple[str, str]:
    """Return (status, error_short) for one dataset.

    Args:
        parquet_path: Path to ``run_log.parquet``.
        dataset_dir: Path to the dataset directory (used to detect PENDING/IN_PROGRESS).

    Returns:
        Tuple of (status string, truncated error string or "").
    """
    if parquet_path.exists() is False:
        if dataset_dir.exists() is False:
            return ("PENDING", "")
        # Directory exists but no parquet — likely IN_PROGRESS.
        return ("IN_PROGRESS", "")

    dataframe: pd.DataFrame = pd.read_parquet(parquet_path)
    if len(dataframe) == 0:
        return ("IN_PROGRESS", "")

    last_row: Dict[str, Any] = dataframe.iloc[-1].to_dict()
    last_status: str = str(last_row.get("status", ""))
    last_error: str = str(last_row.get("error", "")) if last_row.get("error") is not None else ""

    if last_status == "ok" and len(last_error) == 0:
        return ("SUCCEEDED", "")
    if last_status == "error" or len(last_error) > 0:
        error_class: str = _classify_error(last_error)
        short: str = _shorten_error(last_error)
        return (error_class, short)
    return ("IN_PROGRESS", "")


def _scan_run(*, run_dir: Path) -> Dict[str, Any]:
    """Scan one run directory and return status counts with error details.

    Args:
        run_dir: Path like ``outputs/<expt>/<timestamp>/<data_name>``.

    Returns:
        Dict with total, PENDING, IN_PROGRESS, SUCCEEDED, PYMC_ERROR,
        OTHER_ERROR counts, plus a dict of dataset-names -> (error_class, error_short)
        for every errored dataset.
    """
    if run_dir.exists() is False:
        return {}

    expected_total: int = _get_expected_total(run_dir=run_dir)
    pending: int = 0
    in_process: int = 0
    succeeded: int = 0
    pymc_error: int = 0
    other_error: int = 0
    error_details: Dict[str, Tuple[str, str]] = {}

    dataset_dirs: List[Path] = sorted([d for d in run_dir.iterdir() if d.is_dir()])

    for dataset_dir in dataset_dirs:
        parquet_path: Path = dataset_dir / "run_log.parquet"
        status, err_short = _classify_from_parquet(parquet_path=parquet_path, dataset_dir=dataset_dir)
        if status == "PENDING":
            pending += 1
        elif status == "IN_PROGRESS":
            in_process += 1
        elif status == "SUCCEEDED":
            succeeded += 1
        elif status == "PYMC_ERROR":
            pymc_error += 1
            error_details[dataset_dir.name] = (status, err_short)
        elif status == "OTHER_ERROR":
            other_error += 1
            error_details[dataset_dir.name] = (status, err_short)

    # Count implicit PENDING dirs that don't exist yet.
    remaining: int = max(expected_total - pending - in_process - succeeded - pymc_error - other_error, 0)
    pending += remaining

    return {
        "run_dir": str(run_dir),
        "data_name": run_dir.name,
        "total": expected_total,
        "pending": pending,
        "in_process": in_process,
        "succeeded": succeeded,
        "pymc_error": pymc_error,
        "other_error": other_error,
        "error_details": error_details,
    }


def _build_tree(*, outputs_root: str) -> List[Dict[str, Any]]:
    """Build a tree of all experiments under outputs/.

    Args:
        outputs_root: Absolute path to the ``outputs/`` directory.

    Returns:
        List of experiment dicts, each with timestamp_runs list.
    """
    outputs_path: Path = Path(outputs_root)
    if outputs_path.exists() is False:
        return []

    experiments: Dict[str, Dict[str, Any]] = {}

    for expt_dir in sorted(outputs_path.iterdir()):
        if expt_dir.is_dir() is False:
            continue

        expt_name: str = expt_dir.name
        experiments[expt_name] = {
            "name": expt_name,
            "runs": [],
        }

        for ts_dir in sorted(expt_dir.iterdir()):
            if ts_dir.is_dir() is False:
                continue

            # Look for data_name subdirectories under ts_dir.
            for data_dir in sorted(ts_dir.iterdir()):
                if data_dir.is_dir() is False:
                    continue

                run_info: Dict[str, Any] = _scan_run(run_dir=data_dir)
                if len(run_info) > 0:
                    experiments[expt_name]["runs"].append(run_info)

    return list(experiments.values())


def _format_counts(run_info: Dict[str, Any]) -> str:
    """Format a compact counts string.

    Args:
        run_info: Dict from ``_scan_run``.

    Returns:
        Formatted string like ``50 TOTAL, 9 PENDING, 6 IN_PROGRESS,
        25 SUCCEEDED, 3 PYMC_ERROR, 7 OTHER_ERROR``.
    """
    parts: List[str] = [f"{run_info['total']} TOTAL"]
    if run_info["pending"] > 0:
        parts.append(f"{run_info['pending']} PENDING")
    if run_info["in_process"] > 0:
        parts.append(f"{run_info['in_process']} IN_PROGRESS")
    if run_info["succeeded"] > 0:
        parts.append(f"{run_info['succeeded']} SUCCEEDED")
    if run_info.get("pymc_error", 0) > 0:
        parts.append(f"{run_info['pymc_error']} PYMC_ERROR")
    if run_info.get("other_error", 0) > 0:
        parts.append(f"{run_info['other_error']} OTHER_ERROR")
    return ", ".join(parts)


def _print_error_summary(*, run_info: Dict[str, Any]) -> None:
    """Print per-dataset error details grouped by error class.

    Args:
        run_info: Dict from ``_scan_run``.
    """
    details: Dict[str, Tuple[str, str]] = run_info.get("error_details", {})
    if len(details) == 0:
        return

    # Group by error class.
    by_class: Dict[str, List[str]] = defaultdict(list)
    for ds_name, (cls, short) in sorted(details.items()):
        by_class[cls].append((ds_name, short))

    for cls_name in ("PYMC_ERROR", "OTHER_ERROR"):
        items: List[Tuple[str, str]] = by_class.get(cls_name, [])
        if len(items) == 0:
            continue
        print(f"\n  [{cls_name}] {len(items)} dataset(s):")
        for ds_name, short in items:
            # Show dataset name and the last ~120 chars of the error
            last_line: str = " ".join(short.split()[-30:]) if len(short) > 0 else "(no error string)"
            print(f"    {ds_name}: {last_line}")


# ---------------------------------------------------------------------------
#  Top-level reporting
# ---------------------------------------------------------------------------

outputs_root: str = os.path.abspath(_OUTPUTS_DIR)


def print_tree(
    *, experiments: List[Dict[str, Any]], show_errors: bool = False, latest_n: Optional[int] = None
) -> None:
    """Print a tree-style summary of all experiments.

    Args:
        experiments: List from ``_build_tree``.
        show_errors: If True, also print per-dataset error details.
        latest_n: If set, only show the last N runs per experiment.
    """
    if len(experiments) == 0:
        print("No experiments found under outputs/.")
        return

    for expt_idx, expt in enumerate(experiments):
        is_last_expt: bool = expt_idx == len(experiments) - 1
        connector: str = "" if is_last_expt else ""
        runs: List[Dict[str, Any]] = expt["runs"]
        if latest_n is not None:
            runs = runs[-latest_n:]
            print(f"{connector}{expt['name']} (latest: {latest_n})")
        else:
            print(f"{connector}{expt['name']}")

        for run_idx, run_info in enumerate(runs):
            is_last_run: bool = run_idx == len(runs) - 1
            branch: str = "\u2514\u2500\u2500 " if is_last_run else "\u251c\u2500\u2500 "
            counts_str: str = _format_counts(run_info)
            # Strip the experiment-name prefix from the path since it is already
            # the parent tree node.
            full_path: str = run_info["run_dir"]
            rel_from_root: str = full_path.replace(outputs_root + "/", "")
            expt_prefix: str = expt["name"] + "/"
            if rel_from_root.startswith(expt_prefix):
                rel_path: str = rel_from_root[len(expt_prefix) :]
            else:
                rel_path: str = rel_from_root
            print(f"{branch}{rel_path}: {counts_str}")

            if show_errors:
                _print_error_summary(run_info=run_info)

        if len(runs) == 0:
            print("  (no runs)")

        if is_last_expt is False:
            print()


def main() -> int:
    """Build and print the tree.

    Returns:
        0 on success.
    """
    show_errors: bool = "--errors" in sys.argv[1:]
    latest_n: Optional[int] = None
    i: int = sys.argv.index("--latest") if "--latest" in sys.argv else -1
    if i >= 0 and i < len(sys.argv) - 1:
        latest_n = int(sys.argv[i + 1])

    experiments: List[Dict[str, Any]] = _build_tree(outputs_root=outputs_root)
    print_tree(experiments=experiments, show_errors=show_errors, latest_n=latest_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
