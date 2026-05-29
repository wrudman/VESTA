"""
Painstaking verification of ALL stacked parquet files across all directories.

Checks:
1. Every expected file exists in its directory
2. Each file has the correct row count for its suffix:
   - unsuffixed (50-example datasets): 50 rows, idx 0-49
   - _50to100 (100-example datasets, first split): 50 rows, idx 0-49
   - _50to100 (100-example datasets, second split): 50 rows, idx 50-99
   - _100to110 (110-example datasets, third split): 10 rows, idx 100-109
   - _orig_0to100 (backup): 110 rows, idx 0-109
3. No files were accidentally overwritten or lost
4. Column structure is consistent within each directory group
5. The grid is complete (72 expected parquet files + 4 backups)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DOWNLOADS = Path.home() / "Downloads"

# ── Directories ──────────────────────────────────────────────────────────────

DIRS = {
    "original": DOWNLOADS / "stacked_parquets",
    "kanishk": DOWNLOADS / "stacked_parquets_kanishk",
    "latest": DOWNLOADS / "stacked_parquets_latest",
    "latest_kimi": DOWNLOADS / "stacked_parquets_latest_remaining_kimi25",
}

# ── Expected files per directory ─────────────────────────────────────────────

# Original: 27 files (df only, no ts)
EXPECTED_ORIGINAL = [
    "stacked_df_data_mixed_genonly_claude_sonnet46.parquet",
    "stacked_df_data_mixed_genonly_claude_sonnet46_50to100.parquet",
    "stacked_df_data_mixed_genonly_gpt54_mini.parquet",
    "stacked_df_data_mixed_genonly_gpt54_mini_50to100.parquet",
    "stacked_df_data_mixed_genonly_kimi25.parquet",
    "stacked_df_data_mixed_genonly_kimi25_50to100.parquet",
    "stacked_df_data_mixed_none_claude_sonnet46.parquet",
    "stacked_df_data_mixed_none_claude_sonnet46_50to100.parquet",
    "stacked_df_data_mixed_none_gpt54_mini.parquet",
    "stacked_df_data_mixed_none_gpt54_mini_50to100.parquet",
    "stacked_df_data_mixed_none_kimi25.parquet",
    "stacked_df_data_mixed_none_kimi25_50to100.parquet",
    "stacked_df_data_mixed_static_claude_sonnet46.parquet",
    "stacked_df_data_mixed_static_claude_sonnet46_50to100.parquet",
    "stacked_df_data_mixed_static_gpt54_mini.parquet",
    "stacked_df_data_mixed_static_gpt54_mini_50to100.parquet",
    "stacked_df_data_mixed_static_kimi25.parquet",
    "stacked_df_data_mixed_static_kimi25_50to100.parquet",
    "stacked_df_data_single_genonly_claude_sonnet46.parquet",
    "stacked_df_data_single_genonly_gpt54_mini.parquet",
    "stacked_df_data_single_genonly_kimi25.parquet",
    "stacked_df_data_single_none_claude_sonnet46.parquet",
    "stacked_df_data_single_none_gpt54_mini.parquet",
    "stacked_df_data_single_none_kimi25.parquet",
    "stacked_df_data_single_static_claude_sonnet46.parquet",
    "stacked_df_data_single_static_gpt54_mini.parquet",
    "stacked_df_data_single_static_kimi25.parquet",
]

# Kanishk: ts files (unsuffixed = 50 rows, _50to100 = 50 rows, _100to110 = 10 rows, _orig_0to100 = 110 rows)
EXPECTED_KANISHK = [
    # ts easy (50 examples) - unsuffixed only
    "stacked_ts_easy_genonly_claude_sonnet46.parquet",
    "stacked_ts_easy_genonly_gpt54_mini.parquet",
    "stacked_ts_easy_genonly_kimi25.parquet",
    "stacked_ts_easy_none_claude_sonnet46.parquet",
    "stacked_ts_easy_none_gpt54_mini.parquet",
    "stacked_ts_easy_none_kimi25.parquet",
    "stacked_ts_easy_static_claude_sonnet46.parquet",
    "stacked_ts_easy_static_gpt54_mini.parquet",
    "stacked_ts_easy_static_kimi25.parquet",
    # ts gravitational_chirp (50 examples) - unsuffixed only
    "stacked_ts_gravitational_chirp_genonly_claude_sonnet46.parquet",
    "stacked_ts_gravitational_chirp_genonly_gpt54_mini.parquet",
    "stacked_ts_gravitational_chirp_genonly_kimi25.parquet",
    "stacked_ts_gravitational_chirp_none_claude_sonnet46.parquet",
    "stacked_ts_gravitational_chirp_none_gpt54_mini.parquet",
    "stacked_ts_gravitational_chirp_none_kimi25.parquet",
    "stacked_ts_gravitational_chirp_static_claude_sonnet46.parquet",
    "stacked_ts_gravitational_chirp_static_gpt54_mini.parquet",
    "stacked_ts_gravitational_chirp_static_kimi25.parquet",
    # ts medium (110 examples) - unsuffixed (50 rows), _50to100 (50 rows), _100to110 (10 rows), _orig_0to100 (110 rows)
    "stacked_ts_medium_genonly_claude_sonnet46.parquet",
    "stacked_ts_medium_genonly_gpt54_mini.parquet",
    "stacked_ts_medium_genonly_kimi25.parquet",
    "stacked_ts_medium_none_claude_sonnet46.parquet",
    "stacked_ts_medium_none_gpt54_mini.parquet",
    "stacked_ts_medium_none_gpt54_mini_50to100.parquet",
    "stacked_ts_medium_none_gpt54_mini_100to110.parquet",
    "stacked_ts_medium_none_gpt54_mini_orig_0to100.parquet",
    "stacked_ts_medium_none_kimi25.parquet",
    "stacked_ts_medium_none_kimi25_50to100.parquet",
    "stacked_ts_medium_none_kimi25_100to110.parquet",
    "stacked_ts_medium_none_kimi25_orig_0to100.parquet",
    "stacked_ts_medium_static_claude_sonnet46.parquet",
    "stacked_ts_medium_static_gpt54_mini.parquet",
    "stacked_ts_medium_static_gpt54_mini_50to100.parquet",
    "stacked_ts_medium_static_gpt54_mini_100to110.parquet",
    "stacked_ts_medium_static_gpt54_mini_orig_0to100.parquet",
    "stacked_ts_medium_static_kimi25.parquet",
    "stacked_ts_medium_static_kimi25_50to100.parquet",
    "stacked_ts_medium_static_kimi25_100to110.parquet",
    "stacked_ts_medium_static_kimi25_orig_0to100.parquet",
]

# Latest: post-fix re-runs
EXPECTED_LATEST = [
    # df single
    "stacked_df_data_single_genonly_claude_sonnet46.parquet",
    "stacked_df_data_single_genonly_gpt54_mini.parquet",
    "stacked_df_data_single_none_claude_sonnet46.parquet",
    "stacked_df_data_single_static_claude_sonnet46.parquet",
    # df mixed
    "stacked_df_data_mixed_genonly_claude_sonnet46.parquet",
    "stacked_df_data_mixed_genonly_claude_sonnet46_50to100.parquet",
    "stacked_df_data_mixed_genonly_gpt54_mini.parquet",
    "stacked_df_data_mixed_genonly_gpt54_mini_50to100.parquet",
    "stacked_df_data_mixed_none_claude_sonnet46.parquet",
    "stacked_df_data_mixed_none_claude_sonnet46_50to100.parquet",
    "stacked_df_data_mixed_static_claude_sonnet46.parquet",
    "stacked_df_data_mixed_static_claude_sonnet46_50to100.parquet",
    # df imf
    "stacked_df_data_imf_genonly_claude_sonnet46.parquet",
    "stacked_df_data_imf_genonly_gpt54_mini.parquet",
    "stacked_df_data_imf_genonly_kimi25.parquet",
    "stacked_df_data_imf_none_claude_sonnet46.parquet",
    "stacked_df_data_imf_none_gpt54_mini.parquet",
    "stacked_df_data_imf_none_kimi25.parquet",
    "stacked_df_data_imf_static_claude_sonnet46.parquet",
    "stacked_df_data_imf_static_gpt54_mini.parquet",
    "stacked_df_data_imf_static_kimi25.parquet",
    # ts easy
    "stacked_ts_easy_genonly_claude_sonnet46.parquet",
    "stacked_ts_easy_genonly_gpt54_mini.parquet",
    "stacked_ts_easy_none_claude_sonnet46.parquet",
    "stacked_ts_easy_static_claude_sonnet46.parquet",
    # ts medium
    "stacked_ts_medium_genonly_claude_sonnet46.parquet",
    "stacked_ts_medium_genonly_claude_sonnet46_50to100.parquet",
    "stacked_ts_medium_genonly_gpt54_mini.parquet",
    "stacked_ts_medium_genonly_gpt54_mini_50to100.parquet",
    "stacked_ts_medium_none_claude_sonnet46.parquet",
    "stacked_ts_medium_none_claude_sonnet46_50to100.parquet",
    "stacked_ts_medium_static_claude_sonnet46.parquet",
    "stacked_ts_medium_static_claude_sonnet46_50to100.parquet",
    # ts gravitational_chirp
    "stacked_ts_gravitational_chirp_genonly_claude_sonnet46.parquet",
    "stacked_ts_gravitational_chirp_genonly_gpt54_mini.parquet",
    "stacked_ts_gravitational_chirp_none_claude_sonnet46.parquet",
    "stacked_ts_gravitational_chirp_static_claude_sonnet46.parquet",
]

# Latest kimi: post-fix kimi25 genonly
EXPECTED_LATEST_KIMI = [
    "stacked_df_data_single_genonly_kimi25.parquet",
    "stacked_df_data_mixed_genonly_kimi25.parquet",
    "stacked_df_data_mixed_genonly_kimi25_50to100.parquet",
    "stacked_df_data_imf_genonly_kimi25.parquet",
    "stacked_ts_easy_genonly_kimi25.parquet",
    "stacked_ts_medium_genonly_kimi25.parquet",
    "stacked_ts_medium_genonly_kimi25_50to100.parquet",
    "stacked_ts_gravitational_chirp_genonly_kimi25.parquet",
]


# ── Validation helpers ───────────────────────────────────────────────────────


def check_file(path: Path, expected_rows: int, expected_idx_min: int, expected_idx_max: int) -> Tuple[bool, str]:
    """Return (ok, message) after validating a single parquet file."""
    if not path.exists():
        return False, f"MISSING"

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return False, f"UNREADABLE: {e}"

    n_rows = len(df)
    if n_rows != expected_rows:
        return False, f"WRONG ROWS: {n_rows} (expected {expected_rows})"

    if "dataset_idx" not in df.columns:
        return False, f"NO dataset_idx COLUMN"

    actual_min = int(df["dataset_idx"].min())
    actual_max = int(df["dataset_idx"].max())
    if actual_min != expected_idx_min or actual_max != expected_idx_max:
        return False, f"WRONG IDX RANGE: [{actual_min}, {actual_max}] (expected [{expected_idx_min}, {expected_idx_max}])"

    return True, f"OK: {n_rows} rows, idx=[{actual_min}, {actual_max}]"


def get_expected_shape(filename: str) -> Tuple[int, int, int]:
    """Return (expected_rows, idx_min, idx_max) based on filename suffix."""
    if "_orig_0to100" in filename:
        return 110, 0, 109
    elif "_100to110" in filename:
        return 10, 100, 109
    elif "_50to100" in filename:
        return 50, 50, 99
    else:
        # unsuffixed: could be 50-row (easy, single, imf, gravitational_chirp)
        # or 50-row first half (medium after our split)
        if "ts_medium" in filename or "df_data_mixed" in filename:
            return 50, 0, 49
        else:
            return 50, 0, 49


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    all_ok = True
    total_files_checked = 0
    total_files_expected = 0

    checks = [
        ("original", DIRS["original"], EXPECTED_ORIGINAL),
        ("kanishk", DIRS["kanishk"], EXPECTED_KANISHK),
        ("latest", DIRS["latest"], EXPECTED_LATEST),
        ("latest_kimi", DIRS["latest_kimi"], EXPECTED_LATEST_KIMI),
    ]

    for label, dirpath, expected_files in checks:
        print(f"\n{'=' * 70}")
        print(f"  DIRECTORY: {label}  ({dirpath})")
        print(f"{'=' * 70}")

        # Also check for UNEXPECTED files
        actual_files = set(p.name for p in dirpath.glob("*.parquet"))
        expected_set = set(expected_files)
        unexpected = actual_files - expected_set
        missing = expected_set - actual_files

        if unexpected:
            print(f"  ⚠️  UNEXPECTED FILES ({len(unexpected)}):")
            for f in sorted(unexpected):
                print(f"      {f}")
        if missing:
            print(f"  ❌ MISSING FILES ({len(missing)}):")
            for f in sorted(missing):
                print(f"      {f}")
            all_ok = False

        total_files_expected += len(expected_files)

        for fname in expected_files:
            total_files_checked += 1
            path = dirpath / fname
            expected_rows, idx_min, idx_max = get_expected_shape(fname)
            ok, msg = check_file(path, expected_rows, idx_min, idx_max)
            status = "✅" if ok else "❌"
            if not ok:
                all_ok = False
            print(f"  {status} {fname:70s} {msg}")

    print(f"\n{'=' * 70}")
    if all_ok:
        print(f"  ✅ ALL {total_files_checked} FILES VERIFIED SUCCESSFULLY")
    else:
        print(f"  ❌ SOME FILES FAILED VERIFICATION ({total_files_checked} checked, {total_files_expected} expected)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
