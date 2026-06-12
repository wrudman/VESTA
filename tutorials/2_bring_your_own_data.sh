#!/usr/bin/env bash
#
# Tutorial 2: Run VESTA on your own data (CSV or Parquet).
#
# Mirrors the "Run VESTA on your own data" section of the top-level README.
#
# The distribution-fitting Harbor task loads its dataset from
#   harbor/tasks/vesta-distribution-fitting/data/data.parquet
# as a single column named "value" (one column of numeric observations).
# This tutorial converts YOUR file into that exact layout, then runs the task.
#
# Usage (from the repo root):
#   bash tutorials/2_bring_your_own_data.sh path/to/my_data.csv     value_col
#   bash tutorials/2_bring_your_own_data.sh path/to/my_data.parquet value_col
#
# If you omit the column name, VESTA uses the single column in the file.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK_PATH="harbor/tasks/vesta-distribution-fitting/"
TASK_DATA="$TASK_PATH/data/data.parquet"

SRC_FILE="${1:-}"
VALUE_COLUMN="${2:-}"

if [ -z "$SRC_FILE" ]; then
    echo "Usage: bash tutorials/2_bring_your_own_data.sh <my_data.csv|my_data.parquet> [value_column]"
    exit 1
fi
if [ ! -f .env ]; then
    echo "No .env found. Run tutorials/1_run_with_your_api_key.sh first to set up .env."
    exit 1
fi

# 1. Convert the user's CSV or Parquet into the task's "value"-column Parquet.
#    Both filetypes are supported by vesta.data loaders.
echo "Converting $SRC_FILE -> $TASK_DATA"
SRC_FILE="$SRC_FILE" VALUE_COLUMN="$VALUE_COLUMN" TASK_DATA="$TASK_DATA" python - <<'PY'
import os
from pathlib import Path

import pandas as pd

from vesta.data import load_distribution_csv, load_distribution_parquet

src = Path(os.environ["SRC_FILE"])
value_column = os.environ["VALUE_COLUMN"] or None
task_data = Path(os.environ["TASK_DATA"])

suffix = src.suffix.lower()
if suffix == ".csv":
    datasets = load_distribution_csv(src, value_column=value_column)
elif suffix == ".parquet":
    datasets = load_distribution_parquet(src, value_column=value_column)
else:
    raise ValueError(
        f"Unsupported file extension {suffix!r} for {src}. "
        f"Use a .csv or .parquet file."
    )

# The task expects exactly one dataset stored as a single 'value' column.
observations = datasets[0]["data"]
task_data.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"value": observations}).to_parquet(task_data, index=False)
print(f"Wrote {len(observations)} observations to {task_data}")
PY

# 2. Run the task on your data via Harbor.
CMD=(harbor run --path "$TASK_PATH" --agent oracle --env-file .env)
echo "+ ${CMD[*]}"
"${CMD[@]}"
