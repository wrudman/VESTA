#!/bin/bash
set -euo pipefail

# Box Loop Time Series example runner.
#
# Required env vars by model:
# - Azure GPT: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
# - OpenRouter Claude/Kimi: OPENROUTER_API_KEY

DATA="../../dataset_time_series/dataset_ts_easy_50.pkl"
OUTPUT="outputs/box_loop_ts_easy_claude_sonnet46.pkl"
MODEL="openrouter/anthropic/claude-sonnet-4.6"
ROUNDS=5
DATASET_IDX="0:50"
NPROC=0
MAX_RPM=30

python run.py \
    --data "$DATA" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --rounds "$ROUNDS" \
    --dataset-idx "$DATASET_IDX" \
    --nproc "$NPROC" \
    --parallel.max-rpm "$MAX_RPM" \
    --task "box_loop_ts"
