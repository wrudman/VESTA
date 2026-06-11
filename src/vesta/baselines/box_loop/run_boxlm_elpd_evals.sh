#!/usr/bin/env bash
# Run Box LM ELPD evaluations for all 9 combinations (3 LLMs × 3 datasets).
# Same MCMC settings as the main eval runs.
set -euo pipefail
cd "$(dirname "$0")"

DRAWS=200
TUNE=200
CHAINS=2
CORES=2
MAX_OBS=150
N_REPS=3

BOXLM_DIR="box-lm-runs"
DATASETS_DIR="datasets_time_series"
OUTPUT_BASE="evals"
PYTHON="${PYTHON:-python}"

# ── Easy dataset (50 samples) ───────────────────────────────────────────────
declare -a EASY_RUNS=(
    "claude:box_loop_ts_easy_claude_sonnet46_20260505_165351.csv"
    "gpt:box_loop_ts_easy_gpt54_mini_20260505_170007.csv"
    "kimi:box_loop_ts_easy_kimi25_20260505_165959.csv"
)

# ── Gravitational chirp dataset (50 samples) ────────────────────────────────
declare -a CHIRP_RUNS=(
    "claude:box_loop_ts_gravitational_chirp_claude_sonnet46_20260505_170640.csv"
    "gpt:box_loop_ts_gravitational_chirp_gpt54_mini_20260505_170718.csv"
    "kimi:box_loop_ts_gravitational_chirp_kimi25_20260505_170654.csv"
)

# ── Medium dataset (110 samples, split across multiple CSVs) ────────────────
# Format: llm:csv1,csv2,csv3
declare -a MEDIUM_RUNS=(
    "claude:box_loop_ts_medium_claude_sonnet46_20260505_170024.csv,box_loop_ts_medium_claude_sonnet46_50to100_20260505_170121.csv,box_loop_ts_medium_claude_sonnet46_100to110_20260505_170135.csv"
    "gpt:box_loop_ts_medium_gpt54_mini_20260505_170410.csv,box_loop_ts_medium_gpt54_mini_50to100_20260505_170422.csv,box_loop_ts_medium_gpt54_mini_100to110_20260505_170430.csv"
    "kimi:box_loop_ts_medium_kimi25_20260505_170150.csv,box_loop_ts_medium_kimi25_50to100_20260505_170207.csv,box_loop_ts_medium_kimi25_100to110_20260505_170222.csv"
)

TOTAL=9
IDX=0
START_TIME=$SECONDS

mkdir -p "${OUTPUT_BASE}"

# ── Function to run a single evaluation ─────────────────────────────────────
run_eval() {
    local llm="$1"
    local dataset_name="$2"
    local pkl_file="$3"
    local out_dir="$4"
    shift 4
    local csv_args=("$@")

    IDX=$((IDX + 1))
    echo ""
    echo "========================================================================"
    echo "[${IDX}/${TOTAL}] Box LM ${llm} - ${dataset_name}"
    echo "  Dataset: ${pkl_file}"
    echo "  Output:  ${out_dir}"
    echo "========================================================================"

    elapsed=$((SECONDS - START_TIME))
    if [ "$IDX" -gt 1 ]; then
        per_run=$((elapsed / (IDX - 1)))
        remaining=$(( (TOTAL - IDX + 1) * per_run / 60 ))
        echo "  (elapsed: ${elapsed}s, ETA: ~${remaining}m)"
    fi

    mkdir -p "${out_dir}"

    "$PYTHON" evaluate_boxlm_elpd.py \
        "${csv_args[@]}" \
        --dataset-pkl "${DATASETS_DIR}/${pkl_file}" \
        --output-dir "${out_dir}" \
        --draws ${DRAWS} \
        --tune ${TUNE} \
        --chains ${CHAINS} \
        --cores ${CORES} \
        --max-obs ${MAX_OBS} \
        --n-subsample-reps ${N_REPS} \
        --target-accept 0.85

    echo "  ✓ Done: Box LM ${llm} - ${dataset_name}"
}

# ── Easy dataset runs ────────────────────────────────────────────────────────
for entry in "${EASY_RUNS[@]}"; do
    IFS=: read -r llm csv_file <<< "${entry}"
    run_eval "${llm}" "easy_50" "dataset_ts_easy_50.pkl" \
        "${OUTPUT_BASE}/boxlm_${llm}_easy_50" \
        "${BOXLM_DIR}/${csv_file}"
done

# ── Chirp dataset runs ───────────────────────────────────────────────────────
for entry in "${CHIRP_RUNS[@]}"; do
    IFS=: read -r llm csv_file <<< "${entry}"
    run_eval "${llm}" "gravitational_chirp_50" "dataset_ts_gravitational_chirp_50.pkl" \
        "${OUTPUT_BASE}/boxlm_${llm}_gravitational_chirp_50" \
        "${BOXLM_DIR}/${csv_file}"
done

# ── Medium dataset runs (multiple CSVs per LLM) ─────────────────────────────
for entry in "${MEDIUM_RUNS[@]}"; do
    IFS=: read -r llm csv_list <<< "${entry}"
    # Split comma-separated CSV list into array
    IFS=, read -ra csv_files <<< "${csv_list}"
    csv_args=()
    for f in "${csv_files[@]}"; do
        csv_args+=("${BOXLM_DIR}/${f}")
    done
    run_eval "${llm}" "medium_110" "dataset_ts_medium_110.pkl" \
        "${OUTPUT_BASE}/boxlm_${llm}_medium_110" \
        "${csv_args[@]}"
done

echo ""
echo "========================================================================"
echo "All ${TOTAL} Box LM ELPD evaluations complete!"
echo "Total time: $(( (SECONDS - START_TIME) / 60 ))m (${SECONDS}s)"
echo "========================================================================"
