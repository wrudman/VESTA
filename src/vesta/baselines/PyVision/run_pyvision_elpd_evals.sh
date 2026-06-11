#!/usr/bin/env bash
# Run PyVision ELPD evaluations for all 6 CSV files (GPT + Claude × 3 datasets).
# Same MCMC settings as the main eval runs.
set -euo pipefail
cd "$(dirname "$0")"

DRAWS=200
TUNE=200
CHAINS=2
CORES=2
MAX_OBS=150
N_REPS=3

PYVISION_DIR="pyvision-ts-runs"
DATASETS_DIR="datasets_time_series"

# CSV → dataset mapping
declare -a CSV_FILES=(
    "pyvision_gpt_output_single_ts_results_ts.csv:dataset_ts_easy_50.pkl"
    "pyvision_gpt_output_medium_ts_results_ts.csv:dataset_ts_medium_110.pkl"
    "pyvision_gpt_output_chirp_ts_results_ts.csv:dataset_ts_gravitational_chirp_50.pkl"
    "pyvision_claude_output_easy_ts_results_ts.csv:dataset_ts_easy_50.pkl"
    "pyvision_claude_output_medium_ts_results_ts.csv:dataset_ts_medium_110.pkl"
    "pyvision_claude_output_chirp_ts_results_ts.csv:dataset_ts_gravitational_chirp_50.pkl"
)

TOTAL=${#CSV_FILES[@]}
START_TIME=$SECONDS

for i in "${!CSV_FILES[@]}"; do
    IFS=: read -r csv_file pkl_file <<< "${CSV_FILES[$i]}"
    idx=$((i + 1))

    csv_path="${PYVISION_DIR}/${csv_file}"
    pkl_path="${DATASETS_DIR}/${pkl_file}"
    out_dir="${PYVISION_DIR}/${csv_file%.csv}_elpd"

    echo ""
    echo "========================================================================"
    echo "[${idx}/${TOTAL}] ${csv_file}"
    echo "  Dataset: ${pkl_file}"
    echo "  Output:  ${out_dir}"
    echo "========================================================================"

    elapsed=$((SECONDS - START_TIME))
    if [ "$i" -gt 0 ]; then
        per_file=$((elapsed / i))
        remaining=$(( (TOTAL - i) * per_file / 60 ))
        echo "  (elapsed: ${elapsed}s, ETA: ~${remaining}m)"
    fi

    conda run -n vesta python evaluate_pyvision_elpd.py \
        "${csv_path}" \
        --dataset-pkl "${pkl_path}" \
        --output-dir "${out_dir}" \
        --draws ${DRAWS} \
        --tune ${TUNE} \
        --chains ${CHAINS} \
        --cores ${CORES} \
        --max-obs ${MAX_OBS} \
        --n-subsample-reps ${N_REPS} \
        --target-accept 0.85

    echo "  ✓ Done: ${csv_file}"
done

echo ""
echo "========================================================================"
echo "All ${TOTAL} PyVision ELPD evaluations complete!"
echo "Total time: $(( (SECONDS - START_TIME) / 60 ))m"
echo "========================================================================"
