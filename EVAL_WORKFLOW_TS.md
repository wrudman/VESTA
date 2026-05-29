# Evaluation Workflow (Runs -> Evals -> Summary)

This document describes the standard process for generating evaluation outputs and summary metrics for time-series runs.

## Prerequisites

1. You are in the repository root.
2. Your Python environment is active.
3. Raw run folders are present in `outputs/` with normalized structure, for example:
   - `outputs/claude_ts_runs_new/`
   - `outputs/gpt_ts_runs_new/`
   - `outputs/kimi_ts_runs_new/`

## Step 1: Generate Eval Folders From Raw Runs

Use the unified script:

```bash
./generate_all_model_evals.sh all
```

You can also run per model:

```bash
./generate_all_model_evals.sh claude
./generate_all_model_evals.sh gpt
./generate_all_model_evals.sh kimi
```

This generates model-specific eval directories:

- `claude_ts_runs_eval_new/`
- `gpt_ts_runs_eval/`
- `kimi_ts_runs_eval/`

Each eval subfolder contains:

- `ts_evaluation_results.csv`
- `ts_evaluation_best_per_sample.csv`
- `eval_output.txt`
- optional boxplot PNG files

## Step 2: Summarize Metrics in Notebook

Open and run all cells in:

- `eval_summary_tables.ipynb`

The notebook reads the eval folders and generates aggregate metric summaries (for example CRPS and R2-based summaries), plus comparison tables/plots.

## Recommended Validation

Before running the notebook, quickly verify eval output exists:

```bash
ls claude_ts_runs_eval_new
ls gpt_ts_runs_eval
ls kimi_ts_runs_eval
```

If any folder is missing or incomplete, rerun Step 1 for that model.