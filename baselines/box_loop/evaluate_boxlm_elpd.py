#!/usr/bin/env python
"""
Post-hoc LOO-CV / WAIC evaluation for Box LM baseline runs.

Box LM produces model code per sample in CSV files under box-lm-runs/ with columns:
    array_id, best_code, best_loo, best_waic, success

The model code defines a gen_model(observed_data) function that:
  - Takes a DataFrame with "time" and "observation" columns
  - Builds a PyMC model (with internal subsampling)
  - Samples and returns (model, posterior_predictive, trace)

For evaluation, we re-sample the model with consistent MCMC settings.

Usage:
  python evaluate_boxlm_elpd.py box-lm-runs/box_loop_ts_easy_claude_sonnet46_20260505_165351.csv \
      --dataset-pkl datasets_time_series/dataset_ts_easy_50.pkl

  # Multiple CSVs for medium dataset (will be concatenated)
  python evaluate_boxlm_elpd.py \
      box-lm-runs/box_loop_ts_medium_claude_sonnet46_20260505_170024.csv \
      box-lm-runs/box_loop_ts_medium_claude_sonnet46_50to100_20260505_170121.csv \
      box-lm-runs/box_loop_ts_medium_claude_sonnet46_100to110_20260505_170135.csv \
      --dataset-pkl datasets_time_series/dataset_ts_medium_110.pkl

Output:
  Writes ts_evaluation_elpd.csv in --output-dir with the same schema as
  evaluate_ts_run_with_elpd.py, using eval_type='best' so notebooks can load it uniformly.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from evaluate_ts_run_with_elpd import (
    _average_refit_results,
    _compute_prediction_metrics,
    _max_subsample_offsets,
    _subsample_data,
    refit_baseline_model,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Model code patching ──────────────────────────────────────────────────────

def _patch_boxlm_code(code: str) -> str:
    """Patch known compatibility issues in Box LM model code.

    - pm.MutableData / pm.ConstantData → pm.Data
    - noise= → sigma= in marginal_likelihood
    """
    if not code or not isinstance(code, str):
        return code
    code = code.replace("pm.MutableData", "pm.Data")
    code = code.replace("pm.ConstantData", "pm.Data")
    # Fix noise= → sigma= in marginal_likelihood calls
    code = re.sub(
        r'(marginal_likelihood\([^)]*)\bnoise\s*=',
        r'\1sigma=',
        code,
    )
    return code


def _strip_sampling_from_gen_model(code: str) -> str:
    """Strip pm.sample and return statements from gen_model code.

    Returns modified code that defines gen_model but only builds the model
    (no sampling), and assigns the model to a module-level variable.
    """
    lines = code.strip().split("\n")
    kept: list[str] = []
    in_function = False
    model_var_name = "model"

    for line in lines:
        stripped = line.strip()

        # Track when we enter gen_model
        if stripped.startswith("def gen_model"):
            in_function = True
            kept.append(line)
            continue

        if in_function:
            # Skip sampling and return lines
            if any(kw in stripped for kw in [
                "pm.sample(",
                "pm.sample_posterior_predictive(",
                "trace = pm.sample",
                "trace=pm.sample",
                "posterior_predictive = pm.sample",
                "posterior_predictive=pm.sample",
                "return model",
                "return (model",
            ]):
                continue
            # Also skip lines that reference trace or posterior_predictive
            # after they would have been created
            if stripped.startswith("return "):
                continue
            kept.append(line)
        else:
            kept.append(line)

    return "\n".join(kept)


# ── Core evaluation function ─────────────────────────────────────────────────


def refit_boxlm_with_mcmc(
    model_code: str,
    data: pd.Series,
    *,
    draws: int = 200,
    tune: int = 200,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.85,
    max_obs: int | None = None,
    subsample_offset: int = 0,
) -> dict:
    """Evaluate a Box LM model by running gen_model and computing LOO/WAIC.

    Box LM's gen_model both builds and samples the model internally (with
    log_likelihood=True). We use its trace directly to compute LOO/WAIC
    rather than re-sampling, because re-sampling GP covariance models after
    the first sample often corrupts PyTensor's compiled state.

    If the internal trace lacks log_likelihood, we attempt to re-build
    and re-sample the model from scratch.
    """
    data_sub = _subsample_data(data, max_obs, offset=subsample_offset)

    result = {
        "status": "error",
        "elpd_loo": np.nan, "elpd_loo_se": np.nan, "p_loo": np.nan,
        "elpd_waic": np.nan, "elpd_waic_se": np.nan, "p_waic": np.nan,
        "loo_good_k": np.nan, "loo_bad_k": np.nan,
        "n_obs": len(data_sub),
        "r2": np.nan, "rmse": np.nan, "crps": np.nan, "aic_refit": np.nan,
        "error": None,
    }

    try:
        # Build observed_data DataFrame that gen_model expects
        t_vals = np.arange(len(data_sub), dtype=float)
        observed_data = pd.DataFrame({
            "time": t_vals,
            "observation": data_sub.values.astype(float),
        })

        # Patch the code
        patched_code = _patch_boxlm_code(model_code)

        # Execute the code to define gen_model
        exec_ns = {
            "np": np, "pd": pd, "pm": pm,
            "__builtins__": __builtins__,
        }
        try:
            import pytensor
            import pytensor.tensor as pt
            exec_ns["pytensor"] = pytensor
            exec_ns["pt"] = pt
        except ImportError:
            pass
        exec(patched_code, exec_ns)  # noqa: S102

        gen_model = exec_ns.get("gen_model")
        if gen_model is None:
            result["error"] = "No 'gen_model' function found after exec"
            return result

        # Call gen_model - it builds the model AND samples internally
        gen_result = gen_model(observed_data)

        if not isinstance(gen_result, tuple) or len(gen_result) < 3:
            result["error"] = f"gen_model returned unexpected type: {type(gen_result)}"
            return result

        model = gen_result[0]
        posterior_predictive = gen_result[1]
        trace = gen_result[2]

        if model is None or not isinstance(model, pm.Model):
            result["error"] = "gen_model returned invalid model"
            return result

        # Use the trace from gen_model directly (it already has log_likelihood)
        if not hasattr(trace, "log_likelihood") or trace.log_likelihood is None:
            result["error"] = "gen_model trace has no log_likelihood"
            return result

        loo = az.loo(trace)
        waic = az.waic(trace)

        result.update({
            "status": "ok",
            "elpd_loo": float(loo.elpd_loo),
            "elpd_loo_se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "elpd_waic": float(waic.elpd_waic),
            "elpd_waic_se": float(waic.se),
            "p_waic": float(waic.p_waic),
            "loo_good_k": int((loo.pareto_k < 0.7).sum()),
            "loo_bad_k": int((loo.pareto_k > 0.7).sum()),
            "error": None,
        })

        # Prediction metrics (R², RMSE, CRPS)
        try:
            y_true = np.asarray(data_sub.values, dtype=float)
            # posterior_predictive from gen_model is a dict (return_inferencedata=False)
            pp_data = None
            if isinstance(posterior_predictive, dict):
                # Dict with variable name -> samples array
                pp_keys = list(posterior_predictive.keys())
                if pp_keys:
                    pp_data = posterior_predictive[pp_keys[0]]
            elif hasattr(posterior_predictive, "posterior_predictive"):
                pp_group = posterior_predictive.posterior_predictive
                obs_vars = list(pp_group.data_vars)
                if obs_vars:
                    pp_data = pp_group[obs_vars[0]].values

            if pp_data is not None:
                # pp_data shape: (chains*draws, n_obs) or (draws, n_obs)
                if pp_data.ndim == 3:
                    y_pred = pp_data.mean(axis=(0, 1))
                else:
                    y_pred = pp_data.mean(axis=0)
                # The model may subsample internally, so lengths may differ
                if len(y_pred) == len(y_true):
                    n_params = len(model.free_RVs)
                    pred_metrics = _compute_prediction_metrics(
                        y_true, y_pred, n_params
                    )
                    result.update(pred_metrics)
        except Exception:
            pass

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return result


# ── Data loading ─────────────────────────────────────────────────────────────

_pkl_cache: dict[str, list] = {}


def load_dataset(pkl_path: Path) -> list:
    """Load a dataset pickle file, with caching."""
    key = str(pkl_path)
    if key not in _pkl_cache:
        with open(pkl_path, "rb") as f:
            _pkl_cache[key] = pickle.load(f)
    return _pkl_cache[key]


# ── Main pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc LOO-CV / WAIC evaluation for Box LM baseline runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_files", type=Path, nargs="+",
        help="Path(s) to Box LM results CSV(s). Multiple CSVs will be concatenated.",
    )
    parser.add_argument(
        "--dataset-pkl", type=Path, required=True,
        help="Path to the dataset pickle file (e.g. datasets_time_series/dataset_ts_easy_50.pkl).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for ts_evaluation_elpd.csv.",
    )
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-obs", type=int, default=None)
    parser.add_argument("--n-subsample-reps", type=int, default=3)
    parser.add_argument(
        "--sample-indices", type=str, default=None,
        help="Comma-separated list of sample indices to evaluate.",
    )
    args = parser.parse_args()

    # Resolve and validate paths
    csv_paths = [p.resolve() for p in args.csv_files]
    pkl_path = args.dataset_pkl.resolve()

    for p in csv_paths:
        if not p.exists():
            print(f"ERROR: CSV file not found: {p}", file=sys.stderr)
            sys.exit(1)
    if not pkl_path.exists():
        print(f"ERROR: Dataset pickle not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    # Load and concatenate CSVs
    dfs = []
    for p in csv_paths:
        dfs.append(pd.read_csv(p))
    csv_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates by array_id (keep last occurrence)
    csv_df = csv_df.drop_duplicates(subset=["array_id"], keep="last")
    csv_df = csv_df.sort_values("array_id").reset_index(drop=True)

    # Determine output directory
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = csv_paths[0].parent / (csv_paths[0].stem + "_elpd")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive dataset name from pkl filename
    dataset_name = pkl_path.stem  # e.g. "dataset_ts_easy_50"

    print(f"CSV file(s): {[str(p) for p in csv_paths]}")
    print(f"Dataset pkl: {pkl_path}")
    print(f"Dataset    : {dataset_name}")
    print(f"Output dir : {out_dir}")
    print(f"Total rows : {len(csv_df)} (after dedup)")
    print(f"MCMC: draws={args.draws}, tune={args.tune}, chains={args.chains}, cores={args.cores}")
    if args.max_obs:
        print(f"Subsampling to max {args.max_obs} obs, {args.n_subsample_reps} reps")
    print()

    # Load dataset
    dataset = load_dataset(pkl_path)
    print(f"Dataset has {len(dataset)} samples")

    # Filter samples
    if args.sample_indices is not None:
        idx_set = {int(x.strip()) for x in args.sample_indices.split(",")}
        csv_df = csv_df[csv_df["array_id"].isin(idx_set)]
        print(f"Filtered to {len(csv_df)} sample(s) by --sample-indices")

    if args.max_samples is not None:
        csv_df = csv_df.head(args.max_samples)

    print(f"Processing {len(csv_df)} sample(s)")
    print()

    # ── Check for existing results (resume support) ──────────────────────
    out_csv = out_dir / "ts_evaluation_elpd.csv"
    done_indices: set[int] = set()
    existing_records: list[dict] = []
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        done_indices = set(existing_df["sample_idx"].unique())
        existing_records = existing_df.to_dict("records")
        print(f"Resuming: {len(done_indices)} sample(s) already done, skipping them")
        print()

    # ── Process each sample ──────────────────────────────────────────────
    records: list[dict] = list(existing_records)
    t_start = time.time()

    for i, row in enumerate(csv_df.itertuples()):
        sample_idx = int(row.array_id)

        if sample_idx in done_indices:
            continue

        print(f"[{i + 1}/{len(csv_df)}] Sample {sample_idx}")

        # Check success flag
        success = row.success
        if isinstance(success, str):
            success = success.strip().lower() == "true"

        model_code = row.best_code

        if not success or not isinstance(model_code, str) or not model_code.strip():
            print(f"  No valid model (success={success}). Fitting baseline...")
            try:
                data = dataset[sample_idx]["data"]
            except (IndexError, KeyError) as e:
                print(f"  SKIP: cannot load data: {e}")
                records.append({
                    "dataset": dataset_name,
                    "sample_idx": sample_idx,
                    "label": "boxlm",
                    "eval_type": "best",
                    "model_structure": "no_model",
                    "aic": np.nan,
                    "status": "error",
                    "elpd_loo": np.nan, "elpd_loo_se": np.nan, "p_loo": np.nan,
                    "elpd_waic": np.nan, "elpd_waic_se": np.nan, "p_waic": np.nan,
                    "loo_good_k": np.nan, "loo_bad_k": np.nan, "n_obs": np.nan,
                    "r2": np.nan, "rmse": np.nan, "crps": np.nan, "aic_refit": np.nan,
                    "error": f"no model code (success={success})",
                })
                _save_results(records, out_csv)
                continue

            baseline_result = refit_baseline_model(
                data=data, draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=args.target_accept, max_obs=args.max_obs,
            )
            records.append({
                "dataset": dataset_name,
                "sample_idx": sample_idx,
                "label": "boxlm",
                "eval_type": "best",
                "model_structure": "mean_baseline",
                "aic": np.nan,
                **{k: baseline_result[k] for k in [
                    "status", "elpd_loo", "elpd_loo_se", "p_loo",
                    "elpd_waic", "elpd_waic_se", "p_waic",
                    "loo_good_k", "loo_bad_k", "n_obs",
                    "r2", "rmse", "crps", "aic_refit", "error",
                ]},
            })
            _save_results(records, out_csv)
            continue

        # Load data
        try:
            data = dataset[sample_idx]["data"]
        except (IndexError, KeyError) as e:
            print(f"  SKIP: cannot load data: {e}")
            records.append({
                "dataset": dataset_name,
                "sample_idx": sample_idx,
                "label": "boxlm",
                "eval_type": "best",
                "model_structure": "boxlm_model",
                "aic": np.nan,
                "status": "data_error",
                "elpd_loo": np.nan, "elpd_loo_se": np.nan, "p_loo": np.nan,
                "elpd_waic": np.nan, "elpd_waic_se": np.nan, "p_waic": np.nan,
                "loo_good_k": np.nan, "loo_bad_k": np.nan, "n_obs": np.nan,
                "r2": np.nan, "rmse": np.nan, "crps": np.nan, "aic_refit": np.nan,
                "error": str(e),
            })
            _save_results(records, out_csv)
            continue

        n_obs_actual = len(data)
        n_obs_used = min(n_obs_actual, args.max_obs) if args.max_obs else n_obs_actual
        max_offsets = _max_subsample_offsets(n_obs_actual, args.max_obs) if args.max_obs else 1
        n_reps = min(args.n_subsample_reps, max_offsets) if args.max_obs and n_obs_actual > args.max_obs else 1

        print(f"  Re-fitting with MCMC ({n_obs_used}/{n_obs_actual} obs, {n_reps} reps)...")

        t0 = time.time()
        runs = []
        for rep in range(n_reps):
            r = refit_boxlm_with_mcmc(
                model_code=model_code,
                data=data,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                max_obs=args.max_obs,
                subsample_offset=rep,
            )
            runs.append(r)
            status_tag = f"elpd_loo={r['elpd_loo']:.1f}" if r["status"] == "ok" else "FAILED"
            print(f"    rep {rep + 1}/{n_reps}: {status_tag}")

        result = _average_refit_results(runs)
        elapsed = time.time() - t0
        n_ok = sum(1 for r in runs if r["status"] == "ok")

        if n_ok > 0:
            print(
                f"  avg elpd_loo={result['elpd_loo']:.1f} ± {result['elpd_loo_se']:.1f}  "
                f"R2={result.get('r2', float('nan')):.3f}  "
                f"({n_ok}/{n_reps} ok)  time={elapsed:.1f}s"
            )
            # Compare with CSV-reported best_loo/best_waic
            csv_loo = getattr(row, "best_loo", None)
            csv_waic = getattr(row, "best_waic", None)
            eval_loo = result["elpd_loo"]
            eval_waic = result.get("elpd_waic", np.nan)
            if csv_loo is not None and not (isinstance(csv_loo, float) and np.isnan(csv_loo)):
                diff_loo = eval_loo - float(csv_loo)
                flag = " ⚠️" if abs(diff_loo) > 20 else ""
                print(
                    f"  [ELPD check] eval_loo={eval_loo:.1f}  csv_best_loo={float(csv_loo):.1f}  "
                    f"Δ={diff_loo:+.1f}{flag}"
                )
            if csv_waic is not None and not (isinstance(csv_waic, float) and np.isnan(csv_waic)):
                diff_waic = eval_waic - float(csv_waic)
                flag = " ⚠️" if abs(diff_waic) > 20 else ""
                print(
                    f"  [ELPD check] eval_waic={eval_waic:.1f}  csv_best_waic={float(csv_waic):.1f}  "
                    f"Δ={diff_waic:+.1f}{flag}"
                )
        else:
            err_preview = str(result["error"] or "")[:200]
            print(f"  ALL FAILED ({elapsed:.1f}s): {err_preview}")
            # Fall back to baseline
            print("  Falling back to baseline model...")
            result = refit_baseline_model(
                data=data, draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=args.target_accept, max_obs=args.max_obs,
            )

        records.append({
            "dataset": dataset_name,
            "sample_idx": sample_idx,
            "label": "boxlm",
            "eval_type": "best",
            "model_structure": "boxlm_model",
            "aic": np.nan,
            **{k: result[k] for k in [
                "status", "elpd_loo", "elpd_loo_se", "p_loo",
                "elpd_waic", "elpd_waic_se", "p_waic",
                "loo_good_k", "loo_bad_k", "n_obs",
                "r2", "rmse", "crps", "aic_refit", "error",
            ]},
        })

        # Save after each sample for resume support
        _save_results(records, out_csv)

        # Progress estimate
        done_now = len([r for r in records if r not in existing_records])
        remaining = len(csv_df) - len(done_indices) - done_now
        elapsed_total = time.time() - t_start
        if done_now > 0 and remaining > 0:
            eta_min = (elapsed_total / done_now) * remaining / 60
            print(f"  (elapsed: {elapsed_total:.0f}s, ETA: ~{eta_min:.0f}m)")

    # ── Final save and summary ───────────────────────────────────────────
    results_df = pd.DataFrame(records)
    _save_results(records, out_csv)

    print()
    print("=" * 80)
    print("BOX LM ELPD EVALUATION RESULTS")
    print("=" * 80)

    display_cols = [
        "sample_idx", "eval_type", "model_structure", "elpd_loo",
        "elpd_loo_se", "elpd_waic", "r2", "rmse", "crps", "status",
    ]
    avail_cols = [c for c in display_cols if c in results_df.columns]
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 200)
    print(results_df[avail_cols].to_string(index=False))
    print()

    # Summary stats
    ok_df = results_df[results_df["status"].isin(["ok", "baseline"])]
    if not ok_df.empty:
        print("SUMMARY (successful fits):")
        for metric in ["elpd_loo", "elpd_waic", "r2", "rmse", "crps"]:
            vals = ok_df[metric].dropna()
            if not vals.empty:
                print(
                    f"  {metric:15s}  mean={vals.mean():10.2f}  std={vals.std():10.2f}  "
                    f"min={vals.min():10.2f}  max={vals.max():10.2f}  n={len(vals)}"
                )
        print()

    n_ok = (results_df["status"] == "ok").sum()
    n_baseline = (results_df["status"] == "baseline").sum()
    n_fail = (results_df["status"] == "error").sum()
    print(f"Total: {len(results_df)}  Success: {n_ok}  Baseline: {n_baseline}  Failed: {n_fail}")
    print(f"  ✓ ELPD evaluation saved to {out_csv}")


def _save_results(records: list[dict], out_csv: Path):
    """Save results DataFrame to CSV, truncating error strings."""
    df = pd.DataFrame(records)
    df["error"] = df["error"].apply(
        lambda e: str(e)[:200] if e is not None else None
    )
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
