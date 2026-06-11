#!/usr/bin/env python
"""
Post-hoc LOO-CV / WAIC evaluation for PyVision baseline runs.

PyVision produces a single final model per sample (no iterative refinement),
stored in CSV files under pyvision-ts-runs/ with columns:
    idx, final_response, tool_calls

The model code is extracted from <final_pymc_model> tags in final_response.

Usage:
  python evaluate_pyvision_elpd.py pyvision-ts-runs/pyvision_gpt_output_single_ts_results_ts.csv \
      --dataset-pkl datasets_time_series/dataset_ts_easy_50.pkl

  python evaluate_pyvision_elpd.py pyvision-ts-runs/pyvision_claude_output_astro_chirp_ts_results_ts.csv \
      --dataset-pkl datasets_time_series/dataset_ts_astro_chirp_50.pkl \
      --max-obs 120 --draws 200 --tune 200 --chains 2 --cores 2

Output:
  Writes ts_evaluation_elpd.csv in --output-dir (default: <csv_stem>_elpd/)
  with the same schema as evaluate_ts_run_with_elpd_basis_ref.py,
  using eval_type='best' so notebooks can load it uniformly.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning)


# ── Model code extraction from PyVision responses ───────────────────────────

_FINAL_MODEL_RE = re.compile(
    r"<final_pymc_model>\s*(.*?)\s*</final_pymc_model>", re.DOTALL
)


def extract_model_from_response(response: str) -> dict | None:
    """Extract model code and kernel list from a PyVision final_response.

    Returns dict with keys: model_code, kernels
    or None if no valid model found.
    When extraction fails, returns dict with keys: model_code=None, extraction_error=<reason>.
    """
    if not response or not isinstance(response, str):
        return {"model_code": None, "kernels": "[]", "extraction_error": "empty or non-string response"}

    m = _FINAL_MODEL_RE.search(response)
    if m is None:
        return {"model_code": None, "kernels": "[]", "extraction_error": "no <final_pymc_model> tags found"}

    content = m.group(1).strip()
    lines = content.split("\n")

    # Extract kernels line
    kernels = "[]"
    for line in lines:
        if line.strip().startswith("kernels:"):
            kernels = line.strip()[len("kernels:"):].strip()
            break

    # Extract code after "code:" line
    code_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("code:"):
            code_start = i
            break

    if code_start is None:
        return {"model_code": None, "kernels": kernels, "extraction_error": "no 'code:' section in <final_pymc_model> block"}

    code = "\n".join(lines[code_start + 1:])
    if not code.strip():
        return {"model_code": None, "kernels": kernels, "extraction_error": "code section is empty"}

    # Patch compatibility issues in PyVision-generated code
    code = _patch_model_code(code)

    return {"model_code": code, "kernels": kernels}


def _patch_model_code(code: str) -> str:
    """Patch known compatibility issues in PyVision model code.

    - pm.MutableData / pm.ConstantData → pm.Data (renamed in PyMC v5+)
    - noise= → sigma= in marginal_likelihood (renamed in PyMC 5.x)
    - WhiteNoise(1) or WhiteNoise(input_dim=1) → WhiteNoise()
      (old API passed input_dim; new API only takes sigma)
    - Ensure marginal_likelihood has sigma= if missing
    """
    code = code.replace("pm.MutableData", "pm.Data")
    code = code.replace("pm.ConstantData", "pm.Data")

    # Fix noise= → sigma= in marginal_likelihood calls
    code = re.sub(
        r'(marginal_likelihood\([^)]*)\bnoise\s*=',
        r'\1sigma=',
        code,
    )

    # Fix WhiteNoise(1) / WhiteNoise(input_dim=1) → WhiteNoise()
    # The old API accepted input_dim as first positional arg; new API has sigma only
    code = re.sub(
        r'pm\.gp\.cov\.WhiteNoise\(\s*(?:input_dim\s*=\s*)?\d+\s*\)',
        'pm.gp.cov.WhiteNoise()',
        code,
    )

    # If marginal_likelihood call has no sigma=, add sigma=1e-6
    # Match lines like: gp.marginal_likelihood("name", X=X, y=y)
    # but NOT lines that already have sigma=
    code = re.sub(
        r'(\.marginal_likelihood\([^)]+)((?<!\bsigma=)[^)]*)\)',
        _add_sigma_if_missing,
        code,
    )

    return code


def _add_sigma_if_missing(match: re.Match) -> str:
    """Callback for re.sub: add sigma=1e-6 if not present."""
    full = match.group(0)
    if 'sigma=' in full:
        return full
    # Insert sigma=1e-6 before the closing paren
    return full[:-1] + ', sigma=1e-6)'


# ── WhiteNoise compatibility shim ───────────────────────────────────────────
# PyMC 5.28: WhiteNoise extends BaseCovariance (no input_dim), so it can't be
# composed with Covariance-based kernels via + or *. We provide a drop-in
# replacement that extends Covariance and can be used in kernel sums.

class _WhiteNoiseCompat(pm.gp.cov.Covariance):
    """WhiteNoise kernel compatible with Covariance-based composition."""

    def __init__(self, sigma=1.0, input_dim=1, active_dims=None):
        super().__init__(input_dim=input_dim, active_dims=active_dims)
        self._sigma = sigma

    def diag(self, X):
        import pytensor.tensor as pt
        return pt.ones(X.shape[0]) * self._sigma ** 2

    def full(self, X, Xs=None):
        import pytensor.tensor as pt
        if Xs is None:
            return pt.eye(X.shape[0]) * self._sigma ** 2
        else:
            return pt.zeros((X.shape[0], Xs.shape[0]))


# ── Reuse core functions from evaluate_ts_run_with_elpd_basis_ref ────────────

from evaluate_ts_run_with_elpd_basis_ref import (
    _average_refit_results,
    _compute_prediction_metrics,
    _max_subsample_offsets,
    _subsample_data,
    refit_baseline_model,
    strip_model_code_for_sampling,
)


def _make_patched_pm():
    """Create a patched copy of the pm.gp.cov module with WhiteNoise replaced."""
    import types

    # Create a proxy for pm.gp.cov that replaces WhiteNoise
    class _CovProxy(types.ModuleType):
        def __getattr__(self, name):
            if name == "WhiteNoise":
                return _WhiteNoiseCompat
            return getattr(pm.gp.cov, name)

    class _GpProxy(types.ModuleType):
        def __init__(self):
            super().__init__("pm.gp")
            self.cov = _CovProxy("pm.gp.cov")

        def __getattr__(self, name):
            if name == "cov":
                return self.cov
            return getattr(pm.gp, name)

    class _PmProxy(types.ModuleType):
        def __init__(self):
            super().__init__("pm")
            self.gp = _GpProxy()

        def __getattr__(self, name):
            if name == "gp":
                return self.gp
            return getattr(pm, name)

    return _PmProxy()


def refit_pyvision_with_mcmc(
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
    """Re-fit a PyVision GP model with MCMC and compute LOO/WAIC.

    Like refit_with_mcmc but injects WhiteNoise compatibility shim
    into the exec namespace for PyVision-generated code.
    """
    data = _subsample_data(data, max_obs, offset=subsample_offset)
    result = {
        "status": "error",
        "elpd_loo": np.nan, "elpd_loo_se": np.nan, "p_loo": np.nan,
        "elpd_waic": np.nan, "elpd_waic_se": np.nan, "p_waic": np.nan,
        "loo_good_k": np.nan, "loo_bad_k": np.nan,
        "n_obs": len(data),
        "r2": np.nan, "rmse": np.nan, "crps": np.nan, "aic_refit": np.nan,
        "error": None,
    }

    try:
        model_only_code = strip_model_code_for_sampling(model_code)
        pm_patched = _make_patched_pm()
        exec_ns = {"pm": pm_patched, "np": np, "pd": pd, "data": data}
        exec(model_only_code, exec_ns)  # noqa: S102

        model = exec_ns.get("model")
        if model is None:
            result["error"] = "No 'model' variable after exec"
            return result

        with model:
            trace = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept,
                nuts={"max_treedepth": 8},
                progressbar=True,
                discard_tuned_samples=True,
                idata_kwargs={"log_likelihood": True},
                cores=cores,
            )

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

        # Prediction metrics
        try:
            gp_obj = exec_ns.get("gp")
            X_pred = exec_ns.get("X_warp", exec_ns.get("X"))
            data_mean = float(np.mean(data.values))
            if gp_obj is not None and X_pred is not None:
                posterior_mean = {
                    var_name: float(trace.posterior[var_name].mean())
                    for var_name in trace.posterior.data_vars
                }
                with model:
                    mu_pred, _ = gp_obj.predict(X_pred, point=posterior_mean, diag=True)
                y_pred = mu_pred.flatten() + data_mean
                y_true = np.asarray(data.values, dtype=float)
                n_params = len(model.free_RVs)
                pred_metrics = _compute_prediction_metrics(y_true, y_pred, n_params)
                result.update(pred_metrics)
        except Exception:
            pass

    except Exception as e:
        import traceback
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
        description="Post-hoc LOO-CV / WAIC evaluation for PyVision baseline runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_file", type=Path,
        help="Path to the PyVision results CSV (e.g. pyvision_gpt_output_single_ts_results_ts.csv).",
    )
    parser.add_argument(
        "--dataset-pkl", type=Path, required=True,
        help="Path to the dataset pickle file (e.g. datasets_time_series/dataset_ts_easy_50.pkl).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for ts_evaluation_elpd.csv. Default: <csv_stem>_elpd/",
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

    csv_path: Path = args.csv_file.resolve()
    pkl_path: Path = args.dataset_pkl.resolve()

    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not pkl_path.exists():
        print(f"ERROR: Dataset pickle not found: {pkl_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = csv_path.parent / (csv_path.stem + "_elpd")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive dataset name from pkl filename
    dataset_name = pkl_path.stem  # e.g. "dataset_ts_easy_50"

    print(f"CSV file   : {csv_path}")
    print(f"Dataset pkl: {pkl_path}")
    print(f"Dataset    : {dataset_name}")
    print(f"Output dir : {out_dir}")
    print(f"MCMC: draws={args.draws}, tune={args.tune}, chains={args.chains}, cores={args.cores}")
    if args.max_obs:
        print(f"Subsampling to max {args.max_obs} obs, {args.n_subsample_reps} reps")
    print()

    # ── 1. Load CSV and dataset ──────────────────────────────────────────
    csv_df = pd.read_csv(csv_path)
    dataset = load_dataset(pkl_path)

    print(f"CSV has {len(csv_df)} rows, dataset has {len(dataset)} samples")

    # Filter samples
    if args.sample_indices is not None:
        idx_set = {int(x.strip()) for x in args.sample_indices.split(",")}
        csv_df = csv_df[csv_df["idx"].isin(idx_set)]
        print(f"Filtered to {len(csv_df)} sample(s) by --sample-indices")

    if args.max_samples is not None:
        csv_df = csv_df.head(args.max_samples)

    print(f"Processing {len(csv_df)} sample(s)")
    print()

    # ── 2. Check for existing results (resume support) ───────────────────
    out_csv = out_dir / "ts_evaluation_elpd.csv"
    done_indices: set[int] = set()
    existing_records: list[dict] = []
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        done_indices = set(existing_df["sample_idx"].unique())
        existing_records = existing_df.to_dict("records")
        print(f"Resuming: {len(done_indices)} sample(s) already done, skipping them")
        print()

    # ── 3. Process each sample ───────────────────────────────────────────
    records: list[dict] = list(existing_records)
    t_start = time.time()

    for i, row in enumerate(csv_df.itertuples()):
        sample_idx = int(row.idx)

        if sample_idx in done_indices:
            continue

        print(f"[{i + 1}/{len(csv_df)}] Sample {sample_idx}")

        # Extract model code
        model_info = extract_model_from_response(row.final_response)
        extraction_failed = model_info.get("model_code") is None
        extraction_error = model_info.get("extraction_error", "")

        if extraction_failed:
            print(f"  No valid model code found ({extraction_error}). Fitting baseline...")
            try:
                data = dataset[sample_idx]["data"]
            except (IndexError, KeyError) as e:
                print(f"  SKIP: cannot load data: {e}")
                records.append({
                    "dataset": dataset_name,
                    "sample_idx": sample_idx,
                    "label": "pyvision",
                    "eval_type": "best",
                    "model_structure": "no_model",
                    "aic": np.nan,
                    "status": "error",
                    "elpd_loo": np.nan, "elpd_loo_se": np.nan, "p_loo": np.nan,
                    "elpd_waic": np.nan, "elpd_waic_se": np.nan, "p_waic": np.nan,
                    "loo_good_k": np.nan, "loo_bad_k": np.nan, "n_obs": np.nan,
                    "r2": np.nan, "rmse": np.nan, "crps": np.nan, "aic_refit": np.nan,
                    "error": f"extraction_failed: {extraction_error}; also cannot load data: {e}",
                })
                continue

            baseline_result = refit_baseline_model(
                data=data, draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=args.target_accept, max_obs=args.max_obs,
            )
            records.append({
                "dataset": dataset_name,
                "sample_idx": sample_idx,
                "label": "pyvision",
                "eval_type": "best",
                "model_structure": "mean_baseline",
                "aic": np.nan,
                **{k: baseline_result[k] for k in [
                    "status", "elpd_loo", "elpd_loo_se", "p_loo",
                    "elpd_waic", "elpd_waic_se", "p_waic",
                    "loo_good_k", "loo_bad_k", "n_obs",
                    "r2", "rmse", "crps", "aic_refit",
                ]},
                "error": f"extraction_failed: {extraction_error}",
            })
            continue

        # Load data
        try:
            data = dataset[sample_idx]["data"]
        except (IndexError, KeyError) as e:
            print(f"  SKIP: cannot load data: {e}")
            records.append({
                "dataset": dataset_name,
                "sample_idx": sample_idx,
                "label": "pyvision",
                "eval_type": "best",
                "model_structure": model_info["kernels"],
                "aic": np.nan,
                "status": "data_error",
                "elpd_loo": np.nan, "elpd_loo_se": np.nan, "p_loo": np.nan,
                "elpd_waic": np.nan, "elpd_waic_se": np.nan, "p_waic": np.nan,
                "loo_good_k": np.nan, "loo_bad_k": np.nan, "n_obs": np.nan,
                "r2": np.nan, "rmse": np.nan, "crps": np.nan, "aic_refit": np.nan,
                "error": str(e),
            })
            continue

        n_obs_actual = len(data)
        n_obs_used = min(n_obs_actual, args.max_obs) if args.max_obs else n_obs_actual
        max_offsets = _max_subsample_offsets(n_obs_actual, args.max_obs) if args.max_obs else 1
        n_reps = min(args.n_subsample_reps, max_offsets) if args.max_obs and n_obs_actual > args.max_obs else 1

        print(f"  Kernels: {model_info['kernels']}")
        print(f"  Re-fitting with MCMC ({n_obs_used}/{n_obs_actual} obs, {n_reps} reps)...")

        t0 = time.time()
        runs = []
        for rep in range(n_reps):
            r = refit_pyvision_with_mcmc(
                model_code=model_info["model_code"],
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
        else:
            err_preview = str(result["error"] or "")[:200]
            print(f"  ALL FAILED ({elapsed:.1f}s): {err_preview}")
            # Preserve the original error from the failed model
            original_error = f"model_exec_failed: {err_preview}"
            # Fall back to baseline
            print(f"  Falling back to baseline model...")
            result = refit_baseline_model(
                data=data, draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=args.target_accept, max_obs=args.max_obs,
            )
            model_info["kernels"] = "mean_baseline"
            # Overwrite error with the original failure reason
            result["error"] = original_error

        records.append({
            "dataset": dataset_name,
            "sample_idx": sample_idx,
            "label": "pyvision",
            "eval_type": "best",
            "model_structure": model_info["kernels"],
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

    # ── 4. Final save and summary ────────────────────────────────────────
    results_df = pd.DataFrame(records)
    _save_results(records, out_csv)

    print()
    print("=" * 80)
    print("PYVISION ELPD EVALUATION RESULTS")
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
