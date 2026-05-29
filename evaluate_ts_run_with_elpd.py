#!/usr/bin/env python
"""
Post-hoc LOO-CV / WAIC evaluation for time-series GP model-selection runs.

For each sample's best model (by R²), re-fits the model with full MCMC
(pm.sample) and computes PSIS-LOO-CV and WAIC via arviz.

Outputs:
  a) Per-sample ELPD metrics table (CSV).
  b) Summary statistics across all samples.

Usage:
  python evaluate_ts_run_with_elpd.py outputs/claude_ts_runs_new/ts_genonly_forced_dataset_ts_easy_50
  python evaluate_ts_run_with_elpd.py outputs/claude_ts_runs_new/ts_genonly_forced_dataset_ts_easy_50 --draws 500 --tune 500
  python evaluate_ts_run_with_elpd.py outputs/claude_ts_runs_new/ts_genonly_forced_dataset_ts_easy_50 --max-samples 5
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

# Suppress noisy sampling output
warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_DIR_RE = re.compile(r"^(\d+)_(.+)$")


# ── Discovery (mirrors evaluate_ts_run.py) ──────────────────────────────────


def discover_samples(run_dir: Path) -> pd.DataFrame:
    """Return a DataFrame of (dataset, sample_idx, label, parquet_path, data_pkl)."""
    rows: list[dict] = []
    flat_cfg_path = run_dir / "config.json"
    if flat_cfg_path.exists():
        cfg = json.loads(flat_cfg_path.read_text())
        data_pkl = cfg.get("data_pkl")
        for sample_dir in sorted(run_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            m = SAMPLE_DIR_RE.match(sample_dir.name)
            if m is None:
                continue
            parquet = sample_dir / "run_log.parquet"
            if not parquet.exists():
                continue
            rows.append(
                {
                    "dataset": run_dir.name,
                    "sample_idx": int(m.group(1)),
                    "label": m.group(2),
                    "parquet_path": parquet,
                    "data_pkl": data_pkl,
                }
            )
        return pd.DataFrame(rows)

    for ds_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        cfg_path = ds_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        data_pkl = cfg.get("data_pkl")
        for sample_dir in sorted(ds_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            m = SAMPLE_DIR_RE.match(sample_dir.name)
            if m is None:
                continue
            parquet = sample_dir / "run_log.parquet"
            if not parquet.exists():
                continue
            rows.append(
                {
                    "dataset": ds_dir.name,
                    "sample_idx": int(m.group(1)),
                    "label": m.group(2),
                    "parquet_path": parquet,
                    "data_pkl": data_pkl,
                }
            )
    return pd.DataFrame(rows)


# ── Data loading ─────────────────────────────────────────────────────────────

_pkl_cache: dict[str, list] = {}


def resolve_pkl_path(pkl_name: str, data_dir: Path) -> Path:
    """Resolve a pkl path, handling common path mismatches."""
    candidate = data_dir / pkl_name
    if candidate.exists():
        return candidate
    # Try common alternatives: dataset_time_series -> datasets_time_series
    alt = pkl_name.replace("dataset_time_series/", "datasets_time_series/")
    candidate = data_dir / alt
    if candidate.exists():
        return candidate
    # Try just the filename in data_dir
    candidate = data_dir / Path(pkl_name).name
    if candidate.exists():
        return candidate
    # Try datasets_time_series subdir
    candidate = data_dir / "datasets_time_series" / Path(pkl_name).name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find dataset pickle '{pkl_name}' relative to '{data_dir}'. "
        f"Tried: {pkl_name}, {alt}"
    )


def load_pickle(pkl_name: str, data_dir: Path) -> list:
    pkl_path = resolve_pkl_path(pkl_name, data_dir)
    key = str(pkl_path)
    if key not in _pkl_cache:
        with open(pkl_path, "rb") as f:
            _pkl_cache[key] = pickle.load(f)
    return _pkl_cache[key]


def _safe_json(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


# ── Model code preparation ──────────────────────────────────────────────────


def strip_model_code_for_sampling(code: str) -> str:
    """Strip post-model lines (find_MAP, predict, trend) from the model code.

    Keeps everything up to and including gp.marginal_likelihood(...).
    The code is meant to be exec'd to produce a `model` variable in scope.
    """
    lines = code.strip().split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Stop before MAP, predict, or trend lines
        if any(
            kw in stripped
            for kw in [
                "pm.find_MAP",
                "find_MAP",
                "gp.predict",
                "trend =",
                "trend=",
            ]
        ):
            break
        kept.append(line)
    return "\n".join(kept)


# ── Prediction metrics ───────────────────────────────────────────────────────


def _compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict:
    """Compute R², RMSE, CRPS, AIC from true and predicted values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    # R²
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # CRPS (Gaussian, closed-form)
    resid = y_true - y_pred
    sigma = max(rmse, 1e-10)
    z = resid / sigma
    crps = float(np.mean(
        sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    ))

    # AIC (Gaussian log-likelihood)
    rss = max(ss_res, 1e-10)
    aic_refit = 2.0 * n_params + n * np.log(rss / n) + n * (1.0 + np.log(2.0 * np.pi))

    return {"r2": r2, "rmse": rmse, "crps": crps, "aic_refit": float(aic_refit)}


# ── MCMC re-fitting ─────────────────────────────────────────────────────────


def _subsample_data(data: pd.Series, max_obs: int | None, offset: int = 0) -> pd.Series:
    """Subsample a time-series to at most max_obs equally-spaced points.

    Uses stride-based subsampling to preserve uniform time spacing.
    Different offsets (0, 1, ..., stride-1) yield different phase-shifted
    subsamples while keeping the same temporal resolution.
    """
    if max_obs is None or len(data) <= max_obs:
        return data
    stride = len(data) // max_obs
    start = offset % stride
    idx = np.arange(start, len(data), stride)[:max_obs]
    return data.iloc[idx].reset_index(drop=True)


def _max_subsample_offsets(n_obs: int, max_obs: int) -> int:
    """Return the number of distinct stride offsets available."""
    if max_obs is None or n_obs <= max_obs:
        return 1
    return n_obs // max_obs


def refit_with_mcmc(
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
    map_estimate_json: str | None = None,
) -> dict:
    """Re-fit a GP time-series model with full MCMC and compute LOO/WAIC.

    Parameters
    ----------
    model_code : str
        The full PyMC model code (will be stripped of find_MAP/predict lines).
    data : pd.Series
        The raw time-series data.
    draws, tune, chains, cores, target_accept
        MCMC sampler settings.
    max_obs : int or None
        If set, uniformly subsample data to this many observations.
    map_estimate_json : str or None
        JSON string of the MAP estimate to use as initvals for MCMC.

    Returns
    -------
    dict with keys:
        status, elpd_loo, elpd_loo_se, p_loo, elpd_waic, elpd_waic_se,
        p_waic, loo_good_k, loo_bad_k, n_obs, error
    """
    data = _subsample_data(data, max_obs, offset=subsample_offset)
    result = {
        "status": "error",
        "elpd_loo": np.nan,
        "elpd_loo_se": np.nan,
        "p_loo": np.nan,
        "elpd_waic": np.nan,
        "elpd_waic_se": np.nan,
        "p_waic": np.nan,
        "loo_good_k": np.nan,
        "loo_bad_k": np.nan,
        "n_obs": len(data),
        "r2": np.nan,
        "rmse": np.nan,
        "crps": np.nan,
        "aic_refit": np.nan,
        "error": None,
    }

    try:
        # Prepare model code for sampling (strip find_MAP, predict, trend)
        model_only_code = strip_model_code_for_sampling(model_code)

        # Execute model definition
        exec_ns = {"pm": pm, "np": np, "pd": pd, "data": data}
        exec(model_only_code, exec_ns)  # noqa: S102

        model = exec_ns.get("model")
        if model is None:
            result["error"] = "No 'model' variable after exec"
            return result

        # Parse MAP estimate for initvals
        initvals = None
        if map_estimate_json:
            try:
                map_dict = json.loads(map_estimate_json) if isinstance(map_estimate_json, str) else map_estimate_json
                # Filter to only untransformed parameters (no _log__ suffixes)
                # that exist as model variables
                model_var_names = {v.name for v in model.free_RVs}
                initvals = {
                    k: float(v)
                    for k, v in map_dict.items()
                    if k in model_var_names and not k.endswith("__")
                }
                if not initvals:
                    initvals = None
            except (json.JSONDecodeError, TypeError, ValueError):
                initvals = None

        # Run MCMC
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts={"max_treedepth": 8},
                progressbar=True,
                discard_tuned_samples=True,
                idata_kwargs={"log_likelihood": True},
                cores=cores,
                initvals=initvals,
            )

        # Compute LOO and WAIC
        loo = az.loo(trace)
        waic = az.waic(trace)

        result.update(
            {
                "status": "ok",
                "elpd_loo": float(loo.elpd_loo),
                "elpd_loo_se": float(loo.se),
                "p_loo": float(loo.p_loo),
                "elpd_waic": float(waic.elpd_waic),
                "elpd_waic_se": float(waic.se),
                "p_waic": float(waic.p_waic),
                "loo_i": np.array(loo.loo_i.values).flatten().tolist(),
                "loo_good_k": int((loo.pareto_k < 0.7).sum()),
                "loo_bad_k": int((loo.pareto_k > 0.7).sum()),
                "error": None,
            }
        )

        # Compute prediction-based metrics (R², RMSE, CRPS, AIC)
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
            pass  # Prediction metrics stay NaN

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return result


def refit_baseline_model(
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
    """Fit a simple Normal(mu, sigma) baseline model and compute LOO/WAIC.

    Serves as a baseline for samples where the model-selection run failed.
    Analogous to the mean_fallback (R²=0) in evaluate_ts_run.py.
    """
    data = _subsample_data(data, max_obs, offset=subsample_offset)
    result = {
        "status": "baseline",
        "elpd_loo": -np.inf,
        "elpd_loo_se": np.nan,
        "p_loo": np.nan,
        "elpd_waic": -np.inf,
        "elpd_waic_se": np.nan,
        "p_waic": np.nan,
        "loo_good_k": np.nan,
        "loo_bad_k": np.nan,
        "n_obs": len(data),
        "r2": np.nan,
        "rmse": np.nan,
        "crps": np.nan,
        "aic_refit": np.nan,
        "error": None,
    }

    try:
        y = np.asarray(data.values, dtype=float)
        with pm.Model() as baseline_model:
            mu = pm.Normal("mu", mu=float(np.mean(y)), sigma=float(np.std(y)) * 10)
            sigma = pm.HalfNormal("sigma", sigma=float(np.std(y)) * 5)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        with baseline_model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts={"max_treedepth": 8},
                progressbar=False,
                discard_tuned_samples=True,
                idata_kwargs={"log_likelihood": True},
                cores=cores,
            )

        loo = az.loo(trace)
        waic = az.waic(trace)

        result.update(
            {
                "elpd_loo": float(loo.elpd_loo),
                "elpd_loo_se": float(loo.se),
                "p_loo": float(loo.p_loo),
                "elpd_waic": float(waic.elpd_waic),
                "elpd_waic_se": float(waic.se),
                "p_waic": float(waic.p_waic),
                "loo_good_k": int((loo.pareto_k < 0.7).sum()),
                "loo_bad_k": int((loo.pareto_k > 0.7).sum()),
                "error": None,
            }
        )

        # Prediction metrics for baseline: predict the posterior mean of mu
        try:
            mu_post = float(trace.posterior["mu"].mean())
            y_pred = np.full_like(y, mu_post)
            pred_metrics = _compute_prediction_metrics(y, y_pred, n_params=2)
            result.update(pred_metrics)
        except Exception:
            pass

    except Exception as e:
        result["error"] = f"Baseline fit failed: {type(e).__name__}: {e}"

    return result


# ── Extract best model per sample from parquet ───────────────────────────────


_AVG_METRICS = [
    "elpd_loo", "elpd_loo_se", "p_loo",
    "elpd_waic", "elpd_waic_se", "p_waic",
    "loo_good_k", "loo_bad_k", "n_obs",
    "r2", "rmse", "crps", "aic_refit",
]


def _average_refit_results(results: list[dict]) -> dict:
    """Average numeric ELPD metrics across multiple refit runs.

    Only averages over runs with status='ok' (or 'baseline').
    If no run succeeded, returns the first result as-is.
    """
    ok = [r for r in results if r["status"] in ("ok", "baseline")]
    if not ok:
        return results[0]

    avg = dict(ok[0])  # copy first successful result as template
    for m in _AVG_METRICS:
        vals = [r[m] for r in ok if np.isfinite(r.get(m, np.nan))]
        avg[m] = float(np.mean(vals)) if vals else ok[0][m]
    # Round k-counts to ints
    for k in ("loo_good_k", "loo_bad_k"):
        if np.isfinite(avg[k]):
            avg[k] = int(round(avg[k]))
    avg["error"] = None
    return avg


def get_best_model_from_parquet(parquet_path: Path) -> dict | None:
    """Extract the run-best model code, structure, and MAP estimate from a parquet.

    Searches backwards from the last step to find the best carried-forward model.
    The parquet tracks `run_best_model_*` columns which accumulate the best model
    across all steps (by AIC). We take the latest row that has a valid
    `run_best_model_code`, preferring rows with status='ok' but falling back to
    earlier steps if the final step errored.

    Returns dict with keys: model_code, model_structure, map_estimate, aic
    or None if no step produced a valid model.
    """
    df = pd.read_parquet(parquet_path)

    # Walk backwards from the last step to find a row with valid model code
    for i in range(len(df) - 1, -1, -1):
        row = df.iloc[i]
        code = row.get("run_best_model_code", "")
        if code and isinstance(code, str) and len(code.strip()) > 0:
            return {
                "model_code": code,
                "model_structure": str(_safe_json(row.get("run_best_model_structure", "[]"))),
                "map_estimate": row.get("run_best_map_estimate", None),
                "aic": float(row.get("run_best_model_aic", np.nan)),
            }

    return None


def get_step0_model_from_parquet(parquet_path: Path) -> dict | None:
    """Extract the step-0 (initial proposal) model from a parquet.

    Returns dict with keys: model_code, model_structure, map_estimate, aic
    or None if step 0 has no valid model.
    """
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return None

    row = df.iloc[0]
    code = row.get("step_best_proposed_model_code", "")
    if not code or not isinstance(code, str) or len(code.strip()) == 0:
        return None

    return {
        "model_code": code,
        "model_structure": str(_safe_json(row.get("step_best_proposed_model_structure", "[]"))),
        "map_estimate": row.get("step_best_proposed_map_estimate", None),
        "aic": float(row.get("step_best_proposed_model_aic", np.nan)),
    }


# ── Main pipeline ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc LOO-CV / WAIC evaluation for time-series runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", type=Path, help="Path to the run directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing dataset_*.pkl files. Defaults to the repo root.",
    )
    parser.add_argument("--draws", type=int, default=200, help="MCMC draws per chain (default: 200).")
    parser.add_argument("--tune", type=int, default=200, help="MCMC tuning steps (default: 200).")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains (default: 4).")
    parser.add_argument("--cores", type=int, default=4, help="Parallel cores for sampling (default: 4).")
    parser.add_argument(
        "--target-accept", type=float, default=0.85, help="NUTS target_accept (default: 0.85)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing).",
    )
    parser.add_argument(
        "--max-obs",
        type=int,
        default=None,
        help="Subsample each time-series to at most this many observations (e.g. 100, 150).",
    )
    parser.add_argument(
        "--n-subsample-reps",
        type=int,
        default=3,
        help="Number of random subsample repetitions to average over (default: 3).",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    data_dir: Path = (args.data_dir or Path(__file__).resolve().parent).resolve()

    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Run dir : {run_dir}")
    print(f"Data dir: {data_dir}")
    print(f"MCMC settings: draws={args.draws}, tune={args.tune}, chains={args.chains}, cores={args.cores}")
    if args.max_obs:
        print(f"Subsampling to max {args.max_obs} observations per sample, averaging over {args.n_subsample_reps} reps")
    print()

    # ── 1. Discover samples ──────────────────────────────────────────────
    samples_df = discover_samples(run_dir)
    if samples_df.empty:
        print("No samples discovered.", file=sys.stderr)
        sys.exit(1)

    if args.max_samples is not None:
        samples_df = samples_df.head(args.max_samples)

    print(f"Discovered {len(samples_df)} sample(s)")
    print()

    # ── 2. Process each sample ───────────────────────────────────────────
    records: list[dict] = []
    for i, row in enumerate(samples_df.itertuples()):
        print(f"[{i + 1}/{len(samples_df)}] Sample {row.sample_idx} ({row.label})")

        # Get best model from parquet
        best = get_best_model_from_parquet(row.parquet_path)
        if best is None:
            print(f"  No valid model (run status not 'ok'). Fitting baseline...")
            # Try to load data for baseline model
            try:
                dataset = load_pickle(row.data_pkl, data_dir)
                data = dataset[row.sample_idx]["data"]
            except Exception as e:
                print(f"  SKIP: cannot load data for baseline: {e}")
                for et in ["step_0", "best"]:
                    records.append(
                        {
                            "dataset": row.dataset,
                            "sample_idx": row.sample_idx,
                            "label": row.label,
                            "eval_type": et,
                            "model_structure": "mean_baseline",
                            "aic": np.nan,
                            "status": "baseline",
                            "elpd_loo": -np.inf,
                            "elpd_loo_se": np.nan,
                            "p_loo": np.nan,
                            "elpd_waic": -np.inf,
                            "elpd_waic_se": np.nan,
                            "p_waic": np.nan,
                            "loo_good_k": np.nan,
                            "loo_bad_k": np.nan,
                            "n_obs": np.nan,
                            "r2": np.nan,
                            "rmse": np.nan,
                            "crps": np.nan,
                            "aic_refit": np.nan,
                            "error": f"no valid model and cannot load data: {e}",
                        }
                    )
                continue

            t0 = time.time()
            baseline_result = refit_baseline_model(
                data=data,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                max_obs=args.max_obs,
            )
            elapsed = time.time() - t0

            if np.isfinite(baseline_result["elpd_loo"]):
                print(
                    f"  Baseline elpd_loo={baseline_result['elpd_loo']:.1f} ± "
                    f"{baseline_result['elpd_loo_se']:.1f}  time={elapsed:.1f}s"
                )
            else:
                print(f"  Baseline fit FAILED ({elapsed:.1f}s), using -inf")

            for et in ["step_0", "best"]:
                records.append(
                    {
                        "dataset": row.dataset,
                        "sample_idx": row.sample_idx,
                        "label": row.label,
                        "eval_type": et,
                        "model_structure": "mean_baseline",
                        "aic": np.nan,
                        **{
                            k: baseline_result[k]
                            for k in [
                                "status",
                                "elpd_loo",
                                "elpd_loo_se",
                                "p_loo",
                                "elpd_waic",
                                "elpd_waic_se",
                                "p_waic",
                                "loo_good_k",
                                "loo_bad_k",
                                "n_obs",
                                "r2",
                                "rmse",
                                "crps",
                                "aic_refit",
                                "error",
                            ]
                        },
                    }
                )
            continue

        # Load data
        try:
            dataset = load_pickle(row.data_pkl, data_dir)
            data = dataset[row.sample_idx]["data"]
        except Exception as e:
            print(f"  SKIP: cannot load data: {e}")
            for et in ["step_0", "best"]:
                records.append(
                    {
                        "dataset": row.dataset,
                        "sample_idx": row.sample_idx,
                        "label": row.label,
                        "eval_type": et,
                        "model_structure": best["model_structure"],
                        "aic": best["aic"],
                        "status": "data_error",
                        "elpd_loo": np.nan,
                        "elpd_loo_se": np.nan,
                        "p_loo": np.nan,
                        "elpd_waic": np.nan,
                        "elpd_waic_se": np.nan,
                        "p_waic": np.nan,
                        "loo_good_k": np.nan,
                        "loo_bad_k": np.nan,
                        "n_obs": np.nan,
                        "r2": np.nan,
                        "rmse": np.nan,
                        "crps": np.nan,
                        "aic_refit": np.nan,
                        "error": str(e),
                    }
                )
            continue

        n_obs_actual = len(data)
        n_obs_used = min(n_obs_actual, args.max_obs) if args.max_obs else n_obs_actual

        # ── Step 0 ELPD ──────────────────────────────────────────────────
        max_offsets = _max_subsample_offsets(n_obs_actual, args.max_obs) if args.max_obs else 1
        n_reps = min(args.n_subsample_reps, max_offsets) if args.max_obs and n_obs_actual > args.max_obs else 1
        step0 = get_step0_model_from_parquet(row.parquet_path)
        if step0 is not None:
            print(f"  Step 0: {step0['model_structure']}  AIC: {step0['aic']:.1f}")
            print(f"  Re-fitting step_0 with MCMC ({n_obs_used}/{n_obs_actual} obs, {n_reps} reps)...")
            t0 = time.time()
            step0_runs = []
            for rep in range(n_reps):
                r = refit_with_mcmc(
                    model_code=step0["model_code"],
                    data=data,
                    draws=args.draws,
                    tune=args.tune,
                    chains=args.chains,
                    cores=args.cores,
                    target_accept=args.target_accept,
                    max_obs=args.max_obs,
                    subsample_offset=rep,
                    map_estimate_json=step0["map_estimate"],
                )
                step0_runs.append(r)
                status_tag = f"elpd_loo={r['elpd_loo']:.1f}" if r["status"] == "ok" else "FAILED"
                print(f"    rep {rep + 1}/{n_reps}: {status_tag}")
            step0_result = _average_refit_results(step0_runs)
            elapsed = time.time() - t0
            n_ok = sum(1 for r in step0_runs if r["status"] == "ok")
            if n_ok > 0:
                print(f"  step_0 avg elpd_loo={step0_result['elpd_loo']:.1f} ({n_ok}/{n_reps} ok)  time={elapsed:.1f}s")
            else:
                print(f"  step_0 ALL FAILED ({elapsed:.1f}s), falling back to baseline")
                step0_result = refit_baseline_model(
                    data=data, draws=args.draws, tune=args.tune,
                    chains=args.chains, cores=args.cores,
                    target_accept=args.target_accept, max_obs=args.max_obs,
                )
                step0["model_structure"] = "mean_baseline"
                step0["aic"] = np.nan
        else:
            print(f"  Step 0 has no model code, using baseline")
            step0_result = refit_baseline_model(
                data=data, draws=args.draws, tune=args.tune,
                chains=args.chains, cores=args.cores,
                target_accept=args.target_accept, max_obs=args.max_obs,
            )
            step0 = {"model_structure": "mean_baseline", "aic": np.nan}

        records.append(
            {
                "dataset": row.dataset,
                "sample_idx": row.sample_idx,
                "label": row.label,
                "eval_type": "step_0",
                "model_structure": step0["model_structure"],
                "aic": step0["aic"],
                **{
                    k: step0_result[k]
                    for k in [
                        "status",
                        "elpd_loo",
                        "elpd_loo_se",
                        "p_loo",
                        "elpd_waic",
                        "elpd_waic_se",
                        "p_waic",
                        "loo_good_k",
                        "loo_bad_k",
                        "n_obs",
                        "r2",
                        "rmse",
                        "crps",
                        "aic_refit",
                        "error",
                    ]
                },
            }
        )

        # ── Best model ELPD ───────────────────────────────────────────────
        print(f"  Best:   {best['model_structure']}  AIC: {best['aic']:.1f}")
        print(f"  Re-fitting best with MCMC ({n_obs_used}/{n_obs_actual} obs, {n_reps} reps)...")

        t0 = time.time()
        best_runs = []
        for rep in range(n_reps):
            r = refit_with_mcmc(
                model_code=best["model_code"],
                data=data,
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=args.cores,
                target_accept=args.target_accept,
                max_obs=args.max_obs,
                subsample_offset=rep,
                map_estimate_json=best["map_estimate"],
            )
            best_runs.append(r)
            status_tag = f"elpd_loo={r['elpd_loo']:.1f}" if r["status"] == "ok" else "FAILED"
            print(f"    rep {rep + 1}/{n_reps}: {status_tag}")
        elpd_result = _average_refit_results(best_runs)
        elapsed = time.time() - t0
        n_ok = sum(1 for r in best_runs if r["status"] == "ok")

        if n_ok > 0:
            print(
                f"  best avg elpd_loo={elpd_result['elpd_loo']:.1f} ± {elpd_result['elpd_loo_se']:.1f}  "
                f"elpd_waic={elpd_result['elpd_waic']:.1f} ± {elpd_result['elpd_waic_se']:.1f}  "
                f"({n_ok}/{n_reps} ok)  time={elapsed:.1f}s"
            )
        else:
            err_preview = str(elpd_result["error"] or "")[:200]
            print(f"  best ALL FAILED ({elapsed:.1f}s): {err_preview}")

        records.append(
            {
                "dataset": row.dataset,
                "sample_idx": row.sample_idx,
                "label": row.label,
                "eval_type": "best",
                "model_structure": best["model_structure"],
                "aic": best["aic"],
                **{
                    k: elpd_result[k]
                    for k in [
                        "status",
                        "elpd_loo",
                        "elpd_loo_se",
                        "p_loo",
                        "elpd_waic",
                        "elpd_waic_se",
                        "p_waic",
                        "loo_good_k",
                        "loo_bad_k",
                        "n_obs",
                        "r2",
                        "rmse",
                        "crps",
                        "aic_refit",
                        "error",
                    ]
                },
            }
        )

    # ── 3. Build results DataFrame ───────────────────────────────────────
    results_df = pd.DataFrame(records)

    print()
    print("=" * 100)
    print("ELPD EVALUATION RESULTS")
    print("=" * 100)
    display_cols = [
        "sample_idx",
        "label",
        "eval_type",
        "model_structure",
        "aic",
        "elpd_loo",
        "elpd_loo_se",
        "p_loo",
        "elpd_waic",
        "elpd_waic_se",
        "p_waic",
        "r2",
        "rmse",
        "crps",
        "aic_refit",
        "loo_good_k",
        "loo_bad_k",
        "status",
    ]
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 200)
    print(results_df[display_cols].to_string(index=False))
    print()

    # ── 4. Summary statistics ────────────────────────────────────────────
    for eval_type in ["step_0", "best"]:
        et_df = results_df[results_df["eval_type"] == eval_type]
        ok_df = et_df[et_df["status"] == "ok"]
        baseline_df = et_df[et_df["status"] == "baseline"]
        if not ok_df.empty:
            print("=" * 100)
            print(f"SUMMARY STATISTICS — {eval_type.upper()} (successful fits only)")
            print("=" * 100)
            for metric in ["elpd_loo", "elpd_waic", "p_loo", "p_waic", "r2", "rmse", "crps", "aic_refit", "aic"]:
                vals = ok_df[metric].dropna()
                if not vals.empty:
                    print(
                        f"  {metric:15s}  mean={vals.mean():10.2f}  std={vals.std():10.2f}  "
                        f"min={vals.min():10.2f}  max={vals.max():10.2f}  n={len(vals)}"
                    )
            print(f"\n  Pareto k diagnostics:")
            print(f"    Total good (k < 0.7): {ok_df['loo_good_k'].sum():.0f}")
            print(f"    Total bad  (k > 0.7): {ok_df['loo_bad_k'].sum():.0f}")
            print()

        if not baseline_df.empty:
            finite_bl = baseline_df[np.isfinite(baseline_df["elpd_loo"])]
            print("=" * 100)
            print(f"BASELINE STATISTICS — {eval_type.upper()} ({len(baseline_df)} sample(s))")
            print("=" * 100)
            if not finite_bl.empty:
                for metric in ["elpd_loo", "elpd_waic"]:
                    vals = finite_bl[metric].dropna()
                    if not vals.empty:
                        print(
                            f"  {metric:15s}  mean={vals.mean():10.2f}  std={vals.std():10.2f}  "
                            f"min={vals.min():10.2f}  max={vals.max():10.2f}  n={len(vals)}"
                        )
            else:
                print("  All baseline fits failed (using -inf).")
            print()

    # Overall counts
    n_ok = (results_df["status"] == "ok").sum()
    n_baseline = (results_df["status"] == "baseline").sum()
    n_fail = (results_df["status"] == "error").sum()
    n_skip = (results_df["status"].isin(["data_error"])).sum()
    n_samples = len(results_df) // 2  # each sample has 2 rows
    print(f"Samples: {n_samples}  Total rows: {len(results_df)}")
    print(f"Success: {n_ok}  Baseline: {n_baseline}  Failed: {n_fail}  Skipped: {n_skip}")

    # ── 5. Save CSV ──────────────────────────────────────────────────────
    out_csv = run_dir / "ts_evaluation_elpd.csv"
    # Don't save the full traceback error column to CSV
    save_df = results_df.copy()
    save_df["error"] = save_df["error"].apply(
        lambda e: str(e)[:200] if e is not None else None
    )
    save_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
