#!/usr/bin/env python
"""
Evaluate step-by-step trend predictions from a PyMC model-selection run.

Outputs:
  a) Raw table with per-sample, per-step metrics (R², RMSE, AIC, CRPS, family).
  b) Best-performance table: per sample, the best step (by R²) with all metrics.
  c) Box-plot of best-performance spread across samples for a chosen metric.

Usage examples:
  python evaluate_ts_run.py outputs/ts_expert_forced/20260423_185203
  python evaluate_ts_run.py outputs/ts_expert_forced/20260423_185203 --box-metric rmse
  python evaluate_ts_run.py outputs/ts_expert_forced/20260423_185203 --data-dir /path/to/pkls
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score

SAMPLE_DIR_RE = re.compile(r"^(\d+)_(.+)$")


# ── Discovery ───────────────────────────────────────────────────────────────


def discover_samples(run_dir: Path) -> pd.DataFrame:
    """Return a DataFrame of (dataset, sample_idx, label, parquet_path, data_pkl).
    
    Supports two layouts:
    1. Nested: run_dir/<dataset_dir>/config.json + sample folders
    2. Flat: run_dir/config.json + sample folders
    """
    rows: list[dict] = []
    
    # Check if flat layout (config.json at root)
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
    
    # Fall back to nested layout (config.json in subdirectories)
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


def load_pickle(pkl_name: str, data_dir: Path) -> list:
    key = str(data_dir / pkl_name)
    if key not in _pkl_cache:
        pkl_path = data_dir / pkl_name
        with open(pkl_path, "rb") as f:
            _pkl_cache[key] = pickle.load(f)
    return _pkl_cache[key]


def get_actual(dataset_name: str, sample_idx: int, data_pkl: str, data_dir: Path) -> np.ndarray:
    dataset = load_pickle(data_pkl, data_dir)
    return np.asarray(dataset[sample_idx]["data"].values, dtype=float)


def _safe_json(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, dict)):
        return val
    try:
        return json.loads(val)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def load_run_log(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "trend" in df.columns:
        df["trend_arr"] = df["trend"].apply(
            lambda v: np.asarray(_safe_json(v), dtype=float) if _safe_json(v) is not None else None
        )
    else:
        df["trend_arr"] = None
    if "metrics" in df.columns:
        df["metrics_dict"] = df["metrics"].apply(_safe_json)
        df["n_params_logged"] = df["metrics_dict"].apply(
            lambda d: int(d["n_params"]) if isinstance(d, dict) and "n_params" in d else np.nan
        )
    else:
        df["n_params_logged"] = np.nan
    if "predicted_dist" in df.columns:
        df["family"] = df["predicted_dist"].apply(
            lambda v: "_".join(_safe_json(v)) if isinstance(_safe_json(v), list) else str(_safe_json(v))
        )
    elif "kernels" in df.columns:
        df["family"] = df["kernels"].apply(
            lambda v: "_".join(_safe_json(v)) if isinstance(_safe_json(v), list) else str(_safe_json(v))
        )
    else:
        df["family"] = "unknown"
    return df[["step", "family", "trend_arr", "n_params_logged"]]


# ── Metrics ──────────────────────────────────────────────────────────────────


def _align(y_true: np.ndarray, y_pred: np.ndarray):
    if y_pred is None:
        return None
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return None
    return y_true[:n], y_pred[:n]


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pair = _align(y_true, y_pred)
    if pair is None:
        return np.nan
    y_t, y_p = pair
    if np.any(np.isnan(y_t)) or np.any(np.isnan(y_p)):
        return np.nan
    return float(r2_score(y_t, y_p))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pair = _align(y_true, y_pred)
    if pair is None:
        return np.nan
    y_t, y_p = pair
    if np.any(np.isnan(y_t)) or np.any(np.isnan(y_p)):
        return np.nan
    return float(np.sqrt(mean_squared_error(y_t, y_p)))


def compute_aic(y_true: np.ndarray, y_pred: np.ndarray, num_params: int) -> float:
    pair = _align(y_true, y_pred)
    if pair is None or not np.isfinite(num_params):
        return np.nan
    resid = pair[0] - pair[1]
    n = len(resid)
    rss = float(np.sum(resid**2))
    if rss <= 0 or n == 0:
        return np.nan
    return 2.0 * num_params + n * np.log(rss / n) + n * (1.0 + np.log(2.0 * np.pi))


def compute_crps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pair = _align(y_true, y_pred)
    if pair is None:
        return np.nan
    resid = pair[0] - pair[1]
    sigma = float(np.sqrt(np.mean(resid**2)))
    if sigma <= 0:
        return np.nan
    z = resid / sigma
    crps_pts = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps_pts))


# ── Main pipeline ────────────────────────────────────────────────────────────


def build_eval_df(samples_df: pd.DataFrame, actuals: dict, data_dir: Path) -> pd.DataFrame:
    records: list[dict] = []
    for row in samples_df.itertuples():
        y_true = actuals[(row.dataset, row.sample_idx)]
        log_df = load_run_log(row.parquet_path)
        for _, r in log_df.iterrows():
            trend = r["trend_arr"]
            n_params = r["n_params_logged"]
            rec = {
                "dataset": row.dataset,
                "sample_idx": row.sample_idx,
                "label": row.label,
                "step": int(r["step"]),
                "family": r["family"],
                "n_params": None if pd.isna(n_params) else int(n_params),
                "r2": compute_r2(y_true, trend),
                "rmse": compute_rmse(y_true, trend),
                "aic": compute_aic(y_true, trend, int(n_params) if not pd.isna(n_params) else 0),
                "crps": compute_crps(y_true, trend),
            }
            records.append(rec)
    eval_df = pd.DataFrame(records)
    eval_df = eval_df.sort_values(["dataset", "sample_idx", "step"]).reset_index(drop=True)
    return eval_df


def best_per_sample(eval_df: pd.DataFrame) -> pd.DataFrame:
    """For each sample, pick the step with the highest R² and report all metrics."""
    # Drop groups where all R² values are NaN (errored samples with no trend)
    valid = eval_df.dropna(subset=["r2"])
    if valid.empty:
        return pd.DataFrame(columns=["dataset", "sample_idx", "label", "best_step", "family", "r2", "rmse", "aic", "crps"])
    idx = valid.groupby(["dataset", "sample_idx"])["r2"].idxmax()
    best_df = eval_df.loc[idx].copy()
    best_df = best_df.rename(columns={"step": "best_step"})
    best_df = (
        best_df[["dataset", "sample_idx", "label", "best_step", "family", "r2", "rmse", "aic", "crps"]]
        .sort_values(["dataset", "sample_idx"])
        .reset_index(drop=True)
    )
    return best_df


def plot_best_boxplot(best_df: pd.DataFrame, metric: str, run_name: str, save_path: Path | None):
    metric_lower = metric.lower()
    valid = ["r2", "rmse", "aic", "crps"]
    if metric_lower not in valid:
        print(f"Unknown metric '{metric}'. Choose from: {valid}", file=sys.stderr)
        sys.exit(1)

    labels = {"r2": "R²", "rmse": "RMSE", "aic": "AIC", "crps": "CRPS"}
    higher_better = {"r2": True, "rmse": False, "aic": False, "crps": False}

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    data = best_df[metric_lower].dropna()
    ax.boxplot(data.values, showmeans=True, tick_labels=[labels[metric_lower]])
    sns.stripplot(y=data.values, color="steelblue", alpha=0.5, jitter=True, ax=ax)
    direction = "↑ higher is better" if higher_better[metric_lower] else "↓ lower is better"
    ax.set_title(f"Best {labels[metric_lower]} across samples — {run_name}\n({direction})", fontweight="bold")
    ax.set_ylabel(labels[metric_lower])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved box-plot: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a time-series model-selection run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", type=Path, help="Path to the timestamped run directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing dataset_*.pkl files. Defaults to the repo root.",
    )
    parser.add_argument(
        "--box-metric",
        type=str,
        default="r2",
        choices=["r2", "rmse", "aic", "crps"],
        help="Metric for the best-performance box-plot (default: r2).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots as PNG files into the run directory.",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    data_dir: Path = (args.data_dir or Path(__file__).resolve().parent).resolve()

    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Run dir : {run_dir}")
    print(f"Data dir: {data_dir}")

    # ── 1. Discover samples ──────────────────────────────────────────────
    samples_df = discover_samples(run_dir)
    if samples_df.empty:
        print("No samples discovered. Check the run directory layout.", file=sys.stderr)
        sys.exit(1)
    print(f"Discovered {len(samples_df)} sample(s)\n")

    # ── 2. Load ground-truth actuals ─────────────────────────────────────
    actuals: dict[tuple[str, int], np.ndarray] = {}
    for row in samples_df.itertuples():
        actuals[(row.dataset, row.sample_idx)] = get_actual(
            row.dataset, row.sample_idx, row.data_pkl, data_dir
        )

    # ── 3. Compute per-step evaluation table ─────────────────────────────
    eval_df = build_eval_df(samples_df, actuals, data_dir)
    print("=" * 80)
    print("(a) RAW EVALUATION TABLE — per sample × step")
    print("=" * 80)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 200)
    print(eval_df.to_string(index=False))
    print(f"\n({len(eval_df)} rows)\n")

    # ── 4. Best performance per sample (by highest R²) ───────────────────
    best_df = best_per_sample(eval_df)
    print("=" * 80)
    print("(b) BEST PERFORMANCE PER SAMPLE (highest R²)")
    print("=" * 80)
    print(best_df.to_string(index=False))
    print()

    # ── 5. Export CSVs ───────────────────────────────────────────────────
    raw_csv = run_dir / "ts_evaluation_results.csv"
    best_csv = run_dir / "ts_evaluation_best_per_sample.csv"
    eval_df.to_csv(raw_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    print(f"Saved raw table  : {raw_csv}")
    print(f"Saved best table : {best_csv}")

    # ── 6. Box-plot of best-performance spread ───────────────────────────
    save_path = (run_dir / f"ts_boxplot_best_{args.box_metric}.png") if args.save_plots else None
    plot_best_boxplot(best_df, args.box_metric, run_dir.name, save_path)


if __name__ == "__main__":
    main()
