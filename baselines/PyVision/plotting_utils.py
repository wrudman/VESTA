"""Plotting utilities for histogram overlays and fitted model visualization."""

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
#from morphic.string import format_exception_msg
from scipy import stats

logger: logging.Logger = logging.getLogger("plotting_utils")

DISTRIBUTION_MAP: Dict[str, Dict[str, Any]] = {
    "gaussian": {"scipy_name": "norm", "params": {"loc": "mu", "scale": "sigma"}},
    "normal": {
        "scipy_name": "norm",
        "params": {"loc": "mu", "scale": "sigma"},
    },
    "lognormal": {"scipy_name": "lognorm", "params": {"s": "sigma", "scale": lambda p: np.exp(p["mu"])}},
    "cauchy": {"scipy_name": "cauchy", "params": {"loc": "location", "scale": "scale"}},
    "laplace": {"scipy_name": "laplace", "params": {"loc": "location", "scale": "scale"}},
    "student-t": {"scipy_name": "t", "params": {"df": "nu", "loc": "mu", "scale": "sigma"}},
    "studentt": {
        "scipy_name": "t",
        "params": {"df": "nu", "loc": "mu", "scale": "sigma"},
    },
    "student_t": {
        "scipy_name": "t",
        "params": {"df": "nu", "loc": "mu", "scale": "sigma"},
    },
    "exponential": {
        "scipy_name": "expon",
        "params": {"scale": lambda p: 1 / p["rate"]},
    },
    "uniform": {
        "scipy_name": "uniform",
        "params": {"loc": "lower", "scale": lambda p: p["upper"] - p["lower"]},
    },
    "weibull": {"scipy_name": "weibull_min", "params": {"c": "alpha", "scale": "beta"}},
    "gamma": {
        "scipy_name": "gamma",
        "params": {"a": "alpha", "scale": lambda p: 1 / p["beta"]},
    },
    "beta": {"scipy_name": "beta", "params": {"a": "alpha", "b": "beta", "loc": 0, "scale": 1}},
    "poisson": {
        "scipy_name": "poisson",
        "params": {"mu": "lambda"},
    },
    "negative-binomial": {"scipy_name": "nbinom", "params": {"n": "n", "p": "p"}},
    "negativebinomial": {
        "scipy_name": "nbinom",
        "params": {"n": "n", "p": "p"},
    },
    "binomial": {"scipy_name": "binom", "params": {"n": "n", "p": "p"}},
    "geometric": {"scipy_name": "geom", "params": {"p": "p"}},
    "pareto": {"scipy_name": "pareto", "params": {"b": "alpha", "scale": "m"}},
    "chi-squared": {"scipy_name": "chi2", "params": {"df": "df"}},
    "chisquared": {
        "scipy_name": "chi2",
        "params": {"df": "df"},
    },
    "f-distribution": {"scipy_name": "f", "params": {"dfn": "dfn", "dfd": "dfd"}},
    "rayleigh": {"scipy_name": "rayleigh", "params": {"scale": "scale"}},
    "gumbel": {"scipy_name": "gumbel_r", "params": {"loc": "mu", "scale": "beta"}},
    "logistic": {"scipy_name": "logistic", "params": {"loc": "mu", "scale": "s"}},
    "wald": {
        "scipy_name": "invgauss",
        "params": {"mu": "mu", "scale": lambda p: p["mu"] ** 3 / p["lambda"]},
    },
    "inverse-gamma": {"scipy_name": "invgamma", "params": {"a": "alpha", "scale": "beta"}},
    "inversegamma": {
        "scipy_name": "invgamma",
        "params": {"a": "alpha", "scale": "beta"},
    },
    "half-normal": {"scipy_name": "halfnorm", "params": {"scale": "sigma"}},
    "halfnormal": {
        "scipy_name": "halfnorm",
        "params": {"scale": "sigma"},
    },
    "half-cauchy": {"scipy_name": "halfcauchy", "params": {"scale": "beta"}},
    "halfcauchy": {
        "scipy_name": "halfcauchy",
        "params": {"scale": "beta"},
    },
    "triangular": {
        "scipy_name": "triang",
        "params": {
            "c": lambda p: (p["mode"] - p["lower"]) / (p["upper"] - p["lower"]),
            "loc": "lower",
            "scale": lambda p: p["upper"] - p["lower"],
        },
    },
    "vonmises": {"scipy_name": "vonmises", "params": {"kappa": "kappa", "loc": "mu"}},
}


def normalize_family_name(family_name: Union[str, List[str]]) -> str:
    """Normalize family name to match DISTRIBUTION_MAP keys.

    Accepts a string like 'gaussian' or a list like ['gaussian', 'lognormal']
    (in which case only the first element is used).
    """
    if isinstance(family_name, list):
        family_name = family_name[0] if len(family_name) > 0 else ""

    normalized: str = family_name.lower().replace("_", "-").replace(" ", "-")

    if "log-normal" in normalized or "lognormal" in normalized:
        return "lognormal"
    elif "student" in normalized or "t-dist" in normalized:
        return "student-t"
    elif "normal" in normalized or "gauss" in normalized:
        return "gaussian"
    elif "expo" in normalized:
        return "exponential"

    return normalized


def get_scipy_distribution(
    *,
    family_name: Union[str, List[str]],
    extracted_params: Dict[str, float],
) -> Any:
    """Get scipy distribution object with proper parameters.

    Args:
        family_name: Distribution name as a string or single-element list.
        extracted_params: Dict mapping parameter names to values.

    Raises:
        ValueError: If the distribution is not supported or params are missing.
    """
    if isinstance(family_name, list):
        family_name = family_name[0] if len(family_name) > 0 else ""

    normalized_name: str = normalize_family_name(family_name)

    if normalized_name not in DISTRIBUTION_MAP:
        raise ValueError(
            f"Distribution {family_name!r} (normalized: {normalized_name!r}) not supported. "
            f"Supported: {list(DISTRIBUTION_MAP.keys())}"
        )

    dist_info: Dict[str, Any] = DISTRIBUTION_MAP[normalized_name]
    scipy_dist: Any = getattr(stats, dist_info["scipy_name"])

    scipy_params: Dict[str, Any] = {}
    for scipy_param, pymc_param in dist_info["params"].items():
        if callable(pymc_param):
            scipy_params[scipy_param] = pymc_param(extracted_params)
        elif isinstance(pymc_param, (int, float)):
            scipy_params[scipy_param] = pymc_param
        else:
            if pymc_param not in extracted_params:
                raise ValueError(
                    f"Missing parameter {pymc_param!r} for distribution {family_name!r}. "
                    f"Available: {list(extracted_params.keys())}"
                )
            scipy_params[scipy_param] = extracted_params[pymc_param]

    return scipy_dist(**scipy_params)


def extract_distribution_params(
    *,
    map_estimate: Dict[str, Any],
    family_name: Union[str, List[str]],
    component_idx: int,
) -> Dict[str, float]:
    """Extract distribution parameters from MAP estimate.

    Args:
        map_estimate: MAP parameter estimates dict.
        family_name: Distribution name as a string or list.
        component_idx: Which component to extract (0-indexed).
    """
    params: Dict[str, float] = {}

    if isinstance(family_name, list):
        resolved_name: str = (
            family_name[component_idx] if component_idx < len(family_name) else family_name[0]
        )
    else:
        resolved_name = family_name

    family_normalized: str = resolved_name.lower().replace("-", "_")
    family_no_sep: str = family_normalized.replace("_", "")

    family_prefixes: List[str] = [family_normalized, family_no_sep]
    if "student" in family_normalized:
        family_prefixes.append("student")

    for key, value in map_estimate.items():
        if "simplex" in key.lower() or "log__" in key or key in ["w", "weights"]:
            continue

        key_lower: str = key.lower()

        if f"_{component_idx}" in key:
            if any(prefix in key_lower for prefix in family_prefixes):
                param_name: Optional[str] = extract_param_name_from_key(
                    key=key,
                    family_normalized=family_normalized,
                    component_idx=component_idx,
                )
                if param_name is not None:
                    params[param_name] = float(np.atleast_1d(value).item())

        elif component_idx == 0:
            if any(key_lower.startswith(prefix) for prefix in family_prefixes):
                param_name = extract_param_name_from_key(
                    key=key,
                    family_normalized=family_normalized,
                    component_idx=None,
                )
                if param_name is not None:
                    params[param_name] = float(np.atleast_1d(value).item())

    return params


_KNOWN_PARAM_NAMES: List[str] = [
    "mu",
    "sigma",
    "alpha",
    "beta",
    "nu",
    "lambda",
    "lam",
    "rate",
    "location",
    "scale",
    "lower",
    "upper",
    "df",
    "s",
    "m",
]


def extract_param_name_from_key(
    *,
    key: str,
    family_normalized: Optional[str],
    component_idx: Optional[int],
) -> Optional[str]:
    """Helper to extract parameter name from a key string."""
    parts: List[str] = key.split("_")

    if component_idx is None:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] in _KNOWN_PARAM_NAMES:
                param_name: str = parts[i]
                if param_name == "rate":
                    return "lam"
                return param_name
        return None

    found_param: Optional[str] = None
    for part in parts:
        if part == str(component_idx):
            continue
        if part in _KNOWN_PARAM_NAMES:
            found_param = part
            break

    if found_param == "rate":
        found_param = "lam"

    return found_param


def plot_best_fit(
    data: np.ndarray,
    map_estimate: Dict[str, Any],
    model_info: Dict[str, Any],
    *,
    path: str,
    best_idx: Any,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot histogram of data with fitted model PDF overlay.

    Handles single distributions and mixtures. distribution_family values
    may be lists (e.g. ['gaussian', 'lognormal']) or strings.
    """
    plt.figure(figsize=figsize)
    plt.hist(data, bins=75, density=True, alpha=0.6, color="lightblue", edgecolor="black", label="Data")

    x: np.ndarray = np.linspace(data.min() - 1, data.max() + 1, 1000)

    dist_family_raw: Any = model_info["distribution_family"][best_idx]
    is_mixture_val: Any = model_info["is_mixture"][best_idx]

    if isinstance(dist_family_raw, list):
        families: List[str] = dist_family_raw
    elif isinstance(dist_family_raw, str) and len(dist_family_raw) > 0:
        families = dist_family_raw.split("_")
    else:
        families = []

    is_mixture: bool = is_mixture_val is True or is_mixture_val == "true" or len(families) > 1

    def _get_weights(map_est: Dict[str, Any], n_components: int) -> np.ndarray:
        if "w" in map_est:
            return np.array(map_est["w"])
        if "weights" in map_est:
            return np.array(map_est["weights"])
        w_key: Optional[str] = next(
            (
                k
                for k in map_est
                if k.startswith("w_") and not k.endswith("_log__") and not k.endswith("_simplex__")
            ),
            None,
        )
        if w_key is not None:
            return np.array(map_est[w_key])
        return np.ones(n_components) / n_components

    if is_mixture and len(families) > 0:
        weights: np.ndarray = _get_weights(map_estimate, len(families))
        if not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()

        total_pdf: np.ndarray = np.zeros_like(x)

        for component_idx, component_family in enumerate(families):
            try:
                weight: float = (
                    weights[component_idx] if component_idx < len(weights) else 1.0 / len(families)
                )
                extracted_params: Dict[str, float] = extract_distribution_params(
                    map_estimate=map_estimate,
                    family_name=component_family,
                    component_idx=component_idx,
                )

                if len(extracted_params) == 0:
                    logger.warning(
                        "No parameters extracted for %s component %d. Available keys: %s",
                        component_family,
                        component_idx,
                        [k for k in map_estimate.keys() if component_family in k.lower()],
                    )
                    continue

                logger.info("Component %d (%s): %s", component_idx, component_family, extracted_params)

                dist: Any = get_scipy_distribution(
                    family_name=component_family,
                    extracted_params=extracted_params,
                )
                component_pdf: np.ndarray = weight * dist.pdf(x)
                total_pdf += component_pdf

            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning(
                    "Could not plot %s component: %s", component_family, format_exception_msg(exc)
                )
                logger.debug(traceback.format_exc())
                continue

        if np.any(total_pdf > 0):
            plt.plot(x, total_pdf, "-", linewidth=2.5, label="Predicted Model Fit", color="red")
        else:
            logger.warning("Total PDF is zero")

    else:
        single_family: str = families[0] if len(families) > 0 else ""

        try:
            extracted_params = extract_distribution_params(
                map_estimate=map_estimate,
                family_name=single_family,
                component_idx=0,
            )

            if len(extracted_params) == 0:
                logger.warning(
                    "No parameters extracted for %s. Available keys: %s",
                    single_family,
                    list(map_estimate.keys()),
                )
                plt.title("Data (Parameter extraction failed)", fontsize=24)
                plt.savefig(path)
                plt.close()
                return

            logger.info("Single distribution (%s): %s", single_family, extracted_params)

            dist = get_scipy_distribution(family_name=single_family, extracted_params=extracted_params)
            pdf: np.ndarray = dist.pdf(x)
            plt.plot(x, pdf, "-", linewidth=2.5, color="red", label="Predicted Model Fit")

        except (ValueError, TypeError, RuntimeError) as exc:
            logger.warning("Could not plot %s: %s", single_family, format_exception_msg(exc))
            logger.debug(traceback.format_exc())

    plt.xlabel("Value", fontsize=12)
    plt.tick_params(axis="x", labelsize=14)
    plt.ylabel("Density", fontsize=12)
    plt.title("Data vs Predicted Model Fit", fontsize=20)
    plt.legend(fontsize=20, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hist(data: np.ndarray, *, save_path: str = "fit.png") -> None:
    """Plot histogram of data."""
    plt.figure(figsize=(8, 5))
    plt.hist(
        data,
        bins=75,
        density=True,
        alpha=0.5,
        label="Data",
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
    )
    plt.grid(alpha=0.5)
    plt.savefig(save_path)
    plt.close()


def plot_time_series(
    segment: Any,
    *,
    save_path: str = "timeseries.png",
) -> None:
    """Plot a time series segment and save the figure.

    Args:
        segment: pandas Series (index=time, values=signal).
        save_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(segment.index, segment.values, color="tab:blue", linewidth=1.5, label="Segment")
    plt.subplots_adjust(left=0.05, right=0.95)
    plt.grid(alpha=0.5)
    plt.savefig(save_path)
    plt.close(fig)


def plot_best_ts_fit(
    series: Any,
    trend: Any,
    kernels_dict: Dict[str, List[str]],
    *,
    path: str,
    best_idx: int,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot time series with Gaussian Process fitted trend overlay."""
    kernels: List[str] = kernels_dict[str(best_idx)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, color="grey", alpha=0.6, linewidth=1.5, label="Observed Data")
    ax.plot(trend.index, trend.values, color="orange", linewidth=2.5, label="GP Fit")
    ax.set_title(f"Time Series Fit | Kernels: {', '.join(kernels)}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_fit_vs_actuals_with_residuals_distribution(
    series: Any,
    trend: Any,
    kernels_dict: Dict[str, List[str]],
    *,
    path: str,
    best_idx: int,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Plot time series fit along with residual series and residual distribution."""
    kernels: List[str] = kernels_dict[str(best_idx)]
    residuals: Any = series - trend
    sigma: float = residuals.std()
    upper_lim: float = 3 * sigma
    lower_lim: float = -3 * sigma

    fig = plt.figure(figsize=figsize)
    gs: GridSpec = GridSpec(2, 4, figure=fig, height_ratios=[1.5, 1], wspace=0.15, hspace=0.15)

    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(series.index, series.values, color="grey", alpha=0.6, lw=1.5, label="Observed Data")
    ax1.plot(trend.index, trend.values, color="orange", alpha=0.9, lw=2, label="GP Fit")
    ax1.set_title(f"Time Series Fit | Kernels: {', '.join(kernels)}")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :3], sharex=ax1)
    ax3.plot(residuals.index, residuals.values, color="#1f77b4", lw=1, label="Residuals")
    ax3.axhspan(lower_lim, upper_lim, color="green", alpha=0.05)
    ax3.axhline(lower_lim, color="red", linestyle="--")
    ax3.axhline(upper_lim, color="red", linestyle="--")
    ax3.set_ylabel("Residual")
    ax3.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 3], sharey=ax3)
    ax2.hist(residuals.values, bins=20, orientation="horizontal", color="purple", alpha=0.5)
    ax2.axhline(lower_lim, color="red", linestyle="--")
    ax2.axhline(upper_lim, color="red", linestyle="--")
    ax2.set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_residuals_auto_correlation(
    series: Any,
    trend: Any,
    kernels_dict: Dict[str, List[str]],
    *,
    path: str,
    best_idx: int,
    lags: int = 40,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """Plot autocorrelation function (ACF) of residuals."""
    kernels: List[str] = kernels_dict[str(best_idx)]
    residuals: Any = series - trend

    fig, ax = plt.subplots(figsize=figsize)

    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(
        residuals.values,
        lags=lags,
        ax=ax,
        alpha=0.05,
        title=f"Residual Autocorrelation | Kernels: {', '.join(kernels)}",
    )

    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def residuals_auto_correlation_score(
    series: Any,
    trend: Any,
    kernels_dict: Dict[str, List[str]],
    *,
    best_idx: int,
    lags: int = 20,
) -> str:
    """Compute Ljung-Box autocorrelation test score for residuals."""
    kernels: List[str] = kernels_dict[str(best_idx)]
    residuals: Any = series - trend

    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_test = acorr_ljungbox(residuals.values, lags=[lags], return_df=True)

    statistic: float = float(lb_test["lb_stat"].iloc[0])
    p_value: float = float(lb_test["lb_pvalue"].iloc[0])

    if p_value > 0.05:
        interpretation: str = (
            "Residuals behave approximately like white noise with no significant autocorrelation."
        )
    else:
        interpretation = (
            "Residuals show statistically significant autocorrelation, "
            "indicating that the model likely missed temporal structure."
        )

    return (
        f"Residual Autocorrelation Diagnostic | "
        f"Kernels Used: {', '.join(kernels)} | "
        f"Ljung-Box Test Lag: {lags} | "
        f"Statistic: {statistic:.4f} | "
        f"P-Value: {p_value:.4f} | "
        f"Interpretation: {interpretation}"
    )


def get_dominant_period(series: Any) -> Optional[str]:
    """Robust Period Detector using Autocorrelation (ACF).

    Best for structural/non-sinusoidal signals (ECG, Square Waves).
    """
    from scipy.signal import correlate, detrend, find_peaks

    vals: np.ndarray = series.values.astype(float)
    vals = detrend(vals, type="linear")

    corr: np.ndarray = correlate(vals, vals, mode="full")
    corr = corr[len(corr) // 2 :]

    if corr[0] == 0:
        return None
    corr /= corr[0]

    peaks, props = find_peaks(corr, height=0.2, distance=10, prominence=0.1)

    if len(peaks) == 0:
        return None

    best_period: float = float(peaks[0])

    return (
        f"Dominant Period Detected: {best_period / len(series)}. "
        f"This suggests a strong periodic component in the data with a cycle "
        f"length of approximately {best_period / len(series)}."
    )
