"""Distribution fitting domain — plotting, scipy distribution construction, and fit state extraction."""

import logging
import re
import traceback
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from morphic import validate
from morphic.string import format_exception_msg
from scipy import stats

from domains import _MATPLOTLIB_LOCK, DomainPlotting, FitState
from domains.distribution_fitting import DOMAIN_ALIASES

logger: logging.Logger = logging.getLogger("domains.distribution_fitting.plotting")


def _get_exponential_rate(params: Dict[str, float]) -> float:
    """Extract the rate parameter from an exponential distribution's MAP estimate.

    PyMC uses ``lam`` (lambda). The VLM may also generate ``rate`` or ``lambda``.
    Tries all known names; falls back to the single parameter if only one exists.
    """
    for key in ("rate", "lam", "lambda"):
        if key in params:
            return float(params[key])
    if len(params) == 1:
        return float(next(iter(params.values())))
    raise KeyError(
        f"Cannot find rate/lam/lambda in exponential params. Available keys: {list(params.keys())}"
    )


DISTRIBUTION_MAP: Dict[str, Dict[str, Any]] = {
    "gaussian": {"scipy_name": "norm", "params": {"loc": "mu", "scale": "sigma"}},
    "normal": {
        "scipy_name": "norm",
        "params": {"loc": "mu", "scale": "sigma"},
    },
    "lognormal": {"scipy_name": "lognorm", "params": {"s": "sigma", "scale": lambda p: np.exp(p["mu"])}},
    "cauchy": {"scipy_name": "cauchy", "params": {"loc": "alpha", "scale": "beta"}},
    "laplace": {"scipy_name": "laplace", "params": {"loc": "mu", "scale": "b"}},
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
        "params": {"scale": lambda p: 1 / _get_exponential_rate(p)},
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


@validate
def normalize_family_name(family_name: str) -> str:
    """Normalize a single family name (string) to a DISTRIBUTION_MAP key.

    Callers are responsible for iterating over ``FitState.family_name``
    (which is now canonically ``List[str]``) and passing one component
    at a time.  Accepting a list here would silently collapse mixtures
    to their first component; that was the old behaviour and it was
    wrong.
    """
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


@validate
def get_scipy_distribution(
    *,
    family_name: str,
    extracted_params: Dict[str, float],
) -> Any:
    """Get scipy distribution object with proper parameters.

    Args:
        family_name: A single distribution component name (e.g.
            ``"gaussian"``). For mixtures, iterate over
            ``FitState.family_name`` and call this once per component.
        extracted_params: Dict mapping parameter names to values.

    Raises:
        ValueError: If the distribution is not supported or params are missing.
    """
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


_PYMC_TRANSFORM_SUFFIXES: Tuple[str, ...] = (
    "_log__",
    "_logodds__",
    "_interval__",
    "_ordered__",
    "_sum_to_zero__",
    "_circular__",
    "_simplex__",
    "_chain__",
)


def _is_pymc_transform_key(key: str) -> bool:
    """True if the key corresponds to a PyMC unconstrained-space transform.

    Any key ending in one of PyMC's reserved transform suffixes (``_log__``,
    ``_interval__``, ``_ordered__``, etc.) is unconstrained-real-valued; using
    it as a constrained-space parameter silently corrupts downstream plots.
    Filter these out aggressively — checking only ``_log__`` (the old behavior)
    missed ``_interval__`` (bounded priors) and ``_ordered__`` (ordered RVs).
    """
    if "simplex" in key.lower():
        return True
    return any(key.endswith(suffix) for suffix in _PYMC_TRANSFORM_SUFFIXES)


_COMPONENT_INDEX_PATTERN: re.Pattern = re.compile(r"_(\d+)(?:_|$)")


def _component_indices_in_key(key: str) -> List[int]:
    """Return every integer ``i`` appearing as a ``_i`` or ``_i_`` token.

    Uses a regex with explicit boundaries so component 1 never matches
    component 10's keys (the old ``f"_{idx}" in key`` substring check did).
    """
    return [int(match) for match in _COMPONENT_INDEX_PATTERN.findall(key)]


def _has_component_suffix(key: str, component_idx: int) -> bool:
    """True iff ``key`` carries the token ``_{component_idx}`` (boundary-safe)."""
    return component_idx in _component_indices_in_key(key)


def _has_any_component_suffix(key: str) -> bool:
    """True iff ``key`` carries *any* numeric component suffix like ``_0`` or ``_1``."""
    return len(_component_indices_in_key(key)) > 0


@validate
def extract_distribution_params(
    *,
    map_estimate: Dict[str, Any],
    family_name: str,
    component_idx: int,
) -> Dict[str, float]:
    """Extract distribution parameters from MAP estimate.

    Args:
        map_estimate: MAP parameter estimates dict.
        family_name: A single component family name (e.g. ``"gaussian"``).
            Callers iterate over ``FitState.family_name`` (List[str]) and
            pass one element per call together with the matching
            ``component_idx``.
        component_idx: Which component to extract (0-indexed).
    """
    params: Dict[str, float] = {}

    family_normalized: str = family_name.lower().replace("-", "_")
    family_no_sep: str = family_normalized.replace("_", "")

    family_prefixes: List[str] = [family_normalized, family_no_sep]
    if "student" in family_normalized:
        family_prefixes.append("student")

    for key, value in map_estimate.items():
        if key in ("w", "weights") or _is_pymc_transform_key(key):
            continue

        key_lower: str = key.lower()

        if _has_component_suffix(key, component_idx):
            if any(prefix in key_lower for prefix in family_prefixes):
                param_name: Optional[str] = extract_param_name_from_key(
                    key=key,
                    family_normalized=family_normalized,
                    component_idx=component_idx,
                )
                if param_name is not None:
                    params[param_name] = float(np.atleast_1d(value).item())

        elif component_idx == 0 and not _has_any_component_suffix(key):
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
    "b",
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


@validate
def plot_best_fit(
    *,
    data: np.ndarray,
    map_estimate: Dict[str, Any],
    model_info: Dict[str, Any],
    path: str,
    best_idx: Union[str, int],
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot histogram of data with fitted model PDF overlay.

    Handles single distributions and mixtures. distribution_family values
    may be lists (e.g. ['gaussian', 'lognormal']) or strings.

    Uses _MATPLOTLIB_LOCK for thread safety in parallel runs.
    """
    with _MATPLOTLIB_LOCK:
        plt.figure(figsize=figsize)
        plt.hist(data, bins=75, density=True, alpha=0.6, color="lightblue", edgecolor="black", label="Data")

        x: np.ndarray = np.linspace(data.min() - 1, data.max() + 1, 1000)

        dist_family_raw: List[str] = list(model_info["distribution_family"][best_idx])
        is_mixture_val: Union[bool, str] = model_info["is_mixture"][best_idx]

        families: List[str] = dist_family_raw
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
            raise ValueError(
                f"No mixture weights found in map_estimate for {n_components}-component mixture. "
                f"Tried keys 'w', 'weights', and 'w_*'. Available keys: {list(map_est.keys())}"
            )

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
                            f"No parameters extracted for {component_family} component {component_idx}. "
                            f"Available keys: {[k for k in map_estimate.keys() if component_family in k.lower()]}"
                        )
                        continue

                    logger.info(f"Component {component_idx} ({component_family}): {extracted_params}")

                    dist: Any = get_scipy_distribution(
                        family_name=component_family,
                        extracted_params=extracted_params,
                    )
                    component_pdf: np.ndarray = weight * dist.pdf(x)
                    total_pdf += component_pdf

                except (ValueError, TypeError, RuntimeError) as exc:
                    logger.warning(
                        f"Could not plot {component_family} component: {format_exception_msg(exc)}"
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
                        f"No parameters extracted for {single_family}. "
                        f"Available keys: {list(map_estimate.keys())}"
                    )
                    plt.title("Data (Parameter extraction failed)", fontsize=24)
                    plt.savefig(path)
                    plt.close()
                    return

                logger.info(f"Single distribution ({single_family}): {extracted_params}")

                dist = get_scipy_distribution(family_name=single_family, extracted_params=extracted_params)
                pdf: np.ndarray = dist.pdf(x)
                plt.plot(x, pdf, "-", linewidth=2.5, color="red", label="Predicted Model Fit")

            except (ValueError, TypeError, RuntimeError) as exc:
                logger.warning(f"Could not plot {single_family}: {format_exception_msg(exc)}")
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


@validate
def plot_hist(data: np.ndarray, *, save_path: str = "fit.png") -> None:
    """Plot histogram of data.

    Uses _MATPLOTLIB_LOCK for thread safety in parallel runs.
    """
    with _MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            data,
            bins=75,
            density=True,
            alpha=0.5,
            label="Data",
            color="tab:blue",
            edgecolor="black",
            linewidth=1.0,
        )
        ax.grid(alpha=0.5)
        fig.savefig(save_path)
        plt.close(fig)


class DistFittingFitState(FitState):
    """Fit state for distribution fitting: MAP estimate + VLM response accumulator."""

    ans: Dict[str, Any]


class DistributionFittingPlotting(DomainPlotting):
    """Visualization for 1-D distribution fitting: histograms + PDF overlays."""

    aliases: ClassVar[List[str]] = DOMAIN_ALIASES

    def plot_initial_data(self, data: Any, *, save_path: str) -> None:
        plot_hist(data, save_path=save_path)

    def plot_fit_overlay(
        self,
        *,
        data: Any,
        fit_state: FitState,
        path: str,
        best_idx: Union[str, int],
    ) -> None:
        plot_best_fit(
            data=data,
            map_estimate=fit_state.map_estimate,
            model_info=fit_state.ans,
            best_idx=best_idx,
            path=path,
        )

    def get_default_plot_description(self) -> str:
        return "histogram"

    def extract_fit_state(
        self,
        *,
        fit_results: Dict[str, Any],
        best_idx: Union[str, int],
        ans: Dict[str, Any],
    ) -> DistFittingFitState:
        raw_family: Any = ans["distribution_family"][best_idx]
        if not isinstance(raw_family, list):
            raise TypeError(
                f"ans['distribution_family'][{best_idx!r}] must be a List[str]; "
                f"got {type(raw_family).__name__}: {raw_family!r}. "
                f"The VLM-facing prompt requires distribution_family to be a list "
                f"(even for single distributions)."
            )
        return DistFittingFitState(
            map_estimate=fit_results[best_idx]["map_estimate"],
            ans=ans,
            family_name=list(raw_family),
        )
