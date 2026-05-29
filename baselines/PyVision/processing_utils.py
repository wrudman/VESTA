# processing_utils.py
"""Processing utilities: model fitting, parameter cleaning, and helpers."""

import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pymc as pm
import pytensor
from morphic.string import format_exception_msg
from pymc.exceptions import SamplingError

from sandbox_namespaces import get_pymc_namespace

logger: logging.Logger = logging.getLogger("processing_utils")

pytensor.config.mode = "NUMBA"
# ---------------------------------------------------------------------------
#  PyMC code sanitizer — fixes common LLM-generated code mistakes
# ---------------------------------------------------------------------------

_DIST_NAMES: List[str] = [
    "Normal",
    "HalfNormal",
    "LogNormal",
    "StudentT",
    "Cauchy",
    "Laplace",
    "Exponential",
    "Uniform",
    "Weibull",
    "HalfCauchy",
    "Gamma",
    "Beta",
    "Dirichlet",
    "Mixture",
]


def sanitize_pymc_code(code: str) -> str:
    """Fix systematic mistakes that VLMs make when generating PyMC code."""
    fixes: List[str] = []

    if "pm.Gaussian" in code:
        code = code.replace("pm.Gaussian", "pm.Normal")
        fixes.append("pm.Gaussian -> pm.Normal")

    if ", sd=" in code or "(sd=" in code:
        code = code.replace(", sd=", ", sigma=").replace("(sd=", "(sigma=")
        fixes.append("sd= -> sigma=")

    for dist in _DIST_NAMES:
        pattern_sq: str = rf"pm\.{dist}\.dist\(\s*'[^']*'\s*,"
        pattern_dq: str = rf'pm\.{dist}\.dist\(\s*"[^"]*"\s*,'
        if re.search(pattern_sq, code) or re.search(pattern_dq, code):
            code = re.sub(pattern_sq, f"pm.{dist}.dist(", code)
            code = re.sub(pattern_dq, f"pm.{dist}.dist(", code)
            fixes.append(f"pm.{dist}.dist('name', ...) -> pm.{dist}.dist(...)")

    if re.search(r"\.dist\([^)]*observed\s*=", code):
        code = re.sub(
            r"(\w+)\s*=\s*pm\.(\w+)\.dist\(([^)]*),\s*observed\s*=\s*(\w+)\s*\)",
            r"\1 = pm.\2('\1', \3, observed=\4)",
            code,
        )
        code = re.sub(
            r"(\w+)\s*=\s*pm\.(\w+)\.dist\(\s*observed\s*=\s*(\w+)\s*\)",
            r"\1 = pm.\2('\1', observed=\3)",
            code,
        )
        fixes.append(".dist(..., observed=data) -> ('name', ..., observed=data)")

    lines: List[str] = code.split("\n")
    cleaned_pass1: List[str] = []
    unassigned_removed: int = 0
    for line in lines:
        stripped: str = line.strip()
        if re.match(r"^pm\.\w+\(", stripped) and not re.match(r"^\w+\s*=\s*pm\.", stripped):
            fixes.append(f"REMOVED (unassigned pm call): {stripped}")
            unassigned_removed += 1
            continue
        cleaned_pass1.append(line)

    if unassigned_removed:
        code = "\n".join(cleaned_pass1)
        fixes.append(f"Removed {unassigned_removed} unassigned pm.*() duplicate declaration(s)")

    lines = code.split("\n")
    cleaned: List[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r".*=\s*pm\.Prior\(", stripped):
            fixes.append(f"REMOVED (pm.Prior): {stripped}")
            continue
        if re.match(r".*_val\s*=.*\+\s*\d", stripped) and "pm." not in stripped:
            fixes.append(f"REMOVED (val arithmetic): {stripped}")
            continue
        if re.match(r"\s*if\s+\w+_val\s*<\s*\d", stripped):
            fixes.append(f"REMOVED (val guard): {stripped}")
            continue
        cleaned.append(line)

    code = "\n".join(cleaned)

    if len(fixes) > 0:
        logger.info("[sanitize_pymc_code] Applied %d fix(es):", len(fixes))
        for fix in fixes:
            logger.info("  - %s", fix)

    return code


# ---------------------------------------------------------------------------
#  Parameter cleaning
# ---------------------------------------------------------------------------


def clean_params(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays in a MAP estimate dict to JSON-friendly types."""
    clean: Dict[str, Any] = {}
    for k, v in params_dict.items():
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                clean[k] = v.item()
            elif v.size == 1:
                clean[k] = v.flat[0].item() if hasattr(v.flat[0], "item") else float(v.flat[0])
            else:
                clean[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


# ---------------------------------------------------------------------------
#  Model fitting
# ---------------------------------------------------------------------------


def execute_and_fit_models(
    data: np.ndarray,
    ans: Dict[str, Any],
    task: str,
) -> Tuple[str, Any, Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """Fit all proposed PyMC models and return the best one based on AIC.

    Raises:
        RuntimeError: If all models fail to fit.
    """

    all_results: Dict[str, Dict[str, Any]] = {}

    def _drain_pymc_context() -> None:
        """Pop any stale PyMC model contexts left by a failed exec."""
        while pm.Model.get_context(error_if_none=False) is not None:
            try:
                pm.Model.get_context().pop_context()
            except (RuntimeError, SamplingError):
                break

    for idx, model_code in ans["pymc_models"].items():
        logger.info("=" * 60)
        logger.info("FITTING MODEL %s", idx)
        logger.info("=" * 60)

        try:
            _drain_pymc_context()

            exec_namespace: Dict[str, Any] = get_pymc_namespace(data=data)

            model_code = sanitize_pymc_code(model_code)

            try:
                exec(model_code, exec_namespace)
            except SamplingError:
                logger.warning("SamplingError on initial MAP, retrying with jittered starts...")
                logger.warning("FAILING MODEL CODE:\n%s", model_code)
                _drain_pymc_context()

            if "model" not in exec_namespace or exec_namespace["model"] is None:
                raise RuntimeError(
                    f"'model' not defined after exec for model {idx}. Check the generated code."
                )
            if "map_estimate" not in exec_namespace or exec_namespace["map_estimate"] is None:
                raise RuntimeError(
                    f"'map_estimate' not defined after exec for model {idx}. Check the generated code."
                )

            model: Any = exec_namespace["model"]
            map_estimate: Dict[str, Any] = exec_namespace["map_estimate"]

            logger.info("MAP ESTIMATE: %s", map_estimate)
            with model:
                f_logp = model.compile_logp()
                point: Dict[str, Any] = {v.name: map_estimate[v.name] for v in model.value_vars}
                log_likelihood: float = f_logp(point)

                if task == "time_series":
                    trend = exec_namespace.get("trend")

            num_pymc_parameters: int = sum(np.size(map_estimate[v.name]) for v in model.value_vars)
            aic: float = 2 * num_pymc_parameters - 2 * log_likelihood
            bic: float = -2 * log_likelihood + num_pymc_parameters * np.log(len(data))

            fit_metrics: Dict[str, float] = {
                "aic": aic,
                "bic": bic,
                "n_params": num_pymc_parameters,
            }

            if task == "fitting":
                all_results[idx] = {
                    "model": model,
                    "map_estimate": map_estimate,
                    "metrics": fit_metrics,
                    "distribution_family": ans["distribution_family"][idx],
                    "is_mixture": ans["is_mixture"][idx],
                }
            elif task == "time_series":
                all_results[idx] = {
                    "model": model,
                    "map_estimate": map_estimate,
                    "metrics": fit_metrics,
                    "kernels": ", ".join(ans["kernels"][idx]),
                    "trend": trend,
                }
            else:
                raise ValueError(f"Unexpected task={task!r}. Must be 'fitting' or 'time_series'.")

            logger.info("Model %s fit metrics: %s", idx, fit_metrics)

        except (SyntaxError, RuntimeError, SamplingError, ValueError) as exc:
            _drain_pymc_context()
            logger.error("MODEL %s FIT FAILED", idx)
            logger.error("FAILING MODEL CODE:\n%s", model_code)
            logger.error("Error: %s", format_exception_msg(exc))

            if task == "fitting":
                all_results[idx] = {
                    "model": None,
                    "map_estimate": None,
                    "metrics": {"aic": np.inf, "bic": np.inf, "n_params": None},
                    "distribution_family": ans["distribution_family"][idx],
                    "is_mixture": ans["is_mixture"][idx],
                    "error": format_exception_msg(exc),
                }
            elif task == "time_series":
                all_results[idx] = {
                    "model": None,
                    "map_estimate": None,
                    "metrics": {"aic": np.inf, "bic": np.inf, "n_params": None},
                    "kernels": ", ".join(ans["kernels"][idx]),
                    "trend": None,
                    "error": format_exception_msg(exc),
                }

    successful: Dict[str, Dict[str, Any]] = {k: v for k, v in all_results.items() if v["model"] is not None}
    if len(successful) == 0:
        raise RuntimeError(
            f"All {len(all_results)} models failed to fit. "
            f"Model indices attempted: {list(all_results.keys())}. "
            f"Errors: {[v.get('error', 'unknown') for v in all_results.values()]}"
        )

    best_idx: str = min(all_results.keys(), key=lambda k: all_results[k]["metrics"]["aic"])
    best_result: Dict[str, Any] = all_results[best_idx]

    logger.info("=" * 60)
    logger.info("BEST MODEL: %s", best_idx)

    if task == "fitting":
        logger.info("Distribution family: %s", best_result["distribution_family"])
        logger.info("Is mixture: %s", best_result["is_mixture"])
    elif task == "time_series":
        logger.info("Kernels: %s", best_result["kernels"])

    logger.info("AIC: %.2f", best_result["metrics"]["aic"])
    logger.info("=" * 60)

    return (
        best_idx,
        best_result["model"],
        best_result["map_estimate"],
        best_result["metrics"],
        all_results,
    )
