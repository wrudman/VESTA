# processing_utils.py
"""Processing utilities: model fitting, parameter cleaning, and helpers."""

import ast
import codecs
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
from morphic import validate

from vesta.core.sandbox_namespaces import get_pymc_namespace
from morphic.string import format_exception_msg
from pymc.exceptions import SamplingError
from pymc.model.core import MODEL_MANAGER

logger: logging.Logger = logging.getLogger("processing_utils")

pytensor.config.mode = "NUMBA"


def _clear_pymc_contexts() -> None:
    """Clear all active PyMC model contexts in the current thread.

    PyMC 5 tracks model contexts in ``MODEL_MANAGER.active_contexts``.
    ``MODEL_MANAGER`` inherits from ``threading.local``, so this clears
    only the current thread's context stack and does not affect parallel
    model fits running in other threads of the same process.
    """
    active_context_count: int = len(MODEL_MANAGER.active_contexts)
    if active_context_count == 0:
        return
    MODEL_MANAGER.active_contexts.clear()
    logger.debug("Cleared %d active PyMC context(s).", active_context_count)


# ---------------------------------------------------------------------------
#  PyMC code sanitizer — fixes common LLM-generated code mistakes
# ---------------------------------------------------------------------------



FAMILY_CANONICAL = {

    # ── Gaussian / Normal ──────────────────────────────────────────────────────
    r"\bgaussian\b"
    r"|\bnormal\b"
    r"|\bnorm\b"
    r"|\bstandard[-\s]?normal\b"
    r"|\bnormal[-\s]?distribution\b"
    r"|\bbell[-\s]?curve\b"
    r"|\bbell[-\s]?shaped\b"
    r"|\bGaussian\b"
    r"|\bN\(.*\)\b": "gaussian",

    # ── Gaussian Mixture ───────────────────────────────────────────────────────
    r"\bgaussian[-_\s]?mixture\b"
    r"|\bmixture[-\s]?of[-\s]?gaussians\b"
    r"|\bmixture[-\s]?of[-\s]?normals\b"
    r"|\bnormal[-\s]?mixture\b"
    r"|\bGMM\b"
    r"|\bmog\b": "gaussian_mixture",

    # ── Uniform ────────────────────────────────────────────────────────────────
    r"\buniform\b"
    r"|\bflat\b"
    r"|\bflat[-\s]?prior\b"
    r"|\beven[-\s]?spread\b"
    r"|\brectangular\b"
    r"|\beven[-\s]?distribution\b"
    r"|\bunif\b": "uniform",

    # ── Exponential ───────────────────────────────────────────────────────────
    r"\bexponential\b"
    r"|\bexpon\b"
    r"|\bexp[-\s]?dist(ribution)?\b"
    r"|\bexponential[-\s]?distribution\b"
    r"|\bexp[-\s]?decay\b"
    r"|\bdecay[-\s]?distribution\b": "exponential",

    # ── Laplace / Double Exponential ──────────────────────────────────────────
    r"\blaplace\b"
    r"|\bdouble[-\s]?exponential\b"
    r"|\bdbl[-\s]?exp\b"
    r"|\bbilateral[-\s]?exponential\b"
    r"|\blaplace[-\s]?distribution\b": "laplace",

    # ── Log-Normal ────────────────────────────────────────────────────────────
    r"\blog[-\s]?normal\b"
    r"|\blognormal\b"
    r"|\blog[-\s]?norm\b"
    r"|\bln[-\s]?normal\b"
    r"|\bgalton\b"
    r"|\bgalton[-\s]?distribution\b"
    r"|\blog[-\s]?gaussian\b": "log-normal",

    # ── Student-t ─────────────────────────────────────────────────────────────
    r"\bstudent[-_\s]?t\b"
    r"|\bstudentt\b"
    r"|\bt[-\s]?distribution\b"
    r"|\bt[-\s]?dist\b"
    r"|\bstudent'?s[-\s]?t\b"
    r"|\bstudent[-\s]?t[-\s]?distribution\b"
    r"|\bheavy[-\s]?tailed[-\s]?normal\b"
    r"|\brobust[-\s]?normal\b"
    r"|\bstudent\b": "student-t",

    # ── Cauchy ────────────────────────────────────────────────────────────────
    r"\bcauchy\b"
    r"|\bcauchy[-\s]?distribution\b"
    r"|\blorentz(ian)?\b"
    r"|\bbreit[-\s]?wigner\b"
    r"|\bcauchy[-\s]?lorentz\b": "cauchy",

    # ── Weibull ───────────────────────────────────────────────────────────────
    r"\bweibull\b"
    r"|\bweibull[-\s]?distribution\b"
    r"|\bwbl\b"
    r"|\bweibul\b"
    r"|\bweilbull\b"
    r"|\bwibull\b": "weibull",

    # ── Pareto ────────────────────────────────────────────────────────────────
    r"\bpareto\b"
    r"|\bpareto[-\s]?distribution\b"
    r"|\bpareto[-\s]?i\b"
    r"|\b80[-/]20[-\s]?(rule|distribution)?\b": "pareto",

    # ── Pareto-II / Lomax ─────────────────────────────────────────────────────
    r"\bpareto[-\s]?ii\b"
    r"|\blomax\b"
    r"|\bpareto[-\s]?type[-\s]?ii\b": "pareto-ii",

    # ── Gamma ─────────────────────────────────────────────────────────────────
    r"\bgamma\b"
    r"|\bgamma[-\s]?distribution\b"
    r"|\berlang\b": "gamma",

    # ── Beta ──────────────────────────────────────────────────────────────────
    r"\bbeta\b"
    r"|\bbeta[-\s]?distribution\b"
    r"|\bbeta[-\s]?prime\b": "beta",

    # ── Beta-Prime ────────────────────────────────────────────────────────────
    r"\bbeta[-\s]?prime\b"
    r"|\bbetaprime\b"
    r"|\binverted[-\s]?beta\b"
    r"|\bpearson[-\s]?vi\b": "beta-prime",

    # ── Chi-Squared ───────────────────────────────────────────────────────────
    r"\bchi[-\s]?squared\b"
    r"|\bchi[-\s]?square\b"
    r"|\bchi2\b"
    r"|\bchi\^2\b"
    r"|\bχ2\b"
    r"|\bchisq\b"
    r"|\bchi[-\s]?sq\b": "chi-squared",

    # ── Rayleigh ──────────────────────────────────────────────────────────────
    r"\brayleigh\b"
    r"|\brayleigh[-\s]?distribution\b": "rayleigh",

    # ── Gumbel ────────────────────────────────────────────────────────────────
    r"\bgumbel[-\s]?(left|right|min|max)?\b"
    r"|\bextreme[-\s]?value\b"
    r"|\bgeneralized[-\s]?extreme[-\s]?value\b"
    r"|\bGEV\b"
    r"|\btype[-\s]?i[-\s]?extreme[-\s]?value\b"
    r"|\bfisher[-\s]?tippett\b": "gumbel",

    # ── Logistic ──────────────────────────────────────────────────────────────
    r"\blogistic\b"
    r"|\blogistic[-\s]?distribution\b"
    r"|\bsech[-\s]?squared\b": "logistic",

    # ── Power-law ─────────────────────────────────────────────────────────────
    r"\bpower[-\s]?law\b"
    r"|\bpowerlaw\b"
    r"|\bscale[-\s]?free\b"
    r"|\bpower[-\s]?function\b": "powerlaw",

    # ── Zipf ──────────────────────────────────────────────────────────────────
    r"\bzipf\b"
    r"|\bzipfian\b"
    r"|\bzipf[-\s]?mandelbrot\b"
    r"|\bzipf'?s[-\s]?law\b": "zipf",

    # ── Inverse Gaussian ──────────────────────────────────────────────────────
    r"\binv[-\s]?gauss(ian)?\b"
    r"|\binverse[-\s]?gaussian\b"
    r"|\bwald\b"
    r"|\bwald[-\s]?distribution\b"
    r"|\binverse[-\s]?normal\b": "inv-gaussian",

    # ── Poisson ───────────────────────────────────────────────────────────────
    r"\bpoisson\b"
    r"|\bpoisson[-\s]?distribution\b"
    r"|\bpois\b": "poisson",

    # ── Binomial ──────────────────────────────────────────────────────────────
    r"\bbinomial\b"
    r"|\bbinom\b"
    r"|\bbinomial[-\s]?distribution\b"
    r"|\bb\([0-9n,\s]+\)\b": "binomial",

    # ── Bernoulli ─────────────────────────────────────────────────────────────
    r"\bbernoulli\b"
    r"|\bbernoulli[-\s]?trial\b"
    r"|\bbinary[-\s]?distribution\b"
    r"|\bbinomial\(.*[,\s]?1\)\b": "bernoulli",

    # ── Geometric ─────────────────────────────────────────────────────────────
    r"\bgeometric\b"
    r"|\bgeometric[-\s]?distribution\b"
    r"|\bgeom\b": "geometric",

    # ── Negative Binomial ─────────────────────────────────────────────────────
    r"\bnegative[-\s]?binomial\b"
    r"|\bnbinom\b"
    r"|\bneg[-\s]?binom\b"
    r"|\bnegbinom\b"
    r"|\bpascal[-\s]?distribution\b"
    r"|\bpolya\b": "negative-binomial",

    # ── Hypergeometric ────────────────────────────────────────────────────────
    r"\bhypergeometric\b"
    r"|\bhypergeom\b"
    r"|\bhyper[-\s]?geometric\b": "hypergeometric",

    # ── Triangular ────────────────────────────────────────────────────────────
    r"\btriangular\b"
    r"|\btriangle[-\s]?distribution\b"
    r"|\btriang\b": "triangular",

    # ── Arcsine ───────────────────────────────────────────────────────────────
    r"\barcsine\b"
    r"|\barc[-\s]?sine\b"
    r"|\barcsin\b": "arcsine",

    # ── Cosine ────────────────────────────────────────────────────────────────
    r"\bcosine\b"
    r"|\bcos[-\s]?distribution\b": "cosine",

    # ── Log-Gamma ─────────────────────────────────────────────────────────────
    r"\bloggamma\b"
    r"|\blog[-\s]?gamma\b"
    r"|\blogamma\b": "loggamma",

    # ── Von Mises ─────────────────────────────────────────────────────────────
    r"\bvon[-\s]?mises\b"
    r"|\bvonmises\b"
    r"|\bcircular[-\s]?normal\b"
    r"|\bvon[-\s]?mises[-\s]?fisher\b": "von-mises",

    # ── Rice ──────────────────────────────────────────────────────────────────
    r"\brice\b"
    r"|\brician\b"
    r"|\brice[-\s]?distribution\b": "rice",

    # ── Anglit ────────────────────────────────────────────────────────────────
    r"\banglit\b": "anglit",

    # ── Maxwell ───────────────────────────────────────────────────────────────
    r"\bmaxwell\b"
    r"|\bmaxwell[-\s]?boltzmann\b"
    r"|\bmaxwell[-\s]?speed\b": "maxwell",

    # ── F-distribution ────────────────────────────────────────────────────────
    r"\bf[-\s]?distribution\b"
    r"|\bf[-\s]?dist\b"
    r"|\bfisher[-\s]?distribution\b"
    r"|\bfisher[-\s]?snedecor\b"
    r"|\bf[-\s]?ratio\b"
    r"|\bsnedecor'?s[-\s]?f\b": "f-distribution",

    # ── General Mixture / Composite ───────────────────────────────────────────
    r"\bmixture\b"
    r"|\bmixture[-\s]?of\b"
    r"|\bblend\b"
    r"|\bcombo\b"
    r"|\bcombination\b"
    r"|\bmixed\b"
    r"|\bcompound\b"
    r"|\bhierarchical[-\s]?mixture\b": "mixture",
}


def canonicalize_family_name(family_name):
    """
    Canonicalize an input family name (string or list) to a standardized form.
    Handles lists of families (e.g. mixtures).
    """
    if isinstance(family_name, list):
        return [canonicalize_family_name(f) for f in family_name]
    if not isinstance(family_name, str):
        raise TypeError(f"Expected str or list, got {type(family_name)}")
    family_name = family_name.strip().lower()
    for pattern, canonical in FAMILY_CANONICAL.items():
        if re.search(pattern, family_name, flags=re.IGNORECASE):
            return canonical
    return family_name



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


@validate
def unescape_broken_code_if_syntax_error(code: str) -> str:
    """Repair VLM-generated code that contains literal backslash-n escapes.

    Background:
      The code-gen prompt instructs the VLM to return a single-line JSON
      string with ``\\n`` for newlines.  Most responses come back with the
      ``\\n`` correctly JSON-escaped so that after parsing the string value
      contains real newlines.  A minority of responses double-escape, so
      the string value received by the pipeline literally contains the
      2-character sequence backslash + ``n`` where newlines belong.  When
      Python's ``exec()`` (or ``ast.parse``) sees that, it interprets
      ``\\`` as a statement line-continuation marker and then chokes on
      ``n``, raising ``SyntaxError: unexpected character after line
      continuation character``.

    Strategy:
      1. If ``ast.parse(code)`` succeeds, return unchanged.
      2. If it fails with ``SyntaxError`` **and** the source contains a
         literal backslash-n, try ``codecs.decode(code, 'unicode_escape')``
         and re-parse.  If the decoded text parses, log and return it.
      3. Otherwise return the original ``code`` so the downstream
         ``exec()`` call raises the original ``SyntaxError`` with the
         original source in the traceback (no silent mangling).

    This function is intentionally conservative: it only rewrites code
    that already fails to parse.  Well-formed multi-line code and code
    containing raw-string literals (``r"\\n"``) parse fine on step 1 and
    are returned unchanged.
    """
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    if "\\n" not in code:
        return code

    try:
        decoded: str = codecs.decode(code, "unicode_escape")
    except UnicodeDecodeError:
        return code

    try:
        ast.parse(decoded)
    except SyntaxError:
        return code

    logger.warning(
        "Auto-unescaped VLM-generated code: source failed to parse and "
        "contained literal backslash-n sequences; applied "
        "codecs.decode(..., 'unicode_escape') and the result parses cleanly."
    )
    return decoded


@validate
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

    # ── FIX: pm.Deterministic('name', <numeric>) -> name = <numeric> ──
    # pm.Deterministic requires a PyTensor tensor; passing a raw float/int
    # crashes with "AttributeError: 'float' object has no attribute 'type'".
    # Only match literal numeric values (e.g. 0.133, 42) — NOT expressions
    # with parentheses like pt.as_tensor_variable(...) which work correctly.
    det_pattern = re.compile(
        r"""(\s*)(\w+)\s*=\s*pm\.Deterministic\(\s*['"][^'"]*['"]\s*,\s*(-?[\d.]+(?:e[+-]?\d+)?)\s*\)""",
        re.IGNORECASE,
    )
    det_lines = code.split("\n")
    det_cleaned: List[str] = []
    for line in det_lines:
        m_det = det_pattern.match(line)
        if m_det:
            indent, var, value = m_det.group(1), m_det.group(2), m_det.group(3).strip()
            det_cleaned.append(f"{indent}{var} = {value}")
            fixes.append(f"pm.Deterministic('{var}', {value}) -> {var} = {value}")
        else:
            det_cleaned.append(line)
    code = "\n".join(det_cleaned)

    # ── FIX 1: Remove broken "var = # comment" lines (no RHS, just a comment) ──
    lines: List[str] = code.split("\n")
    cleaned_comments: List[str] = []
    for line in lines:
        stripped: str = line.strip()
        if re.match(r"^\w+\s*=\s*#", stripped):
            fixes.append(f"REMOVED (broken comment assignment): {stripped[:80]}")
            continue
        cleaned_comments.append(line)
    code = "\n".join(cleaned_comments)

    # ── Assign bare pm.*() calls ──
    lines = code.split("\n")
    cleaned_pass1: List[str] = []
    unassigned_removed: int = 0
    unassigned_assigned: int = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r"^pm\.\w+\(", stripped) and not re.match(r"^\w+\s*=\s*pm\.", stripped):
            name_match: Optional[re.Match] = re.search(r"pm\.\w+\(\s*['\"](\w+)['\"]", stripped)
            if name_match is not None:
                var_name: str = name_match.group(1)
                indent: str = line[: len(line) - len(line.lstrip())]
                assigned_line: str = f"{indent}{var_name} = {stripped}"
                cleaned_pass1.append(assigned_line)
                fixes.append(f"ASSIGNED (unassigned pm call): {stripped} -> {var_name} = ...")
                unassigned_assigned += 1
            else:
                fixes.append(f"REMOVED (unassigned pm call, no extractable name): {stripped}")
                unassigned_removed += 1
            continue
        cleaned_pass1.append(line)
    if unassigned_removed > 0 or unassigned_assigned > 0:
        code = "\n".join(cleaned_pass1)
        if unassigned_assigned > 0:
            fixes.append(f"Auto-assigned {unassigned_assigned} unassigned pm.*() call(s)")
        if unassigned_removed > 0:
            fixes.append(f"Removed {unassigned_removed} unassigned pm.*() call(s) with no extractable name")

    # ── FIX 2: Deduplicate — remove second occurrence of any "var = pm.*(" line ──
    lines = code.split("\n")
    seen_var_assignments: set = set()
    deduped: List[str] = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r"^(\w+)\s*=\s*pm\.", stripped)
        if m:
            var_name = m.group(1)
            if var_name in seen_var_assignments:
                fixes.append(f"REMOVED (duplicate assignment): {stripped[:80]}")
                continue
            seen_var_assignments.add(var_name)
        deduped.append(line)
    code = "\n".join(deduped)

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
        logger.debug(f"[sanitize_pymc_code] Applied {len(fixes)} fix(es):")
        for fix in fixes:
            logger.debug(f"  - {fix}")
    return code


# ---------------------------------------------------------------------------
#  Parameter cleaning
# ---------------------------------------------------------------------------


@validate
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


@validate
def fit_single_model(
    *,
    data: Union[np.ndarray, pd.Series],
    ans: Dict[str, Any],
    model_idx: str,
    model_code: str,
    task: str,
) -> Dict[str, Any]:
    """Execute and fit one generated PyMC model.

    Args:
        data: Observed dataset passed into the generated model namespace.
        ans: VLM proposal accumulator containing domain metadata for
            ``model_idx``.
        model_idx: Proposal key being fitted.
        model_code: Generated PyMC code for this proposal.
        task: Domain task string, either ``"fitting"`` or ``"time_series"``.

    Returns:
        A fit result dict with ``model``, ``map_estimate``, ``metrics``, and
        domain-specific metadata. Failed fits return ``model=None`` and an
        ``error`` string; unexpected infrastructure exceptions propagate after
        context cleanup.
    """
    logger.debug("=" * 60)
    logger.debug(f"FITTING MODEL {model_idx}")
    logger.debug("=" * 60)
    model_code = unescape_broken_code_if_syntax_error(model_code)
    model_code = sanitize_pymc_code(model_code)

    try:
        _clear_pymc_contexts()
        exec_namespace: Dict[str, Any] = get_pymc_namespace(data=data)
        try:
            exec(model_code, exec_namespace)
        except (NameError, TypeError, ValueError, AttributeError) as exec_error:
            _clear_pymc_contexts()
            logger.warning(
                f"Model {model_idx} exec failed: {format_exception_msg(exec_error)}\n"
                f"FAILING CODE:\n{model_code}"
            )
            raise RuntimeError(
                f"Model {model_idx} exec failed: {format_exception_msg(exec_error)}"
            ) from exec_error

        if "model" not in exec_namespace or exec_namespace["model"] is None:
            visible_keys: List[str] = [key for key in exec_namespace.keys() if not key.startswith("__")]
            raise RuntimeError(
                f"'model' not defined after exec for model {model_idx}. "
                f"Available namespace keys: {visible_keys}."
            )
        if "map_estimate" not in exec_namespace or exec_namespace["map_estimate"] is None:
            visible_keys = [key for key in exec_namespace.keys() if not key.startswith("__")]
            raise RuntimeError(
                f"'map_estimate' not defined after exec for model {model_idx}. "
                f"Available namespace keys: {visible_keys}."
            )

        model: Any = exec_namespace["model"]
        map_estimate: Dict[str, Any] = exec_namespace["map_estimate"]
        logger.debug(f"MAP ESTIMATE: {map_estimate}")
        with model:
            compiled_logp: Any = model.compile_logp()
            point: Dict[str, Any] = {value_var.name: map_estimate[value_var.name] for value_var in model.value_vars}
            log_likelihood: float = compiled_logp(point)

            if task == "time_series":
                if "trend" not in exec_namespace:
                    visible_keys = [key for key in exec_namespace.keys() if not key.startswith("__")]
                    raise RuntimeError(
                        f"'trend' not defined after exec for model {model_idx}. "
                        f"The PyMC code for time-series models must define a 'trend' variable. "
                        f"Available namespace keys: {visible_keys}."
                    )
                trend: Any = exec_namespace["trend"]

        num_pymc_parameters: int = sum(np.size(map_estimate[value_var.name]) for value_var in model.value_vars)
        aic: float = 2 * num_pymc_parameters - 2 * log_likelihood
        bic: float = -2 * log_likelihood + num_pymc_parameters * np.log(len(data))
        fit_metrics: Dict[str, float] = {
            "aic": aic,
            "bic": bic,
            "n_params": num_pymc_parameters,
        }

        if task == "fitting":
            fit_result: Dict[str, Any] = {
                "model": model,
                "map_estimate": map_estimate,
                "metrics": fit_metrics,
                "distribution_family": ans["distribution_family"][model_idx],
                "is_mixture": ans["is_mixture"][model_idx],
            }
        elif task == "time_series":
            fit_result = {
                "model": model,
                "map_estimate": map_estimate,
                "metrics": fit_metrics,
                "kernels": ", ".join(ans["kernels"][model_idx]),
                "trend": trend,
            }
        else:
            raise ValueError(f"Unexpected task={task!r}. Must be 'fitting' or 'time_series'.")

        logger.debug(f"Model {model_idx} fit metrics: {fit_metrics}")
        return fit_result
    except (SyntaxError, RuntimeError, SamplingError, ValueError) as exc:
        _clear_pymc_contexts()
        logger.error(f"MODEL {model_idx} FIT FAILED")
        logger.error(f"FAILING MODEL CODE:\n{model_code}")
        logger.error(f"Error: {format_exception_msg(exc)}")

        if task == "fitting":
            return {
                "model": None,
                "map_estimate": None,
                "metrics": {"aic": np.inf, "bic": np.inf, "n_params": None},
                "distribution_family": ans["distribution_family"][model_idx],
                "is_mixture": ans["is_mixture"][model_idx],
                "error": format_exception_msg(exc),
            }
        elif task == "time_series":
            return {
                "model": None,
                "map_estimate": None,
                "metrics": {"aic": np.inf, "bic": np.inf, "n_params": None},
                "kernels": ", ".join(ans["kernels"][model_idx]),
                "trend": None,
                "error": format_exception_msg(exc),
            }
        else:
            raise ValueError(f"Unexpected task={task!r}. Must be 'fitting' or 'time_series'.") from exc
    finally:
        _clear_pymc_contexts()


@validate
def select_best_fit_result(
    *,
    fit_results: Dict[str, Dict[str, Any]],
) -> Tuple[str, Any, Dict[str, Any], Dict[str, float]]:
    """Select the successful fit with the lowest AIC.

    Args:
        fit_results: Per-model fit results keyed by proposal index.

    Returns:
        ``(best_idx, model, map_estimate, metrics)`` for the best successful
        model.

    Raises:
        RuntimeError: If no model fitted successfully.
    """
    successful: Dict[str, Dict[str, Any]] = {
        model_idx: fit_result
        for model_idx, fit_result in fit_results.items()
        if fit_result["model"] is not None
    }
    if len(successful) == 0:
        raise RuntimeError(
            f"All {len(fit_results)} models failed to fit. "
            f"Model indices attempted: {list(fit_results.keys())}. "
            f"Errors: {[fit_result['error'] if 'error' in fit_result else 'unknown' for fit_result in fit_results.values()]}"
        )

    best_idx: str = min(successful.keys(), key=lambda model_idx: successful[model_idx]["metrics"]["aic"])
    best_result: Dict[str, Any] = successful[best_idx]
    logger.debug("=" * 60)
    logger.debug(f"BEST MODEL: {best_idx}")
    if "distribution_family" in best_result:
        logger.debug(f"Distribution family: {best_result['distribution_family']}")
        logger.debug(f"Is mixture: {best_result['is_mixture']}")
    elif "kernels" in best_result:
        logger.debug(f"Kernels: {best_result['kernels']}")
    else:
        raise ValueError(
            f"Best fit result for {best_idx!r} has no recognized domain metadata. "
            f"Keys: {list(best_result.keys())}."
        )
    logger.debug(f"AIC: {best_result['metrics']['aic']:.2f}")
    logger.debug("=" * 60)
    return (
        best_idx,
        best_result["model"],
        best_result["map_estimate"],
        best_result["metrics"],
    )


@validate
def execute_and_fit_models(
    *,
    data: Union[np.ndarray, pd.Series],
    ans: Dict[str, Any],
    task: str,
) -> Tuple[str, Any, Dict[str, Any], Dict[str, float], Dict[str, Any]]:
    """Fit all proposed PyMC models and return the best one based on AIC.

    Raises:
        RuntimeError: If all models fail to fit.
    """
    all_results: Dict[str, Dict[str, Any]] = {
        model_idx: fit_single_model(
            data=data,
            ans=ans,
            model_idx=model_idx,
            model_code=model_code,
            task=task,
        )
        for model_idx, model_code in ans["pymc_models"].items()
    }
    best_idx: str
    model: Any
    map_estimate: Dict[str, Any]
    metrics: Dict[str, float]
    best_idx, model, map_estimate, metrics = select_best_fit_result(fit_results=all_results)
    return best_idx, model, map_estimate, metrics, all_results


def sanitize_box_llm_model_code(raw_code: str) -> str:
    code = raw_code
    code = _extract_gen_model_body(code)
    code = _inject_data_coercion(code)
    code = _replace_data_references(code)
    code = _remove_multiline_call(code, "pm.sample_posterior_predictive")
    code = _remove_multiline_call(code, "pm.sample")
    code = re.sub(r"^\s*return\b.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*rng\w*\s*=.*$", "", code, flags=re.MULTILINE)
    code = _fix_rv_arithmetic(code)
    code = _inject_map_estimate(code)
    code = _strip_redundant_imports(code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def _extract_gen_model_body(code: str) -> str:
    lines = code.splitlines()
    func_start = None
    func_indent = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*def gen_model\s*\(", line):
            func_start = i
            func_indent = len(line) - len(line.lstrip())
            break
    if func_start is None:
        return code
    pre_imports = [
        ln.strip()
        for ln in lines[:func_start]
        if ln.strip().startswith(("import ", "from "))
    ]
    body_lines = []
    for line in lines[func_start + 1:]:
        stripped = line.strip()
        if stripped == "":
            body_lines.append("")
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= func_indent and stripped:
            break
        body_lines.append(line)
    body = textwrap.dedent("\n".join(body_lines))
    if pre_imports:
        return "\n".join(pre_imports) + "\n\n" + body
    return body


def _inject_data_coercion(code: str) -> str:
    # Only inject when the code accesses data by column name (gen_model style)
    if not re.search(r"observed_data\[", code):
        return code
    guard = textwrap.dedent("""\
        import pandas as _pd, numpy as _np
        if isinstance(data, _pd.DataFrame):
            _obs_array = data.iloc[:, 0].values.astype(float)
        elif isinstance(data, _pd.Series):
            _obs_array = data.values.astype(float)
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
            _obs_array = _np.array([list(d.values())[0] for d in data], dtype=float)
        else:
            _obs_array = _np.asarray(data, dtype=float).ravel()
        data = _pd.DataFrame({"observation": _obs_array})
    """)
    return guard + "\n" + code


def _replace_data_references(code: str) -> str:
    code = re.sub(r"\bobserved_data\b", "data", code)
    return code


def _remove_multiline_call(code: str, call_name: str) -> str:
    lines = code.splitlines()
    result = []
    depth = 0
    inside = False
    for line in lines:
        if not inside:
            if call_name in line:
                inside = True
                depth = line.count("(") - line.count(")")
                if depth <= 0:
                    inside = False
            else:
                result.append(line)
        else:
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                inside = False
    return "\n".join(result)


def _fix_rv_arithmetic(code: str) -> str:
    pattern = re.compile(
        r"^(\s*)"
        r"(\w+)"
        r"\s*=\s*"
        r"(pm\.\w+\([^)]*\))"
        r"\s*([+\-\*\/]\s*[\d\.]+)\s*$",
        re.MULTILINE,
    )
    def _rewrite(m):
        indent, var, pm_call, op = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        raw_name = f"{var}_rv"
        pm_call_raw = re.sub(r'(pm\.\w+\(")[^"]*(")', rf'\g<1>{raw_name}\2', pm_call, count=1)
        pm_call_raw = re.sub(r"(pm\.\w+\(')[^']*(')", rf"\g<1>{raw_name}\g<2>", pm_call_raw, count=1)
        return (
            f"{indent}{raw_name} = {pm_call_raw}\n"
            f"{indent}{var} = pm.Deterministic(\"{var}\", {raw_name} {op})"
        )
    return pattern.sub(_rewrite, code)


def _inject_map_estimate(code: str) -> str:
    if "map_estimate" in code:
        return code
    lines = code.splitlines()
    model_idx = None
    model_indent = 0
    for i, line in enumerate(lines):
        if re.search(r"with\s+pm\.Model\(", line):
            model_idx = i
            model_indent = len(line) - len(line.lstrip())
            break
    if model_idx is None:
        return code + "\nmap_estimate = pm.find_MAP()\n"
    body_indent = model_indent + 4
    last_body_idx = model_idx
    for i in range(model_idx + 1, len(lines)):
        ln = lines[i]
        if ln.strip() == "":
            continue
        if (len(ln) - len(ln.lstrip())) > model_indent:
            last_body_idx = i
        else:
            break
    map_line = " " * body_indent + "map_estimate = pm.find_MAP()"
    lines.insert(last_body_idx + 1, map_line)
    return "\n".join(lines)


def _strip_redundant_imports(code: str) -> str:
    already_available = {"pm", "np", "pymc", "numpy", "pandas", "pd"}
    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            # Always preserve imports that use a private underscore alias
            # (e.g. `import pandas as _pd`) — these are injected internally
            if re.search(r"\bas\s+_\w+", stripped):
                cleaned.append(line)
                continue
            tokens = re.findall(r"\b\w+\b", stripped)
            if any(t in already_available for t in tokens):
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


    
PYMC_DISTRIBUTIONS = {
    "normal":                       ["Normal"],
    "studentt":                     ["StudentT"],
    "halfnormal":                   ["HalfNormal"],
    "halfstudent":                  ["HalfStudentT"],
    "cauchy":                       ["Cauchy"],
    "halfcauchy":                   ["HalfCauchy"],
    "laplace":                      ["Laplace"],
    "asymmetriclaplace":            ["AsymmetricLaplace"],
    "beta":                         ["Beta"],
    "gamma":                        ["Gamma"],
    "inversegamma":                 ["InverseGamma"],
    "exponential":                  ["Exponential"],
    "lognormal":                    ["LogNormal"],
    "logitnormal":                  ["LogitNormal"],
    "skewnormal":                   ["SkewNormal"],
    "weibull":                      ["Weibull"],
    "uniform":                      ["Uniform"],
    "triangular":                   ["Triangular"],
    "gumbel":                       ["Gumbel"],
    "logistic":                     ["Logistic"],
    "pareto":                       ["Pareto"],
    "vonmises":                     ["VonMises"],
    "rice":                         ["Rice"],
    "moyal":                        ["Moyal"],
    "chisquared":                   ["ChiSquared"],
    "flat":                         ["Flat"],
    "halfflat":                     ["HalfFlat"],
    "poisson":                      ["Poisson"],
    "negativebinomial":             ["NegativeBinomial"],
    "binomial":                     ["Binomial"],
    "bernoulli":                    ["Bernoulli"],
    "geometric":                    ["Geometric"],
    "zeroinflatedpoisson":          ["ZeroInflatedPoisson"],
    "zeroinflatedbinomial":         ["ZeroInflatedBinomial"],
    "zeroinflatednegativebinomial": ["ZeroInflatedNegativeBinomial"],
    "hurdle_poisson":               ["HurdlePoisson"],
    "betabinomial":                 ["BetaBinomial"],
    "categorical":                  ["Categorical"],
    "discreteuniform":              ["DiscreteUniform"],
    "discreteweibull":              ["DiscreteWeibull"],
    "dirichlet":                    ["Dirichlet"],
    "multinomial":                  ["Multinomial"],
    "dirichletmultinomial":         ["DirichletMultinomial"],
    "mv_normal":                    ["MvNormal", "MatrixNormal"],
}

_REVERSE_LOOKUP: dict[str, str] = {
    cls: canonical
    for canonical, class_names in PYMC_DISTRIBUTIONS.items()
    for cls in class_names
}

_MIXTURE_RE  = re.compile(r"\bpm\.Mixture\s*\(")

# Matches:  varname = pm.SomeDistribution.dist(
_COMP_ASSIGN_RE = re.compile(
    r"(\w+)\s*=\s*pm\.([A-Z][A-Za-z]+)\.dist\s*\("
)

# Matches inline pm.Foo.dist( inside comp_dists=[...]
_INLINE_COMP_RE = re.compile(r"pm\.([A-Z][A-Za-z]+)\.dist\s*\(")

# Matches comp_dists=[var1, var2, ...] (variable names, not inline calls)
_COMP_DISTS_VARS_RE = re.compile(r"comp_dists\s*=\s*\[([^\]]+)\]")


def extract_model_properties(code: str) -> dict:
    """
    Returns:
      - is_mixture:            bool
      - mixture_count:         int
      - distribution_families: list of canonical family names for the
                               MIXTURE COMPONENTS ONLY (not priors)
    """
    mixture_matches = _MIXTURE_RE.findall(code)
    is_mixture      = len(mixture_matches) > 0
    mixture_count   = len(mixture_matches)

    families: list[str] = []

    if is_mixture:
        # Strategy 1: variable assignments used as components
        #   e.g. comp1 = pm.StudentT.dist(...)
        #        comp_dists=[comp1, comp2]
        assigned: dict[str, str] = {}  # varname → canonical family
        for var, cls_name in _COMP_ASSIGN_RE.findall(code):
            canonical = _REVERSE_LOOKUP.get(cls_name)
            if canonical:
                assigned[var] = canonical

        comp_dists_match = _COMP_DISTS_VARS_RE.search(code)
        if comp_dists_match:
            var_names = [v.strip() for v in comp_dists_match.group(1).split(",")]
            for var in var_names:
                # Could be a variable name OR an inline pm.Foo.dist( call
                inline = _INLINE_COMP_RE.match(var)
                if inline:
                    canonical = _REVERSE_LOOKUP.get(inline.group(1))
                elif var in assigned:
                    canonical = assigned[var]
                else:
                    canonical = None
                if canonical:
                    families.append(canonical)

        # Strategy 2: inline comp_dists=[pm.Foo.dist(...), pm.Bar.dist(...)]
        # (fires when strategy 1 yields nothing)
        if not families:
            for cls_name in _INLINE_COMP_RE.findall(code):
                canonical = _REVERSE_LOOKUP.get(cls_name)
                if canonical:
                    families.append(canonical)

    return {
        "is_mixture":            is_mixture,
        "distribution_families": families,
    }