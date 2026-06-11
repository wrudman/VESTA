"""Distribution fitting domain — toolkit dispatch, tool Registry, and tool implementations.

Tool Architecture (see ``domains/__init__.py`` module docstring for full details):

    ``DistributionFittingTool(Tool, Registry, ABC)`` is this domain's tool Registry.
    Each concrete tool (e.g., ``QQPlot``, ``CalculateMoments``) subclasses it.
    Morphic auto-registers them under their snake_case class name, so
    ``DistributionFittingTool.of("qq_plot")`` resolves ``QQPlot``.

    ``DistributionFittingToolkit(DomainToolkit)`` is the dispatch class called by
    ``experiments.py``.  Its ``execute_tool()`` delegates to
    ``DistributionFittingTool.of(selected_tool).execute(...)`` — no if/elif chain.

    To add a new distribution-fitting tool:
        1. Define a class that subclasses ``DistributionFittingTool``.
        2. Set ``tool_description``, ``output_type``, ``parameters_schema``.
        3. Implement ``execute()``.
        That's it — no schema constants, no dispatch branches, no wiring.

Contains:
- ``DistributionBuilder`` — MAP estimate → callable PDF construction
- Statistical diagnostic functions (``compute_moments``, ``factorize_gmm``)
- Diagnostic plot functions (``plot_qq``, ``plot_probability``, ``plot_tail_transforms``)
- ``DistributionFittingTool`` — Tool Registry + 5 concrete tool subclasses
- ``DistributionFittingToolkit`` — DomainToolkit dispatch class
"""

import json
import logging
from abc import ABC
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from morphic import Registry, validate
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture

from vesta.domains import _MATPLOTLIB_LOCK, DiagnosticArtifact, DiagnosticToolResult, DomainToolkit, Tool
from vesta.domains.distribution_fitting import DOMAIN_ALIASES
from vesta.domains.distribution_fitting.plotting import DistFittingFitState, plot_best_fit

logger: logging.Logger = logging.getLogger("domains.distribution_fitting.toolkit")


class DistributionBuilder:
    """Build PDF functions from MAP estimates.

    Canonical input form: ``components: List[str]``, matching
    ``FitState.family_name``.  Each element is a PyMC-named family
    (e.g. ``"gaussian"``, ``"student_t"``, ``"lognormal"``).  Never
    re-split or re-join this list — doing so is lossy for multi-token
    family names like ``student_t``.
    """

    @staticmethod
    def build_gaussian_pdf(mu: float, sigma: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Gaussian PDF function."""
        return lambda x: stats.norm.pdf(x, loc=mu, scale=sigma)

    @staticmethod
    def build_cauchy_pdf(loc: float, scale: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Cauchy PDF function."""
        return lambda x: stats.cauchy.pdf(x, loc=loc, scale=scale)

    @staticmethod
    def build_laplace_pdf(loc: float, scale: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Laplace PDF function."""
        return lambda x: stats.laplace.pdf(x, loc=loc, scale=scale)

    @staticmethod
    def build_student_t_pdf(loc: float, scale: float, df: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Student's t PDF function."""
        return lambda x: stats.t.pdf(x, df=df, loc=loc, scale=scale)

    @staticmethod
    def build_lognormal_pdf(mu: float, sigma: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Lognormal PDF function.

        Args:
            mu: Mean of underlying normal distribution
            sigma: Standard deviation of underlying normal distribution
        """
        return lambda x: stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    @staticmethod
    def build_exponential_pdf(loc: float, scale: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Exponential PDF function.

        Args:
            loc: Location parameter (shift)
            scale: Scale parameter (1/rate)
        """
        return lambda x: stats.expon.pdf(x, loc=loc, scale=scale)

    @staticmethod
    def build_uniform_pdf(low: float, high: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Uniform PDF function.

        Args:
            low: Lower bound
            high: Upper bound
        """
        return lambda x: stats.uniform.pdf(x, loc=low, scale=high - low)

    @staticmethod
    def build_weibull_pdf(loc: float, scale: float, alpha: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Weibull PDF function.

        Args:
            loc: Location parameter (shift)
            scale: Scale parameter
            alpha: Shape parameter
        """
        return lambda x: stats.weibull_min.pdf(x, c=alpha, loc=loc, scale=scale)

    @staticmethod
    def build_pareto_pdf(alpha: float, m: float) -> Callable[[np.ndarray], np.ndarray]:
        """Build Pareto PDF function.

        Matches PyMC's parameterisation: pm.Pareto(alpha=alpha, m=m).
        The scipy equivalent is stats.pareto(b=alpha, scale=m, loc=0),
        giving the PDF  f(x) = alpha * m^alpha / x^(alpha+1)  for x >= m.

        Args:
            alpha: Shape parameter / tail index (must be > 0).
                   Larger alpha = lighter tail (faster decay).
            m:     Scale parameter / minimum value (must be > 0).
                   The distribution has zero density for x < m.
        """
        return lambda x: stats.pareto.pdf(x, b=alpha, scale=m, loc=0)

    @classmethod
    @validate
    def extract_component_params(
        cls, *, map_estimate: Dict[str, Any], component_name: str, index: int
    ) -> Dict[str, float]:
        """Extract parameters for a specific distribution component.

        Args:
            map_estimate: Dictionary of MAP estimates
            component_name: Name of distribution (e.g., 'gaussian', 'cauchy')
            index: Component index (0, 1, 2, ...)

        Returns:
            Dictionary of parameters for that component
        """
        params: Dict[str, float] = {}

        component_name = component_name.lower().replace("-", "_")
        if component_name == "normal":
            component_name = "gaussian"
        elif component_name == "studentt":
            component_name = "student_t"

        if component_name == "gaussian":
            mu_key: str = f"gaussian_mu_{index}"
            mu_key_no_idx: str = "gaussian_mu"
            sigma_key: str = f"gaussian_sigma_{index}"
            sigma_key_no_idx: str = "gaussian_sigma"
            sigma_log_key: str = f"gaussian_sigma_{index}_log__"
            sigma_log_key_no_idx: str = "gaussian_sigma_log__"

            if mu_key in map_estimate:
                params["mu"] = float(map_estimate[mu_key])
            elif mu_key_no_idx in map_estimate:
                params["mu"] = float(map_estimate[mu_key_no_idx])
            else:
                raise ValueError(
                    f"Missing mu for gaussian component {index}. "
                    f"Tried {mu_key!r} and {mu_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if sigma_key in map_estimate:
                params["sigma"] = float(map_estimate[sigma_key])
            elif sigma_log_key in map_estimate:
                params["sigma"] = float(np.exp(map_estimate[sigma_log_key]))
            elif sigma_key_no_idx in map_estimate:
                params["sigma"] = float(map_estimate[sigma_key_no_idx])
            elif sigma_log_key_no_idx in map_estimate:
                params["sigma"] = float(np.exp(map_estimate[sigma_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing sigma for gaussian component {index}. "
                    f"Tried {sigma_key!r} and {sigma_log_key!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "cauchy":
            loc_key: str = f"cauchy_loc_{index}"
            loc_key_no_idx: str = "cauchy_loc"
            scale_key: str = f"cauchy_scale_{index}"
            scale_key_no_idx: str = "cauchy_scale"
            scale_log_key: str = f"cauchy_scale_{index}_log__"
            scale_log_key_no_idx: str = "cauchy_scale_log__"
            alpha_key: str = f"cauchy_alpha_{index}"
            alpha_key_no_idx: str = "cauchy_alpha"
            beta_key: str = f"cauchy_beta_{index}"
            beta_key_no_idx: str = "cauchy_beta"
            beta_log_key: str = f"cauchy_beta_{index}_log__"
            beta_log_key_no_idx: str = "cauchy_beta_log__"

            if loc_key in map_estimate:
                params["loc"] = float(map_estimate[loc_key])
            elif alpha_key in map_estimate:
                params["loc"] = float(map_estimate[alpha_key])
            elif loc_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[loc_key_no_idx])
            elif alpha_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[alpha_key_no_idx])
            else:
                raise ValueError(
                    f"Missing loc/alpha for cauchy component {index}. "
                    f"Tried {loc_key!r}, {alpha_key!r}, "
                    f"{loc_key_no_idx!r}, {alpha_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if scale_key in map_estimate:
                params["scale"] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key]))
            elif beta_key in map_estimate:
                params["scale"] = float(map_estimate[beta_key])
            elif beta_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[beta_log_key]))
            elif scale_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[scale_key_no_idx])
            elif scale_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key_no_idx]))
            elif beta_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[beta_key_no_idx])
            elif beta_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[beta_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing scale/beta for cauchy component {index}. "
                    f"Tried {scale_key!r}, {scale_log_key!r}, {beta_key!r}, "
                    f"{beta_log_key!r}, {scale_key_no_idx!r}, "
                    f"{scale_log_key_no_idx!r}, {beta_key_no_idx!r}, "
                    f"{beta_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "laplace":
            loc_key: str = f"laplace_loc_{index}"
            loc_key_no_idx: str = "laplace_loc"
            scale_key: str = f"laplace_scale_{index}"
            scale_key_no_idx: str = "laplace_scale"
            scale_log_key: str = f"laplace_scale_{index}_log__"
            scale_log_key_no_idx: str = "laplace_scale_log__"

            mu_key: str = f"laplace_mu_{index}"
            mu_key_no_idx: str = "laplace_mu"
            b_key: str = f"laplace_b_{index}"
            b_key_no_idx: str = "laplace_b"
            b_log_key: str = f"laplace_b_{index}_log__"
            b_log_key_no_idx: str = "laplace_b_log__"

            if loc_key in map_estimate:
                params["loc"] = float(map_estimate[loc_key])
            elif mu_key in map_estimate:
                params["loc"] = float(map_estimate[mu_key])
            elif loc_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[loc_key_no_idx])
            elif mu_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[mu_key_no_idx])
            else:
                raise ValueError(
                    f"Missing loc/mu for laplace component {index}. "
                    f"Tried {loc_key!r}, {mu_key!r}, "
                    f"{loc_key_no_idx!r}, {mu_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if scale_key in map_estimate:
                params["scale"] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key]))
            elif b_key in map_estimate:
                params["scale"] = float(map_estimate[b_key])
            elif b_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[b_log_key]))
            elif scale_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[scale_key_no_idx])
            elif scale_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key_no_idx]))
            elif b_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[b_key_no_idx])
            elif b_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[b_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing scale/b for laplace component {index}. "
                    f"Tried {scale_key!r}, {scale_log_key!r}, {b_key!r}, "
                    f"{b_log_key!r}, {scale_key_no_idx!r}, "
                    f"{scale_log_key_no_idx!r}, {b_key_no_idx!r}, "
                    f"{b_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name in ["student_t", "studentt", "student-t"]:
            loc_key: str = f"student_t_loc_{index}"
            loc_key_no_idx: str = "student_t_loc"
            scale_key: str = f"student_t_scale_{index}"
            scale_key_no_idx: str = "student_t_scale"
            df_key: str = f"student_t_df_{index}"
            df_key_no_idx: str = "student_t_df"

            mu_key: str = f"student_t_mu_{index}"
            mu_key_no_idx: str = "student_t_mu"
            sigma_key: str = f"student_t_sigma_{index}"
            sigma_key_no_idx: str = "student_t_sigma"
            sigma_log_key: str = f"student_t_sigma_{index}_log__"
            sigma_log_key_no_idx: str = "student_t_sigma_log__"
            nu_key: str = f"student_t_nu_{index}"
            nu_key_no_idx: str = "student_t_nu"
            nu_log_key: str = f"student_t_nu_{index}_log__"
            nu_log_key_no_idx: str = "student_t_nu_log__"

            if loc_key in map_estimate:
                params["loc"] = float(map_estimate[loc_key])
            elif mu_key in map_estimate:
                params["loc"] = float(map_estimate[mu_key])
            elif loc_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[loc_key_no_idx])
            elif mu_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[mu_key_no_idx])
            else:
                raise ValueError(
                    f"Missing loc/mu for student-t component {index}. "
                    f"Tried {loc_key!r}, {mu_key!r}, "
                    f"{loc_key_no_idx!r}, {mu_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if scale_key in map_estimate:
                params["scale"] = float(map_estimate[scale_key])
            elif sigma_key in map_estimate:
                params["scale"] = float(map_estimate[sigma_key])
            elif sigma_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[sigma_log_key]))
            elif scale_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[scale_key_no_idx])
            elif sigma_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[sigma_key_no_idx])
            elif sigma_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[sigma_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing scale/sigma for student-t component {index}. "
                    f"Tried {scale_key!r}, {sigma_key!r}, {sigma_log_key!r}, "
                    f"{scale_key_no_idx!r}, {sigma_key_no_idx!r}, "
                    f"{sigma_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if df_key in map_estimate:
                params["df"] = float(map_estimate[df_key])
            elif nu_key in map_estimate:
                params["df"] = float(map_estimate[nu_key])
            elif nu_log_key in map_estimate:
                params["df"] = float(np.exp(map_estimate[nu_log_key]))
            elif df_key_no_idx in map_estimate:
                params["df"] = float(map_estimate[df_key_no_idx])
            elif nu_key_no_idx in map_estimate:
                params["df"] = float(map_estimate[nu_key_no_idx])
            elif nu_log_key_no_idx in map_estimate:
                params["df"] = float(np.exp(map_estimate[nu_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing df/nu for student-t component {index}. "
                    f"Tried {df_key!r}, {nu_key!r}, {nu_log_key!r}, "
                    f"{df_key_no_idx!r}, {nu_key_no_idx!r}, "
                    f"{nu_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "lognormal":
            mu_key: str = f"lognormal_mu_{index}"
            mu_key_no_idx: str = "lognormal_mu"
            sigma_key: str = f"lognormal_sigma_{index}"
            sigma_key_no_idx: str = "lognormal_sigma"
            sigma_log_key: str = f"lognormal_sigma_{index}_log__"
            sigma_log_key_no_idx: str = "lognormal_sigma_log__"

            if mu_key in map_estimate:
                params["mu"] = float(map_estimate[mu_key])
            elif mu_key_no_idx in map_estimate:
                params["mu"] = float(map_estimate[mu_key_no_idx])
            else:
                raise ValueError(
                    f"Missing mu for lognormal component {index}. "
                    f"Tried {mu_key!r} and {mu_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if sigma_key in map_estimate:
                params["sigma"] = float(map_estimate[sigma_key])
            elif sigma_log_key in map_estimate:
                params["sigma"] = float(np.exp(map_estimate[sigma_log_key]))
            elif sigma_key_no_idx in map_estimate:
                params["sigma"] = float(map_estimate[sigma_key_no_idx])
            elif sigma_log_key_no_idx in map_estimate:
                params["sigma"] = float(np.exp(map_estimate[sigma_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing sigma for lognormal component {index}. "
                    f"Tried {sigma_key!r} and {sigma_log_key!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "exponential":
            loc_key: str = f"exponential_loc_{index}"
            loc_key_no_idx: str = "exponential_loc"
            scale_key: str = f"exponential_scale_{index}"
            scale_key_no_idx: str = "exponential_scale"
            scale_log_key: str = f"exponential_scale_{index}_log__"
            scale_log_key_no_idx: str = "exponential_scale_log__"
            lam_key: str = f"exponential_lam_{index}"
            lam_key_no_idx: str = "exponential_lam"
            lam_log_key: str = f"exponential_lam_{index}_log__"
            lam_log_key_no_idx: str = "exponential_lam_log__"

            if loc_key in map_estimate:
                params["loc"] = float(map_estimate[loc_key])
            elif loc_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[loc_key_no_idx])
            else:
                params["loc"] = 0.0

            if scale_key in map_estimate:
                params["scale"] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key]))
            elif scale_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[scale_key_no_idx])
            elif scale_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key_no_idx]))
            elif lam_key in map_estimate:
                params["scale"] = 1.0 / float(map_estimate[lam_key])
            elif lam_log_key in map_estimate:
                params["scale"] = 1.0 / float(np.exp(map_estimate[lam_log_key]))
            elif lam_key_no_idx in map_estimate:
                params["scale"] = 1.0 / float(map_estimate[lam_key_no_idx])
            elif lam_log_key_no_idx in map_estimate:
                params["scale"] = 1.0 / float(np.exp(map_estimate[lam_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing scale for exponential component {index}. "
                    f"Tried {scale_key!r}, {scale_log_key!r}, "
                    f"{scale_key_no_idx!r}, {scale_log_key_no_idx!r}, "
                    f"{lam_key!r}, {lam_log_key!r}, "
                    f"{lam_key_no_idx!r}, {lam_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "uniform":
            low_key: str = f"uniform_low_{index}"
            low_key_no_idx: str = "uniform_low"
            high_key: str = f"uniform_high_{index}"
            high_key_no_idx: str = "uniform_high"

            if low_key in map_estimate:
                params["low"] = float(map_estimate[low_key])
            elif low_key_no_idx in map_estimate:
                params["low"] = float(map_estimate[low_key_no_idx])
            else:
                raise ValueError(
                    f"Missing low for uniform component {index}. "
                    f"Tried {low_key!r} and {low_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if high_key in map_estimate:
                params["high"] = float(map_estimate[high_key])
            elif high_key_no_idx in map_estimate:
                params["high"] = float(map_estimate[high_key_no_idx])
            else:
                raise ValueError(
                    f"Missing high for uniform component {index}. "
                    f"Tried {high_key!r} and {high_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "weibull":
            loc_key: str = f"weibull_loc_{index}"
            loc_key_no_idx: str = "weibull_loc"
            scale_key: str = f"weibull_scale_{index}"
            scale_key_no_idx: str = "weibull_scale"
            scale_log_key: str = f"weibull_scale_{index}_log__"
            scale_log_key_no_idx: str = "weibull_scale_log__"
            alpha_key: str = f"weibull_alpha_{index}"
            alpha_key_no_idx: str = "weibull_alpha"
            alpha_log_key: str = f"weibull_alpha_{index}_log__"
            alpha_log_key_no_idx: str = "weibull_alpha_log__"

            if loc_key in map_estimate:
                params["loc"] = float(map_estimate[loc_key])
            elif loc_key_no_idx in map_estimate:
                params["loc"] = float(map_estimate[loc_key_no_idx])
            else:
                params["loc"] = 0.0

            if scale_key in map_estimate:
                params["scale"] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key]))
            elif scale_key_no_idx in map_estimate:
                params["scale"] = float(map_estimate[scale_key_no_idx])
            elif scale_log_key_no_idx in map_estimate:
                params["scale"] = float(np.exp(map_estimate[scale_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing scale for weibull component {index}. "
                    f"Tried {scale_key!r}, {scale_log_key!r}, "
                    f"{scale_key_no_idx!r}, {scale_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            if alpha_key in map_estimate:
                params["alpha"] = float(map_estimate[alpha_key])
            elif alpha_log_key in map_estimate:
                params["alpha"] = float(np.exp(map_estimate[alpha_log_key]))
            elif alpha_key_no_idx in map_estimate:
                params["alpha"] = float(map_estimate[alpha_key_no_idx])
            elif alpha_log_key_no_idx in map_estimate:
                params["alpha"] = float(np.exp(map_estimate[alpha_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing alpha for weibull component {index}. "
                    f"Tried {alpha_key!r}, {alpha_log_key!r}, "
                    f"{alpha_key_no_idx!r}, {alpha_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

        elif component_name == "pareto":
            alpha_key: str = f"pareto_alpha_{index}"
            alpha_key_no_idx: str = "pareto_alpha"
            alpha_log_key: str = f"pareto_alpha_{index}_log__"
            alpha_log_key_no_idx: str = "pareto_alpha_log__"
            m_key: str = f"pareto_m_{index}"
            m_key_no_idx: str = "pareto_m"
            m_interval_key: str = f"pareto_m_{index}_interval__"
            m_interval_key_no_idx: str = "pareto_m_interval__"
            m_log_key: str = f"pareto_m_{index}_log__"
            m_log_key_no_idx: str = "pareto_m_log__"

            # --- alpha (shape / tail index) ---
            if alpha_key in map_estimate:
                params["alpha"] = float(map_estimate[alpha_key])
            elif alpha_log_key in map_estimate:
                params["alpha"] = float(np.exp(map_estimate[alpha_log_key]))
            elif alpha_key_no_idx in map_estimate:
                params["alpha"] = float(map_estimate[alpha_key_no_idx])
            elif alpha_log_key_no_idx in map_estimate:
                params["alpha"] = float(np.exp(map_estimate[alpha_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing alpha for pareto component {index}. "
                    f"Tried {alpha_key!r}, {alpha_log_key!r}, "
                    f"{alpha_key_no_idx!r}, {alpha_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            # --- m (minimum / scale) ---
            if m_key in map_estimate:
                params["m"] = float(map_estimate[m_key])
            elif m_interval_key in map_estimate:
                params["m"] = float(np.exp(map_estimate[m_interval_key]))
            elif m_log_key in map_estimate:
                params["m"] = float(np.exp(map_estimate[m_log_key]))
            elif m_key_no_idx in map_estimate:
                params["m"] = float(map_estimate[m_key_no_idx])
            elif m_interval_key_no_idx in map_estimate:
                params["m"] = float(np.exp(map_estimate[m_interval_key_no_idx]))
            elif m_log_key_no_idx in map_estimate:
                params["m"] = float(np.exp(map_estimate[m_log_key_no_idx]))
            else:
                raise ValueError(
                    f"Missing m for pareto component {index}. "
                    f"Tried {m_key!r}, {m_interval_key!r}, {m_log_key!r}, "
                    f"{m_key_no_idx!r}, {m_interval_key_no_idx!r}, "
                    f"{m_log_key_no_idx!r}. "
                    f"Available keys: {list(map_estimate.keys())}"
                )

            # Hard safety clamp: m must be strictly positive.
            params["m"] = max(params["m"], 1e-9)

        if len(params) == 0:
            raise ValueError(
                f"Unknown distribution family {component_name!r} for component "
                f"index {index}. Supported families: gaussian/normal, cauchy, "
                f"laplace, student_t/studentt, lognormal, exponential, uniform, "
                f"weibull, pareto."
            )

        return params

    @classmethod
    @validate
    def build_pdf_from_map(
        cls, *, map_estimate: Dict[str, Any], components: List[str]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Build PDF function from MAP estimates.

        Args:
            map_estimate: Dictionary of MAP estimates.
            components: Canonical list of component family names, matching
                ``FitState.family_name`` (e.g. ``["gaussian"]``,
                ``["student_t"]``, or ``["gaussian", "cauchy"]``). Each
                element maps 1:1 to a PyMC variable-name prefix used in
                ``map_estimate``.

        Returns:
            PDF function that takes x (scalar or array) and returns probability density.
        """
        if len(components) == 0:
            raise ValueError("build_pdf_from_map requires at least one component; received empty list.")

        weights = cls._extract_mixture_weights(
            map_estimate=map_estimate, n_components=len(components)
        )

        weights = weights / np.sum(weights)

        component_pdfs: List[Callable[[np.ndarray], np.ndarray]] = []
        for idx, component_name in enumerate(components):
            component_name = component_name.lower().replace("-", "_")

            params: Dict[str, float] = cls.extract_component_params(
                map_estimate=map_estimate, component_name=component_name, index=idx
            )

            if component_name in ["gaussian", "normal"]:
                pdf: Callable[[np.ndarray], np.ndarray] = cls.build_gaussian_pdf(
                    params["mu"], params["sigma"]
                )
            elif component_name == "cauchy":
                pdf = cls.build_cauchy_pdf(params["loc"], params["scale"])
            elif component_name == "laplace":
                pdf = cls.build_laplace_pdf(params["loc"], params["scale"])
            elif component_name in ["student_t", "studentt"]:
                pdf = cls.build_student_t_pdf(params["loc"], params["scale"], params["df"])
            elif component_name == "lognormal":
                pdf = cls.build_lognormal_pdf(params["mu"], params["sigma"])
            elif component_name == "exponential":
                pdf = cls.build_exponential_pdf(params["loc"], params["scale"])
            elif component_name == "uniform":
                pdf = cls.build_uniform_pdf(params["low"], params["high"])
            elif component_name == "weibull":
                pdf = cls.build_weibull_pdf(params["loc"], params["scale"], params["alpha"])
            elif component_name == "pareto":
                pdf = cls.build_pareto_pdf(params["alpha"], params["m"])
            else:
                raise ValueError(f"Unknown distribution: {component_name}")

            component_pdfs.append(pdf)

        def mixture_pdf(x: np.ndarray) -> np.ndarray:
            x_arr: np.ndarray = np.atleast_1d(np.asarray(x, dtype=float))
            result: np.ndarray = np.zeros_like(x_arr, dtype=float)
            for weight, pdf in zip(weights, component_pdfs):
                result += weight * pdf(x_arr)
            return result

        return mixture_pdf

    @classmethod
    def _extract_mixture_weights(
        cls, *, map_estimate: Dict[str, Any], n_components: int
    ) -> np.ndarray:
        """Resolve mixture weights from a MAP estimate, tolerant to VLM naming.

        The canonical PyMC name for mixture weights is ``w`` (constrained on the
        simplex) with companion ``w_simplex__`` (unconstrained, stick-breaking).
        In practice, VLM-generated code sometimes uses a different name such as
        ``weights``, ``mix_w``, or ``w_gaussian_cauchy``. We therefore also scan
        for any key matching ``w*`` / ``weights*`` that is not a transform
        suffix (``_log__``, ``_simplex__``, etc.), falling back to inverting
        the simplex transform when only the unconstrained form is present.

        Single-component models ignore weights entirely and default to ``[1.0]``.
        If no recognisable multi-component weight variable is available, this
        preserves the legacy behavior of falling back to uniform weights.
        """
        if n_components == 1:
            return np.ones(1)

        for key in ("w", "weights"):
            if key in map_estimate:
                return np.asarray(map_estimate[key], dtype=float)

        fuzzy_w_key: Optional[str] = next(
            (
                k
                for k in map_estimate
                if (k == "w" or k.startswith("w_") or k == "weights" or k.startswith("weights_"))
                and not k.endswith("_log__")
                and not k.endswith("_simplex__")
                and not k.endswith("_interval__")
                and not k.endswith("_ordered__")
            ),
            None,
        )
        if fuzzy_w_key is not None:
            candidate: np.ndarray = np.asarray(map_estimate[fuzzy_w_key], dtype=float)
            if candidate.shape == (n_components,):
                return candidate

        for simplex_key in list(map_estimate.keys()):
            if simplex_key.endswith("_simplex__") and (
                simplex_key.startswith("w") or "simplex" in simplex_key
            ):
                candidate_simplex: np.ndarray = np.asarray(
                    map_estimate[simplex_key], dtype=float
                )
                if candidate_simplex.shape == (n_components - 1,):
                    return cls.simplex_to_weights(candidate_simplex, n_components)

        return np.ones(n_components) / n_components

    @staticmethod
    def simplex_to_weights(w_simplex: np.ndarray, n_components: int) -> np.ndarray:
        """Invert PyMC's StickBreaking transform: unconstrained logits -> simplex.

        PyMC's ``_simplex__`` transform is *not* ``[w_0, ..., w_{K-2}]`` with an
        implicit last component; it is the output of stick-breaking over the
        unconstrained reals. For a K-component Dirichlet/simplex, the transform
        maps ``R^(K-1)`` -> the K-simplex via::

            z_i = sigmoid(y_i - log(K - i - 1))     for i = 0 .. K-2
            w_i = z_i * prod_{j<i} (1 - z_j)
            w_{K-1} = prod_{j<K-1} (1 - z_j)

        Falling back to ``w[-1] = 1 - sum(w[:-1])`` (the old implementation) was
        mathematically wrong and silently produced nonsensical mixture weights.

        Only invoked when ``'w'`` (the constrained simplex) is absent from the
        MAP estimate, which is itself rare — PyMC ordinarily surfaces both.
        """
        if len(w_simplex) != n_components - 1:
            raise ValueError(
                f"w_simplex__ length {len(w_simplex)} does not match expected "
                f"stick-breaking length {n_components - 1} for a "
                f"{n_components}-component simplex."
            )

        K: int = n_components
        y: np.ndarray = np.asarray(w_simplex, dtype=float)
        weights: np.ndarray = np.zeros(K)
        remaining: float = 1.0
        for i in range(K - 1):
            offset: float = float(np.log(K - i - 1))
            z_i: float = float(1.0 / (1.0 + np.exp(-(y[i] - offset))))
            weights[i] = z_i * remaining
            remaining = remaining * (1.0 - z_i)
        weights[K - 1] = remaining
        return weights


# ---------------------------------------------------------------------------
#  Data Factorization
# ---------------------------------------------------------------------------


@validate
def factorize_gmm(
    *,
    data: np.ndarray,
    n_components: int,
    fit_path: str,
    random_state: int = 42,
    plot: bool = True,
    covariance_type: str = "full",
) -> Tuple[np.ndarray, GaussianMixture]:
    """
    Factorize a 1-D dataset into ``n_components`` groups using a Gaussian
    Mixture Model (GMM), returning a hard label for every data point.

    This is useful for:
    - Identifying sub-populations in mixture distributions.
    - Separating signal from noise in peak-fitting problems.

    Parameters
    ----------
    data           : 1-D array
    n_components   : number of mixture components to fit
    random_state   : random seed for reproducibility
    plot           : if True, show histogram coloured by component
    covariance_type: GMM covariance structure ('full', 'tied', 'diag', 'spherical')

    Returns
    -------
    labels : integer array of shape (n,) with component assignments 0 … K-1
    gmm    : fitted sklearn GaussianMixture object
    """
    X: np.ndarray = data.reshape(-1, 1)
    gmm: GaussianMixture = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    gmm.fit(X)
    labels: np.ndarray = gmm.predict(X)

    separator: str = "=" * 55
    component_lines: List[str] = []
    for k in range(n_components):
        mask: np.ndarray = labels == k
        w: float = gmm.weights_[k]
        mu: float = gmm.means_[k, 0]
        sigma: float = np.sqrt(gmm.covariances_[k].ravel()[0])
        component_lines.append(
            f"  Component {k}: weight={w:.3f}  mu={mu:.4g}  sigma={sigma:.4g}  n={mask.sum()}"
        )
    logger.info(
        "\n".join(
            [
                separator,
                f"  GMM Factorization ({n_components} components)",
                separator,
                *component_lines,
                f"  BIC = {gmm.bic(X):.2f}   AIC = {gmm.aic(X):.2f}",
                separator,
            ]
        )
    )

    if plot:
        with _MATPLOTLIB_LOCK:
            colors: Tuple[Any, ...] = plt.cm.tab10.colors
            fig, ax = plt.subplots(figsize=(9, 4))
            try:
                bins: int = min(80, max(20, int(np.sqrt(len(data)) * 2)))
                for k in range(n_components):
                    ax.hist(
                        data[labels == k],
                        bins=bins,
                        alpha=0.55,
                        color=colors[k % 10],
                        label=f"Component {k}  (w={gmm.weights_[k]:.2f})",
                        density=True,
                    )
                x_plot: np.ndarray = np.linspace(data.min(), data.max(), 500)
                pdf: np.ndarray = np.exp(gmm.score_samples(x_plot.reshape(-1, 1)))
                ax.plot(x_plot, pdf, "k--", lw=1.5, label="GMM total PDF")
                ax.set_xlabel("x")
                ax.set_ylabel("Density")
                ax.set_title("GMM Factorization")
                ax.legend()
                fig.tight_layout()
                fig.savefig(fit_path)
            finally:
                plt.close(fig)

    return labels, gmm


@validate
def compute_moments(
    data: np.ndarray,
    verbose: bool = True,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute mean, variance, skewness, and excess kurtosis, and print
    a plain-language interpretation that aids distribution selection.
    If ``labels`` is provided (e.g. from factorize_gmm), moments are also
    computed per mixture component and appended to the summary string.

    Parameters
    ----------
    data    : 1-D array
    verbose : print the summary table
    labels  : optional integer array of component assignments from factorize_gmm.
              If supplied, per-component moments are computed and included in
              the returned summary_str.
    Returns
    -------
    dict with keys: mean, variance, skewness, kurtosis (excess), summary_str,
                    and optionally component_moments (list of per-component dicts)
    """
    mean: float = float(np.mean(data))
    variance: float = float(np.var(data, ddof=1))
    skewness: float = float(stats.skew(data))
    kurt: float = float(stats.kurtosis(data))

    def _interpret(skewness: float, kurt: float) -> Tuple[str, str, str, str]:
        if abs(skewness) < 0.5:
            sym_label: str = "approximately symmetric"
            sym_hint: str = "Normal, Student-t, Cauchy, Logistic, Uniform"
        elif skewness > 0.5:
            sym_label = f"right-skewed (skew={skewness:.2f})"
            sym_hint = "Gamma, Lognormal, Weibull, Exponential, Chi-squared"
        else:
            sym_label = f"left-skewed (skew={skewness:.2f})"
            sym_hint = "reflected Gamma/Weibull, Beta (α>β)"

        if kurt < -0.5:
            kurt_label: str = "platykurtic (light tails)"
            kurt_hint: str = "Uniform, Beta"
        elif kurt < 1.0:
            kurt_label = "mesokurtic (normal tails)"
            kurt_hint = "Normal, Logistic"
        elif kurt < 5.0:
            kurt_label = "leptokurtic (heavier tails)"
            kurt_hint = "Student-t (moderate ν), Laplace"
        else:
            kurt_label = "very leptokurtic (fat tails)"
            kurt_hint = "Student-t (small ν), Cauchy, Lognormal"

        return sym_label, sym_hint, kurt_label, kurt_hint

    sym_label: str
    sym_hint: str
    kurt_label: str
    kurt_hint: str
    sym_label, sym_hint, kurt_label, kurt_hint = _interpret(skewness, kurt)

    summary_str: str = (
        f"The data has mean={mean:.4g}, variance={variance:.4g}, "
        f"skewness={skewness:.4g} ({sym_label}, consistent with {sym_hint}), "
        f"and excess kurtosis={kurt:.4g} ({kurt_label}, consistent with {kurt_hint}). "
        f"Use these moments to anchor your prior choices: the location prior should be "
        f"centred near {mean:.4g}, the scale prior should reflect a spread of roughly "
        f"{variance**0.5:.4g} (std dev), and the chosen distribution family should be "
        f"consistent with both the skewness and kurtosis hints above. If the red fitted "
        f"line does not reflect these properties (e.g. it is too symmetric for the observed "
        f"skew, or its tails are too light for the observed kurtosis), adjust the family or "
        f"parameters accordingly."
    )

    moments: Dict[str, Any] = dict(
        mean=mean,
        variance=variance,
        skewness=skewness,
        kurtosis=kurt,
        summary_str=summary_str,
    )

    per_component_log_lines: List[str] = []

    if labels is not None:
        unique_components: np.ndarray = np.unique(labels)
        component_moments: List[Dict[str, Any]] = []
        component_strs: List[str] = []

        for k in unique_components:
            mask: np.ndarray = labels == k
            c_data: np.ndarray = data[mask]
            weight: float = mask.sum() / len(data)

            c_mean: float = float(np.mean(c_data))
            c_var: float = float(np.var(c_data, ddof=1))
            c_skew: float = float(stats.skew(c_data))
            c_kurt: float = float(stats.kurtosis(c_data))

            c_sym_label: str
            c_sym_hint: str
            c_kurt_label: str
            c_kurt_hint: str
            c_sym_label, c_sym_hint, c_kurt_label, c_kurt_hint = _interpret(c_skew, c_kurt)

            component_moments.append(
                dict(
                    component=int(k),
                    weight=weight,
                    mean=c_mean,
                    variance=c_var,
                    skewness=c_skew,
                    kurtosis=c_kurt,
                )
            )

            component_strs.append(
                f"  Component {k} (empirical weight={weight:.3f}): "
                f"mean={c_mean:.4g}, std={c_var**0.5:.4g}, "
                f"skewness={c_skew:.4g} ({c_sym_label} → {c_sym_hint}), "
                f"excess kurtosis={c_kurt:.4g} ({c_kurt_label} → {c_kurt_hint}). "
                f"Set this component's location prior near {c_mean:.4g} and scale prior "
                f"near {c_var**0.5:.4g}; choose a family consistent with {c_sym_hint} "
                f"and tail weight consistent with {c_kurt_hint}."
            )

            per_component_log_lines.extend(
                [
                    (
                        f"  Component {k} (weight={weight:.3f}): "
                        f"mean={c_mean:.4g}  std={c_var**0.5:.4g}  "
                        f"skew={c_skew:.4g}  kurt={c_kurt:.4g}"
                    ),
                    f"    Symmetry : {c_sym_label} -> {c_sym_hint}",
                    f"    Tails    : {c_kurt_label} -> {c_kurt_hint}",
                ]
            )

        mixture_summary: str = (
            f"This appears to be a mixture model with {len(unique_components)} components. "
            f"Each component should have its location, scale, and family set independently "
            f"based on the per-component moments below — do not apply global estimates to "
            f"individual components:\n" + "\n".join(component_strs)
        )

        moments["summary_str"] = mixture_summary
        moments["component_moments"] = component_moments

    if verbose:
        major_separator: str = "=" * 55
        minor_separator: str = "-" * 55
        moment_log_lines: List[str] = []
        if len(per_component_log_lines) > 0:
            moment_log_lines.extend([major_separator, "  Per-Component Moments", major_separator])
            moment_log_lines.extend(per_component_log_lines)
        moment_log_lines.extend(
            [
                major_separator,
                "  Moment Summary",
                major_separator,
                f"  Mean              : {mean:.4g}",
                f"  Variance          : {variance:.4g}",
                f"  Skewness          : {skewness:.4g}",
                f"  Excess Kurtosis   : {kurt:.4g}",
                minor_separator,
                f"  Symmetry hint     : {sym_label} -> {sym_hint}",
                f"  Tail-weight hint  : {kurt_label} -> {kurt_hint}",
                major_separator,
            ]
        )
        logger.info("\n".join(moment_log_lines))

    return moments


# ---------------------------------------------------------------------------
# Helpers: numerical CDF + quantile function from an arbitrary PDF callable
# ---------------------------------------------------------------------------


def _build_numerical_cdf(
    pdf: Callable,
    x_min: float,
    x_max: float,
    n_points: int = 2_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically integrate ``pdf`` over [x_min, x_max] to obtain a CDF.

    Returns
    -------
    xs   : evaluation grid
    cdf  : cumulative probabilities at each grid point (starts at 0, ends ≈ 1)
    """
    xs: np.ndarray = np.linspace(x_min, x_max, n_points)
    pdf_vals: np.ndarray = np.asarray(pdf(xs), dtype=float)
    cdf_raw: np.ndarray = cumulative_trapezoid(pdf_vals, xs, initial=0.0)
    cdf: np.ndarray = cdf_raw / cdf_raw[-1]
    return xs, cdf


def _build_quantile_function(
    pdf: Callable,
    x_min: float,
    x_max: float,
    n_points: int = 2_000,
) -> Callable:
    """
    Return a callable Q(p) that maps probabilities → quantiles for ``pdf``.
    """
    xs: np.ndarray
    cdf: np.ndarray
    xs, cdf = _build_numerical_cdf(pdf, x_min, x_max, n_points)
    _: np.ndarray
    uniq: np.ndarray
    _, uniq = np.unique(cdf, return_index=True)
    return interp1d(cdf[uniq], xs[uniq], bounds_error=False, fill_value=(xs[0], xs[-1]))


def _plot_range(data: np.ndarray, pad: float = 0.10) -> Tuple[float, float]:
    """Return a padded (x_min, x_max) spanning the data."""
    lo: float = data.min()
    hi: float = data.max()
    rng: float = hi - lo if hi > lo else 1.0
    return lo - pad * rng, hi + pad * rng


# ---------------------------------------------------------------------------
# 2. Empirical CDF vs Mixture CDF
# ---------------------------------------------------------------------------


@validate
def plot_probability(
    *,
    data: np.ndarray,
    mixture_pdf: Callable,
    fit_path: str,
    ax: Optional[Axes] = None,
    n_cdf_points: int = 2_000,
    title: str = "Probability Plot (Empirical vs Mixture CDF)",
) -> Axes:
    """
    Plot the empirical CDF of ``data`` against the numerically integrated
    CDF of ``mixture_pdf``. Linearity in the scatter indicates a good fit.

    Parameters
    ----------
    data         : 1-D observed array
    mixture_pdf  : callable from ``DistributionBuilder.build_pdf_from_map``
    ax           : optional Axes
    n_cdf_points : resolution for numerical CDF integration
    title        : plot title

    Returns
    -------
    ax
    """
    with _MATPLOTLIB_LOCK:
        created_fig: bool = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure

        try:
            x_min: float
            x_max: float
            x_min, x_max = _plot_range(data)
            xs: np.ndarray
            cdf_grid: np.ndarray
            xs, cdf_grid = _build_numerical_cdf(mixture_pdf, x_min, x_max, n_cdf_points)
            mixture_cdf: interp1d = interp1d(xs, cdf_grid, bounds_error=False, fill_value=(0.0, 1.0))

            x_sorted: np.ndarray = np.sort(data)
            n: int = len(x_sorted)
            p_empirical: np.ndarray = (np.arange(1, n + 1) - 0.5) / n
            p_theoretical: np.ndarray = mixture_cdf(x_sorted).clip(0, 1)

            # Scatter: theoretical prob on x, empirical prob on y
            ax.scatter(
                p_theoretical,
                p_empirical,
                s=8,
                alpha=0.55,
                color="#7C3AED",
                label="Data points",
                zorder=3,
            )
            ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect fit", zorder=4)

            ks_stat: float = float(np.max(np.abs(p_empirical - p_theoretical)))
            ax.text(
                0.05,
                0.92,
                f"KS stat = {ks_stat:.4f}",
                transform=ax.transAxes,
                fontsize=8,
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", alpha=0.8),
            )

            ax.set_xlabel("Theoretical CDF  P(X ≤ x)")
            ax.set_ylabel("Empirical CDF  F̂(x)")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.35)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.savefig(fit_path)
            return ax
        finally:
            if created_fig:
                plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Q-Q Plot
# ---------------------------------------------------------------------------


@validate
def plot_qq(
    *,
    data: np.ndarray,
    mixture_pdf: Callable,
    fit_path: str,
    ax: Optional[Axes] = None,
    n_quantiles: Optional[int] = None,
    n_cdf_points: int = 2_000,
    title: str = "Q-Q Plot (Empirical vs Mixture)",
) -> Axes:
    """
    Quantile-Quantile plot: empirical quantiles of ``data`` vs theoretical
    quantiles from the numerical inverse-CDF of ``mixture_pdf``.

    Parameters
    ----------
    data         : 1-D observed array
    mixture_pdf  : callable from ``DistributionBuilder.build_pdf_from_map``
    ax           : optional Axes
    n_quantiles  : number of quantile points to plot (defaults to len(data))
    n_cdf_points : grid resolution for numerical CDF integration
    title        : plot title

    Returns
    -------
    ax
    """
    with _MATPLOTLIB_LOCK:
        created_fig: bool = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        else:
            fig = ax.figure

        try:
            x_min: float
            x_max: float
            x_min, x_max = _plot_range(data)
            Q: Callable = _build_quantile_function(mixture_pdf, x_min, x_max, n_cdf_points)

            n: int = n_quantiles if n_quantiles is not None else len(data)
            probs: np.ndarray = (np.arange(1, n + 1) - 0.5) / n

            theoretical_q: np.ndarray = Q(probs)
            empirical_q: np.ndarray = np.quantile(data, probs)

            ax.scatter(
                theoretical_q,
                empirical_q,
                s=8,
                alpha=0.55,
                color="#2563EB",
                label="Data quantiles",
                zorder=3,
            )

            q25_t: float = Q(0.25)
            q75_t: float = Q(0.75)
            q25_e: float = float(np.quantile(data, 0.25))
            q75_e: float = float(np.quantile(data, 0.75))
            slope: float = (q75_e - q25_e) / (q75_t - q25_t) if q75_t != q25_t else 1.0
            intercept: float = q25_e - slope * q25_t
            x_ref: np.ndarray = np.array([theoretical_q.min(), theoretical_q.max()])
            ax.plot(
                x_ref,
                slope * x_ref + intercept,
                "r--",
                lw=1.5,
                label="Reference line",
                zorder=4,
            )

            ax.set_xlabel("Theoretical quantiles  (Mixture)")
            ax.set_ylabel("Sample quantiles")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, ls="--", alpha=0.35)
            fig.savefig(fit_path)
            return ax
        finally:
            if created_fig:
                plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Tail transform plots (unchanged API, no dist needed)
# ---------------------------------------------------------------------------


@validate
def plot_tail_transforms(
    *,
    data: np.ndarray,
    mixture_pdf: Optional[Callable],
    fit_path: str,
    ax_loglog: Optional[Axes] = None,
    ax_semilog: Optional[Axes] = None,
    n_cdf_points: int = 2_000,
    title_prefix: str = "",
) -> Tuple[Axes, Axes]:
    """
    Log-log and semi-log CCDF plots of ``data``, with an optional mixture
    survival-function overlay.

    Parameters
    ----------
    data        : 1-D observed array
    mixture_pdf : optional callable — if supplied, its survival function is
                  overlaid on both plots for direct comparison
    n_cdf_points: grid resolution for numerical integration of mixture_pdf

    Returns
    -------
    (ax_loglog, ax_semilog)
    """
    with _MATPLOTLIB_LOCK:
        created_fig: bool = ax_loglog is None or ax_semilog is None
        if ax_loglog is None or ax_semilog is None:
            fig, (ax_loglog, ax_semilog) = plt.subplots(1, 2, figsize=(12, 4))
        else:
            fig = ax_loglog.figure

        try:
            x_sorted: np.ndarray = np.sort(data)
            n: int = len(x_sorted)
            ecdf: np.ndarray = np.arange(1, n + 1) / n
            sf: np.ndarray = 1.0 - ecdf

            pos_mask: np.ndarray = (x_sorted > 0) & (sf > 0)
            for ax in (ax_loglog, ax_semilog):
                ax.plot(
                    x_sorted[pos_mask],
                    sf[pos_mask],
                    "o",
                    markersize=2,
                    alpha=0.55,
                    color="#2563EB",
                    label="Empirical CCDF",
                )

            if mixture_pdf is not None:
                x_min: float
                x_max: float
                x_min, x_max = _plot_range(data)
                xs: np.ndarray
                cdf_grid: np.ndarray
                xs, cdf_grid = _build_numerical_cdf(mixture_pdf, x_min, x_max, n_cdf_points)
                sf_mix: np.ndarray = 1.0 - cdf_grid
                mix_pos: np.ndarray = (xs > 0) & (sf_mix > 0)
                for ax in (ax_loglog, ax_semilog):
                    ax.plot(
                        xs[mix_pos],
                        sf_mix[mix_pos],
                        "-",
                        lw=1.8,
                        color="#DC2626",
                        label="Mixture CCDF",
                        zorder=3,
                    )

            # Formatting
            ax_loglog.set_xscale("log")
            ax_loglog.set_yscale("log")
            ax_loglog.set_xlabel("x  (log)")
            ax_loglog.set_ylabel("P(X > x)  (log)")
            ax_loglog.set_title(f"{title_prefix}Log–Log CCDF")

            ax_semilog.set_yscale("log")
            ax_semilog.set_xlabel("x")
            ax_semilog.set_ylabel("P(X > x)  (log)")
            ax_semilog.set_title(f"{title_prefix}Semi-Log CCDF")

            for ax in (ax_loglog, ax_semilog):
                ax.legend(fontsize=8)
                ax.grid(True, which="both", ls="--", alpha=0.35)

            fig.savefig(fit_path)

            return ax_loglog, ax_semilog
        finally:
            if created_fig:
                plt.close(fig)


def _plot_raw_histogram(data: Any, *, fit_path: str) -> None:
    """Plot a plain histogram of the raw data (used at step 0 when no model is fitted).

    Uses ``_MATPLOTLIB_LOCK`` for thread-safety in parallel runs; without the
    lock, concurrent callers race on matplotlib globals and produce corrupt PNGs.
    """
    with _MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            ax.hist(np.asarray(data), bins=50, density=True, alpha=0.7, edgecolor="black", linewidth=0.5)
            ax.set_title("Raw data histogram (no model fitted yet)")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            fig.tight_layout()
            fig.savefig(fit_path, dpi=150)
        finally:
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Tool Registry + concrete tool subclasses
# ══════════════════════════════════════════════════════════════════════════════


class DistributionFittingTool(Tool, Registry, ABC):
    """Registry of all static distribution-fitting diagnostic tools.

    Concrete subclasses auto-register under their snake_case class name.
    Use ``DistributionFittingTool.of("qq_plot")`` to resolve by name,
    or ``DistributionFittingTool.subclasses()`` to list all registered tools.
    """

    pass


# ── Concrete tools ────────────────────────────────────────────────────────────


class CalculateMoments(DistributionFittingTool):
    """Compute mean, variance, skewness, and excess kurtosis of the data."""

    tool_description: ClassVar[str] = (
        "Compute mean, variance, skewness, and excess kurtosis of the data. "
        "Returns a plain-language interpretation that aids distribution selection, "
        "including symmetry hints (e.g. right-skewed suggests Gamma/Lognormal/Weibull) "
        "and tail-weight hints (e.g. leptokurtic suggests Student-t/Cauchy/Laplace). "
        "Use this when you need quantitative guidance on location, scale, and shape "
        "to anchor your prior choices."
    )
    output_type: ClassVar[str] = "numeric"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[DistFittingFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        """Compute moments and return them as a JSON artifact.

        ``calculate_moments`` is a numeric-only tool: it does NOT produce a new
        visualization. The current fit overlay is already attached as the
        Phase-2 base context image (rendered once by the pipeline, described
        in DIAGNOSTIC RESULTS under ``1) Context image``). Re-rendering the
        fit here would duplicate that file on disk and spuriously list it as
        an output of this tool.
        """
        moments: Dict[str, Any] = compute_moments(data=np.asarray(data))
        moments_json: str = json.dumps(moments, indent=2, default=str)
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="json",
                    description="Moment statistics and distribution-shape interpretation",
                    inline_content=moments_json,
                    attachment_path=None,
                    truncated=False,
                ),
            ],
        )


class SegmentDistributionsAndCalculateMoments(DistributionFittingTool):
    """Segment data into mixture components using GMM, then compute per-component moments."""

    tool_description: ClassVar[str] = (
        "Segment the data into mixture components using a Gaussian Mixture Model, "
        "then compute per-component moments (mean, variance, skewness, kurtosis). "
        "Use this when the histogram appears multimodal and you need to identify "
        "each sub-population's center, spread, and shape independently. "
        "Returns per-component moment summaries with distribution family hints."
    )
    output_type: ClassVar[str] = "numeric"
    parameters_schema: ClassVar[Dict[str, Any]] = {
        "n_components": {
            "type": "integer",
            "description": (
                "Number of mixture components to segment into. "
                "Should match the number of visible modes in the histogram."
            ),
        },
    }

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[DistFittingFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        n_comp: int = 1
        if selected_tool_args is not None and "n_components" in selected_tool_args:
            n_comp = selected_tool_args["n_components"]
        elif fit_state is not None:
            n_comp = len(fit_state.family_name)
        labels: np.ndarray
        _gmm: GaussianMixture
        labels, _gmm = factorize_gmm(data=np.asarray(data), n_components=n_comp, fit_path=fit_path)
        moments: Dict[str, Any] = compute_moments(data=np.asarray(data), labels=labels)
        moments_json: str = json.dumps(moments, indent=2, default=str)
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="GMM component segmentation with total mixture overlay",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                ),
                DiagnosticArtifact(
                    artifact_type="json",
                    description="Per-component moment statistics and interpretation",
                    inline_content=moments_json,
                    attachment_path=None,
                    truncated=False,
                ),
            ],
        )


class QQPlot(DistributionFittingTool):
    """Quantile-Quantile plot comparing empirical vs theoretical quantiles."""

    tool_description: ClassVar[str] = (
        "Generate a Quantile-Quantile plot comparing empirical data quantiles "
        "to theoretical quantiles from the currently fitted distribution. "
        "Linearity indicates a good fit; S-shaped curvature indicates tail mismatch; "
        "one-sided curvature suggests skew; sharp tail departures may indicate "
        "outliers or heavier tails than modeled."
    )
    output_type: ClassVar[str] = "visualization"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[DistFittingFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        if fit_state is None:
            _plot_raw_histogram(data, fit_path=fit_path)
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Raw histogram (no fitted model yet)",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )
        builder: DistributionBuilder = DistributionBuilder()
        pdf_pred = builder.build_pdf_from_map(
            map_estimate=fit_state.map_estimate, components=fit_state.family_name
        )
        plot_qq(data=data, mixture_pdf=pdf_pred, fit_path=fit_path)
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="QQ plot comparing empirical vs fitted quantiles",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                )
            ],
        )


class PlotTailsTransform(DistributionFittingTool):
    """Log-log and semi-log CCDF plots to diagnose tail behavior."""

    tool_description: ClassVar[str] = (
        "Generate log-log and semi-log CCDF (complementary CDF) plots to diagnose "
        "tail behavior. A straight line on the log-log plot indicates power-law / "
        "Pareto-type heavy tails; a straight line on the semi-log plot indicates "
        "exponential decay. Use this to distinguish between heavy-tailed and "
        "light-tailed distributions when the histogram alone is ambiguous."
    )
    output_type: ClassVar[str] = "visualization"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[DistFittingFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        if fit_state is None:
            _plot_raw_histogram(data, fit_path=fit_path)
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Raw histogram (no fitted model yet)",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )
        builder: DistributionBuilder = DistributionBuilder()
        pdf_pred = builder.build_pdf_from_map(
            map_estimate=fit_state.map_estimate, components=fit_state.family_name
        )
        plot_tail_transforms(data=data, mixture_pdf=pdf_pred, fit_path=fit_path)
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="Log-log and semi-log CCDF tail-transform diagnostics",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                )
            ],
        )


class ProbabilityPlot(DistributionFittingTool):
    """Probability plot comparing empirical CDF to fitted CDF."""

    tool_description: ClassVar[str] = (
        "Generate a probability plot comparing the empirical CDF to the fitted "
        "distribution's theoretical CDF. A consistent horizontal shift indicates "
        "mis-specified location; a slope mismatch indicates scale misfit; systematic "
        "tail deviations suggest distributional misfit. Reports a KS statistic "
        "for quantitative goodness-of-fit assessment."
    )
    output_type: ClassVar[str] = "visualization"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[DistFittingFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        if fit_state is None:
            _plot_raw_histogram(data, fit_path=fit_path)
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Raw histogram (no fitted model yet)",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )
        builder: DistributionBuilder = DistributionBuilder()
        pdf_pred = builder.build_pdf_from_map(
            map_estimate=fit_state.map_estimate, components=fit_state.family_name
        )
        plot_probability(data=data, mixture_pdf=pdf_pred, fit_path=fit_path)
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="Probability plot comparing empirical and fitted CDFs",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                )
            ],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  DomainToolkit dispatch class
# ══════════════════════════════════════════════════════════════════════════════


class DistributionFittingToolkit(DomainToolkit):
    """Toolkit dispatch for distribution fitting.

    Delegates tool execution to ``DistributionFittingTool.of(selected_tool)``.
    The if/elif chain is replaced by Registry lookup.  Only three special cases
    remain: ``None`` (default plot), ``generate_new_tool`` (dynamic generation),
    and unknown tools (fallback to dynamic registry, then default plot).
    """

    aliases: ClassVar[List[str]] = DOMAIN_ALIASES

    def get_static_tools(self) -> List[Dict[str, Any]]:
        return [tool_cls.to_openai_schema() for tool_cls in DistributionFittingTool.subclasses()]

    def supports_dynamic_generation(self) -> bool:
        return True

    def execute_tool(
        self,
        *,
        selected_tool: Optional[str],
        selected_tool_args: Dict[str, Any],
        data: Any,
        fit_state: Optional[DistFittingFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        plot_type_descriptions: Dict[str, str],
    ) -> DiagnosticToolResult:
        """Run a distribution-fitting toolkit function.

        ``generate_new_tool`` and previously-registered dynamic tools are
        both handled by ``_run_diagnostic_rounds()`` in ``experiments.py``
        BEFORE this method is called (the pipeline checks
        ``deps.dynamic_tools`` first and dispatches there directly).
        This method therefore only handles the static-tool path for
        this domain.

        Returns a structured diagnostic tool result.
        """
        if selected_tool is None or selected_tool == "None":
            if fit_state is None:
                _plot_raw_histogram(data, fit_path=fit_path)
                return DiagnosticToolResult(
                    tool_name="histogram",
                    tool_description=plot_type_descriptions["histogram"],
                    artifacts=[
                        DiagnosticArtifact(
                            artifact_type="image",
                            description="Raw histogram (no fitted model yet)",
                            inline_content=None,
                            attachment_path=fit_path,
                            truncated=False,
                        )
                    ],
                )
            plot_best_fit(
                data=data,
                map_estimate=fit_state.map_estimate,
                model_info=fit_state.ans,
                best_idx=best_idx,
                path=fit_path,
            )
            return DiagnosticToolResult(
                tool_name="histogram",
                tool_description=plot_type_descriptions["histogram"],
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Histogram with fitted distribution overlay",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )

        tool: Optional[DistributionFittingTool]
        try:
            tool = DistributionFittingTool.of(selected_tool)
        except KeyError:
            logger.debug(
                f"Tool {selected_tool!r} not in DistributionFittingTool registry. "
                f"Available: {[cls.tool_name for cls in DistributionFittingTool.subclasses()]}"
            )
            tool = None

        if tool is not None:
            tool_result: DiagnosticToolResult = tool.execute(
                data=data,
                fit_state=fit_state,
                best_idx=best_idx,
                fit_path=fit_path,
                selected_tool_args=selected_tool_args,
            )
            static_description: str = plot_type_descriptions.get(
                tool_result.tool_name,
                tool_result.tool_description,
            )
            return DiagnosticToolResult(
                tool_name=tool_result.tool_name,
                tool_description=static_description,
                artifacts=tool_result.artifacts,
            )

        raise ValueError(
            f"Unknown static tool {selected_tool!r} for distribution-fitting. "
            f"Available: {[cls.tool_name for cls in DistributionFittingTool.subclasses()]}. "
            f"If this was meant to be a dynamic tool, the pipeline should have "
            f"dispatched it via deps.dynamic_tools before reaching execute_tool()."
        )
