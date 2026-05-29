import ast
import re
import pickle
import traceback
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor.tensor as pt

from scipy import stats, integrate
from scipy.stats import wasserstein_distance, entropy
from scipy.integrate import simpson
from scipy.special import kl_div

import pandas as pd
import glob
import os


from processing_utils import *

def compare_fit_metrics(true_fit_metrics, pred_fit_metrics, N, plot=True):
    """
    Robust comparison of fit metrics.
    Returns structured output even if inputs are invalid.
    """

    # ---------- 1. Validate inputs ----------
    if true_fit_metrics is None or pred_fit_metrics is None:
        return {
            "status": "invalid",
            "reason": "One or both fit_metrics dictionaries are None",
            "true_metrics": true_fit_metrics,
            "pred_metrics": pred_fit_metrics
        }

    required_keys = [
        'elpd_loo', 'elpd_loo_se', 'loo_i',
        'elpd_waic', 'elpd_waic_se',
        'p_loo', 'p_waic'
    ]

    for key in required_keys:
        if key not in true_fit_metrics or key not in pred_fit_metrics:
            return {
                "status": "invalid",
                "reason": f"Missing required key: {key}",
                "true_metrics": true_fit_metrics,
                "pred_metrics": pred_fit_metrics
            }

    # ---------- 2. Safe extraction ----------
    try:
        # LOO — use pointwise per-observation differences (loo_i arrays, shape [N])
        # so that np.std operates over N values, not a scalar.
        loo_i_diff = np.asarray(pred_fit_metrics['loo_i']) - np.asarray(true_fit_metrics['loo_i'])
        elpd_diff_loo = pred_fit_metrics['elpd_loo'] - true_fit_metrics['elpd_loo']
        elpd_diff_se_loo = np.sqrt(N) * np.std(loo_i_diff)
        elpd_diff_z_loo = (
            elpd_diff_loo / elpd_diff_se_loo
            if elpd_diff_se_loo > 0 and not np.isnan(elpd_diff_se_loo)
            else np.nan
        )

        # WAIC
        elpd_diff_waic = pred_fit_metrics['elpd_waic'] - true_fit_metrics['elpd_waic']
        elpd_diff_se_waic = np.sqrt(
            pred_fit_metrics['elpd_waic_se']**2 +
            true_fit_metrics['elpd_waic_se']**2
        )
        elpd_diff_z_waic = (
            elpd_diff_waic / elpd_diff_se_waic
            if elpd_diff_se_waic > 0 and not np.isnan(elpd_diff_se_waic)
            else np.nan
        )

        p_diff_loo = pred_fit_metrics['p_loo'] - true_fit_metrics['p_loo']
        p_diff_waic = pred_fit_metrics['p_waic'] - true_fit_metrics['p_waic']

    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "true_metrics": true_fit_metrics,
            "pred_metrics": pred_fit_metrics
        }

    # ---------- 3. Interpretation ----------
    def interpret_diff(z_score):
        if np.isnan(z_score):
            return "Undefined"
        if abs(z_score) < 2:
            return "Essentially equivalent"
        elif abs(z_score) < 4:
            return "Small difference"
        elif abs(z_score) < 10:
            return "Moderate difference"
        else:
            return "Substantial difference"

    interpretation_loo = interpret_diff(elpd_diff_z_loo)
    interpretation_waic = interpret_diff(elpd_diff_z_waic)

    results = {
        'status': 'ok',
        'elpd_diff_loo': elpd_diff_loo,
        'elpd_diff_se_loo': elpd_diff_se_loo,
        'elpd_diff_z_loo': elpd_diff_z_loo,
        'interpretation_loo': interpretation_loo,
        'elpd_diff_waic': elpd_diff_waic,
        'elpd_diff_se_waic': elpd_diff_se_waic,
        'elpd_diff_z_waic': elpd_diff_z_waic,
        'interpretation_waic': interpretation_waic,
        'p_diff_loo': p_diff_loo,
        'p_diff_waic': p_diff_waic,
        'true_metrics': true_fit_metrics,
        'pred_metrics': pred_fit_metrics,
        # --- NEW: absolute values for both models ---
        'true_elpd_loo': true_fit_metrics['elpd_loo'],
        'true_elpd_loo_se': true_fit_metrics['elpd_loo_se'],
        'true_elpd_waic': true_fit_metrics['elpd_waic'],
        'true_elpd_waic_se': true_fit_metrics['elpd_waic_se'],
        'true_p_loo': true_fit_metrics['p_loo'],
        'true_p_waic': true_fit_metrics['p_waic'],
        'pred_elpd_loo': pred_fit_metrics['elpd_loo'],
        'pred_elpd_loo_se': pred_fit_metrics['elpd_loo_se'],
        'pred_elpd_waic': pred_fit_metrics['elpd_waic'],
        'pred_elpd_waic_se': pred_fit_metrics['elpd_waic_se'],
        'pred_p_loo': pred_fit_metrics['p_loo'],
        'pred_p_waic': pred_fit_metrics['p_waic']
    }

    # ---------- 4. Plot only if valid ----------
    if plot:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            models = ['True', 'Predicted']
            elpd_loo_vals = [
                true_fit_metrics['elpd_loo'],
                pred_fit_metrics['elpd_loo']
            ]
            elpd_loo_ses = [
                true_fit_metrics['elpd_loo_se'],
                pred_fit_metrics['elpd_loo_se']
            ]

            ax1.errorbar(
                models,
                elpd_loo_vals,
                yerr=[2*se for se in elpd_loo_ses],
                fmt='o',
                capsize=10
            )
            ax1.set_ylabel('ELPD LOO')
            ax1.set_title('ELPD LOO (±2 SE)')
            ax1.grid(True, alpha=0.3)

            p_loo_vals = [
                true_fit_metrics['p_loo'],
                pred_fit_metrics['p_loo']
            ]
            p_waic_vals = [
                true_fit_metrics['p_waic'],
                pred_fit_metrics['p_waic']
            ]

            x = np.arange(len(models))
            width = 0.35

            ax2.bar(x - width/2, p_loo_vals, width, label='p_loo')
            ax2.bar(x + width/2, p_waic_vals, width, label='p_waic')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models)
            ax2.legend()
            ax2.grid(True, axis='y', alpha=0.3)

            plt.tight_layout()
            plt.show()

        except Exception:
            pass  # plotting failure shouldn't break loop

    return results

# First, we have a simple matching score based on distribution family names.
def calculate_dist_match_score(true_dist, pred_dist):
    """
    Calculate matching score between true_dist and pred_dist.
    
    Parameters:
    - true_dist: LIST of distributions
    - pred_dist: LIST of distributions
    
    Returns:
    - score: float between 0 and 1
    """
    # Split the distributions
    true_components = true_dist
    pred_components = pred_dist

    # Count matches
    matches = len(true_components.intersection(pred_components))

    # Number of components in true_dist
    n_true = len(true_components)

    # Calculate score
    if matches == n_true and len(pred_components) == n_true:
        # Perfect match
        return 1.0
    else:
        # Partial match: proportion of true components found in pred
        return matches / n_true

class DistributionBuilder:
    """Build PDF functions from MAP estimates."""

    @staticmethod
    def parse_distribution_name(dist_name):
        """Parse distribution name into components.
        
        Args:
            dist_name: String like 'gaussian', 'cauchy', or 'gaussian_cauchy'
        
        Returns:
            List of distribution component names
        """
        return dist_name.split('_')

    @staticmethod
    def build_gaussian_pdf(mu, sigma):
        """Build Gaussian PDF function."""
        return lambda x: stats.norm.pdf(x, loc=mu, scale=sigma)

    @staticmethod
    def build_cauchy_pdf(loc, scale):
        """Build Cauchy PDF function."""
        return lambda x: stats.cauchy.pdf(x, loc=loc, scale=scale)

    @staticmethod
    def build_laplace_pdf(loc, scale):
        """Build Laplace PDF function."""
        return lambda x: stats.laplace.pdf(x, loc=loc, scale=scale)

    @staticmethod
    def build_student_t_pdf(loc, scale, df):
        """Build Student's t PDF function."""
        return lambda x: stats.t.pdf(x, df=df, loc=loc, scale=scale)

    @staticmethod
    def build_lognormal_pdf(mu, sigma):
        """Build Lognormal PDF function."""
        return lambda x: stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    @staticmethod
    def build_exponential_pdf(loc, scale):
        """Build Exponential PDF function."""
        return lambda x: stats.expon.pdf(x, loc=loc, scale=scale)

    @staticmethod
    def build_uniform_pdf(low, high):
        """Build Uniform PDF function."""
        return lambda x: stats.uniform.pdf(x, loc=low, scale=high - low)

    @staticmethod
    def build_weibull_pdf(loc, scale, alpha):
        """Build Weibull PDF function."""
        return lambda x: stats.weibull_min.pdf(x, c=alpha, loc=loc, scale=scale)

    @classmethod
    def extract_component_params(cls, map_estimate, component_name, index):
        """Extract parameters for a specific distribution component."""
        params = {}

        component_name = component_name.lower().replace('-', '')

        if component_name == 'gaussian':
            mu_key = f'gaussian_mu_{index}'
            sigma_key = f'gaussian_sigma_{index}'
            sigma_log_key = f'gaussian_sigma_{index}_log__'

            params['mu'] = float(map_estimate.get(mu_key, 0.0))

            if sigma_key in map_estimate:
                params['sigma'] = float(map_estimate[sigma_key])
            elif sigma_log_key in map_estimate:
                params['sigma'] = float(np.exp(map_estimate[sigma_log_key]))
            else:
                params['sigma'] = 1.0

        elif component_name == 'cauchy':
            loc_key = f'cauchy_loc_{index}'
            scale_key = f'cauchy_scale_{index}'
            scale_log_key = f'cauchy_scale_{index}_log__'
            alpha_key = f'cauchy_alpha_{index}'
            beta_key = f'cauchy_beta_{index}'

            if loc_key in map_estimate:
                params['loc'] = float(map_estimate[loc_key])
            elif alpha_key in map_estimate:
                params['loc'] = float(map_estimate[alpha_key])
            else:
                params['loc'] = 0.0

            if scale_key in map_estimate:
                params['scale'] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params['scale'] = float(np.exp(map_estimate[scale_log_key]))
            elif beta_key in map_estimate:
                params['scale'] = float(map_estimate[beta_key])
            else:
                params['scale'] = 1.0

        elif component_name == 'laplace':
            loc_key = f'laplace_loc_{index}'
            scale_key = f'laplace_scale_{index}'
            scale_log_key = f'laplace_scale_{index}_log__'
            mu_key = f'laplace_mu_{index}'
            b_key = f'laplace_b_{index}'

            if loc_key in map_estimate:
                params['loc'] = float(map_estimate[loc_key])
            elif mu_key in map_estimate:
                params['loc'] = float(map_estimate[mu_key])
            else:
                params['loc'] = 0.0

            if scale_key in map_estimate:
                params['scale'] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params['scale'] = float(np.exp(map_estimate[scale_log_key]))
            elif b_key in map_estimate:
                params['scale'] = float(map_estimate[b_key])
            else:
                params['scale'] = 1.0

        elif component_name in ['student', 't', 'studentt']:
            loc_key = f'student_loc_{index}'
            scale_key = f'student_scale_{index}'
            df_key = f'student_df_{index}'
            mu_key = f'student_mu_{index}'
            sigma_key = f'student_sigma_{index}'
            nu_key = f'student_nu_{index}'

            if loc_key in map_estimate:
                params['loc'] = float(map_estimate[loc_key])
            elif mu_key in map_estimate:
                params['loc'] = float(map_estimate[mu_key])
            else:
                params['loc'] = 0.0

            if scale_key in map_estimate:
                params['scale'] = float(map_estimate[scale_key])
            elif sigma_key in map_estimate:
                params['scale'] = float(map_estimate[sigma_key])
            else:
                params['scale'] = 1.0

            if df_key in map_estimate:
                params['df'] = float(map_estimate[df_key])
            elif nu_key in map_estimate:
                params['df'] = float(map_estimate[nu_key])
            else:
                params['df'] = 3.0

        elif component_name == 'lognormal':
            mu_key = f'lognormal_mu_{index}'
            sigma_key = f'lognormal_sigma_{index}'
            sigma_log_key = f'lognormal_sigma_{index}_log__'

            params['mu'] = float(map_estimate.get(mu_key, 0.0))

            if sigma_key in map_estimate:
                params['sigma'] = float(map_estimate[sigma_key])
            elif sigma_log_key in map_estimate:
                params['sigma'] = float(np.exp(map_estimate[sigma_log_key]))
            else:
                params['sigma'] = 1.0

        elif component_name == 'exponential':
            loc_key = f'exponential_loc_{index}'
            scale_key = f'exponential_scale_{index}'
            scale_log_key = f'exponential_scale_{index}_log__'

            params['loc'] = float(map_estimate.get(loc_key, 0.0))

            if scale_key in map_estimate:
                params['scale'] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params['scale'] = float(np.exp(map_estimate[scale_log_key]))
            else:
                params['scale'] = 1.0

        elif component_name == 'uniform':
            low_key = f'uniform_low_{index}'
            high_key = f'uniform_high_{index}'

            params['low'] = float(map_estimate.get(low_key, 0.0))
            params['high'] = float(map_estimate.get(high_key, 1.0))

        elif component_name == 'weibull':
            loc_key = f'weibull_loc_{index}'
            scale_key = f'weibull_scale_{index}'
            alpha_key = f'weibull_alpha_{index}'
            scale_log_key = f'weibull_scale_{index}_log__'
            alpha_log_key = f'weibull_alpha_{index}_log__'

            params['loc'] = float(map_estimate.get(loc_key, 0.0))

            if scale_key in map_estimate:
                params['scale'] = float(map_estimate[scale_key])
            elif scale_log_key in map_estimate:
                params['scale'] = float(np.exp(map_estimate[scale_log_key]))
            else:
                params['scale'] = 1.0

            if alpha_key in map_estimate:
                params['alpha'] = float(map_estimate[alpha_key])
            elif alpha_log_key in map_estimate:
                params['alpha'] = float(np.exp(map_estimate[alpha_log_key]))
            else:
                params['alpha'] = 1.5

        return params

    @classmethod
    def build_pdf_from_map(cls, map_estimate, dist_name):
        """Build PDF function from MAP estimates."""
        components = dist_name
    
        # Find weight key — handle any name like 'w', 'w_gaussian_cauchy', etc.
        w_key = next(
            (k for k in map_estimate
             if k.startswith('w')
             and 'simplex' not in k
             and isinstance(map_estimate[k], (np.ndarray, list))
             and len(np.atleast_1d(map_estimate[k])) == len(components)),
            None
        )
        simplex_key = next(
            (k for k in map_estimate if k.startswith('w') and 'simplex' in k),
            None
        )
    
        if w_key:
            weights = np.array(map_estimate[w_key])
        elif simplex_key and len(components) > 1:
            w_simplex = np.array(map_estimate[simplex_key])
            weights = cls.simplex_to_weights(w_simplex, len(components))
        else:
            weights = np.ones(len(components)) / len(components)
    
        weights = weights / np.sum(weights)
    
        component_pdfs = []
        for idx, component_name in enumerate(components):
            component_name = component_name.lower().replace('-', '')
            params = cls.extract_component_params(map_estimate, component_name, idx)
    
            if component_name == 'gaussian':
                pdf = cls.build_gaussian_pdf(params['mu'], params['sigma'])
            elif component_name == 'cauchy':
                pdf = cls.build_cauchy_pdf(params['loc'], params['scale'])
            elif component_name == 'laplace':
                pdf = cls.build_laplace_pdf(params['loc'], params['scale'])
            elif component_name in ['student', 't', 'studentt']:
                pdf = cls.build_student_t_pdf(params['loc'], params['scale'], params['df'])
            elif component_name == 'lognormal':
                pdf = cls.build_lognormal_pdf(params['mu'], params['sigma'])
            elif component_name == 'exponential':
                pdf = cls.build_exponential_pdf(params['loc'], params['scale'])
            elif component_name == 'uniform':
                pdf = cls.build_uniform_pdf(params['low'], params['high'])
            elif component_name == 'weibull':
                pdf = cls.build_weibull_pdf(params['loc'], params['scale'], params['alpha'])
            else:
                raise ValueError(f"Unknown distribution: {component_name}")
    
            component_pdfs.append(pdf)
    
        def mixture_pdf(x):
            x = np.atleast_1d(x)
            result = np.zeros_like(x, dtype=float)
            for weight, pdf in zip(weights, component_pdfs):
                result += weight * pdf(x)
            return result if len(x) > 1 else result[0]
    
        return mixture_pdf

    @staticmethod
    def simplex_to_weights(w_simplex, n_components):
        """Convert simplex representation to weights."""
        if len(w_simplex) == n_components - 1:
            weights = np.zeros(n_components)
            weights[:-1] = w_simplex
            weights[-1] = 1.0 - np.sum(w_simplex)
            return weights
        else:
            return w_simplex

class DivergenceCalculator:
    """Calculate divergence metrics between two PDFs."""

    def __init__(self, x_range=None, n_points=10000):
        self.x_range = x_range if x_range is not None else (-50, 50)
        self.n_points = n_points

    def get_integration_points(self, pdf_p, pdf_q):
        """Determine appropriate integration range based on PDFs."""
        if self.x_range is not None:
            return self.x_range

        x_test = np.linspace(-100, 100, 1000)
        p_vals = pdf_p(x_test)
        q_vals = pdf_q(x_test)

        threshold = 1e-6
        p_support = x_test[(p_vals > threshold)]
        q_support = x_test[(q_vals > threshold)]

        if len(p_support) > 0 and len(q_support) > 0:
            x_min = min(p_support.min(), q_support.min()) - 5
            x_max = max(p_support.max(), q_support.max()) + 5
            return (x_min, x_max)

        return (-50, 50)

    def kl_divergence(self, pdf_p, pdf_q, x=None):
        """Calculate KL divergence D_KL(P || Q)."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)

        mask = (p > 1e-10) & (q > 1e-10)

        if not np.any(mask):
            return np.inf

        integrand = np.zeros_like(p)
        integrand[mask] = p[mask] * np.log(p[mask] / q[mask])

        return float(np.trapezoid(integrand, x))

    def js_divergence(self, pdf_p, pdf_q, x=None):
        """Calculate Jensen-Shannon divergence."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)
        m = (p + q) / 2

        kl_pm = self.kl_divergence(lambda xi: np.interp(xi, x, p),
                                   lambda xi: np.interp(xi, x, m), x)
        kl_qm = self.kl_divergence(lambda xi: np.interp(xi, x, q),
                                   lambda xi: np.interp(xi, x, m), x)

        return float(0.5 * (kl_pm + kl_qm))

    def hellinger_distance(self, pdf_p, pdf_q, x=None):
        """Calculate Hellinger distance."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)

        integrand = (np.sqrt(p) - np.sqrt(q)) ** 2
        h_squared = np.trapezoid(integrand, x) / 2

        return float(np.sqrt(h_squared))

    def bhattacharyya_distance(self, pdf_p, pdf_q, x=None):
        """Calculate Bhattacharyya distance."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)

        bc = np.trapezoid(np.sqrt(p * q), x)
        bc = np.clip(bc, 0, 1)

        if bc == 0:
            return np.inf

        return float(-np.log(bc))

    def total_variation(self, pdf_p, pdf_q, x=None):
        """Calculate total variation distance."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)

        integrand = np.abs(p - q)

        return float(0.5 * np.trapezoid(integrand, x))

    def l2_distance(self, pdf_p, pdf_q, x=None):
        """Calculate L2 (Euclidean) distance."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)

        integrand = (p - q) ** 2

        return float(np.sqrt(np.trapezoid(integrand, x)))

    def chi_squared(self, pdf_p, pdf_q, x=None):
        """Calculate chi-squared divergence."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        p = pdf_p(x)
        q = pdf_q(x)

        mask = q > 1e-10

        if not np.any(mask):
            return np.inf

        integrand = np.zeros_like(p)
        integrand[mask] = (p[mask] - q[mask]) ** 2 / q[mask]

        return float(np.trapezoid(integrand, x))

    def compute_all_metrics(self, pdf_p, pdf_q, x=None):
        """Compute all divergence metrics."""
        if x is None:
            x_range = self.get_integration_points(pdf_p, pdf_q)
            x = np.linspace(x_range[0], x_range[1], self.n_points)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            kl_pq = self.kl_divergence(pdf_p, pdf_q, x)
            kl_qp = self.kl_divergence(pdf_q, pdf_p, x)
            js_divergence = self.js_divergence(pdf_p, pdf_q, x)
            hellinger = self.hellinger_distance(pdf_p, pdf_q, x)
            bhattacharyya = self.bhattacharyya_distance(pdf_p, pdf_q, x)
            total_variation = self.total_variation(pdf_p, pdf_q, x)
            l2_distance = self.l2_distance(pdf_p, pdf_q, x)
            chi_squared = self.chi_squared(pdf_p, pdf_q, x)

        return {
            'kl_divergence_pq': float(kl_pq),
            'kl_divergence_qp': float(kl_qp),
            'js_divergence': float(js_divergence),
            'hellinger_distance': float(hellinger),
            'bhattacharyya_distance': float(bhattacharyya),
            'total_variation': float(total_variation),
            'l2_distance': float(l2_distance),
            'chi_squared': float(chi_squared)
        }


def compare_distributions(true_map_estimate, pred_map_estimate,
                         true_dist_name, pred_dist_name,
                         x_range=None, n_points=10000):
    """
    Complete pipeline to compare two distributions.
    Raises immediately if either map_estimate is None rather than propagating
    into build_pdf_from_map where the error message would be cryptic.
    """
    if true_map_estimate is None:
        raise ValueError("true_map_estimate is None — true model fitting must have failed silently")
    if pred_map_estimate is None:
        raise ValueError("pred_map_estimate is None — pred model fitting must have failed silently")

    builder = DistributionBuilder()
    pdf_true = builder.build_pdf_from_map(true_map_estimate, true_dist_name)
    pdf_pred = builder.build_pdf_from_map(pred_map_estimate, pred_dist_name)

    calculator = DivergenceCalculator(x_range=x_range, n_points=n_points)
    metrics = calculator.compute_all_metrics(pdf_true, pdf_pred)

    return metrics, pdf_true, pdf_pred
    
def sample_from_map_estimate(map_estimate, dist_name, n_samples=50000):
    """Generate samples from a distribution given its MAP estimate."""
    builder = DistributionBuilder()
    components = dist_name

    # Find weight key — handle any name like 'w', 'w_gaussian_cauchy', etc.
    w_key = next(
        (k for k in map_estimate
         if k.startswith('w')
         and 'simplex' not in k
         and isinstance(map_estimate[k], (np.ndarray, list))
         and len(np.atleast_1d(map_estimate[k])) == len(components)),
        None
    )
    simplex_key = next(
        (k for k in map_estimate if k.startswith('w') and 'simplex' in k),
        None
    )

    if w_key:
        weights = np.array(map_estimate[w_key])
    elif simplex_key and len(components) > 1:
        w_simplex = np.array(map_estimate[simplex_key])
        weights = builder.simplex_to_weights(w_simplex, len(components))
    else:
        weights = np.ones(len(components)) / len(components)

    weights = weights / np.sum(weights)

    component_samples = np.random.choice(len(components), size=n_samples, p=weights)
    samples = np.zeros(n_samples)

    def _get(keys, params, required=True):
        for k in keys:
            if k in params:
                return params[k]
        if required:
            raise ValueError(f"Required parameter(s) {keys} not found in params: {params}")
        return None

    for idx, component_name in enumerate(components):
        mask = component_samples == idx
        n_comp_samples = int(np.sum(mask))
        if n_comp_samples == 0:
            continue

        params = builder.extract_component_params(map_estimate, component_name, idx)

        if component_name in ['gaussian', 'normal']:
            mu = _get(['mu', 'mean', f'{component_name}_mu', f'gaussian_mu_{idx}'], params)
            sigma = _get(['sigma', 'std', 'sd', f'{component_name}_sigma', f'gaussian_sigma_{idx}'], params)
            samples[mask] = stats.norm.rvs(loc=mu, scale=sigma, size=n_comp_samples)

        elif component_name in ['lognormal', 'log-normal', 'log_normal']:
            mu = _get(['mu', 'lognormal_mu', f'lognormal_mu_{idx}'], params)
            sigma = _get(['sigma', 'lognormal_sigma', f'lognormal_sigma_{idx}'], params)
            samples[mask] = stats.lognorm.rvs(s=sigma, scale=np.exp(mu), size=n_comp_samples)

        elif component_name in ['cauchy']:
            loc = _get(['loc', 'alpha', 'cauchy_alpha', f'cauchy_alpha_{idx}'], params)
            scale = _get(['scale', 'beta', 'cauchy_beta', f'cauchy_beta_{idx}'], params)
            samples[mask] = stats.cauchy.rvs(loc=loc, scale=scale, size=n_comp_samples)

        elif component_name in ['laplace']:
            mu = _get(['mu', 'loc', 'laplace_mu', f'laplace_mu_{idx}'], params)
            b = _get(['b', 'scale', 'laplace_b', f'laplace_b_{idx}'], params)
            samples[mask] = stats.laplace.rvs(loc=mu, scale=b, size=n_comp_samples)

        elif component_name in ['student', 'student-t', 'studentt', 't']:
            df = _get(['df', 'nu', 'studentt_nu', f'studentt_nu_{idx}'], params)
            mu = _get(['mu', 'loc', 'studentt_mu', f'studentt_mu_{idx}'], params)
            sigma = _get(['sigma', 'scale', 'studentt_sigma', f'studentt_sigma_{idx}'], params)
            samples[mask] = stats.t.rvs(df=df, loc=mu, scale=sigma, size=n_comp_samples)

        elif component_name in ['exponential', 'exp']:
            if 'scale' in params:
                scale = params['scale']
            else:
                lam = _get(
                    ['lam', 'lambda', 'exponential_lam', f'exponential_lam_{idx}'],
                    params
                )
                if lam <= 0:
                    raise ValueError(f"Invalid exponential rate lam={lam}")
                scale = 1.0 / lam

            loc = _get(
                ['loc', 'exponential_loc', f'exponential_loc_{idx}'],
                params,
                required=False
            )
            if loc is None:
                loc = 0.0

            samples[mask] = stats.expon.rvs(scale=scale, loc=loc, size=n_comp_samples)

        elif component_name in ['uniform']:
            lower = _get(['lower', 'low', 'uniform_lower', f'uniform_lower_{idx}'], params)
            upper = _get(['upper', 'high', 'uniform_upper', f'uniform_upper_{idx}'], params)
            if upper <= lower:
                raise ValueError(f"Uniform upper ({upper}) must be > lower ({lower}).")
            samples[mask] = stats.uniform.rvs(loc=lower, scale=(upper - lower), size=n_comp_samples)

        elif component_name in ['weibull']:
            alpha = _get(['alpha', 'weibull_alpha', f'weibull_alpha_{idx}'], params)
            scale = _get(['scale', 'weibull_scale', f'weibull_scale_{idx}', 'beta', 'weibull_beta'], params)
            loc = _get(['loc', 'weibull_loc', f'weibull_loc_{idx}'], params, required=False)
            if loc is None:
                loc = 0.0
            samples[mask] = stats.weibull_min.rvs(c=alpha, scale=scale, loc=loc, size=n_comp_samples)

        else:
            raise ValueError(
                f"Unsupported component family: '{component_name}'. "
                f"Supported: gaussian, lognormal, cauchy, laplace, student/t, exponential, uniform, weibull."
            )

    return samples

def plot_suite(true_map_estimate, pred_map_estimate,
                                true_dist_name, pred_dist_name,
                                data, title=None, figsize=(18, 5),
                                n_samples=50000):
    """
    Visualization with Q-Q plot, PDF comparison, and KDE + PDFs overlay.
    """
    metrics, true_pdf, pred_pdf = compare_distributions(
        true_map_estimate, pred_map_estimate,
        true_dist_name, pred_dist_name
    )

    true_samples = sample_from_map_estimate(true_map_estimate, true_dist_name, n_samples)
    pred_samples = sample_from_map_estimate(pred_map_estimate, pred_dist_name, n_samples)

    x_min = min(data.min(), true_samples.min(), pred_samples.min())
    x_max = max(data.max(), true_samples.max(), pred_samples.max())
    padding = 0.1 * (x_max - x_min)
    x = np.linspace(x_min - padding, x_max + padding, 1000)

    true_pdf_vals = true_pdf(x)
    pred_pdf_vals = pred_pdf(x)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    true_sorted = np.sort(true_samples)
    pred_sorted = np.sort(pred_samples)
    n_points = min(len(true_sorted), len(pred_sorted))
    true_qq = true_sorted[:n_points]
    pred_qq = pred_sorted[:n_points]

    axes[0].scatter(true_qq, pred_qq, alpha=0.3, s=1, color='green')
    qq_min = min(true_qq.min(), pred_qq.min())
    qq_max = max(true_qq.max(), pred_qq.max())
    axes[0].plot([qq_min, qq_max], [qq_min, qq_max], 'r--', linewidth=2, label='Perfect match')
    axes[0].set_xlabel('True Distribution Quantiles', fontsize=13)
    axes[0].set_ylabel('Predicted Distribution Quantiles', fontsize=13)
    axes[0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_aspect('equal', adjustable='box')

    axes[1].plot(x, true_pdf_vals, label='True Distribution', linewidth=3,
                 color='#1f77b4', alpha=0.9, zorder=3)
    axes[1].plot(x, pred_pdf_vals, label='Predicted Distribution', linewidth=3,
                 color='#ff7f0e', linestyle='--', alpha=0.9, zorder=2)
    axes[1].fill_between(x, true_pdf_vals, alpha=0.15, color='#1f77b4', zorder=1)
    axes[1].fill_between(x, pred_pdf_vals, alpha=0.15, color='#ff7f0e', zorder=0)
    axes[1].set_xlabel('x', fontsize=13)
    axes[1].set_ylabel('Probability Density', fontsize=13)
    axes[1].set_title('PDF Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')

    kde = stats.gaussian_kde(data)
    kde_vals = kde(x)

    axes[2].plot(x, kde_vals, label='Data KDE', linewidth=3,
                 color='black', alpha=0.8, zorder=4)
    axes[2].plot(x, true_pdf_vals, label='True PDF', linewidth=2.5,
                 color='#1f77b4', alpha=0.7, zorder=3)
    axes[2].plot(x, pred_pdf_vals, label='Predicted PDF', linewidth=2.5,
                 color='#ff7f0e', linestyle='--', alpha=0.7, zorder=2)
    axes[2].fill_between(x, kde_vals, alpha=0.1, color='black', zorder=1)
    axes[2].set_xlabel('x', fontsize=13)
    axes[2].set_ylabel('Probability Density', fontsize=13)
    axes[2].set_title('Data KDE vs PDFs', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=12, loc='best', framealpha=0.9)
    axes[2].grid(True, alpha=0.3, linestyle='--')

    if title is None:
        title = f'{pred_dist_name} | JS Div: {metrics["js_divergence"]:.4f}'

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig, metrics


def clean_params(params_dict):
    clean = {}
    for k, v in params_dict.items():
        if isinstance(v, np.ndarray):
            if v.shape == () or v.shape == (1,):
                clean[k] = float(v)
            else:
                clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean


# FITTING INITIAL MODEL:
def execute_and_fit_model(data, model_code):
    exec_namespace = {'pm': pm, 'np': np, 'data': data}

    exec(model_code, exec_namespace)

    model = exec_namespace['model']
    map_estimate = exec_namespace['map_estimate']

    print("MODEL CODE:\n", model_code)
    print("MODEL:", model)
    print("MAP ESTIMATE:", map_estimate)

    with model:
        trace = pm.sample(
            draws=200,
            tune=200,
            chains=4,
            target_accept=0.85,
            nuts={"max_treedepth": 8},
            progressbar=True,
            discard_tuned_samples=True,
            idata_kwargs={"log_likelihood": True},
            cores=4,
        )

    waic = az.waic(trace)
    loo = az.loo(trace)

    fit_metrics = {
        "elpd_waic": float(waic.elpd_waic),
        "elpd_waic_se": float(waic.se),
        "p_waic": float(waic.p_waic),
        "elpd_loo": float(loo.elpd_loo),
        "elpd_loo_se": float(loo.se),
        "p_loo": float(loo.p_loo),
        "loo_i": np.array(loo.loo_i.values).flatten(),
        "loo_good_k": int((loo.pareto_k < 0.7).sum()),
        "loo_bad_k": int((loo.pareto_k > 0.7).sum()),
    }

    return model, map_estimate, fit_metrics


def execute_model(data, ans):
    model_code = ans['pymc_model']
    model_code = model_code.replace(', sd=', ', sigma=').replace('(sd=', '(sigma=')

    exec_namespace = {'pm': pm, 'np': np, 'data': data}

    exec(model_code, exec_namespace)

    model = exec_namespace['model']
    map_estimate = exec_namespace['map_estimate']

    return model, map_estimate


def fix_densitydist_pattern(model_code: str):
    """
    Fixes the PyMC3 -> PyMC4+ breaking pattern.
    """
    if not isinstance(model_code, str):
        return model_code

    if '.dist(' not in model_code or 'DensityDist' not in model_code:
        return model_code

    lines = model_code.split('\n')

    dist_pattern = re.compile(
        r'^(\s*)(\w+)\s*=\s*pm\.(\w+)\.dist\((.+)\)\s*$'
    )
    density_pattern = re.compile(
        r'^(\s*)(\w+)\s*=\s*pm\.DensityDist\(\s*[\'"](\w+)[\'"]\s*,\s*(\w+)\.logp\s*,\s*observed\s*=\s*\{[\'"]value[\'"]\s*:\s*(\w+)\}\s*\)\s*$'
    )

    dist_map = {}
    for line in lines:
        m = dist_pattern.match(line)
        if m:
            _, var_name, dist_name, dist_args = m.groups()
            dist_map[var_name] = (dist_name, dist_args)

    if not dist_map:
        return model_code

    new_lines = []
    removed_vars = set()

    for line in lines:
        m = density_pattern.match(line)
        if m:
            indent, lhs, obs_name, comp_var, data_var = m.groups()
            if comp_var in dist_map:
                dist_name, dist_args = dist_map[comp_var]
                new_lines.append(
                    f"{indent}{lhs} = pm.{dist_name}('{obs_name}', {dist_args}, observed={data_var})"
                )
                removed_vars.add(comp_var)
                continue

        m2 = dist_pattern.match(line)
        if m2:
            _, var_name, _, _ = m2.groups()
            if var_name in removed_vars:
                continue

        new_lines.append(line)

    return '\n'.join(new_lines)


def fix_pymc_model_code_column(df):
    df = df.copy()
    df['pymc_model_code_fixed'] = df['pymc_model_code'].apply(fix_densitydist_pattern)

    changed = (df['pymc_model_code'] != df['pymc_model_code_fixed']).sum()
    print(f"Fixed {changed} / {len(df)} rows")

    return df

def extract_model_code(value):
    """
    Normalize pymc_model_code values.
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if stripped.startswith("{"):
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, dict):
                return parsed.get('0', parsed)
        except (ValueError, SyntaxError):
            pass

    return value


def fit_true_model(data, true_params):
    """
    Fit a PyMC model using the exact true parameters that generated the data.

    FIX: All initval= arguments have been moved out of the distribution
    constructors and into a separate initvals dict passed to pm.sample().
    This prevents the PyMC fgraph_from_model error:
        NotImplementedError: Cannot convert models with non-default initial_values
    """
    # Normalize single-distribution params into mixture format
    if 'dist_choice' in true_params:
        true_params = {
            'num_components': 1,
            'total_n': true_params.get('n', len(data)),
            'weights': [1.0],
            'components': [true_params],
        }

    try:
        num_components = true_params['num_components']
        components = true_params['components']
        weights = true_params['weights']

        # Collect initial values separately — never pass initval= to constructors
        # as that populates model.rvs_to_initial_values and breaks fgraph_from_model.
        initvals = {}

        with pm.Model() as model:

            if num_components > 1:
                w = pm.Dirichlet('w', a=np.ones(num_components))
                initvals['w'] = weights
            else:
                w = np.array([1.0])

            comp_dists = []

            for i, comp in enumerate(components):
                dist_type = comp['dist_choice']

                if dist_type == 'gaussian':
                    gaussian_mu = pm.Normal(f'gaussian_mu_{i}', mu=comp['mean'], sigma=10)
                    gaussian_sigma = pm.HalfNormal(f'gaussian_sigma_{i}', sigma=5)
                    initvals[f'gaussian_mu_{i}'] = comp['mean']
                    initvals[f'gaussian_sigma_{i}'] = comp['std']
                    comp_dists.append(pm.Normal.dist(mu=gaussian_mu, sigma=gaussian_sigma))

                elif dist_type == 'studentt' or dist_type == 'student-t':
                    studentt_nu = pm.Gamma(f'studentt_nu_{i}', alpha=2, beta=0.1)
                    studentt_mu = pm.Normal(f'studentt_mu_{i}', mu=comp['mu'], sigma=10)
                    studentt_sigma = pm.HalfNormal(f'studentt_sigma_{i}', sigma=5)
                    initvals[f'studentt_nu_{i}'] = comp['nu']
                    initvals[f'studentt_mu_{i}'] = comp['mu']
                    initvals[f'studentt_sigma_{i}'] = comp['sigma']
                    comp_dists.append(pm.StudentT.dist(nu=studentt_nu, mu=studentt_mu, sigma=studentt_sigma))

                elif dist_type == 'lognormal':
                    lognormal_mu = pm.Normal(f'lognormal_mu_{i}', mu=comp['mu'], sigma=10)
                    lognormal_sigma = pm.HalfNormal(f'lognormal_sigma_{i}', sigma=5)
                    initvals[f'lognormal_mu_{i}'] = comp['mu']
                    initvals[f'lognormal_sigma_{i}'] = max(comp['sigma'], 0.1)
                    comp_dists.append(pm.LogNormal.dist(mu=lognormal_mu, sigma=lognormal_sigma))

                elif dist_type == 'cauchy':
                    cauchy_alpha = pm.Normal(f'cauchy_alpha_{i}', mu=comp['alpha'], sigma=10)
                    cauchy_beta = pm.HalfNormal(f'cauchy_beta_{i}', sigma=5)
                    initvals[f'cauchy_alpha_{i}'] = comp['alpha']
                    initvals[f'cauchy_beta_{i}'] = comp['beta']
                    comp_dists.append(pm.Cauchy.dist(alpha=cauchy_alpha, beta=cauchy_beta))

                elif dist_type == 'laplace':
                    laplace_mu = pm.Normal(f'laplace_mu_{i}', mu=comp['mu'], sigma=10)
                    laplace_b = pm.HalfNormal(f'laplace_b_{i}', sigma=5)
                    initvals[f'laplace_mu_{i}'] = comp['mu']
                    initvals[f'laplace_b_{i}'] = comp['b']
                    comp_dists.append(pm.Laplace.dist(mu=laplace_mu, b=laplace_b))

                elif dist_type == 'uniform':
                    mid = (comp['low'] + comp['high']) / 2.0
                    half_width = (comp['high'] - comp['low']) / 2.0
                    uniform_mid = pm.Normal(f'uniform_mid_{i}', mu=mid, sigma=5)
                    uniform_hw = pm.HalfNormal(f'uniform_hw_{i}', sigma=half_width * 2)
                    uniform_lower = pm.Deterministic(f'uniform_lower_{i}', uniform_mid - uniform_hw)
                    uniform_upper = pm.Deterministic(f'uniform_upper_{i}', uniform_mid + uniform_hw)
                    initvals[f'uniform_mid_{i}'] = mid
                    initvals[f'uniform_hw_{i}'] = half_width
                    comp_dists.append(pm.Uniform.dist(lower=uniform_lower, upper=uniform_upper))

                elif dist_type == 'exponential':
                    true_scale = comp['scale']
                    true_loc = comp['loc']
                    safe_loc = min(true_loc, np.min(data) - 0.1)

                    exponential_scale = pm.HalfNormal(f'exponential_scale_{i}', sigma=5)
                    exponential_loc = pm.Normal(f'exponential_loc_{i}', mu=safe_loc, sigma=10)
                    initvals[f'exponential_scale_{i}'] = true_scale
                    initvals[f'exponential_loc_{i}'] = safe_loc

                    class ShiftedExponential:
                        @staticmethod
                        def dist(scale, loc, **kwargs):
                            def logp(value, scale, loc):
                                shifted_value = value - loc
                                return pt.switch(
                                    shifted_value >= 0,
                                    -pt.log(scale) - shifted_value / scale,
                                    -np.inf
                                )
                            return pm.CustomDist.dist(scale, loc, logp=logp, **kwargs)

                    comp_dists.append(ShiftedExponential.dist(exponential_scale, exponential_loc))

                elif dist_type == 'weibull':
                    true_alpha = comp['alpha']
                    true_scale = comp['scale']
                    true_loc = comp['loc']
                    safe_loc = min(true_loc, np.min(data) - 0.1)

                    weibull_alpha = pm.Gamma(f'weibull_alpha_{i}', alpha=2, beta=1)
                    weibull_scale = pm.HalfNormal(f'weibull_scale_{i}', sigma=5)
                    weibull_loc = pm.Normal(f'weibull_loc_{i}', mu=safe_loc, sigma=10)
                    initvals[f'weibull_alpha_{i}'] = true_alpha
                    initvals[f'weibull_scale_{i}'] = true_scale
                    initvals[f'weibull_loc_{i}'] = safe_loc

                    class ShiftedWeibull:
                        @staticmethod
                        def dist(alpha, scale, loc, **kwargs):
                            def logp(value, alpha, scale, loc):
                                shifted_value = (value - loc) / scale
                                return pt.switch(
                                    shifted_value >= 0,
                                    pt.log(alpha) - pt.log(scale) + (alpha - 1) * pt.log(shifted_value) - pt.power(shifted_value, alpha),
                                    -np.inf
                                )
                            return pm.CustomDist.dist(alpha, scale, loc, logp=logp, **kwargs)

                    comp_dists.append(ShiftedWeibull.dist(weibull_alpha, weibull_scale, weibull_loc))

                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")

            # Create the likelihood
            if num_components == 1:
                dist_type = components[0]['dist_choice']
                comp = components[0]
                if dist_type == 'gaussian':
                    likelihood = pm.Normal('likelihood', mu=model['gaussian_mu_0'], sigma=model['gaussian_sigma_0'], observed=data)
                elif dist_type in ('studentt', 'student-t'):
                    likelihood = pm.StudentT('likelihood', nu=model['studentt_nu_0'], mu=model['studentt_mu_0'], sigma=model['studentt_sigma_0'], observed=data)
                elif dist_type == 'lognormal':
                    likelihood = pm.LogNormal('likelihood', mu=model['lognormal_mu_0'], sigma=model['lognormal_sigma_0'], observed=data)
                elif dist_type == 'cauchy':
                    likelihood = pm.Cauchy('likelihood', alpha=model['cauchy_alpha_0'], beta=model['cauchy_beta_0'], observed=data)
                elif dist_type == 'laplace':
                    likelihood = pm.Laplace('likelihood', mu=model['laplace_mu_0'], b=model['laplace_b_0'], observed=data)
                elif dist_type == 'uniform':
                    likelihood = pm.Uniform('likelihood', lower=model['uniform_lower_0'], upper=model['uniform_upper_0'], observed=data)
                elif dist_type == 'exponential':
                    scale_var = model['exponential_scale_0']
                    loc_var = model['exponential_loc_0']
                    def single_exp_logp(value, scale, loc):
                        shifted_value = value - loc
                        return pt.switch(shifted_value >= 0, -pt.log(scale) - shifted_value / scale, -np.inf)
                    likelihood = pm.CustomDist('likelihood', scale_var, loc_var, logp=single_exp_logp, observed=data)
                elif dist_type == 'weibull':
                    alpha_var = model['weibull_alpha_0']
                    scale_var = model['weibull_scale_0']
                    loc_var = model['weibull_loc_0']
                    def single_weibull_logp(value, alpha, scale, loc):
                        shifted_value = (value - loc) / scale
                        return pt.switch(shifted_value >= 0, pt.log(alpha) - pt.log(scale) + (alpha - 1) * pt.log(shifted_value) - pt.power(shifted_value, alpha), -np.inf)
                    likelihood = pm.CustomDist('likelihood', alpha_var, scale_var, loc_var, logp=single_weibull_logp, observed=data)
                else:
                    raise ValueError(f"Unsupported single distribution type: {dist_type}")
            else:
                likelihood = pm.Mixture('likelihood', w=w, comp_dists=comp_dists, observed=data)

            # MAP estimation
            try:
                map_estimate = pm.find_MAP(method='L-BFGS-B', maxeval=10000)
            except Exception as map_error:
                print(f"MAP estimation failed with L-BFGS-B, trying Powell: {map_error}")
                try:
                    map_estimate = pm.find_MAP(method='Powell', maxeval=10000)
                except Exception as map_error2:
                    print(f"MAP estimation failed with Powell too: {map_error2}")
                    return None, None, None

        print("TRUE MODEL STRUCTURE:")
        print(model)
        print("\nTRUE MAP ESTIMATE:")
        print(map_estimate)

        # Pass initvals to pm.sample() instead of setting them on the constructors.
        # This keeps model.rvs_to_initial_values empty, which is required for
        # fgraph_from_model to succeed when computing log likelihood.
        with model:
            trace = pm.sample(
                draws=200,
                tune=200,
                chains=4,
                target_accept=0.85,
                nuts={"max_treedepth": 8},
                progressbar=True,
                discard_tuned_samples=True,
                idata_kwargs={'log_likelihood': True},
                cores=4,
                initvals=initvals,
            )

        waic = az.waic(trace)
        loo = az.loo(trace)

        fit_metrics = {
            "elpd_waic": float(waic.elpd_waic),
            "elpd_waic_se": float(waic.se),
            "p_waic": float(waic.p_waic),
            "elpd_loo": float(loo.elpd_loo),
            "elpd_loo_se": float(loo.se),
            "p_loo": float(loo.p_loo),
            "loo_i": np.array(loo.loo_i.values).flatten(),
            "loo_good_k": int((loo.pareto_k < 0.7).sum()),
            "loo_bad_k": int((loo.pareto_k > 0.7).sum()),
        }

        return model, map_estimate, fit_metrics

    except Exception as e:
        print("TRUE MODEL FIT FAILED")
        print("Error:", e)
        traceback.print_exc()
        return None, None, None


def extract_pymc_model_pyvision(text: str) -> dict | None:
    """
    Extracts the PyMC model block from a string containing a <final_pymc_model> block.
    """
    block_match = re.search(
        r'<final_pymc_model>(.*?)</final_pymc_model>',
        text,
        flags=re.DOTALL
    )
    if not block_match:
        return None

    block = block_match.group(1)

    family_match = re.search(r'distribution_family:\s*\[([^\]]+)\]', block)
    distribution_family = (
        [f.strip().strip('"').strip("'") for f in family_match.group(1).split(",")]
        if family_match else []
    )

    mixture_match = re.search(r'is_mixture:\s*(true|false)', block, flags=re.IGNORECASE)
    is_mixture = mixture_match.group(1).lower() == "true" if mixture_match else None

    code_match = re.search(r'(?:^|\n)code:\n(.*)', block, flags=re.DOTALL)
    code = code_match.group(1).strip() if code_match else None

    return {
        "distribution_family": distribution_family,
        "is_mixture": is_mixture,
        "code": code,
    }

def parse_row(text):
    if not isinstance(text, str):
        return None, None, None
    result = extract_pymc_model_pyvision(text)
    if result is None:
        return None, None, None
    return result["code"], result["is_mixture"], result["distribution_family"]

def is_mixture(model_structure):
    """Return True if model_structure contains multiple families (i.e. is a mixture)."""
    if isinstance(model_structure, (list, tuple)):
        return len(model_structure) > 1
    if isinstance(model_structure, str):
        # Handle stringified lists like "['gaussian', 'student_t']"
        try:
            parsed = ast.literal_eval(model_structure)
            return isinstance(parsed, (list, tuple)) and len(parsed) > 1
        except (ValueError, SyntaxError):
            pass
    return False


def main():

    parser = argparse.ArgumentParser(description="Distribution Fitting with VLM Assistance")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--distribution_type", type=str, required=True, help="Select from 'imf', 'mixed' or 'single'.")
    parser.add_argument("--df_split", type=str, help="'run_log' for only the best proposed model or 'run_log' for all steps.")
    parser.add_argument("--base_path", type=str, required=True, help="Base directory where each individual")
    args = parser.parse_args()

    if '/' in args.base_path:
        parquet_files = glob.glob(os.path.join(args.base_path, f"*/{args.df_split}.parquet"))
        dfs = []
        for path in parquet_files:
            run_dir = os.path.basename(os.path.dirname(path))
            df_sample = pd.read_parquet(path)
    
            # Take ONLY the last row
            last_row = df_sample.iloc[[-1]].copy()  # double brackets keep it as a DataFrame
            last_row["run_dir"] = run_dir
            last_row["best_model_code"] = last_row["run_best_model_code"].iloc[0]
            last_row["final_family"] = last_row["run_best_model_structure"].iloc[0]
            last_row["is_mixture"] = is_mixture(last_row["run_best_model_structure"].iloc[0])
            last_row["dataset_idx"] = last_row["dataset_idx"].iloc[0]
    
            dfs.append(last_row)
    
        df = pd.concat(dfs, ignore_index=True)

        print("DF SHAPE: ", len(df))

        # print("WARNING EVALUATING ON 2 SAMPLES")
        # df = df.head(2)
        
        
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                print(f"Failed to parse final_family value: {repr(x)}")
                return x  # keep raw value so you can inspect/handle downstream

        
        df['final_family'] = df['final_family'].apply(safe_literal_eval)

    elif args.base_path == 'pyvision':
        print("RUNNING PYVISION", args.base_path)
        df = pd.read_csv(f'pyvision_{args.distribution_type}_results.csv')
        df[["best_model_code", "is_mixture", "final_family"]] = df["final_response"].apply(lambda x: pd.Series(parse_row(x)))
        df['dataset_idx'] = df['idx']

    elif args.base_path == 'boxing_llm': 
        df = pd.read_csv('boxing_llm.csv')
        # CHANGE TO BEST_MODEL_CODE LATER
        df['best_model_code'] = df['best_code'].apply(sanitize_box_llm_model_code)
        df[["is_mixture", "final_family"]] = df["best_model_code"].apply(
            lambda code: pd.Series(extract_model_properties(code))
        )
        # CAN DROP LATER
        df['dataset_idx'] = df['array_id']

    #df = df.head(1)

    df['pymc_model_code'] = df['best_model_code'].apply(extract_model_code)
    # Guard against NaN before sanitize (pydantic can't handle it)
    df['pymc_model_code'] = df['pymc_model_code'].apply(
        lambda x: sanitize_pymc_code(x) if pd.notna(x) else None
    )

    df = fix_pymc_model_code_column(df)
    
    print("MODEL CODE", df['pymc_model_code'])

    model = args.model_name
    distribution_type = args.distribution_type

    with open(f'data_{distribution_type}.pkl', 'rb') as f:
        data = pickle.load(f)

    print("DATA LOADED")

    results_list = []

    for idx in list(range(len(df))):
        print("CURRENT INDEX: ", idx)

        data_idx = df.iloc[idx]['dataset_idx']

        row_result = {
            "idx": df.iloc[idx]['dataset_idx'],
            "status": "ok",
            "error_stage": None,
            "error_message": None
        }

        try:
            # ---------------- TRUE MODEL ----------------
            try:
                true_model, true_map_estimate, true_fit_metrics = fit_true_model(
                    data[data_idx]['data'],
                    data[data_idx]['true_params']
                )
                print("TRUE MODEL FIT: ", true_model)
                # fit_true_model catches its own exceptions and returns None on failure.
                # Detect that here so we log it under the right error_stage.
                if true_model is None or true_map_estimate is None:
                    raise RuntimeError("fit_true_model returned None (internal silent failure — check logs above)")
            except Exception as e:
                row_result.update({
                    "status": "failed",
                    "error_stage": "true_model_fit",
                    "error_message": str(e)
                })
                results_list.append(row_result)
                continue

            # ---------------- PRED MODEL ----------------
            try:
                pred_model, pred_map_estimate, pred_fit_metrics = execute_and_fit_model(
                    data[data_idx]['data'],
                    df.iloc[idx]['pymc_model_code']
                )
                print("PRED MODEL FIT: ", pred_model)
                # Safety net in case execute_and_fit_model error handling is re-enabled
                # and it silently returns None.
                if pred_model is None or pred_map_estimate is None:
                    raise RuntimeError("execute_and_fit_model returned None (internal silent failure — check logs above)")
            except Exception as e:
                row_result.update({
                    "status": "failed",
                    "error_stage": "pred_model_fit",
                    "error_message": str(e)
                })
                results_list.append(row_result)
                continue

            # ---------------- FIT METRICS ----------------
            try:
                data_results = compare_fit_metrics(
                    true_fit_metrics,
                    pred_fit_metrics,
                    N=len(data[data_idx]['data']),
                    plot=False
                )
                print("COMPARE FIT", data_results)
            except Exception as e:
                row_result.update({
                    "status": "failed",
                    "error_stage": "compare_fit_metrics",
                    "error_message": str(e)
                })
                results_list.append(row_result)
                continue

            # ---------------- PDF COMPARISON ----------------
            try:
                print("TME", true_map_estimate)
                print("PME", pred_map_estimate)
                print("DATA DIST", data[data_idx]['dist_choice'])
                print("PRED DIST", df.iloc[idx]['final_family'])
                pdf_results, pdf_true, pdf_pred = compare_distributions(
                    true_map_estimate,
                    pred_map_estimate,
                    data[data_idx]['dist_choice'],
                    canonicalize_family_name(df.iloc[idx]['final_family'])
                )
            except Exception as e:
                row_result.update({
                    "status": "failed",
                    "error_stage": "compare_distributions",
                    "error_message": str(e)
                })
                results_list.append(row_result)
                continue

            # ---------------- MERGE RESULTS ----------------
            combined = {
                **row_result,
                **data_results,
                **pdf_results
            }

            print(combined)
            print("=" * 50)

            results_list.append(combined)

        except Exception as e:
            # Catastrophic failure safeguard
            row_result.update({
                "status": "failed",
                "error_stage": "unknown",
                "error_message": traceback.format_exc()
            })
            results_list.append(row_result)

    results_df = pd.DataFrame(results_list)
    if '/' in args.base_path:
        results_df.to_csv(f"EVALUATION_vae_{args.base_path.split('/')[0]}_{args.base_path.split('/')[-1]}_{args.df_split}.csv", index=False)
    else:
        results_df.to_csv(f"EVALUATION_{args.base_path}_{args.model_name}_{args.distribution_type}.csv", index=False)

if __name__ == "__main__":
    main()