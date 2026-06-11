"""
IMF Sampling Pipeline
=====================
Self-contained sampling for Salpeter, Kroupa, Chabrier, and 8-parameter
freeform IMFs via exact inverse-CDF methods (no external IMF package needed).

Dependencies: numpy, scipy, matplotlib
"""

import json
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import lognorm

# =============================================================================
# CONSTANTS & METADATA
# =============================================================================

IMF_COLORS = {
    "salpeter": "#E8621A",
    "kroupa": "#3A86FF",
    "chabrier": "#8338EC",
    "freeform_wide": "#FF006E",
    "freeform_tight": "#06D6A0",
}

IMF_LABELS = {
    "salpeter": "Salpeter",
    "kroupa": "Kroupa",
    "chabrier": "Chabrier",
    "freeform_wide": "Free-form (wide)",
    "freeform_tight": "Free-form (tight)",
}

BREAKPOINT_KEYS = {
    "salpeter": [],
    "kroupa": ["break1", "break2"],
    "chabrier": [],  # lognormal/powerlaw join fixed at 1 Msun
    "freeform_wide": ["b1", "b2", "b3"],
    "freeform_tight": ["b1", "b2", "b3"],
}

SLOPE_LABELS = {
    "salpeter": ["alpha"],
    "kroupa": ["p1", "p2", "p3"],
    "chabrier": ["alpha"],
    "freeform_wide": ["a0", "a1", "a2", "a3"],
    "freeform_tight": ["a0", "a1", "a2", "a3"],
}

# Bright yellow for mmin / mmax cutoff lines in all plots
CUTOFF_COLOR = "#FFE600"


# =============================================================================
# SHARED INVERSE-CDF HELPERS  (power-law segments)
# =============================================================================


def _bpl_norm(alpha: float, lo: float, hi: float) -> float:
    """Integral of m^-alpha dm over [lo, hi]."""
    if abs(alpha - 1.0) < 1e-10:
        return np.log(hi / lo)
    return (hi ** (1 - alpha) - lo ** (1 - alpha)) / (1 - alpha)


def _bpl_inverse_cdf(u: np.ndarray, alpha: float, lo: float, hi: float) -> np.ndarray:
    """Exact inverse CDF of m^-alpha / Z  for uniform u in [0, 1]."""
    if abs(alpha - 1.0) < 1e-10:
        return lo * (hi / lo) ** u
    exp = 1.0 - alpha
    return (lo**exp + u * (hi**exp - lo**exp)) ** (1.0 / exp)


def _sample_powerlaw_segment(rng, n: int, alpha: float, lo: float, hi: float) -> np.ndarray:
    """Draw n samples from m^-alpha on [lo, hi]."""
    return _bpl_inverse_cdf(rng.uniform(size=n), alpha, lo, hi)


# =============================================================================
# SALPETER  — single power law
# Canonical: alpha = 2.35  (Salpeter 1955)
# Free param: alpha
# mmin = 0.08 Msun  (hydrogen-burning limit)
# mmax = 120.0 Msun (revised stellar upper-mass limit; Figer 2005)
# =============================================================================


def _sample_salpeter(rng, n: int, alpha: float, mmin: float = 0.08, mmax: float = 120.0) -> np.ndarray:
    return _sample_powerlaw_segment(rng, n, alpha, mmin, mmax)


# =============================================================================
# KROUPA  — 3-segment broken power law
# Canonical: p1=0.3, p2=1.3, p3=2.3  (Kroupa 2001)
# Break masses FIXED: break1=0.08, break2=0.5  (stellar physics)
# Free params: p1, p2, p3
# mmin = 0.009 Msun  (below deuterium-burning limit ~0.012; probes BD tail)
# mmax = 120.0 Msun
# =============================================================================


def _sample_kroupa(
    rng,
    n: int,
    p1: float,
    p2: float,
    p3: float,
    break1: float = 0.08,
    break2: float = 0.5,
    mmin: float = 0.009,
    mmax: float = 120.0,
) -> np.ndarray:
    # Continuity constants
    k2 = break1 ** (p2 - p1)
    k3 = k2 * break2 ** (p3 - p2)

    # Segment weights
    w1 = _bpl_norm(p1, mmin, break1)
    w2 = k2 * _bpl_norm(p2, break1, break2)
    w3 = k3 * _bpl_norm(p3, break2, mmax)
    W = w1 + w2 + w3

    counts = rng.multinomial(n, [w1 / W, w2 / W, w3 / W])

    parts = [
        _sample_powerlaw_segment(rng, counts[0], p1, mmin, break1),
        _sample_powerlaw_segment(rng, counts[1], p2, break1, break2),
        _sample_powerlaw_segment(rng, counts[2], p3, break2, mmax),
    ]
    return np.concatenate([p for p in parts if len(p) > 0])


# =============================================================================
# CHABRIER  — lognormal below 1 Msun, power law above
# Canonical: mchar=0.22, sigma=0.57, alpha=2.3  (Chabrier 2003)
# Free params: mchar, sigma, alpha
# mmin = 0.009 Msun
# mmax = 120.0 Msun
# =============================================================================


def _sample_chabrier(
    rng,
    n: int,
    mchar: float,
    sigma: float,
    alpha: float,
    mmin: float = 0.009,
    mmax: float = 120.0,
    mjoin: float = 1.0,
) -> np.ndarray:
    """
    Lognormal ∝ (1/m) exp(-(log m - log mchar)^2 / 2σ^2)  for m < mjoin
    Power law ∝ m^-alpha                                    for m >= mjoin
    """
    # --- Numerical weight of lognormal part via integration ---
    # Use the scipy lognorm CDF (parameterised as scale=mchar, s=sigma in log)
    ln_dist = lognorm(s=sigma, scale=mchar)
    w_ln = ln_dist.cdf(mjoin) - ln_dist.cdf(mmin)  # prob in [mmin, mjoin]
    w_pl = _bpl_norm(alpha, mjoin, mmax)  # unnorm power-law integral

    # Match heights at mjoin for consistent normalisation
    # lognormal pdf at mjoin (as dN/dm):
    f_join_ln = ln_dist.pdf(mjoin)
    # power-law pdf at mjoin (unnorm): mjoin^-alpha
    f_join_pl = mjoin ** (-alpha)
    # scale power-law weight to match lognormal at junction
    scale_pl = f_join_ln / f_join_pl if f_join_pl > 0 else 1.0
    w_pl_eff = scale_pl * w_pl

    W = w_ln + w_pl_eff
    frac_ln = w_ln / W

    counts = rng.multinomial(n, [frac_ln, 1.0 - frac_ln])

    # Sample lognormal part via rejection on [mmin, mjoin]
    ln_samples = []
    needed = counts[0]
    while len(ln_samples) < needed:
        draws = ln_dist.rvs(size=needed * 3, random_state=rng.integers(2**31))
        valid = draws[(draws >= mmin) & (draws < mjoin)]
        ln_samples.append(valid)
    ln_samples = np.concatenate(ln_samples)[:needed]

    # Sample power-law part exactly
    pl_samples = _sample_powerlaw_segment(rng, counts[1], alpha, mjoin, mmax)

    return np.concatenate([ln_samples, pl_samples])


# =============================================================================
# FREEFORM  — 8-parameter broken power law  (Bastian+ 2010)
# 4 slopes (a0..a3) + 3 break masses (b1, b2, b3) + mmin = 8 free params
# mmax fixed at 120 Msun
# =============================================================================


def _sample_freeform(
    rng,
    n: int,
    a0: float,
    a1: float,
    a2: float,
    a3: float,
    mmin: float,
    b1: float,
    b2: float,
    b3: float,
    mmax: float = 120.0,
) -> np.ndarray:
    # Enforce break mass ordering
    b1, b2, b3 = sorted([b1, b2, b3])

    # Continuity constants
    k1 = b1 ** (a1 - a0)
    k2 = k1 * b2 ** (a2 - a1)
    k3 = k2 * b3 ** (a3 - a2)

    # Segment weights
    w0 = _bpl_norm(a0, mmin, b1)
    w1 = k1 * _bpl_norm(a1, b1, b2)
    w2 = k2 * _bpl_norm(a2, b2, b3)
    w3 = k3 * _bpl_norm(a3, b3, mmax)
    W = w0 + w1 + w2 + w3

    counts = rng.multinomial(n, [w0 / W, w1 / W, w2 / W, w3 / W])

    segs = [
        (counts[0], a0, mmin, b1),
        (counts[1], a1, b1, b2),
        (counts[2], a2, b2, b3),
        (counts[3], a3, b3, mmax),
    ]
    parts = [_sample_powerlaw_segment(rng, cnt, alpha, lo, hi) for cnt, alpha, lo, hi in segs if cnt > 0]

    samples = np.concatenate(parts)
    rng.shuffle(samples)
    return samples


# =============================================================================
# PUBLIC SAMPLERS
# =============================================================================


def sample_random_imf(imf_type: str, n: int = 5000, seed: int = None) -> tuple[np.ndarray, dict]:
    """
    Sample from a randomly-parameterised Salpeter, Kroupa, or Chabrier IMF.
    All free parameters are drawn close to their canonical values.

    Returns
    -------
    samples : np.ndarray  — stellar masses in solar masses
    params  : dict        — exact parameters used (sufficient for reconstruction)
    """
    rng = np.random.default_rng(seed)

    if imf_type == "salpeter":
        # Canonical alpha=2.35 | Range [1.9, 2.7]
        # mmin=0.08 Msun (H-burning limit), mmax=120 Msun (Figer 2005)
        alpha = float(rng.uniform(1.9, 2.7))
        params = {"alpha": round(alpha, 4), "mmin": 0.08, "mmax": 120.0}
        samples = _sample_salpeter(rng, n, **params)

    elif imf_type == "kroupa":
        # Canonical p1=0.3, p2=1.3, p3=2.3 | break masses FIXED
        # mmin=0.009 Msun (extends into BD tail), mmax=120 Msun
        p1 = float(rng.uniform(0.1, 0.5))
        p2 = float(rng.uniform(1.0, 1.6))
        p3 = float(rng.uniform(1.9, 2.7))
        params = {
            "p1": round(p1, 4),
            "p2": round(p2, 4),
            "p3": round(p3, 4),
            "break1": 0.08,
            "break2": 0.5,
            "mmin": 0.009,
            "mmax": 120.0,
        }
        samples = _sample_kroupa(rng, n, **params)

    elif imf_type == "chabrier":
        # Canonical mchar=0.22, sigma=0.57, alpha=2.3
        # mmin=0.009 Msun, mmax=120 Msun
        mchar = float(rng.uniform(0.15, 0.35))
        sigma = float(rng.uniform(0.45, 0.70))
        alpha = float(rng.uniform(1.9, 2.7))
        params = {
            "mchar": round(mchar, 4),
            "sigma": round(sigma, 4),
            "alpha": round(alpha, 4),
            "mmin": 0.009,
            "mmax": 120.0,
        }
        samples = _sample_chabrier(rng, n, **params)

    else:
        raise ValueError(f"Unknown IMF type '{imf_type}'. Choose: 'salpeter', 'kroupa', 'chabrier'")

    return samples, params


def sample_freeform_imf(n: int = 5000, seed: int = None, wide: bool = True) -> tuple[np.ndarray, dict]:
    """
    Sample from a randomly-parameterised 8-parameter broken power law IMF
    (Bastian, Covey & Meyer 2010 extended Kroupa formulation).

    Parameters
    ----------
    wide : bool  — True = exploratory ranges; False = tight around canonical
    """
    rng = np.random.default_rng(seed)
    mmax = 120.0  # revised upper stellar mass limit (Figer 2005)

    if wide:
        a0 = float(rng.uniform(-0.5, 1.0))
        a1 = float(rng.uniform(0.5, 2.0))
        a2 = float(rng.uniform(1.5, 3.0))
        a3 = float(rng.uniform(1.8, 3.5))
        # mmin range spans BD regime down to ~opacity limit (~0.007 Msun)
        mmin = float(rng.uniform(0.007, 0.040))
        b1 = float(rng.uniform(0.06, 0.12))
        b2 = float(rng.uniform(0.30, 0.80))
        b3 = float(rng.uniform(0.80, 2.00))
    else:
        a0 = float(rng.uniform(0.1, 0.5))
        a1 = float(rng.uniform(1.0, 1.6))
        a2 = float(rng.uniform(1.9, 2.7))
        a3 = float(rng.uniform(1.9, 2.7))
        # mmin range tightly around H-burning limit
        mmin = float(rng.uniform(0.007, 0.012))
        b1 = float(rng.uniform(0.07, 0.09))
        b2 = float(rng.uniform(0.40, 0.60))
        b3 = float(rng.uniform(0.80, 1.20))

    # Sort breaks so ordering is guaranteed
    b1, b2, b3 = sorted([b1, b2, b3])

    params = {
        "a0": round(a0, 4),
        "a1": round(a1, 4),
        "a2": round(a2, 4),
        "a3": round(a3, 4),
        "mmin": round(mmin, 5),
        "b1": round(b1, 4),
        "b2": round(b2, 4),
        "b3": round(b3, 4),
        "mmax": mmax,
    }
    samples = _sample_freeform(rng, n, **params)
    return samples, params


# =============================================================================
# PLOTTING
# =============================================================================


def _build_param_string(imf_type: str, params: dict) -> str:
    """Compact parameter annotation for plot."""
    lines = []
    for k in SLOPE_LABELS[imf_type]:
        if k in params:
            lines.append(f"{k} = {params[k]:.3f}")
    if imf_type == "chabrier":
        lines += [f"μ = {params['mchar']:.3f}", f"σ = {params['sigma']:.3f}"]
    return "\n".join(lines)


def _draw_cutoff_lines(ax, mmin: float, mmax: float, y_min: float, y_max: float) -> None:
    """
    Draw dashed yellow vertical lines at the exact mmin and mmax cutoffs,
    with small labels. Shared by both plot_imf_result and plot_survey_overview.
    """
    y_label = y_min * 3.5  # position label just above the floor

    for cutoff, clabel, ha, xoff_fac in [
        (mmin, f"mmin={mmin:.4g} M☉", "right", 0.90),
        (mmax, f"mmax={mmax:.4g} M☉", "left", 1.10),
    ]:
        ax.axvline(cutoff, color=CUTOFF_COLOR, lw=1.5, ls="--", alpha=0.90, zorder=3)
        ax.text(
            cutoff * xoff_fac,
            y_label,
            clabel,
            color=CUTOFF_COLOR,
            fontsize=6.5,
            alpha=0.95,
            va="bottom",
            ha=ha,
            fontfamily="monospace",
            zorder=4,
        )


def plot_imf_result(result: dict, ax: plt.Axes = None, bins: int = 60, show: bool = True) -> plt.Figure:
    """
    Log-log histogram of one IMF result with breakpoint annotations
    and dashed yellow mmin/mmax cutoff lines.

    Parameters
    ----------
    result : dict   — one entry from generate_imf_survey()
    ax     : Axes   — optional existing axes
    bins   : int    — number of log-spaced bins
    show   : bool   — call plt.show()
    """
    imf_type = result["imf_type"]
    params = result["params"]
    samples = result["samples"]
    color = IMF_COLORS[imf_type]
    label = IMF_LABELS[imf_type]
    mmin = params.get("mmin", 0.009)
    mmax = params.get("mmax", 120.0)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#0F1117")
    else:
        fig = ax.get_figure()

    # --- Log-log histogram (dN/dm density) ---
    log_bins = np.logspace(np.log10(samples.min()), np.log10(samples.max()), bins)
    counts, edges = np.histogram(samples, bins=log_bins)
    widths = np.diff(edges)
    bin_centres = np.sqrt(edges[:-1] * edges[1:])  # geometric midpoints
    dN_dm = counts / widths
    mask = dN_dm > 0

    ax.fill_between(bin_centres[mask], dN_dm[mask], alpha=0.15, color=color)
    ax.plot(bin_centres[mask], dN_dm[mask], color=color, lw=2, label=label)

    y_max = dN_dm[mask].max()
    y_min = dN_dm[mask][dN_dm[mask] > 0].min()

    # --- Dashed yellow mmin / mmax cutoff lines ---
    _draw_cutoff_lines(ax, mmin, mmax, y_min, y_max)

    # --- Breakpoint vertical lines ---
    for bp_key in BREAKPOINT_KEYS[imf_type]:
        bp_val = params[bp_key]
        ax.axvline(bp_val, color=color, lw=1.2, ls="--", alpha=0.7)
        ax.text(
            bp_val * 1.08,
            np.exp(0.5 * (np.log(y_max) + np.log(y_min))),
            f"{bp_key} = {bp_val:.3f} M☉",
            color=color,
            fontsize=7.5,
            alpha=0.9,
            va="center",
            rotation=90,
            fontfamily="monospace",
        )

    # Chabrier: mark the lognormal / power-law junction
    if imf_type == "chabrier":
        ax.axvline(1.0, color=color, lw=1.2, ls=":", alpha=0.6)
        ax.text(
            1.0 * 1.08,
            np.exp(0.5 * (np.log(y_max) + np.log(y_min))),
            "lognormal | PL\n1.0 M☉",
            color=color,
            fontsize=7.5,
            alpha=0.8,
            va="center",
            rotation=90,
            fontfamily="monospace",
        )

    # --- Parameter box (top-right) ---
    param_str = _build_param_string(imf_type, params)
    ax.text(
        0.97,
        0.97,
        param_str,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        fontfamily="monospace",
        color="white",
        alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E2130", edgecolor=color, linewidth=0.8, alpha=0.9),
    )

    # --- Metadata box (top-left) ---
    meta = f"Type: {label}\nRealization: {result['realization']}\nSeed: {result['seed']}"
    ax.text(
        0.03,
        0.97,
        meta,
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        fontfamily="monospace",
        color="#9BA3BF",
        alpha=0.9,
    )

    # --- Axes formatting ---
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Stellar Mass  $m\;[M_\odot]$", color="#C8CCDB", fontsize=11)
    ax.set_ylabel(r"$dN/dm$", color="#C8CCDB", fontsize=11)
    ax.set_title(f"{label} IMF  —  log-log", color="white", fontsize=13, pad=10, fontweight="bold")
    ax.tick_params(colors="#9BA3BF", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2E3347")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.grid(True, which="major", color="#2E3347", lw=0.8)
    ax.grid(True, which="minor", color="#1E2130", lw=0.4)

    if standalone:
        fig.tight_layout()
        if show:
            plt.show()

    return fig


def _analytical_dN_dlogm(imf_type: str, params: dict, m_grid: np.ndarray) -> np.ndarray:
    """
    Evaluate the analytical dN/d(log m) ∝ m · xi(m) on a mass grid.
    Returns zeros outside [mmin, mmax]. Used for clean curve rendering.
    """
    mmin = params.get("mmin", 0.009)
    mmax = params.get("mmax", 120.0)
    out = np.zeros_like(m_grid, dtype=float)
    mask = (m_grid >= mmin) & (m_grid <= mmax)
    m = m_grid[mask]

    if imf_type == "salpeter":
        # xi(m) ∝ m^-alpha  =>  dN/dlogm ∝ m^(1-alpha)
        vals = m ** (1.0 - params["alpha"])

    elif imf_type == "kroupa":
        p1, p2, p3 = params["p1"], params["p2"], params["p3"]
        b1, b2 = params["break1"], params["break2"]
        k2 = b1 ** (p2 - p1)
        k3 = k2 * b2 ** (p3 - p2)
        vals = np.where(m < b1, m ** (1.0 - p1), np.where(m < b2, k2 * m ** (1.0 - p2), k3 * m ** (1.0 - p3)))

    elif imf_type == "chabrier":
        from scipy.stats import lognorm as _lognorm

        mchar, sigma, alpha = params["mchar"], params["sigma"], params["alpha"]
        mjoin = 1.0
        ln_dist = _lognorm(s=sigma, scale=mchar)
        f_join_ln = ln_dist.pdf(mjoin)
        f_join_pl = mjoin ** (-alpha)
        scale_pl = f_join_ln / f_join_pl if f_join_pl > 0 else 1.0
        vals = np.where(
            m < mjoin,
            ln_dist.pdf(m),  # lognormal part
            scale_pl * m ** (-alpha),
        )  # power-law part
        vals *= m  # convert pdf → dN/dlogm  (pdf is already dN/dm for lognorm)

    elif imf_type in ("freeform_wide", "freeform_tight"):
        a0, a1, a2, a3 = params["a0"], params["a1"], params["a2"], params["a3"]
        b1, b2, b3 = params["b1"], params["b2"], params["b3"]
        k1 = b1 ** (a1 - a0)
        k2 = k1 * b2 ** (a2 - a1)
        k3 = k2 * b3 ** (a3 - a2)
        vals = np.where(
            m < b1,
            m ** (1.0 - a0),
            np.where(
                m < b2, k1 * m ** (1.0 - a1), np.where(m < b3, k2 * m ** (1.0 - a2), k3 * m ** (1.0 - a3))
            ),
        )
    else:
        vals = np.ones_like(m)

    # Normalise to peak = 1 for plotting (log scale — shape is what matters)
    if imf_type != "chabrier":  # chabrier already multiplied by m above
        pass
    v_max = vals.max()
    if v_max > 0:
        vals /= v_max

    out[mask] = vals
    return out


def plot_survey_overview(results: list, bins: int = 60, show: bool = True) -> plt.Figure:
    """
    Grid of all survey results on a log-log scale using dN/d(log m).
    Uses histogram bins for a realistic noisy appearance, with dashed
    yellow mmin/mmax cutoff lines.
    """
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    fig.patch.set_facecolor("#0F1117")
    axes_flat = np.array(axes).flatten()

    for i, result in enumerate(results):
        ax = axes_flat[i]
        ax.set_facecolor("#0F1117")

        imf_type = result["imf_type"]
        params = result["params"]
        samples = result["samples"]
        color = IMF_COLORS[imf_type]
        label = IMF_LABELS[imf_type]
        mmin = params.get("mmin", 0.009)
        mmax = params.get("mmax", 120.0)

        # --- dN/d(log m) via histogram (keeps realistic noise) ---
        log_bins = np.logspace(np.log10(mmin), np.log10(mmax), bins)
        counts, edges = np.histogram(samples, bins=log_bins)
        bin_centres = np.sqrt(edges[:-1] * edges[1:])  # geometric midpoints
        dlog_widths = np.diff(np.log10(edges))  # widths in log10 space
        dN_dlogm = counts / dlog_widths
        mask = dN_dlogm > 0

        ax.fill_between(bin_centres[mask], dN_dlogm[mask], alpha=0.15, color=color)
        ax.plot(bin_centres[mask], dN_dlogm[mask], color=color, lw=1.8)

        y_max = dN_dlogm[mask].max()
        y_min = dN_dlogm[mask][dN_dlogm[mask] > 0].min()
        y_mid = np.exp(0.5 * (np.log(y_max) + np.log(y_min)))

        # --- Dashed yellow mmin / mmax cutoff lines ---
        _draw_cutoff_lines(ax, mmin, mmax, y_min, y_max)

        # --- Internal breakpoint lines (IMF-family colour) ---
        for bp_key in BREAKPOINT_KEYS[imf_type]:
            bp_val = params[bp_key]
            ax.axvline(bp_val, color=color, lw=1.1, ls="--", alpha=0.6)
            ax.text(
                bp_val * 1.08,
                y_mid,
                f"{bp_key}={bp_val:.3f}",
                color=color,
                fontsize=6.5,
                alpha=0.85,
                va="center",
                rotation=90,
                fontfamily="monospace",
            )

        if imf_type == "chabrier":
            ax.axvline(1.0, color=color, lw=1.1, ls=":", alpha=0.55)
            ax.text(
                1.0 * 1.08,
                y_mid,
                "PL join",
                color=color,
                fontsize=6.5,
                alpha=0.75,
                va="center",
                rotation=90,
                fontfamily="monospace",
            )

        # --- Parameter box (top-right) ---
        param_str = _build_param_string(imf_type, params)
        ax.text(
            0.97,
            0.97,
            param_str,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="right",
            fontfamily="monospace",
            color="white",
            alpha=0.9,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="#1E2130", edgecolor=color, linewidth=0.8, alpha=0.9
            ),
        )

        # --- Meta box (top-left) ---
        meta = f"{label}\nr={result['realization']}  seed={result['seed']}"
        ax.text(
            0.03,
            0.97,
            meta,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            fontfamily="monospace",
            color="#9BA3BF",
            alpha=0.9,
        )

        # --- Axes ---
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(mmin * 0.3, mmax * 3)
        ax.set_ylim(y_min * 0.5, y_max * 5)
        ax.set_xlabel(r"$m\;[M_\odot]$", color="#C8CCDB", fontsize=10)
        ax.set_ylabel(r"$dN/d\log m$", color="#C8CCDB", fontsize=10)
        ax.set_title(f"{label}", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#9BA3BF", which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E3347")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.grid(True, which="major", color="#2E3347", lw=0.8)
        ax.grid(True, which="minor", color="#1E2130", lw=0.4)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_handles = [Line2D([0], [0], color=c, lw=2, label=IMF_LABELS[t]) for t, c in IMF_COLORS.items()]
    # Add cutoff line to legend
    legend_handles.append(Line2D([0], [0], color=CUTOFF_COLOR, lw=1.5, ls="--", label="mmin / mmax cutoff"))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=9,
        facecolor="#1E2130",
        edgecolor="#2E3347",
        labelcolor="white",
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        r"IMF Survey  —  $dN/d\log m$ vs $m$  (log-log)",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if show:
        plt.show()
    return fig


# =============================================================================
# SURVEY GENERATOR
# =============================================================================


def generate_imf_survey(
    imf_types: list,
    n_samples: int = 5000,
    n_realizations: int = 1,
    seed: int = None,
    plot_individual: bool = True,
    plot_overview: bool = True,
    save_plots: bool = False,
) -> list[dict]:
    """
    Generate random IMF samples across multiple types and realizations.

    Parameters
    ----------
    imf_types      : list of str — any of 'salpeter', 'kroupa', 'chabrier',
                                   'freeform_wide', 'freeform_tight'
    n_samples      : int  — stellar masses drawn per realization
    n_realizations : int  — independent random draws per IMF type
    seed           : int  — master seed (all child seeds derived from this)
    plot_individual: bool — plot each realization as it is generated
    plot_overview  : bool — plot a summary grid at the end
    save_plots     : bool — save PNGs to disk instead of displaying

    Returns
    -------
    results : list of dict, each entry contains:
        'imf_type'    : str
        'realization' : int
        'seed'        : int       ← exact child seed; enough to reconstruct
        'params'      : dict      ← all parameters used
        'samples'     : np.ndarray
    """
    valid = {"salpeter", "kroupa", "chabrier", "freeform_wide", "freeform_tight"}
    bad = set(imf_types) - valid
    if bad:
        raise ValueError(f"Unknown IMF type(s): {bad}. Choose from {valid}")

    master_rng = np.random.default_rng(seed)
    results = []

    for imf_type in imf_types:
        for i in range(n_realizations):
            child_seed = int(master_rng.integers(0, 2**31))

            if imf_type in {"salpeter", "kroupa", "chabrier"}:
                samples, params = sample_random_imf(imf_type, n_samples, child_seed)
            elif imf_type == "freeform_wide":
                samples, params = sample_freeform_imf(n_samples, child_seed, wide=True)
            elif imf_type == "freeform_tight":
                samples, params = sample_freeform_imf(n_samples, child_seed, wide=False)

            result = {
                "imf_type": imf_type,
                "realization": i,
                "seed": child_seed,
                "params": params,
                "samples": samples,
            }
            results.append(result)
            print(f"  [{imf_type}] realization {i}  seed={child_seed}  params={params}")

            if plot_individual:
                fig = plot_imf_result(result, show=not save_plots)
                if save_plots:
                    fname = f"imf_{imf_type}_r{i}_seed{child_seed}.png"
                    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="#0F1117")
                    plt.close(fig)
                    print(f"    → saved {fname}")

    if plot_overview:
        fig_ov = plot_survey_overview(results, show=not save_plots)
        if save_plots:
            fig_ov.savefig("imf_survey_overview.png", dpi=150, bbox_inches="tight", facecolor="#0F1117")
            plt.close(fig_ov)
            print("  → saved imf_survey_overview.png")

    return results


# =============================================================================
# RECONSTRUCTION  —  rebuild identical samples from saved seed alone
# =============================================================================


def reconstruct_from_result(result: dict) -> np.ndarray:
    """
    Exactly reconstruct samples from a saved result dict using only its seed.
    Guaranteed to return the identical array as the original draw.
    """
    imf_type = result["imf_type"]
    seed = result["seed"]
    n = len(result["samples"])

    if imf_type in {"salpeter", "kroupa", "chabrier"}:
        samples, _ = sample_random_imf(imf_type, n, seed)
    elif imf_type == "freeform_wide":
        samples, _ = sample_freeform_imf(n, seed, wide=True)
    elif imf_type == "freeform_tight":
        samples, _ = sample_freeform_imf(n, seed, wide=False)

    return samples


# =============================================================================
# SAVE / LOAD  — .npz round-trip
# =============================================================================


def save_survey(results: list, path: str = "imf_survey.npz") -> None:
    """Save all samples + params to a single .npz file."""
    save_dict = {}
    metadata = []
    for idx, r in enumerate(results):
        save_dict[f"samples_{idx}"] = r["samples"]
        metadata.append(
            {
                "idx": idx,
                "imf_type": r["imf_type"],
                "realization": r["realization"],
                "seed": r["seed"],
                "params": r["params"],
            }
        )
    save_dict["metadata"] = np.array(json.dumps(metadata))
    np.savez(path, **save_dict)
    print(f"Saved {len(results)} realizations → {path}")


def load_survey(path: str = "imf_survey.npz") -> list[dict]:
    """Reload a saved survey back into the same list-of-dicts format."""
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))
    return [
        {
            "imf_type": m["imf_type"],
            "realization": m["realization"],
            "seed": m["seed"],
            "params": m["params"],
            "samples": data[f"samples_{m['idx']}"],
        }
        for m in metadata
    ]


# =============================================================================
# PDF-PARAM BUILDER  —  converts raw sampler params → segment dicts
# =============================================================================


def _build_pdf_params(imf_type: str, params: dict, samples: np.ndarray) -> dict:
    """
    Convert the raw parameter dict produced by the samplers into a structured
    'true_params' block that fully specifies the PDF for each segment.

    Every segment carries enough information to evaluate its unnormalised PDF
    p_seg(m) directly:

      powerlaw  →  continuity_k * m^{-alpha}  on [mmin_seg, mmax_seg]
      lognormal →  lognorm.pdf(m, s=sigma, scale=mchar)  on [mmin_seg, mmax_seg]

    The global PDF is:
        p(m) = sum_i  weight_i * p_seg_i(m) / Z_i
    where Z_i is the normalising integral of p_seg_i over its segment range and
    weight_i is the fraction of total stars that fall in segment i.
    """
    n_total = len(samples)
    mmin = params.get("mmin", 0.009)
    mmax = params.get("mmax", 120.0)

    # ------------------------------------------------------------------
    # SALPETER — single power-law
    # ------------------------------------------------------------------
    if imf_type == "salpeter":
        alpha = params["alpha"]
        Z = _bpl_norm(alpha, mmin, mmax)
        seg = {
            "dist_choice": "powerlaw",
            "alpha": alpha,
            "mmin_seg": mmin,
            "mmax_seg": mmax,
            "continuity_k": 1.0,  # single segment — no rescaling needed
            "norm_integral": Z,  # ∫ m^{-alpha} dm over segment
            "weight": 1.0,
            "n": n_total,
            # PDF hint: p(m) = m^{-alpha} / Z  on [mmin, mmax]
        }
        return {
            "imf_type": imf_type,
            "mmin": mmin,
            "mmax": mmax,
            "num_components": 1,
            "total_n": n_total,
            "weights": [1.0],
            "components": [seg],
        }

    # ------------------------------------------------------------------
    # KROUPA — 3 power-law segments with continuity constants
    # ------------------------------------------------------------------
    elif imf_type == "kroupa":
        p1, p2, p3 = params["p1"], params["p2"], params["p3"]
        b1, b2 = params["break1"], params["break2"]

        k2 = b1 ** (p2 - p1)  # continuity at break1
        k3 = k2 * b2 ** (p3 - p2)  # continuity at break2

        raw_w1 = _bpl_norm(p1, mmin, b1)
        raw_w2 = k2 * _bpl_norm(p2, b1, b2)
        raw_w3 = k3 * _bpl_norm(p3, b2, mmax)
        W = raw_w1 + raw_w2 + raw_w3

        w1, w2, w3 = raw_w1 / W, raw_w2 / W, raw_w3 / W
        n1 = round(n_total * w1)
        n2 = round(n_total * w2)
        n3 = n_total - n1 - n2

        segs = [
            {
                "dist_choice": "powerlaw",
                "alpha": p1,
                "mmin_seg": mmin,
                "mmax_seg": b1,
                "continuity_k": 1.0,
                "norm_integral": _bpl_norm(p1, mmin, b1),
                "weight": round(w1, 6),
                "n": n1,
            },
            {
                "dist_choice": "powerlaw",
                "alpha": p2,
                "mmin_seg": b1,
                "mmax_seg": b2,
                "continuity_k": k2,
                "norm_integral": _bpl_norm(p2, b1, b2),
                "weight": round(w2, 6),
                "n": n2,
            },
            {
                "dist_choice": "powerlaw",
                "alpha": p3,
                "mmin_seg": b2,
                "mmax_seg": mmax,
                "continuity_k": k3,
                "norm_integral": _bpl_norm(p3, b2, mmax),
                "weight": round(w3, 6),
                "n": n3,
            },
        ]
        return {
            "imf_type": imf_type,
            "mmin": mmin,
            "mmax": mmax,
            "break1": b1,
            "break2": b2,
            "num_components": 3,
            "total_n": n_total,
            "weights": [round(w1, 6), round(w2, 6), round(w3, 6)],
            "components": segs,
        }

    # ------------------------------------------------------------------
    # CHABRIER — lognormal below mjoin=1 M☉, power-law above
    # ------------------------------------------------------------------
    elif imf_type == "chabrier":
        mchar, sigma, alpha = params["mchar"], params["sigma"], params["alpha"]
        mjoin = 1.0

        ln_dist = lognorm(s=sigma, scale=mchar)
        w_ln = ln_dist.cdf(mjoin) - ln_dist.cdf(mmin)
        w_pl_raw = _bpl_norm(alpha, mjoin, mmax)

        f_join_ln = ln_dist.pdf(mjoin)
        f_join_pl = mjoin ** (-alpha)
        scale_pl = f_join_ln / f_join_pl if f_join_pl > 0 else 1.0
        w_pl_eff = scale_pl * w_pl_raw

        W = w_ln + w_pl_eff
        frac_ln = w_ln / W
        frac_pl = w_pl_eff / W

        n_ln = round(n_total * frac_ln)
        n_pl = n_total - n_ln

        segs = [
            {
                "dist_choice": "lognormal",
                "mchar": mchar,
                "sigma": sigma,
                "mmin_seg": mmin,
                "mmax_seg": mjoin,
                "cdf_lo": float(ln_dist.cdf(mmin)),
                "cdf_hi": float(ln_dist.cdf(mjoin)),
                "norm_integral": float(w_ln),
                "weight": round(frac_ln, 6),
                "n": n_ln,
            },
            {
                "dist_choice": "powerlaw",
                "alpha": alpha,
                "mmin_seg": mjoin,
                "mmax_seg": mmax,
                "continuity_k": scale_pl,
                "norm_integral": float(w_pl_raw),
                "weight": round(frac_pl, 6),
                "n": n_pl,
            },
        ]
        return {
            "imf_type": imf_type,
            "mmin": mmin,
            "mmax": mmax,
            "mjoin": mjoin,
            "num_components": 2,
            "total_n": n_total,
            "weights": [round(frac_ln, 6), round(frac_pl, 6)],
            "components": segs,
        }

    # ------------------------------------------------------------------
    # FREEFORM (wide / tight) — 4 power-law segments
    # ------------------------------------------------------------------
    elif imf_type in ("freeform_wide", "freeform_tight"):
        a0, a1, a2, a3 = params["a0"], params["a1"], params["a2"], params["a3"]
        b1, b2, b3 = params["b1"], params["b2"], params["b3"]

        k1 = b1 ** (a1 - a0)
        k2 = k1 * b2 ** (a2 - a1)
        k3 = k2 * b3 ** (a3 - a2)

        raw_w0 = _bpl_norm(a0, mmin, b1)
        raw_w1 = k1 * _bpl_norm(a1, b1, b2)
        raw_w2 = k2 * _bpl_norm(a2, b2, b3)
        raw_w3 = k3 * _bpl_norm(a3, b3, mmax)
        W = raw_w0 + raw_w1 + raw_w2 + raw_w3

        ws = [raw_w0 / W, raw_w1 / W, raw_w2 / W, raw_w3 / W]
        ns = [round(n_total * w) for w in ws]
        ns[-1] = n_total - sum(ns[:-1])  # absorb rounding residual

        seg_specs = [
            (a0, 1.0, mmin, b1),
            (a1, k1, b1, b2),
            (a2, k2, b2, b3),
            (a3, k3, b3, mmax),
        ]
        segs = [
            {
                "dist_choice": "powerlaw",
                "alpha": alpha_s,
                "mmin_seg": lo,
                "mmax_seg": hi,
                "continuity_k": k,
                "norm_integral": _bpl_norm(alpha_s, lo, hi),
                "weight": round(ws[j], 6),
                "n": ns[j],
            }
            for j, (alpha_s, k, lo, hi) in enumerate(seg_specs)
        ]
        return {
            "imf_type": imf_type,
            "mmin": mmin,
            "mmax": mmax,
            "breaks": [b1, b2, b3],
            "num_components": 4,
            "total_n": n_total,
            "weights": [round(w, 6) for w in ws],
            "components": segs,
        }

    else:
        raise ValueError(f"Unknown imf_type: {imf_type}")


# =============================================================================
# SAVE AS PKL  —  list-of-dicts with PDF-ready true_params
# =============================================================================


def save_survey_pkl(results: list, path: str = "data_imf.pkl") -> None:
    """
    Save the survey results as a list of dicts to a pickle file.

    Each dict has the structure::

        {
            'idx'        : int,           # global index in the list
            'combo_idx'  : int,           # index within this IMF type's realizations
            'dist_choice': [str],         # e.g. ['salpeter'], ['kroupa'], …
            'data'       : np.ndarray,    # sampled stellar masses [M☉]
            'true_params': {              # fully specifies the PDF
                'imf_type'      : str,
                'mmin'          : float,
                'mmax'          : float,
                'num_components': int,
                'total_n'       : int,
                'weights'       : [float, …],   # fraction of stars per component
                'components'    : [             # one dict per piecewise piece
                    {
                      'dist_choice'   : 'powerlaw' | 'lognormal',
                      # powerlaw keys:
                      'alpha'         : float,   # slope  (xi ∝ m^{-alpha})
                      'continuity_k'  : float,   # rescaling constant for continuity
                      'norm_integral' : float,   # ∫ m^{-alpha} dm over segment
                      # lognormal keys (Chabrier only):
                      'mchar'         : float,   # characteristic (median) mass
                      'sigma'         : float,   # log-space width
                      'cdf_lo'        : float,   # lognorm CDF at mmin_seg
                      'cdf_hi'        : float,   # lognorm CDF at mmax_seg
                      # shared:
                      'mmin_seg'      : float,
                      'mmax_seg'      : float,
                      'weight'        : float,   # fraction of total stars
                      'n'             : int,     # expected star count
                    },
                    …
                ],
            },
            'n_points'   : int,
        }

    PDF evaluation recipe
    ---------------------
    For a powerlaw component:
        p_seg(m) = m^{-alpha} / norm_integral

    For the lognormal component (Chabrier):
        p_seg(m) = lognorm.pdf(m, s=sigma, scale=mchar)
                   / (cdf_hi - cdf_lo)

    Global PDF:
        p(m) = sum_i  weight_i * p_seg_i(m)
    """
    records = []
    type_counter: dict[str, int] = {}

    for global_idx, r in enumerate(results):
        imf_type = r["imf_type"]
        combo_idx = type_counter.get(imf_type, 0)
        type_counter[imf_type] = combo_idx + 1

        true_params = _build_pdf_params(imf_type, r["params"], r["samples"])

        records.append(
            {
                "idx": global_idx,
                "combo_idx": combo_idx,
                "dist_choice": [imf_type],  # list, matching the example schema
                "data": r["samples"],
                "true_params": true_params,
                "n_points": len(r["samples"]),
            }
        )

    with open(path, "wb") as fh:
        pickle.dump(records, fh)

    print(f"Saved {len(records)} records → {path}")
    for rec in records[:3]:
        comp_summary = [(c["dist_choice"], c["weight"]) for c in rec["true_params"]["components"]]
        print(
            f"  idx={rec['idx']}  dist_choice={rec['dist_choice']}  "
            f"n={rec['n_points']}  components={comp_summary}"
        )
    if len(records) > 3:
        print(f"  … ({len(records) - 3} more records)")


def load_survey_pkl(path: str = "data_imf.pkl") -> list[dict]:
    """Reload a pickle saved by save_survey_pkl()."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    results = generate_imf_survey(
        imf_types=["salpeter", "kroupa", "chabrier", "freeform_wide", "freeform_tight"],
        n_samples=5000,
        n_realizations=2,
        seed=42,
        plot_individual=True,
        plot_overview=True,
        save_plots=True,
    )

    # Save as pkl (list-of-dicts with PDF-ready true_params)
    save_survey_pkl(results, "data_imf.pkl")
    reloaded_pkl = load_survey_pkl("data_imf.pkl")
    assert np.allclose(reloaded_pkl[0]["data"], results[0]["samples"])
    print("PKL round-trip check passed.")

    # Verify reconstruction from seed alone
    for r in results:
        rebuilt = reconstruct_from_result(r)
        assert np.allclose(rebuilt, r["samples"]), (
            f"Reconstruction failed for {r['imf_type']} r{r['realization']}"
        )
    print("All reconstruction checks passed.")
