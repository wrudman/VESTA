"""Time series astro-domain prompts — extends the base time-series prompts
with handling for chirp-like signals that require a WARPED PERIODIC kernel.

The base domain (``prompts.py``) assumes stationary periodicity.  For
gravitational-wave chirps and similar astro signals, the instantaneous
frequency sweeps monotonically across the window, so a stationary
Periodic kernel cannot fit them.  The fix is to apply a quadratic
input-warp ``x_warp = warp_a * x + warp_b * x^2`` to the normalized time
axis and then use a standard Periodic kernel on the warped inputs
(period fixed to 1 to break the (a, b, period) scale redundancy).

This module mirrors the structure of ``prompts.py`` -- each piece (shared
rules, proposal prompt, code-gen prompt, feedback template, summary
prompts, plot-type descriptions, and the ``DomainPrompts`` class) is
re-derived so the astro variant can be registered alongside the base
variant without touching it.
"""

import textwrap
from typing import Any, ClassVar, Dict, List, Tuple, Type

from morphic import Typed
from pydantic import field_validator

from domains import DomainPrompts
from domains.time_series import DOMAIN_ALIASES
from domains.time_series.plotting import TimeSeriesFitState

# ---------------------------------------------------------------------------
#  Domain-specific CodeGenResponse
# ---------------------------------------------------------------------------


class TimeSeriesCodeGenResponse(Typed):
    """Response from a code-generation LLM call for time-series GP fitting.

    Pydantic validates at construction time; if ``code`` is missing or not
    a string, construction raises ``ValueError`` which triggers SlowBurn retry.
    """

    code: str

# ---------------------------------------------------------------------------
#  Shared rules — astro variant
# ---------------------------------------------------------------------------


def _ts_shared_rules(*, num_proposals: int) -> str:
    """Shared rules block with astro/chirp awareness."""
    max_key: int = num_proposals - 1
    return f"""\
CORE MODELING PHILOSOPHY (OCCAM'S RAZOR):
  Prefer the simplest explanation that broadly fits the signal.
  Start with Linear or Periodic kernels.
  Escalate to WarpedPeriodic if the periodicity is clearly non-stationary.
  Add complexity (RBF or Matern) ONLY if the signal visibly wanders after structure is accounted for.
  Avoid fitting isolated spikes or temporary disruptions as structure.

TV STATIC TEST (Noise vs Structure):
  White Noise: points jump randomly like TV static -> treat as observation noise.
  Correlated Structure: points form a continuous path throughout the series -> use Matern or RBF.
  NEVER use Matern and RBF together.

Stationary vs Non-Stationary Periodicity Check:
  Look at successive peak-to-peak (or zero-crossing) spacings across the window.
  Constant spacing -> stationary oscillation -> use Periodic or PeriodicComplex.
  Monotonically SHRINKING or GROWING spacing -> the instantaneous frequency is sweeping -> use WarpedPeriodic.
  Visual tell-tales: cycles compress toward one edge of the window, or the signal looks like a "whistle"/"ringdown" waveform.
  Require at least 3-4 visible cycles before proposing WarpedPeriodic; with fewer cycles the evidence is too weak and RBF/Matern may explain it better.

KERNEL INTERPRETATION GUIDE:
  Linear -> persistent upward or downward drift across the window.
  Periodic -> smooth repeating sinusoidal patterns with CONSTANT cycle spacing.  Only use if 2-3+ cycles are clearly visible with uniform spacing.
  PeriodicComplex -> square waves, sawtooths, ECG-like spikes (sum of 3 periodic kernels at harmonic periods).  Mutually exclusive with Periodic.
  WarpedPeriodic -> sinusoidal pattern whose cycle spacing CHANGES monotonically across the window.  Mutually exclusive with Periodic and PeriodicComplex.
  RBF -> extremely smooth "lazy" curves (logistic or U-shaped).  Avoid with Matern.
  Matern (Matern52) -> ONLY for rough jagged paths (stock prices, ARMA, random walk).  Avoid if RBF or any Periodic variant is selected.
  Do not repeat kernels.  If multiple stationary periodicities exist, use periodic_complex.

OVERFITTING PREVENTION GUARDRAILS:
  Propose RBF / Matern ONLY if the pattern is NOT captured by Linear, Periodic, or WarpedPeriodic.
  Do NOT combine WarpedPeriodic with Periodic or PeriodicComplex in the same proposal.
  WarpedPeriodic + Linear is a valid combination when a non-stationary periodic wave rides on a linear drift.

KERNEL IMPLEMENTATION GUIDANCE (PyMC):
  Linear: pm.gp.cov.Linear(input_dim=1, c=c)
  Periodic: pm.gp.cov.Periodic(input_dim=1, period=period, ls=ls)
  RBF: pm.gp.cov.ExpQuad(input_dim=1, ls=ls)
  Matern: pm.gp.cov.Matern52(input_dim=1, ls=ls)
  PeriodicComplex: Sum of 3 periodic kernels at harmonic periods (dominant_period, dominant_period/2, dominant_period/3).
  WarpedPeriodic: Apply quadratic warp to normalized inputs and then a Periodic kernel with period FIXED to 1:
    X_warp = warp_a * X + warp_b * pm.math.sqr(X)   # X is the normalized (0-1) time column
    cov = amp**2 * pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ls)
    gp  = pm.gp.Marginal(cov_func=cov)
    y_  = gp.marginal_likelihood('y', X=X_warp, y=y, sigma=sigma)
  Combine kernels by addition (cov = cov1 + cov2).

WARPED-PERIODIC INITIALIZATION (read this before writing any warped_periodic code):
  Find_MAP is very sensitive to where the optimizer starts for warped-periodic models.  If warp_a and warp_b start near their prior means (e.g., 0), the optimizer collapses to a trivial fit with warp_b ~= 0 and amp ~= 0.  The fix is a tiny data-driven initializer that the code itself should compute BEFORE the `with pm.Model()` block, and then feed into `pm.find_MAP(start=...)`.
  The trick: for a non-stationary periodic wave y ~ sin(2*pi*(a*x + b*x^2)) on normalized x in [0, 1], the number of zero crossings up to x is approximately 2 * (a*x + b*x^2).  Counting crossings at x=0.5 and x=1 gives two equations in two unknowns that solve trivially, so you do NOT need least-squares or linear algebra.  Just compute:
    y_centered = data.values - np.mean(data.values)
    signs      = np.sign(y_centered)
    N_tot      = int((np.diff(signs) != 0).sum())                  # total zero crossings
    N_half     = int((np.diff(signs[: len(signs) // 2]) != 0).sum())  # zero crossings in the first half
    a_init     = max(2.0 * N_half - 0.5 * N_tot, 1.0)               # clamp to stay positive
    b_init     = float(N_tot - 2.0 * N_half)                         # can be negative for decreasing-frequency chirps
  Use these as prior means inside the Priors dict (e.g., `warp_a` ~ Normal(a_init, max(0.3*abs(a_init), 1.0))).  Equally important, pass them to the optimizer via the `start=` argument of `pm.find_MAP` — a dict keyed by variable name mapping `warp_a` -> `a_init`, `warp_b` -> `b_init`, plus sensible defaults for `ls`, `amp`, and `sigma` based on the data scale (the concrete code-gen prompt shows the exact form).
  Whenever you write any code that uses warp_a and warp_b, build this init block and plumb it through `start=` to `pm.find_MAP`.  If you omit `start=`, the model will underfit.

KERNEL PARAMETER PRIORS GUIDANCE:
  Adjust priors based on visual feedback.  If amplitude is too small, increase sigma in pm.HalfNormal('amp', sigma=...).
  lengthscale (ls): pm.Gamma('ls', alpha=2, beta=1) or pm.Exponential('ls', 1).  For WarpedPeriodic prefer pm.HalfNormal('ls', sigma=1.0).
  period: pm.Gamma('period', alpha=2, beta=1) for stationary Periodic.  For WarpedPeriodic FIX period=1.0 (do NOT declare a prior) — the (a, b, period) triple is redundant and all frequency information lives in the warp coefficients.
  warp_a (linear warp coefficient): pm.HalfNormal('warp_a', sigma=20.0).  Positive constraint avoids sign ambiguity.  If a toolkit has estimated a_init, switch to pm.Normal('warp_a', mu=a_init, sigma=max(0.3*abs(a_init), 1.0)).
  warp_b (quadratic warp coefficient): pm.Normal('warp_b', mu=0.0, sigma=20.0).  If a toolkit has estimated b_init, switch to pm.Normal('warp_b', mu=b_init, sigma=max(0.3*abs(b_init), 1.0)).
  amplitude (amp): pm.HalfNormal('amp', sigma=2.0)  # roughly 2 x std(y) for centered y.
  linear coefficient c: pm.Normal('c', mu=0, sigma=5) — update based on visible slope.
  observation noise sigma: pm.Exponential('sigma', 1) or pm.HalfNormal('sigma', sigma=0.5).

  CRITICAL TIMING RULE: If you request a toolkit in this step, its result will NOT be available until the NEXT step.  For THIS step's priors you MUST use a standard distribution.  NEVER use placeholder variables like ``a_init`` in the prior string.

PROPOSAL DIVERSITY RULES:
  Proposal '0' must be your PRIMARY/BEST guess.
  Proposals '1' to '{max_key}' should explore structurally different kernel combinations or different priors.
  For non-stationary periodic suspected series, at least one proposal should include WarpedPeriodic and at least one should NOT, so the agent can compare.

ANTI-PATTERNS:
  WRONG: ``"period": "pm.Deterministic('period', dominant_period)"`` -- dominant_period is undefined at runtime.
  WRONG: ``"period": "pm.Deterministic('period', 0.133)"`` -- pm.Deterministic requires a PyTensor tensor, not a raw float.  Use a plain assignment instead: ``period = 0.133``.
  WRONG: ``"warp_a": "pm.Normal('warp_a', mu=a_init, sigma=1)"`` -- a_init is undefined at runtime.
  WRONG: ``"kernels": ["warped_periodic", "periodic"]`` -- WarpedPeriodic subsumes Periodic; do not combine.
  WRONG: ``"kernels": ["periodic"]`` applied to a visibly sweeping non-stationary periodic wave.
  RIGHT: ``"kernels": ["warped_periodic"]`` with priors: warp_a ~ HalfNormal(20), warp_b ~ Normal(0, 20), ls ~ HalfNormal(1), amp ~ HalfNormal(2), sigma ~ Exponential(1).
  RIGHT: ``"kernels": ["linear", "warped_periodic"]`` when a non-stationary periodic wave rides on a linear drift.

CRITICAL RULES:
  Always output exactly {num_proposals} proposals (keys '0' to '{max_key}').
  kernels must always be a LIST.
  If description is COMPLETE, still provide the current best model under proposals '0' (other {max_key} may repeat it).
  NEVER output anything outside the JSON object."""


# ---------------------------------------------------------------------------
#  Proposal prompt — astro variant
# ---------------------------------------------------------------------------


def _build_ts_proposal_prompt(*, num_proposals: int) -> str:
    example_other_keys: str = "\n".join(
        [f'    "{i}": {{ ... }},' for i in range(1, min(num_proposals, 5))]
        + (
            [f'    "{i}": {{ ... }}' for i in range(min(num_proposals, 5), num_proposals)]
            if num_proposals > 5
            else []
        )
    )
    example_other_keys_str: str = ",\n" + example_other_keys if num_proposals > 1 else ""
    return (
        f"Analyze the time series chart from the provided image and infer the structural "
        f"components that most likely generated the signal."
        f"Be explicit about whether the periodicity (if any) is stationary or sweeping "
        f"(a non-stationary periodic wave).  Propose exactly {num_proposals} diverse Gaussian Process models, "
        f"recommending kernels and priors that could plausibly explain the series.\n\n"
        + _ts_shared_rules(num_proposals=num_proposals)
        + "\n\n"
        "YOUR TASK:\n"
        f"Propose EXACTLY {num_proposals} diverse model candidates following the rules above.\n"
        "In the description, state explicitly whether the Stationary v/s Non-Stationary wave check passes and which direction "
        "(frequency increasing or decreasing across the window) if so.\n\n"
        "Return ONLY a valid JSON object with EXACTLY this format:\n"
        "{\n"
        '  "description": "Two to three sentences describing the visible structure (trend, periodicity, whether cycle spacing is constant or sweeping, smoothness, anomalies, noise).",\n'
        '  "proposals": {\n'
        '    "0": {\n'
        '      "kernels": ["warped_periodic"],\n'
        '      "priors": {\n'
        '        "warp_a": "pm.HalfNormal(\'warp_a\', sigma=20.0)",\n'
        '        "warp_b": "pm.Normal(\'warp_b\', mu=0.0, sigma=20.0)",\n'
        '        "ls":     "pm.HalfNormal(\'ls\', sigma=1.0)",\n'
        '        "amp":    "pm.HalfNormal(\'amp\', sigma=2.0)",\n'
        '        "sigma":  "pm.Exponential(\'sigma\', 1)"\n'
        "      }\n"
        "    }" + example_other_keys_str + "\n  }\n}"
    )


# ---------------------------------------------------------------------------
#  Code-gen prompt — astro variant
# ---------------------------------------------------------------------------


TS_CODE_GEN_PROMPT: str = """\
You are an expert in PyMC.  Write a PyMC model for a time series based EXACTLY on these specifications:
Kernels: {kernels}
Priors: {priors}

KERNEL IMPLEMENTATION GUIDANCE:
  Linear: pm.gp.cov.Linear(input_dim=1, c=c)
  Periodic: pm.gp.cov.Periodic(input_dim=1, period=period, ls=ls)
  RBF: pm.gp.cov.ExpQuad(input_dim=1, ls=ls)
  Matern: pm.gp.cov.Matern52(input_dim=1, ls=ls)
  PeriodicComplex: Sum of 3 periodic kernels at harmonic periods (dominant_period, dominant_period/2, dominant_period/3).
  WarpedPeriodic: Build X_warp = warp_a * X + warp_b * pm.math.sqr(X); use pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ls) wrapped by amp**2; pass X_warp (NOT X) to marginal_likelihood and predict.
  Combine kernels by addition (cov = cov1 + cov2).

MODEL STRUCTURE REQUIREMENTS:
  Do NOT write import statements.
  Begin code directly with: with pm.Model() as model:
  Assume data exists as a pandas Series.
  The generated code MUST begin with this exact preamble BEFORE `with pm.Model() as model:
  a) Define X: X = np.arange(len(data))[:, None].astype(float) / len(data)
  b) Define y: y = data.values - np.mean(data.values)
  Never use X, y before defining them in this preamble.
  Do NOT define X, y for the first time inside the `with pm.Model()` block.
  If 'warped_periodic' is in Kernels, BEFORE the `with pm.Model()` block compute a cheap data-driven initializer for the warp coefficients so that pm.find_MAP does not collapse to warp_b=0 and amp=0.  The idea is simple: for a chirp, the number of zero crossings of the centered signal up to normalized time x is approximately 2*(warp_a*x + warp_b*x^2), so counting crossings in the first half of the window and across the full window gives two equations that solve trivially for `a_init` and `b_init` (no least-squares needed).  Store the results as plain Python floats named `a_init` and `b_init`; clamp `a_init` to be at least 1 (sign ambiguity) and allow `b_init` to be negative for decreasing-frequency chirps.  Priors in the Priors dict still use only plain numeric constants — do NOT put `a_init` / `b_init` inside the prior strings.
  Declare the priors EXACTLY as provided in the Priors dictionary.
  NEVER use pm.Deterministic to pass fixed numeric constants (e.g., a known period value).  pm.Deterministic requires a PyTensor tensor and will crash on raw floats with ``AttributeError: 'float' object has no attribute 'type'``.  Instead, assign fixed values as plain Python variables, for instance: ``period = 0.06667``.
  Build the covariance function using the specified kernels.
  If 'warped_periodic' is in Kernels:
    Construct X_warp = warp_a * X + warp_b * pm.math.sqr(X) AFTER priors are declared.
    Do NOT declare a 'period' prior; use period=1.0 fixed inside pm.gp.cov.Periodic.
    Pass X_warp (not X) to gp.marginal_likelihood and to gp.predict.
    When combining with Linear (or other non-warped kernels), apply the non-warped kernel on X and the periodic kernel on X_warp by using pm.gp.cov.Periodic().warp or by summing two separate gps: prefer the simpler path of summing cov_linear(X) + cov_periodic(X_warp) via a single marginal by evaluating both on the SAME inputs -- in that case simplify to WarpedPeriodic alone unless the Linear component is clearly present.
  Define the GP: gp = pm.gp.Marginal(cov_func=cov)
  Define noise prior: sigma = pm.Exponential('sigma', 1) (unless overwritten by the Priors dict).
  Define likelihood: y_ = gp.marginal_likelihood('y', X=X_input, y=y, sigma=sigma)
    where X_input is X_warp if warped_periodic is used, otherwise X.
  Extract MAP: map_estimate = pm.find_MAP().  IMPORTANT: if you computed `a_init` / `b_init` (warped_periodic case) or any other data-driven starting values, pass them through the `start=` argument as a dict keyed by the corresponding PyMC variable names so the optimizer actually uses them (otherwise it starts from prior means and collapses).  Include sensible defaults for the remaining free parameters (lengthscale, amplitude, noise) based on the data scale.  If no inits were computed (plain Periodic / Linear / RBF / Matern), call pm.find_MAP() with no arguments.
  Extract predictions: mu, var = gp.predict(X_input, point=map_estimate, diag=True)
  Calculate trend: trend = pd.Series(mu.flatten() + np.mean(data.values), index=data.index)

CRITICAL JSON FORMATTING RULES:
  Return ONLY a JSON-parseable dictionary containing the code string.
  Each model code must be a SINGLE-LINE string with \\n for newlines
  Use double quotes for the JSON string and escape quotes in code with \\"
  Do NOT include actual newlines in the JSON
  Do NOT output anything outside the JSON object

EXAMPLE OUTPUT (warped periodic, with zero-crossing init and start= passed to find_MAP):
{{"code": "X = np.arange(len(data))[:, None].astype(float) / len(data)\ny = data.values - np.mean(data.values)\nsigns = np.sign(y)\nN_tot = int((np.diff(signs) != 0).sum())\nN_half = int((np.diff(signs[: len(signs) // 2]) != 0).sum())\na_init = max(2.0 * N_half - 0.5 * N_tot, 1.0)\nb_init = float(N_tot - 2.0 * N_half)\nwith pm.Model() as model:\n    warp_a = pm.Normal('warp_a', mu=a_init, sigma=max(0.3*abs(a_init), 1.0))\n    warp_b = pm.Normal('warp_b', mu=b_init, sigma=max(0.3*abs(b_init), 1.0))\n    ls = pm.HalfNormal('ls', sigma=1.0)\n    amp = pm.HalfNormal('amp', sigma=2.0)\n    sigma = pm.Exponential('sigma', 1)\n    X_warp = warp_a * X + warp_b * pm.math.sqr(X)\n    cov = amp**2 * pm.gp.cov.Periodic(input_dim=1, period=1.0, ls=ls)\n    gp = pm.gp.Marginal(cov_func=cov)\n    y_ = gp.marginal_likelihood('y', X=X_warp, y=y, sigma=sigma)\n    map_estimate = pm.find_MAP(start={{'warp_a': a_init, 'warp_b': b_init, 'ls': 0.5, 'amp': float(np.std(y)) or 1.0, 'sigma': 0.3 * (float(np.std(y)) or 1.0)}})\n    mu, var = gp.predict(X_warp, point=map_estimate, diag=True)\n    trend = pd.Series(mu.flatten() + np.mean(data.values), index=data.index)"}}"""


# ---------------------------------------------------------------------------
#  Feedback prompt template — astro variant
# ---------------------------------------------------------------------------


def _build_ts_feedback_prompt_template(*, num_proposals: int, max_steps: int) -> str:
    num_feedback_slots: int = max(max_steps - 1, 1)
    example_other_keys: str = ",\n".join([f'    "{i}": {{{{ ... }}}}' for i in range(1, num_proposals)])
    return (
        "You are a Model Validation Engineer responsible for validating Gaussian Process "
        "(GP) models for time series structure extraction, with explicit support for "
        "non-stationary wave (warped periodic) signals.\n\n"
        "GOAL: Determine whether the current GP model correctly separates underlying "
        "temporal structure from noise and anomalies, and whether a WarpedPeriodic kernel "
        "is warranted for any observed non-stationary periodicity.\n\n"
        "DIAGNOSTIC CONTEXT:\n{plot_type_description}\n\n"
        "CURRENT MODEL INFORMATION:\n"
        "  Current PyMC Code:\n{current_model}\n"
        "  Current Kernel Configuration: {model_structure}\n"
        "  Previously Tested Kernel Configurations: {tested_model_structures}\n"
        "  Toolkit History: {selected_tool_history}\n\n"
        "HISTORY OF RECOMMENDATIONS:\n"
        "  Original Recommendation: {initial_summary}\n"
        + "".join(
            f"  Feedback Step {i}: {{feedback_summary_step_{i}}}\n" for i in range(1, num_feedback_slots + 1)
        )
        + "\n"
        "IMPORTANT CONSTRAINTS:\n"
        "  Evaluate the visualization and model performance based on the most recent feedback run.\n"
        "  Pay attention to the trajectory of past runs, experimented kernels, and their results.\n"
        "  You MUST either change the combination of kernels OR use the same kernels with different prior values if the fit is unsatisfactory.\n"
        "  Do NOT repeat a toolkit more than twice across all iterations.\n"
        "  Ensure interpretable kernel combinations and avoid unnecessary complexity.\n\n"
        "MODEL FIT DIAGNOSTICS -- what to look for:\n"
        "  Anomaly Ignoring: The GP fit should ignore isolated spikes.\n"
        "  Noise Filtering: The GP fit should pass through the center of noisy fluctuations.\n"
        "  Structure Capture: Persistent patterns (trends, stationary cycles, or non-stationary periodic waves) should be captured.\n"
        "  Non-Stationary Periodic Wave Diagnosis: If residuals show systematic oscillation whose spacing sweeps across the window, the current model missed a non-stationary periodic wave — escalate to WarpedPeriodic.\n"
        "  Over-warping: If warp_b is large and the fitted trend oscillates wildly at the edges while the raw data doesn't, the warp is overfitting — tighten priors on warp_b.\n"
        "  Residual Independence: Residuals should resemble white noise.\n\n"
        + _ts_shared_rules(num_proposals=num_proposals)
        + "\n\n"
        "DIAGNOSTIC RESULTS (from this step's analysis):\n"
        "{diagnostic_results}\n\n"
        "YOUR TASK:\n"
        "Determine if the fit is satisfactory (COMPLETE) or if new models should be tested.\n"
        f"If unsatisfactory, propose EXACTLY {num_proposals} diverse revised models following the rules above.\n"
        "If a previous run tried stationary Periodic on what now appears to be a non-stationary periodic wave, switch to WarpedPeriodic.\n"
        "If a previous WarpedPeriodic run overfit (warp coefficients exploded), tighten warp_a / warp_b priors or fall back to Periodic.\n\n"
        "Return ONLY a valid JSON object.  Example:\n"
        "{{\n"
        '  "description": "2-3 sentences evaluating the fit. If satisfactory, set EXACTLY to COMPLETE.",\n'
        '  "proposals": {{\n'
        '    "0": {{\n'
        '      "kernels": ["warped_periodic"],\n'
        '      "priors": {{\n'
        '        "warp_a": "pm.HalfNormal(\'warp_a\', sigma=20.0)",\n'
        '        "warp_b": "pm.Normal(\'warp_b\', mu=0.0, sigma=20.0)",\n'
        '        "ls":     "pm.HalfNormal(\'ls\', sigma=1.0)",\n'
        '        "amp":    "pm.HalfNormal(\'amp\', sigma=2.0)",\n'
        '        "sigma":  "pm.Exponential(\'sigma\', 1)"\n'
        "      }}\n"
        "    }},\n" + example_other_keys + "\n  }}\n}}"
    )


# ---------------------------------------------------------------------------
#  Summary prompts — astro variants
# ---------------------------------------------------------------------------


TS_INITIAL_SUMMARY_PROMPT: str = """\
You are an AI assistant summarizing the initial result of a time series modeling run.

Given the inputs below, generate a concise structured summary following the narrative structure shown in the example.  Limit to 4-5 sentences total.

INSTRUCTIONS:
1. Briefly describe the observed data shape from the description, explicitly noting whether the periodicity is stationary or non-stationary (sweeping).
2. State the chosen kernels (including whether WarpedPeriodic was used) and the specific priors (extract from the PyMC code — name distribution type and key hyperparameters, including warp_a / warp_b if present).
3. Report the AIC score directly.
4. Describe the visualization using the plot description.

EXAMPLE OUTPUT (warped periodic):
Description: The model analyzed a noisy oscillatory series whose cycle spacing visibly compresses from left to right — a clear non-stationary periodic wave.  Model Implementation: The PyMC implementation used a WarpedPeriodic kernel with a quadratic input warp, applying priors `warp_a` ~ HalfNormal(20), `warp_b` ~ Normal(0, 20), `ls` ~ HalfNormal(1), `amp` ~ HalfNormal(2), and `sigma` ~ Exponential(1); period was fixed at 1.0 to break scale redundancy.  Metric: The reported AIC score is 2345.6.  Visualization: The attached plot shows the raw non-stationary periodic wave (grey) and the GP fit (orange), useful for visually confirming that the model tracks the sweeping cycles without overfitting the noise.

EXAMPLE OUTPUT (stationary periodic):
Description: The model analyzed a smooth, oscillatory series with constant cycle spacing and a mild drift.  Model Implementation: The PyMC implementation combined Linear and Periodic kernels with priors `c` ~ Normal(0, 5), `period` ~ Gamma(2, 1), `ls` ~ Gamma(2, 1), `amp` ~ HalfNormal(3), and `sigma` ~ Exponential(1).  Metric: The reported AIC score is 1201.3.  Visualization: The attached plot shows the raw series and the GP fit to help visually inspect structure capture vs noise filtering.

INPUTS:
  Kernels: {kernels}
  PyMC Code:
{pymc_code}
  Data Description: {description}
  AIC Score: {aic_score}
  Diagnostic Context: {plot_description}

Now generate the summary for the provided inputs."""


TS_FEEDBACK_SUMMARY_PROMPT: str = """\
You are an AI assistant summarizing an iterative time series modeling refinement step.

Given the inputs below, generate a concise summary following the EXACT narrative structure shown in the examples.  This summary will be passed to future iterations to guide further refinement, so be precise and informative.

INSTRUCTIONS:
1. Begin by briefly describing the previous fit's issue (from the description), noting explicitly if the previous model misdiagnosed a non-stationary periodic wave as stationary (or vice versa).
2. State the newly chosen kernels (flag WarpedPeriodic explicitly if used) and the specific priors (extract from the PyMC code, including warp_a / warp_b if present).
3. Report the AIC score directly without evaluating it.
4. Explicitly state the toolkit chosen and describe its output:
   If toolkit_type is visualization: state it is a visual tool with no numeric summary.
   If toolkit_type is numeric: include the toolkit_summary value and note any actionable conclusion (e.g., a non-stationary periodic wave warp estimator suggested warp_a ~= 10, warp_b ~= 25 -- use as informative priors next step).
5. Briefly describe the visualization attached.

EXAMPLE OUTPUT (non-stationary periodic wave correction, visualization toolkit):
Basis the previous run's model fit, the description from feedback was that a stationary Periodic kernel left systematic residuals whose spacing compressed across the window, indicating a missed non-stationary periodic wave.  Accordingly, WarpedPeriodic was selected with priors `warp_a` ~ HalfNormal(20), `warp_b` ~ Normal(0, 20), `ls` ~ HalfNormal(1), `amp` ~ HalfNormal(2), `sigma` ~ Exponential(1), and period fixed at 1.0.  This led to a model with an AIC of 1820.4.  The `fit_vs_actuals` toolkit (visualization) was selected; no numeric summary.  The attached plot shows the raw non-stationary periodic wave compared to the new GP fit to help visually confirm that the sweeping cycles are now captured.

EXAMPLE OUTPUT (non-stationary periodic wave warp estimator, numeric toolkit):
Basis the previous run's model fit, the description from feedback was that WarpedPeriodic worked directionally but warp coefficients hit broad-prior boundaries.  Accordingly, WarpedPeriodic was re-run with informative priors `warp_a` ~ Normal(10, 3), `warp_b` ~ Normal(25, 7.5), `ls` ~ HalfNormal(1), `amp` ~ HalfNormal(2), `sigma` ~ Exponential(1), period fixed at 1.0.  This led to a model with an AIC of 1742.1.  The `estimate_chirp_warp` toolkit (numeric) was selected; summary: a_init=10.2, b_init=24.8 from a quadratic fit to cumulative zero-crossings, which motivated the tightened priors above.  The attached plot shows the raw non-stationary periodic wave and the refined GP fit.

INPUTS:
  Kernels: {kernels}
  PyMC Code:
{pymc_code}
  Feedback Description: {description}
  AIC Score: {aic_score}
  Toolkit Used: {selected_tool}, Type: {tool_output_type}, Summary: {tool_output_summary}
  Diagnostic Context: {plot_description}

Now generate the summary for the provided inputs using the mandated narrative structure."""


# ---------------------------------------------------------------------------
#  Plot type descriptions — astro variant
# ---------------------------------------------------------------------------


PLOT_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "fit_vs_actuals": (
        """Creates a visualization showing the raw time series (grey line) and the current GP model fit (orange line).  Used to visually inspect whether the model captures the main structure (trend, seasonality, chirp sweep) while ignoring noise and anomalies.  Helps identify underfitting (model too flat) or overfitting (model chasing noise or wild warp)."""
    ),
    "fit_vs_actuals_with_residuals_distribution": (
        """Creates a visualization showing the raw data vs model fit along with the distribution of residuals (data minus model prediction).  Used to verify whether residuals resemble white noise centered around zero.  If the residual distribution is skewed, heavy-tailed, or multimodal, it suggests the model has not captured the full structure -- for astro signals, a bimodal residual pattern often signals a missed chirp."""
    ),
    "residuals_auto_correlation_plot": (
        """Generates an autocorrelation (ACF) plot of the model residuals.  Used to check whether residuals are temporally independent.  Significant autocorrelation at non-zero lags indicates the model has missed temporal structure such as periodicity, smooth trends, or a chirp sweep."""
    ),
    "residuals_auto_correlation_score": (
        """Computes a numerical diagnostic of residual autocorrelation using the Ljung-Box statistical test across multiple lags.  The output summarizes the lag tested, the Ljung-Box statistic, the p-value, and a plain-language interpretation.  A high p-value (> 0.05) suggests residuals are approximately independent and the model has captured the main structure.  A low p-value indicates significant autocorrelation -- for signals this often means a chirp was fit with a stationary Periodic kernel and the sweep was missed."""
    ),
    "instantaneous_frequency_plot": (
        """Plots an estimate of the instantaneous frequency of the raw signal across the window (e.g., via successive zero-crossing spacings or a short-time transform).  Used to visually diagnose whether the oscillation is stationary (flat line) or a chirp (monotonically rising or falling).  A clearly monotonic instantaneous-frequency curve is the single strongest visual cue to select WarpedPeriodic."""
    )
}

# ---------------------------------------------------------------------------
#  Typed response models
# ---------------------------------------------------------------------------


class TimeSeriesProposal(Typed):
    """One model proposal from the VLM for time-series GP fitting."""

    kernels: List[str]
    priors: Dict[str, str]

    @field_validator("priors")
    @classmethod
    def priors_must_be_nonempty(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) == 0:
            raise ValueError("Each proposal must define at least one prior.")
        return v

    def __str__(self) -> str:
        kernels_str: str = " + ".join(self.kernels)
        priors_lines: List[str] = [f"      {k}: {v}" for k, v in self.priors.items()]
        return f"{kernels_str}\n    priors:\n" + "\n".join(priors_lines)


class TimeSeriesVLMResponse(Typed):
    """Validated VLM response for the time-series domain.

    Pydantic validates all fields at construction time.  If the VLM
    returns malformed JSON, construction raises ``ValidationError``
    which the SlowBurn validator wraps as ``ValueError`` for retry.

    This model is used in Phase 2 (Proposal) of the agentic tool loop.
    Phase 1 (Diagnostic) uses native tool calling via ``call_for_tool()``
    and does not go through this response model.
    """

    description: str
    proposals: Dict[str, TimeSeriesProposal]

    @field_validator("proposals")
    @classmethod
    def proposals_must_be_nonempty(cls, v: Dict[str, TimeSeriesProposal]) -> Dict[str, TimeSeriesProposal]:
        if len(v) == 0:
            raise ValueError("VLM response must include at least one model proposal.")
        return v

    def __str__(self) -> str:
        lines: List[str] = [f"  description: {self.description}"]
        lines.append(f"  proposals ({len(self.proposals)}):")
        for ix, proposal in self.proposals.items():
            lines.append(f"    [{ix}] {proposal}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  DomainPrompts implementation — astro variant
# ---------------------------------------------------------------------------


class TimeSeriesPrompts(DomainPrompts):
    """Astro-aware prompt rendering for time-series GP fitting, with explicit
    support for warped periodic (chirp) signals.

    Drop-in replacement for ``TimeSeriesPrompts`` in the time-series domain
    when the input series are expected to include astro/chirp-like signals.
    Reuses the base ``TimeSeriesProposal`` / ``TimeSeriesVLMResponse`` types
    so downstream tooling needs no changes.
    """

    aliases: ClassVar[List[str]] = DOMAIN_ALIASES

    task_string: ClassVar[str] = "time_series"

    def get_response_type(self) -> Type[Typed]:
        return TimeSeriesVLMResponse

    def render_proposal_prompt(self, *, num_proposals: int) -> str:
        return _build_ts_proposal_prompt(num_proposals=num_proposals)

    def render_code_gen_prompt(self, *, entity_value: Any, priors: Dict[str, str]) -> str:
        return TS_CODE_GEN_PROMPT.format(
            kernels=entity_value,
            priors=priors,
        )

    def render_code_repair_prompt(
        self,
        *,
        base_prompt: str,
        previous_code: str,
        error_message: str,
        repair_context: str,
    ) -> str:
        return (
            f"{base_prompt}\n\n"
            f"The previous PyMC time-series model code or JSON code response failed. "
            f"The full traceback is included below. Fix the response while preserving "
            f"the same requested kernels and priors.\n\n"
            f"{'═' * 60}\n"
            f"RUNTIME API DISCOVERY (from the live Python environment)\n"
            f"{'═' * 60}\n"
            f"{repair_context}\n"
            f"{'═' * 60}\n\n"
            f"Return ONLY a JSON-parseable dictionary with a single key named \"code\".\n\n"
            f"Previous code:\n```python\n{previous_code}\n```\n\n"
            f"Error (full traceback):\n{error_message}"
        )

    def get_feedback_prompt_template(self, *, num_proposals: int, max_steps: int) -> str:
        return _build_ts_feedback_prompt_template(
            num_proposals=num_proposals, max_steps=max_steps
        )

    def render_initial_summary(
        self,
        *,
        entity_value: Any,
        pymc_code: str,
        description: str,
        aic_score: float,
        plot_description: str,
    ) -> str:
        return TS_INITIAL_SUMMARY_PROMPT.format(
            kernels=entity_value,
            pymc_code=textwrap.indent(pymc_code, "    "),
            description=description,
            aic_score=aic_score,
            plot_description=plot_description,
        )

    def render_feedback_summary(
        self,
        *,
        entity_value: Any,
        pymc_code: str,
        description: str,
        aic_score: float,
        plot_description: str,
        tool_name: str,
        tool_output_type: str,
        tool_output_summary: str,
    ) -> str:
        return TS_FEEDBACK_SUMMARY_PROMPT.format(
            kernels=entity_value,
            pymc_code=textwrap.indent(pymc_code, "    "),
            description=description,
            aic_score=aic_score,
            plot_description=plot_description,
            selected_tool=tool_name,
            tool_output_type=tool_output_type,
            tool_output_summary=tool_output_summary,
        )

    def get_entity_key(self) -> str:
        return "kernels"

    def build_ans_dict(self, *, description: str) -> Dict[str, Any]:
        return {
            "description": description,
            "kernels": {},
            "pymc_models": {},
        }

    def extract_proposal_fields(
        self,
        *,
        proposal_config: Dict[str, Any],
        ans: Dict[str, Any],
        ix: str,
    ) -> Tuple[Any, Dict[str, str]]:
        entity_value: List[str] = proposal_config["kernels"]
        priors: Dict[str, str] = proposal_config["priors"]
        ans["kernels"][ix] = entity_value
        return entity_value, priors

    def extract_dataset_fields(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        dist_choice: str = dataset["anomaly_info"]
        dist_label: str = str(dist_choice).lower().replace(" | ", "_")
        return {
            "dataset_idx": dataset["series_id"],
            "dist_label": dist_label,
        }

    def build_step_record_extras(
        self,
        *,
        ans: Dict[str, Any],
        fit_state: TimeSeriesFitState,
    ) -> Dict[str, Any]:
        return {
            "kernels": ans["kernels"],
            "trend": fit_state.trend,
        }

    def build_result_extras(
        self,
        *,
        dataset: Dict[str, Any],
        fit_state: TimeSeriesFitState,
    ) -> Dict[str, Any]:
        return {
            "trend": fit_state.trend,
            "unique_id": dataset["unique_id"] if "unique_id" in dataset else dataset["series_id"],
        }

    def should_log_map_estimate(self) -> bool:
        return False

    def get_plot_type_descriptions(self) -> Dict[str, str]:
        return PLOT_TYPE_DESCRIPTIONS