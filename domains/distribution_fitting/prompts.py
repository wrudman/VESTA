"""Distribution fitting domain — prompt constants, Typed response models, and data shapes."""

import textwrap
from typing import Any, ClassVar, Dict, List, Tuple, Type

from morphic import Typed
from pydantic import field_validator

from domains import DomainPrompts, FitState
from domains.distribution_fitting import DOMAIN_ALIASES

# ---------------------------------------------------------------------------
#  Domain-specific CodeGenResponse
# ---------------------------------------------------------------------------


class DistFittingCodeGenResponse(Typed):
    """Response from a code-generation LLM call for distribution fitting.

    Pydantic validates at construction time; if ``code`` is missing or not
    a string, construction raises ``ValueError`` which triggers SlowBurn retry.
    """

    code: str


# ---------------------------------------------------------------------------
#  Prompt constants (moved from prompts.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Shared rules (included in both predict and feedback prompts via string
#  concatenation — single source of truth for family guide, prior rules,
#  naming conventions, pareto constraints, and forbidden patterns)
# ---------------------------------------------------------------------------


def _df_shared_rules(*, num_proposals: int) -> str:
    """Build the shared rules block with the given proposal count."""
    max_key: int = num_proposals - 1
    return f"""\
DISTRIBUTION FAMILY GUIDE:
  Single families: gaussian, lognormal, cauchy, laplace, student_t, exponential, uniform, weibull, pareto
  Single distribution: distribution_family = ["family"], is_mixture = false
  Mixture of two families: distribution_family = ["family1", "family2"], is_mixture = true

DISTRIBUTION PARAMETRIZATION REFERENCE (use to avoid parametrization mistakes):
  The parameter names below (mu, sigma, alpha, beta, nu, lam, b, m) are
  MATHEMATICAL ROLES, NOT PyMC variable names. PyMC RVs in one model must have
  unique names, so the actual variable names MUST follow NAMING CONVENTION
  below: `<family>_<role>` for single distributions (e.g. gaussian_mu,
  lognormal_sigma) and `<family>_<role>_<component_index>` for mixtures
  (e.g. gaussian_mu_0, pareto_alpha_1). This is how a weibull+pareto mixture
  keeps its two `alpha` parameters distinct as weibull_alpha_0 and pareto_alpha_1.

  gaussian(mu, sigma)     : mu is mean = median = mode (data space); sigma is std.
  lognormal(mu, sigma)    : mu, sigma are in LOG space, NOT data space.
                            median(X) = exp(mu); mean(X) = exp(mu + sigma**2/2).
                            Set mu near log(peak_x), not the peak x-value itself.
  cauchy(alpha, beta)     : alpha is location (median); beta is scale (half-width
                            at half-max). No finite mean or variance.
  laplace(mu, b)          : mu is mean = median = mode; b is scale; Var = 2*b**2.
  student_t(nu, mu, sigma): nu is degrees of freedom (small nu => heavy tails;
                            nu >> 30 approaches Normal); mu, sigma are location
                            and scale in DATA space.
  exponential(lam)        : lam is the RATE (not scale). mean(X) = 1/lam.
  uniform(lower, upper)   : flat density on [lower, upper]. When used as the
                            OBSERVATION likelihood, lower and upper must be fixed
                            numeric literals chosen so they bracket every
                            observed value.
  weibull(alpha, beta)    : alpha is SHAPE (<1: decreasing hazard; =1: exponential;
                            >1: increasing hazard). beta is SCALE.
  pareto(alpha, m)        : alpha is the tail index (smaller => heavier tail);
                            m is the minimum. mean(X) = alpha*m/(alpha-1) for
                            alpha > 1.

SHAPE INTERPRETATION GUIDE:
  Symmetric bell curve -> gaussian or student_t (student_t if heavy tails visible)
  Right-skewed with every observed value strictly positive
                       -> lognormal, exponential, weibull, or pareto
                          (see DATA SUPPORT VALIDATION before selecting)
  Heavy tails on both sides -> cauchy or laplace
  Two visible modes -> mixture of two families
  Flat / bounded -> uniform
  Power-law tail (very heavy right tail) -> pareto (see DATA SUPPORT VALIDATION)

DATA SUPPORT VALIDATION:
  lognormal, exponential, weibull, pareto have support restricted to positive
  values and cannot assign density to observations outside their support:
    - lognormal, weibull : support (0, +inf)  -- every observation must be > 0.
    - exponential        : support [0, +inf)  -- every observation must be >= 0.
    - pareto             : support [m, +inf)  -- every observation must be >= m.
  Read the smallest visible x-value from the histogram before proposing any of
  these families. If the histogram shows values at or below zero (or below m
  for pareto), do NOT propose that family as a SINGLE distribution. It MAY
  appear inside a mixture, but only paired with a real-line component
  (gaussian, cauchy, laplace, or student_t) that covers the remaining
  observations -- a mixture of only positive-support families still cannot
  explain non-positive data.

PARETO-SPECIFIC CONSTRAINTS -- VIOLATIONS CAUSE SAMPLING FAILURE:
  pareto_alpha (tail index): MUST be strictly positive.
    Use pm.HalfNormal('pareto_alpha', sigma=2.0) or pm.Gamma('pareto_alpha', alpha=2.0, beta=1.0).
    NEVER pm.Normal.
  pareto_m (scale / minimum): MUST be strictly positive AND its prior's UPPER
    bound MUST be strictly less than the smallest data value visible in the
    histogram.
    Use pm.Uniform('pareto_m', lower=1e-6, upper=<visual_min * 0.99>).
    NEVER allow m >= min(data).

PRIOR SPECIFICATION RULES:
  ALL hyperparameters must be plain numeric literals derived directly from
  visual inspection of the histogram.

  STEP 1 — READ THE HISTOGRAM FIRST, WRITE NUMBERS DOWN:
    Before naming any distribution, extract these quantities from the plot:
      - Peak location: the x-value where the histogram is tallest.
      - Spread: the approximate half-width of the bulk of the data
        (from the peak to where the density has fallen to roughly half its max).
      - Support: does the data include zero or negative values?
        Is there a hard lower bound visible?
      - Tail behavior: does one tail fall off sharply or drag out slowly?
      - Multimodality: are there two (or more) distinct humps?

  STEP 2 — TRANSLATE OBSERVATIONS INTO HYPERPARAMETERS:
    Location parameters (mu, alpha, or equivalent center):
      Set equal to the visually estimated peak location from Step 1.

    Scale parameters (sigma, b, beta, or equivalent spread):
      Set equal to the visually estimated spread from Step 1.

    Shape parameters (nu, alpha for Weibull/Pareto, etc.):
      Choose a value consistent with the tail heaviness observed:
        - Light, rapidly-decaying tail  → higher shape value
        - Heavy, slowly-decaying tail   → lower shape value

    Strictly-positive parameters (e.g. scale, rate, minimum bound, degrees of
    freedom, tail index):
      Use pm.HalfNormal, pm.Gamma, pm.Exponential, pm.LogNormal, or pm.Uniform
      with lower > 0. NEVER pm.Normal, pm.Cauchy, or any other real-line prior.
      Set the scale hyperparameter to match the visually estimated spread.

    Mixture weights:
      Estimate the approximate fraction of data belonging to each component
      visually (e.g. left hump ≈ 30 %, right hump ≈ 70 %).
      Encode this as a Dirichlet: a=np.array([<frac1>, <frac2>]) scaled to
      sum to a reasonable pseudo-count (e.g. [1.0, 2.3] for a 30/70 split).

  NAMING CONVENTION (must match the code-gen naming exactly):
    Single model  : <family>_<param>          e.g. gaussian_mu, gaussian_sigma
    Mixture model : <family>_<param>_<index>  e.g. gaussian_mu_0, cauchy_beta_1
    Mixture weights: w_<family1>_<family2>    e.g. w_gaussian_cauchy
    NEVER use a bare 'w' as the variable name.

  PRIOR FORMAT — each entry must be a valid PyMC expression string:
    "pm.Normal('gaussian_mu', mu=<peak_value>, sigma=<spread_value>)"
    "pm.HalfNormal('gaussian_sigma', sigma=<spread_value>)"
    "pm.HalfNormal('exponential_lam', sigma=<rate_estimate>)"
    "pm.Dirichlet('w_gaussian_cauchy', a=np.array([<w1>, <w2>]))"

PROPOSAL DIVERSITY RULES:
  Proposal '0' must be your PRIMARY/BEST guess.
  Proposals '1' to '{max_key}' should explore plausible but structurally different alternatives
  (different families, different mixture combinations, or significantly different priors).
  Mix of single distributions and mixtures across the {num_proposals} proposals.

CRITICAL RULES:
  Always output exactly {num_proposals} proposals (keys '0' to '{max_key}').
  distribution_family must always be a LIST (even for single distributions).
  is_mixture must be a JSON boolean: true or false (NOT a quoted string).
  Mixture proposals must include a 'w_<family1>_<family2>' prior using pm.Dirichlet (e.g. w_gaussian_cauchy). NEVER use a bare 'w'.
  Variable names in priors must follow the naming convention exactly.
  If description is COMPLETE, still provide the current best model under proposals "0" (other {max_key} can repeat it).
  NEVER output anything outside the JSON object.

ANTI-PATTERNS:
  WRONG: is_mixture as string: "is_mixture": "true"
  RIGHT: is_mixture as boolean: "is_mixture": true
  WRONG: wrong mixture naming: mu instead of gaussian_mu_0
  RIGHT: correct mixture naming: gaussian_mu_0, cauchy_beta_1"""


def _df_build_proposal_prompt(*, num_proposals: int) -> str:
    """Build the distribution fitting proposal prompt with the given proposal count."""
    example_keys: str = ",\n".join(
        [
            '    "0": {\n      "distribution_family": ["gaussian"],\n      "is_mixture": false,\n      "priors": {\n        "gaussian_mu": "pm.Normal(\'gaussian_mu\', mu=0.0, sigma=5.0)",\n        "gaussian_sigma": "pm.HalfNormal(\'gaussian_sigma\', sigma=2.0)"\n      }\n    }',
            '    "1": {\n      "distribution_family": ["gaussian", "cauchy"],\n      "is_mixture": true,\n      "priors": {\n        "w_gaussian_cauchy": "pm.Dirichlet(\'w_gaussian_cauchy\', a=np.array([1.0, 1.0]))",\n        "gaussian_mu_0": "pm.Normal(\'gaussian_mu_0\', mu=0.0, sigma=5.0)",\n        "gaussian_sigma_0": "pm.HalfNormal(\'gaussian_sigma_0\', sigma=2.0)",\n        "cauchy_alpha_1": "pm.Normal(\'cauchy_alpha_1\', mu=5.0, sigma=3.0)",\n        "cauchy_beta_1": "pm.HalfNormal(\'cauchy_beta_1\', sigma=1.0)"\n      }\n    }',
        ]
        + [f'    "{i}": {{ ... }}' for i in range(2, num_proposals)]
    )
    return (
        f"Analyze the histogram from the provided image and infer the distribution family "
        f"(or mixture) that most likely generated the data.\n\n"
        + _df_shared_rules(num_proposals=num_proposals)
        + "\n\n"
        "YOUR TASK:\n"
        f"Propose EXACTLY {num_proposals} diverse models following the rules above.\n\n"
        "Return ONLY a valid JSON object with EXACTLY this structure:\n"
        '{\n  "description": "Two to three sentences describing shape, modality, symmetry, skew, tails.",\n'
        '  "proposals": {\n' + example_keys + "\n  }\n}"
    )


CODE_GEN_PROMPT: str = """\
You are an expert in PyMC. Write a PyMC model for distribution fitting based EXACTLY on these specifications:
Distribution Family: {distribution_family}
Priors: {priors}

DETERMINE MODEL TYPE:
  If distribution_family has ONE element -> build a single-distribution model.
  If distribution_family has TWO elements -> build a two-component mixture model using pm.Mixture + pm.Dirichlet.

MODEL STRUCTURE REQUIREMENTS:
  Do NOT write import statements.
  Begin code directly with: with pm.Model() as model:
  Assume data exists in scope as a numpy array (passed to observed=data only).
  Declare ALL priors EXACTLY as provided in the Priors dictionary.
  End with: map_estimate = pm.find_MAP()

SINGLE DISTRIBUTION TEMPLATE:
  with pm.Model() as model:
      <family>_param1 = pm.SomePrior('<family>_param1', ...)
      <family>_param2 = pm.SomePrior('<family>_param2', ...)
      obs = pm.ChosenDist('obs', param1=<family>_param1, param2=<family>_param2, observed=data)
      map_estimate = pm.find_MAP()

MIXTURE DISTRIBUTION TEMPLATE:
  with pm.Model() as model:
      w_<family1>_<family2> = pm.Dirichlet('w_<family1>_<family2>', a=np.array([1.0, 1.0]))
      <family1>_param1_0 = pm.SomePrior('<family1>_param1_0', ...)
      <family2>_param1_1 = pm.SomePrior('<family2>_param1_1', ...)
      comp_dists = [pm.Dist1.dist(...), pm.Dist2.dist(...)]
      obs = pm.Mixture('obs', w=w_<family1>_<family2>, comp_dists=comp_dists, observed=data)
      map_estimate = pm.find_MAP()

CRITICAL PyMC SYNTAX REFERENCE:
  pm.Normal('name', mu=..., sigma=...)
  pm.LogNormal('name', mu=..., sigma=...)  [.dist(): pm.LogNormal.dist(mu=..., sigma=...)]
  pm.Exponential('name', lam=...)           [.dist(): pm.Exponential.dist(lam=...)]
  pm.Cauchy('name', alpha=..., beta=...)    [.dist(): pm.Cauchy.dist(alpha=..., beta=...)]
  pm.Laplace('name', mu=..., b=...)         [.dist(): pm.Laplace.dist(mu=..., b=...)]
  pm.StudentT('name', nu=..., mu=..., sigma=...) [.dist(): pm.StudentT.dist(...)]
  pm.Uniform('name', lower=..., upper=...)  [.dist(): pm.Uniform.dist(lower=..., upper=...)]
  pm.Weibull('name', alpha=..., beta=...)   [.dist(): pm.Weibull.dist(alpha=..., beta=...)]
  pm.Pareto('name', alpha=..., m=...)       [.dist(): pm.Pareto.dist(alpha=..., m=...)]

NAMING CONVENTION (must match Priors dict exactly):
  Single: <family>_<param>  e.g. gaussian_mu, gaussian_sigma
  Mixture: <family>_<param>_<component_index>  e.g. gaussian_mu_0, cauchy_beta_1
  Mixture weights: w_<family1>_<family2>  e.g. w_gaussian_cauchy -- NEVER use a bare 'w'

PARETO-SPECIFIC CONSTRAINTS:
  pareto_alpha: MUST be strictly positive. NEVER use pm.Normal.
  pareto_m: MUST be strictly positive AND strictly less than min(data).

DATA SUPPORT:
  lognormal, exponential, weibull, pareto require data > 0 as single distributions.
  They may appear in mixtures even when some data is non-positive.

CRITICAL JSON FORMATTING RULES:
  Return ONLY a JSON-parseable dictionary containing the code string.
  The model code must be a SINGLE-LINE string with \\n for newlines.
  Use double quotes for the JSON string; escape inner double quotes with \\".
  Do NOT include actual newlines inside the JSON string value.
  Do NOT output anything outside the JSON object.

EXAMPLE OUTPUT (single distribution):
{{"code": "with pm.Model() as model:\\n    gaussian_mu = pm.Normal('gaussian_mu', mu=0.0, sigma=5.0)\\n    gaussian_sigma = pm.HalfNormal('gaussian_sigma', sigma=2.0)\\n    obs = pm.Normal('obs', mu=gaussian_mu, sigma=gaussian_sigma, observed=data)\\n    map_estimate = pm.find_MAP()"}}

EXAMPLE OUTPUT (mixture distribution):
{{"code": "with pm.Model() as model:\\n    w_gaussian_cauchy = pm.Dirichlet('w_gaussian_cauchy', a=np.array([1.0, 1.0]))\\n    gaussian_mu_0 = pm.Normal('gaussian_mu_0', mu=0.0, sigma=3.0)\\n    gaussian_sigma_0 = pm.HalfNormal('gaussian_sigma_0', sigma=2.0)\\n    cauchy_alpha_1 = pm.Normal('cauchy_alpha_1', mu=5.0, sigma=3.0)\\n    cauchy_beta_1 = pm.HalfNormal('cauchy_beta_1', sigma=1.0)\\n    comp_dists = [pm.Normal.dist(mu=gaussian_mu_0, sigma=gaussian_sigma_0), pm.Cauchy.dist(alpha=cauchy_alpha_1, beta=cauchy_beta_1)]\\n    obs = pm.Mixture('obs', w=w_gaussian_cauchy, comp_dists=comp_dists, observed=data)\\n    map_estimate = pm.find_MAP()"}}"""


def _df_build_model_spec_feedback_prompt_template(*, num_proposals: int, max_steps: int) -> str:
    """Build the distribution fitting feedback prompt template with the given proposal count.

    Returns a string with {placeholder} slots for _build_model_spec_feedback_prompt to fill.
    Literal braces in JSON examples use {{{{ / }}}} escaping for .format() compatibility.
    """
    max_key: int = num_proposals - 1
    num_feedback_slots: int = max(max_steps - 1, 1)
    example_other_keys: str = ",\n".join([f'    "{i}": {{{{ ... }}}}' for i in range(1, num_proposals)])
    return (
        "You are a Distribution Fitting Validation Engineer. Your task is to evaluate "
        "the current model fit and decide whether to refine it or declare it complete.\n\n"
        "GOAL: Determine whether the current distribution model correctly captures the "
        "shape, location, scale, and tail behavior of the data.\n\n"
        "DIAGNOSTIC CONTEXT:\n{plot_type_description}\n\n"
        "CURRENT MODEL INFORMATION:\n"
        "  Current PyMC Code:\n{current_model}\n"
        "  Current Distribution Family: {model_structure}\n"
        "  Previously Tested Distribution Families: {tested_model_structures}\n"
        "  Toolkit History: {selected_tool_history}\n\n"
        "HISTORY OF RECOMMENDATIONS:\n"
        "  1) Original Recommendation: {initial_summary}\n"
        + "".join(
            f"  {i + 1}) Feedback Step {i}: {{feedback_summary_step_{i}}}\n"
            for i in range(1, num_feedback_slots + 1)
        )
        + "\n"
        "IMPORTANT CONSTRAINTS:\n"
        "  Evaluate fit quality based on the MOST RECENT feedback step.\n"
        "  Pay attention to the trajectory of past runs and their results.\n"
        "  You MAY revisit a previously tested family, but you MUST use sufficiently different priors.\n"
        "  Do NOT repeat a toolkit more than twice across all iterations.\n"
        "  Ensure interpretable distributions and avoid unnecessary complexity.\n\n"
        "MODEL FIT DIAGNOSTICS -- what to look for:\n"
        "  Shape Capture: Does the fitted curve follow the overall histogram shape?\n"
        "  Peak Alignment: Is the mode of the fit aligned with the histogram peak(s)?\n"
        "  Tail Behaviour: Are the tails of the fit consistent with the histogram tails?\n"
        "  Mixture Detection: Are there multiple modes or a bimodal shape requiring a mixture?\n"
        "  Overfitting: Is the model chasing noise rather than the underlying shape?\n\n"
        + _df_shared_rules(num_proposals=num_proposals)
        + "\n\n"
        "DIAGNOSTIC RESULTS (from this step's analysis):\n"
        "{diagnostic_results}\n\n"
        "YOUR TASK:\n"
        "Determine if the fit is satisfactory (COMPLETE) or if new models should be tested.\n"
        f"If unsatisfactory, propose EXACTLY {num_proposals} diverse revised models following the rules above.\n\n"
        "Return ONLY a valid JSON object. Single-distribution example:\n"
        "{{\n"
        '  "description": "2-3 sentences evaluating the fit. If satisfactory, set EXACTLY to COMPLETE.",\n'
        '  "proposals": {{\n'
        '    "0": {{\n'
        '      "distribution_family": ["gaussian"],\n'
        '      "is_mixture": false,\n'
        '      "priors": {{\n'
        '        "gaussian_mu": "pm.Normal(\'gaussian_mu\', mu=0.0, sigma=5.0)",\n'
        '        "gaussian_sigma": "pm.HalfNormal(\'gaussian_sigma\', sigma=2.0)"\n'
        "      }}\n"
        "    }},\n" + example_other_keys + "\n  }}\n}}\n\n"
        "MIXTURE EXAMPLE (only use when bimodal):\n"
        "{{\n"
        '  "description": "The histogram shows two clear modes, suggesting a mixture.",\n'
        '  "proposals": {{\n'
        '    "0": {{\n'
        '      "distribution_family": ["gaussian", "cauchy"],\n'
        '      "is_mixture": true,\n'
        '      "priors": {{\n'
        '        "w_gaussian_cauchy": "pm.Dirichlet(\'w_gaussian_cauchy\', a=np.array([1.0, 1.0]))",\n'
        '        "gaussian_mu_0": "pm.Normal(\'gaussian_mu_0\', mu=0.0, sigma=3.0)",\n'
        '        "gaussian_sigma_0": "pm.HalfNormal(\'gaussian_sigma_0\', sigma=2.0)",\n'
        '        "cauchy_alpha_1": "pm.Normal(\'cauchy_alpha_1\', mu=6.0, sigma=3.0)",\n'
        '        "cauchy_beta_1": "pm.HalfNormal(\'cauchy_beta_1\', sigma=1.0)"\n'
        "      }}\n"
        "    }}\n"
        "  }}\n}}"
    )


INITIAL_SUMMARY_PROMPT: str = """\
You are an AI assistant summarizing the initial result of a distribution fitting run.

Given the inputs below, generate a concise structured summary following the narrative structure shown in the example. Limit to 4-5 sentences total.

INSTRUCTIONS:
1. Briefly describe the observed data shape from the description.
2. State the chosen distribution family (or mixture) and the specific priors used (extract from the PyMC code -- name the distribution type and key hyperparameters).
3. Report the AIC score directly. 
4. Describe the visualization using the plot description.

EXAMPLE OUTPUT:
Description: The data appears right-skewed with a single mode near zero and a heavy right tail, suggesting a positive-support distribution. Model Implementation: The initial model selected the lognormal family with priors lognormal_mu ~ Normal(mu=1.5, sigma=1.0) and lognormal_sigma ~ HalfNormal(sigma=1.0). Metric: The reported AIC score is 1234.5. Visualization: The attached histogram overlays the fitted PDF on the empirical data to allow visual inspection of shape, peak, and tail alignment.

INPUTS:
  Distribution Family: {distribution_family}
  PyMC Code:
{pymc_code}
  Data Description: {description}
  AIC Score: {aic_score}
  Diagnostic Context: {plot_description}

Now generate the summary for the provided inputs."""


FEEDBACK_SUMMARY_PROMPT: str = """\
You are an AI assistant summarizing an iterative distribution fitting refinement step.

Given the inputs below, generate a concise summary following the EXACT narrative structure shown in the examples. This summary will be passed to future iterations to guide further refinement, so be precise and informative.

INSTRUCTIONS:
1. Begin by briefly describing what the previous fit's issue was (from the description).
2. State the newly chosen distribution family (or mixture) and the specific priors used (extract from the PyMC code).
3. Report the AIC score directly without evaluating it.
4. Explicitly state the toolkit chosen and describe its output:
   If toolkit_type is visualization: state it is a visual tool with no numeric summary.
   If toolkit_type is numeric: include the toolkit_summary value and note any actionable conclusion.
5. Briefly describe the visualization attached.

EXAMPLE OUTPUT (visualization toolkit):
Basis the previous run, the description indicated the fit underestimated the right tail, suggesting a heavier-tailed family. Accordingly, the student_t family was selected with priors student_t_nu ~ Gamma(alpha=2.0, beta=0.5), student_t_mu ~ Normal(mu=3.0, sigma=2.0), and student_t_sigma ~ HalfNormal(sigma=2.0). This led to an AIC of 987.3. The qq_plot toolkit (visualization) was selected; no numeric summary. The attached plot shows a QQ plot comparing empirical quantiles to the fitted student_t quantiles, used to assess tail alignment.

EXAMPLE OUTPUT (numeric toolkit):
Basis the previous run, the description indicated two distinct modes suggesting a mixture. Accordingly, the gaussian_cauchy mixture was selected with priors w ~ Dirichlet([1.0,1.0]), gaussian_mu_0 ~ Normal(mu=0.0, sigma=2.0), cauchy_alpha_1 ~ Normal(mu=5.0, sigma=2.0), cauchy_beta_1 ~ HalfNormal(sigma=1.0). This led to an AIC of 1102.7. The calculate_moments toolkit (numeric) was selected; summary: mean=4.2, skew=1.3, kurtosis=2.1 (right skew consistent with the mixture hypothesis). The attached histogram overlays the fitted mixture PDF for visual comparison.

INPUTS:
  Distribution Family: {distribution_family}
  PyMC Code:
{pymc_code}
  Feedback Description: {description}
  AIC Score: {aic_score}
  Toolkit Used: {selected_tool}, Type: {tool_output_type}, Summary: {tool_output_summary}
  Diagnostic Context: {plot_description}

Now generate the summary for the provided inputs using the mandated narrative structure."""


PLOT_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "histogram": (
        """Evaluate whether a fitted model (red line) matches the data histogram. If the model's distribution family matches the overall shape (modality, skew, tail behavior) but is misaligned or mis-scaled, retain the same family (Model 0) and adjust the priors."""
    ),
    "qq_plot": (
        """You are evaluating a QQ plot (quantile–quantile plot) that compares empirical data quantiles to theoretical model quantiles along a 45° reference line. Assess whether the points follow an approximately straight line across the full range of quantiles, with particular attention to tail behavior. If deviations are approximately linear but shifted up or down, adjust the location priors. If the line is too steep or too flat, adjust the scale prior to better match the overall spread. Retain the same distribution family when deviations are linear, as this indicates parameter misalignment rather than structural error. If the middle aligns but the tails systematically diverge, adjust tail parameters (e.g., degrees of freedom in heavy-tailed families); when switching family is warranted, choose one whose support covers the data per DATA SUPPORT VALIDATION. Strong S-shaped curvature indicates tail mismatch, one-sided curvature suggests skew, and sharp tail departures may indicate outliers or heavier tails than modeled. For mixture models, examine whether different quantile regions align with distinct linear segments; if so, adjust each component's location, scale, and tail parameters independently rather than applying global shifts."""
    ),
    "plot_tails_transform": (
        """You are evaluating two tail diagnostic plots returned by plot_tail_transforms: a left log–log CCDF and a right semi-log (log-y) CCDF. Each plot contains two lines: blue/green dots representing the empirical survival function (1 - ECDF), and a red line representing the predicted model fit. Your evaluation should assess both the shape of the empirical tail and how well the red predicted line tracks it. A straight line on the log–log plot indicates power-law / Pareto-type heavy tails; a straight line on the semi-log plot indicates exponential decay. A good fit requires the red line to closely track the empirical dots across the full tail rangee red line decays too fast relative to the empirical dots, the model tail is too light; if it decays too slowly, the tail is too heavy. If neither plot shows a clear linear regime in the empirical data, the assumed tail family is structurally incorrect. Downward curvature in the empirical log–log plot indicates the true tail is lighter than power-law; upward curvature indicates it is heavier. If the semi-log empirical curve bends rather than remaining linear, exponential decay is inappropriate. If the empirical tail shows multiple linear regimes or a visible slope change, consider mixture structure and adjust the tail component separately rather than applying global shifts. If the red line tracks one regime but misses another, the mixture weights or tail parameters need adjustment. Adjust tail parameters (e.g., Pareto shape, Student-t degrees of freedom, exponential rate, lognormal) to match the slope of the empirical line — subject to DATA SUPPORT VALIDATION: positive-support families are only valid as single distributions when every observed value falls within their support."""
    ),
    "probability_plot": (
        """Evaluate whether the empirical cumulative probabilities (purple points) track the fitted distribution's CDF (red dashed line) across the full range of observed values. If alignment is close and approximately linear, retain the same likelihood family and only refine priors for stability. A consistent horizontal shift indicates mis-specified location, so adjust the prior on the mean toward the central mass of the data. A slope mismatch indicates scale misfit: if the empirical curve rises too steeply the variance is too small (increase the scale prior), and if too shallow the variance is too large (decrease it). Systematic tail deviations suggest distributional misfit, such as heavier or lighter tails, motivating alternatives like a Student-t or mixture model. An S-shaped pattern indicates skew not captured by a symmetric likelihood, while localized deviations or step-like behavior suggest multimodality and support a mixture model with component-specific location and scale priors rather than a single broad distribution."""
    ),
    "segment_distributions_and_calculate_moments": (
        """Evaluate the GMM factorization plot to identify each colored component by its visible center, spread, and relative size, cross-referencing each component's color with the printed component index (Component 0, 1, …) to confirm which histogram mass belongs to which label. Then apply the per-component moments from compute_moments to set PyMC priors independently for each component — do not apply global mean or std estimates to individual components. For each component, set the location prior (mu) near the component mean and the scale prior (sigma) near the component std. Choose the distribution family based on the skewness and excess kurtosis hints: if |skew| < 0.5 and |kurt| < 1.0, use Normal; if skew > 0.5, prefer Gamma, Lognormal, or Weibull; if skew < -0.5, prefer a reflected or Beta family; if kurt > 5.0, prefer StudentT with small nu or Laplace; if kurt is between 1.0 and 5.0, prefer StudentT with moderate nu. Family suggestions above are shape-based only; before committing to one, verify via DATA SUPPORT VALIDATION that the family's support covers the component's observed range, and prefer a real-line family (gaussian, cauchy, laplace, student_t) for any component whose data spans zero. If the black dashed total PDF does not envelope the visible histogram well — for example a component peak is missed or a tail is clipped — check whether the GMM weight for that component is consistent with the visible mass fraction and adjust the mixture weight prior accordingly. Retain the same family as Model 0 if the shape is broadly correct and only location or scale need tuning, and propose an alternative family in Models 1–4 only when skewness or kurtosis clearly violates the assumptions of the base family. Each component's priors must be set and updated separately — treat the mixture as K independent sub-models sharing only the weight vector."""
    ),
}


# ---------------------------------------------------------------------------
#  Typed response models
# ---------------------------------------------------------------------------


class DistFittingProposal(Typed):
    """One model proposal from the VLM for distribution fitting."""

    distribution_family: List[str]
    is_mixture: bool
    priors: Dict[str, str]

    @field_validator("priors")
    @classmethod
    def priors_must_be_nonempty(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) == 0:
            raise ValueError("Each proposal must define at least one prior.")
        return v

    def __str__(self) -> str:
        family_str: str = ", ".join(self.distribution_family)
        mixture_label: str = " (mixture)" if self.is_mixture else ""
        priors_lines: List[str] = [f"      {k}: {v}" for k, v in self.priors.items()]
        return f"{family_str}{mixture_label}\n    priors:\n" + "\n".join(priors_lines)


class DistFittingVLMResponse(Typed):
    """Validated VLM response for the distribution-fitting domain.

    Pydantic validates all fields at construction time.  If the VLM
    returns malformed JSON, construction raises ``ValidationError``
    which the SlowBurn validator wraps as ``ValueError`` for retry.

    This model is used in Phase 2 (Proposal) of the agentic tool loop.
    Phase 1 (Diagnostic) uses native tool calling via ``call_for_tool()``
    and does not go through this response model.

    Fields ``toolkit`` and ``tool_calls`` have been removed — tool
    selection is handled entirely by native function-calling in Phase 1.
    """

    description: str
    proposals: Dict[str, DistFittingProposal]

    @field_validator("proposals")
    @classmethod
    def proposals_must_be_nonempty(cls, v: Dict[str, DistFittingProposal]) -> Dict[str, DistFittingProposal]:
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
#  DomainPrompts implementation
# ---------------------------------------------------------------------------


class DistributionFittingPrompts(DomainPrompts):
    """Prompt rendering, VLM response parsing, and data shape definitions
    for 1-D distribution fitting."""

    aliases: ClassVar[List[str]] = DOMAIN_ALIASES

    task_string: ClassVar[str] = "fitting"

    def get_response_type(self) -> Type[Typed]:
        return DistFittingVLMResponse

    def render_proposal_prompt(self, *, num_proposals: int) -> str:
        return _df_build_proposal_prompt(num_proposals=num_proposals)

    def render_code_gen_prompt(self, *, entity_value: Any, priors: Dict[str, str]) -> str:
        return CODE_GEN_PROMPT.format(
            distribution_family=entity_value,
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
            f"The previous PyMC model code or JSON code response failed. "
            f"The full traceback is included below. Fix the response while preserving "
            f"the same requested distribution family and priors.\n\n"
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
        return _df_build_model_spec_feedback_prompt_template(
            num_proposals=num_proposals,
            max_steps=max_steps,
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
        return INITIAL_SUMMARY_PROMPT.format(
            distribution_family=entity_value,
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
        return FEEDBACK_SUMMARY_PROMPT.format(
            distribution_family=entity_value,
            pymc_code=textwrap.indent(pymc_code, "    "),
            description=description,
            aic_score=aic_score,
            plot_description=plot_description,
            selected_tool=tool_name,
            tool_output_type=tool_output_type,
            tool_output_summary=tool_output_summary,
        )

    def get_entity_key(self) -> str:
        return "distribution_family"

    def build_ans_dict(self, *, description: str) -> Dict[str, Any]:
        return {
            "description": description,
            "distribution_family": {},
            "is_mixture": {},
            "pymc_models": {},
        }

    def extract_proposal_fields(
        self,
        *,
        proposal_config: Dict[str, Any],
        ans: Dict[str, Any],
        ix: str,
    ) -> Tuple[Any, Dict[str, str]]:
        entity_value: List[str] = proposal_config["distribution_family"]
        priors: Dict[str, str] = proposal_config["priors"]
        ans["distribution_family"][ix] = entity_value
        ans["is_mixture"][ix] = proposal_config["is_mixture"]
        return entity_value, priors

    def extract_dataset_fields(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        dist_choice: List[str] = dataset["dist_choice"]
        dist_label: str = "_".join(dist_choice) if isinstance(dist_choice, list) else str(dist_choice)
        return {
            "dataset_idx": dataset["idx"],
            "dist_label": dist_label,
        }

    def build_step_record_extras(
        self,
        *,
        ans: Dict[str, Any],
        fit_state: FitState,
    ) -> Dict[str, Any]:
        return {
            "distribution_family": ans["distribution_family"],
            "is_mixture": ans["is_mixture"],
        }

    def build_result_extras(
        self,
        *,
        dataset: Dict[str, Any],
        fit_state: FitState,
    ) -> Dict[str, Any]:
        return {"true_params": dataset["true_params"]}

    def should_log_map_estimate(self) -> bool:
        return True

    def get_plot_type_descriptions(self) -> Dict[str, str]:
        return PLOT_TYPE_DESCRIPTIONS
