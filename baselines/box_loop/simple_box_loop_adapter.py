"""
SIMPLIFIED BOX'S APPRENTICE LOOP FOR YOUR USE CASE
====================================================
200 independent numpy arrays → PyMC models (no priors, no critic)

USAGE:
    Use run.py as the entry point. It owns CLI defaults for the model name,
    number of rounds, temperature, max output tokens, checkpoint path,
    resume behavior, task, process count, and request rate limit.
"""

import pickle
import re
import time
import traceback
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from concurry import gather
from morphic.string import format_exception_msg
from scipy.stats import skew


# ============================================================================
# 1. ENVIRONMENT  (minimal wrapper around your numpy array)
# ============================================================================
class SimpleDistributionEnvironment:
    """
    Wraps a single 1-D numpy array so the Box loop can work with it.
    Instantiate one of these per array inside the loop.
    """

    def __init__(self, observed_array: np.ndarray, array_id: int):
        if observed_array.ndim != 1:
            raise ValueError(
                f"observed_array must be 1-D, got shape {observed_array.shape}. "
                "Flatten or pass each column separately."
            )
        self.observed_array = observed_array.astype(float)
        self.array_id = array_id
        self.env_name = "simple_distribution"

        self.df = pd.DataFrame({"observation": self.observed_array})
        self.df.index = [f"True Observation {i}" for i in range(len(self.df))]

    def describe(self) -> Dict[str, Any]:
        """Summary statistics shown to the LLM in every prompt."""
        a = self.observed_array
        return {
            "n": len(a),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
            "skew": float(skew(a)),
            "is_non_negative": bool(np.all(a >= 0)),
            "is_bounded_01": bool(np.all((a >= 0) & (a <= 1))),
        }

    # --- required if you ever plug into the full BoxLoop_Experiment --------
    def get_description(self) -> str:
        return f"Unknown distribution, {len(self.observed_array)} samples."

    def describe_data_columns(self) -> str:
        return "observation: numeric value"

    def get_ordered_column_names(self) -> List[str]:
        return ["observation"]

    def get_data(self) -> List[list]:
        return [[x] for x in self.observed_array]


class SimpleTimeSeriesEnvironment:
    def __init__(self, series_dict, array_id: int, max_obs: int = 80):
        if isinstance(series_dict, dict):
            s = series_dict["data"]
            self.unique_id = series_dict.get("unique_id", str(array_id))
            self.category = series_dict.get("category", "")
            self.name = series_dict.get("name", "")
            self.anomaly_info = series_dict.get("anomaly_info", "")
        elif isinstance(series_dict, pd.Series):
            s = series_dict
            self.unique_id = str(array_id)
            self.category = ""
            self.name = ""
            self.anomaly_info = ""
        else:
            raise TypeError(f"Expected dict or pd.Series, got {type(series_dict)}")

        t_ordinal = (s.index - s.index[0]).days.astype(float).values
        obs = s.values.astype(float)

        # ── Fixed-stride subsampling ──────────────────────────────────────
        # Stride chosen so we keep at most max_obs evenly-spaced points.
        # e.g. 365 obs, max_obs=80 → stride=4 → indices 0,4,8,…,364
        n = len(t_ordinal)
        stride = max(1, n // max_obs)
        idx = np.arange(0, n, stride)  # 0, stride, 2*stride, …

        self._original_n = n
        self._stride = stride
        self._sub_n = len(idx)

        full_df = pd.DataFrame({"time": t_ordinal, "observation": obs})
        self.df = full_df.iloc[idx].reset_index(drop=True)
        self.df.index = [f"True Observation {i}" for i in range(len(self.df))]

        self.array_id = array_id
        self.env_name = "simple_timeseries"
        self._dates = s.index

    def get_description(self) -> str:
        return (
            f"Daily time series '{self.name}' (category: {self.category}). "
            f"{self._sub_n} observations (every {self._stride}th point from "
            f"{self._original_n} total) from "
            f"{self._dates[0].date()} to {self._dates[-1].date()}. "
            f"Anomaly info: {self.anomaly_info}."
        )

    def describe_data_columns(self) -> str:
        return (
            "time        : integer day offset from series start (0 = first day)\n"
            "observation : measured value (float)"
        )

    def get_ordered_column_names(self) -> list:
        return ["time", "observation"]

    def get_data(self) -> list:
        return self.df[["time", "observation"]].values.tolist()


# ============================================================================
# 2. PROMPTS
# ============================================================================

SYSTEM_PROMPT = """\
You are an expert Bayesian statistician specialising in PyMC.

You will be given samples from an UNKNOWN distribution.
You have NO prior knowledge about where the data comes from.
Your job: propose a PyMC probabilistic model that explains the data.

Modelling rules:
- Ground every decision in the empirical statistics shown to you.
- Use appropriate families:
    * HalfNormal / Exponential / Gamma for strictly-positive data
    * Beta for data bounded in [0, 1]
    * Normal / StudentT for symmetric unbounded data
    * Mixture models if the data looks multi-modal
- The code MUST define a function `gen_model(observed_data)`.
- `observed_data` is a pandas DataFrame with one column: "observation".
- The function MUST return exactly `(model, posterior_predictive, trace)`.
- Use EXACTLY these sampling calls (do not change them):
    trace = pm.sample(1000, tune=1000, target_accept=0.90,
                      chains=2, cores=1, random_seed=rng1,
                      idata_kwargs={"log_likelihood": True})
    posterior_predictive = pm.sample_posterior_predictive(
        trace, random_seed=rng2, return_inferencedata=False)
- Do NOT add plotting, printing, or file I/O inside gen_model.

Response format (always use these three sections):
<analysis>
What you observe in the data (shape, range, skew, modality, etc.)
</analysis>

<rationale>
Why you chose this specific model structure
</rationale>

```python
import numpy as np
import pymc as pm
import pandas as pd

def gen_model(observed_data):
    obs  = observed_data["observation"].values
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(314)

    with pm.Model() as model:
        # --- priors ---
        # e.g. mu = pm.Normal("mu", mu=0, sigma=10)

        # --- likelihood ---
        # e.g. y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=obs)

        trace = pm.sample(1000, tune=1000, target_accept=0.90,
                          chains=2, cores=1, random_seed=rng1,
                          idata_kwargs={"log_likelihood": True})
        posterior_predictive = pm.sample_posterior_predictive(
            trace, random_seed=rng2, return_inferencedata=False)

    return model, posterior_predictive, trace
```
"""

TS_SYSTEM_PROMPT = """
You are an expert Bayesian statistician specialising in PyMC Gaussian Processes.
You will be given a time series as a pandas DataFrame with columns "time" and "observation".
Your job: propose a PyMC GP model that explains the series.

Modelling rules:
- Normalise time to [0,1] and centre observations BEFORE the model block.
- Kernel selection:
    * Linear   (pm.gp.cov.Linear(1, c=0.0))              → persistent trend
      NOTE: c is REQUIRED. Omitting c causes TypeError.
    * Periodic (pm.gp.cov.Periodic(1, ls=ls, period=period)) → seasonal pattern
    * ExpQuad  (pm.gp.cov.ExpQuad(1, ls=ls))              → smooth variation
    * Matern52 (pm.gp.cov.Matern52(1, ls=ls))             → rougher variation
  Combine with + or *. Do NOT combine ExpQuad and Matern52.
  Default to a SINGLE kernel — complex models sample much slower.
- Use sigma= not sd= for all distributions.

CRITICAL — HOW TO BUILD THE GP LIKELIHOOD:
Do NOT use gp.marginal_likelihood(). It stores the likelihood as a Potential,
which means log_likelihood will NOT appear in the trace and LOO/WAIC will fail.
Instead, build the covariance matrix explicitly and use pm.MvNormal as the
observed variable. This is the ONLY approach that produces valid LOO/WAIC scores.

    import pytensor.tensor as pt
    K = cov(X, X) + (noise**2 + 1e-6) * pt.eye(n)
    y_obs = pm.MvNormal("y_obs", mu=pt.zeros(n), cov=K, observed=y)

SPEED RULES — mandatory:

1. SUBSAMPLE large series: if len(observed_data) > 80, subsample to exactly 80
   evenly-spaced points BEFORE building X and y. Do NOT use MarginalApprox or
   MarginalSparse — they break log likelihood tracking.

       if n > 80:
           idx = np.linspace(0, n - 1, 80, dtype=int)
           X, y = X[idx], y[idx]
           n = 80

2. FLOAT32: set pytensor.config.floatX = "float32" as the very first line of
   gen_model. Cast X and y to float32 explicitly.

3. PRIORS — use only these:
   - Length-scale: pm.Gamma("ls", alpha=2, beta=1)
   - Period:       pm.Gamma("period", alpha=2, beta=2)
   - Amplitude:    pm.HalfNormal("amp", sigma=1.0)
   - Noise:        pm.HalfNormal("noise", sigma=0.5)
   Do NOT use pm.Uniform, pm.HalfFlat, or pm.Exponential with small rate.

The code MUST define `gen_model(observed_data)` returning `(model, posterior_predictive, trace)`.
Use EXACTLY these sampling calls — do not change any argument:

    trace = pm.sample(50, tune=50, target_accept=0.80,
                      chains=2, cores=1,
                      progressbar=False,
                      compute_convergence_checks=False,
                      random_seed=rng1,
                      idata_kwargs={"log_likelihood": True})
    posterior_predictive = pm.sample_posterior_predictive(
        trace, random_seed=rng2, return_inferencedata=False)

Response format — output ONLY a Python code block, no prose:
```python
import numpy as np
import pymc as pm
import pandas as pd
import pytensor
import pytensor.tensor as pt

def gen_model(observed_data):
    pytensor.config.floatX = "float32"
    t   = observed_data["time"].values
    obs = observed_data["observation"].values
    t_n   = (t - t.min()) / (t.max() - t.min())
    obs_c = obs - obs.mean()
    n = len(t_n)
    X = t_n[:, None].astype("float32")
    y = obs_c.astype("float32")
    if n > 80:
        idx = np.linspace(0, n - 1, 80, dtype=int)
        X, y = X[idx], y[idx]
        n = 80
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(314)
    with pm.Model() as model:
        # priors
        # covariance  (remember: Linear requires c=, e.g. pm.gp.cov.Linear(1, c=0.0))
        # K = cov(X, X) + (noise**2 + 1e-6) * pt.eye(n)
        # y_obs = pm.MvNormal("y_obs", mu=pt.zeros(n), cov=K, observed=y)
        trace = pm.sample(50, tune=50, target_accept=0.80,
                          chains=2, cores=1,
                          progressbar=False,
                          compute_convergence_checks=False,
                          random_seed=rng1,
                          idata_kwargs={"log_likelihood": True})
        posterior_predictive = pm.sample_posterior_predictive(
            trace, random_seed=rng2, return_inferencedata=False)
    return model, posterior_predictive, trace
```
"""


def _stats_block(stats: Dict[str, Any]) -> str:
    flags = []
    if stats["is_bounded_01"]:
        flags.append("bounded in [0, 1]")
    if stats["is_non_negative"]:
        flags.append("all values ≥ 0")
    flag_line = f"  flags    : {', '.join(flags)}\n" if flags else ""
    return (
        f"  n        : {stats['n']}\n"
        f"  mean     : {stats['mean']:.4f}\n"
        f"  std      : {stats['std']:.4f}\n"
        f"  min      : {stats['min']:.4f}\n"
        f"  25th pct : {stats['p25']:.4f}\n"
        f"  median   : {stats['median']:.4f}\n"
        f"  75th pct : {stats['p75']:.4f}\n"
        f"  max      : {stats['max']:.4f}\n"
        f"  skewness : {stats['skew']:.4f}\n"
        f"{flag_line}"
    )


def _build_round1_prompt(env: SimpleDistributionEnvironment) -> str:
    stats = env.describe()
    sample_str = "\n".join(f"  {v:.6f}" for v in env.observed_array[:20])
    return (
        f"Round 1 — initial model\n\n"
        f"Data summary statistics:\n{_stats_block(stats)}\n"
        f"First 20 of {stats['n']} observations:\n{sample_str}\n\n"
        "Propose a PyMC model for this data."
    )


def _build_round1_prompt_ts(env: SimpleTimeSeriesEnvironment) -> str:
    t = env.df["time"].values
    obs = env.df["observation"].values

    # Show first 20 rows with actual dates for readability
    date_strs = [str(env._dates[i].date()) for i in range(min(20, len(t)))]
    sample_rows = "\n".join(
        f"  {date_strs[i]}  (t={int(t[i])})  obs={obs[i]:.6f}" for i in range(len(date_strs))
    )

    return (
        f"Round 1 — initial GP model\n\n"
        f"Series    : {env.name}  (category: {env.category})\n"
        f"Anomalies : {env.anomaly_info}\n\n"
        f"Time series summary:\n"
        f"  n          : {len(obs)}\n"
        f"  date range : {env._dates[0].date()} → {env._dates[-1].date()}\n"
        f"  time range : [0, {int(t.max())}] (integer day offset)\n"
        f"  obs mean   : {obs.mean():.4f}\n"
        f"  obs std    : {obs.std():.4f}\n"
        f"  obs min    : {obs.min():.4f}\n"
        f"  obs max    : {obs.max():.4f}\n\n"
        f"First {len(date_strs)} of {len(t)} rows:\n{sample_rows}\n\n"
        "Propose a PyMC GP model for this time series."
    )


def _build_followup_prompt_ts(
    env: SimpleTimeSeriesEnvironment,
    round_num: int,
    prev_code: str,
    prev_loo: float,
    prev_error,
) -> str:
    t = env.df["time"].values
    obs = env.df["observation"].values

    if prev_error:
        feedback = (
            f"Your previous GP model FAILED:\n\n```\n{prev_error[:800]}\n```\n\n"
            "Fix the error. Common GP causes: wrong X shape (needs 2-D column vector), "
            "sigma ≤ 0, incompatible kernel combination, "
            "or forgetting to normalise the time axis inside gen_model."
        )
    else:
        feedback = (
            f"Your previous GP achieved LOO (elpd_loo) = {prev_loo:.2f}.\n"
            "Higher is better. Consider adjusting the kernel combination, "
            "length-scale priors, or adding a noise floor.\n\n"
            f"Previous code:\n```python\n{prev_code}\n```"
        )

    return (
        f"Round {round_num} — improve your GP model\n\n"
        f"Series: {env.name}  |  n={len(obs)}"
        f"  t∈[0,{int(t.max())}] days"
        f"  obs mean={obs.mean():.4f}  std={obs.std():.4f}\n\n"
        f"{feedback}\n\n"
        "Propose an improved PyMC GP model."
    )


def _build_followup_prompt(
    env: SimpleDistributionEnvironment,
    round_num: int,
    prev_code: str,
    prev_loo: float,
    prev_error: Optional[str],
) -> str:
    stats = env.describe()

    if prev_error:
        feedback = (
            f"Your previous model FAILED with this error:\n\n"
            f"```\n{prev_error[:800]}\n```\n\n"
            "Fix the error. Check shapes, distribution support, and that "
            "the returned variable names are correct."
        )
    else:
        feedback = (
            f"Your previous model achieved LOO (elpd_loo) = {prev_loo:.2f}.\n"
            "Higher LOO is better (less negative = better fit).\n"
            "Consider whether a different distribution family, "
            "heavier tails, or a mixture model would better capture the data.\n\n"
            f"Previous code:\n```python\n{prev_code}\n```"
        )

    return (
        f"Round {round_num} — improve your model\n\n"
        f"Data summary statistics:\n{_stats_block(stats)}\n"
        f"{feedback}\n\n"
        "Propose an improved PyMC model."
    )


# ============================================================================
# 3. CODE EXTRACTION  (handles both `python\n` and `python \n`)
# ============================================================================


def extract_code(response: str) -> str:
    """
    Extract the first ```python ... ``` block from an LLM response.
    Handles both 'python\\n' and 'python \\n' (GPT-4o produces both).
    Raises ValueError if no code block is found.
    """
    for pattern in (r"```python\n(.*?)```", r"```python \n(.*?)```"):
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    raise ValueError(f"No ```python ... ``` block found.\nResponse preview:\n{response[:500]}")


# ============================================================================
# 4. MODEL FITTING & SCORING
# ============================================================================


def fit_model(
    code: str,
    df: pd.DataFrame,
) -> Tuple[Any, Any, Any, Optional[str]]:
    """
    Execute LLM-generated code and call gen_model(df).

    Returns:
        (model, posterior_predictive, trace, error_string)
        On success  → error_string is None.
        On failure  → first three are None, error_string is set.
    """
    try:
        # Second argument to exec() is the globals dict visible inside the
        # executed code, including inside any functions defined there.
        g: Dict[str, Any] = {"np": np, "pm": pm, "pd": pd}
        exec(code, g)

        if "gen_model" not in g:
            return None, None, None, "Code did not define `gen_model`."

        model, pp, trace = g["gen_model"](df)
        return model, pp, trace, None

    except Exception:
        with StringIO() as buf:
            traceback.print_exc(file=buf)
            return None, None, None, buf.getvalue()


def score_trace(trace) -> Tuple[float, float]:
    """
    Returns (elpd_loo, elpd_waic).  Higher = better for both.
    Returns (-inf, -inf) on failure.
    """
    try:
        return float(az.loo(trace).elpd_loo), float(az.waic(trace).elpd_waic)
    except Exception as e:
        print(f"    [scoring error] {e}")
        return float("-inf"), float("-inf")


# ============================================================================
# 5. SINGLE-ARRAY LOOP
# ============================================================================


def run_box_loop_for_array(
    observed_array: Any,
    array_id: Union[int, str],
    *,
    model_name: str,
    num_rounds: int,
    task: str,
    temperature: float,
    max_tokens: int,
    throttle_llm_call: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Run Box's Apprentice loop for ONE numpy array.

    Args:
        observed_array : 1-D numpy array of samples
        array_id       : integer label 0..N-1 (for logging / bookkeeping)
        model_name     : Azure or OpenRouter model string
        num_rounds     : total LLM calls; 5 means 1 proposal + 4 improvements
        task           : original distribution task or time-series GP task
        temperature    : LLM sampling temperature
        max_tokens     : maximum output tokens per LLM call

    Returns dict with keys:
        array_id, best_code, best_model, best_trace,
        best_loo, best_waic, all_rounds, success
    """
    from agent import LMExperimenter

    env = (
        SimpleTimeSeriesEnvironment(observed_array, array_id)
        if task == "box_loop_ts"
        else SimpleDistributionEnvironment(observed_array, array_id)
    )
    agent = LMExperimenter(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        throttle_llm_call=throttle_llm_call,
    )
    agent.set_system_message(TS_SYSTEM_PROMPT if task == "box_loop_ts" else SYSTEM_PROMPT)

    # Best-so-far state  (LOO: higher = better → initialise to -inf)
    best_loo = float("-inf")
    best_waic = float("-inf")
    best_code = None
    best_model = None
    best_trace = None
    all_rounds: List[Dict] = []

    # Inter-round feedback
    prev_code: Optional[str] = None
    prev_loo: Optional[float] = None
    prev_error: Optional[str] = None

    print(f"\n{'=' * 60}")
    print(f"Array {array_id}")
    # print(f"Array {array_id}  |  n={len(observed_array)}"
    #       f"  mean={observed_array.mean():.3f}  std={observed_array.std():.3f}")
    # print(f"{'='*60}")

    for round_num in range(1, num_rounds + 1):
        print(f"\n  Round {round_num}/{num_rounds} … ", end="", flush=True)

        # Build prompt
        if round_num == 1:
            user_msg = _build_round1_prompt_ts(env) if task == "box_loop_ts" else _build_round1_prompt(env)
        else:
            user_msg = (
                _build_followup_prompt_ts(env, round_num, prev_code, prev_loo, prev_error)
                if task == "box_loop_ts"
                else _build_followup_prompt(env, round_num, prev_code, prev_loo, prev_error)
            )

        # Call LLM
        try:
            response = agent.prompt_llm(user_msg)
            print("LLM RESPONSE: ", response)
        except Exception as e:
            print(f"[LLM error: {e}]")
            prev_error = str(e)
            all_rounds.append({"round": round_num, "error": str(e)})
            continue

        # Extract code
        try:
            code = extract_code(response)
        except ValueError as e:
            print("[code extraction failed]")
            prev_error = str(e)
            all_rounds.append({"round": round_num, "error": str(e)})
            continue

        # Fit
        model, pp, trace, fit_error = fit_model(code, env.df)
        if fit_error:
            print("[model fit failed]")
            prev_code = code
            prev_error = fit_error
            all_rounds.append({"round": round_num, "code": code, "error": fit_error})
            continue

        # Score
        loo, waic = score_trace(trace)
        marker = "  ✓ new best" if loo > best_loo else ""
        print(f"LOO={loo:.2f}  WAIC={waic:.2f}{marker}")

        all_rounds.append(
            {
                "round": round_num,
                "code": code,
                "loo": loo,
                "waic": waic,
                "trace": trace,
                "model": model,
                "error": None,
            }
        )

        # Update best
        if loo > best_loo:
            best_loo, best_waic = loo, waic
            best_code, best_model, best_trace = code, model, trace

        # Pass to next round
        prev_code, prev_loo, prev_error = code, loo, None

    success = best_code is not None
    print(f"\n  → best LOO = {best_loo:.2f}" if success else "\n  → all rounds failed")

    return {
        "array_id": array_id,
        "best_code": best_code,
        "best_model": best_model,
        "best_trace": best_trace,
        "best_loo": best_loo,
        "best_waic": best_waic,
        "all_rounds": all_rounds,
        "success": success,
    }


# ============================================================================
# 6. BATCH RUNNER
# ============================================================================
def _sanitize_round(round_result: Dict[str, Any]) -> Dict[str, Any]:
    keep: Set[str] = {"round", "code", "loo", "waic", "error"}
    return {key: round_result[key] for key in keep if key in round_result}


def _sanitize(result: Dict[str, Any]) -> Dict[str, Any]:
    keep: Set[str] = {"array_id", "best_code", "best_loo", "best_waic", "success"}
    sanitized: Dict[str, Any] = {key: result[key] for key in keep if key in result}
    if "all_rounds" in result:
        sanitized["all_rounds"] = [
            _sanitize_round(round_result=round_result) for round_result in result["all_rounds"]
        ]
    return sanitized


def _checkpoint_results(*, results: List[Dict[str, Any]], save_path: Optional[str]) -> None:
    if save_path is None:
        return

    cleaned_results: List[Dict[str, Any]] = [_sanitize(result=result) for result in results]
    output_path: Path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(cleaned_results, output_file)

    csv_rows: List[Dict[str, Any]] = [
        {
            "array_id": result["array_id"],
            "best_code": result.get("best_code"),
            "best_loo": result.get("best_loo"),
            "best_waic": result.get("best_waic"),
            "success": result.get("success"),
        }
        for result in cleaned_results
    ]
    csv_path: Path = output_path.with_suffix(".csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)


def _make_llm_call_throttler(*, per_worker_rpm: int) -> Optional[Callable[[], None]]:
    if per_worker_rpm <= 0:
        return None

    last_llm_call_monotonic: float = 0.0

    def throttle_llm_call() -> None:
        nonlocal last_llm_call_monotonic
        min_interval_seconds: float = 60.0 / per_worker_rpm
        now: float = time.monotonic()
        elapsed_seconds: float = now - last_llm_call_monotonic
        if elapsed_seconds < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed_seconds)
        last_llm_call_monotonic = time.monotonic()

    return throttle_llm_call


def _error_result_from_exception(*, array_id: Union[int, str], exc: Exception) -> Dict[str, Any]:
    return {
        "array_id": array_id,
        "best_code": None,
        "best_loo": float("-inf"),
        "best_waic": float("-inf"),
        "success": False,
        "all_rounds": [
            {
                "round": 0,
                "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }
        ],
    }


def _run_pending_arrays_sequentially(
    *,
    pending_items: List[Tuple[Union[int, str], Any]],
    results: List[Dict[str, Any]],
    model_name: str,
    num_rounds: int,
    task: str,
    temperature: float,
    max_tokens: int,
    save_path: Optional[str],
    total: int,
    per_worker_rpm: int,
) -> None:
    throttle_llm_call: Optional[Callable[[], None]] = _make_llm_call_throttler(
        per_worker_rpm=per_worker_rpm,
    )
    for idx, (array_id, array) in enumerate(pending_items):
        print(f"\n[{idx + 1}/{total}]", end="")
        result: Dict[str, Any] = run_box_loop_for_array(
            observed_array=array,
            array_id=array_id,
            model_name=model_name,
            num_rounds=num_rounds,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
            throttle_llm_call=throttle_llm_call,
        )
        results.append(_sanitize(result=result))
        _checkpoint_results(results=results, save_path=save_path)


def _run_pending_arrays_with_concurry(
    *,
    pending_items: List[Tuple[Union[int, str], Any]],
    results: List[Dict[str, Any]],
    model_name: str,
    num_rounds: int,
    task: str,
    temperature: float,
    max_tokens: int,
    save_path: Optional[str],
    nproc: int,
    per_worker_rpm: int,
) -> None:
    from box_loop_workers import BoxLoopDatasetWorker

    max_workers: int = min(nproc, len(pending_items))
    worker = BoxLoopDatasetWorker.options(
        mode="process",
        max_workers=max_workers,
    ).init(
        model_name=model_name,
        num_rounds=num_rounds,
        task=task,
        temperature=temperature,
        max_tokens=max_tokens,
        per_worker_rpm=per_worker_rpm,
    )
    try:
        futures: List[Any] = [
            worker.run_dataset(array_id=array_id, observed_array=observed_array)
            for array_id, observed_array in pending_items
        ]
        resolved_results: List[Union[Dict[str, Any], Exception]] = gather(
            futures,
            return_exceptions=True,
            progress=len(futures) > 1,
        )
        for (array_id, _), resolved_result in zip(pending_items, resolved_results):
            if isinstance(resolved_result, Exception):
                print(f"\n[Worker error] array {array_id}: {format_exception_msg(resolved_result)}")
                result: Dict[str, Any] = _error_result_from_exception(
                    array_id=array_id,
                    exc=resolved_result,
                )
            else:
                result = resolved_result
            results.append(_sanitize(result=result))
            _checkpoint_results(results=results, save_path=save_path)
    finally:
        worker.stop()


def run_all_arrays(
    arrays_dict: Dict[Union[int, str], Any],
    *,
    model_name: str,
    num_rounds: int,
    temperature: float,
    max_tokens: int,
    save_path: Optional[str],
    resume: bool,
    task: str,
    nproc: int,
    per_worker_rpm: int,
) -> List[Dict[str, Any]]:
    """
    Run Box's Apprentice on all arrays independently.

    Args:
        arrays_dict : {array_id: 1-D numpy array or time-series record}
        model_name  : Azure or OpenRouter model string.
        num_rounds  : total LLM calls per array; 5 means 1 proposal + 4 improvements.
        temperature : LLM sampling temperature.
        max_tokens  : Maximum output tokens per LLM call.
        save_path   : pickle file checkpointed after every array. Pass None to disable.
        resume      : if True and save_path exists, skip already-done arrays.
        task        : original distribution task or time-series GP task.
        nproc       : 0 runs inline; >=1 uses one Concurry process-worker layer.
        per_worker_rpm: LLM requests per minute allowed in each worker process.

    Returns:
        Sanitized result dictionaries.
    """
    if nproc < 0:
        raise ValueError(f"nproc must be >= 0, got {nproc}.")

    completed: Dict[Union[int, str], Dict[str, Any]] = {}
    if resume and save_path is not None:
        try:
            with Path(save_path).open("rb") as input_file:
                previous_results: List[Dict[str, Any]] = pickle.load(input_file)
            completed = {result["array_id"]: result for result in previous_results}
            completed_success_count: int = sum(1 for result in completed.values() if result["success"])
            completed_failure_count: int = len(completed) - completed_success_count
            print(
                f"Resuming: {len(completed)}/{len(arrays_dict)} already done "
                f"({completed_success_count} succeeded, {completed_failure_count} failed)."
            )
            if completed_failure_count > 0:
                print(
                    "Existing checkpoint contains failed results. "
                    "Use --no-resume or delete the output .pkl/.csv to rerun them."
                )
        except FileNotFoundError:
            pass

    results: List[Dict[str, Any]] = list(completed.values())
    pending_items: List[Tuple[Union[int, str], Any]] = [
        (array_id, array) for array_id, array in arrays_dict.items() if array_id not in completed
    ]
    total: int = len(arrays_dict)

    if len(pending_items) == 0:
        _checkpoint_results(results=results, save_path=save_path)
    elif nproc == 0:
        _run_pending_arrays_sequentially(
            pending_items=pending_items,
            results=results,
            model_name=model_name,
            num_rounds=num_rounds,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
            save_path=save_path,
            total=total,
            per_worker_rpm=per_worker_rpm,
        )
    else:
        _run_pending_arrays_with_concurry(
            pending_items=pending_items,
            results=results,
            model_name=model_name,
            num_rounds=num_rounds,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
            save_path=save_path,
            nproc=nproc,
            per_worker_rpm=per_worker_rpm,
        )

    n_ok: int = sum(result["success"] for result in results)
    print(f"\nFinished. {n_ok}/{total} arrays succeeded.")
    return results
