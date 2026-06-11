"""Time series domain — toolkit dispatch, tool Registry, and diagnostic functions.

Tool Architecture (see ``domains/__init__.py`` module docstring for full details):

    ``TimeSeriesStaticTool(Tool, Registry, ABC)`` is this domain's tool Registry.
    Each concrete tool (e.g., ``FitVsActuals``, ``GetDominantPeriod``) subclasses it.
    Morphic auto-registers them under their snake_case class name, so
    ``TimeSeriesStaticTool.of("fit_vs_actuals")`` resolves ``FitVsActuals``.

    ``TimeSeriesToolkit(DomainToolkit)`` is the dispatch class called by
    ``experiments.py``.  Its ``execute_tool()`` delegates to
    ``TimeSeriesStaticTool.of(selected_tool).execute(...)`` — no if/elif chain.

    To add a new time-series tool:
        1. Define a class that subclasses ``TimeSeriesStaticTool``.
        2. Set ``tool_description``, ``output_type``, ``parameters_schema``.
        3. Implement ``execute()``.
        That's it — no schema constants, no dispatch branches, no wiring.
"""

import logging
import threading
from abc import ABC
from typing import Any, ClassVar, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from morphic import Registry, validate
from scipy.signal import correlate, detrend, find_peaks
from statsmodels.stats.diagnostic import acorr_ljungbox

from vesta.domains import DiagnosticArtifact, DiagnosticToolResult, DomainToolkit, FitState, Tool
from vesta.domains.time_series import DOMAIN_ALIASES
from vesta.domains.time_series.plotting import (
    TimeSeriesFitState,
    plot_best_ts_fit,
    plot_fit_vs_actuals_with_residuals_distribution,
    plot_residuals_auto_correlation,
)

logger: logging.Logger = logging.getLogger("domains.time_series.toolkit")


# ══════════════════════════════════════════════════════════════════════════════
#  Tool Registry + concrete tool subclasses
# ══════════════════════════════════════════════════════════════════════════════


class TimeSeriesStaticTool(Tool, Registry, ABC):
    """Registry of all static time-series diagnostic tools.

    Concrete subclasses auto-register under their snake_case class name.
    Use ``TimeSeriesStaticTool.of("fit_vs_actuals")`` to resolve by name,
    or ``TimeSeriesStaticTool.subclasses()`` to list all registered tools.
    """

    pass


# ── Concrete tools ────────────────────────────────────────────────────────────


class GetDominantPeriod(TimeSeriesStaticTool):
    """Extract the dominant period from the time series using autocorrelation."""

    tool_description: ClassVar[str] = (
        "Extract the dominant period from the time series using FFT analysis. "
        "Use this when Periodic or PeriodicComplex kernels are identified and "
        "the period has not yet been numerically calculated. The result will "
        "be available in the NEXT feedback iteration."
    )
    output_type: ClassVar[str] = "numeric"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[TimeSeriesFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        """Return the dominant-period text summary only.

        ``get_dominant_period`` is a numeric-only tool. The current fit overlay
        is already attached as the Phase-2 base context image and described in
        DIAGNOSTIC RESULTS as ``1) Context image``, so we do not re-render it
        here. If the VLM wants a residuals-distribution view, it can call
        ``fit_vs_actuals_with_residuals_distribution`` explicitly.
        """
        summary: Optional[str] = _get_dominant_period(data)
        tool_output_summary: str = "Period extraction summary: " + (
            summary if summary is not None else "No dominant period detected."
        )
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="text",
                    description="Dominant period diagnostic summary",
                    inline_content=tool_output_summary,
                    attachment_path=None,
                    truncated=False,
                ),
            ],
        )


class FitVsActuals(TimeSeriesStaticTool):
    """Visual inspection of GP fit overlaid on raw time series data."""

    tool_description: ClassVar[str] = (
        "Visual inspection of the GP fit overlaid on the raw time series data. "
        "Essential for checking whether the model captures trend and seasonality "
        "while ignoring noise."
    )
    output_type: ClassVar[str] = "visualization"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[TimeSeriesFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        if fit_state is None:
            _plot_raw_series(series=data, path=fit_path)
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Raw time-series plot (no model fitted yet)",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )
        plot_best_ts_fit(
            series=data,
            trend=fit_state.trend,
            kernels=fit_state.kernels,
            path=fit_path,
        )
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="GP fit overlaid on the observed time series",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                )
            ],
        )


class FitVsActualsWithResidualsDistribution(TimeSeriesStaticTool):
    """Combined plot of fit vs actuals AND the residual distribution."""

    tool_description: ClassVar[str] = (
        "Combined plot of fit vs actuals AND the residual distribution. "
        "Checks if the residual distribution appears like white noise. "
        "If residuals are broadly normal, the fit is good."
    )
    output_type: ClassVar[str] = "visualization"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[TimeSeriesFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        if fit_state is None:
            _plot_raw_series(series=data, path=fit_path)
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Raw time-series plot (no model fitted yet)",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )
        plot_fit_vs_actuals_with_residuals_distribution(
            series=data,
            trend=fit_state.trend,
            kernels=fit_state.kernels,
            path=fit_path,
        )
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="Fit overlay with residual distribution diagnostics",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                )
            ],
        )


class ResidualsAutoCorrelationPlot(TimeSeriesStaticTool):
    """ACF plot of residuals to check for temporal independence."""

    tool_description: ClassVar[str] = (
        "ACF plot of residuals to check for temporal independence. "
        "Significant spikes above the confidence band suggest the model "
        "is missing structure."
    )
    output_type: ClassVar[str] = "visualization"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[TimeSeriesFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        if fit_state is None:
            _plot_raw_series(series=data, path=fit_path)
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="Raw time-series plot (residuals unavailable before fitting)",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )
        plot_residuals_auto_correlation(
            series=data,
            trend=fit_state.trend,
            kernels=fit_state.kernels,
            path=fit_path,
        )
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="image",
                    description="Residual autocorrelation (ACF) diagnostic plot",
                    inline_content=None,
                    attachment_path=fit_path,
                    truncated=False,
                )
            ],
        )


class ResidualsAutoCorrelationScore(TimeSeriesStaticTool):
    """Ljung-Box test for residual independence."""

    tool_description: ClassVar[str] = (
        "Ljung-Box test for residual independence. "
        "A p-value > 0.05 indicates residuals are consistent with white noise."
    )
    output_type: ClassVar[str] = "numeric"
    parameters_schema: ClassVar[Dict[str, Any]] = {}

    @validate
    def execute(
        self,
        *,
        data: Any,
        fit_state: Optional[TimeSeriesFitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        selected_tool_args: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticToolResult:
        """Return the Ljung-Box residual independence text summary only.

        ``residuals_auto_correlation_score`` is numeric-only. The current fit
        overlay is already attached as the Phase-2 base context image, so we
        do not re-render it here. The VLM can request
        ``fit_vs_actuals_with_residuals_distribution`` or
        ``residuals_auto_correlation_plot`` separately if it wants a
        visualization of the residual structure.
        """
        if fit_state is None:
            return DiagnosticToolResult(
                tool_name=self.tool_name,
                tool_description=self.tool_description,
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="text",
                        description="Residual autocorrelation score summary",
                        inline_content=(
                            "Residual autocorrelation summary: residuals unavailable before fitting."
                        ),
                        attachment_path=None,
                        truncated=False,
                    ),
                ],
            )
        summary: str = _residuals_auto_correlation_score(
            series=data,
            trend=fit_state.trend,
            kernels=fit_state.kernels,
        )
        tool_output_summary: str = "Residual autocorrelation summary: " + summary
        return DiagnosticToolResult(
            tool_name=self.tool_name,
            tool_description=self.tool_description,
            artifacts=[
                DiagnosticArtifact(
                    artifact_type="text",
                    description="Ljung-Box residual independence summary",
                    inline_content=tool_output_summary,
                    attachment_path=None,
                    truncated=False,
                ),
            ],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic implementation functions (called by the tool classes above)
# ══════════════════════════════════════════════════════════════════════════════

_MATPLOTLIB_LOCK: threading.Lock = threading.Lock()


def _plot_raw_series(*, series: Any, path: str) -> None:
    """Plot just the raw time series (no model overlay). Used at step 0."""
    with _MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(series.index, series.values, color="grey", alpha=0.7, linewidth=0.8, label="Raw data")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Raw Time Series (no model fitted yet)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _residuals_auto_correlation_score(
    *,
    series: Any,
    trend: Any,
    kernels: List[str],
    lags: int = 20,
) -> str:
    """Compute Ljung-Box autocorrelation test score for residuals."""
    residuals: Any = series - trend

    lb_test: Any = acorr_ljungbox(residuals.values, lags=[lags], return_df=True)

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


def _get_dominant_period(series: Any) -> Optional[str]:
    """Robust Period Detector using Autocorrelation (ACF).

    Best for structural/non-sinusoidal signals (ECG, Square Waves).
    """
    vals: np.ndarray = series.values.astype(float)
    vals = detrend(vals, type="linear")

    corr: np.ndarray = correlate(vals, vals, mode="full")
    corr = corr[len(corr) // 2 :]

    if corr[0] == 0:
        return None
    corr /= corr[0]

    peaks: np.ndarray
    props: Dict[str, Any]
    peaks, props = find_peaks(corr, height=0.2, distance=10, prominence=0.1)

    if len(peaks) == 0:
        return None

    best_period: float = float(peaks[0])

    return (
        f"Dominant Period Detected: {best_period / len(series)}. "
        f"This suggests a strong periodic component in the data with a cycle "
        f"length of approximately {best_period / len(series)}."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DomainToolkit dispatch class
# ══════════════════════════════════════════════════════════════════════════════


class TimeSeriesToolkit(DomainToolkit):
    """Toolkit dispatch for time-series GP fitting.

    Delegates tool execution to ``TimeSeriesStaticTool.of(selected_tool)``.
    The if/elif chain is replaced by Registry lookup.
    """

    aliases: ClassVar[List[str]] = DOMAIN_ALIASES

    def get_static_tools(self) -> List[Dict[str, Any]]:
        return [tool_cls.to_openai_schema() for tool_cls in TimeSeriesStaticTool.subclasses()]

    def supports_dynamic_generation(self) -> bool:
        return True

    def execute_tool(
        self,
        *,
        selected_tool: Optional[str],
        selected_tool_args: Dict[str, Any],
        data: Any,
        fit_state: Optional[FitState],
        best_idx: Optional[Union[str, int]],
        fit_path: str,
        plot_type_descriptions: Dict[str, str],
    ) -> DiagnosticToolResult:
        """Run a time-series toolkit function.

        Returns a structured diagnostic tool result.
        """
        if selected_tool is None or selected_tool == "None":
            if fit_state is None:
                _plot_raw_series(series=data, path=fit_path)
                return DiagnosticToolResult(
                    tool_name=FitVsActuals.tool_name,
                    tool_description=plot_type_descriptions.get(
                        FitVsActuals.tool_name,
                        FitVsActuals.tool_description,
                    ),
                    artifacts=[
                        DiagnosticArtifact(
                            artifact_type="image",
                            description="Raw time-series plot (no model fitted yet)",
                            inline_content=None,
                            attachment_path=fit_path,
                            truncated=False,
                        )
                    ],
                )
            plot_best_ts_fit(
                series=data,
                trend=fit_state.trend,
                kernels=fit_state.kernels,
                path=fit_path,
            )
            return DiagnosticToolResult(
                tool_name=FitVsActuals.tool_name,
                tool_description=plot_type_descriptions.get(
                    FitVsActuals.tool_name,
                    FitVsActuals.tool_description,
                ),
                artifacts=[
                    DiagnosticArtifact(
                        artifact_type="image",
                        description="GP fit overlaid on the observed time series",
                        inline_content=None,
                        attachment_path=fit_path,
                        truncated=False,
                    )
                ],
            )

        tool: Optional[TimeSeriesStaticTool]
        try:
            tool = TimeSeriesStaticTool.of(selected_tool)
        except KeyError:
            logger.debug(
                f"Tool {selected_tool!r} not in TimeSeriesStaticTool registry. "
                f"Available: {[cls.tool_name for cls in TimeSeriesStaticTool.subclasses()]}"
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
            f"Unknown static tool {selected_tool!r} for time-series. "
            f"Available: {[cls.tool_name for cls in TimeSeriesStaticTool.subclasses()]}. "
            f"If this was meant to be a dynamic tool, the pipeline should have "
            f"dispatched it via deps.dynamic_tools before reaching execute_tool()."
        )
