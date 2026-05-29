"""Time series domain — plotting and fit state extraction."""

from typing import Any, ClassVar, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf

from domains import _MATPLOTLIB_LOCK, DomainPlotting, FitState
from domains.time_series import DOMAIN_ALIASES


def plot_time_series(
    segment: Any,
    *,
    save_path: str = "timeseries.png",
) -> None:
    """Plot a time series segment and save the figure.

    Args:
        segment: pandas Series (index=time, values=signal).
        save_path: Output file path.
    """
    with _MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(segment.index, segment.values, color="tab:blue", linewidth=1.5, label="Segment")
        fig.subplots_adjust(left=0.05, right=0.95)
        ax.grid(alpha=0.5)
        fig.savefig(save_path)
        plt.close(fig)


def plot_best_ts_fit(
    *,
    series: Any,
    trend: Any,
    kernels: List[str],
    path: str,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Plot time series with Gaussian Process fitted trend overlay."""
    with _MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(series.index, series.values, color="grey", alpha=0.6, linewidth=1.5, label="Observed Data")
        ax.plot(trend.index, trend.values, color="orange", linewidth=2.5, label="GP Fit")
        ax.set_title(f"Time Series Fit | Kernels: {', '.join(kernels)}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


def plot_fit_vs_actuals_with_residuals_distribution(
    *,
    series: Any,
    trend: Any,
    kernels: List[str],
    path: str,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Plot time series fit along with residual series and residual distribution."""
    residuals: Any = series - trend
    sigma: float = residuals.std()
    upper_lim: float = 3 * sigma
    lower_lim: float = -3 * sigma

    with _MATPLOTLIB_LOCK:
        fig = plt.figure(figsize=figsize)
        gs: GridSpec = GridSpec(2, 4, figure=fig, height_ratios=[1.5, 1], wspace=0.15, hspace=0.15)

        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(series.index, series.values, color="grey", alpha=0.6, lw=1.5, label="Observed Data")
        ax1.plot(trend.index, trend.values, color="orange", alpha=0.9, lw=2, label="GP Fit")
        ax1.set_title(f"Time Series Fit | Kernels: {', '.join(kernels)}")
        ax1.legend()
        ax1.grid(alpha=0.25)

        ax3 = fig.add_subplot(gs[1, :3], sharex=ax1)
        ax3.plot(residuals.index, residuals.values, color="#1f77b4", lw=1, label="Residuals")
        ax3.axhspan(lower_lim, upper_lim, color="green", alpha=0.05)
        ax3.axhline(lower_lim, color="red", linestyle="--")
        ax3.axhline(upper_lim, color="red", linestyle="--")
        ax3.set_ylabel("Residual")
        ax3.grid(alpha=0.25)

        ax2 = fig.add_subplot(gs[1, 3], sharey=ax3)
        ax2.hist(residuals.values, bins=20, orientation="horizontal", color="purple", alpha=0.5)
        ax2.axhline(lower_lim, color="red", linestyle="--")
        ax2.axhline(upper_lim, color="red", linestyle="--")
        ax2.set_title("Residual Distribution")

        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


def plot_residuals_auto_correlation(
    *,
    series: Any,
    trend: Any,
    kernels: List[str],
    path: str,
    lags: int = 40,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """Plot autocorrelation function (ACF) of residuals."""
    residuals: Any = series - trend

    with _MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=figsize)

        plot_acf(
            residuals.values,
            lags=lags,
            ax=ax,
            alpha=0.05,
            title=f"Residual Autocorrelation | Kernels: {', '.join(kernels)}",
        )

        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)


class TimeSeriesFitState(FitState):
    """Fit state for time series: GP trend + kernel configuration.

    ``kernels`` is the resolved kernel list for the best-fitting model only;
    the candidate-model dict is collapsed at construction time so downstream
    consumers never need a ``best_idx`` lookup.
    """

    trend: Any
    kernels: List[str]


class TimeSeriesPlotting(DomainPlotting):
    """Visualization for time-series GP fitting: raw series + GP fit overlays."""

    aliases: ClassVar[List[str]] = DOMAIN_ALIASES

    def plot_initial_data(self, data: Any, *, save_path: str) -> None:
        plot_time_series(data, save_path=save_path)

    def plot_fit_overlay(
        self,
        *,
        data: Any,
        fit_state: FitState,
        path: str,
        best_idx: Union[str, int],
    ) -> None:
        plot_best_ts_fit(
            series=data,
            trend=fit_state.trend,
            kernels=fit_state.kernels,
            path=path,
        )

    def get_default_plot_description(self) -> str:
        return "fit_vs_actuals"

    def extract_fit_state(
        self,
        *,
        fit_results: Dict[str, Any],
        best_idx: Union[str, int],
        ans: Dict[str, Any],
    ) -> TimeSeriesFitState:
        if "trend" not in fit_results[best_idx]:
            raise ValueError(
                f"Expected 'trend' in fit_results[{best_idx!r}], "
                f"but only found keys: {list(fit_results[best_idx].keys())}. "
                f"The PyMC code for this time-series model must define a 'trend' variable."
            )
        best_kernels: List[str] = ans["kernels"][best_idx]
        return TimeSeriesFitState(
            trend=fit_results[best_idx]["trend"],
            kernels=best_kernels,
            map_estimate=fit_results[best_idx]["map_estimate"],
            family_name=best_kernels,
        )
