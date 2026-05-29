"""Time series domain — registers all subclasses into the 3 registries."""

from typing import ClassVar, List

DOMAIN_ALIASES: List[str] = ["time-series", "time_series", "ts"]

# Import subclasses to trigger morphic Registry registration
from domains.time_series.plotting import TimeSeriesPlotting  # noqa: F401
from domains.time_series.prompts import TimeSeriesPrompts  # noqa: F401
from domains.time_series.toolkit import TimeSeriesToolkit  # noqa: F401
