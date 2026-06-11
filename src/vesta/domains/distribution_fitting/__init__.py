"""Distribution fitting domain — registers all subclasses into the 3 registries."""

from typing import List

DOMAIN_ALIASES: List[str] = ["fitting", "distribution-fitting"]

# Import subclasses to trigger morphic Registry registration
from vesta.domains.distribution_fitting.plotting import DistributionFittingPlotting  # noqa: F401
from vesta.domains.distribution_fitting.prompts import DistributionFittingPrompts  # noqa: F401
from vesta.domains.distribution_fitting.toolkit import DistributionFittingToolkit  # noqa: F401
