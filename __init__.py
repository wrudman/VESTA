"""pymc_model_selection — VLM-guided statistical model fitting pipeline.

Public API re-exports for convenience when importing the package.
"""

# Trigger domain registration (importing the subpackages registers subclasses)
import domains.distribution_fitting  # noqa: F401
import domains.time_series  # noqa: F401

from .domains import (
    CodeGenResponse,
    DomainPlotting,
    DomainPrompts,
    DomainToolkit,
    Tool,
)
from .experiments import run
from .processing_utils import clean_params, execute_and_fit_models, sanitize_pymc_code
from .vlm_backends import ToolCallResponse, ToolCallResult, VLMBackend

__all__ = [
    # VLM backends
    "VLMBackend",
    "ToolCallResponse",
    "ToolCallResult",
    # Domain registries (ABCs)
    "DomainPrompts",
    "DomainToolkit",
    "DomainPlotting",
    "Tool",
    "CodeGenResponse",
    # Pipeline
    "run",
    # Utilities
    "sanitize_pymc_code",
    "clean_params",
    "execute_and_fit_models",
]
