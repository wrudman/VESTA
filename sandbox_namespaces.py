"""Shared sandbox namespace builders for exec() and runtime diagnostics.

Provides single-source-of-truth functions for the namespace dicts injected
into exec() sandboxes and used by the repair diagnostics system.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy import stats
from typing import Any, Dict, Optional

# matplotlib is optional for this module — only needed for tool namespace.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


def get_pymc_namespace(*, data: Optional[Any] = None) -> Dict[str, Any]:
    """Return the PyMC model exec/diagnostic namespace.

    When ``data`` is provided, includes it under the ``"data"`` key
    (used for exec sandboxes). When omitted, the namespace is suitable
    for API introspection during repair diagnostics.
    """
    ns: Dict[str, Any] = {"pm": pm, "np": np, "pt": pt, "pd": pd}
    if data is not None:
        ns["data"] = data
    return ns


def get_tool_runtime_namespace(
    *, plt_wrapper: Optional[Any] = None
) -> Dict[str, Any]:
    """Return the dynamic tool diagnostic namespace (libraries only).

    Used by both the repair diagnostics system and ``execute_dynamic_tool``
    to build the full exec namespace.

    Args:
        plt_wrapper: If provided, replaces the raw ``plt`` object in the
            returned namespace. ``execute_dynamic_tool`` passes
            ``_SilentPlt(plt)`` here to suppress interactive display.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for the tool runtime namespace."
        )
    ns: Dict[str, Any] = {"np": np, "plt": plt, "pd": pd, "stats": stats}
    if plt_wrapper is not None:
        ns["plt"] = plt_wrapper
    return ns
