"""Standard delimiters for multi-line log bodies.

The pipeline wraps every prompt, response, and long code body it emits to
stdout / ``run.log`` in a pair of heavy (``=``) delimiters with a light
(``─``) separator under the title. Centralizing the delimiter strings and
the wrapping function here buys us two things:

1. Every multi-line block in every log file has identical structure, so
   readers (humans and scripts alike) can find and parse the blocks
   without having to worry about which module emitted them.
2. Automated tooling such as ``scripts/audit_run_logs.py`` can slice a
   ``run.log`` into blocks by scanning for ``BLOCK_HEAVY_SEP`` instead of
   maintaining copies of the hard-coded delimiter string.

The helper is intentionally thin: it does NOT call ``logger`` itself.
Callers still decide the log level, logger, and anything else; the helper
returns a plain string so the caller can pass it to ``logger.info(...)``,
``logger.debug(...)``, or keep a copy in memory.
"""

from typing import Final

BLOCK_HEAVY_SEP: Final[str] = "=" * 70
BLOCK_LIGHT_SEP: Final[str] = "─" * 70


def format_log_block(*, title: str, body: str, dataset_prefix: str = "") -> str:
    """Wrap ``body`` in a standard heavy/light delimiter block.

    The emitted shape is::

        ======================================================================
        [<dataset_prefix>] <title>
        ──────────────────────────────────────────────────────────────────────
        <body>
        ======================================================================

    Args:
        title: Short descriptive header (one line). Rendered immediately
            after the opening heavy delimiter.
        body: Multi-line body. Rendered between the light separator and
            the closing heavy delimiter. Trailing newline is normalised.
        dataset_prefix: Optional string (e.g. '001_exp_unif') to prepend to the title.

    Returns:
        A string ready to hand to ``logger.info(...)`` or similar. The
        caller is responsible for choosing the log level and logger.
    """
    normalised_body: str = body.rstrip("\n")
    display_title: str = f"[{dataset_prefix}] {title}" if len(dataset_prefix) > 0 else title
    return (
        f"{BLOCK_HEAVY_SEP}\n"
        f"{display_title}\n"
        f"{BLOCK_LIGHT_SEP}\n"
        f"{normalised_body}\n"
        f"{BLOCK_HEAVY_SEP}"
    )
