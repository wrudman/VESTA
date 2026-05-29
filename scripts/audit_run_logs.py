"""Extract every ``format_log_block`` block from one or more run.log files.

Every prompt, response, code body, and backend REQUEST/RESPONSE summary in
the pipeline is wrapped by ``logging_utils.format_log_block`` which produces
a consistent ``==== / ──── / ====`` delimiter structure. This script walks
a ``run.log`` line-by-line, finds every such block, and prints the title
plus body with an easy-to-read header so the agent or a human can audit
all prompts and responses in a single pass without having to scroll
through thousands of log lines.

Usage::

    python scripts/audit_run_logs.py outputs/**/run.log
    python scripts/audit_run_logs.py -- file1.log file2.log
    python scripts/audit_run_logs.py --filter-titles "PROPOSAL PROMPT" "FEEDBACK RESPONSE" file.log

Behaviour:

- ``--filter-titles`` accepts one or more substrings; if provided, only
  blocks whose title contains at least one substring are printed.
- ``--max-body-lines`` caps each block's body at the given number of
  lines (default: 0 = unlimited). Useful when a body is a 500-line
  generated PyMC model and you just want to see the surrounding blocks.
- ``--show-line-range`` toggles printing the ``L<start>-L<end>`` line
  range at the start of each block (on by default).

Exit code is ``0`` on success, ``1`` if any log file could not be read or
contains obviously malformed blocks (unclosed delimiters).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add the repository root to ``sys.path`` so ``logging_utils`` imports cleanly
# whether the script is invoked from the repo root (``python scripts/...``)
# or from anywhere else.
_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from logging_utils import BLOCK_HEAVY_SEP  # noqa: E402  (sys.path fix required first)

logger: logging.Logger = logging.getLogger("scripts.audit_run_logs")


class _Block:
    """A single ``format_log_block`` block extracted from a run.log."""

    __slots__ = ("title", "body", "start_line", "end_line")

    def __init__(
        self,
        *,
        title: str,
        body: str,
        start_line: int,
        end_line: int,
    ) -> None:
        self.title: str = title
        self.body: str = body
        self.start_line: int = start_line
        self.end_line: int = end_line


def _line_ends_with_heavy_sep(line: str) -> bool:
    """Return True when ``line`` ends with ``BLOCK_HEAVY_SEP``.

    The opening delimiter is emitted on the same physical line as the
    logger's timestamp/name/level prefix (``%(asctime)s [%(name)s] INFO``),
    so we look at ``line.endswith(BLOCK_HEAVY_SEP)`` rather than
    ``line.strip() == BLOCK_HEAVY_SEP``. Closing delimiters are emitted on
    their own line because ``format_log_block`` puts them at the end of
    the log string, which ``logging`` writes as-is.
    """
    return line.rstrip("\n").endswith(BLOCK_HEAVY_SEP)


def _line_is_bare_heavy_sep(line: str) -> bool:
    """Return True when ``line`` is exactly the heavy delimiter (closing)."""
    return line.rstrip("\n") == BLOCK_HEAVY_SEP


def extract_blocks(log_path: Path) -> List[_Block]:
    """Parse ``log_path`` and return every ``format_log_block`` block.

    Algorithm:
        1. Read all lines of the log.
        2. Walk linearly. When a line ends with the heavy separator and
           is either the bare separator (closing) or has content before
           (opening emitted by the logging layer), we treat it as an
           opener. The next line is the block title.
        3. The body spans every subsequent line until the next bare
           heavy-separator line.
        4. The inner light separator that ``format_log_block`` inserts
           under the title is stripped from the body (first line).

    Raises ``ValueError`` if an opener is found without a matching close.
    """
    lines: List[str] = log_path.read_text(errors="replace").splitlines()
    blocks: List[_Block] = []

    index: int = 0
    total_lines: int = len(lines)
    while index < total_lines:
        current_line: str = lines[index]
        if not _line_ends_with_heavy_sep(current_line):
            index += 1
            continue

        # ``current_line`` is an opener. The next line is the title, the
        # line after that is the inner light-separator, and the body
        # continues until the next bare heavy-separator line.
        title_index: int = index + 1
        if title_index >= total_lines:
            raise ValueError(
                f"{log_path}: opener at line {index + 1} has no title line following it."
            )
        title: str = lines[title_index].rstrip("\n")

        body_start: int = title_index + 1
        body_end: Optional[int] = None
        for probe in range(body_start, total_lines):
            if _line_is_bare_heavy_sep(lines[probe]):
                body_end = probe
                break
        if body_end is None:
            raise ValueError(
                f"{log_path}: opener at line {index + 1} "
                f"(title={title!r}) has no closing heavy delimiter."
            )

        # The first body line is the inner light separator — skip it.
        raw_body_lines: List[str] = lines[body_start:body_end]
        if len(raw_body_lines) > 0:
            raw_body_lines = raw_body_lines[1:]
        body: str = "\n".join(raw_body_lines)

        blocks.append(
            _Block(
                title=title,
                body=body,
                start_line=index + 1,
                end_line=body_end + 1,
            )
        )
        index = body_end + 1

    return blocks


def _clip_body(body: str, max_body_lines: int) -> str:
    if max_body_lines <= 0:
        return body
    body_lines: List[str] = body.splitlines()
    if len(body_lines) <= max_body_lines:
        return body
    clipped: List[str] = body_lines[:max_body_lines]
    clipped.append(
        f"... (body truncated: {len(body_lines) - max_body_lines} more lines; "
        f"rerun with --max-body-lines 0 for full text)"
    )
    return "\n".join(clipped)


def _block_matches_filter(block: _Block, filter_titles: List[str]) -> bool:
    if len(filter_titles) == 0:
        return True
    return any(pattern in block.title for pattern in filter_titles)


def emit_blocks(
    log_path: Path,
    blocks: List[_Block],
    *,
    filter_titles: List[str],
    max_body_lines: int,
    show_line_range: bool,
) -> int:
    """Print ``blocks`` to stdout. Returns the count of printed blocks."""
    selected: List[_Block] = [
        block for block in blocks if _block_matches_filter(block, filter_titles)
    ]
    print("\n" + "#" * 80)
    print(f"# {log_path}")
    print(
        f"# blocks_total={len(blocks)}  blocks_printed={len(selected)}  "
        f"filter_titles={filter_titles if len(filter_titles) > 0 else '(all)'}"
    )
    print("#" * 80)
    for block_index, block in enumerate(selected, start=1):
        header: str = f"[{block_index}/{len(selected)}] {block.title}"
        if show_line_range:
            header = f"{header}   (L{block.start_line}-L{block.end_line})"
        print()
        print(header)
        print("-" * len(header))
        print(_clip_body(block.body, max_body_lines))
    return len(selected)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="audit_run_logs",
        description=(
            "Extract and print every format_log_block block from one or more "
            "run.log files so every prompt/response pair can be audited at a glance."
        ),
    )
    parser.add_argument("logs", nargs="+", help="One or more run.log paths.")
    parser.add_argument(
        "--filter-titles",
        nargs="*",
        default=[],
        help=(
            "Optional substrings: only blocks whose title contains any of "
            "these will be emitted. When omitted, every block is emitted."
        ),
    )
    parser.add_argument(
        "--max-body-lines",
        type=int,
        default=0,
        help=(
            "Cap each block's body at this many lines. 0 = unlimited "
            "(default). Useful when a body is a long generated PyMC model."
        ),
    )
    parser.add_argument(
        "--no-line-range",
        action="store_true",
        help="Suppress printing the L<start>-L<end> range on each block header.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args: argparse.Namespace = _parse_args(argv)
    had_errors: bool = False
    for log_path_str in args.logs:
        log_path: Path = Path(log_path_str)
        if not log_path.exists():
            logger.error(f"{log_path}: does not exist")
            had_errors = True
            continue
        try:
            blocks: List[_Block] = extract_blocks(log_path)
        except (OSError, ValueError) as exc:
            from morphic.string import format_exception_msg

            logger.error(f"{log_path}: failed to parse blocks: {format_exception_msg(exc)}")
            had_errors = True
            continue
        emit_blocks(
            log_path,
            blocks,
            filter_titles=args.filter_titles,
            max_body_lines=args.max_body_lines,
            show_line_range=not args.no_line_range,
        )
    return 1 if had_errors else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(main())
