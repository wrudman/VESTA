"""Concurry workers for the Box Loop baseline."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Union

from concurry import Worker
from morphic import Typed, validate
from pydantic import PrivateAttr

# ── Ensure the box_loop directory is on sys.path for Concurry child processes ──
# On macOS, multiprocessing defaults to "spawn" which starts a fresh interpreter
# that does not inherit sys.path modifications from run.py.
_BOX_LOOP_DIR: str = str(Path(__file__).resolve().parent)
if _BOX_LOOP_DIR not in sys.path:
    sys.path.insert(0, _BOX_LOOP_DIR)

from simple_box_loop_adapter import _sanitize, run_box_loop_for_array  # noqa: E402


class BoxLoopDatasetWorker(Typed, Worker):
    """Run one Box Loop dataset entry per Concurry process-worker call."""

    model_name: str
    num_rounds: int
    task: str
    temperature: float
    max_tokens: int
    per_worker_rpm: int

    _last_llm_call_monotonic: float = PrivateAttr(default=0.0)

    def post_initialize(self) -> None:
        """Ensure local box_loop imports resolve inside each worker process."""
        if _BOX_LOOP_DIR not in sys.path:
            sys.path.insert(0, _BOX_LOOP_DIR)

    def _throttle_llm_call(self) -> None:
        if self.per_worker_rpm <= 0:
            return

        min_interval_seconds: float = 60.0 / self.per_worker_rpm
        now: float = time.monotonic()
        elapsed_seconds: float = now - self._last_llm_call_monotonic
        if elapsed_seconds < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed_seconds)
        self._last_llm_call_monotonic = time.monotonic()

    @validate
    def run_dataset(self, *, array_id: Union[int, str], observed_array: Any) -> Dict[str, Any]:
        """Run the existing per-dataset core loop and return sanitized results.

        Args:
            array_id: Dataset identifier preserved in outputs.
            observed_array: Numpy array, pandas Series, or time-series record dict.

        Returns:
            Sanitized Box Loop result suitable for process-boundary transport.
        """

        result: Dict[str, Any] = run_box_loop_for_array(
            observed_array=observed_array,
            array_id=array_id,
            model_name=self.model_name,
            num_rounds=self.num_rounds,
            task=self.task,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            throttle_llm_call=self._throttle_llm_call,
        )
        return _sanitize(result=result)
