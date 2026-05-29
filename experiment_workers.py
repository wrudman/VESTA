"""Concurry Worker classes for two-level parallel dataset execution.

These classes are in a separate module (not ``__main__``) because concurry's
``ProcessWorkerProxy`` uses cloudpickle to serialize the worker class to child
processes.  On Python 3.13, cloudpickle cannot serialize classes defined in
``__main__`` due to ``FrameLocalsProxy`` (PEP 667).  Placing them in an
importable module lets cloudpickle store a module-path reference instead of
serializing bytecode.
"""

import contextvars
import datetime
import json
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Tuple, Union

from concurry import Worker, gather
from morphic import Typed, validate
from morphic.string import format_exception_msg
from pydantic import PrivateAttr

from domains import DomainPrompts
from experiment_config import ExperimentConfig
from logging_utils import format_log_block

logger: logging.Logger = logging.getLogger(__name__)

_DATASET_PREFIX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "dataset_prefix",
    default="",
)


class _DatasetPrefixFilter(logging.Filter):
    """Prepend the current dataset prefix to every log record's message.

    Attached to the root logger so that ALL log lines in the parent
    stdout (interleaved across child processes) carry a dataset identifier.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        prefix: str = _DATASET_PREFIX.get()
        if len(prefix) > 0:
            record.msg = f"[{prefix}] {record.msg}"
        return True


_LOG_FORMAT: str = "%(asctime)s.%(msecs)03d [%(name)-20s] %(levelname)-7s %(message)s"
_LOG_DATEFMT: str = "%Y-%m-%d::%H:%M:%S"

_NOISY_LOGGER_NAMES: Tuple[str, ...] = (
    "matplotlib",
    "PIL",
    "urllib3",
    "httpcore",
    "httpx",
    "openai",
    "litellm",
    "LiteLLM",
    "numba",
    "asyncio",
    "arviz",
    "pymc",
    "pytensor",
    "filelock",
)


class _ThreadFilter(logging.Filter):
    """Only passes log records originating from a specific thread.

    Attached to per-dataset FileHandlers so that in thread mode, each
    dataset's ``run.log`` captures only that dataset's logs (no interleaving
    from other concurrently-running datasets).
    """

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id: int = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


_ASYNCIO_NOISE_PATTERNS: Tuple[str, ...] = (
    "Task was destroyed but it is pending",
    "Exception in callback",
)


class _AsyncioTaskDestroyedFilter(logging.Filter):
    """Drop the LiteLLM/asyncio teardown ``Task was destroyed`` noise.

    Duplicated here (same logic as in ``experiments._AsyncioTaskDestroyedFilter``)
    because the per-dataset file handler is created before ``experiments`` is
    imported in the worker process, and we want the suppression applied to
    file output as well as stdout.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message: str = record.getMessage()
        return not any(pattern in message for pattern in _ASYNCIO_NOISE_PATTERNS)


def _configure_dataset_file_logging(*, dataset_out_dir: str) -> logging.FileHandler:
    """Add a per-dataset FileHandler to the root logger.

    The handler writes to ``{dataset_out_dir}/run.log`` and is filtered to
    the current thread so concurrent datasets don't pollute each other's
    log files. The asyncio teardown-race filter is also attached so
    ``Task was destroyed but it is pending!`` records never reach the file.

    Returns the handler so the caller can remove it in a ``finally`` block.
    """
    log_path: str = os.path.join(dataset_out_dir, "run.log")
    file_handler: logging.FileHandler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    file_handler.addFilter(_ThreadFilter(threading.get_ident()))
    file_handler.addFilter(_AsyncioTaskDestroyedFilter())
    logging.getLogger().addHandler(file_handler)
    return file_handler


def _configure_process_logging(*, verbosity: int) -> None:
    """Set up logging inside a child process (forkserver context).

    Forkserver child processes start with a blank logging config (no handlers).
    This adds a StreamHandler to stdout so logs are visible, and suppresses
    noisy third-party loggers.  The StreamHandler level is controlled by
    ``verbosity`` (0=WARNING, 1=INFO, 2=DEBUG).
    """
    if verbosity >= 2:
        stream_level: int = logging.DEBUG
    elif verbosity >= 1:
        stream_level = logging.INFO
    else:
        stream_level = logging.WARNING

    root: logging.Logger = logging.getLogger()
    if len(root.handlers) == 0:
        logging.basicConfig(
            level=logging.DEBUG,
            format=_LOG_FORMAT,
            datefmt=_LOG_DATEFMT,
        )
    dataset_prefix_filter: _DatasetPrefixFilter = _DatasetPrefixFilter()
    asyncio_filter: _AsyncioTaskDestroyedFilter = _AsyncioTaskDestroyedFilter()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(stream_level)
        handler.addFilter(dataset_prefix_filter)
        handler.addFilter(asyncio_filter)
    for noisy_logger_name in _NOISY_LOGGER_NAMES:
        if verbosity >= 3 and noisy_logger_name in ("litellm", "LiteLLM", "openai"):
            logging.getLogger(noisy_logger_name).setLevel(logging.DEBUG)
        else:
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)


def _log_dataset_run_header(
    *,
    config: ExperimentConfig,
    dataset: Dict[str, Any],
    dataset_idx: int,
    dataset_label: str,
    dataset_out_dir: str,
    run_verbosity: int,
) -> None:
    """Emit a verbose header at the top of a per-dataset ``run.log``.

    Must be called AFTER ``_configure_dataset_file_logging`` so the logs
    are captured by the per-dataset FileHandler (which is thread-filtered
    and captures DEBUG-level records).  Written as:

      1. ``logger.info`` one-liner (also appears on parent stdout at
         ``verbosity>=1`` so multi-dataset runs have per-dataset progress
         markers in the interleaved stream).
      2. ``logger.debug`` blocks for dataset identifiers, environment,
         and the full experiment config.  These hit ``run.log`` (DEBUG
         level) but do NOT clutter the parent stdout at ``verbosity=1``.

    This guarantees each ``run.log`` is self-contained for post-hoc
    analysis: you never need the parent stdout to reconstruct which
    config / dataset / library versions produced a given log.  Raw data
    values are intentionally NOT logged — load the dataset pkl directly
    if you need to inspect them.
    """
    process_id: int = os.getpid()
    thread_id: int = threading.get_ident()
    started_at: str = datetime.datetime.now().isoformat(timespec="seconds")

    logger.info(
        f"=== DATASET RUN start  idx={dataset_idx}  label={dataset_label!r}  "
        f"pid={process_id}  tid={thread_id}  started={started_at} ==="
    )

    parent_run_dir: str = os.path.dirname(dataset_out_dir)
    if config.toolkit.code_gen_model is None:
        model_code_generation_model: str = config.model.litellm_model
    else:
        model_code_generation_model = config.toolkit.code_gen_model
    if config.toolkit.tool_gen_model is None:
        tool_generation_model: str = config.model.litellm_model
    else:
        tool_generation_model = config.toolkit.tool_gen_model

    run_context_body: str = (
        f"Data file:                       {config.data_pkl}\n"
        f"Dataset selector:                {config.dataset_idx}\n"
        f"Domain:                          {config.domain}\n"
        f"Max steps:                       {config.max_steps}\n"
        f"Toolkit mode:                    {config.toolkit.mode}\n"
        f"Model code-generation attempts:  {config.toolkit.max_code_generation_attempts}\n"
        f"Tool generation attempts:        {config.toolkit.max_tool_generation_attempts}\n"
        f"Main model:                      {config.model.litellm_model}\n"
        f"Model code-generation model:     {model_code_generation_model}\n"
        f"Tool generation model:           {tool_generation_model}\n"
        f"Parent run directory:            {parent_run_dir}\n"
        f"Dataset output directory:        {dataset_out_dir}\n"
        f"Parallelism:                     nproc={config.parallel.nproc}, nthread={config.parallel.nthread}\n"
        f"Dataset run verbosity:           {run_verbosity}"
    )
    logger.debug(format_log_block(title="RUN CONTEXT", body=run_context_body))

    non_data_fields: Dict[str, Any] = {
        field_name: field_value for field_name, field_value in dataset.items() if field_name != "data"
    }
    dataset_metadata_body: str = (
        f"Dataset index: {dataset_idx}\n"
        f"Dataset label: {dataset_label}\n"
        f"Output directory: {dataset_out_dir}\n"
        f"Process id: {process_id}\n"
        f"Thread id: {thread_id}\n"
        f"\n"
        f"Non-data dataset fields (identifiers / metadata only; raw values "
        f"in 'data' are omitted):\n"
        f"{json.dumps(non_data_fields, indent=2, default=str)}"
    )
    logger.debug(format_log_block(title="DATASET METADATA", body=dataset_metadata_body))

    # Inline imports required: C extensions (numpy/pandas/pymc/pytensor) set up
    # thread-local state on import.  Importing them at module level in the
    # forkserver process causes SIGSEGV in forked children because the child
    # inherits stale thread-local data from the parent's import-time
    # initialization.  Importing them here ensures they only load inside the
    # child process after fork()..
    import numpy
    import pandas
    import pymc
    import pytensor

    environment_body: str = (
        f"Python:     {sys.version.split()[0]}\n"
        f"NumPy:      {numpy.__version__}\n"
        f"pandas:     {pandas.__version__}\n"
        f"PyMC:       {pymc.__version__}\n"
        f"PyTensor:   {pytensor.__version__}\n"
        f"Executable: {sys.executable}\n"
        f"BLAS env vars (OPENBLAS/OMP/MKL/VECLIB/NUMEXPR)_NUM_THREADS = "
        + ", ".join(
            os.environ.get(var, "<unset>")
            for var in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
    )
    logger.debug(format_log_block(title="ENVIRONMENT", body=environment_body))

    config_json_body: str = json.dumps(config.model_dump(mode="json"), indent=4, default=str)
    logger.debug(format_log_block(title="EXPERIMENT CONFIG", body=config_json_body))


class DatasetRunnerThread(Typed, Worker):
    """Inner worker: runs ``run()`` on a single dataset.

    mode=sync when nthread=0, mode=thread when nthread>0.
    Created inside each ``DatasetRunnerProcess`` via ``post_initialize``.
    """

    config: ExperimentConfig
    run_dir: str
    run_verbosity: int

    @validate
    def run_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full pipeline for one dataset."""
        # Inline import required: Worker methods may execute in process remote context
        from experiments import run

        ds_fields: Dict[str, Any] = DomainPrompts.of(self.config.domain).extract_dataset_fields(dataset)
        dataset_idx: int = ds_fields["dataset_idx"]
        dataset_label: str = ds_fields["dist_label"]
        dataset_prefix: str = f"{dataset_idx:03d}_{dataset_label}"
        _DATASET_PREFIX.set(dataset_prefix)

        dataset_out_dir: str = os.path.join(self.run_dir, dataset_prefix)
        os.makedirs(dataset_out_dir, exist_ok=True)

        file_handler: logging.FileHandler = _configure_dataset_file_logging(
            dataset_out_dir=dataset_out_dir,
        )
        _log_dataset_run_header(
            config=self.config,
            dataset=dataset,
            dataset_idx=dataset_idx,
            dataset_label=dataset_label,
            dataset_out_dir=dataset_out_dir,
            run_verbosity=self.run_verbosity,
        )
        try:
            return run(
                config=self.config,
                dataset=dataset,
                out_dir=dataset_out_dir,
                verbosity=self.run_verbosity,
            )
        except Exception as exc:
            logger.error(f"Dataset {dataset_idx} ({dataset_label}) failed: {format_exception_msg(exc)}")
            return {
                "status": "error",
                "dataset_idx": dataset_idx,
                "dataset_label": dataset_label,
                "data": dataset["data"],
                "steps": [],
                "error": format_exception_msg(exc),
            }
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()


class DatasetRunnerProcess(Typed, Worker):
    """Outer worker: holds an inner ``DatasetRunnerThread`` pool.

    In ``post_initialize`` (runs inside the child process for process mode):
    - Configures logging for the child process (StreamHandler + suppresses noisy loggers)
    - Applies runtime backend settings (matplotlib backend, pytensor mode)
    - Creates the inner ``DatasetRunnerThread`` with thread/sync mode
    """

    config: ExperimentConfig
    run_dir: str
    run_verbosity: int
    nthread: int = 0

    _inner: Any = PrivateAttr()  # Concurry Worker proxy wrapping DatasetRunnerThread

    def post_initialize(self) -> None:
        """Set up process-local state and create the inner thread worker.

        BLAS / OpenMP thread caps are NOT set here — they are handled in
        ``_thread_caps.py`` at parent-process import time and inherited
        by child processes through ``os.environ``.  Setting them in
        ``post_initialize`` would be too late: the child process imports
        ``numpy`` as part of unpickling ``self`` (ExperimentConfig
        transitively references numpy types) before this hook runs, by
        which point OpenBLAS has already dlopen'd and latched its
        thread count.
        """
        _configure_process_logging(verbosity=self.run_verbosity)

        # Inline imports required: post_initialize runs in child process/thread context
        # where process-local matplotlib backend and pytensor mode must be configured.
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib as _matplotlib

        _matplotlib.use("Agg")

        import pytensor as _pytensor

        _pytensor.config.mode = self.config.pytensor_mode.value

        import warnings as _warnings

        _warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        _warnings.filterwarnings("ignore", category=FutureWarning)
        _warnings.filterwarnings(
            "ignore",
            message=r".*PyTensor could not link to a BLAS installation.*",
            category=UserWarning,
        )
        _warnings.filterwarnings(
            "ignore",
            message=r".*pytensor\.config\.cxx.*identifiable `g\+\+` compiler.*",
            category=UserWarning,
        )

        inner_mode: str = "sync" if self.nthread == 0 else "thread"
        inner_workers: int = max(self.nthread, 1)

        self._inner = DatasetRunnerThread.options(
            mode=inner_mode,
            max_workers=inner_workers,
        ).init(config=self.config, run_dir=self.run_dir, run_verbosity=self.run_verbosity)

    @validate
    def run_chunk(self, datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a chunk of datasets through the inner thread pool."""
        futures: List[Any] = [self._inner.run_dataset(ds) for ds in datasets]  # Concurry futures
        results: List[Union[Dict[str, Any], Exception]] = gather(futures, return_exceptions=True)
        resolved: List[Dict[str, Any]] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                failed_ds: Dict[str, Any] = datasets[i]
                ds_fields: Dict[str, Any] = DomainPrompts.of(self.config.domain).extract_dataset_fields(
                    failed_ds
                )
                logger.error(
                    f"Chunk item {i} (dataset {ds_fields['dataset_idx']}, "
                    f"{ds_fields['dist_label']}) raised: {format_exception_msg(res)}"
                )
                resolved.append(
                    {
                        "status": "error",
                        "dataset_idx": ds_fields["dataset_idx"],
                        "dataset_label": ds_fields["dist_label"],
                        "steps": [],
                        "error": format_exception_msg(res),
                    }
                )
            else:
                resolved.append(res)
        return resolved

    def stop(self) -> None:
        """Stop the inner thread worker, then stop self."""
        self._inner.stop()
        super().stop()
