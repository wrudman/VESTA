"""Cap BLAS / OpenMP thread pools to a single thread per process.

Why this module must import first
=================================
OpenBLAS and MKL read their thread-count environment variables
(``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ...)
**once at ``dlopen`` time**.  That happens the very first time any library
in the process imports a compiled BLAS user — typically on ``import
numpy``.  Setting these variables *after* ``import numpy`` is a no-op for
the rest of the process.

Therefore this module MUST be imported before ``numpy``, ``scipy``,
``pandas``, ``pymc``, ``pytensor``, or anything that transitively pulls
them in.  The convention in this repo is to place the line::

    import _thread_caps  # noqa: F401  -- see module docstring; MUST be first

as the very first import of every CLI entry point (``experiments.py``,
``tests/conftest.py``).  Library modules (``processing_utils``,
``domains.*``, ``dynamic_toolkit``) MUST NOT import this module directly —
by the time those are imported, ``numpy`` is already loaded and the caps
would silently do nothing.

Why the default is one thread
=============================
On Linux, conda-forge / PyPI NumPy and SciPy wheels ship the pthreads
build of OpenBLAS (``libopenblasp-*.so`` or vendored
``libscipy_openblas*.so``).  The pthreads backend defaults to one worker
thread per logical CPU and does **not** cooperate with the OpenMP runtime
that PyTensor's compiled C-Ops use (``-fopenmp`` / libgomp).  On a
16-core server this interaction produced hour-long hangs in PyMC GP
``find_MAP`` runs that took seconds on Mac + Accelerate:

  * OpenBLAS#3187: "Slowdown when using openblas-pthreads alongside
    openmp based parallel code"
    https://github.com/OpenMathLib/OpenBLAS/issues/3187
  * pymc#6640: "NUTS sampler for Gaussian process regression uses all
    system cores, even with cores=1"
    https://github.com/pymc-devs/pymc/issues/6640

Capping to a single thread sidesteps the contention entirely.  For the
n ≤ ~1000 GP Cholesky workloads this project runs, single-threaded BLAS
is within a few percent of any multi-threaded configuration anyway
(Cholesky on a 600×600 matrix is ~5 ms per call; pthread spin-up dominates
multi-threaded setups at that size).  Users who run larger-matrix
workloads can raise the cap (see *Configuration* below).

Configuration
=============
Controlled by the ``PYMC_PARALLEL__COMPUTE_THREADS`` environment
variable (the pydantic-settings form of
``ExperimentConfig.parallel.compute_threads``):

  * unset              → cap all BLAS / OpenMP vars to 1 thread (default).
  * integer ``N ≥ 0``  → cap all BLAS / OpenMP vars to ``max(N, 1)``.
  * integer ``N < 0``  → do not touch any env var (opt-out sentinel).
  * any other value    → raises ``ValueError`` at import time.

Per-variable overrides always win because we use ``os.environ.setdefault``.
A shell invocation like ``OPENBLAS_NUM_THREADS=4 python experiments.py …``
leaves ``OPENBLAS_NUM_THREADS=4`` even if ``PYMC_PARALLEL__COMPUTE_THREADS``
is 1.

This module does **not** read ``.env`` files.  ``load_dotenv`` in
``experiments.py`` runs long after ``numpy`` is imported, so a ``.env``
entry for ``PYMC_PARALLEL__COMPUTE_THREADS`` would be too late to affect
BLAS threading anyway.  Set the variable in your shell / systemd unit /
CI config, or pass ``--parallel.compute-threads N`` to the CLI (which
only affects child process workers, not the parent's BLAS state).
"""

import os
from typing import Tuple

_CAP_ENV_VARS: Tuple[str, ...] = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_CONFIG_ENV_VAR: str = "PYMC_PARALLEL__COMPUTE_THREADS"
_DEFAULT_THREAD_COUNT: int = 1


def apply_thread_caps() -> None:
    """Set BLAS / OpenMP thread env vars according to ``PYMC_PARALLEL__COMPUTE_THREADS``.

    See the module docstring for the variable's semantics and the
    rationale for the default.  Existing per-variable overrides (e.g. a
    shell ``OMP_NUM_THREADS=4`` export) are preserved — this function
    uses ``os.environ.setdefault``, never ``os.environ[…] = …``.

    Raises:
        ValueError: If ``PYMC_PARALLEL__COMPUTE_THREADS`` is set to a
            value that cannot be parsed as an integer.
    """
    if _CONFIG_ENV_VAR not in os.environ:
        compute_threads: int = _DEFAULT_THREAD_COUNT
    else:
        raw_value: str = os.environ[_CONFIG_ENV_VAR]
        try:
            compute_threads = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{_CONFIG_ENV_VAR} must be an integer, got {raw_value!r}. "
                f"Use -1 to disable thread capping, 1 for single-threaded BLAS "
                f"(recommended for PyMC GP workloads; see OpenBLAS#3187 and "
                f"pymc#6640), or N>1 to cap at N threads."
            ) from exc

    if compute_threads < 0:
        return

    cap_value: str = str(max(compute_threads, 1))
    for env_var_name in _CAP_ENV_VARS:
        os.environ.setdefault(env_var_name, cap_value)


apply_thread_caps()
