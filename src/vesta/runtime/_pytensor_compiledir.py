"""Give each top-level Python invocation its own PyTensor compile directory.

Why this module must import first
=================================
PyTensor's ``base_compiledir`` defaults to ``~/.pytensor`` and the value
is read once at ``import pytensor`` time — it cannot be modified after
the first PyTensor import.  To override it this module sets
``PYTENSOR_FLAGS`` in ``os.environ`` *before* ``pytensor`` is imported.
Like ``_thread_caps.py``, this module MUST be imported before ``numpy``
/ ``scipy`` / ``pymc`` / ``pytensor``.

Why per-invocation separation
=============================
PyTensor serialises writes to the compile directory via a file lock
(``.lock`` in ``base_compiledir``).  When N processes run concurrent
cold-cache compilations, all N contend for the single lock; with more
than ~10 concurrent processes the lock hits its default
``compile__timeout`` (120 s) and raises::

    filelock.Timeout: The file lock '...\\.lock' could not be acquired.

Documented in:
  * Theano#4858 — "Different compile directories for compiling Theano
    functions from several processes"
    https://github.com/Theano/Theano/issues/4858
  * pymc#6818   — "Issue compiling pytensor functions in multiple processes"
    https://github.com/pymc-devs/pymc/issues/6818

By giving each top-level invocation a unique ``base_compiledir`` keyed
on the parent process PID (disambiguated across reboots by the mtime of
``/proc/self`` on Linux — a value that is stable per-boot and changes
on reboot), concurrent invocations never share a lock.  Within one
invocation, child processes spawned by ``--parallel.nproc N`` inherit
``PYTENSOR_FLAGS`` from the parent via ``os.environ``, so they share
the parent's compiledir and benefit from cache reuse.

Configuration
=============
Controlled by two environment variables:

  * ``PYMC_PYTENSOR_COMPILEDIR_ROOT``
      Unset            → root is ``_pytensor_cache/`` in the project
                         directory (default).
      Set to a path    → use that path as the root under which
                         per-invocation ``pid_<pid>_boot_<mtime>/``
                         subdirectories are created.

  * ``PYTENSOR_FLAGS``
      If the user has already set ``PYTENSOR_FLAGS`` with an explicit
      ``base_compiledir=...``, this module does NOT override it.  User
      config wins; the check is a simple substring test.

Opt-out — point all invocations at the PyTensor default ``~/.pytensor``
(useful when running sequentially and you want cache reuse across
runs)::

    PYTENSOR_FLAGS='base_compiledir=~/.pytensor' python experiments.py ...

Disk cost and cleanup
=====================
Each invocation's compiledir typically holds 50–500 MB of compiled
``.so`` modules (more if the LLM emitted many distinct GP graphs).  The
directories accumulate under ``_pytensor_cache/``; they are safe to
delete between experiment batches::

    rm -rf _pytensor_cache
"""

import os

_COMPILEDIR_ROOT_ENV_VAR: str = "PYMC_PYTENSOR_COMPILEDIR_ROOT"
_DEFAULT_COMPILEDIR_ROOT: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_pytensor_cache"
)
_PYTENSOR_FLAGS_ENV_VAR: str = "PYTENSOR_FLAGS"
_BASE_COMPILEDIR_TOKEN: str = "base_compiledir"


def _boot_marker() -> int:
    """Return a per-boot-stable integer to disambiguate reused PIDs.

    Linux resets ``/proc/self``'s stat info on every process start, but
    the process start time (and therefore PID pool reset boundary) is
    effectively tied to machine boot — two reboots apart, a PID collision
    could point a fresh invocation at a stale compiledir.  Using the
    mtime of ``/proc/self`` gives us a stable marker within a boot that
    changes across reboots.

    On macOS and Windows ``/proc/self`` does not exist; returns 0.  PID
    alone is sufficient uniqueness in practice on those platforms
    because we also create the directory fresh on first use.
    """
    if os.path.exists("/proc/self"):
        return int(os.stat("/proc/self").st_mtime)
    return 0


def apply_per_invocation_compiledir() -> None:
    """Set ``PYTENSOR_FLAGS=base_compiledir=<per-invocation-path>``.

    Respects an existing ``base_compiledir`` in the current
    ``PYTENSOR_FLAGS`` (user override wins) and appends to any other
    pre-existing flags (e.g. ``mode=NUMBA``) rather than clobbering
    them.

    Raises:
        OSError: If ``PYMC_PYTENSOR_COMPILEDIR_ROOT`` points at a path
            that cannot be created (permission denied, parent missing,
            etc.).
    """
    existing_flags: str = os.environ.get(_PYTENSOR_FLAGS_ENV_VAR, "")
    if _BASE_COMPILEDIR_TOKEN in existing_flags:
        return

    if _COMPILEDIR_ROOT_ENV_VAR in os.environ:
        compiledir_root: str = os.environ[_COMPILEDIR_ROOT_ENV_VAR]
    else:
        compiledir_root = _DEFAULT_COMPILEDIR_ROOT
    compiledir_root = os.path.expanduser(compiledir_root)
    os.makedirs(compiledir_root, exist_ok=True)

    invocation_compiledir: str = os.path.join(
        compiledir_root, f"pid_{os.getpid()}_boot_{_boot_marker()}"
    )
    flag_fragment: str = f"{_BASE_COMPILEDIR_TOKEN}={invocation_compiledir}"
    if existing_flags:
        os.environ[_PYTENSOR_FLAGS_ENV_VAR] = f"{existing_flags},{flag_fragment}"
    else:
        os.environ[_PYTENSOR_FLAGS_ENV_VAR] = flag_fragment


apply_per_invocation_compiledir()
