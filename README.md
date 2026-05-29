# VESTA

# NOTE: REPO STILL UNDER CONSTRUCTION

VLM-guided PyMC model selection for distribution fitting and time-series
forecasting.  An LLM iteratively proposes model structures (distribution
families, GP kernels, priors), generates the PyMC code, and a scoring
loop picks the best fit by AIC.  See `experiments.py` for the full
pipeline.

## Installation

### macOS / cross-platform default

```bash
conda create -n pymc python=3.13
conda activate pymc
uv pip install -r requirements.txt
```

On macOS this links NumPy/SciPy against Apple's Accelerate framework
automatically, which is already highly optimized for GP Cholesky workloads
— no extra setup needed.

### Linux x86_64 with Intel MKL (recommended for production sweeps)

On Linux x86_64 (Ubuntu / Debian / RHEL / Fedora / Amazon Linux, glibc ≥
2.27), install the base requirements *and* the MKL delta:

```bash
conda create -n pymc python=3.13
conda activate pymc

# 1. Base cross-platform deps (installs NumPy/SciPy with vendored OpenBLAS)
uv pip install -r requirements.txt

# 2. Swap NumPy/SciPy for MKL-linked wheels and install Intel MKL runtime.
#    See the header of requirements-extra-mkl.txt for why these flags are needed.
uv pip install \
    --reinstall-package numpy --reinstall-package scipy \
    --index-strategy unsafe-best-match \
    -r requirements-extra-mkl.txt
```

Verify the swap worked:

```bash
python -c "from threadpoolctl import threadpool_info; \
           [print(d['prefix'], d.get('threading_layer')) for d in threadpool_info()]"
```

You should see `libmkl_rt intel` and `libiomp None`, *not*
`libscipy_openblas pthreads`.  If you see the latter, the MKL wheels did
not take effect — re-run the second `uv pip install` command and check
the `--index-strategy` flag.

### Other platforms

  * **Linux ARM64** (AWS Graviton, Ampere, Raspberry Pi): use only
    `requirements.txt`.  MKL is Intel-only and will not install.  The
    vendored OpenBLAS in the PyPI wheels works, but make sure the thread
    cap (below) is applied.
  * **Windows**: use only `requirements.txt`.  MKL on Windows is
    possible in principle but the urob/numpy-mkl wheels have not been
    tested there.
  * **Linux x86_64 without MKL**: use only `requirements.txt`.  The
    vendored OpenBLAS will be thread-capped automatically (below).

## Pre-import environment setup (critical for Linux)

`experiments.py` has two "must import first" modules that mutate
`os.environ` before NumPy / SciPy / PyMC / PyTensor load:

### `_thread_caps.py` — BLAS / OpenMP thread caps

On Linux, NumPy and SciPy wheels ship with a pthread-backend OpenBLAS
that defaults to one worker thread per logical CPU.  Combined with
PyTensor's OpenMP-compiled C-Ops (`-fopenmp` / libgomp), this produces
catastrophic thread contention for PyMC GP `find_MAP` workloads — a
20-second fit can turn into a multi-hour hang on a 16-core server.  See
[OpenBLAS#3187](https://github.com/OpenMathLib/OpenBLAS/issues/3187)
and [pymc#6640](https://github.com/pymc-devs/pymc/issues/6640) for the
underlying issue.

[`_thread_caps.py`](./_thread_caps.py) sets `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, etc. *before* NumPy / SciPy / PyMC / PyTensor load.
You do not need to export these variables manually.

Controlled via the `PYMC_PARALLEL__COMPUTE_THREADS` environment variable
(the pydantic-settings form of `ExperimentConfig.parallel.compute_threads`):

| Value  | Effect                                                    |
|--------|-----------------------------------------------------------|
| unset  | cap all BLAS / OpenMP vars to 1 (default, recommended).   |
| `N≥0`  | cap to `max(N, 1)` — use `N=4` etc. for large-matrix work.|
| `-1`   | opt-out sentinel; leave env vars untouched.               |

Explicit per-variable shell overrides (e.g.
`OPENBLAS_NUM_THREADS=4 python experiments.py …`) always win, because
`_thread_caps.py` uses `os.environ.setdefault`.

### `_pytensor_compiledir.py` — per-invocation PyTensor compile directory

When multiple concurrent `experiments.py` invocations run on the same
machine (e.g. 6 parallel shell jobs, each with `--parallel.nproc 2`, for
a total of 12 child processes), they all share PyTensor's default
`~/.pytensor/compiledir_<platform>/` directory and contend for its
file lock.  Above ~10 concurrent cold-cache compilations the lock hits
its default 120s timeout and raises:

```
filelock.Timeout: The file lock '…/.lock' could not be acquired.
```

See [pymc#6818](https://github.com/pymc-devs/pymc/issues/6818) and
[Theano#4858](https://github.com/Theano/Theano/issues/4858) for the
underlying issue.

[`_pytensor_compiledir.py`](./_pytensor_compiledir.py) prepends
`base_compiledir=~/.pytensor_jobs/pid_<parent_pid>_boot_<mtime>/` to
`PYTENSOR_FLAGS` before PyTensor imports.  Each top-level invocation
gets its own directory keyed on the parent process PID (so child
processes spawned by `--parallel.nproc` inherit the same compiledir via
`os.environ` and benefit from within-invocation cache reuse).
Concurrent invocations never share a lock.

Controlled via two environment variables:

| Variable                           | Effect                                                            |
|------------------------------------|-------------------------------------------------------------------|
| `PYMC_PYTENSOR_COMPILEDIR_ROOT`    | Root directory for per-invocation subdirs. Default `~/.pytensor_jobs`. |
| `PYTENSOR_FLAGS=base_compiledir=…` | If already set, takes precedence (user override wins).            |

**Disk cost**: each invocation's compiledir holds ~50–500 MB of compiled
`.so` modules.  They accumulate under `~/.pytensor_jobs/` and are safe
to delete between experiment batches:

```bash
rm -rf ~/.pytensor_jobs
```

**Opt-out** (share a single compile cache across all invocations — useful
when running sequentially and you want cache hits across runs):

```bash
PYTENSOR_FLAGS='base_compiledir=~/.pytensor' python experiments.py …
```

## Running experiments

### Single dataset (debugging)

```bash
python experiments.py --domain time-series --pytensor-mode FAST_RUN \
    --data-pkl dataset_ts_no_anomaly_medium.pkl --dataset-idx "0" \
    --max-steps 5 --verbosity 1 \
    --model.id "azure/gpt-5.4-mini" --model.call-timeout 300 \
    --toolkit.mode generate_only \
    --toolkit.code-gen-model "azure/gpt-5.4-mini" \
    --toolkit.force-tool-call \
    --parallel.nproc 0 --parallel.nthread 0 --parallel.max-rpm 120 \
    --output.expt ts_debug
```

### Full parallel sweep (production)

For a dataset with N series on a machine with C cores, run one process
per core with single-threaded BLAS.  Each child inherits
`OPENBLAS_NUM_THREADS=1` from the parent (set by `_thread_caps.py`), so
the C processes fit independent datasets in parallel with zero BLAS
contention:

```bash
python experiments.py --domain time-series --pytensor-mode FAST_RUN \
    --data-pkl dataset_ts_no_anomaly_medium.pkl \
    --max-steps 5 --verbosity 1 \
    --model.id "azure/gpt-5.4-mini" --model.call-timeout 300 \
    --toolkit.mode generate_only \
    --toolkit.code-gen-model "azure/gpt-5.4-mini" \
    --toolkit.force-tool-call \
    --parallel.nproc 16 --parallel.nthread 0 \
    --parallel.max-rpm 1920 \
    --output.expt ts_genonly_forced
```

Key flags:

  * `--parallel.nproc 16` — one child process per core on a 16-core box.
  * `--parallel.nthread 0` — no inner thread pool (sync within each child).
  * `--parallel.max-rpm 1920` — total LLM RPM budget across all children;
    divided automatically by `nproc × max(nthread, 1)` to get per-child RPM.
    Tune to your Azure / OpenAI quota.
  * Drop `--dataset-idx "0"` to process every series in the pkl.

Expected throughput scaling on 16 cores is near-linear (≈14-16× vs
sequential) because each child does single-threaded BLAS, so the `C`
children fit without stepping on each other.

## Reproducing the original BLAS-contention hang

If you want to see the failure mode that motivated the thread-cap work,
`repro_stuck_model.py` runs three exact LLM-generated GP models that
hung in the original report:

```bash
# Default — uses the thread cap (fast, ~100s wall on Linux x86_64).
python repro_stuck_model.py --models 0,1,2 --timeout 900 --pytensor-mode FAST_RUN

# Opt out — reproduces the thread-contention behaviour.
PYMC_PARALLEL__COMPUTE_THREADS=-1 \
    python repro_stuck_model.py --models 0,1,2 --timeout 900 --pytensor-mode FAST_RUN
```

The `[env]` block printed at the top of each run shows which BLAS is
loaded and how many threads each pool is configured for, so you can
visually confirm the cap is in effect.

## Tests

```bash
python -m pytest tests/ -x -q
```
