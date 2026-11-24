"""Project-wide worker allocation and parallel execution for local and HPC jobs.

Import this module before NumPy/BLAS where possible. Environment defaults protect
fresh worker processes against thread oversubscription without overriding the
user's or job scheduler's explicit limits.
"""

import concurrent.futures
import multiprocessing
import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional, TypeVar


_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_Result = TypeVar("_Result")


def configure_worker_threads() -> None:
    """Set single-thread numerical defaults without overriding explicit limits.

    Notes
    -----
    These settings affect libraries initialized *after* this call. For
    already initialized BLAS pools, configure limits before launching a job.
    """
    for variable in _THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")


configure_worker_threads()


def get_mp_context() -> multiprocessing.context.BaseContext:
    """Select a safe multiprocessing context without using plain ``fork``.

    Returns
    -------
    multiprocessing.context.BaseContext
        The ``forkserver`` context if supported, otherwise ``spawn``.
    """
    methods = multiprocessing.get_all_start_methods()
    method = "forkserver" if "forkserver" in methods else "spawn"
    return multiprocessing.get_context(method)


def _slurm_cpu_limit() -> Optional[int]:
    """Read a valid per-task SLURM CPU allocation when one is specified.

    Returns
    -------
    int or None
        Positive allocation, or None if it is absent or invalid.
    """
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "")
    try:
        count = int(raw)
    except ValueError:
        return None
    return count if count > 0 else None


def _allocated_cpus() -> int:
    """Detect CPUs permitted by the OS, cgroups, and HPC scheduler.

    Returns
    -------
    int
        At least one CPU, never more than any known allocation constraint.

    Notes
    -----
    ``joblib.cpu_count`` consults cgroup CPU quotas on supported platforms.
    Explicit affinity and SLURM limits are additional upper bounds.
    """
    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        from joblib import cpu_count
    except ImportError:
        pass
    else:
        limits.append(cpu_count())
    slurm = _slurm_cpu_limit()
    if slurm is not None:
        limits.append(slurm)
    return max(1, min(limit for limit in limits if limit > 0))


def get_n_cores(reserve_cores: int = 2) -> int:
    """Return the default worker count after optional CPU headroom.

    Parameters
    ----------
    reserve_cores : int, default=2
        CPU capacity reserved for the parent process and other work.

    Returns
    -------
    int
        At least one worker, no greater than the CPU allocation.

    Raises
    ------
    ValueError
        If the reservation is not a non-negative integer.
    """
    if isinstance(reserve_cores, bool) or not isinstance(reserve_cores, int) or reserve_cores < 0:
        raise ValueError("reserve_cores must be a non-negative integer.")
    return max(1, _allocated_cpus() - reserve_cores)


def resolve_n_jobs(n_jobs: Optional[int] = -1) -> int:
    """Clamp requested workers to available CPUs instead of oversubscribing.

    Parameters
    ----------
    n_jobs : int or None, default=-1
        Positive number of requested workers, or negative joblib-style offset
        relative to the *reserved* default. ``None`` and ``-1`` use the
        reserved default. Explicit positive requests can use all allocated
        CPUs, even when the request exceeds the allocation.

    Returns
    -------
    int
        Worker count between one and the available CPU allocation.

    Raises
    ------
    ValueError
        If ``n_jobs`` is zero, a boolean, or not an integer or None.

    Examples
    --------
    With eight allocated CPUs, ``resolve_n_jobs(100)`` returns eight.
    """
    if n_jobs is None:
        return get_n_cores()
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer or None.")
    if n_jobs < 0:
        return max(1, get_n_cores() + n_jobs + 1)
    return min(n_jobs, _allocated_cpus())


def run_parallel(
    worker: Callable[..., _Result],
    calls: Iterable[tuple[tuple[Any, ...], Mapping[str, Any]]],
    *,
    n_jobs: Optional[int] = -1,
) -> list[_Result]:
    """Run ordered, independent worker calls using the central CPU policy.

    Parameters
    ----------
    worker : callable
        Function or bound method run on each independent sample.
    calls : iterable of (tuple, mapping)
        Positional and keyword arguments for each call, lazily generated.
    n_jobs : int or None, default=-1
        Worker request, clamped to the allocated CPUs.

    Returns
    -------
    list
        Results in input order; worker exceptions propagate unchanged.

    Notes
    -----
    Joblib uses its default ``loky`` backend for multiple workers and honors
    a caller's ``parallel_backend`` context (for example, during testing).
    """
    workers = resolve_n_jobs(n_jobs)
    if workers == 1:
        return [worker(*args, **kwargs) for args, kwargs in calls]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=workers)(
        delayed(worker)(*args, **kwargs) for args, kwargs in calls
    )


def create_executor(max_workers: Optional[int] = None) -> concurrent.futures.ProcessPoolExecutor:
    """Create a safe process pool constrained to the CPU allocation.

    Parameters
    ----------
    max_workers : int or None, optional
        Requested worker count; None uses the reserved default.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor
        Executor configured with a safe multiprocessing context.
    """
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=resolve_n_jobs(max_workers),
        mp_context=get_mp_context(),
    )
