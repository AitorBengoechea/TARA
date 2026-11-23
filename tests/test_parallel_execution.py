"""Tests for the project-wide local/HPC parallel execution policy."""

import os
from unittest.mock import patch

from joblib import parallel_backend
import pytest

import parallel_execution as parallel


def _double(number: int) -> int:
    """Return twice an input integer for serializable process tests.

    Parameters
    ----------
    number : int
        Input value.

    Returns
    -------
    int
        Doubled value.
    """
    return number * 2


@pytest.mark.parametrize("requested, expected", [
    (None, 6), (-1, 6), (-2, 5), (-20, 1),
    (1, 1), (2, 2), (8, 8), (10000, 8),
])
def test_worker_count_caps_at_available_allocation(monkeypatch, requested, expected):
    """Never oversubscribe, even for an explicit request above the allocation."""
    monkeypatch.setattr(parallel, "_allocated_cpus", lambda: 8)
    assert parallel.resolve_n_jobs(requested) == expected


@pytest.mark.parametrize("invalid", [0, False, True, 1.5, "4", -1.0])
def test_rejects_invalid_worker_requests(invalid):
    """Invalid worker counts fail predictably before creating joblib workers."""
    with pytest.raises(ValueError, match="n_jobs"):
        parallel.resolve_n_jobs(invalid)


@pytest.mark.parametrize("invalid", [True, False, -1, 0.5, "2"])
def test_rejects_invalid_cpu_reservation(invalid):
    """Reserved headroom must be a nonnegative integer."""
    with pytest.raises(ValueError, match="reserve_cores"):
        parallel.get_n_cores(invalid)


def test_hpc_affinity_joblib_slurm_limits_are_combined(monkeypatch):
    """Take the strictest CPU cap across OS, affinity, quotas, and SLURM."""
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda _: set(range(16)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    with patch("joblib.cpu_count", return_value=12):
        assert parallel._allocated_cpus() == 6
        assert parallel.resolve_n_jobs(128) == 6
        assert parallel.get_n_cores() == 4


def test_cpu_detection_handles_missing_affinity_and_invalid_slurm(monkeypatch):
    """Portable CPU detection never depends on Linux-specific OS APIs."""
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 12)
    monkeypatch.delattr(parallel.os, "sched_getaffinity", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-an-integer")
    with patch("joblib.cpu_count", return_value=7):
        assert parallel._allocated_cpus() == 7
        assert parallel.resolve_n_jobs(999) == 7


def test_single_cpu_allocation_never_returns_zero(monkeypatch):
    """Reserved headroom must not eliminate the only allowed worker."""
    monkeypatch.setattr(parallel, "_allocated_cpus", lambda: 1)
    assert parallel.get_n_cores(reserve_cores=16) == 1
    assert parallel.resolve_n_jobs(200) == 1
    assert parallel.resolve_n_jobs(-4) == 1


def test_thread_defaults_preserve_scheduler_overrides(monkeypatch):
    """The shared policy must not overwrite explicit user or HPC thread limits."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    parallel.configure_worker_threads()
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_safe_context_uses_available_start_method(monkeypatch):
    """Never select plain fork for scientific subprocess execution."""
    monkeypatch.setattr(parallel.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])
    assert parallel.get_mp_context().get_start_method() == "spawn"
    monkeypatch.setattr(parallel.multiprocessing, "get_all_start_methods", lambda: ["fork", "forkserver", "spawn"])
    assert parallel.get_mp_context().get_start_method() == "forkserver"


def test_parallel_results_preserve_order_and_keyword_arguments(monkeypatch):
    """Oversized worker requests use the allocation and preserve result order."""
    monkeypatch.setattr(parallel, "_allocated_cpus", lambda: 2)
    with parallel_backend("threading"):
        result = parallel.run_parallel(
            pow, (((number, 2), {}) for number in [4, 1, 3, 2]), n_jobs=10000,
        )
    assert result == [16, 1, 9, 4]


def test_sequential_runner_propagates_errors_and_processes_generators(monkeypatch):
    """Single-worker execution stays lazy and propagates backend exceptions."""
    monkeypatch.setattr(parallel, "_allocated_cpus", lambda: 1)
    assert parallel.run_parallel(_double, (((n,), {}) for n in range(4)), n_jobs=50) == [0, 2, 4, 6]
    with pytest.raises(ZeroDivisionError):
        parallel.run_parallel(lambda: 1 / 0, [((), {})], n_jobs=1)


def test_executor_clamps_worker_request(monkeypatch):
    """ProcessPoolExecutor uses the same worker cap as joblib calls."""
    monkeypatch.setattr(parallel, "_allocated_cpus", lambda: 3)
    with patch.object(parallel.concurrent.futures, "ProcessPoolExecutor") as executor:
        parallel.create_executor(max_workers=100)
    assert executor.call_args.kwargs["max_workers"] == 3
    assert executor.call_args.kwargs["mp_context"].get_start_method() != "fork"
