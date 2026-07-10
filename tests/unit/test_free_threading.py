"""Tests specific to free-threaded (no-GIL) Python builds.

All tests are skipped on standard GIL builds — they only run under cp313t /
cp314t (Py_GIL_DISABLED=1).  The normal pytest suite is unaffected.

Three tests:
  1. gil_stays_off       — import the full stack, assert GIL is disabled.
  2. concurrent_determinism — run compiled kernels from N threads on independent
                              inputs, assert results match serial execution.
  3. thread_scaling_report  — perf marker; measures speedup and checks
                              determinism across thread counts.
"""
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

_FT_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

pytestmark = pytest.mark.skipif(
    not _FT_BUILD,
    reason="requires free-threaded Python (cp313t / cp314t, Py_GIL_DISABLED=1)",
)


# ---------------------------------------------------------------------------
# 1. GIL-stays-off regression
# ---------------------------------------------------------------------------
def test_gil_stays_off():
    """Importing the full openptv2 stack must not re-enable the GIL.

    If this fails, a compiled module is missing Py_MOD_GIL_NOT_USED — most
    likely because stale .c files were used during build.  Fix:
        touch src/openptv2/algorithms/*.py
        python setup.py build_ext --inplace
    """
    # Import the hot-path modules to trigger any GIL re-enable side-effects.
    import openptv2.algorithms.track_kernels  # noqa: F401
    import openptv2.algorithms.track_kernels_corr  # noqa: F401
    import openptv2.algorithms.image_processing  # noqa: F401
    import openptv2.algorithms.correspondences  # noqa: F401

    assert hasattr(sys, "_is_gil_enabled"), "sys._is_gil_enabled missing on ft build"
    assert not sys._is_gil_enabled(), (
        "GIL was re-enabled after importing openptv2 modules.  "
        "Stale .c files are the most likely cause — see docs/developer_guide/free_threading.md."
    )


# ---------------------------------------------------------------------------
# 2. Concurrent determinism / race detector
# ---------------------------------------------------------------------------
_IMX, _IMY = 640, 512
_SPAN = 3
_N_TASKS = 8


def _make_inputs(n, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=_IMX * _IMY, dtype=np.uint8) for _ in range(n)]


def test_concurrent_determinism():
    """Compiled kernels must produce identical results under parallel Python threads.

    Runs fast_box_blur on N independent images, once serially and once from N
    concurrent threads, then compares element-wise.  Any shared mutable state
    (scratch buffers, cached module globals, etc.) that is not thread-safe will
    be caught here.
    """
    from openptv2.algorithms.image_processing import fast_box_blur

    inputs = _make_inputs(_N_TASKS)

    # Serial reference
    serial = [fast_box_blur(img, _SPAN, _IMX, _IMY) for img in inputs]

    # Parallel execution
    with ThreadPoolExecutor(max_workers=_N_TASKS) as pool:
        parallel = list(
            pool.map(lambda img: fast_box_blur(img, _SPAN, _IMX, _IMY), inputs)
        )

    for i, (s, p) in enumerate(zip(serial, parallel)):
        np.testing.assert_array_equal(
            s, p, err_msg=f"Result mismatch at task {i} — possible race condition"
        )


# ---------------------------------------------------------------------------
# 3. Thread-scaling speedup (perf marker)
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_thread_scaling_report():
    """Report thread-scaling speedup; assert determinism at each thread count.

    Does not assert a specific speedup — hardware varies.  The test passes as
    long as results are numerically identical across all thread counts.
    Speedup is printed for human inspection in -s / -v mode.
    """
    from openptv2.algorithms.image_processing import fast_box_blur

    inputs = _make_inputs(_N_TASKS)
    serial_ref = [fast_box_blur(img, _SPAN, _IMX, _IMY) for img in inputs]

    def _run_n(n_threads, n_tasks):
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            t0 = time.perf_counter()
            results = list(
                pool.map(
                    lambda img: fast_box_blur(img, _SPAN, _IMX, _IMY),
                    inputs[:n_tasks],
                )
            )
            return time.perf_counter() - t0, results

    # Warm-up
    _run_n(1, 1)

    t1, _ = _run_n(1, 1)
    print(f"\n[free-threading speedup] baseline 1 task/1 thread: {t1*1000:.1f} ms")

    for n in (2, 4, 8):
        wall, results = _run_n(n, n)
        speedup = (n * t1) / wall
        print(
            f"  {n} tasks / {n} threads: {wall*1000:.1f} ms  speedup={speedup:.2f}x"
        )
        for i, r in enumerate(results):
            np.testing.assert_array_equal(
                serial_ref[i], r,
                err_msg=f"Result mismatch at n_threads={n}, task={i}",
            )
