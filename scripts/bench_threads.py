"""Thread-scaling benchmark: run a compiled compute unit across N threads.

On GIL Python, CPU-bound threads serialize → wall time grows with N.
On free-threaded Python, they run in parallel → wall time stays flat.
Speedup = (N * single-thread time) / (N-thread wall time).

Run from repo root:
    python scripts/bench_threads.py
"""
import sys
import time
import sysconfig
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
import numpy as np
from openptv2.algorithms.image_processing import fast_box_blur

IMX, IMY = 1280, 1024
rng = np.random.default_rng(7)
img = rng.integers(0, 256, size=IMX * IMY, dtype=np.uint8)
SPAN = 3


def work():
    # ~24 ms of pure compiled compute, no I/O
    fast_box_blur(img, SPAN, IMX, IMY)


def run_n(nthreads, tasks):
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        t0 = time.perf_counter()
        list(ex.map(lambda _: work(), range(tasks)))
        return time.perf_counter() - t0


def main():
    gil = sysconfig.get_config_var("Py_GIL_DISABLED")
    runtime_gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    tag = "free-threaded" if gil else "GIL-build"
    print(f"Python {sys.version.split()[0]}  [{tag}]  runtime GIL enabled={runtime_gil}")

    work()
    work()  # warm

    # single-thread time per task
    t1 = min(run_n(1, 1) for _ in range(5))
    print(f"1 task , 1 thread : {t1*1000:6.1f}ms  (baseline unit)")

    for n in (2, 4, 8):
        wall = min(run_n(n, n) for _ in range(3))
        speedup = (n * t1) / wall
        print(
            f"{n} tasks, {n} threads: {wall*1000:6.1f}ms  "
            f"speedup={speedup:.2f}x  (ideal {n}x)"
        )


if __name__ == "__main__":
    main()
