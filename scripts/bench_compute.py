"""I/O-free compiled-hot-path microbench: image filters over a fixed image.

Deterministic input (seeded), no file I/O, times pure Cython compute so the
comparison isolates compiled-code speed across interpreter versions.

Run from repo root:
    python scripts/bench_compute.py
"""
import sys
import sysconfig
import time

sys.path.insert(0, "src")
import numpy as np
from openptv2.algorithms.image_processing import lowpass_3, fast_box_blur

IMX, IMY = 1280, 1024
rng = np.random.default_rng(12345)
img = rng.integers(0, 256, size=IMX * IMY, dtype=np.uint8)


def timeit(fn, reps):
    fn()  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], min(ts)


def main():
    gil = sysconfig.get_config_var("Py_GIL_DISABLED")
    tag = "free-threaded" if gil else "GIL"
    print(f"Python {sys.version.split()[0]}  [{tag}]  image {IMX}x{IMY}")

    med, lo = timeit(lambda: lowpass_3(img, IMX, IMY), 25)
    print(f"lowpass_3     : median={med*1000:.2f}ms  min={lo*1000:.2f}ms")

    med, lo = timeit(lambda: fast_box_blur(img, 3, IMX, IMY), 25)
    print(f"fast_box_blur : median={med*1000:.2f}ms  min={lo*1000:.2f}ms")


if __name__ == "__main__":
    main()
