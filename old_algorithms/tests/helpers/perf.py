"""Performance helper utilities for algorithms tests."""

from __future__ import annotations

import time
from statistics import median
from typing import Callable


def measure_seconds(func: Callable[[], object], repeat: int = 5, warmup: int = 1) -> float:
    """Measure median wall-time in seconds.

    The callable is executed `warmup` times first to avoid first-run overhead
    (for example Numba compilation effects), then timed `repeat` times.
    """
    for _ in range(max(0, warmup)):
        func()

    samples = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        func()
        samples.append(time.perf_counter() - start)

    return float(median(samples))
