"""A single reusable thread pool, shared across per-frame calls.

``match_pairs`` used to open a fresh ``ThreadPoolExecutor`` (and tear it
down) on every frame -- real OS thread creation/teardown 25+ times per batch
run, measured at ~22% of wall time in a profiled run. One pool, sized to the
machine and created once, removes that churn: submitting work to it costs
only a queue push.
"""

import atexit
import os
from concurrent.futures import ThreadPoolExecutor

_executor: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    """The process-wide shared executor, created on first use."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        atexit.register(_executor.shutdown, wait=False, cancel_futures=True)
    return _executor
