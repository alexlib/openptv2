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
_creator_pid: int | None = None


def get_executor() -> ThreadPoolExecutor:
    """The process-wide shared executor, created on first use.

    Recreated whenever the current PID doesn't match the one that created
    it: pyptv_batch_parallel forks worker processes on Unix, and a fork
    only clones the calling thread, so a child inheriting an
    already-initialized executor would have a pool object whose worker
    threads never actually exist in that process -- submissions there
    would have no consumer and hang forever.
    """
    global _executor, _creator_pid
    pid = os.getpid()
    if _executor is None or _creator_pid != pid:
        _executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        _creator_pid = pid
        atexit.register(_executor.shutdown, wait=False, cancel_futures=True)
    return _executor
