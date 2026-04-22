"""Tracker module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.tracker import Tracker, default_naming
    except ImportError:
        from algorithms.compat.tracker import Tracker, default_naming
else:
    from algorithms.compat.tracker import Tracker, default_naming

__all__ = ['Tracker', 'default_naming']
