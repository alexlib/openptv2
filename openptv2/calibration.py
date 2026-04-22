"""Calibration module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.calibration import Calibration
    except ImportError:
        from algorithms.compat.calibration import Calibration
else:
    from algorithms.compat.calibration import Calibration

__all__ = ['Calibration']
