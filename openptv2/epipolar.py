"""Epipolar geometry module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.epipolar import epipolar_curve
    except ImportError:
        from algorithms.compat.epipolar import epipolar_curve
else:
    from algorithms.compat.epipolar import epipolar_curve

__all__ = ['epipolar_curve']
