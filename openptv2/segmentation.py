"""Segmentation module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.segmentation import target_recognition
    except ImportError:
        from algorithms.compat.segmentation import target_recognition
else:
    from algorithms.compat.segmentation import target_recognition

__all__ = ['target_recognition']
