"""Correspondences module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.correspondences import MatchedCoords, correspondences
    except ImportError:
        from algorithms.compat.correspondences import MatchedCoords, correspondences
else:
    from algorithms.compat.correspondences import MatchedCoords, correspondences

__all__ = ['MatchedCoords', 'correspondences']
