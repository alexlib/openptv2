"""Tracking frame buffer module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.tracking_framebuf import Target, TargetArray, Frame, read_targets
    except ImportError:
        from algorithms.compat.tracking_framebuf import (
            Target,
            TargetArray,
            Frame,
            read_targets_compat as read_targets,
        )
else:
    from algorithms.compat.tracking_framebuf import (
        Target,
        TargetArray,
        Frame,
        read_targets_compat as read_targets,
    )

__all__ = ['Target', 'TargetArray', 'Frame', 'read_targets']
