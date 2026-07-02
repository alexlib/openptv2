"""Compatibility forwarder for tracking_framebuf."""
from openptv2.algorithms.tracking_frame_buf import (
    CORRES_NONE,
    Frame,
    Target,
    TargetArray,
    read_targets,
)

__all__ = ["Frame", "Target", "TargetArray", "read_targets", "CORRES_NONE"]
