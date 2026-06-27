"""Compatibility forwarder for tracking_framebuf."""
from openptv2.algorithms.tracking_frame_buf import Frame, Target, TargetArray, read_targets, CORRES_NONE

__all__ = ["Frame", "Target", "TargetArray", "read_targets", "CORRES_NONE"]
