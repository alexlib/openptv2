"""
Numba-accelerated Python implementations of OpenPTV algorithms.

This module provides the main entry point for the Python/Numba engine,
re-exporting the algorithm implementations from the algorithms package.

Usage:
    from openptv2.algorithms import numba_impl
    
    # Use the engine
    tracker = numba_impl.Tracker()
    result = tracker.track(targets)
"""

# Re-export main classes and functions from the algorithms package
from . import (
    calibration,
    correspondences,
    epi,
    image_processing,
    imgcoord,
    multimed,
    orientation,
    parameters,
    ray_tracing,
    segmentation,
    sortgrid,
    track,
    tracking_frame_buf,
    tracking_run,
    trafo,
    vec_utils,
    constants,
    find_candidate,
)

# Re-export commonly used classes
from .tracking_frame_buf import Target, TargetArray, Frame
from .track import Tracker
from .calibration import Calibration
from .orientation import Orientation
from .parameters import ControlParams, TrackingParams, SequenceParams, VolumeParams

# Re-export key functions
from .image_processing import detect_targets
from .segmentation import segment_image
from .correspondences import find_correspondences
from .epi import epipolar_line
from .trafo import pixel_to_world, world_to_pixel, pixel_to_metric, metric_to_pixel

__all__ = [
    # Modules
    "calibration",
    "correspondences",
    "epi",
    "image_processing",
    "imgcoord",
    "multimed",
    "orientation",
    "parameters",
    "ray_tracing",
    "segmentation",
    "sortgrid",
    "track",
    "tracking_frame_buf",
    "tracking_run",
    "trafo",
    "vec_utils",
    "constants",
    "find_candidate",
    # Classes
    "Target",
    "TargetArray",
    "Frame",
    "Tracker",
    "Calibration",
    "Orientation",
    "ControlParams",
    "TrackingParams",
    "SequenceParams",
    "VolumeParams",
    # Functions
    "detect_targets",
    "segment_image",
    "find_correspondences",
    "epipolar_line",
    "pixel_to_world",
    "world_to_pixel",
    "pixel_to_metric",
    "metric_to_pixel",
]
