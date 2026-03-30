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

# Re-export main modules from the algorithms package
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
from .parameters_adapter import (
    ControlParams,
    TrackingParams,
    SequenceParams,
    VolumeParams,
)

# Re-export key functions that exist
from .image_processing import preprocess_image
from .segmentation import target_recognition
from .correspondences import correspondences
from .epi import epipolar_curve
from .trafo import pixel_to_metric, metric_to_pixel

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
    "ControlParams",
    "TrackingParams",
    "SequenceParams",
    "VolumeParams",
    # Functions
    "preprocess_image",
    "target_recognition",
    "correspondences",
    "epipolar_curve",
    "pixel_to_metric",
    "metric_to_pixel",
]
