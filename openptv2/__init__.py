"""
openptv2 - Unified OpenPTV: Particle Tracking Velocimetry

This package provides a unified interface to OpenPTV algorithms with
dual-engine support:
- optv: C/Cython engine (default if available, fastest)
- python: Pure Python/Numba engine (for debugging, cloud deployment)

Engine selection via environment variable:
    export OPENPTV_ENGINE=python  # Use Python engine
    export OPENPTV_ENGINE=optv    # Use C/Cython engine (default)

Example usage:
    >>> import openptv2
    >>> from openptv2 import Tracker, Calibration, ControlParams
    >>>
    >>> # Engine auto-detected based on OPENPTV_ENGINE or availability
    >>> cal = Calibration.from_file("cam1.tif.ori", "cam1.tif.addpar")
    >>>
    >>> # Check current engine
    >>> print(openptv2.get_engine())  # 'optv' or 'python'
"""

__version__ = "0.2.0"
__author__ = "OpenPTV Community"
__email__ = "openptv@googlegroups.com"

# Engine selection API
from .engine import get_engine, set_engine, is_optv_available, is_python_available

# Import all modules (engine-aware)
from .calibration import Calibration
from .parameters import (
    ControlParams,
    VolumeParams,
    TrackingParams,
    SequenceParams,
    TargetParams,
    MultimediaParams,
    TrackParTuple,
    convert_track_par_to_tuple,
)
from .parameter_converters import (
    convert_optv_calibrations,
    get_multimedia_par,
    get_control_par,
    get_sequence_par,
    get_volume_par,
    get_track_par_tuple,
    get_target_par,
    get_calibration_par,
    get_orient_par,
    get_multiplanes_par,
    get_examine_par,
    get_pft_version_par,
    get_all_params,
)
from .correspondences import MatchedCoords, correspondences
from .image_processing import preprocess_image
from .segmentation import target_recognition
from .tracking_framebuf import Target, TargetArray, Frame, read_targets
from .tracker import Tracker, default_naming
from .transforms import (
    convert_arr_pixel_to_metric,
    convert_arr_metric_to_pixel,
    correct_arr_brown_affine,
    distort_arr_brown_affine,
    distorted_to_flat,
)
from .imgcoord import image_coordinates, flat_image_coordinates
from .orientation import (
    point_positions,
    external_calibration,
    full_calibration,
    match_detection_to_ref,
    multi_cam_point_positions,
    dumbbell_target_func,
)
from .epipolar import epipolar_curve


def get_version():
    """Return the version string."""
    return __version__


def get_engine_info():
    """
    Return information about available engines.

    Returns:
        dict: Engine availability and status
    """
    info = {
        "default_engine": get_engine(),
        "optv_available": is_optv_available(),
        "python_available": is_python_available(),
    }

    if is_optv_available():
        try:
            import optv
            info["optv_version"] = getattr(optv, "__version__", "unknown")
        except Exception:
            pass

    return info


# Module exports
__all__ = [
    # Engine API
    'get_engine',
    'set_engine',
    'is_optv_available',
    'is_python_available',
    'get_version',
    'get_engine_info',
    # Core classes
    'Calibration',
    'ControlParams',
    'VolumeParams',
    'TrackingParams',
    'SequenceParams',
    'TargetParams',
    'MultimediaParams',
    'Target',
    'TargetArray',
    'Frame',
    'Tracker',
    'MatchedCoords',
    'TrackParTuple',
    # Functions
    'correspondences',
    'preprocess_image',
    'target_recognition',
    'read_targets',
    'default_naming',
    'convert_arr_pixel_to_metric',
    'convert_arr_metric_to_pixel',
    'correct_arr_brown_affine',
    'distort_arr_brown_affine',
    'distorted_to_flat',
    'image_coordinates',
    'flat_image_coordinates',
    'point_positions',
    'external_calibration',
    'full_calibration',
    'match_detection_to_ref',
    'multi_cam_point_positions',
    'dumbbell_target_func',
    'epipolar_curve',
    'convert_track_par_to_tuple',
    # Parameter converters
    'convert_optv_calibrations',
    'get_multimedia_par',
    'get_control_par',
    'get_sequence_par',
    'get_volume_par',
    'get_track_par_tuple',
    'get_target_par',
    'get_calibration_par',
    'get_orient_par',
    'get_multiplanes_par',
    'get_examine_par',
    'get_pft_version_par',
    'get_all_params',
]
