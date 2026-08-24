"""Public OpenPTV2 API backed by the single Cython-backed algorithms runtime."""

from .algorithms.parameter_converters import (
    convert_optv_calibrations,
    get_all_params,
    get_calibration_par,
    get_control_par,
    get_examine_par,
    get_multimedia_par,
    get_multiplanes_par,
    get_orient_par,
    get_pft_version_par,
    get_sequence_par,
    get_target_par,
    get_track_par_tuple,
    get_volume_par,
)
from .algorithms.parameters import (
    TrackParTuple,
    convert_track_par_to_tuple,
)
from .algorithms.track_kernels import is_compiled as _is_compiled
from .calibration import Calibration
from .correspondences import (
    MatchedCoords,
    correspondences,
    match_correspondences_batch_parallel,
    match_frame_correspondences,
)
from .algorithms.multimed import (
    get_mmf_from_mmlut,
    get_mmf_from_mmlut_batch,
    init_mmlut,
    prepare_mmluts,
)
from .epipolar import epipolar_curve
from .image_processing import preprocess_image
from .imgcoord import flat_image_coordinates, image_coordinates
from .orientation import (
    dumbbell_target_func,
    external_calibration,
    full_calibration,
    match_detection_to_ref,
    multi_cam_point_positions,
    point_positions,
)
from .parameters import (
    ControlParams,
    MultimediaParams,
    SequenceParams,
    TargetParams,
    TrackingParams,
    VolumeParams,
)
from .segmentation import detect_targets_batch_parallel, target_recognition
from .storage import (
    RunStore,
    RunStoreError,
    ZarrFrameStore,
    convert_ascii_to_zarr,
    export_run,
    import_run,
    read_zarr_trajectories,
    seal,
)
from .tracker import Tracker, default_naming
from .tracking_framebuf import Frame, Target, TargetArray, read_targets
from .transforms import (
    convert_arr_metric_to_pixel,
    convert_arr_pixel_to_metric,
    correct_arr_brown_affine,
    distort_arr_brown_affine,
    distorted_to_flat,
)
from .version import __version__

__author__ = "OpenPTV Community"
__email__ = "openptv@googlegroups.com"


def get_version():
    """Return the version string."""
    return __version__


def is_compiled():
    """Return whether the current runtime is using compiled Cython extensions."""
    return _is_compiled()


def get_runtime_info():
    """Return runtime information for the single-engine runtime."""
    return {
        "engine": "cython3-pure-python",
        "compiled": is_compiled(),
        "package": "openptv2",
    }


def get_engine_info():
    """Backward-compatible alias for runtime information."""
    return get_runtime_info()


__all__ = [
    "get_version",
    "is_compiled",
    "get_runtime_info",
    "get_engine_info",
    "Calibration",
    "ControlParams",
    "VolumeParams",
    "TrackingParams",
    "SequenceParams",
    "TargetParams",
    "MultimediaParams",
    "Target",
    "TargetArray",
    "Frame",
    "Tracker",
    "MatchedCoords",
    "TrackParTuple",
    "correspondences",
    "match_frame_correspondences",
    "match_correspondences_batch_parallel",
    "preprocess_image",
    "target_recognition",
    "detect_targets_batch_parallel",
    "read_targets",
    "default_naming",
    "convert_arr_pixel_to_metric",
    "convert_arr_metric_to_pixel",
    "correct_arr_brown_affine",
    "distort_arr_brown_affine",
    "distorted_to_flat",
    "image_coordinates",
    "flat_image_coordinates",
    "point_positions",
    "external_calibration",
    "full_calibration",
    "match_detection_to_ref",
    "multi_cam_point_positions",
    "dumbbell_target_func",
    "epipolar_curve",
    "convert_track_par_to_tuple",
    "convert_optv_calibrations",
    "get_multimedia_par",
    "get_control_par",
    "get_sequence_par",
    "get_volume_par",
    "get_track_par_tuple",
    "get_target_par",
    "get_calibration_par",
    "get_orient_par",
    "get_multiplanes_par",
    "get_examine_par",
    "get_pft_version_par",
    "get_all_params",
    "RunStore",
    "RunStoreError",
    "ZarrFrameStore",
    "convert_ascii_to_zarr",
    "import_run",
    "export_run",
    "read_zarr_trajectories",
    "seal",
]
