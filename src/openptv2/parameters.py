"""Compatibility forwarder for parameters."""
from openptv2.algorithms.compat.parameters import (
    ControlParams,
    MultimediaParams,
    SequenceParams,
    TargetParams,
    TrackingParams,
    VolumeParams,
)
from openptv2.algorithms.parameter_converters import (
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
from openptv2.algorithms.parameters import (
    TrackParTuple,
    convert_track_par_to_tuple,
)

__all__ = [
    "ControlParams",
    "MultimediaParams",
    "SequenceParams",
    "TargetParams",
    "TrackingParams",
    "VolumeParams",
    "convert_optv_calibrations",
    "get_all_params",
    "get_calibration_par",
    "get_control_par",
    "get_examine_par",
    "get_multimedia_par",
    "get_multiplanes_par",
    "get_orient_par",
    "get_pft_version_par",
    "get_sequence_par",
    "get_target_par",
    "get_track_par_tuple",
    "get_volume_par",
    "TrackParTuple",
    "convert_track_par_to_tuple",
]
