"""Compatibility parameter API.

The ``*Params`` names are thin aliases for the canonical ``*Par`` classes in
:mod:`openptv2.algorithms.parameters`. They used to be delegating wrappers with
an optv-compatible ``get_*``/``set_*``/``read_*_par`` surface; that surface now
lives directly on the ``*Par`` classes, so the wrappers were removed and these
names are plain aliases kept for backward compatibility (public API and
``openptv2.__init__`` exports).

The parameter-file converter helpers (``get_control_par`` etc.) are re-exported
here unchanged so ``openptv2.parameters.get_control_par`` keeps resolving.
"""

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
    ControlPar as ControlParams,
)
from openptv2.algorithms.parameters import (
    MmNp as MultimediaParams,
)
from openptv2.algorithms.parameters import (
    SequencePar as SequenceParams,
)
from openptv2.algorithms.parameters import (
    TargetPar as TargetParams,
)
from openptv2.algorithms.parameters import (
    TrackPar as TrackingParams,
)
from openptv2.algorithms.parameters import (
    TrackParTuple,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.parameters import (
    VolumePar as VolumeParams,
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
