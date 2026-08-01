"""Compatibility and direct forwarder for orientation."""

import numpy as np

from openptv2.algorithms.orientation import (
    external_calibration as _external_calibration,
)
from openptv2.algorithms.orientation import (
    full_calibration as _full_calibration,
)
from openptv2.algorithms.orientation import (
    match_detection_to_ref as _match_detection_to_ref,
)
from openptv2.algorithms.orientation import (
    multi_cam_point_positions as _multi_cam_point_positions,
)
from openptv2.algorithms.orientation import (
    point_positions as _point_positions,
)
from openptv2.algorithms.orientation import (
    single_cam_point_positions as _single_cam_point_positions,
)
from openptv2.algorithms.orientation import (
    weighted_dumbbell_precision as _weighted_dumbbell_precision,
)


def _is_empty_targets(targets) -> bool:
    """Check if the given targets structure represents an empty set of targets."""
    if targets is None:
        return True
    if hasattr(targets, "shape"):
        return targets.size == 0 or targets.shape[0] == 0
    if len(targets) == 0:
        return True
    total_elements = 0
    for t in targets:
        if hasattr(t, "_targets"):
            total_elements += len(t._targets)
        elif hasattr(t, "__len__"):
            total_elements += len(t)
        else:
            total_elements += 1
    return total_elements == 0


def external_calibration(cal, ref_pts, img_pts, cpar):
    """
    External calibration wrapper.

    Args:
        cal: Calibration instance
        ref_pts: ndarray[n, 3] of reference points
        img_pts: ndarray[n, 2] of image points
        cpar: ControlParams instance

    Returns:
        bool: True if successful
    """
    return _external_calibration(cal, ref_pts, img_pts, cpar)


def full_calibration(cal, ref_pts, img_pts, cpar, flags=None):
    """
    Full calibration wrapper.

    Args:
        cal: Calibration instance
        ref_pts: ndarray[n, 3] of reference points
        img_pts: list of Target or ndarray[n, 2] of image points
        cpar: ControlParams instance
        flags: List of calibration flags

    Returns:
        tuple: (residuals, used_points, error_estimate)
    """
    # Unwrap img_pts if needed.  CRITICAL: filter out unmatched targets
    # (pnr == -999) so that garbage coordinates are NOT fed into the
    # bundle adjustment as valid correspondences.
    if hasattr(img_pts, "__iter__") and len(img_pts) > 0:
        if hasattr(img_pts[0], "_target"):
            # List of Target wrappers — filter by pnr()
            matched_pts = [t for t in img_pts if t.pnr() != -999]
            ref_pts = ref_pts[[t.pnr() for t in matched_pts]]
            img_array = np.array([[t._target.x, t._target.y] for t in matched_pts])
        elif hasattr(img_pts[0], "x") and hasattr(img_pts[0], "y"):
            # Core Target objects — filter by pnr
            matched_pts = [
                t for t in img_pts if (t.pnr() if callable(t.pnr) else t.pnr) != -999
            ]
            ref_pts = ref_pts[
                [t.pnr() if callable(t.pnr) else t.pnr for t in matched_pts]
            ]
            img_array = np.array([[t.x, t.y] for t in matched_pts])
        else:
            # Already an array or list of targets
            img_array = img_pts
    else:
        img_array = img_pts

    return _full_calibration(cal, ref_pts, img_array, cpar, flags)


def match_detection_to_ref(cal, ref_pts, img_pts, cpar, eps=25):
    """
    Match detected points to reference grid.

    Args:
        cal: Calibration instance
        ref_pts: ndarray[n, 3] of reference points
        img_pts: TargetArray or list of targets
        cpar: ControlParams instance
        eps: Matching tolerance

    Returns:
        TargetArray of matched targets
    """
    # Unwrap img_pts if it's a TargetArray
    if hasattr(img_pts, "_targets"):
        targets = img_pts._targets
    elif hasattr(img_pts, "_target_array"):
        targets = img_pts._target_array
    else:
        targets = img_pts

    matched = _match_detection_to_ref(cal, ref_pts, targets, cpar, eps)

    # Wrap result in TargetArray
    from openptv2.tracking_framebuf import TargetArray

    return TargetArray(matched)


def multi_cam_point_positions(targets, cpar, cals):
    """
    Calculate 3D positions from multi-camera targets.

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance
        cals: List of Calibration instances

    Returns:
        tuple: (positions ndarray[n,3], distances ndarray[n])
    """
    if _is_empty_targets(targets):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, "_targets"):
            unwrapped_targets.append(cam_targets._targets)
        elif hasattr(cam_targets, "_target_array"):
            unwrapped_targets.append(cam_targets._target_array)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = list(cals)

    return _multi_cam_point_positions(unwrapped_targets, cpar, unwrapped_cals)


def point_positions(targets, cpar, cals, vpar=None):
    """
    Calculate 3D positions (dispatcher for single/multi cam).

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance
        cals: List of Calibration instances
        vpar: VolumeParams instance, optional

    Returns:
        tuple: (positions ndarray[n,3], distances ndarray[n])
    """
    if _is_empty_targets(targets):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, "_targets"):
            unwrapped_targets.append(cam_targets._targets)
        elif hasattr(cam_targets, "_target_array"):
            unwrapped_targets.append(cam_targets._target_array)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = list(cals)

    # Unwrap volume params
    unwrapped_vpar = vpar if vpar is not None else None

    return _point_positions(unwrapped_targets, cpar, unwrapped_cals, unwrapped_vpar)


def single_cam_point_positions(targets, cpar, cals, vpar):
    """
    Calculate 3D positions for single camera using ray tracing.

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance
        cals: List of Calibration instances
        vpar: VolumeParams instance

    Returns:
        tuple: (positions ndarray[n,3], distances ndarray[n])
    """
    if _is_empty_targets(targets):
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.float64)

    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, "_targets"):
            unwrapped_targets.append(cam_targets._targets)
        elif hasattr(cam_targets, "_target_array"):
            unwrapped_targets.append(cam_targets._target_array)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = list(cals)

    return _single_cam_point_positions(unwrapped_targets, cpar, unwrapped_cals, vpar)


def dumbbell_target_func(targets, cpar, cals, db_length, db_weight):
    """
    Dumbbell target function for optimization.

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance
        cals: List of Calibration instances
        db_length: Expected dumbbell length
        db_weight: Dumbbell weight

    Returns:
        float: Target function value
    """
    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, "_targets"):
            unwrapped_targets.append(cam_targets._targets)
        elif hasattr(cam_targets, "_target_array"):
            unwrapped_targets.append(cam_targets._target_array)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = list(cals)

    # Unwrap cpar
    raw_cpar = cpar

    # Get number of targets (assuming same for all cams)
    num_targs = len(unwrapped_targets) if unwrapped_targets else 0

    return _weighted_dumbbell_precision(
        unwrapped_targets,
        num_targs,
        raw_cpar.get_num_cams(),
        raw_cpar.mm,
        unwrapped_cals,
        db_length,
        db_weight,
    )


__all__ = [
    "dumbbell_target_func",
    "external_calibration",
    "full_calibration",
    "match_detection_to_ref",
    "multi_cam_point_positions",
    "point_positions",
    "single_cam_point_positions",
]
