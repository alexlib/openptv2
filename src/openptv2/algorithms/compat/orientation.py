"""
Orientation compatibility wrappers providing optv-like API.

Re-exports orientation functions with wrapper conversions for compat objects.
"""

import numpy as np
from openptv2.algorithms.orientation import (
    external_calibration as _external_calibration,
    full_calibration as _full_calibration,
    match_detection_to_ref as _match_detection_to_ref,
    multi_cam_point_positions as _multi_cam_point_positions,
    point_positions as _point_positions,
    single_cam_point_positions as _single_cam_point_positions,
    weighted_dumbbell_precision as _weighted_dumbbell_precision,
)


def external_calibration(cal, ref_pts, img_pts, cpar):
    """
    External calibration wrapper.

    Args:
        cal: Calibration instance (compat wrapper)
        ref_pts: ndarray[n, 3] of reference points
        img_pts: ndarray[n, 2] of image points
        cpar: ControlParams instance (compat wrapper)

    Returns:
        bool: True if successful
    """
    return _external_calibration(cal._cal, ref_pts, img_pts, cpar._cpar)


def full_calibration(cal, ref_pts, img_pts, cpar, flags=None):
    """
    Full calibration wrapper.

    Args:
        cal: Calibration instance (compat wrapper)
        ref_pts: ndarray[n, 3] of reference points
        img_pts: list of Target or ndarray[n, 2] of image points
        cpar: ControlParams instance (compat wrapper)
        flags: List of calibration flags

    Returns:
        tuple: (residuals, used_points, error_estimate)
    """
    # Unwrap img_pts if needed
    if hasattr(img_pts, '__iter__') and len(img_pts) > 0:
        if hasattr(img_pts[0], '_target'):
            # List of Target wrappers
            img_array = np.array([[t._target.x, t._target.y] for t in img_pts])
        else:
            # Already an array or list of targets
            img_array = img_pts
    else:
        img_array = img_pts

    return _full_calibration(cal._cal, ref_pts, img_array, cpar._cpar, flags)


def match_detection_to_ref(cal, ref_pts, img_pts, cpar, eps=25):
    """
    Match detected points to reference grid.

    Args:
        cal: Calibration instance (compat wrapper)
        ref_pts: ndarray[n, 3] of reference points
        img_pts: TargetArray or list of targets
        cpar: ControlParams instance (compat wrapper)
        eps: Matching tolerance

    Returns:
        list of matched targets
    """
    # Unwrap img_pts if it's a TargetArray
    if hasattr(img_pts, '_targets'):
        targets = img_pts._targets
    else:
        targets = img_pts

    matched = _match_detection_to_ref(cal._cal, ref_pts, targets, cpar._cpar, eps)

    # Wrap result in TargetArray
    from openptv2.algorithms.compat.tracking_framebuf import TargetArray
    return TargetArray(matched)


def multi_cam_point_positions(targets, cpar, cals):
    """
    Calculate 3D positions from multi-camera targets.

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance (compat wrapper)
        cals: List of Calibration instances (compat wrappers)

    Returns:
        tuple: (positions ndarray[n,3], distances ndarray[n])
    """
    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, '_targets'):
            unwrapped_targets.append(cam_targets._targets)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = [c._cal for c in cals]

    return _multi_cam_point_positions(unwrapped_targets, cpar._cpar, unwrapped_cals)


def point_positions(targets, cpar, cals, vpar=None):
    """
    Calculate 3D positions (dispatcher for single/multi cam).

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance (compat wrapper)
        cals: List of Calibration instances (compat wrappers)
        vpar: VolumeParams instance (compat wrapper, optional)

    Returns:
        tuple: (positions ndarray[n,3], distances ndarray[n])
    """
    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, '_targets'):
            unwrapped_targets.append(cam_targets._targets)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = [c._cal for c in cals]

    # Unwrap volume params
    unwrapped_vpar = vpar._vpar if vpar is not None else None

    print(f"DEBUG point_positions:")
    print(f"  type(targets): {type(targets)}")
    if hasattr(targets, "shape"):
        print(f"  targets.shape: {targets.shape}")
    print(f"  len(unwrapped_targets): {len(unwrapped_targets)}")
    if len(unwrapped_targets) > 0:
        print(f"  type(unwrapped_targets[0]): {type(unwrapped_targets[0])}")
        if hasattr(unwrapped_targets[0], "shape"):
            print(f"  unwrapped_targets[0].shape: {unwrapped_targets[0].shape}")
    print(f"  len(cals): {len(cals)}")
    print(f"  len(unwrapped_cals): {len(unwrapped_cals)}")

    return _point_positions(unwrapped_targets, cpar._cpar, unwrapped_cals, unwrapped_vpar)


def single_cam_point_positions(targets, cpar, cals, vpar):
    """
    Calculate 3D positions for single camera using ray tracing.

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance (compat wrapper)
        cals: List of Calibration instances (compat wrappers)
        vpar: VolumeParams instance (compat wrapper)

    Returns:
        tuple: (positions ndarray[n,3], distances ndarray[n])
    """
    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, '_targets'):
            unwrapped_targets.append(cam_targets._targets)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = [c._cal for c in cals]

    return _single_cam_point_positions(unwrapped_targets, cpar._cpar, unwrapped_cals, vpar._vpar)


def dumbbell_target_func(targets, cpar, cals, db_length, db_weight):
    """
    Dumbbell target function for optimization.

    Args:
        targets: List of target arrays per camera
        cpar: ControlParams instance (compat wrapper)
        cals: List of Calibration instances (compat wrappers)
        db_length: Expected dumbbell length
        db_weight: Dumbbell weight

    Returns:
        float: Target function value
    """
    # Unwrap targets if they're TargetArrays
    unwrapped_targets = []
    for cam_targets in targets:
        if hasattr(cam_targets, '_targets'):
            unwrapped_targets.append(cam_targets._targets)
        else:
            unwrapped_targets.append(cam_targets)

    # Unwrap calibrations
    unwrapped_cals = [c._cal for c in cals]

    # Get number of targets (assuming same for all cams)
    num_targs = len(unwrapped_targets[0]) if unwrapped_targets else 0

    return _weighted_dumbbell_precision(
        unwrapped_targets, num_targs, cpar.get_num_cams(),
        cpar._cpar.mm, unwrapped_cals, db_length, db_weight
    )
