"""Orientation module with engine-aware dispatch."""

from openptv2.engine import get_engine

_engine = get_engine()

if _engine == "optv":
    try:
        from optv.orientation import (
            point_positions,
            external_calibration,
            full_calibration,
            match_detection_to_ref,
            multi_cam_point_positions,
            dumbbell_target_func,
        )
    except ImportError:
        from algorithms.compat.orientation import (
            point_positions,
            external_calibration,
            full_calibration,
            match_detection_to_ref,
            multi_cam_point_positions,
            dumbbell_target_func,
        )
else:
    from algorithms.compat.orientation import (
        point_positions,
        external_calibration,
        full_calibration,
        match_detection_to_ref,
        multi_cam_point_positions,
        dumbbell_target_func,
    )

__all__ = [
    'point_positions',
    'external_calibration',
    'full_calibration',
    'match_detection_to_ref',
    'multi_cam_point_positions',
    'dumbbell_target_func',
]
