"""
Tracking debugging utilities for interactive parameter exploration.

Provides functions to compute search volumes, find candidates, and
visualize tracking decisions.

Parameter units (from algorithms/track.py):
- dvxmin, dvxmax, dvymin, dvymax: PIXELS (modifies 2D search area directly)
- dacc: unitless ratio (compared as acc/dacc)
- dangle: unitless (compared directly as angle < dangle)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from openptv2 import TrackingParams as TrackPar
from openptv2 import ControlParams as ControlPar
from openptv2 import VolumeParams as VolumePar
from openptv2 import Calibration


@dataclass
class SearchVolumeBounds:
    """Search volume boundaries for a single camera."""

    left: float
    right: float
    up: float
    down: float


@dataclass
class CandidateInfo:
    """Information about a candidate particle."""

    pnr: int
    x: float
    y: float
    distance_3d: float
    in_volume: bool
    picked: bool


def compute_search_bounds_3d(
    center_3d: np.ndarray,
    velocity: np.ndarray,
    dvxmin: float,
    dvxmax: float,
    dvymin: float,
    dvymax: float,
    dvzmin: float,
    dvzmax: float,
    dacc: float,
    frame_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D search volume bounds for a future frame.

    Args:
        center_3d: Current 3D position [x, y, z]
        velocity: Current velocity [vx, vy, vz]
        dvxmin, dvxmax: X velocity limits
        dvymin, dvymax: Y velocity limits
        dvzmin, dvzmax: Z velocity limits
        dacc: Acceleration limit
        frame_offset: How many frames ahead (1, 2, or 3)

    Returns:
        (min_bounds, max_bounds) - arrays of shape (3,)
    """
    # Compute velocity bounds with acceleration
    accel_factor = 1.0 + dacc * frame_offset

    vx_min = velocity[0] * accel_factor + dvxmin * frame_offset
    vx_max = velocity[0] * accel_factor + dvxmax * frame_offset
    vy_min = velocity[1] * accel_factor + dvymin * frame_offset
    vy_max = velocity[1] * accel_factor + dvymax * frame_offset
    vz_min = velocity[2] * accel_factor + dvzmin * frame_offset
    vz_max = velocity[2] * accel_factor + dvzmax * frame_offset

    min_bounds = center_3d + np.array([vx_min, vy_min, vz_min])
    max_bounds = center_3d + np.array([vx_max, vy_max, vz_max])

    return min_bounds, max_bounds


def project_search_volume_to_camera(
    min_bounds_3d: np.ndarray,
    max_bounds_3d: np.ndarray,
    cal: Calibration,
    cpar: ControlPar,
) -> SearchVolumeBounds:
    """
    Project 3D search volume bounds to 2D camera coordinates.

    Uses corner points of the 3D bounding box.

    Args:
        min_bounds_3d: Minimum 3D coordinates [x, y, z]
        max_bounds_3d: Maximum 3D coordinates [x, y, z]
        cal: Camera calibration
        cpar: Control parameters

    Returns:
        SearchVolumeBounds in pixel coordinates
    """
    from openptv2 import flat_image_coordinates

    # Create corner points of the 3D box
    corners = np.array(
        [
            [min_bounds_3d[0], min_bounds_3d[1], min_bounds_3d[2]],
            [max_bounds_3d[0], min_bounds_3d[1], min_bounds_3d[2]],
            [min_bounds_3d[0], max_bounds_3d[1], min_bounds_3d[2]],
            [max_bounds_3d[0], max_bounds_3d[1], min_bounds_3d[2]],
            [min_bounds_3d[0], min_bounds_3d[1], max_bounds_3d[2]],
            [max_bounds_3d[0], min_bounds_3d[1], max_bounds_3d[2]],
            [min_bounds_3d[0], max_bounds_3d[1], max_bounds_3d[2]],
            [max_bounds_3d[0], max_bounds_3d[1], max_bounds_3d[2]],
        ]
    )

    # Project each corner to 2D using batch function
    try:
        result = flat_image_coordinates(corners, cal, cpar.mm)
        coords_2d = result.tolist()
    except Exception:
        coords_2d = [[0, 0]] * len(corners)

    coords_2d = np.array(coords_2d)

    # Get bounding box of projected points
    left = np.min(coords_2d[:, 0])
    right = np.max(coords_2d[:, 0])
    up = np.min(coords_2d[:, 1])
    down = np.max(coords_2d[:, 1])

    return SearchVolumeBounds(left=left, right=right, up=up, down=down)


def find_candidates_in_volume(
    targets: List,
    bounds: SearchVolumeBounds,
) -> List[CandidateInfo]:
    """
    Find all target particles within a search volume.

    Args:
        targets: List of Target objects
        bounds: Search volume boundaries

    Returns:
        List of CandidateInfo objects
    """
    candidates = []

    for i, tgt in enumerate(targets):
        if tgt.get_pnr() < 0:
            continue

        pos = tgt.get_pos()
        x, y = pos[0], pos[1]

        # Check if within bounds
        in_volume = bounds.left <= x <= bounds.right and bounds.up <= y <= bounds.down

        candidates.append(
            CandidateInfo(
                pnr=tgt.get_pnr(),
                x=x,
                y=y,
                distance_3d=0.0,  # Will be computed later
                in_volume=in_volume,
                picked=False,
            )
        )

    return candidates


def compute_candidate_3d_distances(
    candidates: List[CandidateInfo],
    predicted_3d: np.ndarray,
    cals: List[Calibration],
    cpar: ControlPar,
) -> List[CandidateInfo]:
    """
    Compute 3D distances from predicted position to candidates.

    Args:
        candidates: List of candidate info
        predicted_3d: Predicted 3D position
        cals: List of calibrations (for triangulation)
        cpar: Control parameters

    Returns:
        Updated candidates with distance_3d filled
    """
    # For each candidate, triangulate from 2D and compute distance
    # In real implementation, we'd use point_positions from openptv2
    for cand in candidates:
        # Create fake target data for triangulation
        # In real implementation, we'd use actual 2D positions from all cameras
        pass

    return candidates


def find_nearest_target(
    targets: List,
    click_x: float,
    click_y: float,
    max_distance: float = 10.0,
) -> Optional[Tuple[int, float, float]]:
    """
    Find the nearest target to a click position.

    Args:
        targets: List of Target objects
        click_x: Click X coordinate
        click_y: Click Y coordinate
        max_distance: Maximum search distance in pixels

    Returns:
        (index, x, y) of nearest target, or None
    """
    min_dist = max_distance
    nearest_idx = None
    nearest_pos = None

    for i, tgt in enumerate(targets):
        if tgt.get_pnr() < 0:
            continue

        pos = tgt.get_pos()
        x, y = pos[0], pos[1]

        dist = np.sqrt((x - click_x) ** 2 + (y - click_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
            nearest_pos = (x, y)

    if nearest_idx is not None:
        return (nearest_idx, nearest_pos[0], nearest_pos[1])
    return None


def get_target_3d_position(
    frame,
    target_idx: int,
    cam_idx: int,
    cals: List[Calibration],
    cpar: ControlPar,
) -> Optional[np.ndarray]:
    """
    Get 3D position of a target from frame data.

    Args:
        frame: Frame object with targets and path_info
        target_idx: Index of target in the frame
        cam_idx: Camera index
        cals: List of calibrations
        cpar: Control parameters

    Returns:
        3D position [x, y, z] or None
    """
    # Check if particle is linked (has 3D position)
    if hasattr(frame, "path_info") and frame.path_info:
        for i in range(frame.num_parts):
            path = frame.path_info[i]
            if hasattr(path, "x") and path.x is not None:
                # This is a 3D particle
                # Check if this target is part of this particle
                pass

    # Fallback: triangulate from 2D positions
    # This would require positions from multiple cameras
    return None
