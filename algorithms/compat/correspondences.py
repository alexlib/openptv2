"""
Correspondences compatibility wrappers providing optv-like API.
"""

import numpy as np
from algorithms.correspondences import (
    correspondences as _correspondences,
    correct_frame,
)
from algorithms.epi import Coord2d
from algorithms.tracking_frame_buf import Frame as AlgoFrame
from algorithms.compat.transforms import distorted_to_flat


class MatchedCoords:
    """
    Wrapper for metric-corrected target coordinates.

    Applies pixel → metric → distortion correction to targets.
    """

    def __init__(self, targets, cpar, cal, tol=0.00001, reset_numbers=True):
        """
        Initialize MatchedCoords.

        Args:
            targets: TargetArray instance
            cpar: ControlParams instance
            cal: Calibration instance
            tol: Tolerance for distortion correction
            reset_numbers: Whether to reset particle numbers (unused)
        """
        self._targets = targets if hasattr(targets, '_targets') else targets
        self._cpar = cpar
        self._cal = cal
        self._tol = tol
        self._corrected = []

        # Apply corrections to each target
        self._apply_corrections()

    def _apply_corrections(self):
        """Apply pixel → metric → distortion corrections."""
        from algorithms.compat.transforms import convert_arr_pixel_to_metric

        # Get target positions
        num_targets = len(self._targets)
        if num_targets == 0:
            return

        # Extract pixel positions
        if hasattr(self._targets, '_targets'):
            # TargetArray
            positions = np.array([
                [self._targets._targets[i].x, self._targets._targets[i].y]
                for i in range(num_targets)
            ])
            pnrs = [self._targets._targets[i].pnr for i in range(num_targets)]
        else:
            # List of Target
            positions = np.array([
                [self._targets[i].x, self._targets[i].y]
                for i in range(num_targets)
            ])
            pnrs = [self._targets[i].pnr for i in range(num_targets)]

        # Pixel → metric
        metric = convert_arr_pixel_to_metric(positions, self._cpar)

        # Metric → flat (distortion correction)
        flat = distorted_to_flat(metric, self._cal, tol=self._tol)

        # Store as Coord2d objects
        self._corrected = [
            Coord2d(x=flat[i, 0], y=flat[i, 1], pnr=pnrs[i])
            for i in range(num_targets)
        ]

    def as_arrays(self):
        """
        Return corrected coordinates as arrays.

        Returns:
            tuple: (positions ndarray[n,2], pnrs ndarray[n])
        """
        if len(self._corrected) == 0:
            return np.empty((0, 2)), np.empty(0, dtype=np.int32)

        pos = np.array([[c.x, c.y] for c in self._corrected])
        pnr = np.array([c.pnr for c in self._corrected], dtype=np.int32)
        return pos, pnr

    def get_by_pnrs(self, pnrs):
        """
        Filter coordinates by particle numbers.

        Args:
            pnrs: List or array of particle numbers to keep

        Returns:
            ndarray[len(pnrs), 2]: Filtered positions
        """
        pos = np.full((len(pnrs), 2), -999.0, dtype=np.float64)
        
        # Build mapping from pnr to coordinate
        pnr_to_coord = {c.pnr: (c.x, c.y) for c in self._corrected}
        
        for i, p in enumerate(pnrs):
            if p in pnr_to_coord:
                pos[i, 0] = pnr_to_coord[p][0]
                pos[i, 1] = pnr_to_coord[p][1]
                
        return pos


def correspondences(img_pts, flat_coords, cals, vparam, cparam):
    """
    Find correspondences between cameras.

    Args:
        img_pts: List of TargetArray (one per camera) - image targets
        flat_coords: List of MatchedCoords (one per camera) - corrected coords
        cals: List of Calibration instances
        vparam: VolumeParams instance
        cparam: ControlParams instance

    Returns:
        tuple: (sorted_pos, sorted_corresp, num_targs)
            - sorted_pos: ndarray[n, num_cams] particle numbers per camera
            - sorted_corresp: ndarray[n, num_cams] correspondence quality
            - num_targs: list of target counts per camera
    """
    num_cams = cparam.get_num_cams()

    # Build Frame object from img_pts
    frame = AlgoFrame(num_cams=num_cams, max_targets=1000)

    # Copy targets to frame
    for cam in range(num_cams):
        if hasattr(img_pts[cam], '_targets'):
            targets = img_pts[cam]._targets
        else:
            targets = img_pts[cam]

        frame.targets[cam] = targets
        frame.num_targets[cam] = len(targets)

    # Extract corrected coordinates
    corrected = [mc._corrected for mc in flat_coords]

    # Unwrap parameters
    unwrapped_cals = [c._cal for c in cals]

    # Call algorithms correspondences
    ntupels, match_counts = _correspondences(
        frame, corrected, vparam._vpar, cparam._cpar, unwrapped_cals
    )

    # Convert NTupel list to optv format
    sorted_pos = [None] * (num_cams - 1)
    sorted_corresp = [None] * (num_cams - 1)
    last_count = 0

    for clique_type in range(num_cams - 1):
        num_points = match_counts[4 - num_cams + clique_type]
        clique_targs = np.full((num_cams, num_points, 2), -999.0, dtype=np.float64)  # PT_UNUSED = -999
        clique_ids = np.full((num_cams, num_points), -1, dtype=np.intp)              # CORRES_NONE = -1

        for cam in range(num_cams):
            for pt in range(num_points):
                geo_id = ntupels[pt + last_count].p[cam]
                if geo_id < 0:
                    continue

                p1 = corrected[cam][geo_id].pnr
                clique_ids[cam, pt] = p1

                if p1 > -1:
                    if hasattr(img_pts[cam], '_targets'):
                        targ = img_pts[cam]._targets[p1]
                    else:
                        targ = img_pts[cam][p1]
                    clique_targs[cam, pt, 0] = targ.x
                    clique_targs[cam, pt, 1] = targ.y

        last_count += num_points
        sorted_pos[clique_type] = clique_targs
        sorted_corresp[clique_type] = clique_ids

    # Return target counts per camera
    num_targs = [frame.num_targets[cam] for cam in range(num_cams)]

    return sorted_pos, sorted_corresp, num_targs


def single_cam_correspondence(img_pts, flat_coords, cals):
    """
    Single camera correspondence (pass-through).

    For single camera, correspondences are trivial (each target corresponds to itself).

    Args:
        img_pts: List with one TargetArray
        flat_coords: List with one MatchedCoords
        cals: List with one Calibration

    Returns:
        tuple: (sorted_pos, sorted_corresp, num_targs)
    """
    # For single camera, each target corresponds to itself
    if hasattr(img_pts[0], '_targets'):
        targets = img_pts[0]._targets
    else:
        targets = img_pts[0]

    num_targets = len(targets)
    sorted_pos = np.arange(num_targets, dtype=np.int32).reshape(-1, 1)
    sorted_corresp = np.ones((num_targets, 1), dtype=np.float64)
    num_targs = [num_targets]

    return sorted_pos, sorted_corresp, num_targs
