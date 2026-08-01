"""Streamlined correspondences matching API."""

import numpy as np


class MatchedCoords:
    """
    Wrapper for metric-corrected target coordinates.

    Applies pixel → metric → distortion correction to targets.
    """

    def __init__(self, targets, cpar, cal, tol=0.00001, reset_numbers=True):
        self._targets = targets._targets if hasattr(targets, "_targets") else targets
        self._cpar = cpar
        self._cal = cal
        self._tol = tol
        self._corrected = []

        if reset_numbers:
            for i in range(len(self._targets)):
                t = self._targets[i]
                if hasattr(t, "set_pnr") and callable(t.set_pnr):
                    t.set_pnr(i)
                else:
                    try:
                        t.pnr = i
                    except AttributeError:
                        pass

        # Apply corrections to each target
        self._apply_corrections()

    def _apply_corrections(self):
        from openptv2.algorithms.epi import Coord2d
        from openptv2.transforms import convert_arr_pixel_to_metric, distorted_to_flat

        num_targets = len(self._targets)
        if num_targets == 0:
            return

        targets_list = self._targets
        if hasattr(self._targets, "_targets"):
            targets_list = self._targets._targets

        positions = []
        pnrs = []
        for i in range(num_targets):
            t = targets_list[i]
            if hasattr(t, "pos") and callable(t.pos):
                pos_val = t.pos()
                x_val, y_val = pos_val[0], pos_val[1]
            else:
                x_val, y_val = t.x, t.y

            if hasattr(t, "pnr"):
                if callable(t.pnr):
                    pnr_val = t.pnr()
                else:
                    pnr_val = t.pnr
            else:
                pnr_val = 0

            positions.append([x_val, y_val])
            pnrs.append(pnr_val)

        positions = np.array(positions)

        # Pixel → metric
        metric = convert_arr_pixel_to_metric(positions, self._cpar)

        # Metric → flat (distortion correction)
        flat = distorted_to_flat(metric, self._cal, tol=self._tol)

        # Store as Coord2d objects and sort by x coordinate (matching C's quicksort_coord2d_x)
        self._corrected = [
            Coord2d(x=flat[i, 0], y=flat[i, 1], pnr=pnrs[i]) for i in range(num_targets)
        ]
        self._corrected.sort(key=lambda c: c.x)

    def __len__(self):
        return len(self._corrected)

    def __getitem__(self, index):
        return self._corrected[index]

    def as_arrays(self):
        if len(self._corrected) == 0:
            return np.empty((0, 2)), np.empty(0, dtype=np.int32)

        pos = np.array([[c.x, c.y] for c in self._corrected])
        pnr = np.array([c.pnr for c in self._corrected], dtype=np.int32)
        return pos, pnr

    def get_by_pnrs(self, pnrs):
        # COORD_UNUSED sentinel expected by point_position triangulation
        # (must match orientation.COORD_UNUSED, not PT_UNUSED=-999).
        from openptv2.algorithms.constants import COORD_UNUSED

        pos = np.full((len(pnrs), 2), COORD_UNUSED, dtype=np.float64)

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
        vparam: VolumePar instance
        cparam: ControlPar instance

    Returns:
        tuple: (sorted_pos, sorted_corresp, num_targs)
    """
    from openptv2.algorithms.correspondences import correspondences as _correspondences
    from openptv2.algorithms.tracking_frame_buf import Frame as AlgoFrame
    from openptv2.algorithms.tracking_frame_buf import Target as AlgoTarget

    raw_cparam = cparam
    raw_vparam = vparam
    raw_cals = cals

    num_cams = (
        raw_cparam.get_num_cams()
        if hasattr(raw_cparam, "get_num_cams")
        else raw_cparam.num_cams
    )

    # Build Frame object from img_pts
    frame = AlgoFrame(num_cams=num_cams, max_targets=1000)

    # Copy targets to frame
    for cam in range(num_cams):
        if hasattr(img_pts[cam], "_targets"):
            targets = img_pts[cam]._targets
        else:
            targets = img_pts[cam]

        converted_targets = []
        for t in targets:
            if not hasattr(t, "n"):
                # Optv/Cython target
                nx, ny = t.count_pixels()[1], t.count_pixels()[2]
                converted_targets.append(
                    AlgoTarget(
                        pnr=t.pnr(),
                        x=t.pos()[0],
                        y=t.pos()[1],
                        n=t.count_pixels()[0],
                        nx=nx,
                        ny=ny,
                        sumg=t.sum_grey_value(),
                        tnr=t.tnr(),
                    )
                )
            else:
                converted_targets.append(t)

        frame.targets[cam] = converted_targets
        frame.num_targets[cam] = len(targets)

    # Extract corrected coordinates
    corrected = [
        mc._corrected if hasattr(mc, "_corrected") else mc for mc in flat_coords
    ]

    # Call algorithms correspondences
    ntupels, match_counts = _correspondences(
        frame, corrected, raw_vparam, raw_cparam, raw_cals
    )

    # Convert NTupel list to optv format
    sorted_pos = [None] * (num_cams - 1)
    sorted_corresp = [None] * (num_cams - 1)
    last_count = 0

    # Build pnr-to-target mapping for each camera to avoid wrong direct indexing on sorted lists
    pnr_to_targ_maps = []
    for cam in range(num_cams):
        if hasattr(img_pts[cam], "_targets"):
            targets = img_pts[cam]._targets
        else:
            targets = img_pts[cam]
        mapping = {}
        for t in targets:
            if hasattr(t, "pnr"):
                p_val = t.pnr() if callable(t.pnr) else t.pnr
            else:
                p_val = 0
            mapping[p_val] = t
        pnr_to_targ_maps.append(mapping)

    for clique_type in range(num_cams - 1):
        num_points = match_counts[4 - num_cams + clique_type]
        clique_targs = np.full(
            (num_cams, num_points, 2), -999.0, dtype=np.float64
        )  # PT_UNUSED = -999
        clique_ids = np.full(
            (num_cams, num_points), -1, dtype=np.intp
        )  # CORRES_NONE = -1

        for cam in range(num_cams):
            for pt in range(num_points):
                geo_id = ntupels[pt + last_count].p[cam]
                if geo_id < 0:
                    continue

                p1 = corrected[cam][geo_id].pnr
                clique_ids[cam, pt] = p1

                if p1 > -1:
                    targ = pnr_to_targ_maps[cam].get(p1)
                    if targ is not None:
                        if hasattr(targ, "pos") and callable(targ.pos):
                            pos_val = targ.pos()
                            x_val, y_val = pos_val[0], pos_val[1]
                        else:
                            x_val, y_val = targ.x, targ.y
                        clique_targs[cam, pt, 0] = x_val
                        clique_targs[cam, pt, 1] = y_val

        last_count += num_points
        sorted_pos[clique_type] = clique_targs
        sorted_corresp[clique_type] = clique_ids

    # Return target counts per camera
    num_targs = [frame.num_targets[cam] for cam in range(num_cams)]

    return sorted_pos, sorted_corresp, num_targs


def single_cam_correspondence(img_pts, flat_coords, cals):
    if hasattr(img_pts[0], "_targets"):
        targets = img_pts[0]._targets
    else:
        targets = img_pts[0]

    num_targets = len(targets)
    sorted_pos = np.arange(num_targets, dtype=np.int32).reshape(-1, 1)
    sorted_corresp = np.ones((num_targets, 1), dtype=np.float64)
    num_targs = [num_targets]

    return sorted_pos, sorted_corresp, num_targs


__all__ = ["MatchedCoords", "correspondences"]
