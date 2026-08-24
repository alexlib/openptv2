"""Streamlined correspondences matching API."""

import os
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


def match_frame_correspondences(detections, cpar, cals, vpar):
    """Compute multi-camera stereo correspondences and triangulated 3D positions
    for a single frame.

    Args:
        detections: List of TargetArray / list of Target per camera.
        cpar: ControlParams / ControlPar instance.
        cals: List of Calibration instances.
        vpar: VolumeParams / VolumePar instance.

    Returns:
        tuple: (pos_3d, cam_target_ids)
            pos_3d: (N, 3) ndarray of reconstructed 3D particle coordinates in mm.
            cam_target_ids: (N, max(4, num_cams)) ndarray of corresponding target indices (-1 = not seen).
    """
    from openptv2.orientation import point_positions

    num_cams = len(cals)
    corrected = []
    wrapped_detections = []

    for i_cam in range(num_cams):
        targs = detections[i_cam]
        if hasattr(targs, "sort_y") and callable(targs.sort_y):
            if len(targs) > 0:
                targs.sort_y()
        wrapped_detections.append(targs)
        mc = MatchedCoords(targs, cpar, cals[i_cam])
        corrected.append(mc)

    sorted_pos, sorted_corresp, _ = correspondences(
        wrapped_detections, corrected, cals, vpar, cpar
    )

    total_matches = sum(s.shape[1] for s in sorted_pos)
    if total_matches == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, max(4, num_cams)), dtype=np.int32)

    sorted_pos = np.concatenate(sorted_pos, axis=1)
    sorted_corresp = np.concatenate(sorted_corresp, axis=1)

    flat = np.array(
        [
            corr.get_by_pnrs(corresp)
            for corr, corresp in zip(corrected, sorted_corresp)
        ]
    )

    pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

    if num_cams < 4:
        print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]), dtype=np.int32)
        print_corresp[:num_cams, :] = sorted_corresp
    else:
        print_corresp = sorted_corresp.astype(np.int32)

    return pos, print_corresp.T


def _correspondence_worker_chunk(frames, targets_data, cpar, cals, vpar, zarr_store_path=None):
    """Worker function to process a chunk of frames in parallel."""
    results = []
    num_cams = len(cals)

    store = None
    if zarr_store_path is not None and targets_data is None:
        from openptv2.storage import RunStore
        store = RunStore(zarr_store_path, mode="r")

    for frame in frames:
        if targets_data is not None:
            detections = targets_data[frame]
        elif store is not None:
            detections = [store.read_targets(c, frame) for c in range(num_cams)]
        else:
            raise ValueError("Either targets_data or zarr_store_path must be provided")

        pos_3d, cam_target_ids = match_frame_correspondences(detections, cpar, cals, vpar)
        results.append((frame, pos_3d, cam_target_ids))

    return results


def match_correspondences_batch_parallel(
    frames,
    cpar,
    cals,
    vpar,
    targets=None,
    zarr_store_path=None,
    n_workers=None,
    write_to_store=True,
):
    """Process stereo correspondences and 3D point positions for a batch/sequence of frames
    in parallel using multi-processing.

    Args:
        frames: Sequence or iterable of frame integers (e.g. range(10001, 10005)).
        cpar: ControlParams instance.
        cals: List of Calibration instances.
        vpar: VolumeParams instance.
        targets: Optional dict mapping frame -> list of TargetArray/targets per camera.
        zarr_store_path: Optional path to Zarr store (res/run.zarr) to read targets from
            and/or write correspondences to.
        n_workers: Number of parallel worker processes. If None, auto-selects based on CPU count.
        write_to_store: If True and zarr_store_path is given, writes all correspondences into the Zarr store.

    Returns:
        dict: Mapping frame -> (pos_3d, cam_target_ids)
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    frames_list = list(frames)
    if not frames_list:
        return {}

    num_frames = len(frames_list)
    cpu_count = os.cpu_count() or 1
    if n_workers is None:
        n_workers = min(cpu_count, num_frames)
    else:
        n_workers = max(1, min(n_workers, num_frames))

    results = {}

    # If single worker or single frame, run directly without multiprocessing overhead
    if n_workers <= 1:
        chunk_results = _correspondence_worker_chunk(
            frames_list, targets, cpar, cals, vpar, zarr_store_path
        )
        for frame, pos_3d, cam_target_ids in chunk_results:
            results[frame] = (pos_3d, cam_target_ids)
    else:
        # Split frames into roughly equal contiguous chunks
        chunk_size = (num_frames + n_workers - 1) // n_workers
        chunks = [
            frames_list[i : i + chunk_size]
            for i in range(0, num_frames, chunk_size)
        ]

        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp_ctx) as executor:
            futures = []
            for chunk in chunks:
                chunk_targets = (
                    {f: targets[f] for f in chunk} if targets is not None else None
                )
                fut = executor.submit(
                    _correspondence_worker_chunk,
                    chunk,
                    chunk_targets,
                    cpar,
                    cals,
                    vpar,
                    zarr_store_path,
                )
                futures.append(fut)

            for fut in futures:
                chunk_results = fut.result()
                for frame, pos_3d, cam_target_ids in chunk_results:
                    results[frame] = (pos_3d, cam_target_ids)

    # Write to store if requested
    if write_to_store and zarr_store_path is not None:
        from openptv2.storage import RunStore
        store = RunStore(zarr_store_path, mode="a")
        for frame in frames_list:
            if frame in results:
                pos_3d, cam_target_ids = results[frame]
                store.write_correspondences(frame, pos_3d, cam_target_ids)

    return results


__all__ = [
    "MatchedCoords",
    "correspondences",
    "match_frame_correspondences",
    "match_correspondences_batch_parallel",
]
