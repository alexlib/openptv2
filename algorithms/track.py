"""Tracking algorithm."""

import cProfile
import io
import math
import os
import pstats
import time
# from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from numba import float64, njit, types

from .calibration import Calibration
from .constants import (
    ADD_PART,
    COORD_UNUSED,
    CORRES_NONE,
    MAX_CANDS,
    MAX_TARGETS,
    NEXT_NONE,
    POS_INF,
    PREV_NONE,
    PT_UNUSED,
    TR_BUFSPACE,
    TR_MAX_CAMS,
    TR_UNUSED,
)
from .imgcoord import img_coord
from .orientation import point_position
from .parameters import (
    ControlPar,
    SequencePar,
    TrackParTuple,
    VolumePar,
    convert_track_par_to_tuple,
)
from .multimed import fast_point_to_pixel
from .tracking_frame_buf import Frame, Pathinfo, Target
from .tracking_run import TrackingRun
from .trafo import dist_to_flat, metric_to_pixel, pixel_to_metric
from .vec_utils import vec_copy, vec_diff_norm, vec_subt

default_naming = {
    "corres": "res/rt_is",
    "linkage": "res/ptv_is",
    "prio": "res/added",
}


def _tracker_env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value not in ("", "0", "false", "False", "no", "off")


def _tracker_debug_enabled() -> bool:
    return _tracker_env_flag("OPENPTV_TRACK_DEBUG")


def _tracker_profile_enabled() -> bool:
    return _tracker_env_flag("OPENPTV_TRACK_PROFILE")


def _tracker_log(tag: str, step: int, message: str) -> None:
    print(f"[tracker:{tag} step={step}] {message}")


def _tracker_print_profile(
    profile: cProfile.Profile, tag: str, step: int, limit: int = 12
) -> None:
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(limit)
    print(f"[tracker:{tag} step={step}] cProfile top {limit}")
    print(stream.getvalue().rstrip())


def _target_search_arrays(next_frame: List[Target], num_targets: int):
    """Extract contiguous arrays for the candidate search kernel."""
    target_x = np.empty(num_targets, dtype=np.float64)
    target_y = np.empty(num_targets, dtype=np.float64)
    target_tnr = np.empty(num_targets, dtype=np.int32)

    for i in range(num_targets):
        targ = next_frame[i]
        target_x[i] = targ.x
        target_y[i] = targ.y
        target_tnr[i] = targ.tnr

    return target_x, target_y, target_tnr


# @dataclass
# class Foundpix:
#     """A Foundpix object holds the parameters for a found pixel."""

#     ftnr: int = TR_UNUSED
#     freq: int = 0
#     whichcam: List[int] = field(default_factory=list)

#     def __post_init__(self):
#         self.whichcam = [0] * TR_MAX_CAMS


Foundpix_dtype = np.dtype(
    [("ftnr", np.int32), ("freq", np.int32), ("whichcam", np.int32, (TR_MAX_CAMS,))]
)


class FoundpixResult:
    """Lightweight SoA container for sorted candidate results.

    Attributes
    ----------
    ftnr : int32[n]  – target numbers (TR_UNUSED for empty slots)
    freq : int32[n]  – frequency counts
    count : int      – number of valid entries
    """

    __slots__ = ("ftnr", "freq", "count")

    def __init__(self, ftnr, freq, count):
        self.ftnr = ftnr
        self.freq = freq
        self.count = count


def reset_foundpix_array(arr: np.ndarray, arr_len: int, num_cams: int) -> None:
    """Set default values for foundpix objects in an array.

    Arguments:
    ---------
    arr -- the array to reset, dtype = Foundpix_dtype
    arr_len -- array length
    num_cams -- number of places in the whichcam member of foundpix.
    """
    arr['ftnr'][:arr_len] = TR_UNUSED
    arr['freq'][:arr_len] = 0
    arr['whichcam'][:arr_len] = 0
    return None


def copy_foundpix_array(
    dest: np.ndarray, src: np.ndarray, arr_len: int, num_cams: int
) -> None:
    """Copy the relevant part of foundpix array."""
    dest[:arr_len] = src[:arr_len]


@njit(cache=True, fastmath=True, nogil=True)
def _candsearch_in_pix_core(
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_tnr: np.ndarray,
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    imx: float,
    imy: float,
    require_unused: bool,
):
    """Return the four closest candidate indices in pixel space."""
    p = np.empty(MAX_CANDS, dtype=np.int32)
    for i in range(MAX_CANDS):
        p[i] = PT_UNUSED

    dmin = 1e20
    p1 = p2 = p3 = p4 = PT_UNUSED
    d1 = d2 = d3 = d4 = dmin

    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > imx:
        xmax = imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > imy:
        ymax = imy

    scanned_rows = 0
    if 0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy:
        j0 = num_targets // 2
        dj = num_targets // 4
        while dj > 1:
            if target_y[j0] < ymin:
                j0 += dj
            else:
                j0 -= dj
            dj //= 2

        j0 -= 12
        if j0 < 0:
            j0 = 0

        for j in range(j0, num_targets):
            scanned_rows += 1
            if require_unused:
                if target_tnr[j] != TR_UNUSED:
                    continue
            elif target_tnr[j] == TR_UNUSED:
                continue

            if target_y[j] > ymax:
                break

            if xmin < target_x[j] < xmax and ymin < target_y[j] < ymax:
                d = math.sqrt(
                    (cent_x - target_x[j]) * (cent_x - target_x[j])
                    + (cent_y - target_y[j]) * (cent_y - target_y[j])
                )

                if d < dmin:
                    dmin = d

                if d < d1:
                    p4, p3, p2, p1 = p3, p2, p1, j
                    d4, d3, d2, d1 = d3, d2, d1, d
                elif d1 < d < d2:
                    p4, p3, p2 = p3, p2, j
                    d4, d3, d2 = d3, d2, d
                elif d2 < d < d3:
                    p4, p3 = p3, j
                    d4, d3 = d3, d
                elif d3 < d < d4:
                    p4 = j
                    d4 = d

    p[0] = p1
    p[1] = p2
    p[2] = p3
    p[3] = p4
    return p, scanned_rows


def register_closest_neighbs(
    targets: List[Target],
    num_targets: int,
    cam: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    reg: np.ndarray,
    cpar: ControlPar,
    target_x: np.ndarray | None = None,
    target_y: np.ndarray | None = None,
    target_tnr: np.ndarray | None = None,
    reg_ftnr: np.ndarray | None = None,
    reg_whichcam: np.ndarray | None = None,
) -> List[int]:
    """Register_closest_neighbs() finds candidates for continuing a particle's.

    path in the search volume, and registers their data in a foundpix array
    that is later used by the tracking algorithm.

    Arguments:
    ---------
    targets -- the targets list to search.
    num_targets -- target array length.
    cam -- the index of the camera we're working on.
    cent_x -- image coordinate of search area center along x-axis, [pixel]
    cent_y -- image coordinate of search area center along y-axis, [pixel]
    dl -- left distance to the search area border from its center, [pixel]
    dr -- right distance to the search area border from its center, [pixel]
    du -- up distance to the search area border from its center, [pixel]
    dd -- down distance to the search area border from its center, [pixel]
    reg -- an array of foundpix objects (legacy recarray), or None if SoA.
    cpar -- control parameter object
    reg_ftnr -- SoA: int32[MAX_CANDS] slice for ftnr output
    reg_whichcam -- SoA: int32[MAX_CANDS, num_cams] slice for whichcam output
    """
    # all_cands = [-999] * MAX_CANDS  # Initialize all candidate indexes to -999

    if target_x is not None and target_y is not None and target_tnr is not None:
        all_cands, _ = _candsearch_in_pix_core(
            target_x,
            target_y,
            target_tnr,
            num_targets,
            cent_x,
            cent_y,
            dl,
            dr,
            du,
            dd,
            cpar.imx,
            cpar.imy,
            False,
        )
        all_cands = all_cands.tolist()
    else:
        all_cands = candsearch_in_pix(
            targets, num_targets, cent_x, cent_y, dl, dr, du, dd, cpar
        )

    # SoA path: write into plain arrays
    if reg_ftnr is not None and reg_whichcam is not None:
        for cand_idx in range(MAX_CANDS):
            if (
                all_cands[cand_idx] == PT_UNUSED
                or all_cands[cand_idx] < 0
                or all_cands[cand_idx] >= num_targets
            ):
                reg_ftnr[cand_idx] = TR_UNUSED
            else:
                reg_whichcam[cand_idx, cam] = 1
                reg_ftnr[cand_idx] = targets[all_cands[cand_idx]].tnr
    else:
        # Legacy recarray path
        for cand_idx in range(MAX_CANDS):
            if (
                all_cands[cand_idx] == PT_UNUSED
                or all_cands[cand_idx] < 0
                or all_cands[cand_idx] >= num_targets
            ):
                reg[cand_idx].ftnr = TR_UNUSED
            else:
                reg[cand_idx].whichcam[cam] = 1
                reg[cand_idx].ftnr = targets[all_cands[cand_idx]].tnr

    return all_cands


@njit(
    float64[:](float64[:], float64[:]),
    cache=True,
    fastmath=True,
    nogil=True,
    parallel=True,
)
def search_volume_center_moving(
    prev_pos: np.ndarray, curr_pos: np.ndarray
) -> np.ndarray:
    """Find the position of the center of the search volume for a moving.

    particle using the velocity of last step.

    Args:
    ----
        prev_pos (vec3d): Previous position of the particle.
        curr_pos (vec3d): Current position of the particle.
        output (vec3d): Output variable for the calculated position.

    Returns
    -------
        None
    """
    # Multiply current position by 2 and subtract previous position
    # output[0] = 2 * curr_pos[0] - prev_pos[0]
    # output[1] = 2 * curr_pos[1] - prev_pos[1]
    # output[2] = 2 * curr_pos[2] - prev_pos[2]

    return 2 * curr_pos - prev_pos


def predict(prev_pos, curr_pos, output):
    """Predicts the position of a particle in the next_frame frame, using the.

    previous and current positions.

    Args:
    ----
        prev_pos (vec2d): 2D position at previous frame.
        curr_pos (vec2d): 2D position at current frame.
        output (vec2d): Output of the 2D positions of the particle in the next_frame frame.

    Returns
    -------
        None
    """
    # Calculate the position of the particle in the next_frame frame using the current and previous positions
    output[0] = 2 * curr_pos[0] - prev_pos[0]
    output[1] = 2 * curr_pos[1] - prev_pos[1]


@njit(cache=True, fastmath=True, nogil=True)
def pos3d_in_bounds(pos: np.ndarray, bounds: TrackParTuple) -> bool:
    """Check that all components of a pos3d are in their respective bounds.

    taken from a track_par object.

    Args:
    ----
        pos (vec3d): The 3-component array to check.
        bounds (track_par): The struct containing the bounds specification.

    Returns
    -------
        True if all components are in bounds, False otherwise.
    """
    # Check if all three components of pos are within their respective bounds in bounds.
    return (
        bounds.dvxmin < pos[0] < bounds.dvxmax
        and bounds.dvymin < pos[1] < bounds.dvymax
        and bounds.dvzmin < pos[2] < bounds.dvzmax
    )


@njit(
    types.UniTuple(float64, 2)(float64[:], float64[:], float64[:]),
    cache=True,
    fastmath=True,
    nogil=True,
)
def angle_acc(start: np.ndarray, pred: np.ndarray, cand: np.ndarray):
    """Calculate the angle between the (1st order) numerical velocity vectors.

    to the predicted next_frame position and to the candidate actual position. The
    angle is calculated in [gon], see [1]. The predicted position is the
    position if the particle continued at the current velocity.

    Arguments:
    ---------
    start -- vec3d, the particle start position
    pred -- vec3d, predicted position
    cand -- vec3d, possible actual position

    Returns
    -------
    angle -- float, the angle between the two velocity vectors, [gon]
    acc -- float, the 1st-order numerical acceleration embodied in the deviation from prediction.
    """
    v0 = pred - start
    v1 = cand - start

    acc = np.linalg.norm(v0 - v1)

    if np.all(v0 == -v1):
        angle = 200.0
    elif np.all(v0 == v1):
        angle = 0.0
    else:
        dot_product = np.sum(v0 * v1)
        norm_start_pred = np.linalg.norm(start - pred)
        norm_start_cand = np.linalg.norm(start - cand)

        if norm_start_pred == 0.0 or norm_start_cand == 0.0:
            angle = 0.0
        else:
            cosine = dot_product / (norm_start_pred * norm_start_cand)
            cosine = min(1.0, max(-1.0, cosine))
            angle = (200.0 / np.pi) * np.arccos(cosine)

    return angle, acc


def candsearch_in_pix(
    next_frame: List[Target],
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    cpar: ControlPar,
) -> List[int]:
    """Search for a nearest candidate in unmatched target list."""
    debug = _tracker_debug_enabled()
    cand_start = time.perf_counter()
    bounds_start = time.perf_counter()
    target_x, target_y, target_tnr = _target_search_arrays(next_frame, num_targets)
    bounds_elapsed = time.perf_counter() - bounds_start

    scan_start = time.perf_counter()
    p, scanned_rows = _candsearch_in_pix_core(
        target_x,
        target_y,
        target_tnr,
        num_targets,
        cent_x,
        cent_y,
        dl,
        dr,
        du,
        dd,
        cpar.imx,
        cpar.imy,
        False,
    )
    scan_elapsed = time.perf_counter() - scan_start

    p = p.tolist()

    if debug:
        _tracker_log(
            "candsearch_in_pix",
            int(num_targets),
            (
                f"bounds={bounds_elapsed:.3f}s scan={scan_elapsed:.3f}s "
                f"scanned_rows={scanned_rows} total={time.perf_counter() - cand_start:.3f}s"
            ),
        )

        # print("from inside p = ", p)

        # TODO: check why we need counter, we can use counter = len(p) - p.count(-1)
        # for j in range(4):
        #     if p[j] != -1:
        #         counter += 1

    return p


def candsearch_in_pix_rest(
    next_frame: List[Target],
    num_targets: int,
    cent_x: float,
    cent_y: float,
    dl: float,
    dr: float,
    du: float,
    dd: float,
    p: List[int],
    cpar: ControlPar,
) -> int:
    """Search for a nearest candidate in unmatched target list.

    Arguments:
    ---------
    next_frame - 2D numpy array of targets (pointer, x,y, n, nx,ny, sumg, track ID),
        assumed to be y sorted.
    num_targets - number of targets in the next_frame
    cent_x, cent_y - image coordinates of the position of a particle [pixel]
    dl, dr, du, dd - respectively the left, right, up, down distance
        to the search area borders from its center, [pixel]
    cpar - control_par object with attributes imx and imy.

    Returns
    -------
    int - the number of candidates found, between 0 - 1
    """
    counter = 0
    dmin = POS_INF
    p[0] = PT_UNUSED
    xmin, xmax, ymin, ymax = cent_x - dl, cent_x + dr, cent_y - du, cent_y + dd

    xmin = max(xmin, 0.0)
    xmax = min(xmax, cpar.imx)
    ymin = max(ymin, 0.0)
    ymax = min(ymax, cpar.imy)

    if 0 <= cent_x <= cpar.imx and 0 <= cent_y <= cpar.imy:
        # binarized search for start point of candidate search
        j0, dj = num_targets // 2, num_targets // 4
        while dj > 1:
            j0 += dj if next_frame[j0].y < ymin else -dj
            dj //= 2

        j0 -= 12 if j0 >= 12 else j0  # due to trunc
        for j in range(j0, num_targets):
            if next_frame[j].tnr == TR_UNUSED:
                if next_frame[j].y > ymax:
                    break  # finish search
                if xmin < next_frame[j].x < xmax and ymin < next_frame[j].y < ymax:
                    d = math.sqrt(
                        (cent_x - next_frame[j].x) * (cent_x - next_frame[j].x)
                        + (cent_y - next_frame[j].y) * (cent_y - next_frame[j].y)
                    )
                    if d < dmin:
                        dmin = d
                        p[0] = j

        if p[0] != PT_UNUSED:
            counter += 1

    return counter


def searchquader(
    point: np.ndarray, tpar: TrackParTuple, cpar: ControlPar, cal: List[Calibration],
    raw_cals=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the search volume in image space."""
    mins = np.array([tpar.dvxmin, tpar.dvymin, tpar.dvzmin])
    maxes = np.array([tpar.dvxmax, tpar.dvymax, tpar.dvzmax])

    corner = np.empty(3, dtype=np.float64)

    xr = np.empty(cpar.num_cams, dtype=np.float64)
    xl = np.empty(cpar.num_cams, dtype=np.float64)
    yd = np.empty(cpar.num_cams, dtype=np.float64)
    yu = np.empty(cpar.num_cams, dtype=np.float64)

    base_x = point[0]
    base_y = point[1]
    base_z = point[2]

    use_fast = raw_cals is not None

    # calculation of search area in each camera
    for i in range(cpar.num_cams):
        # initially large or small values
        xr[i] = 0
        xl[i] = cpar.imx
        yd[i] = 0
        yu[i] = cpar.imy

        # pixel position of a search center
        if use_fast:
            center_x, center_y = raw_cals[i].project(point)
        else:
            center = np.empty(2, dtype=np.float64)
            _point_to_pixel_into(point, cal[i], cpar, center)
            center_x = center[0]
            center_y = center[1]

        # mark 8 corners of the search region in pixels
        for pt in range(8):
            corner[0] = base_x + (maxes[0] if pt & 1 else mins[0])
            corner[1] = base_y + (maxes[1] if pt & 2 else mins[1])
            corner[2] = base_z + (maxes[2] if pt & 4 else mins[2])

            if use_fast:
                cpx, cpy = raw_cals[i].project(corner)
            else:
                corner_proj = np.empty(2, dtype=np.float64)
                _point_to_pixel_into(corner, cal[i], cpar, corner_proj)
                cpx, cpy = corner_proj[0], corner_proj[1]

            if cpx < xl[i]:
                xl[i] = cpx
            if cpy < yu[i]:
                yu[i] = cpy
            if cpx > xr[i]:
                xr[i] = cpx
            if cpy > yd[i]:
                yd[i] = cpy

        if xl[i] < 0:
            xl[i] = 0
        if yu[i] < 0:
            yu[i] = 0
        if xr[i] > cpar.imx:
            xr[i] = cpar.imx
        if yd[i] > cpar.imy:
            yd[i] = cpar.imy

        # eventually xr, xl, yd, yu are pixel distances relative to the point
        xr[i] = xr[i] - center_x
        xl[i] = center_x - xl[i]
        yd[i] = yd[i] - center_y
        yu[i] = center_y - yu[i]

    return xr, xl, yd, yu


class _CandidateSearchCache:
    """Reusable buffers for profiler-enabled tracking hot paths."""

    def __init__(self, num_cams: int):
        n_fp = num_cams * MAX_CANDS
        self.ftnr = np.full(n_fp, TR_UNUSED, dtype=np.int32)
        self.freq = np.zeros(n_fp, dtype=np.int32)
        self.whichcam = np.zeros((n_fp, num_cams), dtype=np.int32)


@njit(cache=True, nogil=True)
def _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams):
    """Sort candidates by frequency — numba-compiled version on plain arrays.

    Arguments:
    ---------
    ftnr -- int32[n] target numbers (TR_UNUSED for empty)
    freq -- int32[n] frequency counts (zeroed on entry)
    whichcam -- int32[n, num_cams] camera flags (zeroed on entry)
    num_cams -- int, number of cameras

    Returns:
    -------
    int, number of distinct valid candidates after sort
    """
    n = num_cams * 4  # MAX_CANDS = 4

    # Phase 1: mark whichcam — for each item, mark which cameras saw it
    for i in range(n):
        if ftnr[i] == -1:  # TR_UNUSED
            continue
        for j in range(num_cams):
            base = 4 * j
            for m in range(4):
                if ftnr[i] == ftnr[base + m]:
                    whichcam[i, j] = 1

    # Phase 2: count frequency
    for i in range(n):
        if ftnr[i] == -1:
            continue
        for j in range(num_cams):
            if whichcam[i, j] == 1:
                freq[i] += 1

    # Phase 3: sort by freq descending (insertion sort, n<=16)
    for i in range(1, n):
        key_ftnr = ftnr[i]
        key_freq = freq[i]
        key_wc = whichcam[i].copy()
        j = i - 1
        while j >= 0 and freq[j] < key_freq:
            ftnr[j + 1] = ftnr[j]
            freq[j + 1] = freq[j]
            whichcam[j + 1] = whichcam[j]
            j -= 1
        ftnr[j + 1] = key_ftnr
        freq[j + 1] = key_freq
        whichcam[j + 1] = key_wc

    # Phase 4: prune duplicates and singletons
    for i in range(n):
        if ftnr[i] == -1:
            continue
        for j in range(i + 1, n):
            if ftnr[i] == ftnr[j] or freq[j] < 2:
                freq[j] = 0
                ftnr[j] = -1

    # Phase 5: sort again (same insertion sort)
    for i in range(1, n):
        key_ftnr = ftnr[i]
        key_freq = freq[i]
        key_wc = whichcam[i].copy()
        j = i - 1
        while j >= 0 and freq[j] < key_freq:
            ftnr[j + 1] = ftnr[j]
            freq[j + 1] = freq[j]
            whichcam[j + 1] = whichcam[j]
            j -= 1
        ftnr[j + 1] = key_ftnr
        freq[j + 1] = key_freq
        whichcam[j + 1] = key_wc

    different = 0
    for i in range(n):
        if freq[i] != 0:
            different += 1

    return different


def sort_candidates_by_freq(foundpix: np.ndarray, num_cams: int) -> int:
    """Sort candidates by frequency — delegates to @njit kernel."""
    ftnr = foundpix['ftnr']
    freq = foundpix['freq']
    whichcam = foundpix['whichcam']
    return _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams)


def sort(n: int, a: List[float], b: List[int]) -> Tuple[List[float], List[int]]:
    """In-place sorts a float list 'a' and an integer list 'b' equal lengths, sort up to n.

    Arguments:
    ---------
    a -- float array (returned sorted in ascending order)
    b -- integer array (returned sorted according to float array a)

    Returns
    -------
    Sorted arrays a and b.
    """
    # idx = np.argsort(a)
    # a[...] = a[idx]
    # b[...] = b[idx]

    # return a, b

    sorted_pairs = sorted(zip(a[:n], b[:n]))
    a[:n], b[:n] = zip(*sorted_pairs)
    return a, b


def find_candidates_in_3d(frm, pos, dx, dy, dz, max_cands=MAX_CANDS):
    """
    Find particles within a 3D box centered at pos.

    Arguments:
        frm - Frame object with path_info list
        pos - (3,) array-like, center position
        dx, dy, dz - box half-sizes in each dimension
        max_cands - maximum candidates to return

    Returns:
        list of particle indices within the box
    """
    indices = []
    for i in range(frm.num_parts):
        pi = frm.path_info[i]
        if (
            abs(pi.x[0] - pos[0]) < dx
            and abs(pi.x[1] - pos[1]) < dy
            and abs(pi.x[2] - pos[2]) < dz
        ):
            if len(indices) < max_cands:
                indices.append(i)
    return indices


def _point_to_pixel_into(
    point: np.ndarray, cal: Calibration, cpar: ControlPar, out: np.ndarray
) -> None:
    """Write pixel positions (x,y) into *out*.

    Arguments:
    ---------
    point -- vec3d point in 3D space
    cal -- Calibration parameters
    cpar -- Control parameters (num cams, multimedia parameters, cpar->mm, etc.)
    """
    x, y = img_coord(point, cal, cpar.mm)
    x, y = metric_to_pixel(x, y, cpar)
    out[0] = x
    out[1] = y


def point_to_pixel(point: np.ndarray, cal: Calibration, cpar: ControlPar) -> Tuple[float, float]:
    """Return pixel positions (x,y) in the camera."""
    out = np.empty(2, dtype=np.float64)
    _point_to_pixel_into(point, cal, cpar, out)
    return out[0], out[1]


def sorted_candidates_in_volume(
    center: np.ndarray, center_proj: np.ndarray, frm: Frame, run: TrackingRun
) -> FoundpixResult:
    """Find candidates for continuing a particle's path in the search volume."""
    num_cams = frm.num_cams
    profile_mode = _tracker_profile_enabled()

    if profile_mode:
        cache = getattr(run, "_candidate_cache", None)
        if cache is None or cache.ftnr.shape[0] != num_cams * MAX_CANDS:
            cache = _CandidateSearchCache(num_cams)
            run._candidate_cache = cache

        ftnr = cache.ftnr
        freq = cache.freq
        whichcam = cache.whichcam
        ftnr.fill(TR_UNUSED)
        freq.fill(0)
        whichcam.fill(0)
        right, left, down, up = searchquader(
            center, run.tpar, run.cpar, run.cal, raw_cals=run.raw_cal
        )
    else:
        n_fp = num_cams * MAX_CANDS
        ftnr = np.full(n_fp, TR_UNUSED, dtype=np.int32)
        freq = np.zeros(n_fp, dtype=np.int32)
        whichcam = np.zeros((n_fp, num_cams), dtype=np.int32)
        right, left, down, up = searchquader(
            center, run.tpar, run.cpar, run.cal, raw_cals=run.raw_cal
        )

    # search in pix for candidates in the next_frame time step
    for cam in range(num_cams):
        register_closest_neighbs(
            frm.targets[cam],
            frm.num_targets[cam],
            cam,
            center_proj[cam][0],
            center_proj[cam][1],
            left[cam],
            right[cam],
            up[cam],
            down[cam],
            None,
            run.cpar,
            target_x=frm.target_x[cam],
            target_y=frm.target_y[cam],
            target_tnr=frm.target_tnr[cam],
            reg_ftnr=ftnr[cam * MAX_CANDS :],
            reg_whichcam=whichcam[cam * MAX_CANDS :],
        )

    # fill and sort candidate struct
    num_cands = _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams)
    if num_cands > 0:
        return FoundpixResult(
            np.array(ftnr[: num_cands + 1], copy=True),
            np.array(freq[: num_cands + 1], copy=True),
            num_cands + 1,
        )
    else:
        return FoundpixResult(
            np.array([TR_UNUSED], dtype=np.int32),
            np.array([0], dtype=np.int32),
            1,
        )


def assess_new_position(
    pos: np.ndarray, frm: Frame, run: TrackingRun
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Determine the nearest target on each camera around a search position.

    #     and prepares the data structures accordingly with the determined target
    #     info or the unused flag value.

    #     Arguments:
    #     ---------
    #     pos - vec3d, the position around which to search.
    #     targ_pos - vec2d, the determined targets' respective positions.
    #     cand_inds - 2D array of integers, output buffer, the determined targets'
    #         index in the respective camera's target list.
    #     frm - frame object, holdin target data for the search position.
    #     run - TrackingRun object, scene information struct.

    #     Returns:
    #     -------
    #     Integer, the number of cameras where a suitable target was found.

    """
    # Output variables
    nc = run.cpar.num_cams
    targ_pos = np.full((nc, 2), COORD_UNUSED, dtype=np.float64)
    cand_inds = np.full((nc, MAX_CANDS), -1, dtype=np.int32)

    # Search rectangle limits
    left, right, up, down = ADD_PART, ADD_PART, ADD_PART, ADD_PART

    # for cam in range(run.cpar.num_cams):
    #     targ_pos[cam] = [COORD_UNUSED, COORD_UNUSED]

    for cam in range(run.cpar.num_cams):
        # Convert 3D search position to 2D pixel coordinates
        pixel = run.raw_cal[cam].project(pos)
        # print(f"pos {pos}")
        # print(f"pixel {pixel}")

        # Nearest neighbor search
        num_cands = candsearch_in_pix_rest(
            frm.targets[cam],
            frm.num_targets[cam],
            pixel[0],
            pixel[1],
            left,
            right,
            up,
            down,
            cand_inds[cam],
            run.cpar,
        )

        if num_cands > 0:
            _ix = cand_inds[cam][0]  # first nearest neighbour
            targ_pos[cam][0] = frm.targets[cam][_ix].x
            targ_pos[cam][1] = frm.targets[cam][_ix].y

    valid_cams = 0

    for cam in range(run.cpar.num_cams):
        if (targ_pos[cam][0] != COORD_UNUSED) and (targ_pos[cam][1] != COORD_UNUSED):
            # Convert pixel coordinates to metric coordinates
            x, y = pixel_to_metric(targ_pos[cam][0], targ_pos[cam][1], run.cpar)

            # Apply additional transformations
            targ_pos[cam][0], targ_pos[cam][1] = dist_to_flat(
                x, y, run.cal[cam], run.flatten_tol
            )

            valid_cams += 1

    return valid_cams, targ_pos, cand_inds


# def add_particle(frm: Frame, pos: np.ndarray, cand_inds: np.ndarray) -> None:
#     """Add a new particle to the frame buffer."""
#     ref_path_inf = Pathinfo(x=pos)
#     ref_path_inf.reset_links()

#     frm.path_info.append(ref_path_inf)


#     ref_corres =  Corres()
#     ref_targets = frm.targets

#     for cam in range(frm.num_cams):
#         ref_corres.p[cam] = CORRES_NONE

#         # We always take the 1st candidate, apparently. Why did we fetch 4?
#         if cand_inds[cam][0] != PT_UNUSED:
#             _ix = cand_inds[cam][0]
#             ref_targets[cam][_ix].tnr = frm.num_parts
#             ref_corres.p[cam] = _ix
#             ref_corres.nr = frm.num_parts

#     frm.correspond.append(ref_corres)
#     frm.num_parts += 1


def add_particle(frm: Frame, pos: np.ndarray, cand_inds: np.ndarray) -> None:
    """Insert a particle at a given position to the end of the frame, along with associated targets.

    Arguments:
    - frm (frame): The frame to store the particle.
    - pos (vec3d): Position of the inserted particle in global coordinates.
    - cand_inds (list[list[int]]): Indices of candidate targets for association with this particle.
    """
    num_parts = frm.num_parts

    # Ensure path_info has room for this index
    if num_parts < len(frm.path_info):
        ref_path_inf = frm.path_info[num_parts]
    else:
        ref_path_inf = Pathinfo()
        frm.path_info.append(ref_path_inf)

    # Ensure corres arrays have room for this index
    if num_parts >= frm.corres_nr.shape[0]:
        frm.corres_nr = np.resize(frm.corres_nr, num_parts + 1)
        frm.corres_p = np.resize(frm.corres_p, (num_parts + 1, 4))

    ref_path_inf.x = vec_copy(pos)
    ref_path_inf.reset_links()

    ref_targets = frm.targets

    for cam in range(frm.num_cams):
        frm.corres_p[num_parts, cam] = CORRES_NONE

        # We always take the 1st candidate, apparently. Why did we fetch 4?
        if cand_inds[cam][0] != PT_UNUSED:
            _ix = cand_inds[cam][0]
            ref_targets[cam][_ix].tnr = num_parts
            frm.corres_p[num_parts, cam] = _ix
            frm.corres_nr[num_parts] = num_parts

    frm.num_parts += 1


def track_forward_start(tr: TrackingRun):
    """Initialize the tracking frame buffer with the first frames.

    Arguments:
    ---------
    tr - an object holding the per-run tracking parameters, and
         a frame buffer with 4 positions.
    """
    # step = tr.seq_par.first

    # Prime the buffer with first three frames, fourth frame is read when we track
    for step in range(tr.seq_par.first, tr.seq_par.first + TR_BUFSPACE - 1):
        tr.fb.read_frame_at_end(step)
        tr.fb.fb_next()

    tr.fb.fb_prev()


class TrackingObserver:
    """Collects per-particle tracking events for later visualization.

    Each event is a dict describing one particle's tracking decision at one
    frame. Attach an instance to ``trackcorr_c_loop`` via the ``observer``
    parameter to record the full decision tree without modifying the algorithm.

    Attributes
    ----------
    events : list[dict]
        Accumulated tracking events across all frames.
    """

    def __init__(self):
        self.events: list = []

    def record(self, event: dict) -> None:
        self.events.append(event)

    def clear(self):
        self.events.clear()

    def events_for_frame(self, step: int) -> list:
        return [e for e in self.events if e["step"] == step]

    def events_for_particle(self, particle_id: int) -> list:
        return [e for e in self.events if e["particle_id"] == particle_id]


def trackcorr_c_loop(run_info, step, observer=None):
    """Sequence loop."""
    debug = _tracker_debug_enabled()
    profile = cProfile.Profile() if _tracker_profile_enabled() else None
    if profile is not None:
        profile.enable()

    step_start = time.perf_counter()
    # Initialize variables
    philf = np.zeros((4, MAX_CANDS))
    # quali = 0
    diff_pos = np.empty((3,))

    # 7 reference points used in the algorithm, TODO: check if can reuse some
    # angle, acc, angle0, acc0, dl = 0.0, 0.0, 0.0, 0.0, 0.0
    # angle1, acc1 = 0.0, 0.0

    rr = 0.0

    _ix = 0  # For use in any of the complex index expressions below
    num_added = 0
    count1 = 0
    count2 = 0
    count3 = 0

    fb = run_info.fb
    cal = run_info.cal
    raw_cal = run_info.raw_cal
    tpar = convert_track_par_to_tuple(run_info.tpar)
    vpar = run_info.vpar
    cpar = run_info.cpar
    curr_targets = fb.buf[1].targets

    if debug:
        _tracker_log(
            "trackcorr",
            step,
            f"start curr_parts={fb.buf[1].num_parts} next_parts={fb.buf[2].num_parts} prev_parts={fb.buf[0].num_parts}",
        )

    v1 = np.zeros((cpar.num_cams, 2))  # volume center projection on cameras
    v2 = np.zeros((cpar.num_cams, 2))  # volume center projection on cameras

    # try to track correspondences from previous 0 - corp, variable h
    orig_parts = fb.buf[1].num_parts
    particle_loop_start = time.perf_counter()
    proj_time = 0.0
    cand_search_time = 0.0
    decision_time = 0.0
    add_time = 0.0
    progress_interval = 50
    X = np.zeros((6, 3))
    for h in range(orig_parts):
        X[:] = 0.0

        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres_p = fb.buf[1].corres_p[h]

        curr_path_inf.inlist = 0

        # 3D-position
        X[1] = vec_copy(curr_path_inf.x)
        # print(f"X[1] {X[1]}")

        # use information from previous to locate new search position
        # and to calculate values for search area
        if curr_path_inf.prev_frame >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev_frame]
            X[0] = vec_copy(ref_path_inf.x)
            X[2] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            proj_start = time.perf_counter()
            for j in range(fb.num_cams):
                v1[j] = raw_cal[j].project(X[2])
            proj_time += time.perf_counter() - proj_start
        else:
            X[2] = vec_copy(X[1])
            proj_start = time.perf_counter()
            for j in range(fb.num_cams):
                if curr_corres_p[j] == CORRES_NONE or curr_corres_p[j] >= len(
                    curr_targets[j]
                ):
                    v1[j] = raw_cal[j].project(X[2])
                else:
                    _ix = curr_corres_p[j]
                    v1[j, 0] = curr_targets[j][_ix].x
                    v1[j, 1] = curr_targets[j][_ix].y
                    # print(f"v1[{j}], {v1[j]}")
            proj_time += time.perf_counter() - proj_start

        # calculate search cuboid and reproject it to the image space
        # Compute search limits for observer before candidate search
        if observer is not None:
            _obs_xr, _obs_xl, _obs_yd, _obs_yu = searchquader(
                X[2], tpar, cpar, cal, raw_cals=run_info.raw_cal
            )
        cand_start = time.perf_counter()
        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2], run_info)
        cand_search_time += time.perf_counter() - cand_start
        # if not w  # empty
        if w.count == 1 and w.ftnr[0] == TR_UNUSED:  # empty
            if observer is not None:
                observer.record({
                    "step": step,
                    "particle_id": h,
                    "type": "no_candidates",
                    "pos_3d": X[1].copy(),
                    "predicted_3d": X[2].copy(),
                    "prev_3d": X[0].copy() if curr_path_inf.prev_frame >= 0 else None,
                    "search_center_px": v1.copy(),
                    "search_rect": {
                        "xr": _obs_xr.copy(), "xl": _obs_xl.copy(),
                        "yd": _obs_yd.copy(), "yu": _obs_yu.copy(),
                    },
                    "candidates": [],
                })
            continue

        # Continue to find candidates for the candidates.
        count2 += 1
        # Build candidate list for observer
        _obs_candidates = []
        mm = 0
        # counter1-loop
        while (
            mm < w.count
            and w.ftnr[mm] != TR_UNUSED
            and len(fb.buf[2].path_info) > w.ftnr[mm]
        ):
            # search for found corr of current the corr in next_frame with predicted location

            # found 3D-position
            ref_path_inf = fb.buf[2].path_info[w.ftnr[mm]]
            X[3] = vec_copy(ref_path_inf.x)
            # print(f"X[3] {X[3]}")

            if curr_path_inf.prev_frame >= 0:
                # for j in range(3):
                #     X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])
                X[5] = 0.5 * (5 * X[3] - 4 * X[1] + X[0])
            else:
                X[5] = search_volume_center_moving(X[1], X[3])

            # print(f"X[5] {X[5]}")

            for j in range(fb.num_cams):
                v1[j] = raw_cal[j].project(X[5])
                #  print(f"v1[{j}], {v1[j]}")

            # end of search in pix
            wn = sorted_candidates_in_volume(X[5], v1, fb.buf[3], run_info)
            if wn.count > 1:  # not empty means two rows at least.
                count3 += 1
                kk = 0
                while (
                    kk < wn.count
                    and wn.ftnr[kk] != TR_UNUSED
                    and len(fb.buf[3].path_info) > wn.ftnr[kk]
                ):
                    # print(f" inside wn[{kk}].ftnr {wn.ftnr[kk]}")
                    ref_path_inf = fb.buf[3].path_info[wn.ftnr[kk]]
                    X[4] = vec_copy(ref_path_inf.x)
                    #  print(f"X[4] {X[4]}")

                    diff_pos = vec_subt(X[4], X[3])
                    # print(f"inside kk loop {kk}")
                    # print(f"diff_pos {diff_pos}")

                    if pos3d_in_bounds(diff_pos, tpar):
                        angle1, acc1 = angle_acc(X[3], X[4], X[5])
                        if curr_path_inf.prev_frame >= 0:
                            angle0, acc0 = angle_acc(X[1], X[2], X[3])
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) / 2
                        angle = (angle0 + angle1) / 2
                        quali = wn.freq[kk] + w.freq[mm]

                        if (
                            acc < tpar.dacc
                            and angle < tpar.dangle
                            or acc < tpar.dacc / 10
                        ):
                            dl = (
                                vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])
                            ) / 2
                            rr = (
                                dl / run_info.lmax
                                + acc / tpar.dacc
                                + angle / tpar.dangle
                            ) / quali
                            curr_path_inf.register_link_candidate(rr, w.ftnr[mm])
                            # print(f"kk {kk}, rr {rr}, w.ftnr[mm] {w.ftnr[mm]}")

                    kk += 1  # End of searching 2nd-frame candidates.
                    # print(f"kk is {kk}")

            # creating new particle position,
            # reset img coord because of num_cams < 4
            # fix distance of 3 pixels to define xl,xr,yu,yd instead of searchquader
            # and search for unused candidates in next_frame time step

            quali, v2, philf = assess_new_position(X[5], fb.buf[3], run_info)
            # print(f"quali {quali}, v2 {v2}, philf {philf}")

            # quali >=2 means at least in two cameras
            # we found a candidate
            if quali >= 2:
                in_volume = 0  # inside volume

                dl, X[4] = point_position(v2, cpar.num_cams, cpar.mm, cal)

                # volume check
                if (
                    vpar.x_lay[0] < X[4][0]
                    and X[4][0] < vpar.x_lay[1]
                    and run_info.ymin < X[4][1]
                    and X[4][1] < run_info.ymax
                    and vpar.z_min_lay[0] < X[4][2]
                    and X[4][2] < vpar.z_max_lay[1]
                ):
                    in_volume = 1

                diff_pos = vec_subt(X[3], X[4])
                # print(f"second diff_pos {diff_pos}")

                if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    # print(f"angle {angle}, acc {acc}")

                    if acc < tpar.dacc and angle < tpar.dangle or acc < tpar.dacc / 10:
                        dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[4], X[3])) / 2
                        # print(f" dl {dl} ")
                        rr = (
                            dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle
                        ) / (quali + w.freq[mm])

                        # print(f"acc {acc}, angle {angle}, quali {quali}, w.freq[mm] {w.freq[mm]}")
                        # print(f"rr {rr}, w.ftnr[mm] {w.ftnr[mm]}")
                        curr_path_inf.register_link_candidate(rr, w.ftnr[mm])

                        if tpar.add:
                            add_particle(fb.buf[3], X[4], philf)
                            num_added += 1

                in_volume = 0

            quali = 0

            # end of creating new particle position
            # ***************************************************************

            # try to link if kk is not found/good enough
            if curr_path_inf.inlist == 0:
                diff_pos = vec_subt(X[3], X[1])
                if pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[1], X[2], X[3])
                    if (acc < tpar.dacc and angle < tpar.dangle) or (
                        acc < tpar.dacc / 10
                    ):
                        quali = w.freq[mm]
                        dl = (vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])) / 2
                        rr = (
                            dl / run_info.lmax + acc / tpar.dacc + angle / tpar.dangle
                        ) / quali

                        # print(f"prev exists {mm}")
                        # print(f"rr {rr}, w.ftnr[mm] {w.ftnr[mm]}")
                        curr_path_inf.register_link_candidate(rr, w.ftnr[mm])

            del wn
            # Record this candidate for the observer
            if observer is not None:
                _obs_candidates.append({
                    "ftnr": int(w.ftnr[mm]),
                    "cand_3d": X[3].copy(),
                    "freq": int(w.freq[mm]),
                    "registered": curr_path_inf.inlist > 0,
                })
            mm += 1  # increment mm

        decision_start = time.perf_counter()
        # begin of inlist still zero
        if tpar.add:
            if curr_path_inf.inlist == 0 and curr_path_inf.prev_frame >= 0:
                quali, v2, philf = assess_new_position(X[2], fb.buf[2], run_info)
                if quali >= 2:
                    X[3] = vec_copy(X[2])
                    in_volume = 0
                    dl, X[3] = point_position(v2, fb.num_cams, cpar.mm, cal)

                    # in volume check
                    if (
                        vpar.x_lay[0] < X[3][0] < vpar.x_lay[1]
                        and run_info.ymin < X[3][1] < run_info.ymax
                        and vpar.z_min_lay[0] < X[3][2] < vpar.z_max_lay[1]
                    ):
                        in_volume = 1

                    diff_pos = vec_subt(X[2], X[3])
                    if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if (acc < tpar.dacc and angle < tpar.dangle) or (
                            acc < tpar.dacc / 10
                        ):
                            dl = (
                                vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])
                            ) / 2
                            rr = (
                                dl / run_info.lmax
                                + acc / tpar.dacc
                                + angle / tpar.dangle
                            ) / quali
                            curr_path_inf.register_link_candidate(
                                rr, fb.buf[2].num_parts
                            )
                            add_particle(fb.buf[2], X[3], philf)
                            num_added += 1
                    in_volume = 0

                    decision_time += time.perf_counter() - decision_start

        # end of inlist still zero
        # ***********************************

        # Emit full per-particle event for observer
        if observer is not None:
            observer.record({
                "step": step,
                "particle_id": h,
                "type": "tracked",
                "pos_3d": X[1].copy(),
                "predicted_3d": X[2].copy(),
                "prev_3d": X[0].copy() if curr_path_inf.prev_frame >= 0 else None,
                "has_prev": curr_path_inf.prev_frame >= 0,
                "search_center_px": v1.copy(),
                "search_rect": {
                    "xr": _obs_xr.copy(), "xl": _obs_xl.copy(),
                    "yd": _obs_yd.copy(), "yu": _obs_yu.copy(),
                } if observer is not None else {},
                "candidates": _obs_candidates,
                "inlist": curr_path_inf.inlist,
            })

        del w

        if debug and ((h + 1) % progress_interval == 0 or h + 1 == orig_parts):
            _tracker_log(
                "trackcorr",
                step,
                f"progress particle {h + 1}/{orig_parts} cand_rows={count2} two_stage_rows={count3}",
            )

    particle_loop_elapsed = time.perf_counter() - particle_loop_start

    # sort decis and give preliminary "finaldecis"
    sort_start = time.perf_counter()
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]

        if curr_path_inf.inlist > 0:
            curr_path_inf.decis, curr_path_inf.linkdecis = sort(
                curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis
            )
            curr_path_inf.finaldecis = curr_path_inf.decis[0]
            curr_path_inf.next_frame = curr_path_inf.linkdecis[0]
            # print(f"curr_path_inf.finaldecis {curr_path_inf.finaldecis}")
            # print(f"curr_path_inf.next_frame {curr_path_inf.next_frame}")

    # create links with decision check
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]

        if curr_path_inf.inlist > 0:
            ref_path_inf = fb.buf[2].path_info[curr_path_inf.next_frame]

            if ref_path_inf.prev_frame == PREV_NONE:
                # best choice wasn't used yet, so link is created
                ref_path_inf.prev_frame = h
                # print(f"link created {h}, ref_path_inf.next_frame {ref_path_inf.next_frame}")
            else:
                # best choice was already used by mega[2][mega[1][h].next_frame].prev_frame
                # check which is the better choice
                if (
                    fb.buf[1].path_info[ref_path_inf.prev_frame].finaldecis
                    > curr_path_inf.finaldecis
                ):
                    # remove link with prev
                    fb.buf[1].path_info[ref_path_inf.prev_frame].next_frame = NEXT_NONE
                    ref_path_inf.prev_frame = h
                else:
                    curr_path_inf.next_frame = NEXT_NONE

        if curr_path_inf.next_frame != NEXT_NONE:
            count1 += 1

    # end of creation of links with decision check
    sort_elapsed = time.perf_counter() - sort_start

    # Annotate observer events with final link decisions
    if observer is not None:
        for evt in observer.events_for_frame(step):
            h = evt["particle_id"]
            pi = fb.buf[1].path_info[h]
            evt["finaldecis"] = pi.finaldecis
            evt["next_frame"] = pi.next_frame
            if pi.next_frame >= 0 and pi.next_frame < len(fb.buf[2].path_info):
                evt["linked_3d"] = fb.buf[2].path_info[pi.next_frame].x.copy()
            else:
                evt["linked_3d"] = None

    print(
        f"step: {step}, curr: {fb.buf[1].num_parts}, next_frame: {fb.buf[2].num_parts}, \
            links: {count1}, lost: {fb.buf[1].num_parts - count1}, add: {num_added}"
    )

    if debug:
        _tracker_log(
            "trackcorr",
            step,
            (
                f"timings total={time.perf_counter() - step_start:.3f}s "
                f"particle_loop={particle_loop_elapsed:.3f}s link_sort={sort_elapsed:.3f}s"
                f" projection={proj_time:.3f}s cand_search={cand_search_time:.3f}s"
                f" decision={decision_time:.3f}s add={add_time:.3f}s"
            ),
        )

    # for the average of particles and links
    run_info.npart = run_info.npart + fb.buf[1].num_parts
    run_info.nlinks = run_info.nlinks + count1

    fb.fb_next()
    fb.write_frame_from_start(step)

    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, False)
    # end of sequence loop

    if profile is not None:
        profile.disable()
        _tracker_print_profile(profile, "trackcorr", step)


def trackcorr_c_finish(run_info, step: int):
    """Close the links and write the last frame."""
    track_range = run_info.seq_par.last - run_info.seq_par.first
    npart, nlinks = run_info.npart / track_range, run_info.nlinks / track_range
    print(
        f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart - nlinks:.1f}"
    )

    run_info.fb.fb_next()
    run_info.fb.write_frame_from_start(step)


def track3d_loop(run_info, step):
    """
    3D tracking loop - links particles in 3D space without camera projection.

    Three-level linking strategy:
    1. Particles with previous links: predict = 2*curr - prev
    2. No prev link, neighbors have links: predict = curr + avg_neighbor_velocity
    3. No prev link, no neighbor links: predict = curr

    Arguments:
        run_info - TrackingRun object
        step - current frame number
    """
    import math

    debug = _tracker_debug_enabled()
    profile = cProfile.Profile() if _tracker_profile_enabled() else None
    if profile is not None:
        profile.enable()

    step_start = time.perf_counter()
    fb = run_info.fb
    tpar = run_info.tpar

    prev = fb.buf[0]
    curr = fb.buf[1]
    next_buf = fb.buf[2]

    orig_parts = curr.num_parts
    count1 = 0

    dx = tpar.dvxmax
    dy = tpar.dvymax
    dz = tpar.dvzmax

    level1_elapsed = 0.0
    level2_elapsed = 0.0
    level3_elapsed = 0.0

    if debug:
        _tracker_log(
            "track3d",
            step,
            f"start curr_parts={curr.num_parts} next_parts={next_buf.num_parts} prev_parts={prev.num_parts}",
        )

    # Level 1: Particles with previous links
    level1_start = time.perf_counter()
    for i in range(orig_parts):
        curr_pi = curr.path_info[i]
        if curr_pi.prev_frame < 0:
            continue
        prev_idx = curr_pi.prev_frame
        if prev_idx < 0 or prev_idx >= prev.num_parts:
            continue
        prev_pi = prev.path_info[prev_idx]

        # Predict: 2*curr - prev
        predicted = 2 * curr_pi.x - prev_pi.x

        cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)

        decis = [0.0] * len(cand_indices)
        linkdecis = [0] * len(cand_indices)

        for k, cidx in enumerate(cand_indices):
            acc = 0.0
            for d in range(3):
                diff = curr_pi.x[d] - 2 * next_buf.path_info[cidx].x[d] + prev_pi.x[d]
                acc += diff * diff
            decis[k] = math.sqrt(acc)
            linkdecis[k] = cidx

        if len(cand_indices) > 1:
            sort(len(decis), decis, linkdecis)

        if cand_indices and next_buf.path_info[linkdecis[0]].prev_frame < 0:
            curr_pi.next_frame = linkdecis[0]
            next_buf.path_info[linkdecis[0]].prev_frame = i
            count1 += 1
        else:
            curr_pi.next_frame = -1

    level1_elapsed = time.perf_counter() - level1_start

    # Level 2: No previous link, but neighbors have previous links
    level2_start = time.perf_counter()
    for i in range(orig_parts):
        curr_pi = curr.path_info[i]
        if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
            continue

        vel = np.zeros(3)
        nvel = 0
        for j in range(orig_parts):
            if j == i:
                continue
            nbr = curr.path_info[j]
            if (
                abs(nbr.x[0] - curr_pi.x[0]) < dx
                and abs(nbr.x[1] - curr_pi.x[1]) < dy
                and abs(nbr.x[2] - curr_pi.x[2]) < dz
                and nbr.prev_frame >= 0
            ):
                vel += nbr.x - prev.path_info[nbr.prev_frame].x
                nvel += 1

        if nvel == 0:
            continue
        vel /= nvel
        predicted = curr_pi.x + vel

        cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)

        decis = [0.0] * len(cand_indices)
        linkdecis = [0] * len(cand_indices)

        for k, cidx in enumerate(cand_indices):
            acc = 0.0
            for d in range(3):
                diff = curr_pi.x[d] - 2 * next_buf.path_info[cidx].x[d] + predicted[d]
                acc += diff * diff
            decis[k] = math.sqrt(acc)
            linkdecis[k] = cidx

        if len(cand_indices) > 1:
            sort(len(decis), decis, linkdecis)

        if cand_indices and next_buf.path_info[linkdecis[0]].prev_frame < 0:
            curr_pi.next_frame = linkdecis[0]
            next_buf.path_info[linkdecis[0]].prev_frame = i
            count1 += 1
        else:
            curr_pi.next_frame = -1

    level2_elapsed = time.perf_counter() - level2_start

    # Level 3: No previous link, no neighbors with previous links
    level3_start = time.perf_counter()
    for i in range(orig_parts):
        curr_pi = curr.path_info[i]
        if curr_pi.prev_frame >= 0 or curr_pi.next_frame >= 0:
            continue

        predicted = curr_pi.x.copy()

        cand_indices = find_candidates_in_3d(next_buf, predicted, dx, dy, dz)

        decis = [0.0] * len(cand_indices)
        linkdecis = [0] * len(cand_indices)

        for k, cidx in enumerate(cand_indices):
            acc = 0.0
            for d in range(3):
                diff = curr_pi.x[d] - 2 * next_buf.path_info[cidx].x[d] + predicted[d]
                acc += diff * diff
            decis[k] = math.sqrt(acc)
            linkdecis[k] = cidx

        if len(cand_indices) > 1:
            sort(len(decis), decis, linkdecis)

        if cand_indices and next_buf.path_info[linkdecis[0]].prev_frame < 0:
            curr_pi.next_frame = linkdecis[0]
            next_buf.path_info[linkdecis[0]].prev_frame = i
            count1 += 1
        else:
            curr_pi.next_frame = -1

    level3_elapsed = time.perf_counter() - level3_start

    print(
        f"track3d step: {step}, curr: {fb.buf[1].num_parts}, "
        f"next: {fb.buf[2].num_parts}, links: {count1}"
    )

    if debug:
        _tracker_log(
            "track3d",
            step,
            (
                f"timings total={time.perf_counter() - step_start:.3f}s "
                f"level1={level1_elapsed:.3f}s level2={level2_elapsed:.3f}s level3={level3_elapsed:.3f}s"
            ),
        )

    run_info.npart += fb.buf[1].num_parts
    run_info.nlinks += count1

    fb.fb_next()
    fb.write_frame_from_start(step)

    if debug:
        _tracker_log("track3d", step, "frame rotated and written")

    if profile is not None:
        profile.disable()
        _tracker_print_profile(profile, "track3d", step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, 0)


def trackback_c(run_info: TrackingRun):
    """Trackback algorithm in C."""
    count1, count2, num_added, quali = 0, 0, 0, 0
    npart, nlinks = 0.0, 0.0
    Ymin = run_info.ymin
    Ymax = run_info.ymax

    philf = np.zeros((4, MAX_CANDS))
    X = np.empty((6, 3))
    n = np.empty((4, 2))
    v2 = np.empty((4, 2))

    fb = run_info.fb
    seq_par = run_info.seq_par
    tpar = run_info.tpar
    if not isinstance(tpar, TrackParTuple):
        tpar = convert_track_par_to_tuple(tpar)
    vpar = run_info.vpar
    cpar = run_info.cpar
    cal = run_info.cal
    raw_cal = run_info.raw_cal

    step = 0

    # Prime the buffer with first frames
    for step in range(seq_par.last, seq_par.last - 4, -1):
        fb.read_frame_at_end(step, read_links=True)
        fb.fb_next()

    fb.fb_prev()

    # sequence loop
    for step in range(seq_par.last - 1, seq_par.first, -1):
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            # We try to find link only if the forward search failed to.
            if curr_path_inf.next_frame < 0 or curr_path_inf.prev_frame != -1:
                continue

            X[:] = 0.0

            curr_path_inf.inlist = 0

            # 3D-position of current particle
            X[1] = vec_copy(curr_path_inf.x)

            # use information from previous to locate new search position
            # and to calculate values for search area
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.next_frame]
            X[0] = vec_copy(ref_path_inf.x)
            X[2] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            for j in range(fb.num_cams):
                n[j] = raw_cal[j].project(X[2])

            # calculate searchquader and reprojection in image space
            w = sorted_candidates_in_volume(X[2], n, fb.buf[2], run_info)

            if not (w.count == 1 and w.ftnr[0] == TR_UNUSED):
                count2 += 1

                i = 0
                while i < w.count and w.ftnr[i] != TR_UNUSED:
                    ref_path_inf = fb.buf[2].path_info[w.ftnr[i]]
                    X[3] = vec_copy(ref_path_inf.x)

                    diff_pos = vec_subt(X[1], X[3])
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])

                        # *********************check link *****************************
                        if (
                            acc < tpar.dacc
                            and angle < tpar.dangle
                            or acc < tpar.dacc / 10
                        ):
                            dl = (
                                vec_diff_norm(X[1], X[3]) + vec_diff_norm(X[0], X[1])
                            ) / 2  # type: ignore
                            quali = w.freq[i]
                            rr = (
                                dl / run_info.lmax
                                + acc / tpar.dacc
                                + angle / tpar.dangle
                            ) / quali
                            curr_path_inf.register_link_candidate(rr, w.ftnr[i])

                    i += 1

            # if old wasn't found try to create new particle position from rest
            if tpar.add:
                if curr_path_inf.inlist == 0:
                    quali, v2, philf = assess_new_position(X[2], fb.buf[2], run_info)
                    if quali >= 2:
                        # vec_copy(X[3], X[2])
                        in_volume = 0

                        _, X[3] = point_position(v2, fb.num_cams, cpar.mm, cal)

                        # volume check
                        if (
                            vpar.x_lay[0] < X[3][0] < vpar.x_lay[1]
                            and Ymin < X[3][1] < Ymax
                            and vpar.z_min_lay[0] < X[3][2] < vpar.z_max_lay[1]
                        ):
                            in_volume = 1

                        diff_pos = vec_subt(X[1], X[3])
                        if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                            angle, acc = angle_acc(X[1], X[2], X[3])

                            if (
                                acc < tpar.dacc
                                and angle < tpar.dangle
                                or acc < tpar.dacc / 10
                            ):
                                dl = (
                                    vec_diff_norm(X[1], X[3])
                                    + vec_diff_norm(X[0], X[1])
                                ) / 2  # type: ignore
                                rr = (
                                    dl / run_info.lmax
                                    + acc / tpar.dacc
                                    + angle / tpar.dangle
                                ) / (quali)
                                curr_path_inf.register_link_candidate(
                                    rr, fb.buf[2].num_parts
                                )

                                add_particle(fb.buf[2], X[3], philf)

                        in_volume = 0

        # end of h-loop
        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if curr_path_inf.inlist > 0:
                curr_path_inf.decis, curr_path_inf.linkdecis = sort(
                    curr_path_inf.inlist, curr_path_inf.decis, curr_path_inf.linkdecis
                )

        # create links with decision check
        count1 = 0
        num_added = 0

        for h in range(fb.buf[1].num_parts):
            curr_path_inf = fb.buf[1].path_info[h]

            if curr_path_inf.inlist > 0:
                ref_path_inf = fb.buf[2].path_info[curr_path_inf.linkdecis[0]]

                if (
                    ref_path_inf.prev_frame == PREV_NONE
                    and ref_path_inf.next_frame == NEXT_NONE
                ):
                    curr_path_inf.finaldecis = curr_path_inf.decis[0]
                    curr_path_inf.prev_frame = curr_path_inf.linkdecis[0]
                    fb.buf[2].path_info[curr_path_inf.prev_frame].next_frame = h
                    num_added += 1

                if (
                    ref_path_inf.prev_frame != PREV_NONE
                    and ref_path_inf.next_frame == NEXT_NONE
                ):
                    X[0] = vec_copy(
                        fb.buf[0].path_info[curr_path_inf.next_frame].x,
                    )
                    X[1] = vec_copy(curr_path_inf.x)
                    X[3] = vec_copy(ref_path_inf.x)
                    X[4] = vec_copy(fb.buf[3].path_info[ref_path_inf.prev_frame].x)

                    for j in range(3):
                        X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])

                    angle, acc = angle_acc(X[3], X[4], X[5])

                    if (acc < tpar.dacc and angle < tpar.dangle) or (
                        acc < tpar.dacc / 10
                    ):
                        curr_path_inf.finaldecis = curr_path_inf.decis[0]
                        curr_path_inf.prev_frame = curr_path_inf.linkdecis[0]
                        fb.buf[2].path_info[curr_path_inf.prev_frame].next_frame = h
                        num_added += 1

            if curr_path_inf.prev_frame != PREV_NONE:
                count1 += 1

        npart += fb.buf[1].num_parts
        nlinks += count1

        fb.fb_next()
        fb.write_frame_from_start(step)

        if step > seq_par.first + 2:
            fb.read_frame_at_end(step - 3, read_links=True)

        print(
            "step: {}, curr: {}, next_frame: {}, links: {}, lost: {}, add: {}".format(
                step,
                fb.buf[1].num_parts,
                fb.buf[2].num_parts,
                count1,
                fb.buf[1].num_parts - count1,
                num_added,
            )
        )

    npart /= seq_par.last - seq_par.first - 1
    nlinks /= seq_par.last - seq_par.first - 1

    print(
        f"Average over sequence, particles: {npart:.1f}, links: {nlinks:.1f}, lost: {npart - nlinks:.1f}"
    )

    fb.fb_next()
    fb.write_frame_from_start(step)

    return nlinks


default_naming = {
    "corres": "res/rt_is",
    "linkage": "res/ptv_is",
    "prio": "res/added",
}


class Tracker:
    """
    Workflow: instantiate, call restart() to initialize the frame buffer, then
    call either ``step_forward()`` while it still return True, then call
    ``finalize()`` to finish the run. Alternatively, ``full_forward()`` will
    do all this for you.

    This class matches the Cython Tracker API from bindings/optv/tracker.pyx
    for interchangeability with the optv engine.
    """

    def __init__(
        self,
        cpar: ControlPar,
        vpar: VolumePar,
        tpar: TrackParTuple,
        spar: SequencePar,
        cals: List[Calibration],
        naming: dict = None,
        flatten_tol: float = 0.0001,
    ):
        """
        Initialize the tracker.

        Arguments:
        ---------
        cpar: ControlPar object
        vpar: VolumePar object
        tpar: TrackParTuple object (use convert_track_par_to_tuple if needed)
        spar: SequencePar object
        cals: List of Calibration objects
        naming: Dictionary with naming rules for frame buffer files.
            Keys: 'corres', 'linkage', 'prio'. Default is default_naming.
        flatten_tol: Tolerance parameter for flattening operations.
        """
        # We need to keep a reference to the Python objects so that their
        # allocations are not freed.
        self._keepalive = (cpar, vpar, tpar, spar, cals)

        # Handle naming dictionary with defaults
        if naming is None:
            naming = default_naming
        else:
            naming = dict(naming)
            # Ensure all required keys are present
            for key in default_naming:
                if key not in naming:
                    naming[key] = default_naming[key]

        for key in ("corres", "linkage", "prio"):
            value = naming.get(key)
            if isinstance(value, bytes):
                naming[key] = value.decode("utf-8")

        self.run_info = TrackingRun(
            spar,
            tpar,
            vpar,
            cpar,
            TR_BUFSPACE,
            MAX_TARGETS,
            naming["corres"],
            naming["linkage"],
            naming["prio"],
            cals,
            flatten_tol,
        )
        self.step = self.run_info.seq_par.first

    def restart(self):
        """
        Prepare a tracking run. Sets up initial buffers and performs the
        one-time calculations used throughout the loop.
        """
        self.step = self.run_info.seq_par.first
        track_forward_start(self.run_info)

    def step_forward(self, observer=None):
        """
        Perform one tracking step for the current frame of iteration.

        Args:
            observer: Optional TrackingObserver to collect per-particle events.

        Returns:
            bool: True if more frames to process, False if done.
        """
        if self.step >= self.run_info.seq_par.last:
            return False

        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.step, "step_forward start")

        trackcorr_c_loop(self.run_info, self.step, observer=observer)

        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.step, "step_forward done")

        self.step += 1
        return True

    def finalize(self):
        """Finish a tracking run."""
        trackcorr_c_finish(self.run_info, self.step)

    def full_forward(self, observer=None):
        """Do a full tracking run from restart to finalize.

        Args:
            observer: Optional TrackingObserver to collect per-particle events.
        """
        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.run_info.seq_par.first, "full_forward start")

        track_forward_start(self.run_info)
        for step in range(self.run_info.seq_par.first, self.run_info.seq_par.last):
            trackcorr_c_loop(self.run_info, step, observer=observer)
        trackcorr_c_finish(self.run_info, self.run_info.seq_par.last)

        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.run_info.seq_par.last, "full_forward done")

        self.step = 0

    def step_forward_3d(self):
        """
        Perform one 3D tracking step for the current frame of iteration.

        Returns:
            bool: True if more frames to process, False if done.
        """
        if self.step >= self.run_info.seq_par.last:
            return False

        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.step, "step_forward_3d start")

        track3d_loop(self.run_info, self.step)

        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.step, "step_forward_3d done")

        self.step += 1
        return True

    def full_forward_3d(self):
        """Do a full 3D tracking run from restart to finalize."""
        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.run_info.seq_par.first, "full_forward_3d start")

        track_forward_start(self.run_info)
        for step in range(self.run_info.seq_par.first, self.run_info.seq_par.last):
            track3d_loop(self.run_info, step)
        trackcorr_c_finish(self.run_info, self.run_info.seq_par.last)

        if _tracker_debug_enabled():
            _tracker_log("Tracker", self.run_info.seq_par.last, "full_forward_3d done")

        self.step = 0

    def full_backward(self):
        """
        Do a full backward run on existing tracking results.

        Note: Results must exist or this will fail.
        """
        trackback_c(self.run_info)

    def current_step(self):
        """Return the current step number."""
        return self.step

    def track_with_viz(
        self,
        callback,
        on_particle=None,
        on_algorithm_step=None,
    ):
        """
        Track with visualization callbacks - Python engine only.

        This method wraps the tracking loop to inject callbacks at key points,
        enabling real-time visualization of the tracking process.

        Arguments:
        ---------
        callback: Function called after each frame.
            Signature: callback(frame_num: int, state: dict)
            The state dict contains:
                - 'particles': np.ndarray (N, 3) - 3D particle positions
                - 'correspondences': np.ndarray (N, 5) - correspondence data
                - 'added_count': int - particles added in this frame
                - 'lost_count': int - particles lost in this frame
        on_particle: Optional function called for each tracked particle.
            Signature: on_particle(frame_num: int, particle_id: int, details: dict)
        on_algorithm_step: Optional function called during algorithm steps.
            Signature: on_algorithm_step(step_name: str, details: dict)

        Yields:
        -------
        dict: State dictionary after each frame (same format as callback)

        Example:
        -------
        >>> tracker = Tracker(cpar, vpar, tpar, spar, cals)
        >>> def viz_callback(frame_num, state):
        ...     print(f"Frame {frame_num}: {len(state['particles'])} particles")
        >>> for state in tracker.track_with_viz(viz_callback):
        ...     pass  # Process state
        """
        self.restart()

        while self.step_forward():
            state = self._get_current_state()

            # Call per-frame callback
            callback(self.current_step(), state)

            yield state

        self.finalize()

    def _get_current_state(self):
        """
        Extract current tracking state as NumPy arrays.

        Returns:
        -------
        dict: Current state with keys:
            - 'frame_number': int
            - 'particles': np.ndarray (N, 3) - 3D positions
            - 'correspondences': np.ndarray (N, 5) - correspondence data
            - 'added_count': int
            - 'lost_count': int
        """
        fb = self.run_info.fb

        # Extract particle positions (3D)
        if fb.num_parts > 0:
            particles = np.array([list(fb.path_info[i].x) for i in range(fb.num_parts)])

            # Extract correspondences
            correspondences = np.column_stack([
                fb.corres_nr[:fb.num_parts],
                fb.corres_p[:fb.num_parts],
            ]) if fb.num_parts > 0 else np.empty((0, 5), dtype=np.int32)
        else:
            particles = np.empty((0, 3))
            correspondences = np.empty((0, 5), dtype=np.int32)

        # Count added/lost (simplified - would need more tracking for exact counts)
        added_count = fb.num_parts
        lost_count = 0  # Would need to track across frames for accurate count

        return {
            "frame_number": self.current_step(),
            "particles": particles,
            "correspondences": correspondences,
            "added_count": added_count,
            "lost_count": lost_count,
        }
