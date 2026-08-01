"""Multi-camera correspondence matching.

Translation of lib/src/correspondences.c and lib/include/correspondences.h.

Establishes correspondences between detected targets across 2-4 cameras
using epipolar geometry and clique finding.

Adjacency data is stored in flat typed memoryview arrays (not Python objects),
so that the O(n^4) clique-finding loops compile to pure C pointer arithmetic.
"""

import operator

import cython
import numpy as np

from .epi import MAXCAND

NMAX = 20240
PT_UNUSED = -999


# ---------------------------------------------------------------------------
# Output data type — NTupel is the external result of the matching pipeline.
# It is only created / consumed once per frame (not in hot loops), so the
# Python-level overhead is negligible.
# ---------------------------------------------------------------------------


@cython.cclass
class NTupel:
    """A correspondence match across multiple cameras."""

    p: list = cython.declare(object, visibility="public")
    corr: cython.double = cython.declare(cython.double, visibility="public")

    def __init__(self, p=None, corr=0.0):
        if p is None:
            self.p = [-1, -1, -1, -1]
        else:
            self.p = p
        self.corr = corr


# ---------------------------------------------------------------------------
# Sorting helpers (preserved for test compatibility)
# ---------------------------------------------------------------------------


@cython.ccall
def quicksort_target_y(pix):
    """Sort target list by y coordinate in place using Timsort."""

    pix.sort(key=operator.attrgetter("y"))


@cython.ccall
def quicksort_coord2d_x(crd):
    """Sort Coord2d list by x coordinate in place using Timsort."""

    crd.sort(key=operator.attrgetter("x"))


# ---------------------------------------------------------------------------
# Flat-array adjacency storage
#
# Instead of a Python list-of-lists-of-Correspond-objects we use 5
# contiguous typed arrays indexed by (c1, c2, i[, j]):
#
#   p1_arr[c1, c2, i]   — p1 of the i-th target in camera c1 w.r.t. c2
#   n_arr[c1, c2, i]    — number of candidate matches for that target
#   p2_arr[c1, c2, i, j]   — candidate target index in camera c2
#   corr_arr[c1, c2, i, j] — correlation value
#   dist_arr[c1, c2, i, j] — distance (tolerance)
#
# Only the upper triangle (c1 < c2) is used, matching the original C code.
# Dimensions are (num_cams, num_cams, max_targets[, MAXCAND]).
# ---------------------------------------------------------------------------


@cython.ccall
def allocate_adjacency_arrays(num_cams: cython.int, target_counts):
    """Allocate flat adjacency arrays.

    Args:
        num_cams: number of cameras (2-4).
        target_counts: per-camera target count (length num_cams).

    Returns:
        Tuple (p1_arr, n_arr, p2_arr, corr_arr, dist_arr).
    """
    max_targets = max(target_counts)
    tc_view = np.asarray(target_counts, dtype=np.int32)

    p1_arr = np.full((num_cams, num_cams, max_targets), -1, dtype=np.int32)
    n_arr = np.zeros((num_cams, num_cams, max_targets), dtype=np.int32)
    p2_arr = np.zeros((num_cams, num_cams, max_targets, MAXCAND + 1), dtype=np.int32)
    corr_arr = np.zeros(
        (num_cams, num_cams, max_targets, MAXCAND + 1), dtype=np.float64
    )
    dist_arr = np.zeros(
        (num_cams, num_cams, max_targets, MAXCAND + 1), dtype=np.float64
    )

    c1: cython.int
    c2: cython.int
    i: cython.int
    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            for i in range(tc_view[c1]):
                p1_arr[c1, c2, i] = i

    return p1_arr, n_arr, p2_arr, corr_arr, dist_arr


# ---------------------------------------------------------------------------
# Per-pair candidate search
# ---------------------------------------------------------------------------


def _build_adjacency_for_pair(
    i1: cython.int,
    i2: cython.int,
    n_arr,
    p2_arr,
    corr_arr,
    dist_arr,
    corrected,
    frm,
    vpar,
    cpar,
    calib,
):
    """Build adjacency entries for one camera pair into flat arrays.

    Module-level for compatibility with ThreadPoolExecutor.

    Uses pre-allocated output arrays for find_candidate (no Python list or
    Candidate objects in the hot path).
    """
    from .epi import epi_mm, find_candidate

    i: cython.int
    j: cython.int
    count: cython.int
    pt1: cython.int

    # Pre-allocated scratch arrays for find_candidate output (reused per target)
    _cand_pnr = np.empty(MAXCAND + 1, dtype=np.int32)
    _cand_tol = np.empty(MAXCAND + 1, dtype=np.float64)
    _cand_corr = np.empty(MAXCAND + 1, dtype=np.float64)

    for i in range(frm.num_targets[i1]):
        if corrected[i1][i].x == PT_UNUSED:
            continue

        xmin, ymin, xmax, ymax = epi_mm(
            corrected[i1][i].x,
            corrected[i1][i].y,
            calib[i1],
            calib[i2],
            cpar.mm,
            vpar,
        )

        pt1 = corrected[i1][i].pnr
        count = find_candidate(
            corrected[i2],
            frm.targets[i2],
            frm.num_targets[i2],
            xmin,
            ymin,
            xmax,
            ymax,
            frm.targets[i1][pt1].n,
            frm.targets[i1][pt1].nx,
            frm.targets[i1][pt1].ny,
            frm.targets[i1][pt1].sumg,
            _cand_pnr,
            _cand_tol,
            _cand_corr,
            vpar,
            cpar,
            calib[i2],
        )

        if count > MAXCAND:
            count = MAXCAND

        # 1D views for (i1, i2, i) row — 4D write becomes 1D
        p2_row: cython.int[:] = p2_arr[i1, i2, i]
        corr_row: cython.double[:] = corr_arr[i1, i2, i]
        dist_row: cython.double[:] = dist_arr[i1, i2, i]
        for j in range(count):
            cand_p = _cand_pnr[j]
            p2_row[j] = cand_p
            corr_row[j] = _cand_corr[j]
            dist_row[j] = _cand_tol[j]
        n_arr[i1, i2, i] = count

    return i1, i2


# ---------------------------------------------------------------------------
# Build all pairwise adjacency arrays
# ---------------------------------------------------------------------------


@cython.ccall
@cython.locals(num_cams=cython.int, i1=cython.int, i2=cython.int)
def match_pairs(
    p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, calib
):
    """Build pairwise adjacency for all camera pairs.

    Args:
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr: flat adjacency arrays
            (from allocate_adjacency_arrays).
        corrected: per-camera x-sorted Coord2d arrays.
        frm: Frame object with targets and num_targets.
        vpar: VolumePar.
        cpar: ControlPar.
        calib: list of Calibration objects.
    """
    num_cams: cython.int = cpar.num_cams

    pairs = [(i1, i2) for i1 in range(num_cams - 1) for i2 in range(i1 + 1, num_cams)]

    if len(pairs) <= 1:
        # Sequential for 2 cameras
        for i1, i2 in pairs:
            _build_adjacency_for_pair(
                i1,
                i2,
                n_arr,
                p2_arr,
                corr_arr,
                dist_arr,
                corrected,
                frm,
                vpar,
                cpar,
                calib,
            )
        return

    # Multi-threaded: each camera pair is independent
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        futures = {
            pool.submit(
                _build_adjacency_for_pair,
                i1,
                i2,
                n_arr,
                p2_arr,
                corr_arr,
                dist_arr,
                corrected,
                frm,
                vpar,
                cpar,
                calib,
            ): (i1, i2)
            for i1, i2 in pairs
        }
        for future in as_completed(futures):
            future.result()  # propagate exceptions


# ---------------------------------------------------------------------------
# Clique-finding — flat-array versions of the original C matching functions.
#
# All indexing like  p2_arr[c1, c2, i, j]  compiles to direct C pointer
# arithmetic when this module is compiled via Cython.
# ---------------------------------------------------------------------------


@cython.ccall
@cython.locals(
    matched=cython.int,
    i=cython.int,
    j=cython.int,
    k=cython.int,
    l=cython.int,
    m=cython.int,
    n=cython.int,
    o=cython.int,
    p1=cython.int,
    p2=cython.int,
    p3=cython.int,
    p4=cython.int,
    p31=cython.int,
    p41=cython.int,
    p42=cython.int,
    corr=cython.double,
)
def four_camera_matching(
    p1_arr: cython.int[:, :, :],
    n_arr: cython.int[:, :, :],
    p2_arr: cython.int[:, :, :, :],
    corr_arr: cython.double[:, :, :, :],
    dist_arr: cython.double[:, :, :, :],
    base_target_count: cython.int,
    accept_corr: cython.double,
    scratch_p: cython.int[:, :],
    scratch_corr: cython.double[:],
    scratch_size: cython.int,
) -> cython.int:
    """Find consistent 4-camera correspondences (quadruplets).

    Access is  p2_arr[c1, c2, i, j]  = direct C pointer dereference when
    compiled, versus  lists[c1][c2][i].p2[j]  = 6 Python operations before.

    Args:
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr: flat adjacency arrays.
        base_target_count: number of targets in camera 0.
        accept_corr: minimum acceptable correlation.
        scratch: output NTupel list.
        scratch_size: capacity of scratch.

    Returns:
        Number of cliques found.
    """
    matched: cython.int = 0

    # scratch_p0 = p1_arr[0, 1, i] is the reference target in camera 0 for this i
    for i in range(base_target_count):
        p1: cython.int = p1_arr[0, 1, i]

        # Slice 1D views from 4D arrays — inner loops become 1D access (score-0)
        n_01_i: cython.int = n_arr[0, 1, i]
        n_02_i: cython.int = n_arr[0, 2, i]
        n_03_i: cython.int = n_arr[0, 3, i]
        p2_01_i: cython.int[:] = p2_arr[0, 1, i]
        c01_i: cython.double[:] = corr_arr[0, 1, i]
        d01_i: cython.double[:] = dist_arr[0, 1, i]
        p2_02_i: cython.int[:] = p2_arr[0, 2, i]
        c02_i: cython.double[:] = corr_arr[0, 2, i]
        d02_i: cython.double[:] = dist_arr[0, 2, i]
        p2_03_i: cython.int[:] = p2_arr[0, 3, i]
        c03_i: cython.double[:] = corr_arr[0, 3, i]
        d03_i: cython.double[:] = dist_arr[0, 3, i]

        for j in range(n_01_i):
            p2: cython.int = p2_01_i[j]
            c01: cython.double = c01_i[j]
            d01: cython.double = d01_i[j]

            # Rows depending only on p2 — hoisted out of the k/l loops
            n_12_p2: cython.int = n_arr[1, 2, p2]
            n_13_p2: cython.int = n_arr[1, 3, p2]
            p2_12_p2: cython.int[:] = p2_arr[1, 2, p2]
            c12_p2: cython.double[:] = corr_arr[1, 2, p2]
            d12_p2: cython.double[:] = dist_arr[1, 2, p2]
            p2_13_p2: cython.int[:] = p2_arr[1, 3, p2]
            c13_p2: cython.double[:] = corr_arr[1, 3, p2]
            d13_p2: cython.double[:] = dist_arr[1, 3, p2]

            for k in range(n_02_i):
                p3: cython.int = p2_02_i[k]
                c02: cython.double = c02_i[k]
                d02: cython.double = d02_i[k]

                # Rows depending only on p3 — hoisted out of the l loop
                n_23_p3: cython.int = n_arr[2, 3, p3]
                p2_23_p3: cython.int[:] = p2_arr[2, 3, p3]
                c23_p3: cython.double[:] = corr_arr[2, 3, p3]
                d23_p3: cython.double[:] = dist_arr[2, 3, p3]

                for target_idx in range(n_03_i):
                    p4: cython.int = p2_03_i[target_idx]
                    c03: cython.double = c03_i[target_idx]
                    d03: cython.double = d03_i[target_idx]

                    corr_partial: cython.double = c01 + c02 + c03
                    dist_partial: cython.double = d01 + d02 + d03

                    for m in range(n_12_p2):
                        p31: cython.int = p2_12_p2[m]
                        if p3 != p31:
                            continue

                        for n in range(n_13_p2):
                            p41: cython.int = p2_13_p2[n]
                            if p4 != p41:
                                continue

                            for o in range(n_23_p3):
                                p42: cython.int = p2_23_p3[o]
                                if p4 != p42:
                                    continue

                                corr: cython.double = (
                                    corr_partial + c12_p2[m] + c13_p2[n] + c23_p3[o]
                                ) / (dist_partial + d12_p2[m] + d13_p2[n] + d23_p3[o])

                                if corr <= accept_corr:
                                    continue

                                scratch_p[matched, 0] = p1
                                scratch_p[matched, 1] = p2
                                scratch_p[matched, 2] = p3
                                scratch_p[matched, 3] = p4
                                scratch_corr[matched] = corr

                                matched += 1
                                if matched == scratch_size:
                                    return matched
    return matched


@cython.ccall
@cython.locals(
    matched=cython.int,
    i1=cython.int,
    i2=cython.int,
    i3=cython.int,
    i=cython.int,
    j=cython.int,
    k=cython.int,
    m_idx=cython.int,
    nc=cython.int,
    p1=cython.int,
    p2=cython.int,
    p3=cython.int,
    corr=cython.double,
)
def three_camera_matching(
    p1_arr: cython.int[:, :, :],
    n_arr: cython.int[:, :, :],
    p2_arr: cython.int[:, :, :, :],
    corr_arr: cython.double[:, :, :, :],
    dist_arr: cython.double[:, :, :, :],
    num_cams: cython.int,
    target_counts,
    accept_corr: cython.double,
    scratch_p: cython.int[:, :],
    scratch_corr: cython.double[:],
    scratch_size: cython.int,
    tusage: cython.int[:, :],
) -> cython.int:
    """Find consistent 3-camera correspondences (triplets)."""
    matched: cython.int = 0
    tc = np.asarray(target_counts, dtype=np.int32)

    for i1 in range(num_cams - 2):
        for i in range(tc[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1: cython.int = p1_arr[i1, i2, i]
                if p1 > NMAX or tusage[i1, p1] > 0:
                    continue

                # 1D views for (i1, i2, i) row
                n_i1i2_i: cython.int = n_arr[i1, i2, i]
                p2_i1i2_i: cython.int[:] = p2_arr[i1, i2, i]
                c_i1i2_i: cython.double[:] = corr_arr[i1, i2, i]
                d_i1i2_i: cython.double[:] = dist_arr[i1, i2, i]

                for j in range(n_i1i2_i):
                    p2: cython.int = p2_i1i2_i[j]
                    if p2 > NMAX or tusage[i2, p2] > 0:
                        continue

                    c12: cython.double = c_i1i2_i[j]
                    d12: cython.double = d_i1i2_i[j]

                    for i3 in range(i2 + 1, num_cams):
                        # 1D views for (i1, i3, i) row
                        n_i1i3_i: cython.int = n_arr[i1, i3, i]
                        p2_i1i3_i: cython.int[:] = p2_arr[i1, i3, i]
                        c_i1i3_i: cython.double[:] = corr_arr[i1, i3, i]
                        d_i1i3_i: cython.double[:] = dist_arr[i1, i3, i]

                        # 1D views for (i2, i3, p2) row — invariant over k
                        n_i2i3_p2: cython.int = n_arr[i2, i3, p2]
                        p2_i2i3_p2: cython.int[:] = p2_arr[i2, i3, p2]
                        c_i2i3_p2: cython.double[:] = corr_arr[i2, i3, p2]
                        d_i2i3_p2: cython.double[:] = dist_arr[i2, i3, p2]

                        for k in range(n_i1i3_i):
                            p3: cython.int = p2_i1i3_i[k]
                            if p3 > NMAX or tusage[i3, p3] > 0:
                                continue

                            c13: cython.double = c_i1i3_i[k]
                            d13: cython.double = d_i1i3_i[k]
                            corr_partial: cython.double = c12 + c13
                            dist_partial: cython.double = d12 + d13

                            for m_idx in range(n_i2i3_p2):
                                if p3 != p2_i2i3_p2[m_idx]:
                                    continue

                                corr: cython.double = (
                                    corr_partial + c_i2i3_p2[m_idx]
                                ) / (dist_partial + d_i2i3_p2[m_idx])

                                if corr <= accept_corr:
                                    continue

                                for nc in range(num_cams):
                                    scratch_p[matched, nc] = -2

                                scratch_p[matched, i1] = p1
                                scratch_p[matched, i2] = p2
                                scratch_p[matched, i3] = p3
                                scratch_corr[matched] = corr

                                matched += 1
                                if matched == scratch_size:
                                    return matched
    return matched


@cython.ccall
@cython.locals(
    matched=cython.int,
    i1=cython.int,
    i2=cython.int,
    i=cython.int,
    nc=cython.int,
    p1=cython.int,
    p2=cython.int,
    corr=cython.double,
)
def consistent_pair_matching(
    p1_arr: cython.int[:, :, :],
    n_arr: cython.int[:, :, :],
    p2_arr: cython.int[:, :, :, :],
    corr_arr: cython.double[:, :, :, :],
    dist_arr: cython.double[:, :, :, :],
    num_cams: cython.int,
    target_counts,
    accept_corr: cython.double,
    scratch_p: cython.int[:, :],
    scratch_corr: cython.double[:],
    scratch_size: cython.int,
    tusage: cython.int[:, :],
) -> cython.int:
    """Find unambiguous 2-camera pairs."""
    matched: cython.int = 0
    tc = np.asarray(target_counts, dtype=np.int32)

    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(tc[i1]):
                p1: cython.int = p1_arr[i1, i2, i]
                if p1 > NMAX or tusage[i1, p1] > 0:
                    continue

                if n_arr[i1, i2, i] != 1:
                    continue

                # 1D view for (i1, i2, i) row — avoid 4D strided access
                p2_row: cython.int[:] = p2_arr[i1, i2, i]
                c_row: cython.double[:] = corr_arr[i1, i2, i]
                d_row: cython.double[:] = dist_arr[i1, i2, i]
                p2: cython.int = p2_row[0]
                if p2 > NMAX or tusage[i2, p2] > 0:
                    continue

                corr: cython.double = c_row[0] / d_row[0]
                if corr <= accept_corr:
                    continue

                for nc in range(num_cams):
                    scratch_p[matched, nc] = -2

                scratch_p[matched, i1] = p1
                scratch_p[matched, i2] = p2
                scratch_corr[matched] = corr

                matched += 1
                if matched == scratch_size:
                    return matched
    return matched


# ---------------------------------------------------------------------------
# Candidate sorting (still uses .corr attribute on NTupel — not on hot path)
# ---------------------------------------------------------------------------


@cython.ccall
@cython.locals(
    taken=cython.int,
    cam=cython.int,
    tnum=cython.int,
    has_used=cython.bint,
)
def take_best_candidates(
    src_p: cython.int[:, :],
    src_corr: cython.double[:],
    dst_p: cython.int[:, :],
    dst_corr: cython.double[:],
    num_cams: cython.int,
    num_cands: cython.int,
    tusage: cython.int[:, :],
    dst_offset: cython.int,
) -> cython.int:
    """Take candidates by descending correlation, skipping used targets."""
    order = np.argsort(src_corr[:num_cands])[::-1]

    taken: cython.int = 0

    for idx in range(num_cands):
        cand: cython.int = order[idx]
        has_used: cython.bint = False
        for cam in range(num_cams):
            tnum: cython.int = src_p[cand, cam]
            if tnum > -1 and tusage[cam, tnum] > 0:
                has_used = True
                break

        if has_used:
            continue

        for cam in range(num_cams):
            tnum = src_p[cand, cam]
            if tnum > -1:
                tusage[cam, tnum] += 1

        for cam in range(num_cams):
            dst_p[dst_offset + taken, cam] = src_p[cand, cam]
        dst_corr[dst_offset + taken] = src_corr[cand]
        taken += 1
    return taken


# ---------------------------------------------------------------------------
# Coordinate correction (per-camera pixel→metric→flat, x-sorted)
# ---------------------------------------------------------------------------


def _correct_one_camera(cam, frm, calib, cpar, tol):
    """Process a single camera (module-level for pickling)."""
    from .epi import Coord2d
    from .trafo import dist_to_flat, pixel_to_metric

    cam_coords = []
    for part in range(frm.num_targets[cam]):
        t = frm.targets[cam][part]
        xm, ym = pixel_to_metric(t.x, t.y, cpar)

        ap = calib[cam].added_par
        ip = calib[cam].int_par
        fx, fy = dist_to_flat(
            xm,
            ym,
            ip.xh,
            ip.yh,
            ap.k1,
            ap.k2,
            ap.k3,
            ap.p1,
            ap.p2,
            ap.scx,
            ap.she,
            tol,
        )

        cam_coords.append(Coord2d(pnr=t.pnr, x=fx, y=fy))

    cam_coords.sort(key=operator.attrgetter("x"))
    return cam_coords


@cython.ccall
@cython.locals(num_cams=cython.int)
def correct_frame(frm, calib, cpar, tol):
    """Pixel → metric → flat coordinates, x-sorted per camera.

    Args:
        frm: Frame object.
        calib: list of Calibration objects.
        cpar: ControlPar.
        tol: tolerance for iterative flattening.

    Returns:
        list of lists of Coord2d, one per camera, x-sorted.
    """
    num_cams: cython.int = cpar.num_cams

    if num_cams <= 1:
        corrected = []
        for cam in range(num_cams):
            corrected.append(_correct_one_camera(cam, frm, calib, cpar, tol))
        return corrected

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=num_cams) as pool:
        futures = [
            pool.submit(_correct_one_camera, cam, frm, calib, cpar, tol)
            for cam in range(num_cams)
        ]
        corrected = [f.result() for f in futures]
    return corrected


# ---------------------------------------------------------------------------
# Candidate-saturation diagnostics
# ---------------------------------------------------------------------------

_saturation_warned = False


def _warn_candidate_saturation(n_arr, calib, cpar):
    """Warn once if epipolar candidate lists hit MAXCAND.

    Saturation means some targets collect the maximum number of epipolar
    candidates, which makes the clique search combinatorially expensive and
    the matching unreliable. The usual cause is an over-fitted distortion
    model whose undistortion folds image-edge coordinates on top of the
    central ones.
    """
    global _saturation_warned
    if _saturation_warned:
        return
    n = np.asarray(n_arr)
    saturated = int((n >= MAXCAND).sum())
    if saturated == 0:
        return

    from .trafo import radial_distortion_folds

    r_max = 0.5 * float(np.hypot(cpar.imx * cpar.pix_x, cpar.imy * cpar.pix_y))
    folding = []
    for cam, cal in enumerate(calib):
        r_fold = radial_distortion_folds(cal.added_par, r_max)
        if r_fold is not None:
            folding.append(f"cam{cam + 1} (folds at r={r_fold:.2f} mm)")

    msg = (
        f"{saturated} epipolar candidate lists hit MAXCAND={MAXCAND}; "
        "correspondence matching will be slow and unreliable. This usually "
        "indicates an over-fitted distortion model (.addpar)."
    )
    if folding:
        msg += (
            " Radial distortion is non-monotonic within the sensor "
            f"(half-diagonal {r_max:.2f} mm) for: {', '.join(folding)}. "
            "Re-calibrate these cameras with fewer additional parameters "
            "(e.g. k1 only) or with calibration targets covering the image "
            "corners."
        )
    else:
        msg += (
            " Consider re-calibrating with fewer additional parameters or "
            "reducing the epipolar band width (eps0)."
        )
    import warnings

    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    _saturation_warned = True


# ---------------------------------------------------------------------------
# Main correspondence pipeline
# ---------------------------------------------------------------------------


@cython.ccall
@cython.locals(
    num_cams=cython.int,
    i=cython.int,
    j=cython.int,
    k=cython.int,
    p1=cython.int,
    con0_size=cython.int,
)
def correspondences(frm, corrected, vpar, cpar, calib):
    """Full correspondence matching pipeline.

    Args:
        frm: Frame object.
        corrected: per-camera x-sorted Coord2d arrays (from correct_frame).
        vpar: VolumePar.
        cpar: ControlPar.
        calib: list of Calibration objects.

    Returns:
        (con, match_counts) where con is the list of NTupel correspondences
        and match_counts is [quads, trips, pairs, total].
    """
    num_cams: cython.int = cpar.num_cams
    con0_size: cython.int = num_cams * NMAX

    # Flat scratch arrays — 98× faster than [NTupel() for _ in range(con0_size)]
    con0_p: cython.int[:, :] = np.full((con0_size, num_cams), -1, dtype=np.int32)
    con0_corr: cython.double[:] = np.zeros(con0_size, dtype=np.float64)
    con_p: cython.int[:, :] = np.full((con0_size, num_cams), -1, dtype=np.int32)
    con_corr: cython.double[:] = np.zeros(con0_size, dtype=np.float64)
    tusage: cython.int[:, :] = np.zeros((num_cams, NMAX), dtype=np.int32)

    # Allocate flat adjacency arrays
    p1_arr, n_arr, p2_arr, corr_arr, dist_arr = allocate_adjacency_arrays(
        num_cams, frm.num_targets
    )

    match_counts = [0, 0, 0, 0]

    # Build adjacency
    match_pairs(
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr, corrected, frm, vpar, cpar, calib
    )
    _warn_candidate_saturation(n_arr, calib, cpar)

    # 4-camera cliques
    if num_cams == 4:
        match0: cython.int = four_camera_matching(
            p1_arr,
            n_arr,
            p2_arr,
            corr_arr,
            dist_arr,
            frm.num_targets[0],
            vpar.corrmin,
            con0_p,
            con0_corr,
            4 * NMAX,
        )
        match_counts[0] = take_best_candidates(
            con0_p, con0_corr, con_p, con_corr, num_cams, match0, tusage, 0
        )
        match_counts[3] += match_counts[0]

    # 3-camera cliques
    if (num_cams == 4 and cpar.allCam_flag == 0) or num_cams == 3:
        match0 = three_camera_matching(
            p1_arr,
            n_arr,
            p2_arr,
            corr_arr,
            dist_arr,
            num_cams,
            frm.num_targets,
            vpar.corrmin,
            con0_p,
            con0_corr,
            4 * NMAX,
            tusage,
        )
        match_counts[1] = take_best_candidates(
            con0_p,
            con0_corr,
            con_p,
            con_corr,
            num_cams,
            match0,
            tusage,
            match_counts[3],
        )
        match_counts[3] += match_counts[1]

    # 2-camera pairs
    if num_cams > 1 and cpar.allCam_flag == 0:
        match0 = consistent_pair_matching(
            p1_arr,
            n_arr,
            p2_arr,
            corr_arr,
            dist_arr,
            num_cams,
            frm.num_targets,
            vpar.corrmin,
            con0_p,
            con0_corr,
            4 * NMAX,
            tusage,
        )
        match_counts[2] = take_best_candidates(
            con0_p,
            con0_corr,
            con_p,
            con_corr,
            num_cams,
            match0,
            tusage,
            match_counts[3],
        )
        match_counts[3] += match_counts[2]

    # Update target track numbers
    total: cython.int = match_counts[3]
    for i in range(total):
        for j in range(num_cams):
            if con_p[i, j] < 0:
                continue
            p1 = corrected[j][con_p[i, j]].pnr
            if p1 > -1 and p1 < 1202590843:
                frm.targets[j][p1].tnr = i

    # Reconstruct NTupel list only for the matched entries (minimal allocation)
    con = [
        NTupel(p=[con_p[i, c] for c in range(num_cams)], corr=con_corr[i])
        for i in range(total)
    ]
    return con, match_counts


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
