"""Multi-camera correspondence matching.

Translation of lib/src/correspondences.c and lib/include/correspondences.h.

Establishes correspondences between detected targets across 2-4 cameras
using epipolar geometry and clique finding.
"""
import cython


import numpy as np
from dataclasses import dataclass, field

from .epi import MAXCAND

NMAX = 20240
PT_UNUSED = -999


def _default_ntupel_p():
    return [-1, -1, -1, -1]


def _default_correspond_p2():
    return np.zeros(MAXCAND, dtype=np.int32)


def _default_correspond_corr():
    return np.zeros(MAXCAND, dtype=np.float64)


def _default_correspond_dist():
    return np.zeros(MAXCAND, dtype=np.float64)


@cython.cclass
@dataclass
class NTupel:
    """A correspondence match across multiple cameras."""
    p: list = field(default_factory=_default_ntupel_p)
    corr: cython.double = 0.0


@cython.cclass
@dataclass
class Correspond:
    """Adjacency list entry for candidate matching.

    Matches C correspond struct: indexed by target index in source camera.
    """
    p1: cython.int = 0
    n: cython.int = 0
    p2: np.ndarray = field(default_factory=_default_correspond_p2)
    corr: np.ndarray = field(default_factory=_default_correspond_corr)
    dist: np.ndarray = field(default_factory=_default_correspond_dist)


@cython.ccall
def quicksort_target_y(pix):
    """Sort target list by y coordinate in place."""
    i: cython.int
    j: cython.int
    for i in range(1, len(pix)):
        item = pix[i]
        j = i
        while j > 0 and pix[j - 1].y > item.y:
            pix[j] = pix[j - 1]
            j -= 1
        pix[j] = item


@cython.ccall
def quicksort_coord2d_x(crd):
    """Sort Coord2d list by x coordinate in place."""
    i: cython.int
    j: cython.int
    for i in range(1, len(crd)):
        item = crd[i]
        j = i
        while j > 0 and crd[j - 1].x > item.x:
            crd[j] = crd[j - 1]
            j -= 1
        crd[j] = item


@cython.ccall
def safely_allocate_adjacency_lists(num_cams, target_counts):
    """Allocate pairwise adjacency lists.

    Returns lists[c1][c2] as a 2D list where lists[c1][c2] is an array
    of Correspond objects of length target_counts[c1], for c1 < c2.
    """
    lists = [[None] * num_cams for _ in range(num_cams)]
    c1: cython.int
    c2: cython.int
    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            lists[c1][c2] = [Correspond(p1=0, n=0) for _ in range(target_counts[c1])]
    return lists


@cython.ccall
def match_pairs(lists, corrected, frm, vpar, cpar, calib):
    """Build pairwise adjacency lists between all camera pairs.

    Matches C match_pairs exactly. For each target in camera i1,
    projects epipolar lines into camera i2 and finds candidate matches.

    Args:
        lists: adjacency lists[c1][c2], allocated by safely_allocate_adjacency_lists.
        corrected: per-camera x-sorted Coord2d arrays.
        frm: Frame object with targets and num_targets.
        vpar: VolumePar.
        cpar: ControlPar.
        calib: list of Calibration objects.
    """
    from .epi import epi_mm, find_candidate, Candidate

    i1: cython.int
    i2: cython.int
    i: cython.int
    j: cython.int
    pt1: cython.int
    count: cython.int
    for i1 in range(cpar.num_cams - 1):
        for i2 in range(i1 + 1, cpar.num_cams):
            for i in range(frm.num_targets[i1]):
                if corrected[i1][i].x == PT_UNUSED:
                    continue

                xmin, ymin, xmax, ymax = epi_mm(
                    corrected[i1][i].x, corrected[i1][i].y,
                    calib[i1], calib[i2], cpar.mm, vpar)

                lists[i1][i2][i].p1 = i
                pt1 = corrected[i1][i].pnr

                cand = []
                count = find_candidate(
                    corrected[i2], frm.targets[i2],
                    frm.num_targets[i2],
                    xmin, ymin, xmax, ymax,
                    frm.targets[i1][pt1].n, frm.targets[i1][pt1].nx,
                    frm.targets[i1][pt1].ny, frm.targets[i1][pt1].sumg,
                    cand, vpar, cpar, calib[i2])

                if count > MAXCAND:
                    count = MAXCAND

                for j in range(count):
                    lists[i1][i2][i].p2[j] = cand[j].pnr
                    lists[i1][i2][i].corr[j] = cand[j].corr
                    lists[i1][i2][i].dist[j] = cand[j].tol
                lists[i1][i2][i].n = count


@cython.ccall
def four_camera_matching(lists, base_target_count, accept_corr, scratch, scratch_size):
    """Find consistent 4-camera correspondences (quadruplets).

    Matches C four_camera_matching exactly.

    Returns:
        int, the number of candidate cliques found.
    """
    matched: cython.int = 0
    i: cython.int
    j: cython.int
    k: cython.int
    l: cython.int
    m: cython.int
    n: cython.int
    o: cython.int
    p1: cython.int
    p2: cython.int
    p3: cython.int
    p4: cython.int
    p31: cython.int
    p41: cython.int
    p42: cython.int
    corr: cython.double

    for i in range(base_target_count):
        p1 = lists[0][1][i].p1
        for j in range(lists[0][1][i].n):
            for k in range(lists[0][2][i].n):
                for l in range(lists[0][3][i].n):
                    p2 = lists[0][1][i].p2[j]
                    p3 = lists[0][2][i].p2[k]
                    p4 = lists[0][3][i].p2[l]

                    for m in range(lists[1][2][p2].n):
                        p31 = lists[1][2][p2].p2[m]
                        if p3 != p31:
                            continue

                        for n in range(lists[1][3][p2].n):
                            p41 = lists[1][3][p2].p2[n]
                            if p4 != p41:
                                continue

                            for o in range(lists[2][3][p3].n):
                                p42 = lists[2][3][p3].p2[o]
                                if p4 != p42:
                                    continue

                                corr = (lists[0][1][i].corr[j]
                                    + lists[0][2][i].corr[k]
                                    + lists[0][3][i].corr[l]
                                    + lists[1][2][p2].corr[m]
                                    + lists[1][3][p2].corr[n]
                                    + lists[2][3][p3].corr[o]) / (
                                    lists[0][1][i].dist[j]
                                    + lists[0][2][i].dist[k]
                                    + lists[0][3][i].dist[l]
                                    + lists[1][2][p2].dist[m]
                                    + lists[1][3][p2].dist[n]
                                    + lists[2][3][p3].dist[o])

                                if corr <= accept_corr:
                                    continue

                                scratch[matched].p[0] = p1
                                scratch[matched].p[1] = p2
                                scratch[matched].p[2] = p3
                                scratch[matched].p[3] = p4
                                scratch[matched].corr = corr

                                matched += 1
                                if matched == scratch_size:
                                    return matched
    return matched


@cython.ccall
def three_camera_matching(lists, num_cams, target_counts, accept_corr,
                          scratch, scratch_size, tusage):
    """Find consistent 3-camera correspondences (triplets).

    Matches C three_camera_matching exactly.

    Returns:
        int, the number of candidate cliques found.
    """
    matched: cython.int = 0
    i1: cython.int
    i2: cython.int
    i3: cython.int
    i: cython.int
    j: cython.int
    k: cython.int
    m_idx: cython.int
    nc: cython.int
    p1: cython.int
    p2: cython.int
    p3: cython.int
    corr: cython.double

    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = lists[i1][i2][i].p1
                if p1 > NMAX or tusage[i1][p1] > 0:
                    continue

                for j in range(lists[i1][i2][i].n):
                    p2 = lists[i1][i2][i].p2[j]
                    if p2 > NMAX or tusage[i2][p2] > 0:
                        continue

                    for i3 in range(i2 + 1, num_cams):
                        for k in range(lists[i1][i3][i].n):
                            p3 = lists[i1][i3][i].p2[k]
                            if p3 > NMAX or tusage[i3][p3] > 0:
                                continue

                            for m_idx in range(lists[i2][i3][p2].n):
                                if p3 != lists[i2][i3][p2].p2[m_idx]:
                                    continue

                                corr = (lists[i1][i2][i].corr[j]
                                    + lists[i1][i3][i].corr[k]
                                    + lists[i2][i3][p2].corr[m_idx]) / (
                                    lists[i1][i2][i].dist[j]
                                    + lists[i1][i3][i].dist[k]
                                    + lists[i2][i3][p2].dist[m_idx])

                                if corr <= accept_corr:
                                    continue

                                for nc in range(num_cams):
                                    scratch[matched].p[nc] = -2

                                scratch[matched].p[i1] = p1
                                scratch[matched].p[i2] = p2
                                scratch[matched].p[i3] = p3
                                scratch[matched].corr = corr

                                matched += 1
                                if matched == scratch_size:
                                    return matched
    return matched


@cython.ccall
def consistent_pair_matching(lists, num_cams, target_counts, accept_corr,
                             scratch, scratch_size, tusage):
    """Find unambiguous 2-camera pairs.

    Matches C consistent_pair_matching exactly.

    Returns:
        int, the number of pairs found.
    """
    matched: cython.int = 0
    i1: cython.int
    i2: cython.int
    i: cython.int
    nc: cython.int
    p1: cython.int
    p2: cython.int
    corr: cython.double

    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = lists[i1][i2][i].p1
                if p1 > NMAX or tusage[i1][p1] > 0:
                    continue

                if lists[i1][i2][i].n != 1:
                    continue

                p2 = lists[i1][i2][i].p2[0]
                if p2 > NMAX or tusage[i2][p2] > 0:
                    continue

                corr = lists[i1][i2][i].corr[0] / lists[i1][i2][i].dist[0]
                if corr <= accept_corr:
                    continue

                for nc in range(num_cams):
                    scratch[matched].p[nc] = -2

                scratch[matched].p[i1] = p1
                scratch[matched].p[i2] = p2
                scratch[matched].corr = corr

                matched += 1
                if matched == scratch_size:
                    return matched
    return matched


@cython.ccall
def take_best_candidates(src, dst, num_cams, num_cands, tusage):
    """Take candidates by descending correlation, skipping used targets.

    Matches C take_best_candidates exactly.

    Returns:
        int, the number of cliques taken.
    """
    import operator
    
    # Sort the active slice of src using Python's optimized Timsort
    src_slice = src[:num_cands]
    src_slice.sort(key=operator.attrgetter('corr'), reverse=True)
    src[:num_cands] = src_slice

    taken = 0
    for cand in range(num_cands):
        has_used = False
        for cam in range(num_cams):
            tnum = src[cand].p[cam]
            if tnum > -1 and tusage[cam][tnum] > 0:
                has_used = True
                break

        if has_used:
            continue

        for cam in range(num_cams):
            tnum = src[cand].p[cam]
            if tnum > -1:
                tusage[cam][tnum] += 1

        dst[taken] = NTupel(p=list(src[cand].p), corr=src[cand].corr)
        taken += 1
    return taken


@cython.ccall
def correct_frame(frm, calib, cpar, tol):
    """Transition from pixel to metric to flat coordinates, x-sorted.

    Matches C correct_frame from check_correspondences.c.

    Args:
        frm: Frame object.
        calib: list of Calibration objects.
        cpar: ControlPar.
        tol: tolerance for iterative flattening.

    Returns:
        list of lists of Coord2d, one per camera, x-sorted.
    """
    from .epi import Coord2d
    from .trafo import pixel_to_metric, dist_to_flat

    corrected = []
    for cam in range(cpar.num_cams):
        cam_coords = []
        for part in range(frm.num_targets[cam]):
            t = frm.targets[cam][part]
            xm, ym = pixel_to_metric(t.x, t.y, cpar)

            ap = calib[cam].added_par
            ip = calib[cam].int_par
            fx, fy = dist_to_flat(xm, ym,
                ip.xh, ip.yh,
                ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she,
                tol)

            cam_coords.append(Coord2d(pnr=t.pnr, x=fx, y=fy))

        quicksort_coord2d_x(cam_coords)
        corrected.append(cam_coords)

    return corrected


@cython.ccall
def correspondences(frm, corrected, vpar, cpar, calib):
    """Full correspondence matching pipeline.

    Matches C correspondences() exactly.

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
    num_cams = cpar.num_cams
    con0_size = num_cams * NMAX
    con0 = [NTupel() for _ in range(con0_size)]
    con = [NTupel() for _ in range(con0_size)]

    tusage = [[0] * NMAX for _ in range(num_cams)]

    lists = safely_allocate_adjacency_lists(num_cams, frm.num_targets)

    for i in range(NMAX):
        for j in range(num_cams):
            con0[i].p[j] = -1
        con0[i].corr = 0.0

    match_counts = [0, 0, 0, 0]

    match_pairs(lists, corrected, frm, vpar, cpar, calib)

    if num_cams == 4:
        match0 = four_camera_matching(lists, frm.num_targets[0],
            vpar.corrmin, con0, 4 * NMAX)

        match_counts[0] = take_best_candidates(con0, con, num_cams, match0, tusage)
        match_counts[3] += match_counts[0]

    if (num_cams == 4 and cpar.allCam_flag == 0) or num_cams == 3:
        match0 = three_camera_matching(lists, num_cams, frm.num_targets,
            vpar.corrmin, con0, 4 * NMAX, tusage)

        offset = match_counts[3]
        tmp = con[offset:]
        match_counts[1] = take_best_candidates(con0, tmp, num_cams,
            match0, tusage)
        for k in range(match_counts[1]):
            con[offset + k] = tmp[k]
        match_counts[3] += match_counts[1]

    if num_cams > 1 and cpar.allCam_flag == 0:
        match0 = consistent_pair_matching(lists, num_cams, frm.num_targets,
            vpar.corrmin, con0, 4 * NMAX, tusage)

        offset = match_counts[3]
        tmp = con[offset:]
        match_counts[2] = take_best_candidates(con0, tmp, num_cams,
            match0, tusage)
        for k in range(match_counts[2]):
            con[offset + k] = tmp[k]
        match_counts[3] += match_counts[2]

    for i in range(match_counts[3]):
        for j in range(num_cams):
            if con[i].p[j] < 0:
                continue
            p1 = corrected[j][con[i].p[j]].pnr
            if p1 > -1 and p1 < 1202590843:
                frm.targets[j][p1].tnr = i

    return con[:match_counts[3]], match_counts


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
