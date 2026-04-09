"""Correspondences."""

import math
from typing import List, Tuple

import numpy as np
from numba import njit

from .calibration import Calibration
from .constants import (
    CORRES_NONE,
    MAX_TARGETS,
    MAXCAND,
    NMAX,
    PT_UNUSED,
)
from .epi import epi_mm
from .find_candidate import find_candidate, find_start_point_binary, quality_ratio
from .multimed import fast_flat_image_coord_raw, move_along_ray
from .parameters import ControlPar, VolumePar
from .ray_tracing import fast_ray_tracing
from .tracking_frame_buf import Frame, Target, n_tupel_dtype
from .trafo import dist_to_flat, pixel_to_metric



# ---------------------------------------------------------------------------
# Phase 3A: SoA adjacency layout + fused fill kernel
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _fill_adjacency_pair(
    src_x, src_y, num_src,
    src_ref_n, src_ref_nx, src_ref_ny, src_ref_sumg,
    tgt_x, tgt_y, tgt_pnr, num_tgt,
    tgt_targ_n, tgt_targ_nx, tgt_targ_ny, tgt_targ_sumg,
    cal1_pos, cal1_dm, cal1_glass, cal1_cc,
    cal2_pos, cal2_dm, cal2_glass, cal2_cc,
    mm_n1, mm_d, mm_n2, mm_n3,
    mmlut_origin, mmlut_data, mmlut_nz, mmlut_nr, mmlut_rw,
    eps0, cn, cnx, cny, csumg_thresh,
    x_lay, z_min_lay, z_max_lay,
    maxcand,
    out_n, out_p2, out_corr, out_dist,
):
    """Fill adjacency for one camera pair.

    Fused @njit kernel combining epi_mm + candidate search into a single
    compiled loop.  Writes directly to pre-allocated SoA output arrays.
    """
    for i in range(num_src):
        # --- Compute epipolar line (inline epi_mm) ---
        camera = np.array([src_x[i], src_y[i], -cal1_cc], dtype=np.float64)
        pos, direction = fast_ray_tracing(
            camera, cal1_dm, cal1_pos, cal1_glass,
            mm_d[0], mm_n1, mm_n2[0], mm_n3,
        )

        denom_x = x_lay[1] - x_lay[0]
        if denom_x == 0.0:
            denom_x = 1.0
        z_min = z_min_lay[0] + (pos[0] - x_lay[0]) * (z_min_lay[1] - z_min_lay[0]) / denom_x
        z_max = z_max_lay[0] + (pos[0] - x_lay[0]) * (z_max_lay[1] - z_max_lay[0]) / denom_x

        X_min = move_along_ray(z_min, pos, direction)
        xa, ya = fast_flat_image_coord_raw(
            X_min, cal2_pos, cal2_dm, cal2_cc, cal2_glass,
            mm_d, mm_n1, mm_n2, mm_n3,
            mmlut_origin, mmlut_data, mmlut_nz, mmlut_nr, mmlut_rw,
        )

        X_max = move_along_ray(z_max, pos, direction)
        xb, yb = fast_flat_image_coord_raw(
            X_max, cal2_pos, cal2_dm, cal2_cc, cal2_glass,
            mm_d, mm_n1, mm_n2, mm_n3,
            mmlut_origin, mmlut_data, mmlut_nz, mmlut_nr, mmlut_rw,
        )

        # --- Candidate search (inline find_candidate) ---
        if abs(xb - xa) < 1e-15:
            xb = xa + 1e-10

        m = (yb - ya) / (xb - xa)
        b_val = ya - m * xa
        m_norm = math.sqrt(m * m + 1.0)

        xa_lo = min(xa, xb) - eps0
        xa_hi = max(xa, xb) + eps0
        ya_lo = min(ya, yb) - eps0
        ya_hi = max(ya, yb) + eps0

        j0 = find_start_point_binary(tgt_x, num_tgt, xa, eps0)

        ref_n = src_ref_n[i]
        ref_nx = src_ref_nx[i]
        ref_ny = src_ref_ny[i]
        ref_sumg = src_ref_sumg[i]

        count = 0
        for j in range(j0, num_tgt):
            if tgt_x[j] > xa_hi:
                break
            if tgt_x[j] < xa_lo:
                continue
            if tgt_y[j] < ya_lo or tgt_y[j] > ya_hi:
                continue

            d = abs((tgt_y[j] - m * tgt_x[j] - b_val) / m_norm)
            if d >= eps0:
                continue

            pnr_j = tgt_pnr[j]
            cand_n = tgt_targ_n[pnr_j]
            cand_nx = tgt_targ_nx[pnr_j]
            cand_ny = tgt_targ_ny[pnr_j]
            cand_sumg = tgt_targ_sumg[pnr_j]

            qn = quality_ratio(ref_n, cand_n)
            qnx = quality_ratio(ref_nx, cand_nx)
            qny = quality_ratio(ref_ny, cand_ny)
            qsumg = quality_ratio(ref_sumg, cand_sumg)

            if qn < cn or qnx < cnx or qny < cny or qsumg <= csumg_thresh:
                continue

            if count >= maxcand:
                break

            corr = (4.0 * qsumg + 2.0 * qn + qnx + qny) * (ref_sumg + cand_sumg)

            out_p2[i, count] = pnr_j
            out_dist[i, count] = d
            out_corr[i, count] = corr
            count += 1

        out_n[i] = count


# ---------------------------------------------------------------------------
# Phase 3B: Numba matching kernels on SoA arrays
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _four_camera_matching_inner(
    n_01, p2_01, corr_01, dist_01,
    n_02, p2_02, corr_02, dist_02,
    n_03, p2_03, corr_03, dist_03,
    n_12, p2_12, corr_12, dist_12,
    n_13, p2_13, corr_13, dist_13,
    n_23, p2_23, corr_23, dist_23,
    base_target_count, accept_corr,
    scratch_p, scratch_corr, scratch_size,
):
    """Four-camera matching on pre-extracted 2D pair arrays.

    Each n_XX is 1D (N,), each p2_XX/corr_XX/dist_XX is 2D (N, MAXCAND).
    """
    matched = 0

    for i in range(base_target_count):
        for j in range(n_01[i]):
            p2 = p2_01[i, j]
            c01j = corr_01[i, j]
            d01j = dist_01[i, j]

            for k in range(n_02[i]):
                p3 = p2_02[i, k]
                c02k = corr_02[i, k]
                d02k = dist_02[i, k]

                for ll in range(n_03[i]):
                    p4 = p2_03[i, ll]
                    c03ll = corr_03[i, ll]
                    d03ll = dist_03[i, ll]

                    for m in range(n_12[p2]):
                        p31 = p2_12[p2, m]
                        if p3 != p31:
                            continue
                        c12m = corr_12[p2, m]
                        d12m = dist_12[p2, m]

                        for n in range(n_13[p2]):
                            p41 = p2_13[p2, n]
                            if p4 != p41:
                                continue
                            c13n = corr_13[p2, n]
                            d13n = dist_13[p2, n]

                            for o in range(n_23[p3]):
                                p42 = p2_23[p3, o]
                                if p4 != p42:
                                    continue

                                total_dist = d01j + d02k + d03ll + d12m + d13n + dist_23[p3, o]
                                if total_dist == 0.0:
                                    continue
                                corr_val = (c01j + c02k + c03ll + c12m + c13n + corr_23[p3, o]) / total_dist
                                if corr_val <= accept_corr:
                                    continue

                                scratch_p[matched, 0] = i
                                scratch_p[matched, 1] = p2
                                scratch_p[matched, 2] = p3
                                scratch_p[matched, 3] = p4
                                scratch_corr[matched] = corr_val
                                matched += 1
                                if matched == scratch_size:
                                    return matched
    return matched


def _four_camera_matching_numba(
    corr_n, corr_p2, corr_corr, corr_dist,
    base_target_count, accept_corr,
    scratch_p, scratch_corr, scratch_size,
):
    """Four-camera matching wrapper: extracts pair slices then calls @njit kernel."""
    return _four_camera_matching_inner(
        corr_n[0, 1], corr_p2[0, 1], corr_corr[0, 1], corr_dist[0, 1],
        corr_n[0, 2], corr_p2[0, 2], corr_corr[0, 2], corr_dist[0, 2],
        corr_n[0, 3], corr_p2[0, 3], corr_corr[0, 3], corr_dist[0, 3],
        corr_n[1, 2], corr_p2[1, 2], corr_corr[1, 2], corr_dist[1, 2],
        corr_n[1, 3], corr_p2[1, 3], corr_corr[1, 3], corr_dist[1, 3],
        corr_n[2, 3], corr_p2[2, 3], corr_corr[2, 3], corr_dist[2, 3],
        base_target_count, accept_corr,
        scratch_p, scratch_corr, scratch_size,
    )


def _three_camera_matching_numba(
    corr_n, corr_p2, corr_corr, corr_dist,
    num_cams, target_counts, accept_corr,
    scratch_p, scratch_corr, scratch_size,
    tusage, nmax,
):
    """Three-camera matching on SoA arrays (plain Python, no @njit)."""
    matched = 0

    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = i
                if p1 >= nmax or tusage[i1, p1] > 0:
                    continue

                n_i1i2 = corr_n[i1, i2, i]
                for j in range(n_i1i2):
                    p2 = corr_p2[i1, i2, i, j]
                    if p2 >= nmax or tusage[i2, p2] > 0:
                        continue

                    for i3 in range(i2 + 1, num_cams):
                        n_i1i3 = corr_n[i1, i3, i]
                        n_i2i3 = corr_n[i2, i3, p2]

                        for k in range(n_i1i3):
                            p3 = corr_p2[i1, i3, i, k]
                            if p3 >= nmax or tusage[i3, p3] > 0:
                                continue

                            m = -1
                            for idx in range(n_i2i3):
                                if corr_p2[i2, i3, p2, idx] == p3:
                                    m = idx
                                    break
                            if m < 0:
                                continue

                            total_dist = (
                                corr_dist[i1, i2, i, j]
                                + corr_dist[i1, i3, i, k]
                                + corr_dist[i2, i3, p2, m]
                            )
                            if total_dist == 0.0:
                                continue
                            corr_val = (
                                corr_corr[i1, i2, i, j]
                                + corr_corr[i1, i3, i, k]
                                + corr_corr[i2, i3, p2, m]
                            ) / total_dist

                            if corr_val <= accept_corr:
                                continue

                            scratch_p[matched, :] = -2
                            scratch_p[matched, i1] = p1
                            scratch_p[matched, i2] = p2
                            scratch_p[matched, i3] = p3
                            scratch_corr[matched] = corr_val

                            matched += 1
                            if matched == scratch_size:
                                return matched
    return matched


def _consistent_pair_matching_numba(
    corr_n, corr_p2, corr_corr, corr_dist,
    num_cams, target_counts, accept_corr,
    scratch_p, scratch_corr, scratch_size,
    tusage, nmax,
):
    """Consistent pair matching on SoA arrays (plain Python, no @njit)."""
    matched = 0

    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = i
                if p1 >= nmax or tusage[i1, p1] > 0:
                    continue
                if corr_n[i1, i2, i] != 1:
                    continue

                p2 = corr_p2[i1, i2, i, 0]
                if p2 >= nmax or tusage[i2, p2] > 0:
                    continue

                d = corr_dist[i1, i2, i, 0]
                if d == 0.0:
                    continue
                corr_val = corr_corr[i1, i2, i, 0] / d
                if corr_val <= accept_corr:
                    continue

                scratch_p[matched, :] = -2
                scratch_p[matched, i1] = p1
                scratch_p[matched, i2] = p2
                scratch_corr[matched] = corr_val

                matched += 1
                if matched == scratch_size:
                    return matched
    return matched


# ---------------------------------------------------------------------------
# Phase 3C: SoA take_best_candidates
# ---------------------------------------------------------------------------


def _take_best_candidates_soa(
    src_p, src_corr, num_cands, num_cams, tusage,
):
    """Take best candidates from SoA scratch buffers.

    Greedy selection: sort by correlation descending, skip candidates with
    already-used targets.

    Args:
        src_p: int32 array (max_cands, num_cams) — candidate target indices
        src_corr: float64 array (max_cands,) — correlation scores
        num_cands: number of valid candidates in src_p/src_corr
        num_cams: number of cameras
        tusage: int32 array (num_cams, nmax) — target usage marks

    Returns:
        (dst_p, dst_corr, taken) where dst arrays hold the accepted candidates
        and taken is the count.
    """
    if num_cands == 0:
        dst_p = np.empty((0, num_cams), dtype=np.int32)
        dst_corr = np.empty(0, dtype=np.float64)
        return dst_p, dst_corr, 0

    # Sort descending by correlation
    order = np.argsort(-src_corr[:num_cands])

    dst_p = np.full((num_cands, num_cams), -2, dtype=np.int32)
    dst_corr = np.zeros(num_cands, dtype=np.float64)
    taken = 0

    for idx in range(num_cands):
        ci = order[idx]
        has_used = False
        for cam in range(num_cams):
            tnum = int(src_p[ci, cam])
            if tnum > -1 and tusage[cam, tnum] > 0:
                has_used = True
                break

        if has_used:
            continue

        # Mark targets as used
        for cam in range(num_cams):
            tnum = int(src_p[ci, cam])
            if tnum > -1:
                tusage[cam, tnum] += 1

        dst_p[taken] = src_p[ci]
        dst_corr[taken] = src_corr[ci]
        taken += 1

    return dst_p, dst_corr, taken


def match_pairs_soa(
    corrected,
    frm,
    vpar,
    cpar,
    calib,
):
    """SoA version of match_pairs.

    Returns (corr_n, corr_p2, corr_corr, corr_dist) plain numpy arrays
    replacing the Correspond_dtype recarray.
    """
    mm = getattr(cpar, "mm", None)
    if mm is None:
        get_mm = getattr(cpar, "get_multimedia_params", None)
        if callable(get_mm):
            mm = get_mm()
    if mm is None:
        raise AttributeError("Control parameters object does not expose multimedia parameters")

    num_cams = getattr(cpar, "num_cams", None)
    if num_cams is None:
        num_cams = getattr(getattr(cpar, "_control_par", None), "num_cams", None)
    if num_cams is None:
        get_num_cams = getattr(cpar, "get_num_cams", None)
        if callable(get_num_cams):
            num_cams = get_num_cams()
    if num_cams is None:
        raise AttributeError("Control parameters object does not expose num_cams")

    # Pre-extract corrected coords
    crd_x = [np.asarray(corrected[i].x, dtype=np.float64) for i in range(num_cams)]
    crd_y = [np.asarray(corrected[i].y, dtype=np.float64) for i in range(num_cams)]
    crd_pnr = [np.asarray(corrected[i].pnr, dtype=np.int64) for i in range(num_cams)]

    # Pre-extract target properties as float64
    targ_n = []
    targ_nx = []
    targ_ny = []
    targ_sumg = []
    for i in range(num_cams):
        nt = frm.num_targets[i]
        targ_n.append(np.array([frm.targets[i][j].n for j in range(nt)], dtype=np.float64))
        targ_nx.append(np.array([frm.targets[i][j].nx for j in range(nt)], dtype=np.float64))
        targ_ny.append(np.array([frm.targets[i][j].ny for j in range(nt)], dtype=np.float64))
        targ_sumg.append(np.array([frm.targets[i][j].sumg for j in range(nt)], dtype=np.float64))

    # Extract volume params
    eps0 = float(getattr(vpar, "eps0", None) or getattr(vpar, "get_eps0", lambda: 0)())
    cn_val = float(getattr(vpar, "cn", None) or getattr(vpar, "get_cn", lambda: 0)())
    cnx_val = float(getattr(vpar, "cnx", None) or getattr(vpar, "get_cnx", lambda: 0)())
    cny_val = float(getattr(vpar, "cny", None) or getattr(vpar, "get_cny", lambda: 0)())
    csumg_val = float(getattr(vpar, "csumg", None) or getattr(vpar, "get_csumg", lambda: 0)())
    x_lay = np.asarray(vpar.x_lay, dtype=np.float64)
    z_min_lay = np.asarray(vpar.z_min_lay, dtype=np.float64)
    z_max_lay = np.asarray(vpar.z_max_lay, dtype=np.float64)

    # Multimedia params
    mm_n1 = float(mm.n1)
    mm_d = np.asarray(mm.d, dtype=np.float64)
    mm_n2 = np.asarray(mm.n2, dtype=np.float64)
    mm_n3 = float(mm.n3)

    # Extract calibration arrays per camera
    cal_pos = []
    cal_dm = []
    cal_glass = []
    cal_cc = []
    cal_mmlut_origin = []
    cal_mmlut_data = []
    cal_mmlut_nz = []
    cal_mmlut_nr = []
    cal_mmlut_rw = []
    for c in range(num_cams):
        cal_pos.append(np.array(
            [calib[c].ext_par.x0, calib[c].ext_par.y0, calib[c].ext_par.z0],
            dtype=np.float64,
        ))
        cal_dm.append(np.ascontiguousarray(calib[c].ext_par.dm, dtype=np.float64))
        cal_glass.append(np.ascontiguousarray(calib[c].glass_par, dtype=np.float64))
        cal_cc.append(float(calib[c].int_par.cc))
        cal_mmlut_origin.append(np.asarray(calib[c].mmlut["origin"], dtype=np.float64).ravel())
        mmlut_data_c = np.asarray(calib[c].mmlut_data, dtype=np.float64).ravel()
        if mmlut_data_c.size == 0:
            mmlut_data_c = np.zeros(1, dtype=np.float64)
        cal_mmlut_data.append(mmlut_data_c)
        cal_mmlut_nz.append(int(calib[c].mmlut["nz"]))
        cal_mmlut_nr.append(int(calib[c].mmlut["nr"]))
        cal_mmlut_rw.append(max(int(calib[c].mmlut["rw"]), 1))  # avoid /0 in LUT lookup

    # Allocate SoA output
    n_max = max(frm.num_targets)
    corr_n = np.zeros((num_cams, num_cams, n_max), dtype=np.int32)
    corr_p2 = np.zeros((num_cams, num_cams, n_max, MAXCAND), dtype=np.int32)
    corr_corr = np.zeros((num_cams, num_cams, n_max, MAXCAND), dtype=np.float64)
    corr_dist = np.zeros((num_cams, num_cams, n_max, MAXCAND), dtype=np.float64)

    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            num1 = frm.num_targets[i1]

            # Pre-compute source reference properties
            pnr1 = crd_pnr[i1][:num1]
            src_ref_n = targ_n[i1][pnr1]
            src_ref_nx = targ_nx[i1][pnr1]
            src_ref_ny = targ_ny[i1][pnr1]
            src_ref_sumg = targ_sumg[i1][pnr1]

            _fill_adjacency_pair(
                crd_x[i1], crd_y[i1], np.int32(num1),
                src_ref_n, src_ref_nx, src_ref_ny, src_ref_sumg,
                crd_x[i2], crd_y[i2], crd_pnr[i2],
                np.int32(frm.num_targets[i2]),
                targ_n[i2], targ_nx[i2], targ_ny[i2], targ_sumg[i2],
                cal_pos[i1], cal_dm[i1], cal_glass[i1], cal_cc[i1],
                cal_pos[i2], cal_dm[i2], cal_glass[i2], cal_cc[i2],
                mm_n1, mm_d, mm_n2, mm_n3,
                cal_mmlut_origin[i2], cal_mmlut_data[i2],
                cal_mmlut_nz[i2], cal_mmlut_nr[i2], cal_mmlut_rw[i2],
                eps0, cn_val, cnx_val, cny_val, csumg_val,
                x_lay, z_min_lay, z_max_lay,
                np.int32(MAXCAND),
                corr_n[i1, i2], corr_p2[i1, i2],
                corr_corr[i1, i2], corr_dist[i1, i2],
            )

    return corr_n, corr_p2, corr_corr, corr_dist


class MatchedCoords:
    """Python equivalent of the native MatchedCoords wrapper."""

    def __init__(
        self,
        targs,
        cpar: ControlPar,
        cal: Calibration,
        tol: float = 0.00001,
        reset_numbers: bool = True,
    ):
        self._num_pts = len(targs)
        self.buf = np.recarray(
            self._num_pts,
            dtype=[("x", np.float64), ("y", np.float64), ("pnr", np.int_)],
        )

        for tnum in range(self._num_pts):
            targ = targs[tnum]
            if reset_numbers:
                targ.pnr = tnum

            x_m, y_m = pixel_to_metric(targ.x, targ.y, cpar)
            x_f, y_f = dist_to_flat(x_m, y_m, cal, tol)
            self.buf[tnum].x = x_f
            self.buf[tnum].y = y_f
            self.buf[tnum].pnr = targ.pnr

        self.buf = self.buf[np.argsort(self.buf.x)]

    def __getitem__(self, index):
        return self.buf[index]

    @property
    def x(self):
        """Expose x coordinates for compatibility with find_start_point."""
        return self.buf.x

    @property
    def y(self):
        """Expose y coordinates for compatibility with find_candidate."""
        return self.buf.y

    @property
    def pnr(self):
        """Expose pnr for compatibility with find_candidate."""
        return self.buf.pnr

    def as_arrays(self):
        pos = np.empty((self._num_pts, 2), dtype=np.float64)
        pos[:, 0] = self.buf.x
        pos[:, 1] = self.buf.y
        pnr = self.buf.pnr.astype(np.int_)
        return pos, pnr

    def get_by_pnrs(self, pnrs):
        pnrs = np.asarray(pnrs)
        pos = np.full((len(pnrs), 2), np.nan, dtype=np.float64)
        for row in self.buf:
            which = np.flatnonzero(row.pnr == pnrs)
            if len(which) > 0:
                pos[which[0], 0] = row.x
                pos[which[0], 1] = row.y
        return pos


Correspond_dtype = np.dtype(
    [
        ("p1", np.int32),  # PT_UNUSED
        ("n", np.int32),  # 0
        ("p2", (np.int32, MAXCAND)),  # np.zeros
        ("corr", (np.float64, MAXCAND)),  # np.zeros
        ("dist", (np.float64, MAXCAND)),  # np.zeros
    ]
)


def safely_allocate_target_usage_marks(
    num_cams: int, nmax: int = NMAX
) -> np.ndarray:  # num_cams x nmax instead of List[List[int]]:
    """Allocate space for per-camera arrays marking whether a certain target was used.

    If some allocation failed, it cleans up memory and returns NULL. Allocated arrays are zeroed
    out initially by the C library.

    Args:
    ----
        num_cams: The number of cameras.

    Returns
    -------
        A list of lists of integers, or `None` if an allocation failed.
    """
    # tusage = []
    # for cam in range(num_cams):
    #     tusage.append([0] * nmax)  # Initialize the array to all zeros.

    # # Check if any of the allocations failed.
    # for cam in range(num_cams):
    #     if tusage[cam] is None:
    #         return []  # was None

    return np.zeros((num_cams, nmax), dtype=np.int32)


def safely_allocate_adjacency_lists(
    num_cams: int, target_counts: List[int]
) -> np.recarray:
    """Allocate space for the adjacency lists."""
    # one_element = np.array(
    #     [(PT_UNUSED, 0, np.zeros(MAXCAND), np.zeros(MAXCAND), np.zeros(MAXCAND))],
    #     dtype=Correspond_dtype).view(np.recarray)

    try:
        # lists = [
        #     [[one_element for _ in range(target_counts[c1])] for _ in range(num_cams)]
        #     for c1 in range(num_cams)
        # ]

        lists = np.recarray(
            (num_cams, num_cams, max(target_counts)), dtype=Correspond_dtype
        )

    except MemoryError as exc:
        raise MemoryError("Failed to allocate adjacency lists.") from exc
        # lists = [[[one_element]]]

    lists.p1 = PT_UNUSED
    lists.n = 0
    lists.p2 = np.zeros(MAXCAND)
    lists.corr = np.zeros(MAXCAND)
    lists.dist = np.zeros(MAXCAND)

    return lists


def four_camera_matching(
    corr_list: np.recarray,
    base_target_count,
    accept_corr,
    scratch,
    scratch_size,
) -> int:
    """Four-camera matching."""
    matched = 0
    # print(" Four camera matching ")

    for i in range(base_target_count):
        pair_01 = corr_list[0][1][i]
        pair_02 = corr_list[0][2][i]
        pair_03 = corr_list[0][3][i]
        p1 = pair_01.p1

        for j in range(pair_01.n):
            p2 = pair_01.p2[j]
            corr_01_j = pair_01.corr[j]
            dist_01_j = pair_01.dist[j]

            for k in range(pair_02.n):
                p3 = pair_02.p2[k]
                corr_02_k = pair_02.corr[k]
                dist_02_k = pair_02.dist[k]

                for ll in range(pair_03.n):
                    p4 = pair_03.p2[ll]
                    corr_03_ll = pair_03.corr[ll]
                    dist_03_ll = pair_03.dist[ll]

                    pair_12 = corr_list[1][2][p2]
                    pair_13 = corr_list[1][3][p2]
                    pair_23 = corr_list[2][3][p3]

                    for m in range(pair_12.n):
                        p31 = pair_12.p2[m]
                        # print(f" p31 {p31} p3 {p3}")

                        if p3 != p31:
                            continue

                        corr_12_m = pair_12.corr[m]
                        dist_12_m = pair_12.dist[m]

                        for n in range(pair_13.n):
                            p41 = pair_13.p2[n]
                            # print(f" p41 {p41} p4 {p4}")
                            if p4 != p41:
                                continue

                            corr_13_n = pair_13.corr[n]
                            dist_13_n = pair_13.dist[n]

                            for o in range(pair_23.n):
                                p42 = pair_23.p2[o]

                                # print(f" p42 {p42} p4 {p4}")
                                if p4 != p42:
                                    continue

                                corr = (
                                    corr_01_j
                                    + corr_02_k
                                    + corr_03_ll
                                    + corr_12_m
                                    + corr_13_n
                                    + pair_23.corr[o]
                                ) / (
                                    dist_01_j
                                    + dist_02_k
                                    + dist_03_ll
                                    + dist_12_m
                                    + dist_13_n
                                    + pair_23.dist[o]
                                )

                                # print(f" corr {corr}")
                                if corr <= accept_corr:
                                    continue

                                # accept as preliminary match
                                scratch[matched].p[0] = p1
                                scratch[matched].p[1] = p2
                                scratch[matched].p[2] = p3
                                scratch[matched].p[3] = p4
                                scratch[matched].corr = corr

                                matched += 1
                                # print(f" matched {matched} [{p1, p2, p3, p4}]")
                                if matched == scratch_size:
                                    print("Overflow in correspondences.")
                                    return matched

    return matched


def three_camera_matching(
    corr_list: np.recarray,  # num_cam, num_cam, num_targets
    num_cams,
    target_counts,
    accept_corr,
    scratch,
    scratch_size,
    tusage,
) -> int:
    """Three-camera matching."""
    matched = 0
    nmax = NMAX

    for i1 in range(num_cams - 2):
        for i in range(target_counts[i1]):
            for i2 in range(i1 + 1, num_cams - 1):
                p1 = corr_list[i1][i2][i].p1
                if p1 >= nmax or tusage[i1][p1] > 0:
                    continue

                # print(f"p1 {p1} candidates {corr_list[i1][i2][i].n } ")

                for j in range(corr_list[i1][i2][i].n):
                    p2 = corr_list[i1][i2][i].p2[j]
                    if p2 > nmax or tusage[i2][p2] > 0:
                        continue

                    # print(f"p2 {p2}")

                    for i3 in range(i2 + 1, num_cams):
                        pair_13 = corr_list[i1][i3][i]
                        pair_23 = corr_list[i2][i3][p2]

                        for k in range(pair_13.n):
                            p3 = pair_13.p2[k]
                            if p3 > nmax or tusage[i3][p3] > 0:
                                continue

                            # print(f"p3 {p3}")

                            # Direct scan of pair_23 candidates,
                            # matching the C loop structure.
                            m = -1
                            for idx in range(pair_23.n):
                                if pair_23.p2[idx] == p3:
                                    m = idx
                                    break

                            if m < 0:
                                continue

                            corr = (
                                corr_list[i1][i2][i].corr[j]
                                + pair_13.corr[k]
                                + pair_23.corr[m]
                            ) / (
                                corr_list[i1][i2][i].dist[j]
                                + pair_13.dist[k]
                                + pair_23.dist[m]
                            )

                            # print(f"corr {corr}")

                            if corr <= accept_corr:
                                continue

                            p = np.full(num_cams, -2)
                            p[i1], p[i2], p[i3] = p1, p2, p3
                            scratch[matched].p = p
                            scratch[matched].corr = corr

                            matched += 1
                            # print(f"matched: {matched} p: {p}")

                            if matched == scratch_size:
                                print("Overflow in correspondences.\n")
                                return matched
    return matched


def consistent_pair_matching(
    corr_list: np.recarray,
    num_cams: int,
    target_counts: List[int],
    accept_corr: float,
    scratch: np.recarray,
    scratch_size: int,
    tusage: np.ndarray,
) -> int:
    """Find consistent pairs of correspondences."""
    matched = 0
    # nmax = np.inf
    nmax = NMAX
    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            for i in range(target_counts[i1]):
                p1 = corr_list[i1][i2][i].p1
                if p1 >= nmax or tusage[i1][p1] > 0:
                    continue

                if corr_list[i1][i2][i].n != 1:
                    continue

                p2 = corr_list[i1][i2][i].p2[0]
                if p2 >= nmax or tusage[i2][p2] > 0:
                    continue

                corr = corr_list[i1][i2][i].corr[0] / corr_list[i1][i2][i].dist[0]
                if corr <= accept_corr:
                    continue

                for n in range(num_cams):
                    scratch[matched].p[n] = -2

                scratch[matched].p[i1] = p1
                scratch[matched].p[i2] = p2
                scratch[matched].corr = corr

                matched += 1
                if matched == scratch_size:
                    print("Overflow in correspondences.\n")
                    return matched

    return matched


def match_pairs(
    corr_lists: np.recarray,  # num_cam, num_cam, num_targets
    corrected: np.recarray,  # List[List[Coord2d]],
    frm: Frame,
    vpar: VolumePar,
    cpar: ControlPar,
    calib: List[Calibration],
) -> None:
    """Match pairs of cameras.

    **This function matches pairs of cameras by finding corresponding points in each camera.
    The correspondences are stored in the `corr_lists` argument.**

    **The following steps are performed:**

    1. For each pair of cameras, the epipolar lines for the two cameras are calculated.
    2. For each target in the first camera, the corresponding points in the second camera
    are found by searching along the epipolar line.
    3. The correspondences are stored in the `corr_lists` argument.

    **The `corr_lists` argument is a list of lists of lists of `Correspond` objects.
    Each inner list corresponds to a pair of cameras, and each inner-most list corresponds
    to a correspondence between two points in the two cameras. The `Correspond` objects
    have the following attributes:**

    * `p1`: The index of the target in the first camera.
    * `p2`: The index of the target in the second camera.
    * `corr`: The correspondence score.
    * `dist`: The distance between the two points.

    **The following are the arguments for the function:**

    * `corr_lists`: A list of lists of lists of `Correspond` objects. Each inner list
    corresponds to a pair of cameras, and each inner-most list corresponds to a
    correspondence between two points in the two cameras.

    * `corrected`: A list of lists of `coord_2d` objects. Each inner list corresponds to a
    camera, and each inner-most object corresponds to the corrected coordinates of a target in
    that camera.
    * `frm`: A `frame` object.
    * `vpar`: A `volume_par` object.
    * `cpar`: A `control_par` object.
    * `calib`: A list of `Calibration` objects.

    **The function returns None.**
    """
    count = 0
    mm = getattr(cpar, "mm", None)
    if mm is None:
        get_mm = getattr(cpar, "get_multimedia_params", None)
        if callable(get_mm):
            mm = get_mm()
    if mm is None:
        raise AttributeError(
            "Control parameters object does not expose multimedia parameters"
        )
    num_cams = getattr(cpar, "num_cams", None)
    if num_cams is None:
        num_cams = getattr(getattr(cpar, "_control_par", None), "num_cams", None)
    if num_cams is None:
        get_num_cams = getattr(cpar, "get_num_cams", None)
        if callable(get_num_cams):
            num_cams = get_num_cams()
    if num_cams is None:
        raise AttributeError("Control parameters object does not expose num_cams")

    # Pre-extract raw arrays from corrected coords and targets to avoid
    # repeated recarray attribute access overhead.
    _crd_x = [np.asarray(corrected[i].x, dtype=np.float64) for i in range(num_cams)]
    _crd_y = [np.asarray(corrected[i].y, dtype=np.float64) for i in range(num_cams)]
    _crd_pnr = [np.asarray(corrected[i].pnr, dtype=np.int64) for i in range(num_cams)]

    _targ_n = []
    _targ_nx = []
    _targ_ny = []
    _targ_sumg = []
    for i in range(num_cams):
        nt = frm.num_targets[i]
        _targ_n.append(
            np.array([frm.targets[i][j].n for j in range(nt)], dtype=np.int64)
        )
        _targ_nx.append(
            np.array([frm.targets[i][j].nx for j in range(nt)], dtype=np.int64)
        )
        _targ_ny.append(
            np.array([frm.targets[i][j].ny for j in range(nt)], dtype=np.int64)
        )
        _targ_sumg.append(
            np.array([frm.targets[i][j].sumg for j in range(nt)], dtype=np.int64)
        )

    eps0 = getattr(vpar, "eps0", None)
    if eps0 is None:
        get_eps0 = getattr(vpar, "get_eps0", None)
        if callable(get_eps0):
            eps0 = get_eps0()
    cn_val = getattr(vpar, "cn", None)
    if cn_val is None:
        get_cn = getattr(vpar, "get_cn", None)
        if callable(get_cn):
            cn_val = get_cn()
    cnx_val = getattr(vpar, "cnx", None)
    if cnx_val is None:
        get_cnx = getattr(vpar, "get_cnx", None)
        if callable(get_cnx):
            cnx_val = get_cnx()
    cny_val = getattr(vpar, "cny", None)
    if cny_val is None:
        get_cny = getattr(vpar, "get_cny", None)
        if callable(get_cny):
            cny_val = get_cny()
    csumg_val = getattr(vpar, "csumg", None)
    if csumg_val is None:
        get_csumg = getattr(vpar, "get_csumg", None)
        if callable(get_csumg):
            csumg_val = get_csumg()

    for i1 in range(num_cams - 1):
        for i2 in range(i1 + 1, num_cams):
            num1 = frm.num_targets[i1]
            num2 = frm.num_targets[i2]
            crd_x2 = _crd_x[i2]
            crd_y2 = _crd_y[i2]
            crd_pnr2 = _crd_pnr[i2]

            for i in range(num1):
                xa12, ya12, xb12, yb12 = epi_mm(
                    _crd_x[i1][i],
                    _crd_y[i1][i],
                    calib[i1],
                    calib[i2],
                    mm,
                    vpar,
                )

                corr_lists[i1][i2][i].p1 = i
                pt1 = _crd_pnr[i1][i]

                # Vectorized candidate search (replaces find_candidate)
                cand = _find_candidates_vectorized(
                    crd_x2,
                    crd_y2,
                    crd_pnr2,
                    num2,
                    _targ_n[i2],
                    _targ_nx[i2],
                    _targ_ny[i2],
                    _targ_sumg[i2],
                    xa12,
                    ya12,
                    xb12,
                    yb12,
                    _targ_n[i1][pt1],
                    _targ_nx[i1][pt1],
                    _targ_ny[i1][pt1],
                    _targ_sumg[i1][pt1],
                    eps0,
                    cn_val,
                    cnx_val,
                    cny_val,
                    csumg_val,
                )

                count = min(len(cand), MAXCAND)
                for j in range(count):
                    corr_lists[i1][i2][i].p2[j] = cand[j][0]
                    corr_lists[i1][i2][i].corr[j] = cand[j][2]
                    corr_lists[i1][i2][i].dist[j] = cand[j][1]

                corr_lists[i1][i2][i].n = count


def _find_candidates_vectorized(
    crd_x2,
    crd_y2,
    crd_pnr2,
    num2,
    targ_n2,
    targ_nx2,
    targ_ny2,
    targ_sumg2,
    xa,
    ya,
    xb,
    yb,
    ref_n,
    ref_nx,
    ref_ny,
    ref_sumg,
    eps0,
    cn,
    cnx,
    cny,
    csumg,
):
    """Vectorized candidate search replacing find_candidate.

    Uses numpy boolean indexing instead of per-element Python loop.
    Returns list of (pnr, distance, correlation) tuples.
    """
    if num2 == 0:
        return []

    # Line equation: y = m*x + b
    if abs(xb - xa) < 1e-15:
        xb = xa + 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa
    m_norm = math.sqrt(m * m + 1)

    # Normalize search window
    xa_lo = min(xa, xb) - eps0
    xa_hi = max(xa, xb) + eps0
    ya_lo = min(ya, yb) - eps0
    ya_hi = max(ya, yb) + eps0

    # Binary search for starting index in x-sorted array
    j0 = find_start_point_binary(crd_x2, num2, xa, eps0)

    # Extract slice starting from j0
    sl_x = crd_x2[j0:num2]
    sl_y = crd_y2[j0:num2]
    sl_pnr = crd_pnr2[j0:num2]

    # Vectorized bounds check
    mask = (sl_x >= xa_lo) & (sl_x <= xa_hi) & (sl_y >= ya_lo) & (sl_y <= ya_hi)

    # Stop at first x that exceeds upper bound
    beyond = np.where(sl_x > xa_hi)[0]
    if len(beyond) > 0:
        mask[beyond[0] :] = False

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []

    # Vectorized epipolar distance
    cx = sl_x[indices]
    cy = sl_y[indices]
    cpnr = sl_pnr[indices]
    dists = np.abs((cy - m * cx - b) / m_norm)

    # Filter by distance
    dmask = dists < eps0
    if not np.any(dmask):
        return []

    cpnr = cpnr[dmask]
    dists = dists[dmask]

    # Vectorized quality computation
    cand_n = targ_n2[cpnr]
    cand_nx = targ_nx2[cpnr]
    cand_ny = targ_ny2[cpnr]
    cand_sumg = targ_sumg2[cpnr]

    qn = _quality_ratio_vec(ref_n, cand_n)
    qnx = _quality_ratio_vec(ref_nx, cand_nx)
    qny = _quality_ratio_vec(ref_ny, cand_ny)
    qsumg = _quality_ratio_vec(ref_sumg, cand_sumg)

    # Quality filter
    qmask = (qn >= cn) & (qnx >= cnx) & (qny >= cny) & (qsumg > csumg)
    if not np.any(qmask):
        return []

    cpnr = cpnr[qmask]
    dists = dists[qmask]
    qn = qn[qmask]
    qnx = qnx[qmask]
    qny = qny[qmask]
    qsumg = qsumg[qmask]
    cand_sumg = cand_sumg[qmask]

    # Correlation score
    corrs = (4 * qsumg + 2 * qn + qnx + qny) * (ref_sumg + cand_sumg).astype(np.float64)

    # Preserve x-sorted discovery order (matching original find_candidate
    # which takes candidates in the order they appear along the epipolar line).
    # The candidates already maintain order from the numpy filtering,
    # but we need to limit to MAXCAND.
    n_take = min(len(cpnr), MAXCAND)

    return [(int(cpnr[k]), float(dists[k]), float(corrs[k])) for k in range(n_take)]


def _quality_ratio_vec(a, b):
    """Vectorized quality ratio: min(a,b)/max(a,b), handling zeros."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (a == 0) & (b == 0),
            0.0,
            np.minimum(a, b) / np.maximum(a, b),
        )
    return result


def take_best_candidates(
    # List[n_tupel]
    src: np.recarray,
    dst: np.recarray,
    num_cams: int,
    tusage: np.ndarray,
):
    """
    Take the best candidates from the candidate list based on their correlation measure.

    Arguments:
    ---------
    src (list): The list of candidates to choose from.
    dst (list): The list to store the chosen candidates.
    num_cams (int): The number of cameras in the scene.
    tusage (list): Record of currently used/unused targets in each camera.

    Returns
    -------
    int: The number of candidates taken from the source list.

    /*  take_best_candidates() takes candidates out of a candidate list by their
        correlation measure. A candidate is not taken if it has been marked used
        for a larger clique or for a same-size clique with a better correlation
        score.

    Arguments:
    ---------
        n_tupel *src - the array of candidates. sorted in place by correlation
            score.
        n_tupel *dst - an array to receive the chosen cliques in order. Must have
            enough space allocated.
        int num_cams - the number of cameras in the scene, which defines the size
            of other parameters.
        int num_cands - number of elements in ``src``.
        int **tusage - record of currently used/unused targets in each camera.
            Targets that are already marked used (e.g. by quadruplets) will not be
            taken.

    Returns
    -------
        the number of cliques taken from the candidate list.
    */

    """
    taken = 0

    # Sort candidates by match quality (.corr)
    src.sort(order="corr")  # by corr
    src[:] = np.flip(
        src, axis=0
    )  # reverse order in-place while preserving recarray type

    # Take candidates from the top to the bottom of the sorted list
    # Only take if none of the corresponding targets have been used
    for cand in src:
        has_used_target = False
        for cam in range(num_cams):
            tnum = cand.p[cam]

            # If any correspondence in this camera, check if the target is free
            if tnum > -1 and tusage[cam][tnum] > 0:
                has_used_target = True
                break

        if has_used_target:
            continue

        # Mark the targets as used
        for cam in range(num_cams):
            tnum = cand.p[cam]
            if tnum > -1:
                tusage[cam][tnum] += 1

        dst[taken] = cand
        taken += 1

    return taken


def py_correspondences(
    img_pts: List[List[Target]],  # num_cams * num_targets[cam]
    flat_coords: np.recarray,
    calib: List[Calibration],
    vparam: VolumePar,
    cparam: ControlPar,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """
    Get the correspondences for each clique size.

    Arguments:
    ---------
    img_pts - a list of c := len(cals), containing TargetArray objects, each
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property
        set. the pnr property should be set to the target index in its array.
    flat_coords - a list of MatchedCoordinates objects, one per camera, holding
        the x-sorted flat-coordinates conversion of the respective image
        targets.
    cals - a list of Calibration objects, each for the camera taking one image.
    VolumeParams vparam - an object holding observed volume size parameters.
    ControlParams cparam - an object holding general control parameters.

    Returns
    -------
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of
        c image planes of points belonging to quadruplets, triplets, pairs
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belonging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of
        previous 3).
    """
    num_cams = cparam.num_cams
    frm = Frame(num_cams, MAX_TARGETS)

    # Special case of a single camera, follow the single_cam_correspondence docstring
    if num_cams == 1:
        sorted_pos, sorted_corresp, num_targs = single_cam_correspondences(
            img_pts[0],
            flat_coords[0],
        )
        return sorted_pos, sorted_corresp, num_targs

    # cdef:
    #     calibration **calib = <calibration **> malloc(
    #         num_cams * sizeof(calibration *))
    #     coord_2d **corrected = <coord_2d **> malloc(
    #         num_cams * sizeof(coord_2d *))
    #     frame frm

    # np.ndarray[ndim=2, dtype=np.int_t] clique_ids
    # np.ndarray[ndim=3, dtype=np.float64_t] clique_targs

    # Return buffers:
    # int *match_counts = <int *> malloc(num_cams * sizeof(int))
    # n_tupel *corresp_buf

    match_counts = [0] * num_cams
    # corresp_buf = []  # of n_tupel

    # Initialize frame partially, without the extra momory used by init_frame.
    # frm.targets = <target**> calloc(num_cams, sizeof(target*))
    # frm.num_targets = <int *> calloc(num_cams, sizeof(int))
    # frm.targets = [TargetArray(MAX_TARGETS) for _ in range(num_cams)]
    # frm.num_targets = [0] * num_cams

    for cam in range(num_cams):
        # calib[cam] = (<Calibration>cals[cam])._calibration
        # frm.targets[cam] = (<TargetArray>img_pts[cam])._tarr
        frm.num_targets[cam] = len(img_pts[cam])
        frm.targets[cam] = img_pts[cam]

    # The biz:
    corresp_buf = correspondences(frm, flat_coords, vparam, cparam, calib, match_counts)

    # Distribute data to return structures:
    # sorted_pos = [None] * (num_cams - 1)
    # sorted_corresp = [None] * (num_cams - 1)
    sorted_pos, sorted_corresp = [], []

    last_count = 0

    for clique_type in range(num_cams - 1):
        num_points = match_counts[4 - num_cams + clique_type]  # for 1-4 cameras
        clique_targs = np.full((num_cams, num_points, 2), PT_UNUSED, dtype=np.float64)
        clique_ids = np.full((num_cams, num_points), CORRES_NONE, dtype=np.int_)

        # Trace back the pixel target properties through the flat metric
        # intermediary that's x-sorted.
        for cam in range(num_cams):
            for pt in range(num_points):
                geo_id = corresp_buf[pt + last_count].p[cam]
                if geo_id < 0:
                    continue

                p1 = flat_coords[cam][geo_id].pnr
                clique_ids[cam, pt] = p1

                if p1 > -1:
                    targ = img_pts[cam][p1]
                    clique_targs[cam, pt, 0] = targ.x
                    clique_targs[cam, pt, 1] = targ.y

        last_count += num_points
        sorted_pos.append(clique_targs)
        sorted_corresp.append(clique_ids)
        # sorted_pos[clique_type] = clique_targs # type: ignore
        # sorted_corresp[clique_type] = clique_ids # type: ignore

    # Clean up.
    num_targs = match_counts[num_cams - 1]

    return sorted_pos, sorted_corresp, num_targs


def correspondences(
    frm: Frame,
    corrected: np.recarray,  # List[List[Coord2d]],
    vpar: VolumePar,
    cpar: ControlPar,
    calib: List[Calibration],
    match_counts: List[int],
) -> np.recarray:  # n_tupel_dtype
    """Find correspondences between cameras.

    /*  correspondences() generates a list of tuple target numbers (one for each
        camera), denoting the set of targets all corresponding to one 3D position.
        Candidates are preferred by the number of cameras invoilved (more is
        better) and the correspondence score calculated using epipolar lines.

    Arguments:
    ---------
        frame *frm - a frame struct holding the observed targets and their number
            for each camera.
        coord_2d **corrected - for each camera, an array of the flat-image
            coordinates corresponding to the targets in frm (the .pnr property
            says which is which), sorted by the X coordinate.
        volume_par *vpar - epipolar search zone and criteria for correspondence.
        control_par *cpar - general scene parameters s.a. image size.
        Calibration **calib - array of pointers to each camera's calibration
            parameters.

        Output Arguments:
        int match_counts[] - output buffer, as long as the number of cameras.
            stores the number of matches for each clique size, in descending
            clique size order. The last element stores the total.

    Returns
    -------
        n_tupel con - the sorted list of correspondences in descending quality
            order.
    */



    """
    nmax = NMAX
    num_cams = getattr(cpar, "num_cams", None)
    if num_cams is None:
        num_cams = getattr(getattr(cpar, "_control_par", None), "num_cams", None)
    if num_cams is None:
        get_num_cams = getattr(cpar, "get_num_cams", None)
        if callable(get_num_cams):
            num_cams = get_num_cams()
    if num_cams is None:
        raise AttributeError("Control parameters object does not expose num_cams")

    # Allocation of scratch buffers for internal tasks and return-value space
    con0 = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
    con0.p = 0
    con0.corr = 0.0

    con = np.recarray((nmax * num_cams,), dtype=n_tupel_dtype)
    con.p = 0
    con.corr = 0.0

    tim = safely_allocate_target_usage_marks(num_cams, nmax)

    # allocate memory for lists of correspondences
    corr_list = safely_allocate_adjacency_lists(num_cams, frm.num_targets)

    # if I understand correctly, the number of matches cannot be more than the number of
    # targets (dots) in the first image. In the future we'll replace it by the maximum
    # number of targets in any image (if we will implement the cyclic search) but for
    # a while we always start with the cam1

    # Generate adjacency lists: mark candidates for correspondence.
    # matching 1 -> 2,3,4 + 2 -> 3,4 + 3 -> 4
    match_pairs(corr_list, corrected, frm, vpar, cpar, calib)

    # search consistent quadruplets in the corr_list
    if num_cams == 4:
        four_camera_matching(
            corr_list, frm.num_targets[0], vpar.corrmin, con0, 4 * nmax
        )

        match_counts[0] = take_best_candidates(con0, con, num_cams, tim)
        match_counts[3] += match_counts[0]

    # search consistent triplets: 123, 124, 134, 234
    if (num_cams == 4 and cpar.all_cam_flag == 0) or num_cams == 3:
        three_camera_matching(
            corr_list, num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim
        )

        match_counts[1] = take_best_candidates(
            con0, con[match_counts[3] :].view(np.recarray), num_cams, tim
        )
        match_counts[3] += match_counts[1]

    # Search consistent pairs: 12, 13, 14, 23, 24, 34
    if num_cams > 1 and cpar.all_cam_flag == 0:
        consistent_pair_matching(
            corr_list, num_cams, frm.num_targets, vpar.corrmin, con0, 4 * nmax, tim
        )
        match_counts[2] = take_best_candidates(
            con0, con[match_counts[3] :].view(np.recarray), num_cams, tim
        )
        match_counts[3] += match_counts[2]

    # Give each used pix the correspondence number
    for i in range(match_counts[3]):
        for j in range(num_cams):
            # Skip cameras without a correspondence obviously.
            if con[i].p[j] < 0:
                continue

            p1 = corrected[j][con[i].p[j]].pnr
            if p1 > -1 and p1 < 1202590843:
                frm.targets[j][p1].tnr = i

    # Free all other allocations
    # deallocate_adjacency_lists(corr_list, cpar.num_cams)
    # deallocate_target_usage_marks(tim, cpar.num_cams)
    # del con0

    return con


def single_cam_correspondences(
    img_pts: List[Target],
    corrected: np.recarray,  # List[Coord2d]
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """
    Single camera correspondence is not a real correspondence, it will be only a projection.

    of a 2D target from the image space into the 3D position, x,y,z using epi_mm_2d
    function. Here we only update the pointers of the targets and return it in a proper format.

    Arguments:
    ---------
    img_pts - a corr_list of c := len(cals), containing TargetArray objects, each
        with the target coordinates of n detections in the respective image.
        The target arrays are clobbered: returned arrays have the tnr property
        set. the pnr property should be set to the target index in its array.
    flat_coords - a corr_list of MatchedCoordinates objects, one per camera, holding
        the x-sorted flat-coordinates conversion of the respective image
        targets.

    Returns
    -------
    sorted_pos - a tuple of (c,?,2) arrays, each with the positions in each of
        c image planes of points belonging to quadruplets, triplets, pairs
        found.
    sorted_corresp - a tuple of (c,?) arrays, each with the point identifiers
        of targets belonging to a quad/trip/etc per camera.
    num_targs - total number of targets (must be greater than the sum of
        previous 3).
    """
    # cdef:
    #     int pt, num_points
    #     coord_2d *corrected = <coord_2d *> malloc(sizeof(coord_2d *))

    num_points = len(img_pts)

    clique_targs = np.full((1, num_points, 2), PT_UNUSED, dtype=np.float64)
    clique_ids = np.full((1, num_points), CORRES_NONE, dtype=np.int_)

    # Trace back the pixel target properties through the flat metric
    # intermediary that's x-sorted.
    for pt in range(num_points):
        # From Beat code (issue #118) pix[0][geo[0][i].pnr].tnr=i;
        _, pnrs = corrected[pt].as_arrays()
        p1 = int(pnrs[pt])
        clique_ids[0, pt] = p1

        if p1 > -1:
            targ = img_pts[0][p1]
            clique_targs[0, pt, 0] = targ.x
            clique_targs[0, pt, 1] = targ.y
            # we also update the tnr, see docstring of correspondences
            targ.tnr = pt

    sorted_pos = [clique_targs]
    sorted_corresp = [clique_ids]

    return (sorted_pos, sorted_corresp, num_points)


def single_cam_correspondence(
    img_pts: List[Target],
    corrected: np.recarray,  # List[Coord2d]
):
    """Compatibility alias for the native single-camera correspondence API."""
    return single_cam_correspondences(img_pts, corrected)
