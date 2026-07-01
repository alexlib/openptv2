"""Camera orientation and bundle adjustment.

Translation of lib/src/orientation.c and lib/include/orientation.h.

Determines camera exterior orientation and refines interior/distortion
parameters using known 3D points and their 2D image projections.
"""

from __future__ import annotations

import cython

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt

import copy
from pathlib import Path

import numpy as np

NPAR = 19
IDT = 10
NUM_ITER = 80
POS_INF = 1e20
CONVERGENCE = 0.00001
COORD_UNUSED = -1e10


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def _skew_midpoint_core(
    v1_0: cython.double,
    v1_1: cython.double,
    v1_2: cython.double,
    d1_0: cython.double,
    d1_1: cython.double,
    d1_2: cython.double,
    v2_0: cython.double,
    v2_1: cython.double,
    v2_2: cython.double,
    d2_0: cython.double,
    d2_1: cython.double,
    d2_2: cython.double,
    midpoint: cython.double[:],
) -> cython.double:
    sp_x: cython.double = v2_0 - v1_0
    sp_y: cython.double = v2_1 - v1_1
    sp_z: cython.double = v2_2 - v1_2

    perp_x: cython.double = d1_1 * d2_2 - d1_2 * d2_1
    perp_y: cython.double = d1_2 * d2_0 - d1_0 * d2_2
    perp_z: cython.double = d1_0 * d2_1 - d1_1 * d2_0

    scale: cython.double = perp_x * perp_x + perp_y * perp_y + perp_z * perp_z

    if scale < 1e-20:
        midpoint[0] = (v1_0 + v2_0) * 0.5
        midpoint[1] = (v1_1 + v2_1) * 0.5
        midpoint[2] = (v1_2 + v2_2) * 0.5
        return c_sqrt(sp_x * sp_x + sp_y * sp_y + sp_z * sp_z)

    t1_x: cython.double = sp_y * d2_2 - sp_z * d2_1
    t1_y: cython.double = sp_z * d2_0 - sp_x * d2_2
    t1_z: cython.double = sp_x * d2_1 - sp_y * d2_0

    dot1: cython.double = perp_x * t1_x + perp_y * t1_y + perp_z * t1_z
    factor1: cython.double = dot1 / scale
    on1_x: cython.double = v1_0 + d1_0 * factor1
    on1_y: cython.double = v1_1 + d1_1 * factor1
    on1_z: cython.double = v1_2 + d1_2 * factor1

    t2_x: cython.double = sp_y * d1_2 - sp_z * d1_1
    t2_y: cython.double = sp_z * d1_0 - sp_x * d1_2
    t2_z: cython.double = sp_x * d1_1 - sp_y * d1_0

    dot2: cython.double = perp_x * t2_x + perp_y * t2_y + perp_z * t2_z
    factor2: cython.double = dot2 / scale
    on2_x: cython.double = v2_0 + d2_0 * factor2
    on2_y: cython.double = v2_1 + d2_1 * factor2
    on2_z: cython.double = v2_2 + d2_2 * factor2

    diff_x: cython.double = on1_x - on2_x
    diff_y: cython.double = on1_y - on2_y
    diff_z: cython.double = on1_z - on2_z
    dist: cython.double = c_sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)

    midpoint[0] = (on1_x + on2_x) * 0.5
    midpoint[1] = (on1_y + on2_y) * 0.5
    midpoint[2] = (on1_z + on2_z) * 0.5

    return dist


@cython.ccall
def skew_midpoint(
    vert1: np.ndarray,
    direct1: np.ndarray,
    vert2: np.ndarray,
    direct2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find midpoint of shortest distance segment between two skew rays."""
    midpoint = np.empty(3, dtype=np.float64)
    v1: cython.double[:] = vert1
    v2: cython.double[:] = vert2
    d1: cython.double[:] = direct1
    d2: cython.double[:] = direct2
    dist = _skew_midpoint_core(
        v1[0],
        v1[1],
        v1[2],
        d1[0],
        d1[1],
        d1[2],
        v2[0],
        v2[1],
        v2[2],
        d2[0],
        d2[1],
        d2[2],
        midpoint,
    )
    return dist, midpoint


@cython.ccall
def point_position(targets, num_cams, mm, cals):
    """Compute average 3D position from multiple camera rays.

    Args:
        targets: (num_cams, 2) array of metric flat coordinates.
        num_cams: number of cameras.
        mm: MultimediaPar or MmNp with n1, n2, n3, d attributes.
        cals: list of Calibration objects.

    Returns:
        (position, avg_ray_distance) tuple.
    """
    targets_mv: cython.double[:, :] = np.ascontiguousarray(targets, dtype=np.float64)
    t_3d = np.empty((1, num_cams, 2), dtype=np.float64)
    t_3d[0] = targets_mv
    positions, distances = point_position_batch(t_3d, num_cams, mm, cals)
    return positions[0], distances[0]


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def point_position_batch(targets, num_cams: cython.int, mm, cals):
    """Compute 3D positions from multiple camera rays for M targets.

    Args:
        targets: (M, num_cams, 2) array of metric flat coordinates.
        num_cams: number of cameras.
        mm: MmNp multimedia parameters.
        cals: list of Calibration objects.

    Returns:
        (positions, distances) — (M, 3) and (M,) float64 arrays.
    """
    targets_mv: cython.double[:, :, :] = np.ascontiguousarray(targets, dtype=np.float64)
    num_pts: cython.Py_ssize_t = targets_mv.shape[0]
    positions = np.empty((num_pts, 3), dtype=np.float64)
    distances = np.empty(num_pts, dtype=np.float64)
    pos_mv: cython.double[:, :] = positions
    dist_mv: cython.double[:] = distances

    # Preallocate scratch buffers once for the entire batch loop
    vertices = np.empty((num_cams, 3), dtype=np.float64)
    directs = np.empty((num_cams, 3), dtype=np.float64)
    vertices_mv: cython.double[:, :] = vertices
    directs_mv: cython.double[:, :] = directs
    used = np.empty(num_cams, dtype=np.int32)
    used_mv: cython.int[:] = used

    midpoint = np.empty(3, dtype=np.float64)
    midpoint_mv: cython.double[:] = midpoint

    i: cython.Py_ssize_t
    cam: cython.int
    pair: cython.int
    num_used_pairs: cython.int
    dtot: cython.double
    pt_tot_x: cython.double
    pt_tot_y: cython.double
    pt_tot_z: cython.double
    d: cython.double
    x: cython.double
    y: cython.double

    # 1. Unpack multimedia params once
    # Handle wrapper layers if needed
    if hasattr(mm, "_mm"):
        mm_obj = mm._mm
    else:
        mm_obj = mm
    mm_n1: cython.double = mm_obj.n1
    mm_n2_0: cython.double = mm_obj.n2[0]
    mm_n3: cython.double = mm_obj.n3
    mm_d0: cython.double = mm_obj.d[0]

    # 2. Extract calibration attributes for all cams into typed memoryviews
    ext_dm_all = np.empty((num_cams, 3, 3), dtype=np.float64)
    ext_x0_all = np.empty(num_cams, dtype=np.float64)
    ext_y0_all = np.empty(num_cams, dtype=np.float64)
    ext_z0_all = np.empty(num_cams, dtype=np.float64)
    int_cc_all = np.empty(num_cams, dtype=np.float64)

    glass_dir_x = np.empty(num_cams, dtype=np.float64)
    glass_dir_y = np.empty(num_cams, dtype=np.float64)
    glass_dir_z = np.empty(num_cams, dtype=np.float64)
    dist_cam_glass = np.empty(num_cams, dtype=np.float64)

    ext_dm_mv: cython.double[:, :, :] = ext_dm_all
    ext_x0_mv: cython.double[:] = ext_x0_all
    ext_y0_mv: cython.double[:] = ext_y0_all
    ext_z0_mv: cython.double[:] = ext_z0_all
    int_cc_mv: cython.double[:] = int_cc_all

    glass_dir_x_mv: cython.double[:] = glass_dir_x
    glass_dir_y_mv: cython.double[:] = glass_dir_y
    glass_dir_z_mv: cython.double[:] = glass_dir_z
    dist_cam_glass_mv: cython.double[:] = dist_cam_glass

    for cam in range(num_cams):
        cal = cals[cam]
        if hasattr(cal, "_cal"):
            cal_obj = cal._cal
        else:
            cal_obj = cal
        ext_dm_all[cam, :, :] = cal_obj.ext_par.dm
        ext_x0_mv[cam] = cal_obj.ext_par.x0
        ext_y0_mv[cam] = cal_obj.ext_par.y0
        ext_z0_mv[cam] = cal_obj.ext_par.z0
        int_cc_mv[cam] = cal_obj.int_par.cc

        g_x: cython.double = cal_obj.glass_par.vec_x
        g_y: cython.double = cal_obj.glass_par.vec_y
        g_z: cython.double = cal_obj.glass_par.vec_z
        norm_g: cython.double = c_sqrt(g_x * g_x + g_y * g_y + g_z * g_z)

        glass_dir_x_mv[cam] = g_x / norm_g
        glass_dir_y_mv[cam] = g_y / norm_g
        glass_dir_z_mv[cam] = g_z / norm_g

        c: cython.double = norm_g + mm_d0
        dist_cam_glass_mv[cam] = (
            glass_dir_x_mv[cam] * ext_x0_mv[cam]
            + glass_dir_y_mv[cam] * ext_y0_mv[cam]
            + glass_dir_z_mv[cam] * ext_z0_mv[cam]
        ) - c

    # Local variables for inlined ray tracing core
    tx: cython.double
    ty: cython.double
    tz: cython.double
    norm_tmp1: cython.double
    start_dir_x: cython.double
    start_dir_y: cython.double
    start_dir_z: cython.double
    dot_glass_start: cython.double
    d1: cython.double
    Xb_x: cython.double
    Xb_y: cython.double
    Xb_z: cython.double
    n: cython.double
    bp_x: cython.double
    bp_y: cython.double
    bp_z: cython.double
    norm_bp: cython.double
    p: cython.double
    n_glass: cython.double
    a2_x: cython.double
    a2_y: cython.double
    a2_z: cython.double
    dot_glass_a2: cython.double
    d2: cython.double
    X_x: cython.double
    X_y: cython.double
    X_z: cython.double
    n_a2: cython.double
    n_final: cython.double
    out_x: cython.double
    out_y: cython.double
    out_z: cython.double
    g_dx: cython.double
    g_dy: cython.double
    g_dz: cython.double

    for i in range(num_pts):
        for cam in range(num_cams):
            x = targets_mv[i, cam, 0]
            y = targets_mv[i, cam, 1]
            if x == COORD_UNUSED:
                used_mv[cam] = 0
                continue
            used_mv[cam] = 1

            cc: cython.double = int_cc_mv[cam]
            norm_tmp1 = c_sqrt(x * x + y * y + cc * cc)
            tx = x / norm_tmp1
            ty = y / norm_tmp1
            tz = -cc / norm_tmp1

            start_dir_x = (
                ext_dm_mv[cam, 0, 0] * tx
                + ext_dm_mv[cam, 0, 1] * ty
                + ext_dm_mv[cam, 0, 2] * tz
            )
            start_dir_y = (
                ext_dm_mv[cam, 1, 0] * tx
                + ext_dm_mv[cam, 1, 1] * ty
                + ext_dm_mv[cam, 1, 2] * tz
            )
            start_dir_z = (
                ext_dm_mv[cam, 2, 0] * tx
                + ext_dm_mv[cam, 2, 1] * ty
                + ext_dm_mv[cam, 2, 2] * tz
            )

            g_dx = glass_dir_x_mv[cam]
            g_dy = glass_dir_y_mv[cam]
            g_dz = glass_dir_z_mv[cam]

            dot_glass_start = (
                g_dx * start_dir_x + g_dy * start_dir_y + g_dz * start_dir_z
            )
            d1 = -dist_cam_glass_mv[cam] / dot_glass_start

            Xb_x = ext_x0_mv[cam] + start_dir_x * d1
            Xb_y = ext_y0_mv[cam] + start_dir_y * d1
            Xb_z = ext_z0_mv[cam] + start_dir_z * d1

            n = start_dir_x * g_dx + start_dir_y * g_dy + start_dir_z * g_dz
            bp_x = start_dir_x - g_dx * n
            bp_y = start_dir_y - g_dy * n
            bp_z = start_dir_z - g_dz * n
            norm_bp = c_sqrt(bp_x * bp_x + bp_y * bp_y + bp_z * bp_z)
            if norm_bp > 0:
                bp_x /= norm_bp
                bp_y /= norm_bp
                bp_z /= norm_bp

            p = c_sqrt(1.0 - n * n) * mm_n1 / mm_n2_0
            n_glass = -c_sqrt(1.0 - p * p)

            a2_x = bp_x * p + g_dx * n_glass
            a2_y = bp_y * p + g_dy * n_glass
            a2_z = bp_z * p + g_dz * n_glass

            dot_glass_a2 = g_dx * a2_x + g_dy * a2_y + g_dz * a2_z
            d2 = mm_d0 / abs(dot_glass_a2)

            X_x = Xb_x + a2_x * d2
            X_y = Xb_y + a2_y * d2
            X_z = Xb_z + a2_z * d2

            n_a2 = a2_x * g_dx + a2_y * g_dy + a2_z * g_dz
            bp_x = a2_x - g_dx * n_glass
            bp_y = a2_y - g_dy * n_glass
            bp_z = a2_z - g_dz * n_glass
            norm_bp = c_sqrt(bp_x * bp_x + bp_y * bp_y + bp_z * bp_z)
            if norm_bp > 0:
                bp_x /= norm_bp
                bp_y /= norm_bp
                bp_z /= norm_bp

            p = c_sqrt(1.0 - n_a2 * n_a2)
            p = p * mm_n2_0 / mm_n3
            n_final = -c_sqrt(1.0 - p * p)

            out_x = bp_x * p + g_dx * n_final
            out_y = bp_y * p + g_dy * n_final
            out_z = bp_z * p + g_dz * n_final

            vertices_mv[cam, 0] = X_x
            vertices_mv[cam, 1] = X_y
            vertices_mv[cam, 2] = X_z

            directs_mv[cam, 0] = out_x
            directs_mv[cam, 1] = out_y
            directs_mv[cam, 2] = out_z

        dtot = 0.0
        num_used_pairs = 0
        pt_tot_x = 0.0
        pt_tot_y = 0.0
        pt_tot_z = 0.0

        for cam in range(num_cams):
            if used_mv[cam] == 0:
                continue
            for pair in range(cam + 1, num_cams):
                if used_mv[pair] == 0:
                    continue
                num_used_pairs += 1
                d = _skew_midpoint_core(
                    vertices_mv[cam, 0],
                    vertices_mv[cam, 1],
                    vertices_mv[cam, 2],
                    directs_mv[cam, 0],
                    directs_mv[cam, 1],
                    directs_mv[cam, 2],
                    vertices_mv[pair, 0],
                    vertices_mv[pair, 1],
                    vertices_mv[pair, 2],
                    directs_mv[pair, 0],
                    directs_mv[pair, 1],
                    directs_mv[pair, 2],
                    midpoint_mv,
                )
                dtot += d
                pt_tot_x += midpoint_mv[0]
                pt_tot_y += midpoint_mv[1]
                pt_tot_z += midpoint_mv[2]

        if num_used_pairs == 0:
            pos_mv[i, 0] = 0.0
            pos_mv[i, 1] = 0.0
            pos_mv[i, 2] = 0.0
            dist_mv[i] = 0.0
        else:
            pos_mv[i, 0] = pt_tot_x / num_used_pairs
            pos_mv[i, 1] = pt_tot_y / num_used_pairs
            pos_mv[i, 2] = pt_tot_z / num_used_pairs
            dist_mv[i] = dtot / num_used_pairs

    return positions, distances


@cython.ccall
def weighted_dumbbell_precision(
    targets, num_targs, num_cams, mm, cals, db_length, db_weight
):
    """Weighted sum of dumbbell precision measures.

    Args:
        targets: (num_targs, num_cams, 2) array.
        num_targs: number of target points.
        num_cams: number of cameras.
        mm: multimedia parameters.
        cals: list of Calibration objects.
        db_length: expected dumbbell length.
        db_weight: weight of length error vs ray convergence.

    Returns:
        Weighted precision measure (float).
    """
    dtot = 0.0
    len_err_tot = 0.0
    res = [np.zeros(3), np.zeros(3)]

    for pt in range(num_targs):
        targs_pt = targets[pt]
        r, d = point_position(targs_pt, num_cams, mm, cals)
        res[pt % 2] = r
        dtot += d

        if pt % 2 == 1:
            diff = res[0] - res[1]
            dist = np.linalg.norm(diff)
            if dist > db_length:
                len_err_tot += 1 - db_length / dist
            else:
                len_err_tot += 1 - dist / db_length

    return dtot / num_targs + db_weight * len_err_tot / (0.5 * num_targs)


@cython.ccall
def num_deriv_exterior(cal, cpar, dpos, dang, pos):
    """Compute numerical derivatives of image coords w.r.t. exterior params.

    Args:
        cal: Calibration object (temporarily modified, then restored).
        cpar: ControlPar object.
        dpos: position step.
        dang: angle step.
        pos: 3D world position.

    Returns:
        (x_ders, y_ders) each shape (6,).
    """
    from .imgcoord import img_coord

    cal.ext_par.compute_rotation_matrix()
    xs, ys = img_coord(pos, cal, cpar.mm)

    x_ders = np.zeros(6)
    y_ders = np.zeros(6)

    var_names = ["x0", "y0", "z0", "omega", "phi", "kappa"]

    for pd in range(6):
        step = dang if pd > 2 else dpos

        orig = getattr(cal.ext_par, var_names[pd])
        setattr(cal.ext_par, var_names[pd], orig + step)

        if pd > 2:
            cal.ext_par.compute_rotation_matrix()

        xpd, ypd = img_coord(pos, cal, cpar.mm)
        x_ders[pd] = (xpd - xs) / step
        y_ders[pd] = (ypd - ys) / step

        setattr(cal.ext_par, var_names[pd], orig)

    cal.ext_par.compute_rotation_matrix()
    return x_ders, y_ders


@cython.ccall
def raw_orient(cal, cpar, nfix, fix, pix):
    """Simplified orientation using only 6 exterior parameters.

    Args:
        cal: Calibration (modified in place on success).
        cpar: ControlPar.
        nfix: number of fix points.
        fix: list/array of 3D positions, shape (nfix, 3).
        pix: list of Target objects with .x, .y attributes.

    Returns:
        True on success, False on failure.
    """
    from .imgcoord import img_coord
    from .trafo import pixel_to_metric, correct_brown_affin
    from .lsqadj import ata, atl, matinv, matmul

    dm: cython.double = 0.0001
    drad: cython.double = 0.0001

    cal.added_par.k1 = 0
    cal.added_par.k2 = 0
    cal.added_par.k3 = 0
    cal.added_par.p1 = 0
    cal.added_par.p2 = 0
    cal.added_par.scx = 1
    cal.added_par.she = 0

    itnum: cython.int = 0
    stopflag: cython.int = 0
    i: cython.int
    n: cython.int

    while stopflag == 0 and itnum < 20:
        itnum += 1

        X = np.zeros((2 * nfix, 6))
        y = np.zeros(2 * nfix)
        n = 0

        for i in range(nfix):
            x_val = pix[i].x
            x_px = x_val() if callable(x_val) else x_val
            y_val = pix[i].y
            y_px = y_val() if callable(y_val) else y_val
            xc, yc = pixel_to_metric(x_px, y_px, cpar)

            cal.ext_par.compute_rotation_matrix()
            xp, yp = img_coord(np.asarray(fix[i]), cal, cpar.mm)

            x_ders, y_ders = num_deriv_exterior(cal, cpar, dm, drad, np.asarray(fix[i]))

            X[n, :] = x_ders
            X[n + 1, :] = y_ders

            y[n] = xc - xp
            y[n + 1] = yc - yp

            n += 2

        XPX = ata(X[:n], n, 6)
        XPX = matinv(XPX, 6)
        XPy = atl(X[:n], y[:n], n, 6)
        beta = matmul(XPX, XPy, 6, 6)

        stopflag = 1
        for i in range(6):
            if abs(beta[i]) > 0.1:
                stopflag = 0

        cal.ext_par.x0 += beta[0]
        cal.ext_par.y0 += beta[1]
        cal.ext_par.z0 += beta[2]
        cal.ext_par.omega += beta[3]
        cal.ext_par.phi += beta[4]
        cal.ext_par.kappa += beta[5]

    if stopflag:
        cal.ext_par.compute_rotation_matrix()

    return bool(stopflag)


@cython.ccall
def orient(cal_in, cpar, nfix, fix, pix, flags, sigmabeta):
    """Bundle adjustment using Gauss-Markov model.

    Args:
        cal_in: Calibration (modified in place on success).
        cpar: ControlPar.
        nfix: number of fix points.
        fix: (nfix, 3) array of known 3D positions.
        pix: list of Target objects with .x, .y, .pnr attributes.
        flags: OrientPar with flags for which params to adjust.
        sigmabeta: output array of size 20 for parameter deviations.

    Returns:
        Array of residuals on success, None on failure.
    """
    from .imgcoord import img_coord
    from .trafo import pixel_to_metric, correct_brown_affin
    from .lsqadj import ata, atl, matinv, matmul
    from .vec_utils import vec_set, unit_vector, vec_norm

    dm: cython.double = 0.00001
    drad: cython.double = 0.0000001

    cal = copy.deepcopy(cal_in)

    maxsize = nfix * 2 + IDT

    P = np.ones(maxsize)
    y = np.zeros(maxsize)
    X = np.zeros((maxsize, NPAR))

    for i in range(NPAR):
        sigmabeta[i] = 0.0

    numbers = 18 if flags.interfflag else 16

    glass_dir = np.array(
        [cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z]
    )
    nGl = vec_norm(glass_dir)

    e1_x = 2 * cal.glass_par.vec_z - 3 * cal.glass_par.vec_x
    e1_y = 3 * cal.glass_par.vec_x - 1 * cal.glass_par.vec_z
    e1_z = 1 * cal.glass_par.vec_y - 2 * cal.glass_par.vec_y
    e1 = unit_vector(np.array([e1_x, e1_y, e1_z]))

    e2_x = e1[1] * cal.glass_par.vec_z - e1[2] * cal.glass_par.vec_x
    e2_y = e1[2] * cal.glass_par.vec_x - e1[0] * cal.glass_par.vec_z
    e2_z = e1[0] * cal.glass_par.vec_y - e1[1] * cal.glass_par.vec_y
    e2 = unit_vector(np.array([e2_x, e2_y, e2_z]))

    al = 0.0
    be = 0.0
    ga = 0.0

    ident = np.array(
        [
            cal.int_par.cc,
            cal.int_par.xh,
            cal.int_par.yh,
            cal.added_par.k1,
            cal.added_par.k2,
            cal.added_par.k3,
            cal.added_par.p1,
            cal.added_par.p2,
            cal.added_par.scx,
            cal.added_par.she,
        ]
    )

    safety_x = cal.glass_par.vec_x
    safety_y = cal.glass_par.vec_y
    safety_z = cal.glass_par.vec_z

    itnum: cython.int = 0
    stopflag: cython.int = 0
    i: cython.int
    n: cython.int
    n_obs: cython.int
    pd: cython.int

    while stopflag == 0 and itnum < NUM_ITER:
        itnum += 1

        X[:] = 0.0
        y[:] = 0.0
        P[:] = 1.0
        n = 0

        for i in range(nfix):
            pnr_val = pix[i].pnr
            pnr_i = pnr_val() if callable(pnr_val) else pnr_val
            if pnr_i != i:
                continue

            if flags.useflag == 1 and (i % 2) == 0:
                continue
            if flags.useflag == 2 and (i % 2) != 0:
                continue
            if flags.useflag == 3 and (i % 3) == 0:
                continue

            x_val = pix[i].x
            x_px = x_val() if callable(x_val) else x_val
            y_val = pix[i].y
            y_px = y_val() if callable(y_val) else y_val
            xc, yc = pixel_to_metric(x_px, y_px, cpar)
            xc, yc = correct_brown_affin(
                xc,
                yc,
                cal.added_par.k1,
                cal.added_par.k2,
                cal.added_par.k3,
                cal.added_par.p1,
                cal.added_par.p2,
                cal.added_par.scx,
                cal.added_par.she,
            )

            cal.ext_par.compute_rotation_matrix()
            xp, yp = img_coord(np.asarray(fix[i]), cal, cpar.mm)

            r = np.sqrt(xp * xp + yp * yp)

            X[n, 7] = cal.added_par.scx
            X[n + 1, 7] = np.sin(cal.added_par.she)

            X[n, 8] = 0
            X[n + 1, 8] = 1

            X[n, 9] = cal.added_par.scx * xp * r * r
            X[n + 1, 9] = yp * r * r

            X[n, 10] = cal.added_par.scx * xp * r**4
            X[n + 1, 10] = yp * r**4

            X[n, 11] = cal.added_par.scx * xp * r**6
            X[n + 1, 11] = yp * r**6

            X[n, 12] = cal.added_par.scx * (2 * xp * xp + r * r)
            X[n + 1, 12] = 2 * xp * yp

            X[n, 13] = 2 * cal.added_par.scx * xp * yp
            X[n + 1, 13] = 2 * yp * yp + r * r

            qq = cal.added_par.k1 * r * r
            qq += cal.added_par.k2 * r**4
            qq += cal.added_par.k3 * r**6
            qq += 1
            X[n, 14] = (
                xp * qq
                + cal.added_par.p1 * (r * r + 2 * xp * xp)
                + 2 * cal.added_par.p2 * xp * yp
            )
            X[n + 1, 14] = 0

            X[n, 15] = -np.cos(cal.added_par.she) * yp
            X[n + 1, 15] = -np.sin(cal.added_par.she) * yp

            x_ders, y_ders = num_deriv_exterior(cal, cpar, dm, drad, np.asarray(fix[i]))
            X[n, 0:6] = x_ders
            X[n + 1, 0:6] = y_ders

            # cc derivative
            cal.int_par.cc += dm
            cal.ext_par.compute_rotation_matrix()
            xpd, ypd = img_coord(np.asarray(fix[i]), cal, cpar.mm)
            X[n, 6] = (xpd - xp) / dm
            X[n + 1, 6] = (ypd - yp) / dm
            cal.int_par.cc -= dm

            # glass interface derivatives
            al += dm
            cal.glass_par.vec_x += e1[0] * nGl * al
            cal.glass_par.vec_y += e1[1] * nGl * al
            cal.glass_par.vec_z += e1[2] * nGl * al
            xpd, ypd = img_coord(np.asarray(fix[i]), cal, cpar.mm)
            X[n, 16] = (xpd - xp) / dm
            X[n + 1, 16] = (ypd - yp) / dm
            al -= dm
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z

            be += dm
            cal.glass_par.vec_x += e2[0] * nGl * be
            cal.glass_par.vec_y += e2[1] * nGl * be
            cal.glass_par.vec_z += e2[2] * nGl * be
            xpd, ypd = img_coord(np.asarray(fix[i]), cal, cpar.mm)
            X[n, 17] = (xpd - xp) / dm
            X[n + 1, 17] = (ypd - yp) / dm
            be -= dm
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z

            ga += dm
            cal.glass_par.vec_x += cal.glass_par.vec_x * nGl * ga
            cal.glass_par.vec_y += cal.glass_par.vec_y * nGl * ga
            cal.glass_par.vec_z += cal.glass_par.vec_z * nGl * ga
            xpd, ypd = img_coord(np.asarray(fix[i]), cal, cpar.mm)
            X[n, 18] = (xpd - xp) / dm
            X[n + 1, 18] = (ypd - yp) / dm
            ga -= dm
            cal.glass_par.vec_x = safety_x
            cal.glass_par.vec_y = safety_y
            cal.glass_par.vec_z = safety_z

            y[n] = xc - xp
            y[n + 1] = yc - yp

            n += 2

        n_obs = n

        # identity constraints
        for i in range(IDT):
            X[n_obs + i, 6 + i] = 1

        y[n_obs + 0] = ident[0] - cal.int_par.cc
        y[n_obs + 1] = ident[1] - cal.int_par.xh
        y[n_obs + 2] = ident[2] - cal.int_par.yh
        y[n_obs + 3] = ident[3] - cal.added_par.k1
        y[n_obs + 4] = ident[4] - cal.added_par.k2
        y[n_obs + 5] = ident[5] - cal.added_par.k3
        y[n_obs + 6] = ident[6] - cal.added_par.p1
        y[n_obs + 7] = ident[7] - cal.added_par.p2
        y[n_obs + 8] = ident[8] - cal.added_par.scx
        y[n_obs + 9] = ident[9] - cal.added_par.she

        P[n_obs + 0] = 1 if flags.ccflag else POS_INF
        P[n_obs + 1] = 1 if flags.xhflag else POS_INF
        P[n_obs + 2] = 1 if flags.yhflag else POS_INF
        P[n_obs + 3] = 1 if flags.k1flag else POS_INF
        P[n_obs + 4] = 1 if flags.k2flag else POS_INF
        P[n_obs + 5] = 1 if flags.k3flag else POS_INF
        P[n_obs + 6] = 1 if flags.p1flag else POS_INF
        P[n_obs + 7] = 1 if flags.p2flag else POS_INF
        P[n_obs + 8] = 1 if flags.scxflag else POS_INF
        P[n_obs + 9] = 1 if flags.sheflag else POS_INF

        n_obs += IDT

        # homogenize
        Xh = np.zeros_like(X[:n_obs])
        yh = np.zeros(n_obs)
        for i in range(n_obs):
            p = np.sqrt(P[i])
            Xh[i] = p * X[i]
            yh[i] = p * y[i]

        XPX = ata(Xh, n_obs, numbers)
        XPX = matinv(XPX, numbers)
        XPy = atl(Xh, yh, n_obs, numbers)
        beta = matmul(XPX, XPy, numbers, numbers)

        stopflag = 1
        for i in range(numbers):
            if abs(beta[i]) > CONVERGENCE:
                stopflag = 0

        if not flags.ccflag:
            beta[6] = 0.0
        if not flags.xhflag:
            beta[7] = 0.0
        if not flags.yhflag:
            beta[8] = 0.0
        if not flags.k1flag:
            beta[9] = 0.0
        if not flags.k2flag:
            beta[10] = 0.0
        if not flags.k3flag:
            beta[11] = 0.0
        if not flags.p1flag:
            beta[12] = 0.0
        if not flags.p2flag:
            beta[13] = 0.0
        if not flags.scxflag:
            beta[14] = 0.0
        if not flags.sheflag:
            beta[15] = 0.0

        cal.ext_par.x0 += beta[0]
        cal.ext_par.y0 += beta[1]
        cal.ext_par.z0 += beta[2]
        cal.ext_par.omega += beta[3]
        cal.ext_par.phi += beta[4]
        cal.ext_par.kappa += beta[5]
        cal.int_par.cc += beta[6]
        cal.int_par.xh += beta[7]
        cal.int_par.yh += beta[8]
        cal.added_par.k1 += beta[9]
        cal.added_par.k2 += beta[10]
        cal.added_par.k3 += beta[11]
        cal.added_par.p1 += beta[12]
        cal.added_par.p2 += beta[13]
        cal.added_par.scx += beta[14]
        cal.added_par.she += beta[15]

        if flags.interfflag:
            cal.glass_par.vec_x += e1[0] * nGl * beta[16]
            cal.glass_par.vec_y += e1[1] * nGl * beta[16]
            cal.glass_par.vec_z += e1[2] * nGl * beta[16]
            cal.glass_par.vec_x += e2[0] * nGl * beta[17]
            cal.glass_par.vec_y += e2[1] * nGl * beta[17]
            cal.glass_par.vec_z += e2[2] * nGl * beta[17]

    # compute residuals
    beta_full = np.zeros(NPAR)
    beta_full[:numbers] = beta[:numbers]
    Xbeta = X[:n_obs] @ beta_full
    omega = 0.0
    resi = np.zeros(n_obs)
    for i in range(n_obs):
        resi[i] = Xbeta[i] - y[i]
        omega += resi[i] * P[i] * resi[i]

    sigmabeta[NPAR] = np.sqrt(omega / (n_obs - numbers))
    for i in range(numbers):
        sigmabeta[i] = sigmabeta[NPAR] * np.sqrt(XPX[i, i])

    if stopflag:
        cal.ext_par.compute_rotation_matrix()
        cal_in.ext_par = copy.deepcopy(cal.ext_par)
        cal_in.int_par = copy.deepcopy(cal.int_par)
        cal_in.glass_par = copy.deepcopy(cal.glass_par)
        cal_in.added_par = copy.deepcopy(cal.added_par)
        cal_in.mmlut = copy.deepcopy(cal.mmlut)
        return resi
    else:
        return None


@cython.ccall
def read_man_ori_fix(calblock_filename, man_ori_filename, cam):
    """Read manual orientation fix points.

    Args:
        calblock_filename: path to calibration target file.
        man_ori_filename: path to manual orientation parameter file.
        cam: camera index (0-based).

    Returns:
        List of 4 vec3d arrays, or None on failure.
    """
    from .sortgrid import read_calblock as _read_calblock

    man_path = Path(man_ori_filename)
    if not man_path.exists():
        return None

    tokens = man_path.read_text().split()
    if len(tokens) < (cam + 1) * 4:
        return None

    nr = []
    for i in range(4):
        try:
            nr.append(int(tokens[cam * 4 + i]))
        except (ValueError, IndexError):
            return None

    fix, num_fix = _read_calblock(calblock_filename)
    if num_fix < 4:
        return None

    fix4 = []
    for i in range(4):
        pnr = nr[i] - 1
        if 0 <= pnr < num_fix:
            fix4.append(fix[pnr].copy())
        else:
            return None

    return fix4


@cython.ccall
def read_calblock(filename):
    """Read calibration block file. Delegates to sortgrid.read_calblock."""
    from .sortgrid import read_calblock as _read_calblock

    return _read_calblock(filename)


@cython.ccall
def external_calibration(cal, ref_pts, img_pts, cpar):
    """Update exterior calibration from known 3D-2D correspondences.

    Thin wrapper around raw_orient(). Converts pixel-coordinate arrays
    into Target objects and calls the iterative 6-parameter solver.

    Args:
        cal: Calibration object (modified in place on success).
        ref_pts: (n, 3) array of known 3D positions.
        img_pts: (n, 2) array of pixel coordinates.
        cpar: ControlPar object.

    Returns:
        True if iteration converged, False otherwise.
    """
    from .tracking_frame_buf import Target

    ref_pts = np.ascontiguousarray(ref_pts, dtype=np.float64)
    img_pts = np.ascontiguousarray(img_pts, dtype=np.float64)

    targs = []
    for i in range(len(img_pts)):
        targs.append(Target(pnr=i, x=img_pts[i, 0], y=img_pts[i, 1]))

    return raw_orient(cal, cpar, len(ref_pts), ref_pts, targs)


@cython.ccall
def full_calibration(cal, ref_pts, img_pts, cpar, flags=None):
    """Full calibration adjusting exterior, interior, and distortion params.

    Thin wrapper around orient(). Accepts either a list of Target objects
    or a list of flag name strings, converts to OrientPar, and calls the
    full bundle adjustment.

    Args:
        cal: Calibration object (modified in place on success).
        ref_pts: (n, 3) array of known 3D positions.
        img_pts: list of Target objects with .x, .y, .pnr attributes,
            ordered by matching reference point (as done by
            match_detection_to_ref).
        cpar: ControlPar object.
        flags: list of flag name strings to enable. Recognized:
            'cc', 'xh', 'yh', 'k1', 'k2', 'k3', 'p1', 'p2',
            'scale', 'shear'. If None, no flags enabled (raw-like).

    Returns:
        (residuals, used, err_est) tuple:
            residuals: (n, 2) array of x/y residuals per point.
            used: n-length array of target pnr values.
            err_est: (NPAR+1,) array of error estimates per DOF.

    Raises:
        ValueError: if orient() iteration did not converge.
    """
    from .parameters import OrientPar

    if flags is None:
        flags = []

    orient_par = OrientPar(
        useflag=0,
        ccflag=1 if "cc" in flags else 0,
        xhflag=1 if "xh" in flags else 0,
        yhflag=1 if "yh" in flags else 0,
        k1flag=1 if "k1" in flags else 0,
        k2flag=1 if "k2" in flags else 0,
        k3flag=1 if "k3" in flags else 0,
        p1flag=1 if "p1" in flags else 0,
        p2flag=1 if "p2" in flags else 0,
        scxflag=1 if "scale" in flags else 0,
        sheflag=1 if "shear" in flags else 0,
        interfflag=0,
    )

    ref_pts = np.ascontiguousarray(ref_pts, dtype=np.float64)

    # Accept either Target objects or a raw (n, 2) pixel-coordinate array.
    # orient() indexes img_pts[i].pnr/.x/.y, so a bare ndarray (as passed by
    # run_standalone_calibration) must be wrapped first — mirrors the conversion
    # done in external_calibration(). pnr=i keeps every point (orient skips rows
    # whose pnr != i).
    if len(img_pts) > 0 and not hasattr(img_pts[0], "x"):
        from .tracking_frame_buf import Target

        pts = np.ascontiguousarray(img_pts, dtype=np.float64)
        img_pts = [Target(pnr=i, x=pts[i, 0], y=pts[i, 1]) for i in range(len(pts))]

    sigmabeta = np.zeros(NPAR + 1)

    residuals = orient(cal, cpar, len(ref_pts), ref_pts, img_pts, orient_par, sigmabeta)

    if residuals is None:
        raise ValueError("Orientation iteration failed, need better setup.")

    n = len(img_pts)
    ret = np.empty((n, 2))
    used = np.empty(n, dtype=np.int32)

    for i in range(n):
        ret[i, 0] = residuals[2 * i]
        ret[i, 1] = residuals[2 * i + 1]
        pnr_val = img_pts[i].pnr
        used[i] = pnr_val() if callable(pnr_val) else pnr_val

    return ret, used, sigmabeta


@cython.ccall
def match_detection_to_ref(cal, ref_pts, img_pts, cpar, eps=25):
    """Match detected targets to reference 3D points via back-projection.

    Thin wrapper around sortgrid(). Projects reference points into the
    image and matches each to the nearest detected target within eps pixels.

    Args:
        cal: Calibration object.
        ref_pts: (n, 3) array of known 3D positions.
        img_pts: list of Target objects (detected points).
        cpar: ControlPar object.
        eps: pixel search radius (default 25).

    Returns:
        List of Target objects sorted by reference point order.
        Unmatched entries have pnr=-999.
    """
    from .sortgrid import sortgrid

    ref_pts = np.ascontiguousarray(ref_pts, dtype=np.float64)

    return sortgrid(cal, cpar, len(ref_pts), ref_pts, len(img_pts), eps, img_pts)


@cython.ccall
def multi_cam_point_positions(targets, cpar, cals):
    """Calculate 3D positions from multi-camera 2D projections.

    Convenience wrapper matching the Cython binding API signature.
    Uses point_position_batch() directly for speed.

    Args:
        targets: (num_targets, num_cams, 2) array of metric flat coordinates.
        cpar: ControlPar (used for multimedia parameters via cpar.mm).
        cals: list of Calibration objects.

    Returns:
        (positions, rcm) tuple:
            positions: (n, 3) array of 3D positions.
            rcm: n-length array of ray convergence measures.
    """
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    num_cams: cython.int = targets.shape[1]
    return point_position_batch(targets, num_cams, cpar.mm, cals)


@cython.ccall
def point_positions(targets, cpar, cals, vpar=None):
    """Dispatch to single or multi-camera point position calculation.

    Matches the Cython binding API: selects single_cam or multi_cam
    based on the number of calibrations provided.

    Args:
        targets: (num_targets, num_cams, 2) array of metric flat coords.
        cpar: ControlPar (used for multimedia parameters).
        cals: list of Calibration objects.
        vpar: VolumePar (required for single camera case only).

    Returns:
        (positions, rcm) tuple.
    """
    if len(cals) == 1:
        return single_cam_point_positions(targets, cpar, cals, vpar)
    elif len(cals) > 1:
        return multi_cam_point_positions(targets, cpar, cals)
    else:
        raise ValueError("wrong number of cameras in point_positions")


@cython.ccall
def single_cam_point_positions(targets, cpar, cals, vpar):
    """Calculate 3D positions from single-camera 2D projections.

    Uses ray tracing with z-plane intersection. For a single camera,
    the depth (z) is estimated from the volume parameters and the ray
    direction.

    Args:
        targets: (num_targets, 1, 2) array of metric flat coordinates.
        cpar: ControlPar.
        cals: list with one Calibration object.
        vpar: VolumePar with z_min_lay, z_max_lay for depth limits.

    Returns:
        (positions, rcm) tuple where rcm is zeros (no ray convergence
        measure for single camera).
    """
    from .ray_tracing import ray_tracing

    targets = np.ascontiguousarray(targets, dtype=np.float64)
    num_targets: cython.int = targets.shape[0]

    res = np.empty((num_targets, 3), dtype=np.float64)
    rcm = np.zeros(num_targets, dtype=np.float64)

    cal = cals[0]
    mm = cpar.mm
    z_mid: cython.double = 0.5 * (vpar.z_min_lay[0] + vpar.z_max_lay[0])

    pt: cython.int
    x: cython.double
    y: cython.double
    t: cython.double
    for pt in range(num_targets):
        x, y = targets[pt, 0, 0], targets[pt, 0, 1]
        pos, direct = ray_tracing(
            x,
            y,
            cal.ext_par.dm,
            cal.ext_par.x0,
            cal.ext_par.y0,
            cal.ext_par.z0,
            cal.int_par.cc,
            cal.glass_par.vec_x,
            cal.glass_par.vec_y,
            cal.glass_par.vec_z,
            mm.n1,
            mm.n2[0],
            mm.n3,
            mm.d[0],
        )
        if abs(direct[2]) > 1e-10:
            t = (z_mid - pos[2]) / direct[2]
            res[pt] = pos + t * direct
        else:
            res[pt] = pos

    return res, rcm


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
