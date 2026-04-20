"""Camera orientation and bundle adjustment.

Translation of lib/src/orientation.c and lib/include/orientation.h.

Determines camera exterior orientation and refines interior/distortion
parameters using known 3D points and their 2D image projections.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

NPAR = 19
IDT = 10
NUM_ITER = 80
POS_INF = 1e20
CONVERGENCE = 0.00001
COORD_UNUSED = -1e10


def skew_midpoint(
    vert1: np.ndarray,
    direct1: np.ndarray,
    vert2: np.ndarray,
    direct2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Find midpoint of shortest distance segment between two skew rays."""
    sp_diff = vert2 - vert1
    perp_both = np.cross(direct1, direct2)
    scale = np.dot(perp_both, perp_both)

    if scale < 1e-20:
        return np.linalg.norm(sp_diff), (vert1 + vert2) / 2

    temp = np.cross(sp_diff, direct2)
    on1 = vert1 + direct1 * (np.dot(perp_both, temp) / scale)

    temp = np.cross(sp_diff, direct1)
    on2 = vert2 + direct2 * (np.dot(perp_both, temp) / scale)

    dist = np.linalg.norm(on1 - on2)
    midpoint = (on1 + on2) * 0.5

    return dist, midpoint


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
    from .ray_tracing import ray_tracing

    vertices = []
    directs = []

    for cam in range(num_cams):
        x, y = targets[cam, 0], targets[cam, 1]
        if x == COORD_UNUSED:
            vertices.append(None)
            directs.append(None)
            continue

        cal = cals[cam]
        pos, direction = ray_tracing(
            x, y,
            cal.ext_par.dm,
            cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
            cal.int_par.cc,
            cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
            mm.n1, mm.n2[0], mm.n3, mm.d[0],
        )
        vertices.append(pos)
        directs.append(direction)

    dtot = 0.0
    num_used_pairs = 0
    point_tot = np.zeros(3)

    for cam in range(num_cams):
        if vertices[cam] is None:
            continue
        for pair in range(cam + 1, num_cams):
            if vertices[pair] is None:
                continue
            num_used_pairs += 1
            d, point = skew_midpoint(
                vertices[cam], directs[cam],
                vertices[pair], directs[pair],
            )
            dtot += d
            point_tot += point

    if num_used_pairs == 0:
        return np.zeros(3), 0.0

    res = point_tot / num_used_pairs
    return res, dtot / num_used_pairs


def weighted_dumbbell_precision(targets, num_targs, num_cams, mm, cals,
                                db_length, db_weight):
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

    var_names = ['x0', 'y0', 'z0', 'omega', 'phi', 'kappa']

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

    dm = 0.0001
    drad = 0.0001

    cal.added_par.k1 = 0
    cal.added_par.k2 = 0
    cal.added_par.k3 = 0
    cal.added_par.p1 = 0
    cal.added_par.p2 = 0
    cal.added_par.scx = 1
    cal.added_par.she = 0

    itnum = 0
    stopflag = 0

    while stopflag == 0 and itnum < 20:
        itnum += 1

        X = np.zeros((2 * nfix, 6))
        y = np.zeros(2 * nfix)
        n = 0

        for i in range(nfix):
            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)

            cal.ext_par.compute_rotation_matrix()
            xp, yp = img_coord(np.asarray(fix[i]), cal, cpar.mm)

            x_ders, y_ders = num_deriv_exterior(cal, cpar, dm, drad,
                                                np.asarray(fix[i]))

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

    dm = 0.00001
    drad = 0.0000001

    cal = copy.deepcopy(cal_in)

    maxsize = nfix * 2 + IDT

    P = np.ones(maxsize)
    y = np.zeros(maxsize)
    X = np.zeros((maxsize, NPAR))

    for i in range(NPAR):
        sigmabeta[i] = 0.0

    numbers = 18 if flags.interfflag else 16

    glass_dir = np.array([cal.glass_par.vec_x, cal.glass_par.vec_y,
                          cal.glass_par.vec_z])
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

    ident = np.array([
        cal.int_par.cc, cal.int_par.xh, cal.int_par.yh,
        cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
        cal.added_par.p1, cal.added_par.p2,
        cal.added_par.scx, cal.added_par.she,
    ])

    safety_x = cal.glass_par.vec_x
    safety_y = cal.glass_par.vec_y
    safety_z = cal.glass_par.vec_z

    itnum = 0
    stopflag = 0

    while stopflag == 0 and itnum < NUM_ITER:
        itnum += 1

        X[:] = 0.0
        y[:] = 0.0
        P[:] = 1.0
        n = 0

        for i in range(nfix):
            if pix[i].pnr != i:
                continue

            if flags.useflag == 1 and (i % 2) == 0:
                continue
            if flags.useflag == 2 and (i % 2) != 0:
                continue
            if flags.useflag == 3 and (i % 3) == 0:
                continue

            xc, yc = pixel_to_metric(pix[i].x, pix[i].y, cpar)
            xc, yc = correct_brown_affin(
                xc, yc,
                cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
                cal.added_par.p1, cal.added_par.p2,
                cal.added_par.scx, cal.added_par.she,
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
            X[n, 14] = (xp * qq
                        + cal.added_par.p1 * (r * r + 2 * xp * xp)
                        + 2 * cal.added_par.p2 * xp * yp)
            X[n + 1, 14] = 0

            X[n, 15] = -np.cos(cal.added_par.she) * yp
            X[n + 1, 15] = -np.sin(cal.added_par.she) * yp

            x_ders, y_ders = num_deriv_exterior(cal, cpar, dm, drad,
                                                np.asarray(fix[i]))
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

    lines = man_path.read_text().strip().splitlines()
    if len(lines) < (cam + 1) * 4:
        return None

    nr = []
    for i in range(4):
        try:
            nr.append(int(lines[cam * 4 + i].strip()))
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


def read_calblock(filename):
    """Read calibration block file. Delegates to sortgrid.read_calblock."""
    from .sortgrid import read_calblock as _read_calblock
    return _read_calblock(filename)
