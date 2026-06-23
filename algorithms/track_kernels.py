"""Compiled kernels for the tracking hot path.

Optimized implementations of point_to_pixel, multimed_r_nlay_iterative,
and full tracking loops. These run as plain Python when uncompiled and at
C speed when compiled via Cython 3 Pure Python Mode.
"""

import math
import numpy as np

import cython


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled


# Cal array layout (31 float64):
#  0-2:   ext_x0, ext_y0, ext_z0
#  3-11:  dm[0,0], dm[1,0], dm[2,0], dm[0,1], dm[1,1], dm[2,1], dm[0,2], dm[1,2], dm[2,2]
#  12:    int_cc
#  13-14: xh, yh
#  15-17: gx, gy, gz
#  18:    dist_o_glas
#  19:    inv_dog
#  20-23: mm_n1, mm_n2_0, mm_n3, mm_d0
#  24-30: k1, k2, k3, p1, p2, scx, she

CAL_ARRAY_SIZE = 31


@cython.ccall
def pack_cal_array(cal, mm):
    """Pack calibration into a flat float64 array for compiled kernels."""
    dist_o_glas: cython.double
    gx: cython.double; gy: cython.double; gz: cython.double
    ext = cal.ext_par
    ip = cal.int_par
    gp = cal.glass_par
    ap = cal.added_par
    gx, gy, gz = gp.vec_x, gp.vec_y, gp.vec_z
    dist_o_glas = math.sqrt(gx * gx + gy * gy + gz * gz)

    c = np.empty(CAL_ARRAY_SIZE, dtype=np.float64)
    c[0] = ext.x0;  c[1] = ext.y0;  c[2] = ext.z0
    c[3] = ext.dm[0, 0]; c[4] = ext.dm[1, 0]; c[5] = ext.dm[2, 0]
    c[6] = ext.dm[0, 1]; c[7] = ext.dm[1, 1]; c[8] = ext.dm[2, 1]
    c[9] = ext.dm[0, 2]; c[10] = ext.dm[1, 2]; c[11] = ext.dm[2, 2]
    c[12] = ip.cc
    c[13] = ip.xh; c[14] = ip.yh
    c[15] = gx; c[16] = gy; c[17] = gz
    c[18] = dist_o_glas
    c[19] = 1.0 / dist_o_glas
    c[20] = mm.n1; c[21] = mm.n2[0]; c[22] = mm.n3; c[23] = mm.d[0]
    c[24] = ap.k1; c[25] = ap.k2; c[26] = ap.k3
    c[27] = ap.p1; c[28] = ap.p2
    c[29] = ap.scx; c[30] = ap.she
    return c


@cython.ccall
def pack_mmlut(cal):
    """Pack mmlut into kernel-friendly arrays.

    Returns (data, origin, nr, nz, rw). If no mmlut, data has length 0.
    """
    mmlut = cal.mmlut
    if mmlut.data is not None:
        return (mmlut.data.astype(np.float64, copy=False),
                mmlut.origin.astype(np.float64, copy=False),
                mmlut.nr, mmlut.nz, float(mmlut.rw))
    return (np.empty(0, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            0, 0, 0.0)


@cython.boundscheck(False)
@cython.wraparound(False)
def _multimed_r_nlay_1layer(pos_x: cython.double, pos_y: cython.double, pos_z: cython.double,
                             ext_x0: cython.double, ext_y0: cython.double, ext_z0: cython.double,
                             mm_n1: cython.double, mm_n2_0: cython.double, mm_n3: cython.double, mm_d0: cython.double):
    """Single-layer iterative radial shift."""
    zout: cython.double
    dx: cython.double; dy: cython.double; r: cython.double; rq: cython.double
    it: cython.int
    denom: cython.double; beta1: cython.double; sin_beta1: cython.double
    arg: cython.double; beta2_0: cython.double; arg3: cython.double; beta3: cython.double
    rbeta: cython.double; rdiff: cython.double
    if mm_n1 == 1.0 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
        return 1.0

    zout = pos_z
    dx = pos_x - ext_x0
    dy = pos_y - ext_y0
    r = math.sqrt(dx * dx + dy * dy)
    rq = r

    for it in range(40):
        denom = ext_z0 - pos_z
        if denom == 0.0:
            return 1.0
        beta1 = math.atan(rq / denom)
        sin_beta1 = math.sin(beta1)

        arg = sin_beta1 * mm_n1 / mm_n2_0
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        beta2_0 = math.asin(arg)

        arg3 = sin_beta1 * mm_n1 / mm_n3
        if arg3 > 1.0:
            arg3 = 1.0
        elif arg3 < -1.0:
            arg3 = -1.0
        beta3 = math.asin(arg3)

        rbeta = ((ext_z0 - mm_d0) * math.tan(beta1)
                 + mm_d0 * math.tan(beta2_0)
                 - zout * math.tan(beta3))

        rdiff = r - rbeta
        rq += rdiff

        if abs(rdiff) < 0.001:
            break
    else:
        return 1.0

    if r != 0.0:
        return rq / r
    return 1.0


@cython.ccall
def point_to_pixel_fast(pos: cython.double[:], cal: cython.double[:],
                       mmlut_data: cython.double[:], mmlut_origin: cython.double[:],
                       mmlut_nr: cython.int, mmlut_nz: cython.int, mmlut_rw: cython.double,
                       imx_half: cython.double, imy_half: cython.double,
                       inv_pix_x: cython.double, inv_pix_y: cython.double,
                       chfield: cython.int):
    """Project 3D position to pixel coordinates.

    Args:
        pos: (3,) float64 array — 3D position.
        cal: (31,) float64 array — packed calibration.
        mmlut_data: float64 array — LUT data (length 0 if no LUT).
        mmlut_origin: (3,) float64 array — LUT origin.
        mmlut_nr, mmlut_nz: int — LUT grid dimensions.
        mmlut_rw: float — LUT grid spacing.
        imx_half, imy_half: float — half image dimensions.
        inv_pix_x, inv_pix_y: float — 1/pixel_size.
        chfield: int — interlace mode.

    Returns:
        (x_pixel, y_pixel) tuple.
    """
    pos0: cython.double; pos1: cython.double; pos2: cython.double
    ext_x0: cython.double; ext_y0: cython.double; ext_z0: cython.double
    dm00: cython.double; dm10: cython.double; dm20: cython.double
    dm01: cython.double; dm11: cython.double; dm21: cython.double
    dm02: cython.double; dm12: cython.double; dm22: cython.double
    int_cc: cython.double; xh: cython.double; yh: cython.double
    gx: cython.double; gy: cython.double; gz: cython.double
    inv_dog: cython.double; mm_n1: cython.double; mm_n2_0: cython.double; mm_n3: cython.double; mm_d0: cython.double
    k1: cython.double; k2: cython.double; k3: cython.double
    p1: cython.double; p2: cython.double; scx: cython.double; she: cython.double
    dot_cam: cython.double; dist_o_glas: cython.double; dist_cam_glas: cython.double
    dot_pos: cython.double; dist_point_glas: cython.double
    s_cam: cython.double; cc_x: cython.double; cc_y: cython.double; cc_z: cython.double
    s_pt: cython.double; cp_x: cython.double; cp_y: cython.double; cp_z: cython.double
    ext_t_z0: cython.double; s_d: cython.double
    ag_x: cython.double; ag_y: cython.double; ag_z: cython.double
    tmp_x: cython.double; tmp_y: cython.double; tmp_z: cython.double
    pos_t_0: cython.double; pos_t_2: cython.double
    radial_shift: cython.double
    has_mmlut: cython.bint
    tx: cython.double; ty: cython.double; tz: cython.double
    sz: cython.double; iz: cython.int
    R: cython.double; sr: cython.double; ir: cython.int
    v0: cython.int; v3: cython.int; mmf: cython.double
    X_t: cython.double
    s_z: cython.double; bx: cython.double; by: cython.double; bz: cython.double
    s_x: cython.double
    dx: cython.double; dy: cython.double; dz: cython.double; deno: cython.double
    x: cython.double; y: cython.double; r: cython.double; r2: cython.double; r4: cython.double
    radial_factor: cython.double; xd: cython.double; yd: cython.double
    sin_she: cython.double; cos_she: cython.double
    x_dist: cython.double; y_dist: cython.double
    x_pixel: cython.double; y_pixel: cython.double
    pos0 = pos[0]; pos1 = pos[1]; pos2 = pos[2]

    ext_x0 = cal[0]; ext_y0 = cal[1]; ext_z0 = cal[2]
    dm00 = cal[3]; dm10 = cal[4]; dm20 = cal[5]
    dm01 = cal[6]; dm11 = cal[7]; dm21 = cal[8]
    dm02 = cal[9]; dm12 = cal[10]; dm22 = cal[11]
    int_cc = cal[12]; xh = cal[13]; yh = cal[14]
    gx = cal[15]; gy = cal[16]; gz = cal[17]
    inv_dog = cal[19]
    mm_n1 = cal[20]; mm_n2_0 = cal[21]; mm_n3 = cal[22]; mm_d0 = cal[23]
    k1 = cal[24]; k2 = cal[25]; k3 = cal[26]
    p1 = cal[27]; p2 = cal[28]; scx = cal[29]; she = cal[30]

    # trans_cam_point
    dot_cam = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_o_glas = cal[18]
    dist_cam_glas = dot_cam * inv_dog - dist_o_glas - mm_d0

    dot_pos = pos0 * gx + pos1 * gy + pos2 * gz
    dist_point_glas = dot_pos * inv_dog - dist_o_glas

    s_cam = dist_cam_glas * inv_dog
    cc_x = ext_x0 - gx * s_cam
    cc_y = ext_y0 - gy * s_cam
    cc_z = ext_z0 - gz * s_cam

    s_pt = dist_point_glas * inv_dog
    cp_x = pos0 - gx * s_pt
    cp_y = pos1 - gy * s_pt
    cp_z = pos2 - gz * s_pt

    ext_t_z0 = dist_cam_glas + mm_d0

    s_d = mm_d0 * inv_dog
    ag_x = cc_x - gx * s_d
    ag_y = cc_y - gy * s_d
    ag_z = cc_z - gz * s_d
    tmp_x = cp_x - ag_x
    tmp_y = cp_y - ag_y
    tmp_z = cp_z - ag_z

    pos_t_0 = math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
    pos_t_2 = dist_point_glas

    # mmlut lookup + multimed_nlay
    radial_shift = 1.0
    has_mmlut = len(mmlut_data) > 0
    if has_mmlut:
        tx = pos_t_0 - mmlut_origin[0]
        ty = -mmlut_origin[1]
        tz = pos_t_2 - mmlut_origin[2]
        sz = tz / mmlut_rw
        iz = int(sz)
        sz -= iz
        R = math.sqrt(tx * tx + ty * ty)
        sr = R / mmlut_rw
        ir = int(sr)
        sr -= ir
        if ir <= mmlut_nr and iz >= 0 and iz <= mmlut_nz:
            v0 = ir * mmlut_nz + iz
            v3 = v0 + mmlut_nz + 1
            if v0 >= 0 and v3 <= mmlut_nr * mmlut_nz:
                mmf = (mmlut_data[v0] * (1.0 - sr) * (1.0 - sz)
                       + mmlut_data[v0 + 1] * (1.0 - sr) * sz
                       + mmlut_data[v0 + mmlut_nz] * sr * (1.0 - sz)
                       + mmlut_data[v3] * sr * sz)
                if mmf > 0.0:
                    radial_shift = mmf
    if radial_shift == 1.0:
        radial_shift = _multimed_r_nlay_1layer(
            pos_t_0, 0.0, pos_t_2, 0.0, 0.0, ext_t_z0,
            mm_n1, mm_n2_0, mm_n3, mm_d0,
        )
    X_t = pos_t_0 * radial_shift

    # back_trans_point
    s_z = -pos_t_2 * inv_dog
    bx = ag_x - gx * s_z
    by = ag_y - gy * s_z
    bz = ag_z - gz * s_z
    if pos_t_0 > 0.0:
        s_x = -X_t / pos_t_0
        bx -= tmp_x * s_x
        by -= tmp_y * s_x
        bz -= tmp_z * s_x

    # perspective projection
    dx = bx - ext_x0
    dy = by - ext_y0
    dz = bz - ext_z0
    deno = dm02 * dx + dm12 * dy + dm22 * dz
    x = -int_cc * (dm00 * dx + dm10 * dy + dm20 * dz) / deno
    y = -int_cc * (dm01 * dx + dm11 * dy + dm21 * dz) / deno

    # flat_to_dist + distort_brown_affin
    x += xh
    y += yh
    r = math.sqrt(x * x + y * y)
    if r < 1e-10:
        x_dist = 0.0
        y_dist = 0.0
    else:
        r2 = r * r
        r4 = r2 * r2
        radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
        xd = x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
        yd = y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
        sin_she = math.sin(she)
        cos_she = math.cos(she)
        x_dist = scx * (xd - sin_she * yd)
        y_dist = scx * cos_she * yd

    # metric_to_pixel
    x_pixel = x_dist * inv_pix_x + imx_half
    y_pixel = imy_half - y_dist * inv_pix_y
    if chfield == 1:
        y_pixel = (y_pixel - 1.0) * 0.5
    elif chfield == 2:
        y_pixel = y_pixel * 0.5

    return x_pixel, y_pixel


PT_UNUSED = -999

@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def candsearch_in_pix_fast(targ_x: cython.double[:], targ_y: cython.double[:], targ_tnr: cython.int[:],
                           num_targets: cython.int,
                           cent_x: cython.double, cent_y: cython.double,
                           dl: cython.double, dr: cython.double, du: cython.double, dd: cython.double,
                           imx: cython.double, imy: cython.double, tr_unused: cython.int):
    """Find up to 4 closest candidates in pixel search area.

    Args:
        targ_x, targ_y: float64 arrays of target coordinates.
        targ_tnr: int32 array of target numbers (TR_UNUSED = unused).
        num_targets: number of valid targets.
        cent_x, cent_y: search center.
        dl, dr, du, dd: search margins (left, right, up, down).
        imx, imy: image dimensions.
        tr_unused: TR_UNUSED sentinel value.

    Returns:
        (p0, p1, p2, p3) — indices of up to 4 closest candidates,
        PT_UNUSED for empty slots.
    """
    xmin: cython.double; xmax: cython.double; ymin: cython.double; ymax: cython.double
    p1: cython.int; p2: cython.int; p3: cython.int; p4: cython.int
    d1: cython.double; d2: cython.double; d3: cython.double; d4: cython.double
    j0: cython.int; dj: cython.int; j: cython.int
    ty: cython.double; tx: cython.double
    dx: cython.double; dy: cython.double; d: cython.double
    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0: xmin = 0.0
    if xmax > imx: xmax = imx
    if ymin < 0.0: ymin = 0.0
    if ymax > imy: ymax = imy

    p1 = PT_UNUSED; p2 = PT_UNUSED; p3 = PT_UNUSED; p4 = PT_UNUSED
    d1 = 1e20; d2 = 1e20; d3 = 1e20; d4 = 1e20

    if not (0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy):
        return p1, p2, p3, p4

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if targ_y[j0] < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        ty = targ_y[j]
        if targ_tnr[j] != tr_unused:
            if ty > ymax:
                break
            tx = targ_x[j]
            if tx > xmin and tx < xmax and ty > ymin and ty < ymax:
                dx = cent_x - tx
                dy = cent_y - ty
                d = math.sqrt(dx * dx + dy * dy)

                if d < d1:
                    p4 = p3; p3 = p2; p2 = p1; p1 = j
                    d4 = d3; d3 = d2; d2 = d1; d1 = d
                elif d < d2:
                    p4 = p3; p3 = p2; p2 = j
                    d4 = d3; d3 = d2; d2 = d
                elif d < d3:
                    p4 = p3; p3 = j
                    d4 = d3; d3 = d
                elif d < d4:
                    p4 = j
                    d4 = d

    return p1, p2, p3, p4


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def candsearch_in_pix_rest_fast(targ_x: cython.double[:], targ_y: cython.double[:], targ_tnr: cython.int[:],
                                num_targets: cython.int,
                                cent_x: cython.double, cent_y: cython.double,
                                dl: cython.double, dr: cython.double, du: cython.double, dd: cython.double,
                                imx: cython.double, imy: cython.double, tr_unused: cython.int):
    """Find closest unused candidate.

    Returns:
        (index, count) — index of closest candidate with tnr==TR_UNUSED, count (0 or 1).
    """
    xmin: cython.double; xmax: cython.double; ymin: cython.double; ymax: cython.double
    best: cython.int; dmin: cython.double; counter: cython.int
    j0: cython.int; dj: cython.int; j: cython.int
    ty: cython.double; tx: cython.double
    dx: cython.double; dy: cython.double; d: cython.double
    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0: xmin = 0.0
    if xmax > imx: xmax = imx
    if ymin < 0.0: ymin = 0.0
    if ymax > imy: ymax = imy

    best = PT_UNUSED
    dmin = 1e20
    counter = 0

    if not (0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy):
        return best, 0

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if targ_y[j0] < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        ty = targ_y[j]
        if targ_tnr[j] == tr_unused:
            if ty > ymax:
                break
            tx = targ_x[j]
            if tx > xmin and tx < xmax and ty > ymin and ty < ymax:
                dx = cent_x - tx
                dy = cent_y - ty
                d = math.sqrt(dx * dx + dy * dy)
                if d < dmin:
                    dmin = d
                    best = j
                    counter = 1

    return best, counter


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def searchquader_fast(point: cython.double[:], quader: cython.double[:, :], num_cams: cython.int,
                     cal_arrays, mmlut_datas, mmlut_origins,
                     mmlut_nrs, mmlut_nzs, mmlut_rws,
                     imx_half: cython.double, imy_half: cython.double,
                     inv_pix_x: cython.double, inv_pix_y: cython.double,
                     chfield: cython.int, imx: cython.double, imy: cython.double):
    """Compute search area for all cameras.

    Projects point + 8 corner points through all cameras in a single compiled call,
    eliminating per-projection dispatch overhead.

    Args:
        point: (3,) float64 — center position.
        quader: (8, 3) float64 — corner positions.
        num_cams: int.
        cal_arrays: tuple of (31,) float64 arrays, one per camera.
        mmlut_datas/origins/nrs/nzs/rws: tuple of mmlut params per camera.
        imx_half, imy_half, inv_pix_x, inv_pix_y, chfield: pixel conversion params.
        imx, imy: image dimensions.

    Returns:
        (xr, xl, yd, yu) — each (num_cams,) float64.
    """
    i: cython.Py_ssize_t; pt: cython.Py_ssize_t
    xr_i: cython.double; xl_i: cython.double; yd_i: cython.double; yu_i: cython.double
    cx: cython.double; cy: cython.double; corner_x: cython.double; corner_y: cython.double
    mrw: cython.double; mnr: cython.int; mnz: cython.int
    xr = np.zeros(num_cams, dtype=np.float64)
    xl = np.zeros(num_cams, dtype=np.float64)
    yd = np.zeros(num_cams, dtype=np.float64)
    yu = np.zeros(num_cams, dtype=np.float64)

    for i in range(num_cams):
        cal = cal_arrays[i]
        md = mmlut_datas[i]
        mo = mmlut_origins[i]
        mnr = mmlut_nrs[i]
        mnz = mmlut_nzs[i]
        mrw = mmlut_rws[i]

        xr_i = 0.0
        xl_i = float(imx)
        yd_i = 0.0
        yu_i = float(imy)

        cx, cy = point_to_pixel_fast(point, cal, md, mo, mnr, mnz, mrw,
                                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)

        for pt in range(8):
            corner_x, corner_y = point_to_pixel_fast(
                quader[pt], cal, md, mo, mnr, mnz, mrw,
                imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
            if corner_x < xl_i: xl_i = corner_x
            if corner_y < yu_i: yu_i = corner_y
            if corner_x > xr_i: xr_i = corner_x
            if corner_y > yd_i: yd_i = corner_y

        if xl_i < 0.0: xl_i = 0.0
        if yu_i < 0.0: yu_i = 0.0
        if xr_i > imx: xr_i = imx
        if yd_i > imy: yd_i = imy

        xr[i] = xr_i - cx
        xl[i] = cx - xl_i
        yd[i] = yd_i - cy
        yu[i] = cy - yu_i

    return xr, xl, yd, yu


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def sort_candidates_by_freq_fast(ftnr: cython.int[:], freq: cython.int[:],
                                 whichcam: cython.int[:, :],
                                 n: cython.int, num_cams: cython.int, max_cands: cython.int):
    """Sort candidates by frequency, matches C algorithm.

    Args:
        ftnr: (n,) int32 — candidate target numbers (TR_UNUSED = -1).
        freq: (n,) int32 — frequency counts (zeroed on entry).
        whichcam: (n, num_cams) int32 — camera flags.
        n: total number of entries (num_cams * max_cands).
        num_cams: number of cameras.
        max_cands: candidates per camera (4).

    Returns:
        num_valid: number of valid candidates after sort.
    """
    i: cython.int; j: cython.int; m: cython.int; k: cython.int
    ftnr_i: cython.int; num_valid: cython.int
    tr_unused = -1

    for i in range(n):
        ftnr_i = ftnr[i]
        if ftnr_i == tr_unused:
            continue
        for j in range(num_cams):
            for m in range(max_cands):
                if ftnr_i == ftnr[max_cands * j + m]:
                    whichcam[i, j] = 1

    for i in range(n):
        if ftnr[i] != tr_unused:
            for j in range(num_cams):
                if whichcam[i, j] == 1:
                    freq[i] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq[j - 1] < freq[j]:
                ftnr[j - 1], ftnr[j] = ftnr[j], ftnr[j - 1]
                freq[j - 1], freq[j] = freq[j], freq[j - 1]
                for k in range(num_cams):
                    whichcam[j - 1, k], whichcam[j, k] = whichcam[j, k], whichcam[j - 1, k]

    for i in range(n):
        ftnr_i = ftnr[i]
        for j in range(i + 1, n):
            if ftnr[j] == ftnr_i or freq[j] < 2:
                freq[j] = 0
                ftnr[j] = tr_unused

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq[j - 1] < freq[j]:
                ftnr[j - 1], ftnr[j] = ftnr[j], ftnr[j - 1]
                freq[j - 1], freq[j] = freq[j], freq[j - 1]
                for k in range(num_cams):
                    whichcam[j - 1, k], whichcam[j, k] = whichcam[j, k], whichcam[j - 1, k]

    num_valid = 0
    for i in range(n):
        if freq[i] != 0:
            num_valid += 1
    return num_valid


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def sorted_candidates_fast(
    center: cython.double[:], center_proj_x: cython.double[:], center_proj_y: cython.double[:],
    num_cams: cython.int, max_cands: cython.int,
    cal_arrays, mmlut_datas, mmlut_origins, mmlut_nrs, mmlut_nzs, mmlut_rws,
    targ_x_tuple, targ_y_tuple, targ_tnr_tuple, num_targets,
    dvxmin: cython.double, dvxmax: cython.double, dvymin: cython.double, dvymax: cython.double,
    dvzmin: cython.double, dvzmax: cython.double,
    imx_half: cython.double, imy_half: cython.double,
    inv_pix_x: cython.double, inv_pix_y: cython.double,
    chfield: cython.int, imx: cython.double, imy: cython.double, tr_unused: cython.int,
):
    """Fused searchquader + candsearch + sort — single compiled entry.

    Returns (ftnr, freq, whichcam, num_valid).
    """
    n: cython.int; px: cython.double; py: cython.double; pz: cython.double
    i: cython.int; pt: cython.int
    xr_i: cython.double; xl_i: cython.double; yd_i: cython.double; yu_i: cython.double
    cx: cython.double; cy: cython.double; corner_x: cython.double; corner_y: cython.double
    mrw: cython.double; mnr: cython.int; mnz: cython.int
    cam: cython.int; base: cython.int; ci: cython.int; idx: cython.int
    ftnr_i: cython.int; num_valid: cython.int; j: cython.int; m: cython.int; k: cython.int
    p0: cython.int; p1: cython.int; p2: cython.int; p3: cython.int
    n = num_cams * max_cands

    # --- searchquader inlined ---
    px = center[0]; py = center[1]; pz = center[2]
    quader = np.empty((8, 3), dtype=np.float64)
    for pt in range(8):
        quader[pt, 0] = px + (dvxmax if pt & 1 else dvxmin)
        quader[pt, 1] = py + (dvymax if pt & 2 else dvymin)
        quader[pt, 2] = pz + (dvzmax if pt & 4 else dvzmin)

    xr = np.zeros(num_cams, dtype=np.float64)
    xl = np.zeros(num_cams, dtype=np.float64)
    yd = np.zeros(num_cams, dtype=np.float64)
    yu = np.zeros(num_cams, dtype=np.float64)

    for i in range(num_cams):
        cal = cal_arrays[i]
        md = mmlut_datas[i]; mo = mmlut_origins[i]
        mnr = mmlut_nrs[i]; mnz = mmlut_nzs[i]; mrw = mmlut_rws[i]

        xr_i = 0.0; xl_i = float(imx); yd_i = 0.0; yu_i = float(imy)
        cx, cy = point_to_pixel_fast(center, cal, md, mo, mnr, mnz, mrw,
                                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
        for pt in range(8):
            corner_x, corner_y = point_to_pixel_fast(
                quader[pt], cal, md, mo, mnr, mnz, mrw,
                imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
            if corner_x < xl_i: xl_i = corner_x
            if corner_y < yu_i: yu_i = corner_y
            if corner_x > xr_i: xr_i = corner_x
            if corner_y > yd_i: yd_i = corner_y
        if xl_i < 0.0: xl_i = 0.0
        if yu_i < 0.0: yu_i = 0.0
        if xr_i > imx: xr_i = imx
        if yd_i > imy: yd_i = imy
        xr[i] = xr_i - cx; xl[i] = cx - xl_i
        yd[i] = yd_i - cy; yu[i] = cy - yu_i

    # --- candsearch per camera, write directly into ftnr/whichcam ---
    ftnr = np.full(n, tr_unused, dtype=np.int32)
    freq = np.zeros(n, dtype=np.int32)
    whichcam = np.zeros((n, num_cams), dtype=np.int32)

    for cam in range(num_cams):
        p0, p1, p2, p3 = candsearch_in_pix_fast(
            targ_x_tuple[cam], targ_y_tuple[cam], targ_tnr_tuple[cam],
            num_targets[cam], center_proj_x[cam], center_proj_y[cam],
            xl[cam], xr[cam], yu[cam], yd[cam], imx, imy, tr_unused)

        base = cam * max_cands
        cands = (p0, p1, p2, p3)
        for ci in range(4):
            idx = cands[ci]
            if idx != PT_UNUSED:
                whichcam[base + ci, cam] = 1
                ftnr[base + ci] = int(targ_tnr_tuple[cam][idx])

    # --- sort_candidates_by_freq inlined ---
    for i in range(n):
        ftnr_i = ftnr[i]
        if ftnr_i == tr_unused:
            continue
        for j in range(num_cams):
            for m in range(max_cands):
                if ftnr_i == ftnr[max_cands * j + m]:
                    whichcam[i, j] = 1

    for i in range(n):
        if ftnr[i] != tr_unused:
            for j in range(num_cams):
                if whichcam[i, j] == 1:
                    freq[i] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq[j - 1] < freq[j]:
                ftnr[j - 1], ftnr[j] = ftnr[j], ftnr[j - 1]
                freq[j - 1], freq[j] = freq[j], freq[j - 1]
                for k in range(num_cams):
                    whichcam[j - 1, k], whichcam[j, k] = whichcam[j, k], whichcam[j - 1, k]

    for i in range(n):
        ftnr_i = ftnr[i]
        for j in range(i + 1, n):
            if ftnr[j] == ftnr_i or freq[j] < 2:
                freq[j] = 0
                ftnr[j] = tr_unused

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq[j - 1] < freq[j]:
                ftnr[j - 1], ftnr[j] = ftnr[j], ftnr[j - 1]
                freq[j - 1], freq[j] = freq[j], freq[j - 1]
                for k in range(num_cams):
                    whichcam[j - 1, k], whichcam[j, k] = whichcam[j, k], whichcam[j - 1, k]

    num_valid = 0
    for i in range(n):
        if freq[i] != 0:
            num_valid += 1
    return ftnr, freq, whichcam, num_valid


@cython.ccall
def angle_acc_fast(start_x: cython.double, start_y: cython.double, start_z: cython.double,
                  pred_x: cython.double, pred_y: cython.double, pred_z: cython.double,
                  cand_x: cython.double, cand_y: cython.double, cand_z: cython.double):
    """Compute angle and acceleration between predicted and candidate."""
    v0x: cython.double; v0y: cython.double; v0z: cython.double
    v1x: cython.double; v1y: cython.double; v1z: cython.double
    angle: cython.double; norm0: cython.double; norm1: cython.double; dot: cython.double
    dx: cython.double; dy: cython.double; dz: cython.double; acc: cython.double
    v0x = pred_x - start_x
    v0y = pred_y - start_y
    v0z = pred_z - start_z
    v1x = cand_x - start_x
    v1y = cand_y - start_y
    v1z = cand_z - start_z

    if v0x == -v1x and v0y == -v1y and v0z == -v1z:
        angle = 200.0
    elif v0x == v1x and v0y == v1y and v0z == v1z:
        angle = 0.0
    else:
        norm0 = math.sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
        norm1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if norm0 == 0.0 or norm1 == 0.0:
            angle = 0.0
        else:
            dot = (v0x * v1x + v0y * v1y + v0z * v1z) / (norm0 * norm1)
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            angle = math.acos(dot) * 200.0 / math.pi

    dx = v1x - v0x
    dy = v1y - v0y
    dz = v1z - v0z
    acc = math.sqrt(dx * dx + dy * dy + dz * dz)
    return angle, acc


@cython.boundscheck(False)
@cython.wraparound(False)
def _ray_tracing_fast(x: cython.double, y: cython.double, cal: cython.double[:]):
    """Trace ray through multi-media interface.

    Returns (Xx, Xy, Xz, ox, oy, oz) — crossing point and direction.
    """
    ext_x0: cython.double; ext_y0: cython.double; ext_z0: cython.double
    dm00: cython.double; dm10: cython.double; dm20: cython.double
    dm01: cython.double; dm11: cython.double; dm21: cython.double
    dm02: cython.double; dm12: cython.double; dm22: cython.double
    int_cc: cython.double; gx: cython.double; gy: cython.double; gz: cython.double
    mm_n1: cython.double; mm_n2_0: cython.double; mm_n3: cython.double; mm_d0: cython.double
    t0: cython.double; t1: cython.double; t2: cython.double; tn: cython.double
    sd0: cython.double; sd1: cython.double; sd2: cython.double
    gn: cython.double; gd0: cython.double; gd1: cython.double; gd2: cython.double
    c: cython.double; dcg: cython.double; denom: cython.double; d1: cython.double
    Xb0: cython.double; Xb1: cython.double; Xb2: cython.double
    n: cython.double; bp0: cython.double; bp1: cython.double; bp2: cython.double; bpn: cython.double
    p: cython.double; n_glass: cython.double
    a2_0: cython.double; a2_1: cython.double; a2_2: cython.double
    d2_denom: cython.double; d2: cython.double
    Xx: cython.double; Xy: cython.double; Xz: cython.double
    n_a2: cython.double; p2: cython.double; n_final: cython.double
    ox: cython.double; oy: cython.double; oz: cython.double
    ext_x0 = cal[0]; ext_y0 = cal[1]; ext_z0 = cal[2]
    dm00 = cal[3]; dm10 = cal[4]; dm20 = cal[5]
    dm01 = cal[6]; dm11 = cal[7]; dm21 = cal[8]
    dm02 = cal[9]; dm12 = cal[10]; dm22 = cal[11]
    int_cc = cal[12]
    gx = cal[15]; gy = cal[16]; gz = cal[17]
    mm_n1 = cal[20]; mm_n2_0 = cal[21]; mm_n3 = cal[22]; mm_d0 = cal[23]

    # tmp1 = unit_vector([x, y, -int_cc])
    t0 = x; t1 = y; t2 = -int_cc
    tn = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if tn > 0.0:
        t0 /= tn; t1 /= tn; t2 /= tn

    # start_dir = dm @ tmp1
    sd0 = dm00 * t0 + dm01 * t1 + dm02 * t2
    sd1 = dm10 * t0 + dm11 * t1 + dm12 * t2
    sd2 = dm20 * t0 + dm21 * t1 + dm22 * t2

    # glass_dir = unit_vector(glass_vec)
    gn = math.sqrt(gx * gx + gy * gy + gz * gz)
    if gn > 0.0:
        gd0 = gx / gn; gd1 = gy / gn; gd2 = gz / gn
    else:
        gd0 = 0.0; gd1 = 0.0; gd2 = 0.0
    c = gn + mm_d0

    # dist_cam_glass, d1
    dcg = gd0 * ext_x0 + gd1 * ext_y0 + gd2 * ext_z0 - c
    denom = gd0 * sd0 + gd1 * sd1 + gd2 * sd2
    d1 = -dcg / denom

    # Xb = primary_point + start_dir * d1
    Xb0 = ext_x0 + sd0 * d1
    Xb1 = ext_y0 + sd1 * d1
    Xb2 = ext_z0 + sd2 * d1

    # Decompose ray: n = dot(start_dir, glass_dir)
    n = sd0 * gd0 + sd1 * gd1 + sd2 * gd2
    # bp = unit_vector(start_dir - glass_dir * n)
    bp0 = sd0 - gd0 * n; bp1 = sd1 - gd1 * n; bp2 = sd2 - gd2 * n
    bpn = math.sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn; bp1 /= bpn; bp2 /= bpn

    # Snell's law: air -> glass
    p = math.sqrt(1.0 - n * n) * mm_n1 / mm_n2_0
    n_glass = -math.sqrt(1.0 - p * p)

    # a2 = bp * p + glass_dir * n_glass
    a2_0 = bp0 * p + gd0 * n_glass
    a2_1 = bp1 * p + gd1 * n_glass
    a2_2 = bp2 * p + gd2 * n_glass

    d2_denom = gd0 * a2_0 + gd1 * a2_1 + gd2 * a2_2
    d2 = mm_d0 / abs(d2_denom)

    # X = Xb + a2 * d2
    Xx = Xb0 + a2_0 * d2
    Xy = Xb1 + a2_1 * d2
    Xz = Xb2 + a2_2 * d2

    # Direction in next medium: Snell glass -> water
    n_a2 = a2_0 * gd0 + a2_1 * gd1 + a2_2 * gd2
    # bp = unit_vector(a2 - glass_dir * n_glass)
    bp0 = a2_0 - gd0 * n_glass; bp1 = a2_1 - gd1 * n_glass; bp2 = a2_2 - gd2 * n_glass
    bpn = math.sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn; bp1 /= bpn; bp2 /= bpn

    p2 = math.sqrt(1.0 - n_a2 * n_a2) * mm_n2_0 / mm_n3
    n_final = -math.sqrt(1.0 - p2 * p2)

    ox = bp0 * p2 + gd0 * n_final
    oy = bp1 * p2 + gd1 * n_final
    oz = bp2 * p2 + gd2 * n_final

    return Xx, Xy, Xz, ox, oy, oz


COORD_UNUSED = -1e10


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def point_position_fast(targets: cython.double[:, :], num_cams: cython.int, cal_arrays):
    """Compute 3D position from multiple camera rays.

    Args:
        targets: (num_cams, 2) float64 — metric flat coordinates per camera.
        num_cams: int.
        cal_arrays: tuple of (31,) float64 arrays, one per camera.

    Returns:
        (pos, avg_dist) — (3,) float64 position and average ray distance.
    """
    cam: cython.int; pair: cython.int
    tx: cython.double; ty: cython.double
    Xx: cython.double; Xy: cython.double; Xz: cython.double
    ox: cython.double; oy: cython.double; oz: cython.double
    dtot: cython.double; num_used: cython.int
    px: cython.double; py: cython.double; pz: cython.double
    v1x: cython.double; v1y: cython.double; v1z: cython.double
    d1x: cython.double; d1y: cython.double; d1z: cython.double
    v2x: cython.double; v2y: cython.double; v2z: cython.double
    d2x: cython.double; d2y: cython.double; d2z: cython.double
    sp0: cython.double; sp1: cython.double; sp2: cython.double
    pb0: cython.double; pb1: cython.double; pb2: cython.double; scale: cython.double
    dist: cython.double; mx: cython.double; my: cython.double; mz: cython.double
    t0: cython.double; t1: cython.double; t2: cython.double
    s1: cython.double; on1x: cython.double; on1y: cython.double; on1z: cython.double
    s2: cython.double; on2x: cython.double; on2y: cython.double; on2z: cython.double
    ddx: cython.double; ddy: cython.double; ddz: cython.double
    verts_x = np.empty(num_cams, dtype=np.float64)
    verts_y = np.empty(num_cams, dtype=np.float64)
    verts_z = np.empty(num_cams, dtype=np.float64)
    dirs_x = np.empty(num_cams, dtype=np.float64)
    dirs_y = np.empty(num_cams, dtype=np.float64)
    dirs_z = np.empty(num_cams, dtype=np.float64)
    valid = np.zeros(num_cams, dtype=np.int32)

    for cam in range(num_cams):
        tx = targets[cam, 0]; ty = targets[cam, 1]
        if tx == COORD_UNUSED:
            continue
        Xx, Xy, Xz, ox, oy, oz = _ray_tracing_fast(tx, ty, cal_arrays[cam])
        verts_x[cam] = Xx; verts_y[cam] = Xy; verts_z[cam] = Xz
        dirs_x[cam] = ox; dirs_y[cam] = oy; dirs_z[cam] = oz
        valid[cam] = 1

    dtot = 0.0
    num_used = 0
    px = 0.0; py = 0.0; pz = 0.0

    for cam in range(num_cams):
        if valid[cam] == 0:
            continue
        for pair in range(cam + 1, num_cams):
            if valid[pair] == 0:
                continue

            # skew_midpoint inlined
            v1x = verts_x[cam]; v1y = verts_y[cam]; v1z = verts_z[cam]
            d1x = dirs_x[cam]; d1y = dirs_y[cam]; d1z = dirs_z[cam]
            v2x = verts_x[pair]; v2y = verts_y[pair]; v2z = verts_z[pair]
            d2x = dirs_x[pair]; d2y = dirs_y[pair]; d2z = dirs_z[pair]

            sp0 = v2x - v1x; sp1 = v2y - v1y; sp2 = v2z - v1z

            # perp_both = cross(d1, d2)
            pb0 = d1y * d2z - d1z * d2y
            pb1 = d1z * d2x - d1x * d2z
            pb2 = d1x * d2y - d1y * d2x
            scale = pb0 * pb0 + pb1 * pb1 + pb2 * pb2

            if scale < 1e-20:
                dist = math.sqrt(sp0 * sp0 + sp1 * sp1 + sp2 * sp2)
                mx = (v1x + v2x) * 0.5
                my = (v1y + v2y) * 0.5
                mz = (v1z + v2z) * 0.5
            else:
                # temp = cross(sp, d2)
                t0 = sp1 * d2z - sp2 * d2y
                t1 = sp2 * d2x - sp0 * d2z
                t2 = sp0 * d2y - sp1 * d2x
                s1 = (pb0 * t0 + pb1 * t1 + pb2 * t2) / scale
                on1x = v1x + d1x * s1; on1y = v1y + d1y * s1; on1z = v1z + d1z * s1

                # temp = cross(sp, d1)
                t0 = sp1 * d1z - sp2 * d1y
                t1 = sp2 * d1x - sp0 * d1z
                t2 = sp0 * d1y - sp1 * d1x
                s2 = (pb0 * t0 + pb1 * t1 + pb2 * t2) / scale
                on2x = v2x + d2x * s2; on2y = v2y + d2y * s2; on2z = v2z + d2z * s2

                ddx = on1x - on2x; ddy = on1y - on2y; ddz = on1z - on2z
                dist = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                mx = (on1x + on2x) * 0.5
                my = (on1y + on2y) * 0.5
                mz = (on1z + on2z) * 0.5

            num_used += 1
            dtot += dist
            px += mx; py += my; pz += mz

    pos = np.zeros(3, dtype=np.float64)
    if num_used > 0:
        pos[0] = px / num_used
        pos[1] = py / num_used
        pos[2] = pz / num_used
        dtot /= num_used

    return pos, dtot


@cython.ccall
def pixel_to_metric_fast(x_pixel: cython.double, y_pixel: cython.double,
                         imx: cython.int, imy: cython.int,
                         pix_x: cython.double, pix_y: cython.double,
                         chfield: cython.int):
    """Convert pixel to metric coordinates."""
    yp: cython.double; x_metric: cython.double; y_metric: cython.double
    yp = y_pixel
    if chfield == 1:
        yp = 2.0 * yp + 1.0
    elif chfield == 2:
        yp = 2.0 * yp
    x_metric = (x_pixel - imx * 0.5) * pix_x
    y_metric = (imy * 0.5 - yp) * pix_y
    return x_metric, y_metric


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def dist_to_flat_fast(dist_x: cython.double, dist_y: cython.double,
                     xh: cython.double, yh: cython.double,
                     k1: cython.double, k2: cython.double, k3: cython.double,
                     p1: cython.double, p2: cython.double,
                     scx: cython.double, she: cython.double, tol: cython.double):
    """Inverse Brown distortion."""
    r_init: cython.double; sin_she: cython.double; cos_she: cython.double; inv_scx: cython.double
    xq: cython.double; yq: cython.double
    _: cython.int
    r2: cython.double; r4: cython.double; r6: cython.double
    radial_factor: cython.double; dx: cython.double; dy: cython.double
    xq_new: cython.double; yq_new: cython.double
    dx_change: cython.double; dy_change: cython.double
    r_init = math.sqrt(dist_x * dist_x + dist_y * dist_y)
    if r_init < 1e-10:
        return -xh, -yh

    sin_she = math.sin(she)
    cos_she = math.cos(she)
    inv_scx = 1.0 / scx

    xq = (dist_x + dist_y * sin_she) * inv_scx
    yq = dist_y / cos_she

    for _ in range(50):
        r2 = xq * xq + yq * yq
        r4 = r2 * r2
        r6 = r4 * r2

        radial_factor = k1 * r2 + k2 * r4 + k3 * r6

        dx = (xq * radial_factor
              + p1 * (r2 + 2.0 * xq * xq) + 2.0 * p2 * xq * yq)
        dy = (yq * radial_factor
              + p2 * (r2 + 2.0 * yq * yq) + 2.0 * p1 * xq * yq)

        xq_new = (dist_x + dist_y * sin_she) * inv_scx - dx
        yq_new = dist_y / cos_she - dy

        dx_change = xq_new - xq
        dy_change = yq_new - yq

        xq += 0.5 * dx_change
        yq += 0.5 * dy_change

        if math.sqrt(dx_change * dx_change + dy_change * dy_change) < tol:
            break

    return xq - xh, yq - yh


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def assess_new_position_fast(
    pos: cython.double[:], num_cams: cython.int, add_part: cython.double,
    cal_arrays, mmlut_datas, mmlut_origins, mmlut_nrs, mmlut_nzs, mmlut_rws,
    targ_x_tuple, targ_y_tuple, targ_tnr_tuple, num_targets,
    imx_half: cython.double, imy_half: cython.double,
    inv_pix_x: cython.double, inv_pix_y: cython.double,
    chfield: cython.int, imx: cython.int, imy: cython.int,
    pix_x: cython.double, pix_y: cython.double,
    flatten_tol: cython.double, tr_unused: cython.int, coord_unused: cython.double,
):
    """Assess new position: project, find unused targets, undistort.

    Returns (targ_pos, cand_inds, valid_cams).
    """
    cam: cython.int; valid_cams: cython.int; best: cython.int; count: cython.int
    px: cython.double; py: cython.double
    mx: cython.double; my: cython.double; fx: cython.double; fy: cython.double
    targ_pos = np.full((num_cams, 2), coord_unused, dtype=np.float64)
    cand_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)

    for cam in range(num_cams):
        px, py = point_to_pixel_fast(
            pos, cal_arrays[cam],
            mmlut_datas[cam], mmlut_origins[cam],
            mmlut_nrs[cam], mmlut_nzs[cam], mmlut_rws[cam],
            imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)

        best, count = candsearch_in_pix_rest_fast(
            targ_x_tuple[cam], targ_y_tuple[cam], targ_tnr_tuple[cam],
            num_targets[cam], px, py,
            add_part, add_part, add_part, add_part,
            imx, imy, tr_unused)

        if count > 0:
            cand_inds[cam] = best
            targ_pos[cam, 0] = targ_x_tuple[cam][best]
            targ_pos[cam, 1] = targ_y_tuple[cam][best]

    valid_cams = 0
    for cam in range(num_cams):
        if targ_pos[cam, 0] != coord_unused:
            mx, my = pixel_to_metric_fast(
                targ_pos[cam, 0], targ_pos[cam, 1],
                imx, imy, pix_x, pix_y, chfield)

            cal = cal_arrays[cam]
            fx, fy = dist_to_flat_fast(
                mx, my,
                cal[13], cal[14],
                cal[24], cal[25], cal[26],
                cal[27], cal[28],
                cal[29], cal[30],
                flatten_tol)

            targ_pos[cam, 0] = fx
            targ_pos[cam, 1] = fy
            valid_cams += 1

    return targ_pos, cand_inds, valid_cams


POSI_K = 80
MAX_CANDS_K = 4
TR_UNUSED_K = -1
CORRES_NONE_K = -1
PREV_NONE_K = -1
NEXT_NONE_K = -2
COORD_UNUSED_K = -1e10
ADD_PART_K = 3.0


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def trackcorr_loop_fast(
    orig_parts_1: cython.int,
    # Frame 0 (prev — read only)
    path_x_0: cython.double[:, :],
    # Frame 1 (curr — read/write)
    path_x_1: cython.double[:, :], path_prev_1: cython.int[:], path_next_1: cython.int[:], path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:], path_decis_1: cython.double[:, :], path_linkdecis_1: cython.int[:, :],
    corres_p_1: cython.int[:, :], targ_x_1, targ_y_1,
    # Frame 2 (next — read/write)
    path_x_2: cython.double[:, :], path_prev_2: cython.int[:], path_next_2: cython.int[:], path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:], path_finaldecis_2: cython.double[:], path_decis_2: cython.double[:, :], path_linkdecis_2: cython.int[:, :],
    corres_p_2: cython.int[:, :], corres_nr_2: cython.int[:],
    targ_x_2, targ_y_2, targ_tnr_2, num_targets_2: cython.int[:], num_parts_2: cython.int[:],
    # Frame 3 (next-next — read/write)
    path_x_3: cython.double[:, :], path_prev_3: cython.int[:], path_next_3: cython.int[:], path_inlist_3: cython.int[:],
    path_prio_3: cython.int[:], path_finaldecis_3: cython.double[:], path_decis_3: cython.double[:, :], path_linkdecis_3: cython.int[:, :],
    corres_p_3: cython.int[:, :], corres_nr_3: cython.int[:],
    targ_x_3, targ_y_3, targ_tnr_3, num_targets_3: cython.int[:], num_parts_3: cython.int[:],
    # Calibration
    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
    # Tracking params
    dvxmin: cython.double, dvxmax: cython.double, dvymin: cython.double, dvymax: cython.double, dvzmin: cython.double, dvzmax: cython.double,
    dacc: cython.double, dangle: cython.double, add_flag: cython.int, lmax: cython.double,
    # Volume bounds
    X_lay_0: cython.double, X_lay_1: cython.double, ymin: cython.double, ymax: cython.double, Zmin_lay_0: cython.double, Zmax_lay_1: cython.double,
    # Pixel params
    num_cams: cython.int,
    imx_half: cython.double, imy_half: cython.double, inv_pix_x: cython.double, inv_pix_y: cython.double,
    chfield: cython.int, imx: cython.double, imy: cython.double, pix_x: cython.double, pix_y: cython.double, flatten_tol: cython.double,
):
    """Full per-particle tracking loop + link resolution — single compiled entry.

    All internal calls (sorted_candidates, angle_acc, assess_new_position,
    point_position) are compiled with zero dispatch overhead.

    Args:
        num_parts_2, num_parts_3: (1,) int32 arrays — mutable particle counts.

    Returns:
        (count1, num_added) — number of links established and particles added.
    """
    count1: cython.int; num_added: cython.int; n_sc: cython.int
    h: cython.int; j: cython.int; mm: cython.int; kk: cython.int
    prev_h: cython.int; ftnr_mm: cython.int; ftnr_kk: cython.int
    ki: cython.int; ci: cython.int; inlist: cython.int
    np2: cython.int; np3: cython.int; in_volume: cython.int; quali: cython.int
    i: cython.int; ti: cython.int; cand: cython.int
    px: cython.double; py: cython.double
    dp0: cython.double; dp1: cython.double; dp2: cython.double
    angle1: cython.double; acc1: cython.double; angle0: cython.double; acc0: cython.double
    acc: cython.double; angle: cython.double; rr: cython.double
    d13: cython.double; d43: cython.double; dl: cython.double
    d01: cython.double; quali_f: cython.int
    count1 = 0
    num_added = 0
    n_sc = num_cams * MAX_CANDS_K

    cpx = np.empty(num_cams, dtype=np.float64)
    cpy = np.empty(num_cams, dtype=np.float64)
    X = np.zeros((6, 3), dtype=np.float64)

    for h in range(orig_parts_1):
        path_inlist_1[h] = 0

        X[1, 0] = path_x_1[h, 0]
        X[1, 1] = path_x_1[h, 1]
        X[1, 2] = path_x_1[h, 2]

        prev_h = path_prev_1[h]

        if prev_h >= 0:
            X[0, 0] = path_x_0[prev_h, 0]
            X[0, 1] = path_x_0[prev_h, 1]
            X[0, 2] = path_x_0[prev_h, 2]
            X[2, 0] = 2.0 * X[1, 0] - X[0, 0]
            X[2, 1] = 2.0 * X[1, 1] - X[0, 1]
            X[2, 2] = 2.0 * X[1, 2] - X[0, 2]

            for j in range(num_cams):
                px, py = point_to_pixel_fast(
                    X[2], cal_t[j], md_t[j], mo_t[j],
                    mnr_t[j], mnz_t[j], mrw_t[j],
                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
                cpx[j] = px
                cpy[j] = py
        else:
            X[2, 0] = X[1, 0]
            X[2, 1] = X[1, 1]
            X[2, 2] = X[1, 2]

            for j in range(num_cams):
                if corres_p_1[h, j] == CORRES_NONE_K:
                    px, py = point_to_pixel_fast(
                        X[2], cal_t[j], md_t[j], mo_t[j],
                        mnr_t[j], mnz_t[j], mrw_t[j],
                        imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
                    cpx[j] = px
                    cpy[j] = py
                else:
                    _ix = corres_p_1[h, j]
                    cpx[j] = targ_x_1[j][_ix]
                    cpy[j] = targ_y_1[j][_ix]

        # --- sorted_candidates for frame 2 ---
        w_ftnr, w_freq, w_wc, w_nc = sorted_candidates_fast(
            X[2], cpx, cpy, num_cams, MAX_CANDS_K,
            cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
            targ_x_2, targ_y_2, targ_tnr_2, num_targets_2,
            dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax,
            imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
            imx, imy, TR_UNUSED_K)

        if w_nc == 0:
            continue

        mm = 0
        while mm < w_nc:
            ftnr_mm = w_ftnr[mm]
            X[3, 0] = path_x_2[ftnr_mm, 0]
            X[3, 1] = path_x_2[ftnr_mm, 1]
            X[3, 2] = path_x_2[ftnr_mm, 2]

            if prev_h >= 0:
                for j in range(3):
                    X[5, j] = 0.5 * (5.0 * X[3, j] - 4.0 * X[1, j] + X[0, j])
            else:
                X[5, 0] = 2.0 * X[3, 0] - X[1, 0]
                X[5, 1] = 2.0 * X[3, 1] - X[1, 1]
                X[5, 2] = 2.0 * X[3, 2] - X[1, 2]

            for j in range(num_cams):
                px, py = point_to_pixel_fast(
                    X[5], cal_t[j], md_t[j], mo_t[j],
                    mnr_t[j], mnz_t[j], mrw_t[j],
                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
                cpx[j] = px
                cpy[j] = py

            # --- sorted_candidates for frame 3 ---
            wn_ftnr, wn_freq, wn_wc, wn_nc = sorted_candidates_fast(
                X[5], cpx, cpy, num_cams, MAX_CANDS_K,
                cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
                targ_x_3, targ_y_3, targ_tnr_3, num_targets_3,
                dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax,
                imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
                imx, imy, TR_UNUSED_K)

            if wn_nc > 0:
                kk = 0
                while kk < wn_nc:
                    ftnr_kk = wn_ftnr[kk]
                    X[4, 0] = path_x_3[ftnr_kk, 0]
                    X[4, 1] = path_x_3[ftnr_kk, 1]
                    X[4, 2] = path_x_3[ftnr_kk, 2]

                    dp0 = X[4, 0] - X[3, 0]
                    dp1 = X[4, 1] - X[3, 1]
                    dp2 = X[4, 2] - X[3, 2]

                    if (dvxmin < dp0 < dvxmax and
                            dvymin < dp1 < dvymax and
                            dvzmin < dp2 < dvzmax):
                        angle1, acc1 = angle_acc_fast(
                            X[3, 0], X[3, 1], X[3, 2],
                            X[4, 0], X[4, 1], X[4, 2],
                            X[5, 0], X[5, 1], X[5, 2])
                        if prev_h >= 0:
                            angle0, acc0 = angle_acc_fast(
                                X[1, 0], X[1, 1], X[1, 2],
                                X[2, 0], X[2, 1], X[2, 2],
                                X[3, 0], X[3, 1], X[3, 2])
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) * 0.5
                        angle = (angle0 + angle1) * 0.5
                        quali = wn_freq[kk] + w_freq[mm]

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = math.sqrt(
                                (X[1, 0] - X[3, 0]) ** 2 +
                                (X[1, 1] - X[3, 1]) ** 2 +
                                (X[1, 2] - X[3, 2]) ** 2)
                            d43 = math.sqrt(
                                (X[4, 0] - X[3, 0]) ** 2 +
                                (X[4, 1] - X[3, 1]) ** 2 +
                                (X[4, 2] - X[3, 2]) ** 2)
                            dl = (d13 + d43) * 0.5
                            rr = (dl / lmax + acc / dacc +
                                  angle / dangle) / quali

                            inlist = path_inlist_1[h]
                            if inlist < POSI_K:
                                path_decis_1[h, inlist] = rr
                                path_linkdecis_1[h, inlist] = ftnr_mm
                                path_inlist_1[h] = inlist + 1

                    kk += 1

            # --- assess_new_position for X[5] in frame 3 ---
            targ_pos, cand_inds, quali = assess_new_position_fast(
                X[5], num_cams, ADD_PART_K,
                cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
                targ_x_3, targ_y_3, targ_tnr_3, num_targets_3,
                imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
                int(imx), int(imy), pix_x, pix_y, flatten_tol,
                TR_UNUSED_K, COORD_UNUSED_K)

            if quali >= 2:
                in_volume = 0
                pos_new, dl_pp = point_position_fast(
                    targ_pos, num_cams, cal_t)
                X[4, 0] = pos_new[0]
                X[4, 1] = pos_new[1]
                X[4, 2] = pos_new[2]

                if (X_lay_0 < X[4, 0] < X_lay_1 and
                        ymin < X[4, 1] < ymax and
                        Zmin_lay_0 < X[4, 2] < Zmax_lay_1):
                    in_volume = 1

                dp0 = X[3, 0] - X[4, 0]
                dp1 = X[3, 1] - X[4, 1]
                dp2 = X[3, 2] - X[4, 2]

                if (in_volume == 1 and
                        dvxmin < dp0 < dvxmax and
                        dvymin < dp1 < dvymax and
                        dvzmin < dp2 < dvzmax):
                    angle, acc = angle_acc_fast(
                        X[3, 0], X[3, 1], X[3, 2],
                        X[4, 0], X[4, 1], X[4, 2],
                        X[5, 0], X[5, 1], X[5, 2])

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = math.sqrt(
                            (X[1, 0] - X[3, 0]) ** 2 +
                            (X[1, 1] - X[3, 1]) ** 2 +
                            (X[1, 2] - X[3, 2]) ** 2)
                        d43 = math.sqrt(
                            (X[4, 0] - X[3, 0]) ** 2 +
                            (X[4, 1] - X[3, 1]) ** 2 +
                            (X[4, 2] - X[3, 2]) ** 2)
                        dl = (d13 + d43) * 0.5
                        rr = (dl / lmax + acc / dacc +
                              angle / dangle) / (quali + w_freq[mm])

                        inlist = path_inlist_1[h]
                        if inlist < POSI_K:
                            path_decis_1[h, inlist] = rr
                            path_linkdecis_1[h, inlist] = ftnr_mm
                            path_inlist_1[h] = inlist + 1

                        if add_flag:
                            np3 = num_parts_3[0]
                            path_x_3[np3, 0] = X[4, 0]
                            path_x_3[np3, 1] = X[4, 1]
                            path_x_3[np3, 2] = X[4, 2]
                            path_prev_3[np3] = PREV_NONE_K
                            path_next_3[np3] = NEXT_NONE_K
                            path_inlist_3[np3] = 0
                            path_prio_3[np3] = 4
                            path_finaldecis_3[np3] = 1000000.0
                            for ki in range(POSI_K):
                                path_decis_3[np3, ki] = 0.0
                                path_linkdecis_3[np3, ki] = PT_UNUSED
                            for ci in range(num_cams):
                                corres_p_3[np3, ci] = CORRES_NONE_K
                            corres_nr_3[np3] = np3
                            for ci in range(num_cams):
                                if cand_inds[ci] != PT_UNUSED:
                                    idx = cand_inds[ci]
                                    targ_tnr_3[ci][idx] = np3
                                    corres_p_3[np3, ci] = idx
                            num_parts_3[0] = np3 + 1
                            num_added += 1

                in_volume = 0
            quali = 0

            # --- fallback: direct link if no links and prev >= 0 ---
            if path_inlist_1[h] == 0 and prev_h >= 0:
                dp0 = X[3, 0] - X[1, 0]
                dp1 = X[3, 1] - X[1, 1]
                dp2 = X[3, 2] - X[1, 2]

                if (dvxmin < dp0 < dvxmax and
                        dvymin < dp1 < dvymax and
                        dvzmin < dp2 < dvzmax):
                    angle, acc = angle_acc_fast(
                        X[1, 0], X[1, 1], X[1, 2],
                        X[2, 0], X[2, 1], X[2, 2],
                        X[3, 0], X[3, 1], X[3, 2])

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        quali_f = w_freq[mm]
                        d13 = math.sqrt(
                            (X[1, 0] - X[3, 0]) ** 2 +
                            (X[1, 1] - X[3, 1]) ** 2 +
                            (X[1, 2] - X[3, 2]) ** 2)
                        d01 = math.sqrt(
                            (X[0, 0] - X[1, 0]) ** 2 +
                            (X[0, 1] - X[1, 1]) ** 2 +
                            (X[0, 2] - X[1, 2]) ** 2)
                        dl = (d13 + d01) * 0.5
                        rr = (dl / lmax + acc / dacc +
                              angle / dangle) / quali_f

                        inlist = path_inlist_1[h]
                        if inlist < POSI_K:
                            path_decis_1[h, inlist] = rr
                            path_linkdecis_1[h, inlist] = ftnr_mm
                            path_inlist_1[h] = inlist + 1

            mm += 1

        # --- add_particle to frame 2 if no links found ---
        if add_flag:
            if path_inlist_1[h] == 0 and prev_h >= 0:
                targ_pos2, cand_inds2, quali2 = assess_new_position_fast(
                    X[2], num_cams, ADD_PART_K,
                    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
                    targ_x_2, targ_y_2, targ_tnr_2, num_targets_2,
                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
                    int(imx), int(imy), pix_x, pix_y, flatten_tol,
                    TR_UNUSED_K, COORD_UNUSED_K)

                if quali2 >= 2:
                    in_volume = 0
                    pos_new2, dl_pp2 = point_position_fast(
                        targ_pos2, num_cams, cal_t)
                    X[3, 0] = pos_new2[0]
                    X[3, 1] = pos_new2[1]
                    X[3, 2] = pos_new2[2]

                    if (X_lay_0 < X[3, 0] < X_lay_1 and
                            ymin < X[3, 1] < ymax and
                            Zmin_lay_0 < X[3, 2] < Zmax_lay_1):
                        in_volume = 1

                    dp0 = X[2, 0] - X[3, 0]
                    dp1 = X[2, 1] - X[3, 1]
                    dp2 = X[2, 2] - X[3, 2]

                    if (in_volume == 1 and
                            dvxmin < dp0 < dvxmax and
                            dvymin < dp1 < dvymax and
                            dvzmin < dp2 < dvzmax):
                        angle, acc = angle_acc_fast(
                            X[1, 0], X[1, 1], X[1, 2],
                            X[2, 0], X[2, 1], X[2, 2],
                            X[3, 0], X[3, 1], X[3, 2])

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = math.sqrt(
                                (X[1, 0] - X[3, 0]) ** 2 +
                                (X[1, 1] - X[3, 1]) ** 2 +
                                (X[1, 2] - X[3, 2]) ** 2)
                            d01 = math.sqrt(
                                (X[0, 0] - X[1, 0]) ** 2 +
                                (X[0, 1] - X[1, 1]) ** 2 +
                                (X[0, 2] - X[1, 2]) ** 2)
                            dl = (d13 + d01) * 0.5
                            rr = (dl / lmax + acc / dacc +
                                  angle / dangle) / quali2

                            np2 = num_parts_2[0]
                            inlist = path_inlist_1[h]
                            if inlist < POSI_K:
                                path_decis_1[h, inlist] = rr
                                path_linkdecis_1[h, inlist] = np2
                                path_inlist_1[h] = inlist + 1

                            path_x_2[np2, 0] = X[3, 0]
                            path_x_2[np2, 1] = X[3, 1]
                            path_x_2[np2, 2] = X[3, 2]
                            path_prev_2[np2] = PREV_NONE_K
                            path_next_2[np2] = NEXT_NONE_K
                            path_inlist_2[np2] = 0
                            path_prio_2[np2] = 4
                            path_finaldecis_2[np2] = 1000000.0
                            for ki in range(POSI_K):
                                path_decis_2[np2, ki] = 0.0
                                path_linkdecis_2[np2, ki] = PT_UNUSED
                            for ci in range(num_cams):
                                corres_p_2[np2, ci] = CORRES_NONE_K
                            corres_nr_2[np2] = np2
                            for ci in range(num_cams):
                                if cand_inds2[ci] != PT_UNUSED:
                                    idx = cand_inds2[ci]
                                    targ_tnr_2[ci][idx] = np2
                                    corres_p_2[np2, ci] = idx
                            num_parts_2[0] = np2 + 1
                            num_added += 1

                    in_volume = 0

    # ========== LINK RESOLUTION ==========
    # Phase 1: Sort decis/linkdecis, set finaldecis and next
    for h in range(orig_parts_1):
        inlist = path_inlist_1[h]
        if inlist > 0:
            flag = True
            while flag:
                flag = False
                for i in range(inlist - 1):
                    if path_decis_1[h, i] > path_decis_1[h, i + 1]:
                        path_decis_1[h, i], path_decis_1[h, i + 1] = (
                            path_decis_1[h, i + 1], path_decis_1[h, i])
                        path_linkdecis_1[h, i], path_linkdecis_1[h, i + 1] = (
                            path_linkdecis_1[h, i + 1], path_linkdecis_1[h, i])
                        flag = True
            path_finaldecis_1[h] = path_decis_1[h, 0]
            path_next_1[h] = path_linkdecis_1[h, 0]

    # Phase 2: Resolve conflicts (original single-pass)
    for h in range(orig_parts_1):
        if path_inlist_1[h] > 0:
            next_h = path_next_1[h]
            if path_prev_2[next_h] == PREV_NONE_K:
                path_prev_2[next_h] = h
            else:
                prev_of_next = path_prev_2[next_h]
                if path_finaldecis_1[prev_of_next] > path_finaldecis_1[h]:
                    path_next_1[prev_of_next] = NEXT_NONE_K
                    path_prev_2[next_h] = h
                else:
                    path_next_1[h] = NEXT_NONE_K

    # Phase 3: Losers retry with fallback candidates (claim unclaimed only)
    for h in range(orig_parts_1):
        if path_inlist_1[h] > 1 and path_next_1[h] == NEXT_NONE_K:
            for ti in range(1, path_inlist_1[h]):
                cand = path_linkdecis_1[h, ti]
                if path_prev_2[cand] == PREV_NONE_K:
                    path_next_1[h] = cand
                    path_finaldecis_1[h] = path_decis_1[h, ti]
                    path_prev_2[cand] = h
                    break

    for h in range(orig_parts_1):
        if path_next_1[h] != NEXT_NONE_K:
            count1 += 1

    return count1, num_added


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def trackback_loop_fast(
    num_parts_1: cython.int,
    # Frame 0 (forward/next in time — read only)
    path_x_0: cython.double[:, :],
    # Frame 1 (current — read/write)
    path_x_1: cython.double[:, :], path_prev_1: cython.int[:], path_next_1: cython.int[:], path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:], path_decis_1: cython.double[:, :], path_linkdecis_1: cython.int[:, :],
    # Frame 2 (backward/prev in time — read/write)
    path_x_2: cython.double[:, :], path_prev_2: cython.int[:], path_next_2: cython.int[:], num_parts_2: cython.int[:],
    targ_x_2, targ_y_2, targ_tnr_2, num_targets_2: cython.int[:],
    corres_p_2: cython.int[:, :], corres_nr_2: cython.int[:],
    path_inlist_2: cython.int[:], path_prio_2: cython.int[:], path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, :], path_linkdecis_2: cython.int[:, :],
    # Frame 3 (further backward — read only, for extra angle check)
    path_x_3: cython.double[:, :], path_prev_3: cython.int[:],
    # Calibration
    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
    # Tracking params
    dvxmin: cython.double, dvxmax: cython.double, dvymin: cython.double, dvymax: cython.double, dvzmin: cython.double, dvzmax: cython.double,
    dacc: cython.double, dangle: cython.double, add_flag: cython.int, lmax: cython.double,
    # Volume bounds
    X_lay_0: cython.double, X_lay_1: cython.double, ymin: cython.double, ymax: cython.double, Zmin_lay_0: cython.double, Zmax_lay_1: cython.double,
    # Pixel params
    num_cams: cython.int,
    imx_half: cython.double, imy_half: cython.double, inv_pix_x: cython.double, inv_pix_y: cython.double,
    chfield: cython.int, imx: cython.double, imy: cython.double, pix_x: cython.double, pix_y: cython.double, flatten_tol: cython.double,
):
    """Backward tracking loop — compiled compiled.

    For each particle in buf[1] with next >= 0 and prev == -1,
    searches for candidates in buf[2] (backward in time).
    """
    count1: cython.int; num_added: cython.int
    h: cython.int; i: cython.int; j: cython.int; ki: cython.int; ci: cython.int
    next_h: cython.int; prev_h: cython.int; ftnr_i: cython.int
    inlist: cython.int; best_cand: cython.int; prev_of_cand: cython.int
    np2: cython.int; in_volume: cython.int; quali: cython.int; ti: cython.int
    px: cython.double; py: cython.double
    dp0: cython.double; dp1: cython.double; dp2: cython.double
    angle: cython.double; acc: cython.double; rr: cython.double
    d13: cython.double; d01: cython.double; dl: cython.double; idx: cython.int
    flag: cython.bint
    count1 = 0
    num_added = 0
    cpx = np.empty(num_cams, dtype=np.float64)
    cpy = np.empty(num_cams, dtype=np.float64)
    X = np.zeros((6, 3), dtype=np.float64)

    for h in range(num_parts_1):
        next_h = path_next_1[h]
        prev_h = path_prev_1[h]

        if (next_h < 0) or (prev_h != -1):
            continue

        path_inlist_1[h] = 0

        X[1, 0] = path_x_1[h, 0]
        X[1, 1] = path_x_1[h, 1]
        X[1, 2] = path_x_1[h, 2]

        X[0, 0] = path_x_0[next_h, 0]
        X[0, 1] = path_x_0[next_h, 1]
        X[0, 2] = path_x_0[next_h, 2]

        # Predict backward: 2*curr - next (mirror of forward prediction)
        X[2, 0] = 2.0 * X[1, 0] - X[0, 0]
        X[2, 1] = 2.0 * X[1, 1] - X[0, 1]
        X[2, 2] = 2.0 * X[1, 2] - X[0, 2]

        for j in range(num_cams):
            px, py = point_to_pixel_fast(
                X[2], cal_t[j], md_t[j], mo_t[j],
                mnr_t[j], mnz_t[j], mrw_t[j],
                imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)
            cpx[j] = px
            cpy[j] = py

        w_ftnr, w_freq, w_wc, w_nc = sorted_candidates_fast(
            X[2], cpx, cpy, num_cams, MAX_CANDS_K,
            cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
            targ_x_2, targ_y_2, targ_tnr_2, num_targets_2,
            dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax,
            imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
            imx, imy, TR_UNUSED_K)

        if w_nc > 0:
            i = 0
            while i < w_nc:
                ftnr_i = w_ftnr[i]
                X[3, 0] = path_x_2[ftnr_i, 0]
                X[3, 1] = path_x_2[ftnr_i, 1]
                X[3, 2] = path_x_2[ftnr_i, 2]

                dp0 = X[1, 0] - X[3, 0]
                dp1 = X[1, 1] - X[3, 1]
                dp2 = X[1, 2] - X[3, 2]

                if (dvxmin < dp0 < dvxmax and
                        dvymin < dp1 < dvymax and
                        dvzmin < dp2 < dvzmax):
                    angle, acc = angle_acc_fast(
                        X[1, 0], X[1, 1], X[1, 2],
                        X[2, 0], X[2, 1], X[2, 2],
                        X[3, 0], X[3, 1], X[3, 2])

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = math.sqrt(
                            (X[1, 0] - X[3, 0]) ** 2 +
                            (X[1, 1] - X[3, 1]) ** 2 +
                            (X[1, 2] - X[3, 2]) ** 2)
                        d01 = math.sqrt(
                            (X[0, 0] - X[1, 0]) ** 2 +
                            (X[0, 1] - X[1, 1]) ** 2 +
                            (X[0, 2] - X[1, 2]) ** 2)
                        dl = (d13 + d01) * 0.5
                        quali = w_freq[i]
                        rr = (dl / lmax + acc / dacc + angle / dangle) / quali

                        inlist = path_inlist_1[h]
                        if inlist < POSI_K:
                            path_decis_1[h, inlist] = rr
                            path_linkdecis_1[h, inlist] = ftnr_i
                            path_inlist_1[h] = inlist + 1

                i += 1

        if add_flag:
            if path_inlist_1[h] == 0:
                targ_pos, cand_inds, quali = assess_new_position_fast(
                    X[2], num_cams, ADD_PART_K,
                    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
                    targ_x_2, targ_y_2, targ_tnr_2, num_targets_2,
                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
                    int(imx), int(imy), pix_x, pix_y, flatten_tol,
                    TR_UNUSED_K, COORD_UNUSED_K)

                if quali >= 2:
                    in_volume = 0
                    pos_new, dl_pp = point_position_fast(
                        targ_pos, num_cams, cal_t)
                    X[3, 0] = pos_new[0]
                    X[3, 1] = pos_new[1]
                    X[3, 2] = pos_new[2]

                    if (X_lay_0 < X[3, 0] < X_lay_1 and
                            ymin < X[3, 1] < ymax and
                            Zmin_lay_0 < X[3, 2] < Zmax_lay_1):
                        in_volume = 1

                    dp0 = X[1, 0] - X[3, 0]
                    dp1 = X[1, 1] - X[3, 1]
                    dp2 = X[1, 2] - X[3, 2]

                    if (in_volume == 1 and
                            dvxmin < dp0 < dvxmax and
                            dvymin < dp1 < dvymax and
                            dvzmin < dp2 < dvzmax):
                        angle, acc = angle_acc_fast(
                            X[1, 0], X[1, 1], X[1, 2],
                            X[2, 0], X[2, 1], X[2, 2],
                            X[3, 0], X[3, 1], X[3, 2])

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = math.sqrt(
                                (X[1, 0] - X[3, 0]) ** 2 +
                                (X[1, 1] - X[3, 1]) ** 2 +
                                (X[1, 2] - X[3, 2]) ** 2)
                            d01 = math.sqrt(
                                (X[0, 0] - X[1, 0]) ** 2 +
                                (X[0, 1] - X[1, 1]) ** 2 +
                                (X[0, 2] - X[1, 2]) ** 2)
                            dl = (d13 + d01) * 0.5
                            rr = (dl / lmax + acc / dacc + angle / dangle) / quali

                            np2 = num_parts_2[0]
                            inlist = path_inlist_1[h]
                            if inlist < POSI_K:
                                path_decis_1[h, inlist] = rr
                                path_linkdecis_1[h, inlist] = np2
                                path_inlist_1[h] = inlist + 1

                            path_x_2[np2, 0] = X[3, 0]
                            path_x_2[np2, 1] = X[3, 1]
                            path_x_2[np2, 2] = X[3, 2]
                            path_prev_2[np2] = PREV_NONE_K
                            path_next_2[np2] = NEXT_NONE_K
                            path_inlist_2[np2] = 0
                            path_prio_2[np2] = 4
                            path_finaldecis_2[np2] = 1000000.0
                            for ki in range(POSI_K):
                                path_decis_2[np2, ki] = 0.0
                                path_linkdecis_2[np2, ki] = PT_UNUSED
                            for ci in range(num_cams):
                                corres_p_2[np2, ci] = CORRES_NONE_K
                            corres_nr_2[np2] = np2
                            for ci in range(num_cams):
                                if cand_inds[ci] != PT_UNUSED:
                                    idx = cand_inds[ci]
                                    targ_tnr_2[ci][idx] = np2
                                    corres_p_2[np2, ci] = idx
                            num_parts_2[0] = np2 + 1
                            num_added += 1

                    in_volume = 0

    # Sort candidates
    for h in range(num_parts_1):
        inlist = path_inlist_1[h]
        if inlist > 0:
            flag = True
            while flag:
                flag = False
                for i in range(inlist - 1):
                    if path_decis_1[h, i] > path_decis_1[h, i + 1]:
                        path_decis_1[h, i], path_decis_1[h, i + 1] = (
                            path_decis_1[h, i + 1], path_decis_1[h, i])
                        path_linkdecis_1[h, i], path_linkdecis_1[h, i + 1] = (
                            path_linkdecis_1[h, i + 1], path_linkdecis_1[h, i])
                        flag = True

    # Link resolution — trackback style
    for h in range(num_parts_1):
        if path_inlist_1[h] > 0:
            best_cand = path_linkdecis_1[h, 0]

            # Case 1: candidate has no links at all
            if (path_prev_2[best_cand] == PREV_NONE_K and
                    path_next_2[best_cand] == NEXT_NONE_K):
                path_finaldecis_1[h] = path_decis_1[h, 0]
                path_prev_1[h] = best_cand
                path_next_2[best_cand] = h
                num_added += 1

            # Case 2: candidate has a prev but no next — extra angle check
            elif (path_prev_2[best_cand] != PREV_NONE_K and
                    path_next_2[best_cand] == NEXT_NONE_K):
                X[0, 0] = path_x_0[path_next_1[h], 0]
                X[0, 1] = path_x_0[path_next_1[h], 1]
                X[0, 2] = path_x_0[path_next_1[h], 2]
                X[1, 0] = path_x_1[h, 0]
                X[1, 1] = path_x_1[h, 1]
                X[1, 2] = path_x_1[h, 2]
                X[3, 0] = path_x_2[best_cand, 0]
                X[3, 1] = path_x_2[best_cand, 1]
                X[3, 2] = path_x_2[best_cand, 2]

                prev_of_cand = path_prev_2[best_cand]
                X[4, 0] = path_x_3[prev_of_cand, 0]
                X[4, 1] = path_x_3[prev_of_cand, 1]
                X[4, 2] = path_x_3[prev_of_cand, 2]

                for j in range(3):
                    X[5, j] = 0.5 * (5.0 * X[3, j] - 4.0 * X[1, j] + X[0, j])

                angle, acc = angle_acc_fast(
                    X[3, 0], X[3, 1], X[3, 2],
                    X[4, 0], X[4, 1], X[4, 2],
                    X[5, 0], X[5, 1], X[5, 2])

                if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                    path_finaldecis_1[h] = path_decis_1[h, 0]
                    path_prev_1[h] = best_cand
                    path_next_2[best_cand] = h
                    num_added += 1

        if path_prev_1[h] != PREV_NONE_K:
            count1 += 1

    return count1, num_added


@cython.boundscheck(False)
@cython.wraparound(False)
def _find_closest_in_3d(path_x_2: cython.double[:, :], np2: cython.int,
                         pred_x: cython.double, pred_y: cython.double, pred_z: cython.double,
                         dx: cython.double, dy: cython.double, dz: cython.double,
                         max_cands: cython.int,
                         cand_inds: cython.int[:], cand_dists: cython.double[:]):
    """Find up to max_cands closest candidates by distance within a 3D box.

    Maintains a running top-N by distance, matching candsearch_in_pix logic.
    Writes into pre-allocated cand_inds/cand_dists arrays.
    Returns the number of candidates found.
    """
    s: cython.int; k: cython.int; slot: cython.int
    ddx: cython.double; ddy: cython.double; ddz: cython.double; d: cython.double
    n_found = 0
    for s in range(max_cands):
        cand_inds[s] = -1
        cand_dists[s] = 1e20

    for k in range(np2):
        ddx = path_x_2[k, 0] - pred_x
        ddy = path_x_2[k, 1] - pred_y
        ddz = path_x_2[k, 2] - pred_z
        if abs(ddx) < dx and abs(ddy) < dy and abs(ddz) < dz:
            d = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            for slot in range(max_cands):
                if d < cand_dists[slot]:
                    for s in range(max_cands - 1, slot, -1):
                        cand_inds[s] = cand_inds[s - 1]
                        cand_dists[s] = cand_dists[s - 1]
                    cand_inds[slot] = k
                    cand_dists[slot] = d
                    break

    for s in range(max_cands):
        if cand_inds[s] >= 0:
            n_found += 1
    return n_found


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def track3d_loop_fast(
    orig_parts: cython.int,
    # Frame 0 (prev) — read only
    path_x_0: cython.double[:, :], path_prev_0: cython.int[:], num_parts_0: cython.int,
    # Frame 1 (curr) — read/write
    path_x_1: cython.double[:, :], path_prev_1: cython.int[:], path_next_1: cython.int[:], num_parts_1: cython.int,
    # Frame 2 (next) — read/write
    path_x_2: cython.double[:, :], path_prev_2: cython.int[:], path_next_2: cython.int[:], num_parts_2: cython.int,
    # Tracking params
    dx: cython.double, dy: cython.double, dz: cython.double,
    max_cands: cython.int,
):
    """Full track3d loop (3 levels) — single compiled entry.

    Level 1: particles with previous links — predict from velocity.
    Level 2: no prev link — average velocity from neighbors.
    Level 3: no prev link, no neighbor info — use current position.

    Returns count1 (number of links established).
    """
    count1: cython.int; np2: cython.int
    i: cython.int; j: cython.int; ci: cython.int
    prev_idx: cython.int
    pred_x: cython.double; pred_y: cython.double; pred_z: cython.double
    n_cands: cython.int; n_decis: cython.int; k: cython.int
    d0: cython.double; d1: cython.double; d2: cython.double; acc: cython.double
    si: cython.int; sj: cython.int
    vel_x: cython.double; vel_y: cython.double; vel_z: cython.double
    nvel: cython.int; cx: cython.double; cy: cython.double; cz: cython.double
    pj: cython.int; inv_nvel: cython.double
    count1 = 0
    np2 = num_parts_2
    cand_inds = np.empty(max_cands, dtype=np.int32)
    cand_dists = np.empty(max_cands, dtype=np.float64)
    decis_vals = np.empty(max_cands, dtype=np.float64)
    decis_inds = np.empty(max_cands, dtype=np.int32)

    # ===== Level 1: Particles with previous links =====
    for i in range(orig_parts):
        if path_prev_1[i] < 0:
            continue
        prev_idx = path_prev_1[i]
        if prev_idx < 0 or prev_idx >= num_parts_0:
            continue

        pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
        pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
        pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]

        n_cands = _find_closest_in_3d(path_x_2, np2, pred_x, pred_y, pred_z,
                                       dx, dy, dz, max_cands, cand_inds, cand_dists)
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = path_x_1[i, 0] - 2.0 * path_x_2[k, 0] + path_x_0[prev_idx, 0]
            d1 = path_x_1[i, 1] - 2.0 * path_x_2[k, 1] + path_x_0[prev_idx, 1]
            d2 = path_x_1[i, 2] - 2.0 * path_x_2[k, 2] + path_x_0[prev_idx, 2]
            acc = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = decis_vals[sj], decis_vals[sj - 1]
                        decis_inds[sj - 1], decis_inds[sj] = decis_inds[sj], decis_inds[sj - 1]

        if path_prev_2[decis_inds[0]] < 0:
            path_next_1[i] = decis_inds[0]
            path_prev_2[decis_inds[0]] = i
            count1 += 1
        else:
            path_next_1[i] = -1

    # ===== Level 2: No previous link, neighbor velocity =====
    for i in range(orig_parts):
        if path_prev_1[i] >= 0 or path_next_1[i] >= 0:
            continue

        vel_x = 0.0; vel_y = 0.0; vel_z = 0.0
        nvel = 0
        cx = path_x_1[i, 0]; cy = path_x_1[i, 1]; cz = path_x_1[i, 2]

        for j in range(orig_parts):
            if j == i:
                continue
            if (abs(path_x_1[j, 0] - cx) < dx and
                    abs(path_x_1[j, 1] - cy) < dy and
                    abs(path_x_1[j, 2] - cz) < dz and
                    path_prev_1[j] >= 0):
                pj = path_prev_1[j]
                vel_x += path_x_1[j, 0] - path_x_0[pj, 0]
                vel_y += path_x_1[j, 1] - path_x_0[pj, 1]
                vel_z += path_x_1[j, 2] - path_x_0[pj, 2]
                nvel += 1

        if nvel == 0:
            continue

        inv_nvel = 1.0 / nvel
        pred_x = cx + vel_x * inv_nvel
        pred_y = cy + vel_y * inv_nvel
        pred_z = cz + vel_z * inv_nvel

        n_cands = _find_closest_in_3d(path_x_2, np2, pred_x, pred_y, pred_z,
                                       dx, dy, dz, max_cands, cand_inds, cand_dists)
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = cx - 2.0 * path_x_2[k, 0] + pred_x
            d1 = cy - 2.0 * path_x_2[k, 1] + pred_y
            d2 = cz - 2.0 * path_x_2[k, 2] + pred_z
            acc = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = decis_vals[sj], decis_vals[sj - 1]
                        decis_inds[sj - 1], decis_inds[sj] = decis_inds[sj], decis_inds[sj - 1]

        if path_prev_2[decis_inds[0]] < 0:
            path_next_1[i] = decis_inds[0]
            path_prev_2[decis_inds[0]] = i
            count1 += 1
        else:
            path_next_1[i] = -1

    # ===== Level 3: No previous link, no neighbors — static prediction =====
    for i in range(orig_parts):
        if path_prev_1[i] >= 0 or path_next_1[i] >= 0:
            continue

        pred_x = path_x_1[i, 0]
        pred_y = path_x_1[i, 1]
        pred_z = path_x_1[i, 2]

        n_cands = _find_closest_in_3d(path_x_2, np2, pred_x, pred_y, pred_z,
                                       dx, dy, dz, max_cands, cand_inds, cand_dists)
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = pred_x - 2.0 * path_x_2[k, 0] + pred_x
            d1 = pred_y - 2.0 * path_x_2[k, 1] + pred_y
            d2 = pred_z - 2.0 * path_x_2[k, 2] + pred_z
            acc = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = decis_vals[sj], decis_vals[sj - 1]
                        decis_inds[sj - 1], decis_inds[sj] = decis_inds[sj], decis_inds[sj - 1]

        if path_prev_2[decis_inds[0]] < 0:
            path_next_1[i] = decis_inds[0]
            path_prev_2[decis_inds[0]] = i
            count1 += 1
        else:
            path_next_1[i] = -1

    return count1


# ============================================================
# Batch kernels for standalone API acceleration
# ============================================================

@cython.ccall
def metric_to_pixel_fast(x_metric: cython.double, y_metric: cython.double,
                         imx: cython.int, imy: cython.int,
                         pix_x: cython.double, pix_y: cython.double,
                         chfield: cython.int):
    """Convert metric to pixel coordinates."""
    x_pixel: cython.double; y_pixel: cython.double
    x_pixel = x_metric / pix_x + imx * 0.5
    y_pixel = imy * 0.5 - y_metric / pix_y
    if chfield == 1:
        y_pixel = (y_pixel - 1.0) * 0.5
    elif chfield == 2:
        y_pixel = y_pixel * 0.5
    return x_pixel, y_pixel


@cython.boundscheck(False)
@cython.wraparound(False)
def _flat_image_coord_fast(pos: cython.double[:], cal: cython.double[:],
                           mmlut_data: cython.double[:], mmlut_origin: cython.double[:],
                           mmlut_nr: cython.int, mmlut_nz: cython.int, mmlut_rw: cython.double):
    """Project 3D to flat metric image coordinates.

    Returns (x, y) without distortion or pixel conversion.
    """
    pos0: cython.double; pos1: cython.double; pos2: cython.double
    ext_x0: cython.double; ext_y0: cython.double; ext_z0: cython.double
    dm00: cython.double; dm10: cython.double; dm20: cython.double
    dm01: cython.double; dm11: cython.double; dm21: cython.double
    dm02: cython.double; dm12: cython.double; dm22: cython.double
    int_cc: cython.double; gx: cython.double; gy: cython.double; gz: cython.double
    inv_dog: cython.double; mm_n1: cython.double; mm_n2_0: cython.double; mm_n3: cython.double; mm_d0: cython.double
    dot_cam: cython.double; dist_o_glas: cython.double; dist_cam_glas: cython.double
    dot_pos: cython.double; dist_point_glas: cython.double
    s_cam: cython.double; cc_x: cython.double; cc_y: cython.double; cc_z: cython.double
    s_pt: cython.double; cp_x: cython.double; cp_y: cython.double; cp_z: cython.double
    ext_t_z0: cython.double; s_d: cython.double
    ag_x: cython.double; ag_y: cython.double; ag_z: cython.double
    tmp_x: cython.double; tmp_y: cython.double; tmp_z: cython.double
    pos_t_0: cython.double; pos_t_2: cython.double
    radial_shift: cython.double; has_mmlut: cython.bint
    tx: cython.double; ty: cython.double; tz: cython.double
    sz: cython.double; iz: cython.int; R: cython.double; sr: cython.double; ir: cython.int
    v0: cython.int; v3: cython.int; mmf: cython.double; X_t: cython.double
    s_z: cython.double; bx: cython.double; by: cython.double; bz: cython.double; s_x: cython.double
    dx: cython.double; dy: cython.double; dz: cython.double; deno: cython.double
    x: cython.double; y: cython.double
    pos0 = pos[0]; pos1 = pos[1]; pos2 = pos[2]

    ext_x0 = cal[0]; ext_y0 = cal[1]; ext_z0 = cal[2]
    dm00 = cal[3]; dm10 = cal[4]; dm20 = cal[5]
    dm01 = cal[6]; dm11 = cal[7]; dm21 = cal[8]
    dm02 = cal[9]; dm12 = cal[10]; dm22 = cal[11]
    int_cc = cal[12]
    gx = cal[15]; gy = cal[16]; gz = cal[17]
    inv_dog = cal[19]
    mm_n1 = cal[20]; mm_n2_0 = cal[21]; mm_n3 = cal[22]; mm_d0 = cal[23]

    dot_cam = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_o_glas = cal[18]
    dist_cam_glas = dot_cam * inv_dog - dist_o_glas - mm_d0

    dot_pos = pos0 * gx + pos1 * gy + pos2 * gz
    dist_point_glas = dot_pos * inv_dog - dist_o_glas

    s_cam = dist_cam_glas * inv_dog
    cc_x = ext_x0 - gx * s_cam
    cc_y = ext_y0 - gy * s_cam
    cc_z = ext_z0 - gz * s_cam

    s_pt = dist_point_glas * inv_dog
    cp_x = pos0 - gx * s_pt
    cp_y = pos1 - gy * s_pt
    cp_z = pos2 - gz * s_pt

    ext_t_z0 = dist_cam_glas + mm_d0

    s_d = mm_d0 * inv_dog
    ag_x = cc_x - gx * s_d
    ag_y = cc_y - gy * s_d
    ag_z = cc_z - gz * s_d
    tmp_x = cp_x - ag_x
    tmp_y = cp_y - ag_y
    tmp_z = cp_z - ag_z

    pos_t_0 = math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
    pos_t_2 = dist_point_glas

    radial_shift = 1.0
    has_mmlut = len(mmlut_data) > 0
    if has_mmlut:
        tx = pos_t_0 - mmlut_origin[0]
        ty = -mmlut_origin[1]
        tz = pos_t_2 - mmlut_origin[2]
        sz = tz / mmlut_rw
        iz = int(sz)
        sz -= iz
        R = math.sqrt(tx * tx + ty * ty)
        sr = R / mmlut_rw
        ir = int(sr)
        sr -= ir
        if ir <= mmlut_nr and iz >= 0 and iz <= mmlut_nz:
            v0 = ir * mmlut_nz + iz
            v3 = v0 + mmlut_nz + 1
            if v0 >= 0 and v3 <= mmlut_nr * mmlut_nz:
                mmf = (mmlut_data[v0] * (1.0 - sr) * (1.0 - sz)
                       + mmlut_data[v0 + 1] * (1.0 - sr) * sz
                       + mmlut_data[v0 + mmlut_nz] * sr * (1.0 - sz)
                       + mmlut_data[v3] * sr * sz)
                if mmf > 0.0:
                    radial_shift = mmf
    if radial_shift == 1.0:
        radial_shift = _multimed_r_nlay_1layer(
            pos_t_0, 0.0, pos_t_2, 0.0, 0.0, ext_t_z0,
            mm_n1, mm_n2_0, mm_n3, mm_d0,
        )
    X_t = pos_t_0 * radial_shift

    s_z = -pos_t_2 * inv_dog
    bx = ag_x - gx * s_z
    by = ag_y - gy * s_z
    bz = ag_z - gz * s_z
    if pos_t_0 > 0.0:
        s_x = -X_t / pos_t_0
        bx -= tmp_x * s_x
        by -= tmp_y * s_x
        bz -= tmp_z * s_x

    dx = bx - ext_x0
    dy = by - ext_y0
    dz = bz - ext_z0
    deno = dm02 * dx + dm12 * dy + dm22 * dz
    x = -int_cc * (dm00 * dx + dm10 * dy + dm20 * dz) / deno
    y = -int_cc * (dm01 * dx + dm11 * dy + dm21 * dz) / deno

    return x, y


@cython.boundscheck(False)
@cython.wraparound(False)
def _img_coord_fast(pos: cython.double[:], cal: cython.double[:],
                    mmlut_data: cython.double[:], mmlut_origin: cython.double[:],
                    mmlut_nr: cython.int, mmlut_nz: cython.int, mmlut_rw: cython.double):
    """Project 3D to distorted metric image coordinates."""
    xh: cython.double; yh: cython.double
    k1: cython.double; k2: cython.double; k3: cython.double
    p1: cython.double; p2: cython.double; scx: cython.double; she: cython.double
    x: cython.double; y: cython.double; r: cython.double; r2: cython.double; r4: cython.double
    radial_factor: cython.double; xd: cython.double; yd: cython.double
    sin_she: cython.double; cos_she: cython.double
    x_dist: cython.double; y_dist: cython.double
    x, y = _flat_image_coord_fast(pos, cal, mmlut_data, mmlut_origin,
                                  mmlut_nr, mmlut_nz, mmlut_rw)

    xh = cal[13]; yh = cal[14]
    k1 = cal[24]; k2 = cal[25]; k3 = cal[26]
    p1 = cal[27]; p2 = cal[28]; scx = cal[29]; she = cal[30]

    x += xh
    y += yh
    r = math.sqrt(x * x + y * y)
    if r < 1e-10:
        return 0.0, 0.0

    r2 = r * r
    r4 = r2 * r2
    radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
    xd = x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
    yd = y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
    sin_she = math.sin(she)
    cos_she = math.cos(she)
    x_dist = scx * (xd - sin_she * yd)
    y_dist = scx * cos_she * yd

    return x_dist, y_dist


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def img_coord_batch_fast(positions: cython.double[:, :], cal: cython.double[:],
                         mmlut_data: cython.double[:], mmlut_origin: cython.double[:],
                         mmlut_nr: cython.int, mmlut_nz: cython.int, mmlut_rw: cython.double):
    """Project N 3D positions to distorted metric coords."""
    n: cython.Py_ssize_t; i: cython.Py_ssize_t
    n = positions.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = _img_coord_fast(
            positions[i], cal, mmlut_data, mmlut_origin,
            mmlut_nr, mmlut_nz, mmlut_rw)
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def flat_image_coord_batch_fast(positions: cython.double[:, :], cal: cython.double[:],
                                mmlut_data: cython.double[:], mmlut_origin: cython.double[:],
                                mmlut_nr: cython.int, mmlut_nz: cython.int, mmlut_rw: cython.double):
    """Project N 3D positions to flat metric coords."""
    n: cython.Py_ssize_t; i: cython.Py_ssize_t
    n = positions.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = _flat_image_coord_fast(
            positions[i], cal, mmlut_data, mmlut_origin,
            mmlut_nr, mmlut_nz, mmlut_rw)
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def ray_tracing_batch_fast(xy: cython.double[:, :], cal: cython.double[:]):
    """Trace N rays through multi-media interface.

    Args:
        xy: (N, 2) float64 — metric image coordinates.
        cal: (31,) float64 — packed calibration.

    Returns:
        (positions, directions) each (N, 3) float64.
    """
    n: cython.Py_ssize_t; i: cython.Py_ssize_t
    Xx: cython.double; Xy: cython.double; Xz: cython.double
    ox: cython.double; oy: cython.double; oz: cython.double
    n = xy.shape[0]
    positions = np.empty((n, 3), dtype=np.float64)
    directions = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        Xx, Xy, Xz, ox, oy, oz = _ray_tracing_fast(xy[i, 0], xy[i, 1], cal)
        positions[i, 0] = Xx; positions[i, 1] = Xy; positions[i, 2] = Xz
        directions[i, 0] = ox; directions[i, 1] = oy; directions[i, 2] = oz
    return positions, directions


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def point_position_batch_fast(all_targets: cython.double[:, :, :], num_pts: cython.int,
                              num_cams: cython.int, cal_arrays):
    """Triangulate M targets from N cameras.

    Args:
        all_targets: (M, num_cams, 2) float64.
        num_pts: M.
        num_cams: N.
        cal_arrays: tuple of (31,) float64 arrays.

    Returns:
        (positions, distances) — (M, 3) and (M,) float64.
    """
    i: cython.Py_ssize_t
    dist: cython.double
    positions = np.empty((num_pts, 3), dtype=np.float64)
    distances = np.empty(num_pts, dtype=np.float64)
    for i in range(num_pts):
        pos, dist = point_position_fast(all_targets[i], num_cams, cal_arrays)
        positions[i, 0] = pos[0]
        positions[i, 1] = pos[1]
        positions[i, 2] = pos[2]
        distances[i] = dist
    return positions, distances


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def pixel_to_metric_batch_fast(xy: cython.double[:, :], imx: cython.int, imy: cython.int,
                               pix_x: cython.double, pix_y: cython.double, chfield: cython.int):
    """Convert N pixel coordinates to metric."""
    n: cython.Py_ssize_t; i: cython.Py_ssize_t
    n = xy.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = pixel_to_metric_fast(
            xy[i, 0], xy[i, 1], imx, imy, pix_x, pix_y, chfield)
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def metric_to_pixel_batch_fast(xy: cython.double[:, :], imx: cython.int, imy: cython.int,
                               pix_x: cython.double, pix_y: cython.double, chfield: cython.int):
    """Convert N metric coordinates to pixel."""
    n: cython.Py_ssize_t; i: cython.Py_ssize_t
    n = xy.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = metric_to_pixel_fast(
            xy[i, 0], xy[i, 1], imx, imy, pix_x, pix_y, chfield)
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def targ_rec_fast(img, img0, gvthres: cython.int, discont: cython.int,
                 nnmin: cython.int, nnmax: cython.int,
                 nxmin: cython.int, nxmax: cython.int, nymin: cython.int, nymax: cython.int,
                 sumg_min: cython.int,
                 xmin: cython.int, ymin: cython.int, xmax: cython.int, ymax: cython.int,
                 max_targets: cython.int):
    """BFS flood-fill target recognition.

    Args:
        img: original uint8 image (imy, imx) — read-only for neighbor checks.
        img0: writable copy (imy, imx) — pixels zeroed as they are consumed.
        gvthres: grey value threshold.
        discont: maximum grey value discontinuity for BFS growth.
        nnmin, nnmax: min/max pixel count per target.
        nxmin, nxmax, nymin, nymax: min/max bounding box per target.
        sumg_min: minimum sum of grey values.
        xmin, ymin, xmax, ymax: search window (inclusive on ymin/xmin side).
        max_targets: pre-allocated output capacity.

    Returns:
        (n_targets, out_x, out_y, out_n, out_nx, out_ny, out_sumg)
        First n_targets elements of each output array are valid.
    """
    n_targets: cython.int; queue_size: cython.int
    i: cython.int; j: cython.int; d: cython.int
    xa: cython.int; xb: cython.int; ya: cython.int; yb: cython.int
    x_weighted: cython.double; y_weighted: cython.double
    head: cython.int; tail: cython.int
    wj: cython.int; wi: cython.int
    xn4: cython.int; yn4: cython.int
    nx: cython.int; ny: cython.int
    out_x = np.empty(max_targets, dtype=np.float64)
    out_y = np.empty(max_targets, dtype=np.float64)
    out_n = np.empty(max_targets, dtype=np.int64)
    out_nx = np.empty(max_targets, dtype=np.int64)
    out_ny = np.empty(max_targets, dtype=np.int64)
    out_sumg = np.empty(max_targets, dtype=np.int64)

    # BFS circular queue — bounded by nnmax (we stop adding beyond that).
    queue_size = nnmax + 16
    qx = np.empty(queue_size, dtype=np.int32)
    qy = np.empty(queue_size, dtype=np.int32)

    # Offsets for 4-connectivity
    dx4 = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy4 = np.array([0, 0, -1, 1], dtype=np.int32)

    n_targets = 0

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            gv = np.int64(img[i, j])
            if gv <= gvthres:
                continue

            # 8-neighbor local maximum check
            if not (gv >= np.int64(img[i, j - 1]) and
                    gv >= np.int64(img[i, j + 1]) and
                    gv >= np.int64(img[i - 1, j]) and
                    gv >= np.int64(img[i + 1, j]) and
                    gv >= np.int64(img[i - 1, j - 1]) and
                    gv >= np.int64(img[i + 1, j - 1]) and
                    gv >= np.int64(img[i - 1, j + 1]) and
                    gv >= np.int64(img[i + 1, j + 1])):
                continue

            if np.int64(img0[i, j]) <= gvthres:
                continue

            # Start BFS from this peak
            sumg = gv
            img0[i, j] = 0
            xa = j; xb = j
            ya = i; yb = i
            x_weighted = float(j) * float(gv - gvthres)
            y_weighted = float(i) * float(gv - gvthres)
            numpix = np.int64(1)

            head = np.int32(0)
            tail = np.int32(1)
            qx[0] = np.int32(j)
            qy[0] = np.int32(i)

            while head != tail:
                wj = np.int32(qx[head])
                wi = np.int32(qy[head])
                head += 1
                if head >= queue_size:
                    head = 0
                gvref = np.int64(img[wi, wj])

                for d in range(4):
                    xn4 = np.int32(wj + dx4[d])
                    yn4 = np.int32(wi + dy4[d])

                    if xn4 < xmin or xn4 >= xmax or yn4 < ymin or yn4 >= ymax:
                        continue

                    gv4 = np.int64(img0[yn4, xn4])
                    if (gv4 > gvthres and
                            gv4 <= gvref + discont and
                            gvref + discont >= np.int64(img[yn4 - 1, xn4]) and
                            gvref + discont >= np.int64(img[yn4 + 1, xn4]) and
                            gvref + discont >= np.int64(img[yn4, xn4 - 1]) and
                            gvref + discont >= np.int64(img[yn4, xn4 + 1])):
                        sumg += gv4
                        img0[yn4, xn4] = 0
                        if xn4 < xa:
                            xa = xn4
                        if xn4 > xb:
                            xb = xn4
                        if yn4 < ya:
                            ya = yn4
                        if yn4 > yb:
                            yb = yn4
                        x_weighted += float(xn4) * float(gv4 - gvthres)
                        y_weighted += float(yn4) * float(gv4 - gvthres)
                        numpix += 1
                        if numpix <= nnmax:
                            qx[tail] = xn4
                            qy[tail] = yn4
                            tail += 1
                            if tail >= queue_size:
                                tail = 0

            # Skip particles whose bounding box would extend outside the
            # search area (mirrors the original C border check convention).
            if xa == xmin - 1 or ya == ymin - 1 or xb == xmax + 1 or yb == ymax + 1:
                continue

            nx = xb - xa + 1
            ny = yb - ya + 1

            if (nnmin <= numpix <= nnmax and
                    nxmin <= nx <= nxmax and
                    nymin <= ny <= nymax and
                    sumg > sumg_min):
                if n_targets >= max_targets:
                    break

                sumg_adj = sumg - numpix * gvthres
                if sumg_adj > 0:
                    out_x[n_targets] = x_weighted / float(sumg_adj) + 0.5
                    out_y[n_targets] = y_weighted / float(sumg_adj) + 0.5
                else:
                    out_x[n_targets] = float(j) + 0.5
                    out_y[n_targets] = float(i) + 0.5
                out_n[n_targets] = numpix
                out_nx[n_targets] = nx
                out_ny[n_targets] = ny
                out_sumg[n_targets] = sumg
                n_targets += 1

    return n_targets, out_x, out_y, out_n, out_nx, out_ny, out_sumg


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def init_mmlut_data_fast(nr: cython.int, nz: cython.int, rw: cython.double,
                        cal_t_x0: cython.double, cal_t_y0: cython.double, cal_t_z0: cython.double,
                        Zmin_t: cython.double, mm_n1: cython.double, mm_n2_0: cython.double,
                        mm_n3: cython.double, mm_d0: cython.double):
    """Fill mmlut data grid in parallel — single-layer multimedia.

    Computes the radial shift factor for every (R, Z) grid point using
    _multimed_r_nlay_1layer. Rows are R-indices, columns are Z-indices.

    Args:
        nr: number of R grid points.
        nz: number of Z grid points.
        rw: grid cell size (mm).
        cal_t_x0, cal_t_y0, cal_t_z0: transformed camera center.
        Zmin_t: minimum Z of translated grid.
        mm_n1, mm_n2_0, mm_n3, mm_d0: single-layer multimedia parameters.

    Returns:
        data: (nr * nz,) float64 array of radial shift factors.
    """
    i: cython.int; j: cython.int
    R: cython.double; Z: cython.double
    data = np.empty(nr * nz, dtype=np.float64)
    for i in range(nr):
        R = i * rw + cal_t_x0
        for j in range(nz):
            Z = Zmin_t + j * rw
            data[i * nz + j] = _multimed_r_nlay_1layer(
                R, cal_t_y0, Z,
                cal_t_x0, cal_t_y0, cal_t_z0,
                mm_n1, mm_n2_0, mm_n3, mm_d0,
            )
    return data
