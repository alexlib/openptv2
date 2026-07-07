"""Compiled kernels for the tracking hot path.

Auto-generated split from track_kernels.py.
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt,
        sin as c_sin,
        cos as c_cos,
        tan as c_tan,
        asin as c_asin,
        acos as c_acos,
        atan as c_atan,
    )
else:
    from math import (
        sqrt as c_sqrt,
        sin as c_sin,
        cos as c_cos,
        tan as c_tan,
        asin as c_asin,
        acos as c_acos,
        atan as c_atan,
    )

_M_PI: cython.double = 3.141592653589793


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
@cython.nogil
def _multimed_r_nlay_1layer(
    pos_x: cython.double,
    pos_y: cython.double,
    pos_z: cython.double,
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
) -> cython.double:
    """Single-layer iterative radial shift."""
    zout: cython.double
    dx: cython.double
    dy: cython.double
    r: cython.double
    rq: cython.double
    it: cython.int
    denom: cython.double
    beta1: cython.double
    sin_beta1: cython.double
    arg: cython.double
    beta2_0: cython.double
    arg3: cython.double
    beta3: cython.double
    rbeta: cython.double
    rdiff: cython.double
    if mm_n1 == 1.0 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
        return 1.0

    zout = pos_z
    dx = pos_x - ext_x0
    dy = pos_y - ext_y0
    r = c_sqrt(dx * dx + dy * dy)
    rq = r

    for it in range(40):
        denom = ext_z0 - pos_z
        if denom == 0.0:
            return 1.0
        beta1 = c_atan(rq / denom)
        sin_beta1 = c_sin(beta1)

        arg = sin_beta1 * mm_n1 / mm_n2_0
        if arg > 1.0:
            arg = 1.0
        elif arg < -1.0:
            arg = -1.0
        beta2_0 = c_asin(arg)

        arg3 = sin_beta1 * mm_n1 / mm_n3
        if arg3 > 1.0:
            arg3 = 1.0
        elif arg3 < -1.0:
            arg3 = -1.0
        beta3 = c_asin(arg3)

        rbeta = (
            (ext_z0 - mm_d0) * c_tan(beta1)
            + mm_d0 * c_tan(beta2_0)
            - zout * c_tan(beta3)
        )

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
def point_to_pixel_fast(
    pos: cython.double[:],
    cal: cython.double[:],
    mmlut_data: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
    has_mmlut: cython.int,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
):
    """Project 3D position to pixel coordinates.

    Args:
        pos: (3,) float64 array — 3D position.
        cal: (31,) float64 array — packed calibration.
        mmlut_data: float64 array — LUT data (length 0 if no LUT).
        mmlut_origin: (3,) float64 array — LUT origin.
        mmlut_nr, mmlut_nz: int — LUT grid dimensions.
        mmlut_rw: float — LUT grid spacing.
        has_mmlut: int — 1 if mmlut_data is non-empty, 0 otherwise.
        imx_half, imy_half: float — half image dimensions.
        inv_pix_x, inv_pix_y: float — 1/pixel_size.
        chfield: int — interlace mode.

    Returns:
        (x_pixel, y_pixel) tuple.
    """
    pos0: cython.double
    pos1: cython.double
    pos2: cython.double
    ext_x0: cython.double
    ext_y0: cython.double
    ext_z0: cython.double
    dm00: cython.double
    dm10: cython.double
    dm20: cython.double
    dm01: cython.double
    dm11: cython.double
    dm21: cython.double
    dm02: cython.double
    dm12: cython.double
    dm22: cython.double
    int_cc: cython.double
    xh: cython.double
    yh: cython.double
    gx: cython.double
    gy: cython.double
    gz: cython.double
    inv_dog: cython.double
    mm_n1: cython.double
    mm_n2_0: cython.double
    mm_n3: cython.double
    mm_d0: cython.double
    k1: cython.double
    k2: cython.double
    k3: cython.double
    p1: cython.double
    p2: cython.double
    scx: cython.double
    she: cython.double
    dot_cam: cython.double
    dist_o_glas: cython.double
    dist_cam_glas: cython.double
    dot_pos: cython.double
    dist_point_glas: cython.double
    s_cam: cython.double
    cc_x: cython.double
    cc_y: cython.double
    cc_z: cython.double
    s_pt: cython.double
    cp_x: cython.double
    cp_y: cython.double
    cp_z: cython.double
    ext_t_z0: cython.double
    s_d: cython.double
    ag_x: cython.double
    ag_y: cython.double
    ag_z: cython.double
    tmp_x: cython.double
    tmp_y: cython.double
    tmp_z: cython.double
    pos_t_0: cython.double
    pos_t_2: cython.double
    radial_shift: cython.double
    tx: cython.double
    ty: cython.double
    tz: cython.double
    sz: cython.double
    iz: cython.int
    R: cython.double
    sr: cython.double
    ir: cython.int
    v0: cython.int
    v3: cython.int
    mmf: cython.double
    X_t: cython.double
    s_z: cython.double
    bx: cython.double
    by: cython.double
    bz: cython.double
    s_x: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    deno: cython.double
    x: cython.double
    y: cython.double
    r: cython.double
    r2: cython.double
    r4: cython.double
    radial_factor: cython.double
    xd: cython.double
    yd: cython.double
    sin_she: cython.double
    cos_she: cython.double
    x_dist: cython.double
    y_dist: cython.double
    x_pixel: cython.double
    y_pixel: cython.double
    pos0 = pos[0]
    pos1 = pos[1]
    pos2 = pos[2]

    ext_x0 = cal[0]
    ext_y0 = cal[1]
    ext_z0 = cal[2]
    dm00 = cal[3]
    dm10 = cal[4]
    dm20 = cal[5]
    dm01 = cal[6]
    dm11 = cal[7]
    dm21 = cal[8]
    dm02 = cal[9]
    dm12 = cal[10]
    dm22 = cal[11]
    int_cc = cal[12]
    xh = cal[13]
    yh = cal[14]
    gx = cal[15]
    gy = cal[16]
    gz = cal[17]
    inv_dog = cal[19]
    mm_n1 = cal[20]
    mm_n2_0 = cal[21]
    mm_n3 = cal[22]
    mm_d0 = cal[23]
    k1 = cal[24]
    k2 = cal[25]
    k3 = cal[26]
    p1 = cal[27]
    p2 = cal[28]
    scx = cal[29]
    she = cal[30]

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

    pos_t_0 = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
    pos_t_2 = dist_point_glas

    # mmlut lookup + multimed_nlay
    radial_shift = 1.0
    if has_mmlut:
        tx = pos_t_0 - mmlut_origin[0]
        ty = -mmlut_origin[1]
        tz = pos_t_2 - mmlut_origin[2]
        sz = tz / mmlut_rw
        iz = int(sz)
        sz -= iz
        R = c_sqrt(tx * tx + ty * ty)
        sr = R / mmlut_rw
        ir = int(sr)
        sr -= ir
        if ir <= mmlut_nr and iz >= 0 and iz <= mmlut_nz:
            v0 = ir * mmlut_nz + iz
            v3 = v0 + mmlut_nz + 1
            if v0 >= 0 and v3 <= mmlut_nr * mmlut_nz:
                mmf = (
                    mmlut_data[v0] * (1.0 - sr) * (1.0 - sz)
                    + mmlut_data[v0 + 1] * (1.0 - sr) * sz
                    + mmlut_data[v0 + mmlut_nz] * sr * (1.0 - sz)
                    + mmlut_data[v3] * sr * sz
                )
                if mmf > 0.0:
                    radial_shift = mmf
    if radial_shift == 1.0:
        radial_shift = _multimed_r_nlay_1layer(
            pos_t_0,
            0.0,
            pos_t_2,
            0.0,
            0.0,
            ext_t_z0,
            mm_n1,
            mm_n2_0,
            mm_n3,
            mm_d0,
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
    r = c_sqrt(x * x + y * y)
    if r < 1e-10:
        x_dist = 0.0
        y_dist = 0.0
    else:
        r2 = r * r
        r4 = r2 * r2
        radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
        xd = x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
        yd = y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
        sin_she = c_sin(she)
        cos_she = c_cos(she)
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


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.nogil
def _point_to_pixel_out(
    pos: cython.double[:],
    cal: cython.double[:],
    mmlut_data: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
    has_mmlut: cython.int,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    out: cython.double[:],
) -> cython.int:
    """Write pixel coordinates to out[0], out[1] — no tuple creation."""
    pos0: cython.double
    pos1: cython.double
    pos2: cython.double
    ext_x0: cython.double
    ext_y0: cython.double
    ext_z0: cython.double
    dm00: cython.double
    dm10: cython.double
    dm20: cython.double
    dm01: cython.double
    dm11: cython.double
    dm21: cython.double
    dm02: cython.double
    dm12: cython.double
    dm22: cython.double
    int_cc: cython.double
    xh: cython.double
    yh: cython.double
    gx: cython.double
    gy: cython.double
    gz: cython.double
    inv_dog: cython.double
    mm_n1: cython.double
    mm_n2_0: cython.double
    mm_n3: cython.double
    mm_d0: cython.double
    k1: cython.double
    k2: cython.double
    k3: cython.double
    p1: cython.double
    p2: cython.double
    scx: cython.double
    she: cython.double
    dot_cam: cython.double
    dist_o_glas: cython.double
    dist_cam_glas: cython.double
    dot_pos: cython.double
    dist_point_glas: cython.double
    s_cam: cython.double
    cc_x: cython.double
    cc_y: cython.double
    cc_z: cython.double
    s_pt: cython.double
    cp_x: cython.double
    cp_y: cython.double
    cp_z: cython.double
    ext_t_z0: cython.double
    s_d: cython.double
    ag_x: cython.double
    ag_y: cython.double
    ag_z: cython.double
    tmp_x: cython.double
    tmp_y: cython.double
    tmp_z: cython.double
    pos_t_0: cython.double
    pos_t_2: cython.double
    radial_shift: cython.double
    tx: cython.double
    ty: cython.double
    tz: cython.double
    sz: cython.double
    iz: cython.int
    R: cython.double
    sr: cython.double
    ir: cython.int
    v0: cython.int
    v3: cython.int
    mmf: cython.double
    X_t: cython.double
    s_z: cython.double
    bx: cython.double
    by: cython.double
    bz: cython.double
    s_x: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    deno: cython.double
    x: cython.double
    y: cython.double
    r: cython.double
    r2: cython.double
    r4: cython.double
    radial_factor: cython.double
    xd: cython.double
    yd: cython.double
    sin_she: cython.double
    cos_she: cython.double
    x_dist: cython.double
    y_dist: cython.double
    x_pixel: cython.double
    y_pixel: cython.double
    pos0 = pos[0]
    pos1 = pos[1]
    pos2 = pos[2]

    ext_x0 = cal[0]
    ext_y0 = cal[1]
    ext_z0 = cal[2]
    dm00 = cal[3]
    dm10 = cal[4]
    dm20 = cal[5]
    dm01 = cal[6]
    dm11 = cal[7]
    dm21 = cal[8]
    dm02 = cal[9]
    dm12 = cal[10]
    dm22 = cal[11]
    int_cc = cal[12]
    xh = cal[13]
    yh = cal[14]
    gx = cal[15]
    gy = cal[16]
    gz = cal[17]
    inv_dog = cal[19]
    mm_n1 = cal[20]
    mm_n2_0 = cal[21]
    mm_n3 = cal[22]
    mm_d0 = cal[23]
    k1 = cal[24]
    k2 = cal[25]
    k3 = cal[26]
    p1 = cal[27]
    p2 = cal[28]
    scx = cal[29]
    she = cal[30]

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

    pos_t_0 = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
    pos_t_2 = dist_point_glas

    # mmlut lookup + multimed_nlay
    radial_shift = 1.0
    if has_mmlut:
        tx = pos_t_0 - mmlut_origin[0]
        ty = -mmlut_origin[1]
        tz = pos_t_2 - mmlut_origin[2]
        sz = tz / mmlut_rw
        iz = int(sz)
        sz -= iz
        R = c_sqrt(tx * tx + ty * ty)
        sr = R / mmlut_rw
        ir = int(sr)
        sr -= ir
        if ir <= mmlut_nr and iz >= 0 and iz <= mmlut_nz:
            v0 = ir * mmlut_nz + iz
            v3 = v0 + mmlut_nz + 1
            if v0 >= 0 and v3 <= mmlut_nr * mmlut_nz:
                mmf = (
                    mmlut_data[v0] * (1.0 - sr) * (1.0 - sz)
                    + mmlut_data[v0 + 1] * (1.0 - sr) * sz
                    + mmlut_data[v0 + mmlut_nz] * sr * (1.0 - sz)
                    + mmlut_data[v3] * sr * sz
                )
                if mmf > 0.0:
                    radial_shift = mmf
    if radial_shift == 1.0:
        radial_shift = _multimed_r_nlay_1layer(
            pos_t_0,
            0.0,
            pos_t_2,
            0.0,
            0.0,
            ext_t_z0,
            mm_n1,
            mm_n2_0,
            mm_n3,
            mm_d0,
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
    r = c_sqrt(x * x + y * y)
    if r < 1e-10:
        x_dist = 0.0
        y_dist = 0.0
    else:
        r2 = r * r
        r4 = r2 * r2
        radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
        xd = x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
        yd = y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
        sin_she = c_sin(she)
        cos_she = c_cos(she)
        x_dist = scx * (xd - sin_she * yd)
        y_dist = scx * cos_she * yd

    # metric_to_pixel
    x_pixel = x_dist * inv_pix_x + imx_half
    y_pixel = imy_half - y_dist * inv_pix_y
    if chfield == 1:
        y_pixel = (y_pixel - 1.0) * 0.5
    elif chfield == 2:
        y_pixel = y_pixel * 0.5
    out[0] = x_pixel
    out[1] = y_pixel
    return 0


PT_UNUSED = -999


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def searchquader_fast(
    point: cython.double[:],
    quader: cython.double[:, ::1],
    num_cams: cython.int,
    cal_arrays,
    mmlut_datas,
    mmlut_origins,
    mmlut_nrs,
    mmlut_nzs,
    mmlut_rws,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    xr_out: cython.double[:] = None,
    xl_out: cython.double[:] = None,
    yd_out: cython.double[:] = None,
    yu_out: cython.double[:] = None,
):
    """Compute search area for all cameras.

    When xr_out/xl_out/yd_out/yu_out are provided (each (num_cams,) float64),
    they are used as output buffers instead of allocating new arrays internally.

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
    i: cython.Py_ssize_t
    pt: cython.Py_ssize_t
    xr_i: cython.double
    xl_i: cython.double
    yd_i: cython.double
    yu_i: cython.double
    cx: cython.double
    cy: cython.double
    corner_x: cython.double
    corner_y: cython.double
    mrw: cython.double
    mnr: cython.int
    mnz: cython.int
    has_mmlut: cython.int
    xr = np.zeros(num_cams, dtype=np.float64) if xr_out is None else xr_out
    xl = np.zeros(num_cams, dtype=np.float64) if xl_out is None else xl_out
    yd = np.zeros(num_cams, dtype=np.float64) if yd_out is None else yd_out
    yu = np.zeros(num_cams, dtype=np.float64) if yu_out is None else yu_out

    for i in range(num_cams):
        cal = cal_arrays[i]
        md = mmlut_datas[i]
        mo = mmlut_origins[i]
        mnr = mmlut_nrs[i]
        mnz = mmlut_nzs[i]
        mrw = mmlut_rws[i]
        has_mmlut = mnr > 0

        xr_i = 0.0
        xl_i = float(imx)
        yd_i = 0.0
        yu_i = float(imy)

        cx, cy = point_to_pixel_fast(
            point,
            cal,
            md,
            mo,
            mnr,
            mnz,
            mrw,
            has_mmlut,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
        )

        for pt in range(8):
            corner_x, corner_y = point_to_pixel_fast(
                quader[pt],
                cal,
                md,
                mo,
                mnr,
                mnz,
                mrw,
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
            )
            if corner_x < xl_i:
                xl_i = corner_x
            if corner_y < yu_i:
                yu_i = corner_y
            if corner_x > xr_i:
                xr_i = corner_x
            if corner_y > yd_i:
                yd_i = corner_y

        if xl_i < 0.0:
            xl_i = 0.0
        if yu_i < 0.0:
            yu_i = 0.0
        if xr_i > imx:
            xr_i = imx
        if yd_i > imy:
            yd_i = imy

        xr[i] = xr_i - cx
        xl[i] = cx - xl_i
        yd[i] = yd_i - cy
        yu[i] = cy - yu_i

    return xr, xl, yd, yu


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def angle_acc_fast(
    start_x: cython.double,
    start_y: cython.double,
    start_z: cython.double,
    pred_x: cython.double,
    pred_y: cython.double,
    pred_z: cython.double,
    cand_x: cython.double,
    cand_y: cython.double,
    cand_z: cython.double,
):
    """Compute angle and acceleration between predicted and candidate."""
    v0x: cython.double
    v0y: cython.double
    v0z: cython.double
    v1x: cython.double
    v1y: cython.double
    v1z: cython.double
    angle: cython.double
    norm0: cython.double
    norm1: cython.double
    dot: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    acc: cython.double
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
        norm0 = c_sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
        norm1 = c_sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if norm0 == 0.0 or norm1 == 0.0:
            angle = 0.0
        else:
            dot = (v0x * v1x + v0y * v1y + v0z * v1z) / (norm0 * norm1)
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            angle = c_acos(dot) * 200.0 / _M_PI

    dx = v1x - v0x
    dy = v1y - v0y
    dz = v1z - v0z
    acc = c_sqrt(dx * dx + dy * dy + dz * dz)
    return angle, acc


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.nogil
def _angle_acc_out(
    start_x: cython.double,
    start_y: cython.double,
    start_z: cython.double,
    pred_x: cython.double,
    pred_y: cython.double,
    pred_z: cython.double,
    cand_x: cython.double,
    cand_y: cython.double,
    cand_z: cython.double,
    out: cython.double[:],
) -> cython.int:
    """Write angle and acc to out[0], out[1] — no tuple creation."""
    v0x: cython.double
    v0y: cython.double
    v0z: cython.double
    v1x: cython.double
    v1y: cython.double
    v1z: cython.double
    angle: cython.double
    norm0: cython.double
    norm1: cython.double
    dot: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    acc: cython.double
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
        norm0 = c_sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
        norm1 = c_sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if norm0 == 0.0 or norm1 == 0.0:
            angle = 0.0
        else:
            dot = (v0x * v1x + v0y * v1y + v0z * v1z) / (norm0 * norm1)
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            angle = c_acos(dot) * 200.0 / 3.141592653589793

    dx = v1x - v0x
    dy = v1y - v0y
    dz = v1z - v0z
    acc = c_sqrt(dx * dx + dy * dy + dz * dz)
    out[0] = angle
    out[1] = acc
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def _ray_tracing_fast(x: cython.double, y: cython.double, cal: cython.double[:]):
    """Trace ray through multi-media interface.

    Returns (Xx, Xy, Xz, ox, oy, oz) — crossing point and direction.
    """
    ext_x0: cython.double
    ext_y0: cython.double
    ext_z0: cython.double
    dm00: cython.double
    dm10: cython.double
    dm20: cython.double
    dm01: cython.double
    dm11: cython.double
    dm21: cython.double
    dm02: cython.double
    dm12: cython.double
    dm22: cython.double
    int_cc: cython.double
    gx: cython.double
    gy: cython.double
    gz: cython.double
    mm_n1: cython.double
    mm_n2_0: cython.double
    mm_n3: cython.double
    mm_d0: cython.double
    t0: cython.double
    t1: cython.double
    t2: cython.double
    tn: cython.double
    sd0: cython.double
    sd1: cython.double
    sd2: cython.double
    gn: cython.double
    gd0: cython.double
    gd1: cython.double
    gd2: cython.double
    c: cython.double
    dcg: cython.double
    denom: cython.double
    d1: cython.double
    Xb0: cython.double
    Xb1: cython.double
    Xb2: cython.double
    n: cython.double
    bp0: cython.double
    bp1: cython.double
    bp2: cython.double
    bpn: cython.double
    p: cython.double
    n_glass: cython.double
    a2_0: cython.double
    a2_1: cython.double
    a2_2: cython.double
    d2_denom: cython.double
    d2: cython.double
    Xx: cython.double
    Xy: cython.double
    Xz: cython.double
    n_a2: cython.double
    p2: cython.double
    n_final: cython.double
    ox: cython.double
    oy: cython.double
    oz: cython.double
    ext_x0 = cal[0]
    ext_y0 = cal[1]
    ext_z0 = cal[2]
    dm00 = cal[3]
    dm10 = cal[4]
    dm20 = cal[5]
    dm01 = cal[6]
    dm11 = cal[7]
    dm21 = cal[8]
    dm02 = cal[9]
    dm12 = cal[10]
    dm22 = cal[11]
    int_cc = cal[12]
    gx = cal[15]
    gy = cal[16]
    gz = cal[17]
    mm_n1 = cal[20]
    mm_n2_0 = cal[21]
    mm_n3 = cal[22]
    mm_d0 = cal[23]

    # tmp1 = unit_vector([x, y, -int_cc])
    t0 = x
    t1 = y
    t2 = -int_cc
    tn = c_sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if tn > 0.0:
        t0 /= tn
        t1 /= tn
        t2 /= tn

    # start_dir = dm @ tmp1
    sd0 = dm00 * t0 + dm01 * t1 + dm02 * t2
    sd1 = dm10 * t0 + dm11 * t1 + dm12 * t2
    sd2 = dm20 * t0 + dm21 * t1 + dm22 * t2

    # glass_dir = unit_vector(glass_vec)
    gn = c_sqrt(gx * gx + gy * gy + gz * gz)
    if gn > 0.0:
        gd0 = gx / gn
        gd1 = gy / gn
        gd2 = gz / gn
    else:
        gd0 = 0.0
        gd1 = 0.0
        gd2 = 0.0
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
    bp0 = sd0 - gd0 * n
    bp1 = sd1 - gd1 * n
    bp2 = sd2 - gd2 * n
    bpn = c_sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn
        bp1 /= bpn
        bp2 /= bpn

    # Snell's law: air -> glass
    p = c_sqrt(1.0 - n * n) * mm_n1 / mm_n2_0
    n_glass = -c_sqrt(1.0 - p * p)

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
    bp0 = a2_0 - gd0 * n_glass
    bp1 = a2_1 - gd1 * n_glass
    bp2 = a2_2 - gd2 * n_glass
    bpn = c_sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn
        bp1 /= bpn
        bp2 /= bpn

    p2 = c_sqrt(1.0 - n_a2 * n_a2) * mm_n2_0 / mm_n3
    n_final = -c_sqrt(1.0 - p2 * p2)

    ox = bp0 * p2 + gd0 * n_final
    oy = bp1 * p2 + gd1 * n_final
    oz = bp2 * p2 + gd2 * n_final

    return Xx, Xy, Xz, ox, oy, oz


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.nogil
def _ray_tracing_out(
    x: cython.double,
    y: cython.double,
    cal: cython.double[:],
    out: cython.double[:],
) -> cython.int:
    """Write ray tracing results into out[0:6] — no tuple creation."""
    ext_x0: cython.double
    ext_y0: cython.double
    ext_z0: cython.double
    dm00: cython.double
    dm10: cython.double
    dm20: cython.double
    dm01: cython.double
    dm11: cython.double
    dm21: cython.double
    dm02: cython.double
    dm12: cython.double
    dm22: cython.double
    int_cc: cython.double
    gx: cython.double
    gy: cython.double
    gz: cython.double
    mm_n1: cython.double
    mm_n2_0: cython.double
    mm_n3: cython.double
    mm_d0: cython.double
    t0: cython.double
    t1: cython.double
    t2: cython.double
    tn: cython.double
    sd0: cython.double
    sd1: cython.double
    sd2: cython.double
    gn: cython.double
    gd0: cython.double
    gd1: cython.double
    gd2: cython.double
    c: cython.double
    dcg: cython.double
    denom: cython.double
    d1: cython.double
    Xb0: cython.double
    Xb1: cython.double
    Xb2: cython.double
    n: cython.double
    bp0: cython.double
    bp1: cython.double
    bp2: cython.double
    bpn: cython.double
    p: cython.double
    n_glass: cython.double
    a2_0: cython.double
    a2_1: cython.double
    a2_2: cython.double
    d2_denom: cython.double
    d2: cython.double
    Xx: cython.double
    Xy: cython.double
    Xz: cython.double
    n_a2: cython.double
    p2: cython.double
    n_final: cython.double
    ox: cython.double
    oy: cython.double
    oz: cython.double
    ext_x0 = cal[0]
    ext_y0 = cal[1]
    ext_z0 = cal[2]
    dm00 = cal[3]
    dm10 = cal[4]
    dm20 = cal[5]
    dm01 = cal[6]
    dm11 = cal[7]
    dm21 = cal[8]
    dm02 = cal[9]
    dm12 = cal[10]
    dm22 = cal[11]
    int_cc = cal[12]
    gx = cal[15]
    gy = cal[16]
    gz = cal[17]
    mm_n1 = cal[20]
    mm_n2_0 = cal[21]
    mm_n3 = cal[22]
    mm_d0 = cal[23]

    # tmp1 = unit_vector([x, y, -int_cc])
    t0 = x
    t1 = y
    t2 = -int_cc
    tn = c_sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if tn > 0.0:
        t0 /= tn
        t1 /= tn
        t2 /= tn

    # start_dir = dm @ tmp1
    sd0 = dm00 * t0 + dm01 * t1 + dm02 * t2
    sd1 = dm10 * t0 + dm11 * t1 + dm12 * t2
    sd2 = dm20 * t0 + dm21 * t1 + dm22 * t2

    # glass_dir = unit_vector(glass_vec)
    gn = c_sqrt(gx * gx + gy * gy + gz * gz)
    if gn > 0.0:
        gd0 = gx / gn
        gd1 = gy / gn
        gd2 = gz / gn
    else:
        gd0 = 0.0
        gd1 = 0.0
        gd2 = 0.0
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
    bp0 = sd0 - gd0 * n
    bp1 = sd1 - gd1 * n
    bp2 = sd2 - gd2 * n
    bpn = c_sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn
        bp1 /= bpn
        bp2 /= bpn

    # Snell's law: air -> glass
    p = c_sqrt(1.0 - n * n) * mm_n1 / mm_n2_0
    n_glass = -c_sqrt(1.0 - p * p)

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
    bp0 = a2_0 - gd0 * n_glass
    bp1 = a2_1 - gd1 * n_glass
    bp2 = a2_2 - gd2 * n_glass
    bpn = c_sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn
        bp1 /= bpn
        bp2 /= bpn

    p2 = c_sqrt(1.0 - n_a2 * n_a2) * mm_n2_0 / mm_n3
    n_final = -c_sqrt(1.0 - p2 * p2)

    ox = bp0 * p2 + gd0 * n_final
    oy = bp1 * p2 + gd1 * n_final
    oz = bp2 * p2 + gd2 * n_final

    out[0] = Xx
    out[1] = Xy
    out[2] = Xz
    out[3] = ox
    out[4] = oy
    out[5] = oz
    return 0
