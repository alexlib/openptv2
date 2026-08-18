# ruff: noqa: E402,F842
"""Compiled kernels for the tracking hot path.

Auto-generated split from track_kernels.py.
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        asin as c_asin,
    )
    from cython.cimports.libc.math import (
        atan as c_atan,
    )
    from cython.cimports.libc.math import (
        cos as c_cos,
    )
    from cython.cimports.libc.math import (
        sin as c_sin,
    )
    from cython.cimports.libc.math import (
        sqrt as c_sqrt,
    )
    from cython.cimports.libc.math import (
        tan as c_tan,
    )
else:
    from math import (
        asin as c_asin,
    )
    from math import (
        atan as c_atan,
    )
    from math import (
        cos as c_cos,
    )
    from math import (
        sin as c_sin,
    )
    from math import (
        sqrt as c_sqrt,
    )
    from math import (
        tan as c_tan,
    )

_M_PI: cython.double = 3.141592653589793


from .track_kernels_geom import (
    _point_to_pixel_out,
)
from .track_kernels_search import (
    candsearch_in_pix_rest_fast,
)

# Sentinel values — typed C int/double
cython.declare(
    PT_UNUSED=cython.int,
    COORD_UNUSED=cython.double,
)
PT_UNUSED = -999
COORD_UNUSED = -1e10


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

    t0 = x
    t1 = y
    t2 = -int_cc
    tn = c_sqrt(t0 * t0 + t1 * t1 + t2 * t2)
    if tn > 0.0:
        t0 /= tn
        t1 /= tn
        t2 /= tn

    sd0 = dm00 * t0 + dm01 * t1 + dm02 * t2
    sd1 = dm10 * t0 + dm11 * t1 + dm12 * t2
    sd2 = dm20 * t0 + dm21 * t1 + dm22 * t2

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

    dcg = gd0 * ext_x0 + gd1 * ext_y0 + gd2 * ext_z0 - c
    denom = gd0 * sd0 + gd1 * sd1 + gd2 * sd2
    d1 = -dcg / denom

    Xb0 = ext_x0 + sd0 * d1
    Xb1 = ext_y0 + sd1 * d1
    Xb2 = ext_z0 + sd2 * d1

    n = sd0 * gd0 + sd1 * gd1 + sd2 * gd2
    bp0 = sd0 - gd0 * n
    bp1 = sd1 - gd1 * n
    bp2 = sd2 - gd2 * n
    bpn = c_sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn
        bp1 /= bpn
        bp2 /= bpn

    p = c_sqrt(1.0 - n * n) * mm_n1 / mm_n2_0
    n_glass = c_sqrt(1.0 - p * p) if n >= 0 else -c_sqrt(1.0 - p * p)

    a2_0 = bp0 * p + gd0 * n_glass
    a2_1 = bp1 * p + gd1 * n_glass
    a2_2 = bp2 * p + gd2 * n_glass

    d2_denom = gd0 * a2_0 + gd1 * a2_1 + gd2 * a2_2
    d2 = mm_d0 / abs(d2_denom)

    Xx = Xb0 + a2_0 * d2
    Xy = Xb1 + a2_1 * d2
    Xz = Xb2 + a2_2 * d2

    n_a2 = a2_0 * gd0 + a2_1 * gd1 + a2_2 * gd2
    bp0 = a2_0 - gd0 * n_glass
    bp1 = a2_1 - gd1 * n_glass
    bp2 = a2_2 - gd2 * n_glass
    bpn = c_sqrt(bp0 * bp0 + bp1 * bp1 + bp2 * bp2)
    if bpn > 0.0:
        bp0 /= bpn
        bp1 /= bpn
        bp2 /= bpn

    p2 = c_sqrt(1.0 - n_a2 * n_a2) * mm_n2_0 / mm_n3
    n_final = c_sqrt(1.0 - p2 * p2) if n_a2 >= 0 else -c_sqrt(1.0 - p2 * p2)

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


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def _point_position_out(
    targets: cython.double[:, ::1],
    num_cams: cython.int,
    cal_arr: cython.double[:, ::1],
    out: cython.double[:],
    scratch_ray: cython.double[:],
) -> cython.double:
    """Internal — writes 3D position to out[0:3], returns avg_dist (scalar).

    Pure C entry — zero Python object creation, zero tuple overhead.
    """
    cam: cython.int
    pair: cython.int
    tx: cython.double
    ty: cython.double
    Xx: cython.double
    Xy: cython.double
    Xz: cython.double
    ox: cython.double
    oy: cython.double
    oz: cython.double
    dtot: cython.double
    num_used: cython.int
    px: cython.double
    py: cython.double
    pz: cython.double
    v1x: cython.double
    v1y: cython.double
    v1z: cython.double
    d1x: cython.double
    d1y: cython.double
    d1z: cython.double
    v2x: cython.double
    v2y: cython.double
    v2z: cython.double
    d2x: cython.double
    d2y: cython.double
    d2z: cython.double
    sp0: cython.double
    sp1: cython.double
    sp2: cython.double
    pb0: cython.double
    pb1: cython.double
    pb2: cython.double
    scale: cython.double
    dist: cython.double
    mx: cython.double
    my: cython.double
    mz: cython.double
    t0: cython.double
    t1: cython.double
    t2: cython.double
    s1: cython.double
    on1x: cython.double
    on1y: cython.double
    on1z: cython.double
    s2: cython.double
    on2x: cython.double
    on2y: cython.double
    on2z: cython.double
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    verts_x: cython.double[:]
    verts_y: cython.double[:]
    verts_z: cython.double[:]
    dirs_x: cython.double[:]
    dirs_y: cython.double[:]
    dirs_z: cython.double[:]
    valid: cython.int[:]
    _vi: cython.int
    with cython.gil:
        _verts_x_buf = np.zeros(8, dtype=np.float64)
        verts_x = _verts_x_buf
        _verts_y_buf = np.zeros(8, dtype=np.float64)
        verts_y = _verts_y_buf
        _verts_z_buf = np.zeros(8, dtype=np.float64)
        verts_z = _verts_z_buf
        _dirs_x_buf = np.zeros(8, dtype=np.float64)
        dirs_x = _dirs_x_buf
        _dirs_y_buf = np.zeros(8, dtype=np.float64)
        dirs_y = _dirs_y_buf
        _dirs_z_buf = np.zeros(8, dtype=np.float64)
        dirs_z = _dirs_z_buf
        _valid_buf = np.zeros(8, dtype=np.int32)
        valid = _valid_buf

    for _vi in range(8):
        valid[_vi] = 0

    for cam in range(num_cams):
        tx = targets[cam, 0]
        ty = targets[cam, 1]
        if tx == COORD_UNUSED:
            continue
        _ray_tracing_out(tx, ty, cal_arr[cam], scratch_ray)
        verts_x[cam] = scratch_ray[0]
        verts_y[cam] = scratch_ray[1]
        verts_z[cam] = scratch_ray[2]
        dirs_x[cam] = scratch_ray[3]
        dirs_y[cam] = scratch_ray[4]
        dirs_z[cam] = scratch_ray[5]
        valid[cam] = 1

    dtot = 0.0
    num_used = 0
    px = 0.0
    py = 0.0
    pz = 0.0

    for cam in range(num_cams):
        if valid[cam] == 0:
            continue
        for pair in range(cam + 1, num_cams):
            if valid[pair] == 0:
                continue

            v1x = verts_x[cam]
            v1y = verts_y[cam]
            v1z = verts_z[cam]
            d1x = dirs_x[cam]
            d1y = dirs_y[cam]
            d1z = dirs_z[cam]
            v2x = verts_x[pair]
            v2y = verts_y[pair]
            v2z = verts_z[pair]
            d2x = dirs_x[pair]
            d2y = dirs_y[pair]
            d2z = dirs_z[pair]

            sp0 = v2x - v1x
            sp1 = v2y - v1y
            sp2 = v2z - v1z

            pb0 = d1y * d2z - d1z * d2y
            pb1 = d1z * d2x - d1x * d2z
            pb2 = d1x * d2y - d1y * d2x
            scale = pb0 * pb0 + pb1 * pb1 + pb2 * pb2

            if scale < 1e-20:
                dist = c_sqrt(sp0 * sp0 + sp1 * sp1 + sp2 * sp2)
                mx = (v1x + v2x) * 0.5
                my = (v1y + v2y) * 0.5
                mz = (v1z + v2z) * 0.5
            else:
                t0 = sp1 * d2z - sp2 * d2y
                t1 = sp2 * d2x - sp0 * d2z
                t2 = sp0 * d2y - sp1 * d2x
                s1 = (pb0 * t0 + pb1 * t1 + pb2 * t2) / scale
                on1x = v1x + d1x * s1
                on1y = v1y + d1y * s1
                on1z = v1z + d1z * s1

                t0 = sp1 * d1z - sp2 * d1y
                t1 = sp2 * d1x - sp0 * d1z
                t2 = sp0 * d1y - sp1 * d1x
                s2 = (pb0 * t0 + pb1 * t1 + pb2 * t2) / scale
                on2x = v2x + d2x * s2
                on2y = v2y + d2y * s2
                on2z = v2z + d2z * s2

                ddx = on1x - on2x
                ddy = on1y - on2y
                ddz = on1z - on2z
                dist = c_sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                mx = (on1x + on2x) * 0.5
                my = (on1y + on2y) * 0.5
                mz = (on1z + on2z) * 0.5

            num_used += 1
            dtot += dist
            px += mx
            py += my
            pz += mz

    if num_used > 0:
        inv = 1.0 / num_used
        out[0] = px * inv
        out[1] = py * inv
        out[2] = pz * inv
        return dtot * inv
    else:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        return 0.0


@cython.ccall
def point_position_fast(
    targets: cython.double[:, ::1], num_cams: cython.int, cal_arr: cython.double[:, ::1]
):
    """Compute 3D position from multiple camera rays.

    Returns:
        (pos, avg_dist) — (3,) float64 position and average ray distance.
    """
    pos = np.zeros(3, dtype=np.float64)
    pos_mv: cython.double[:] = pos
    scratch_ray = np.zeros(6, dtype=np.float64)
    dtot = _point_position_out(targets, num_cams, cal_arr, pos_mv, scratch_ray)
    return pos, dtot


@cython.ccall
def pixel_to_metric_fast(
    x_pixel: cython.double,
    y_pixel: cython.double,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    chfield: cython.int,
):
    """Convert pixel to metric coordinates."""
    yp: cython.double
    x_metric: cython.double
    y_metric: cython.double
    yp = y_pixel
    if chfield == 1:
        yp = 2.0 * yp + 1.0
    elif chfield == 2:
        yp = 2.0 * yp
    x_metric = (x_pixel - imx * 0.5) * pix_x
    y_metric = (imy * 0.5 - yp) * pix_y
    return x_metric, y_metric


@cython.ccall
@cython.inline
@cython.cdivision(True)
@cython.profile(False)
@cython.nogil
def _pixel_to_metric_out(
    x_pixel: cython.double,
    y_pixel: cython.double,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    chfield: cython.int,
    out: cython.double[:],
) -> cython.int:
    """Write pixel-to-metric coords to out[0], out[1]."""
    yp: cython.double
    yp = y_pixel
    if chfield == 1:
        yp = 2.0 * yp + 1.0
    elif chfield == 2:
        yp = 2.0 * yp
    out[0] = (x_pixel - imx * 0.5) * pix_x
    out[1] = (imy * 0.5 - yp) * pix_y
    return 0


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def dist_to_flat_fast(
    dist_x: cython.double,
    dist_y: cython.double,
    xh: cython.double,
    yh: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    tol: cython.double,
):
    """Inverse Brown distortion."""
    r_init: cython.double
    sin_she: cython.double
    cos_she: cython.double
    inv_scx: cython.double
    xq: cython.double
    yq: cython.double
    _: cython.int
    r2: cython.double
    r4: cython.double
    r6: cython.double
    radial_factor: cython.double
    dx: cython.double
    dy: cython.double
    xq_new: cython.double
    yq_new: cython.double
    dx_change: cython.double
    dy_change: cython.double
    r_init = c_sqrt(dist_x * dist_x + dist_y * dist_y)
    if r_init < 1e-10:
        return -xh, -yh

    sin_she = c_sin(she)
    cos_she = c_cos(she)
    inv_scx = 1.0 / scx

    xq = (dist_x + dist_y * sin_she) * inv_scx
    yq = dist_y / cos_she

    for _ in range(50):
        r2 = xq * xq + yq * yq
        r4 = r2 * r2
        r6 = r4 * r2

        radial_factor = k1 * r2 + k2 * r4 + k3 * r6

        dx = xq * radial_factor + p1 * (r2 + 2.0 * xq * xq) + 2.0 * p2 * xq * yq
        dy = yq * radial_factor + p2 * (r2 + 2.0 * yq * yq) + 2.0 * p1 * xq * yq

        xq_new = (dist_x + dist_y * sin_she) * inv_scx - dx
        yq_new = dist_y / cos_she - dy

        dx_change = xq_new - xq
        dy_change = yq_new - yq

        xq += 0.5 * dx_change
        yq += 0.5 * dy_change

        if c_sqrt(dx_change * dx_change + dy_change * dy_change) < tol:
            break

    return xq - xh, yq - yh


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.nogil
def _dist_to_flat_out(
    dist_x: cython.double,
    dist_y: cython.double,
    xh: cython.double,
    yh: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    tol: cython.double,
    out: cython.double[:],
) -> cython.int:
    """Write dist-to-flat coords to out[0], out[1]."""
    r_init: cython.double = c_sqrt(dist_x * dist_x + dist_y * dist_y)
    if r_init < 1e-10:
        out[0] = -xh
        out[1] = -yh
        return 0
    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)
    inv_scx: cython.double = 1.0 / scx
    xq: cython.double = (dist_x + dist_y * sin_she) * inv_scx
    yq: cython.double = dist_y / cos_she
    _: cython.int
    r2: cython.double
    r4: cython.double
    r6: cython.double
    radial_factor: cython.double
    dx: cython.double
    dy: cython.double
    xq_new: cython.double
    yq_new: cython.double
    dx_change: cython.double
    dy_change: cython.double
    for _ in range(50):
        r2 = xq * xq + yq * yq
        r4 = r2 * r2
        r6 = r4 * r2
        radial_factor = k1 * r2 + k2 * r4 + k3 * r6
        dx = xq * radial_factor + p1 * (r2 + 2.0 * xq * xq) + 2.0 * p2 * xq * yq
        dy = yq * radial_factor + p2 * (r2 + 2.0 * yq * yq) + 2.0 * p1 * xq * yq
        xq_new = (dist_x + dist_y * sin_she) * inv_scx - dx
        yq_new = dist_y / cos_she - dy
        dx_change = xq_new - xq
        dy_change = yq_new - yq
        xq += 0.5 * dx_change
        yq += 0.5 * dy_change
        if c_sqrt(dx_change * dx_change + dy_change * dy_change) < tol:
            break
    out[0] = xq - xh
    out[1] = yq - yh
    return 0


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def assess_new_position_fast(
    pos: cython.double[:],
    num_cams: cython.int,
    add_part: cython.double,
    cal_arr: cython.double[:, ::1],
    md_arr: object,
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    targ_x: cython.double[:, ::1],
    targ_y: cython.double[:, ::1],
    targ_tnr: cython.int[:, ::1],
    num_targets,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
    tr_unused: cython.int,
    coord_unused: cython.double,
    use_proj: cython.bint,
    proj_x: cython.double[:],
    proj_y: cython.double[:],
    targ_pos_out: cython.double[:, ::1] = None,
    cand_inds_out: cython.int[:] = None,
    scratch: cython.double[:] = None,
):
    """Assess new position: project, find unused targets, undistort.

    When use_proj=True, proj_x[cam] and proj_y[cam] provide pre-computed
    pixel projections (avoids redundant _point_to_pixel_out calls).
    When use_proj=False, proj_x/proj_y are unused (can be empty arrays).

    When targ_pos_out, cand_inds_out, and scratch are provided, they are
    used as pre-allocated output buffers instead of allocating new arrays.

    Returns (targ_pos, cand_inds, valid_cams).
    """
    cam: cython.int
    valid_cams: cython.int
    best: cython.int
    count: cython.int
    has_mmlut: cython.int
    px: cython.double
    py: cython.double
    mx: cython.double
    my: cython.double
    fx: cython.double
    fy: cython.double
    _pp_mv: cython.double[:]
    targ_pos: cython.double[:, ::1]
    cand_inds: cython.int[:]
    if scratch is not None:
        _pp_mv = scratch
    else:
        _pp_mv = np.empty(2, dtype=np.float64)
    if targ_pos_out is not None:
        targ_pos = targ_pos_out
    else:
        targ_pos = np.full((num_cams, 2), coord_unused, dtype=np.float64)
    if cand_inds_out is not None:
        cand_inds = cand_inds_out
    else:
        cand_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)

    for cam in range(num_cams):
        if use_proj:
            # Use pre-computed projection (caller already projected this pos)
            px = proj_x[cam]
            py = proj_y[cam]
        else:
            has_mmlut = mnr_arr[cam] > 0
            _point_to_pixel_out(
                pos,
                cal_arr[cam],
                md_arr[cam],
                mo_arr[cam],
                mnr_arr[cam],
                mnz_arr[cam],
                mrw_arr[cam],
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            px = _pp_mv[0]
            py = _pp_mv[1]

        best, count = candsearch_in_pix_rest_fast(
            targ_x[cam],
            targ_y[cam],
            targ_tnr[cam],
            num_targets[cam],
            px,
            py,
            add_part,
            add_part,
            add_part,
            add_part,
            imx,
            imy,
            tr_unused,
        )

        if count > 0:
            cand_inds[cam] = best
            targ_pos[cam, 0] = targ_x[cam, best]
            targ_pos[cam, 1] = targ_y[cam, best]

    valid_cams = 0
    for cam in range(num_cams):
        if targ_pos[cam, 0] != coord_unused:
            _pixel_to_metric_out(
                targ_pos[cam, 0],
                targ_pos[cam, 1],
                imx,
                imy,
                pix_x,
                pix_y,
                chfield,
                _pp_mv,
            )
            mx = _pp_mv[0]
            my = _pp_mv[1]

            cal = cal_arr[cam]
            _dist_to_flat_out(
                mx,
                my,
                cal[13],
                cal[14],
                cal[24],
                cal[25],
                cal[26],
                cal[27],
                cal[28],
                cal[29],
                cal[30],
                flatten_tol,
                _pp_mv,
            )

            targ_pos[cam, 0] = _pp_mv[0]
            targ_pos[cam, 1] = _pp_mv[1]
            valid_cams += 1

    return targ_pos, cand_inds, valid_cams


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def _candsearch_in_pix_rest_nogil(
    targ_x: cython.double[:],
    targ_y: cython.double[:],
    targ_tnr: cython.int[:],
    num_targets: cython.int,
    cent_x: cython.double,
    cent_y: cython.double,
    dl: cython.double,
    dr: cython.double,
    du: cython.double,
    dd: cython.double,
    imx: cython.double,
    imy: cython.double,
    tr_unused: cython.int,
) -> cython.int:
    """Find closest unused candidate GIL-free."""
    xmin: cython.double
    xmax: cython.double
    ymin: cython.double
    ymax: cython.double
    best: cython.int
    dmin: cython.double
    j0: cython.int
    dj: cython.int
    j: cython.int
    ty: cython.double
    tx: cython.double
    dx: cython.double
    dy: cython.double
    d: cython.double
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

    best = tr_unused
    dmin = 1e20

    if not (0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy):
        return best

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
                d = c_sqrt(dx * dx + dy * dy)
                if d < dmin:
                    dmin = d
                    best = j

    return best


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def assess_new_position_fast_nogil(
    pos: cython.double[:],
    num_cams: cython.int,
    add_part: cython.double,
    cal_arr: cython.double[:, ::1],
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    targ_x: cython.double[:, ::1],
    targ_y: cython.double[:, ::1],
    targ_tnr: cython.int[:, ::1],
    num_targets: cython.int[:],
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
    tr_unused: cython.int,
    coord_unused: cython.double,
    proj_x: cython.double[:],
    proj_y: cython.double[:],
    targ_pos_out: cython.double[:, :],
    cand_inds_out: cython.int[:],
    scratch: cython.double[:],
) -> cython.int:
    """Assess new position GIL-free. Assumes use_proj=True."""
    cam: cython.int
    valid_cams: cython.int
    best: cython.int
    px: cython.double
    py: cython.double
    mx: cython.double
    my: cython.double

    for cam in range(num_cams):
        cand_inds_out[cam] = tr_unused
        targ_pos_out[cam, 0] = coord_unused
        targ_pos_out[cam, 1] = coord_unused

    for cam in range(num_cams):
        px = proj_x[cam]
        py = proj_y[cam]

        best = _candsearch_in_pix_rest_nogil(
            targ_x[cam],
            targ_y[cam],
            targ_tnr[cam],
            num_targets[cam],
            px,
            py,
            add_part,
            add_part,
            add_part,
            add_part,
            imx,
            imy,
            tr_unused,
        )

        if best != tr_unused:
            cand_inds_out[cam] = best
            targ_pos_out[cam, 0] = targ_x[cam, best]
            targ_pos_out[cam, 1] = targ_y[cam, best]

    valid_cams = 0
    for cam in range(num_cams):
        if targ_pos_out[cam, 0] != coord_unused:
            _pixel_to_metric_out(
                targ_pos_out[cam, 0],
                targ_pos_out[cam, 1],
                imx,
                imy,
                pix_x,
                pix_y,
                chfield,
                scratch,
            )
            mx = scratch[0]
            my = scratch[1]

            cal = cal_arr[cam]
            _dist_to_flat_out(
                mx,
                my,
                cal[13],
                cal[14],
                cal[24],
                cal[25],
                cal[26],
                cal[27],
                cal[28],
                cal[29],
                cal[30],
                flatten_tol,
                scratch,
            )

            targ_pos_out[cam, 0] = scratch[0]
            targ_pos_out[cam, 1] = scratch[1]
            valid_cams += 1

    return valid_cams


POSI_K = 80
MAX_CANDS_K = 32
TR_UNUSED_K = -1
CORRES_NONE_K = -1
PREV_NONE_K = -1
NEXT_NONE_K = -2
COORD_UNUSED_K = -1e10
ADD_PART_K = 3.0


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def metric_to_pixel_fast(
    x_metric: cython.double,
    y_metric: cython.double,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    chfield: cython.int,
):
    """Convert metric to pixel coordinates."""
    x_pixel: cython.double
    y_pixel: cython.double
    x_pixel = x_metric / pix_x + imx * 0.5
    y_pixel = imy * 0.5 - y_metric / pix_y
    if chfield == 1:
        y_pixel = (y_pixel - 1.0) * 0.5
    elif chfield == 2:
        y_pixel = y_pixel * 0.5
    return x_pixel, y_pixel


@cython.ccall
@cython.inline
@cython.cdivision(True)
@cython.profile(False)
def _metric_to_pixel_out(
    x_metric: cython.double,
    y_metric: cython.double,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    chfield: cython.int,
    out: cython.double[:],
):
    """Write metric-to-pixel coords to out[0], out[1]."""
    x_pixel: cython.double = x_metric / pix_x + imx * 0.5
    y_pixel: cython.double = imy * 0.5 - y_metric / pix_y
    if chfield == 1:
        y_pixel = (y_pixel - 1.0) * 0.5
    elif chfield == 2:
        y_pixel = y_pixel * 0.5
    out[0] = x_pixel
    out[1] = y_pixel


@cython.boundscheck(False)
@cython.wraparound(False)
def _flat_image_coord_fast(
    pos: cython.double[:],
    cal: cython.double[:],
    mmlut_data: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
):
    """Project 3D to flat metric image coordinates.

    Returns (x, y) without distortion or pixel conversion.
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
    gx: cython.double
    gy: cython.double
    gz: cython.double
    inv_dog: cython.double
    mm_n1: cython.double
    mm_n2_0: cython.double
    mm_n3: cython.double
    mm_d0: cython.double
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
    has_mmlut: cython.bint
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
    gx = cal[15]
    gy = cal[16]
    gz = cal[17]
    inv_dog = cal[19]
    mm_n1 = cal[20]
    mm_n2_0 = cal[21]
    mm_n3 = cal[22]
    mm_d0 = cal[23]

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

    radial_shift = 1.0
    has_mmlut = len(mmlut_data) > 0
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
def _img_coord_fast(
    pos: cython.double[:],
    cal: cython.double[:],
    mmlut_data: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
):
    """Project 3D to distorted metric image coordinates."""
    xh: cython.double
    yh: cython.double
    k1: cython.double
    k2: cython.double
    k3: cython.double
    p1: cython.double
    p2: cython.double
    scx: cython.double
    she: cython.double
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
    x, y = _flat_image_coord_fast(
        pos, cal, mmlut_data, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw
    )

    xh = cal[13]
    yh = cal[14]
    k1 = cal[24]
    k2 = cal[25]
    k3 = cal[26]
    p1 = cal[27]
    p2 = cal[28]
    scx = cal[29]
    she = cal[30]

    x += xh
    y += yh
    r = c_sqrt(x * x + y * y)
    if r < 1e-10:
        return 0.0, 0.0

    r2 = r * r
    r4 = r2 * r2
    radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r4 * r2
    xd = x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
    yd = y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
    sin_she = c_sin(she)
    cos_she = c_cos(she)
    x_dist = scx * (xd - sin_she * yd)
    y_dist = scx * cos_she * yd

    return x_dist, y_dist


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def img_coord_batch_fast(
    positions: cython.double[:, ::1],
    cal: cython.double[:],
    mmlut_data: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
):
    """Project N 3D positions to distorted metric coords."""
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    n = positions.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = _img_coord_fast(
            positions[i], cal, mmlut_data, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw
        )
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def flat_image_coord_batch_fast(
    positions: cython.double[:, ::1],
    cal: cython.double[:],
    mmlut_data: cython.double[:],
    mmlut_origin: cython.double[:],
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
):
    """Project N 3D positions to flat metric coords."""
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    n = positions.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = _flat_image_coord_fast(
            positions[i], cal, mmlut_data, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw
        )
    return result
