# ruff: noqa: F842
"""3D position reconstruction via multi-camera ray tracing and assess_new_position."""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt,
    )
else:
    from math import (
        sqrt as c_sqrt,
    )

if cython.compiled:
    from cython.cimports.openptv2.algorithms.track_kernels_pixel import (
        _candsearch_in_pix_rest_nogil,
        _dist_to_flat_out,
        _pixel_to_metric_out,
    )
else:
    from .track_kernels_pixel import (
        _candsearch_in_pix_rest_nogil,
        _dist_to_flat_out,
        _pixel_to_metric_out,
    )

cython.declare(
    PT_UNUSED=cython.int,
    POSI_K=cython.int,
    MAX_CANDS_K=cython.int,
    TR_UNUSED_K=cython.int,
    CORRES_NONE_K=cython.int,
    PREV_NONE_K=cython.int,
    NEXT_NONE_K=cython.int,
    COORD_UNUSED_K=cython.double,
    ADD_PART_K=cython.double,
)
PT_UNUSED = -999
POSI_K = 80
MAX_CANDS_K = 32
TR_UNUSED_K = -1
CORRES_NONE_K = -1
PREV_NONE_K = -1
NEXT_NONE_K = -2
COORD_UNUSED_K = -1e10
ADD_PART_K = 3.0


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
    n_glass = -c_sqrt(1.0 - p * p)

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


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
@cython.exceptval(check=False)
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
        if tx == COORD_UNUSED_K:
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
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
@cython.exceptval(check=False)
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
