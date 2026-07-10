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

if cython.compiled:

    @cython.cfunc
    @cython.cname("__sync_bool_compare_and_swap")
    @cython.nogil
    @cython.exceptval(check=False)
    @cython.returns(cython.int)
    def __sync_bool_compare_and_swap(
        ptr: cython.pointer(cython.int), oldval: cython.int, newval: cython.int
    ) -> cython.int: ...


if not cython.compiled:

    def __sync_bool_compare_and_swap(ptr, oldval, newval):
        return True


from cython.parallel import prange, threadid

from .track_kernels_geom import (
    searchquader_fast,
)
from .track_kernels_search import (
    _sorted_candidates_fast_out,
)
from .track_kernels_transform import (
    assess_new_position_fast,
    point_position_fast,
)

# Constants for tracking kernels — typed C int/double via cython.declare()
# to avoid Python int boxing in every comparison inside the hot particle loop.
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
MAX_CANDS_K = 4
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


@cython.cfunc
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


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def candsearch_in_pix_fast_nogil(
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
    out_indices: cython.p_int,
) -> cython.int:
    xmin: cython.double
    xmax: cython.double
    ymin: cython.double
    ymax: cython.double
    p1: cython.int
    p2: cython.int
    p3: cython.int
    p4: cython.int
    d1: cython.double
    d2: cython.double
    d3: cython.double
    d4: cython.double
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

    p1 = -999  # PT_UNUSED is -999
    p2 = -999
    p3 = -999
    p4 = -999
    d1 = 1e20
    d2 = 1e20
    d3 = 1e20
    d4 = 1e20

    if not (0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy):
        out_indices[0] = p1
        out_indices[1] = p2
        out_indices[2] = p3
        out_indices[3] = p4
        return 0

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
                d = c_sqrt(dx * dx + dy * dy)

                if d < d1:
                    p4 = p3
                    p3 = p2
                    p2 = p1
                    p1 = j
                    d4 = d3
                    d3 = d2
                    d2 = d1
                    d1 = d
                elif d < d2:
                    p4 = p3
                    p3 = p2
                    p2 = j
                    d4 = d3
                    d3 = d2
                    d2 = d
                elif d < d3:
                    p4 = p3
                    p3 = j
                    d4 = d3
                    d3 = d
                elif d < d4:
                    p4 = j
                    d4 = d

    out_indices[0] = p1
    out_indices[1] = p2
    out_indices[2] = p3
    out_indices[3] = p4
    return 0


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def _sorted_candidates_fast_out_nogil(
    center: cython.double[:],
    center_proj_x: cython.double[:],
    center_proj_y: cython.double[:],
    num_cams: cython.int,
    max_cands: cython.int,
    cal_arr: cython.double[:, ::1],
    md0: cython.double[:],
    md1: cython.double[:],
    md2: cython.double[:],
    md3: cython.double[:],
    md4: cython.double[:],
    md5: cython.double[:],
    md6: cython.double[:],
    md7: cython.double[:],
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    targ_x: cython.double[:, ::1],
    targ_y: cython.double[:, ::1],
    targ_tnr: cython.int[:, ::1],
    num_targets: cython.int[:],
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    tr_unused: cython.int,
    ftnr_out: cython.int[:],
    freq_out: cython.int[:],
    whichcam_out: cython.int[:, :],
    pt_buf: cython.double[:],
    _pp: cython.double[:],
) -> cython.int:
    n: cython.int
    px: cython.double
    py: cython.double
    pz: cython.double
    i: cython.int
    pt: cython.int
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
    cam: cython.int
    base: cython.int
    ci: cython.int
    idx: cython.int
    ftnr_i: cython.int
    num_valid: cython.int
    j: cython.int
    m: cython.int
    k: cython.int

    _quader_buf = np.zeros(24, dtype=np.float64)
    quader_buf: cython.double[:] = _quader_buf

    n = num_cams * max_cands

    # --- searchquader inlined ---
    px = center[0]
    py = center[1]
    pz = center[2]
    for pt in range(8):
        quader_buf[pt * 3 + 0] = px + (dvxmax if pt & 1 else dvxmin)
        quader_buf[pt * 3 + 1] = py + (dvymax if pt & 2 else dvymin)
        quader_buf[pt * 3 + 2] = pz + (dvzmax if pt & 4 else dvzmin)

    _xr_buf = np.zeros(8, dtype=np.float64)
    xr: cython.double[:] = _xr_buf
    _xl_buf = np.zeros(8, dtype=np.float64)
    xl: cython.double[:] = _xl_buf
    _yd_buf = np.zeros(8, dtype=np.float64)
    yd: cython.double[:] = _yd_buf
    _yu_buf = np.zeros(8, dtype=np.float64)
    yu: cython.double[:] = _yu_buf

    for i in range(num_cams):
        cal = cal_arr[i]
        mo = mo_arr[i]
        mnr = mnr_arr[i]
        mnz = mnz_arr[i]
        mrw = mrw_arr[i]
        has_mmlut = mnr > 0

        # Select pre-unpacked md memoryview without GIL
        md: cython.double[:]
        if i == 0:
            md = md0
        elif i == 1:
            md = md1
        elif i == 2:
            md = md2
        elif i == 3:
            md = md3
        elif i == 4:
            md = md4
        elif i == 5:
            md = md5
        elif i == 6:
            md = md6
        else:
            md = md7

        xr_i = 0.0
        xl_i = float(imx)
        yd_i = 0.0
        yu_i = float(imy)
        # Use pre-computed center projection
        cx = center_proj_x[i]
        cy = center_proj_y[i]
        for pt in range(8):
            pt_buf[0] = quader_buf[pt * 3 + 0]
            pt_buf[1] = quader_buf[pt * 3 + 1]
            pt_buf[2] = quader_buf[pt * 3 + 2]
            _point_to_pixel_out(
                pt_buf,
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
                _pp,
            )
            corner_x = _pp[0]
            corner_y = _pp[1]
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

    # --- initialize output buffers ---
    for i in range(n):
        ftnr_out[i] = tr_unused
        freq_out[i] = 0
        for j in range(num_cams):
            whichcam_out[i, j] = 0

    # Local buffer for candsearch_in_pix_fast_nogil
    _cands_buf = np.zeros(4, dtype=np.int32)
    cands_buf: cython.int[:] = _cands_buf

    # --- candsearch per camera, write directly into ftnr_out/whichcam_out ---
    for cam in range(num_cams):
        candsearch_in_pix_fast_nogil(
            targ_x[cam],
            targ_y[cam],
            targ_tnr[cam],
            num_targets[cam],
            center_proj_x[cam],
            center_proj_y[cam],
            xl[cam],
            xr[cam],
            yu[cam],
            yd[cam],
            imx,
            imy,
            tr_unused,
            cands_buf,
        )

        base = cam * max_cands
        for ci in range(4):
            idx = cands_buf[ci]
            if idx != -999:  # PT_UNUSED is -999
                whichcam_out[base + ci, cam] = 1
                ftnr_out[base + ci] = int(targ_tnr[cam, idx])

    # --- sort_candidates_by_freq inlined ---
    for i in range(n):
        ftnr_i = ftnr_out[i]
        if ftnr_i == tr_unused:
            continue
        for j in range(num_cams):
            for m in range(max_cands):
                if ftnr_i == ftnr_out[max_cands * j + m]:
                    whichcam_out[i, j] = 1

    for i in range(n):
        if ftnr_out[i] != tr_unused:
            for j in range(num_cams):
                if whichcam_out[i, j] == 1:
                    freq_out[i] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq_out[j - 1] < freq_out[j]:
                ftnr_out[j - 1], ftnr_out[j] = ftnr_out[j], ftnr_out[j - 1]
                freq_out[j - 1], freq_out[j] = freq_out[j], freq_out[j - 1]
                for k in range(num_cams):
                    whichcam_out[j - 1, k], whichcam_out[j, k] = (
                        whichcam_out[j, k],
                        whichcam_out[j - 1, k],
                    )

    for i in range(n):
        ftnr_i = ftnr_out[i]
        for j in range(i + 1, n):
            if ftnr_out[j] == ftnr_i or freq_out[j] < 2:
                freq_out[j] = 0
                ftnr_out[j] = tr_unused

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq_out[j - 1] < freq_out[j]:
                ftnr_out[j - 1], ftnr_out[j] = ftnr_out[j], ftnr_out[j - 1]
                freq_out[j - 1], freq_out[j] = freq_out[j], freq_out[j - 1]
                for k in range(num_cams):
                    whichcam_out[j - 1, k], whichcam_out[j, k] = (
                        whichcam_out[j, k],
                        whichcam_out[j - 1, k],
                    )

    num_valid = 0
    for i in range(n):
        if freq_out[i] != 0:
            num_valid += 1
    return num_valid


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
    _verts_x_buf = np.zeros(8, dtype=np.float64)
    verts_x: cython.double[:] = _verts_x_buf
    _verts_y_buf = np.zeros(8, dtype=np.float64)
    verts_y: cython.double[:] = _verts_y_buf
    _verts_z_buf = np.zeros(8, dtype=np.float64)
    verts_z: cython.double[:] = _verts_z_buf
    _dirs_x_buf = np.zeros(8, dtype=np.float64)
    dirs_x: cython.double[:] = _dirs_x_buf
    _dirs_y_buf = np.zeros(8, dtype=np.float64)
    dirs_y: cython.double[:] = _dirs_y_buf
    _dirs_z_buf = np.zeros(8, dtype=np.float64)
    dirs_z: cython.double[:] = _dirs_z_buf
    _valid_buf = np.zeros(8, dtype=np.int32)
    valid: cython.int[:] = _valid_buf
    _vi: cython.int

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


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def _trackcorr_particle_fast(
    h: cython.int,
    tid: cython.int,
    # Private thread-local buffers
    X_threads: cython.double[:, :, ::1],
    cpx_threads: cython.double[:, :],
    cpy_threads: cython.double[:, :],
    x2_cpx_threads: cython.double[:, :],
    x2_cpy_threads: cython.double[:, :],
    pp_threads: cython.double[:, :],
    assess_targ_threads: cython.double[:, :, ::1],
    assess_inds_threads: cython.int[:, :],
    assess_pp_threads: cython.double[:, :],
    assess_targ2_threads: cython.double[:, :, ::1],
    assess_inds2_threads: cython.int[:, :],
    pos_threads: cython.double[:, :],
    ftnr_buf1_threads: cython.int[:, :],
    freq_buf1_threads: cython.int[:, :],
    wc_buf1_threads: cython.int[:, :, ::1],
    ftnr_buf2_threads: cython.int[:, :],
    freq_buf2_threads: cython.int[:, :],
    wc_buf2_threads: cython.int[:, :, ::1],
    scratch_ray_threads: cython.double[:, :],
    pt_buf_threads: cython.double[:, :],
    # Thread-local added particle buffers
    thread_added_count_3: cython.int[:],
    thread_added_x_3: cython.double[:, :, ::1],
    thread_added_cand_3: cython.int[:, :, ::1],
    thread_added_count_2: cython.int[:],
    thread_added_h_2: cython.int[:, :],
    thread_added_x_2: cython.double[:, :, ::1],
    thread_added_cand_2: cython.int[:, :, ::1],
    thread_added_rr_2: cython.double[:, :],
    # Unpacked md_arr
    md0: cython.double[:],
    md1: cython.double[:],
    md2: cython.double[:],
    md3: cython.double[:],
    md4: cython.double[:],
    md5: cython.double[:],
    md6: cython.double[:],
    md7: cython.double[:],
    # Input arrays and parameters
    path_inlist_1: cython.int[:],
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_x_0: cython.double[:, ::1],
    num_cams: cython.int,
    mnr_arr: cython.int[:],
    cal_arr: cython.double[:, ::1],
    mo_arr: cython.double[:, ::1],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    corres_p_1: cython.int[:, ::1],
    targ_x_1: cython.double[:, ::1],
    targ_y_1: cython.double[:, ::1],
    targ_x_2: cython.double[:, ::1],
    targ_y_2: cython.double[:, ::1],
    targ_tnr_2: cython.int[:, ::1],
    num_targets_2: cython.int[:],
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    imx: cython.double,
    imy: cython.double,
    path_x_2: cython.double[:, ::1],
    targ_x_3: cython.double[:, ::1],
    targ_y_3: cython.double[:, ::1],
    targ_tnr_3: cython.int[:, ::1],
    num_targets_3: cython.int[:],
    path_x_3: cython.double[:, ::1],
    dacc: cython.double,
    dangle: cython.double,
    lmax: cython.double,
    path_decis_1: cython.double[:, ::1],
    path_linkdecis_1: cython.int[:, ::1],
    flatten_tol: cython.double,
    pix_x: cython.double,
    pix_y: cython.double,
    X_lay_0: cython.double,
    X_lay_1: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    Zmin_lay_0: cython.double,
    Zmax_lay_1: cython.double,
    add_flag: cython.int,
) -> cython.int:
    X: cython.double[:, ::1]
    cpx: cython.double[:]
    cpy: cython.double[:]
    x2_cpx: cython.double[:]
    x2_cpy: cython.double[:]
    _pp_mv: cython.double[:]
    _assess_targ: cython.double[:, ::1]
    _assess_inds: cython.int[:]
    _assess_pp: cython.double[:]
    _assess_targ2: cython.double[:, ::1]
    _assess_inds2: cython.int[:]
    _pos_mv: cython.double[:]
    _ftnr_buf1: cython.int[:]
    _freq_buf1: cython.int[:]
    _wc_buf1: cython.int[:, ::1]
    _ftnr_buf2: cython.int[:]
    _freq_buf2: cython.int[:]
    _wc_buf2: cython.int[:, ::1]
    scratch_ray: cython.double[:]
    pt_buf: cython.double[:]

    prev_h: cython.int
    j: cython.int
    has_mmlut: cython.int
    md_j: cython.double[:]
    w_nc: cython.int
    mm: cython.int
    ftnr_mm: cython.int
    wn_nc: cython.int
    kk: cython.int
    ftnr_kk: cython.int
    dp0: cython.double
    dp1: cython.double
    dp2: cython.double
    angle1: cython.double
    acc1: cython.double
    angle0: cython.double
    acc0: cython.double
    acc: cython.double
    angle: cython.double
    d13: cython.double
    d43: cython.double
    dl: cython.double
    rr: cython.double
    inlist: cython.int
    quali: cython.int
    in_volume: cython.int
    idx_add: cython.int
    ci: cython.int
    quali_f: cython.int
    d01: cython.double
    quali2: cython.int
    claimed_ok: cython.int

    if tid < 0:
        tid = 0
    elif tid >= X_threads.shape[0]:
        tid = X_threads.shape[0] - 1

    X = X_threads[tid]
    cpx = cpx_threads[tid]
    cpy = cpy_threads[tid]
    x2_cpx = x2_cpx_threads[tid]
    x2_cpy = x2_cpy_threads[tid]
    _pp_mv = pp_threads[tid]
    _assess_targ = assess_targ_threads[tid]
    _assess_inds = assess_inds_threads[tid]
    _assess_pp = assess_pp_threads[tid]
    _assess_targ2 = assess_targ2_threads[tid]
    _assess_inds2 = assess_inds2_threads[tid]
    _pos_mv = pos_threads[tid]
    _ftnr_buf1 = ftnr_buf1_threads[tid]
    _freq_buf1 = freq_buf1_threads[tid]
    _wc_buf1 = wc_buf1_threads[tid]
    _ftnr_buf2 = ftnr_buf2_threads[tid]
    _freq_buf2 = freq_buf2_threads[tid]
    _wc_buf2 = wc_buf2_threads[tid]
    scratch_ray = scratch_ray_threads[tid]
    pt_buf = pt_buf_threads[tid]

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
            has_mmlut = mnr_arr[j] > 0
            md_j = None
            if j == 0:
                md_j = md0
            elif j == 1:
                md_j = md1
            elif j == 2:
                md_j = md2
            elif j == 3:
                md_j = md3
            elif j == 4:
                md_j = md4
            elif j == 5:
                md_j = md5
            elif j == 6:
                md_j = md6
            elif j == 7:
                md_j = md7

            _point_to_pixel_out(
                X[2],
                cal_arr[j],
                md_j,
                mo_arr[j],
                mnr_arr[j],
                mnz_arr[j],
                mrw_arr[j],
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            cpx[j] = _pp_mv[0]
            cpy[j] = _pp_mv[1]
    else:
        X[2, 0] = X[1, 0]
        X[2, 1] = X[1, 1]
        X[2, 2] = X[1, 2]

        for j in range(num_cams):
            if corres_p_1[h, j] == CORRES_NONE_K:
                has_mmlut = mnr_arr[j] > 0
                md_j = None
                if j == 0:
                    md_j = md0
                elif j == 1:
                    md_j = md1
                elif j == 2:
                    md_j = md2
                elif j == 3:
                    md_j = md3
                elif j == 4:
                    md_j = md4
                elif j == 5:
                    md_j = md5
                elif j == 6:
                    md_j = md6
                elif j == 7:
                    md_j = md7

                _point_to_pixel_out(
                    X[2],
                    cal_arr[j],
                    md_j,
                    mo_arr[j],
                    mnr_arr[j],
                    mnz_arr[j],
                    mrw_arr[j],
                    has_mmlut,
                    imx_half,
                    imy_half,
                    inv_pix_x,
                    inv_pix_y,
                    chfield,
                    _pp_mv,
                )
                cpx[j] = _pp_mv[0]
                cpy[j] = _pp_mv[1]
            else:
                _ix = corres_p_1[h, j]
                cpx[j] = targ_x_1[j, _ix]
                cpy[j] = targ_y_1[j, _ix]

    # Save X[2] projections for later use by assess_new_position_fast
    for j in range(num_cams):
        x2_cpx[j] = cpx[j]
        x2_cpy[j] = cpy[j]

    # --- sorted_candidates for frame 2 ---
    w_nc = _sorted_candidates_fast_out_nogil(
        X[2],
        cpx,
        cpy,
        num_cams,
        MAX_CANDS_K,
        cal_arr,
        md0,
        md1,
        md2,
        md3,
        md4,
        md5,
        md6,
        md7,
        mo_arr,
        mnr_arr,
        mnz_arr,
        mrw_arr,
        targ_x_2,
        targ_y_2,
        targ_tnr_2,
        num_targets_2,
        dvxmin,
        dvxmax,
        dvymin,
        dvymax,
        dvzmin,
        dvzmax,
        imx_half,
        imy_half,
        inv_pix_x,
        inv_pix_y,
        chfield,
        imx,
        imy,
        TR_UNUSED_K,
        _ftnr_buf1,
        _freq_buf1,
        _wc_buf1,
        pt_buf,
        _pp_mv,
    )

    if w_nc == 0:
        return 0

    for mm in range(w_nc):
        ftnr_mm = _ftnr_buf1[mm]
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
            has_mmlut = mnr_arr[j] > 0
            md_j = None
            if j == 0:
                md_j = md0
            elif j == 1:
                md_j = md1
            elif j == 2:
                md_j = md2
            elif j == 3:
                md_j = md3
            elif j == 4:
                md_j = md4
            elif j == 5:
                md_j = md5
            elif j == 6:
                md_j = md6
            elif j == 7:
                md_j = md7

            _point_to_pixel_out(
                X[5],
                cal_arr[j],
                md_j,
                mo_arr[j],
                mnr_arr[j],
                mnz_arr[j],
                mrw_arr[j],
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            cpx[j] = _pp_mv[0]
            cpy[j] = _pp_mv[1]

        # --- sorted_candidates for frame 3 ---
        wn_nc = _sorted_candidates_fast_out_nogil(
            X[5],
            cpx,
            cpy,
            num_cams,
            MAX_CANDS_K,
            cal_arr,
            md0,
            md1,
            md2,
            md3,
            md4,
            md5,
            md6,
            md7,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
            targ_x_3,
            targ_y_3,
            targ_tnr_3,
            num_targets_3,
            dvxmin,
            dvxmax,
            dvymin,
            dvymax,
            dvzmin,
            dvzmax,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            imx,
            imy,
            TR_UNUSED_K,
            _ftnr_buf2,
            _freq_buf2,
            _wc_buf2,
            pt_buf,
            _pp_mv,
        )

        if wn_nc > 0:
            for kk in range(wn_nc):
                ftnr_kk = _ftnr_buf2[kk]
                X[4, 0] = path_x_3[ftnr_kk, 0]
                X[4, 1] = path_x_3[ftnr_kk, 1]
                X[4, 2] = path_x_3[ftnr_kk, 2]

                dp0 = X[4, 0] - X[3, 0]
                dp1 = X[4, 1] - X[3, 1]
                dp2 = X[4, 2] - X[3, 2]

                if (
                    dvxmin < dp0 < dvxmax
                    and dvymin < dp1 < dvymax
                    and dvzmin < dp2 < dvzmax
                ):
                    _angle_acc_out(
                        X[3, 0],
                        X[3, 1],
                        X[3, 2],
                        X[4, 0],
                        X[4, 1],
                        X[4, 2],
                        X[5, 0],
                        X[5, 1],
                        X[5, 2],
                        _pp_mv,
                    )
                    angle1 = _pp_mv[0]
                    acc1 = _pp_mv[1]
                    if prev_h >= 0:
                        _angle_acc_out(
                            X[1, 0],
                            X[1, 1],
                            X[1, 2],
                            X[2, 0],
                            X[2, 1],
                            X[2, 2],
                            X[3, 0],
                            X[3, 1],
                            X[3, 2],
                            _pp_mv,
                        )
                        angle0 = _pp_mv[0]
                        acc0 = _pp_mv[1]
                    else:
                        acc0 = acc1
                        angle0 = angle1

                    acc = (acc0 + acc1) * 0.5
                    angle = (angle0 + angle1) * 0.5
                    quali = _freq_buf2[kk] + _freq_buf1[mm]

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = c_sqrt(
                            (X[1, 0] - X[3, 0]) ** 2
                            + (X[1, 1] - X[3, 1]) ** 2
                            + (X[1, 2] - X[3, 2]) ** 2
                        )
                        d43 = c_sqrt(
                            (X[4, 0] - X[3, 0]) ** 2
                            + (X[4, 1] - X[3, 1]) ** 2
                            + (X[4, 2] - X[3, 2]) ** 2
                        )
                        dl = (d13 + d43) * 0.5
                        rr = (dl / lmax + acc / dacc + angle / dangle) / quali

                        inlist = path_inlist_1[h]
                        if inlist < POSI_K:
                            path_decis_1[h, inlist] = rr
                            path_linkdecis_1[h, inlist] = ftnr_mm
                            path_inlist_1[h] = inlist + 1

        # --- assess_new_position for X[5] in frame 3 ---
        quali = assess_new_position_fast_nogil(
            X[5],
            num_cams,
            ADD_PART_K,
            cal_arr,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
            targ_x_3,
            targ_y_3,
            targ_tnr_3,
            num_targets_3,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            int(imx),
            int(imy),
            pix_x,
            pix_y,
            flatten_tol,
            TR_UNUSED_K,
            COORD_UNUSED_K,
            cpx,
            cpy,
            _assess_targ,
            _assess_inds,
            _assess_pp,
        )

        if quali >= 2:
            in_volume = 0
            _point_position_out(_assess_targ, num_cams, cal_arr, _pos_mv, scratch_ray)
            X[4, 0] = _pos_mv[0]
            X[4, 1] = _pos_mv[1]
            X[4, 2] = _pos_mv[2]

            if (
                X_lay_0 < X[4, 0] < X_lay_1
                and ymin < X[4, 1] < ymax
                and Zmin_lay_0 < X[4, 2] < Zmax_lay_1
            ):
                in_volume = 1

            dp0 = X[3, 0] - X[4, 0]
            dp1 = X[3, 1] - X[4, 1]
            dp2 = X[3, 2] - X[4, 2]

            if (
                in_volume == 1
                and dvxmin < dp0 < dvxmax
                and dvymin < dp1 < dvymax
                and dvzmin < dp2 < dvzmax
            ):
                _angle_acc_out(
                    X[3, 0],
                    X[3, 1],
                    X[3, 2],
                    X[4, 0],
                    X[4, 1],
                    X[4, 2],
                    X[5, 0],
                    X[5, 1],
                    X[5, 2],
                    _pp_mv,
                )
                angle = _pp_mv[0]
                acc = _pp_mv[1]

                if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                    d13 = c_sqrt(
                        (X[1, 0] - X[3, 0]) ** 2
                        + (X[1, 1] - X[3, 1]) ** 2
                        + (X[1, 2] - X[3, 2]) ** 2
                    )
                    d43 = c_sqrt(
                        (X[4, 0] - X[3, 0]) ** 2
                        + (X[4, 1] - X[3, 1]) ** 2
                        + (X[4, 2] - X[3, 2]) ** 2
                    )
                    dl = (d13 + d43) * 0.5
                    rr = (dl / lmax + acc / dacc + angle / dangle) / (
                        quali + _freq_buf1[mm]
                    )

                    inlist = path_inlist_1[h]
                    if inlist < POSI_K:
                        path_decis_1[h, inlist] = rr
                        path_linkdecis_1[h, inlist] = ftnr_mm
                        path_inlist_1[h] = inlist + 1

                    if add_flag:
                        claimed_ok = 1
                        for ci in range(num_cams):
                            cand_idx = _assess_inds[ci]
                            if cand_idx != PT_UNUSED:
                                if not __sync_bool_compare_and_swap(
                                    cython.address(targ_tnr_3[ci, cand_idx]),
                                    TR_UNUSED_K,
                                    -100 - tid,
                                ):
                                    claimed_ok = 0
                                    break

                        if claimed_ok:
                            idx_add = thread_added_count_3[tid]
                            if idx_add < thread_added_x_3.shape[1]:
                                thread_added_x_3[tid, idx_add, 0] = X[4, 0]
                                thread_added_x_3[tid, idx_add, 1] = X[4, 1]
                                thread_added_x_3[tid, idx_add, 2] = X[4, 2]
                                for ci in range(num_cams):
                                    thread_added_cand_3[tid, idx_add, ci] = (
                                        _assess_inds[ci]
                                    )
                                thread_added_count_3[tid] = idx_add + 1
                            else:
                                for ci in range(num_cams):
                                    cand_idx = _assess_inds[ci]
                                    if cand_idx != PT_UNUSED:
                                        __sync_bool_compare_and_swap(
                                            cython.address(targ_tnr_3[ci, cand_idx]),
                                            -100 - tid,
                                            TR_UNUSED_K,
                                        )
                        else:
                            for ci in range(num_cams):
                                cand_idx = _assess_inds[ci]
                                if cand_idx != PT_UNUSED:
                                    __sync_bool_compare_and_swap(
                                        cython.address(targ_tnr_3[ci, cand_idx]),
                                        -100 - tid,
                                        TR_UNUSED_K,
                                    )

            in_volume = 0
        quali = 0

        # --- fallback: direct link if no links and prev >= 0 ---
        if path_inlist_1[h] == 0 and prev_h >= 0:
            dp0 = X[3, 0] - X[1, 0]
            dp1 = X[3, 1] - X[1, 1]
            dp2 = X[3, 2] - X[1, 2]

            if (
                dvxmin < dp0 < dvxmax
                and dvymin < dp1 < dvymax
                and dvzmin < dp2 < dvzmax
            ):
                _angle_acc_out(
                    X[1, 0],
                    X[1, 1],
                    X[1, 2],
                    X[2, 0],
                    X[2, 1],
                    X[2, 2],
                    X[3, 0],
                    X[3, 1],
                    X[3, 2],
                    _pp_mv,
                )
                angle = _pp_mv[0]
                acc = _pp_mv[1]

                if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                    quali_f = _freq_buf1[mm]
                    d13 = c_sqrt(
                        (X[1, 0] - X[3, 0]) ** 2
                        + (X[1, 1] - X[3, 1]) ** 2
                        + (X[1, 2] - X[3, 2]) ** 2
                    )
                    d01 = c_sqrt(
                        (X[0, 0] - X[1, 0]) ** 2
                        + (X[0, 1] - X[1, 1]) ** 2
                        + (X[0, 2] - X[1, 2]) ** 2
                    )
                    dl = (d13 + d01) * 0.5
                    rr = (dl / lmax + acc / dacc + angle / dangle) / quali_f

                    inlist = path_inlist_1[h]
                    if inlist < POSI_K:
                        path_decis_1[h, inlist] = rr
                        path_linkdecis_1[h, inlist] = ftnr_mm
                        path_inlist_1[h] = inlist + 1

    # --- add_particle to frame 2 if no links found ---
    if add_flag:
        if path_inlist_1[h] == 0 and prev_h >= 0:
            quali2 = assess_new_position_fast_nogil(
                X[2],
                num_cams,
                ADD_PART_K,
                cal_arr,
                mo_arr,
                mnr_arr,
                mnz_arr,
                mrw_arr,
                targ_x_2,
                targ_y_2,
                targ_tnr_2,
                num_targets_2,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                int(imx),
                int(imy),
                pix_x,
                pix_y,
                flatten_tol,
                TR_UNUSED_K,
                COORD_UNUSED_K,
                x2_cpx,
                x2_cpy,
                _assess_targ2,
                _assess_inds2,
                _assess_pp,
            )

            if quali2 >= 2:
                in_volume = 0
                _point_position_out(
                    _assess_targ2, num_cams, cal_arr, _pos_mv, scratch_ray
                )
                X[3, 0] = _pos_mv[0]
                X[3, 1] = _pos_mv[1]
                X[3, 2] = _pos_mv[2]

                if (
                    X_lay_0 < X[3, 0] < X_lay_1
                    and ymin < X[3, 1] < ymax
                    and Zmin_lay_0 < X[3, 2] < Zmax_lay_1
                ):
                    in_volume = 1

                dp0 = X[2, 0] - X[3, 0]
                dp1 = X[2, 1] - X[3, 1]
                dp2 = X[2, 2] - X[3, 2]

                if (
                    in_volume == 1
                    and dvxmin < dp0 < dvxmax
                    and dvymin < dp1 < dvymax
                    and dvzmin < dp2 < dvzmax
                ):
                    _angle_acc_out(
                        X[1, 0],
                        X[1, 1],
                        X[1, 2],
                        X[2, 0],
                        X[2, 1],
                        X[2, 2],
                        X[3, 0],
                        X[3, 1],
                        X[3, 2],
                        _pp_mv,
                    )
                    angle = _pp_mv[0]
                    acc = _pp_mv[1]

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = c_sqrt(
                            (X[1, 0] - X[3, 0]) ** 2
                            + (X[1, 1] - X[3, 1]) ** 2
                            + (X[1, 2] - X[3, 2]) ** 2
                        )
                        d01 = c_sqrt(
                            (X[0, 0] - X[1, 0]) ** 2
                            + (X[0, 1] - X[1, 1]) ** 2
                            + (X[0, 2] - X[1, 2]) ** 2
                        )
                        dl = (d13 + d01) * 0.5
                        rr = (dl / lmax + acc / dacc + angle / dangle) / quali2

                        claimed_ok = 1
                        for ci in range(num_cams):
                            cand_idx = _assess_inds2[ci]
                            if cand_idx != PT_UNUSED:
                                if not __sync_bool_compare_and_swap(
                                    cython.address(targ_tnr_2[ci, cand_idx]),
                                    TR_UNUSED_K,
                                    -100 - tid,
                                ):
                                    claimed_ok = 0
                                    break

                        if claimed_ok:
                            idx_add = thread_added_count_2[tid]
                            if idx_add < thread_added_x_2.shape[1]:
                                thread_added_h_2[tid, idx_add] = h
                                thread_added_x_2[tid, idx_add, 0] = X[3, 0]
                                thread_added_x_2[tid, idx_add, 1] = X[3, 1]
                                thread_added_x_2[tid, idx_add, 2] = X[3, 2]
                                thread_added_rr_2[tid, idx_add] = rr
                                for ci in range(num_cams):
                                    thread_added_cand_2[tid, idx_add, ci] = (
                                        _assess_inds2[ci]
                                    )
                                thread_added_count_2[tid] = idx_add + 1
                            else:
                                for ci in range(num_cams):
                                    cand_idx = _assess_inds2[ci]
                                    if cand_idx != PT_UNUSED:
                                        __sync_bool_compare_and_swap(
                                            cython.address(targ_tnr_2[ci, cand_idx]),
                                            -100 - tid,
                                            TR_UNUSED_K,
                                        )
                        else:
                            for ci in range(num_cams):
                                cand_idx = _assess_inds2[ci]
                                if cand_idx != PT_UNUSED:
                                    __sync_bool_compare_and_swap(
                                        cython.address(targ_tnr_2[ci, cand_idx]),
                                        -100 - tid,
                                        TR_UNUSED_K,
                                    )

                in_volume = 0

    return 0


def trackcorr_loop_fast(
    orig_parts_1: cython.int,
    # Frame 0 (prev — read only)
    path_x_0: cython.double[:, ::1],
    # Frame 1 (curr — read/write)
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:],
    path_decis_1: cython.double[:, ::1],
    path_linkdecis_1: cython.int[:, ::1],
    corres_p_1: cython.int[:, ::1],
    targ_x_1: cython.double[:, ::1],
    targ_y_1: cython.double[:, ::1],
    targ_tnr_1: cython.int[:, ::1],
    # Frame 2 (next — read/write)
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:],
    path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, ::1],
    path_linkdecis_2: cython.int[:, ::1],
    corres_p_2: cython.int[:, ::1],
    corres_nr_2: cython.int[:],
    targ_x_2: cython.double[:, ::1],
    targ_y_2: cython.double[:, ::1],
    targ_tnr_2: cython.int[:, ::1],
    num_targets_2: cython.int[:],
    num_parts_2: cython.int[:],
    # Frame 3 (next-next — read/write)
    path_x_3: cython.double[:, ::1],
    path_prev_3: cython.int[:],
    path_next_3: cython.int[:],
    path_inlist_3: cython.int[:],
    path_prio_3: cython.int[:],
    path_finaldecis_3: cython.double[:],
    path_decis_3: cython.double[:, ::1],
    path_linkdecis_3: cython.int[:, ::1],
    corres_p_3: cython.int[:, ::1],
    corres_nr_3: cython.int[:],
    targ_x_3: cython.double[:, ::1],
    targ_y_3: cython.double[:, ::1],
    targ_tnr_3: cython.int[:, ::1],
    num_targets_3: cython.int[:],
    num_parts_3: cython.int[:],
    # Calibration — pre-flattened arrays
    cal_arr: cython.double[:, ::1],
    md_arr: object,
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    # Tracking params
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    dacc: cython.double,
    dangle: cython.double,
    add_flag: cython.int,
    lmax: cython.double,
    # Volume bounds
    X_lay_0: cython.double,
    X_lay_1: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    Zmin_lay_0: cython.double,
    Zmax_lay_1: cython.double,
    # Pixel params
    num_cams: cython.int,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
    num_threads: cython.int = 1,
):
    """Full per-particle tracking loop + link resolution — single compiled entry.

    All internal calls (sorted_candidates, angle_acc, assess_new_position,
    point_position) are compiled with zero dispatch overhead.

    Args:
        num_parts_2, num_parts_3: (1,) int32 arrays — mutable particle counts.

    Returns:
        (count1, num_added) — number of links established and particles added.
    """
    count1: cython.int
    num_added: cython.int
    n_sc: cython.int
    h: cython.int
    j: cython.int
    mm: cython.int
    kk: cython.int
    prev_h: cython.int
    ftnr_mm: cython.int
    ftnr_kk: cython.int
    ki: cython.int
    ci: cython.int
    inlist: cython.int
    np2: cython.int
    np3: cython.int
    in_volume: cython.int
    quali: cython.int
    i: cython.int
    ti: cython.int
    cand: cython.int
    has_mmlut: cython.int
    px: cython.double
    py: cython.double
    dp0: cython.double
    dp1: cython.double
    dp2: cython.double
    angle1: cython.double
    acc1: cython.double
    angle0: cython.double
    acc0: cython.double
    acc: cython.double
    angle: cython.double
    rr: cython.double
    d13: cython.double = 0.0
    d43: cython.double = 0.0
    dl: cython.double = 0.0
    d01: cython.double = 0.0
    quali_f: cython.int
    tid: cython.int
    idx_add: cython.int
    quali2: cython.int
    cand_idx: cython.int
    max_threads_alloc: cython.int
    md_j: cython.double[:]

    # Thread-local private memoryviews are declared inside the prange loop directly.

    count1 = 0
    num_added = 0
    n_sc = num_cams * MAX_CANDS_K

    if num_threads < 1:
        num_threads = 1
    max_threads_alloc = num_threads
    if max_threads_alloc < 64:
        max_threads_alloc = 64

    # Unpack md_arr to individual memoryviews for GIL-free access
    dummy_empty = np.empty(0, dtype=np.float64)
    md0: cython.double[:] = dummy_empty
    md1: cython.double[:] = dummy_empty
    md2: cython.double[:] = dummy_empty
    md3: cython.double[:] = dummy_empty
    md4: cython.double[:] = dummy_empty
    md5: cython.double[:] = dummy_empty
    md6: cython.double[:] = dummy_empty
    md7: cython.double[:] = dummy_empty

    if num_cams > 0:
        md0 = md_arr[0]
    if num_cams > 1:
        md1 = md_arr[1]
    if num_cams > 2:
        md2 = md_arr[2]
    if num_cams > 3:
        md3 = md_arr[3]
    if num_cams > 4:
        md4 = md_arr[4]
    if num_cams > 5:
        md5 = md_arr[5]
    if num_cams > 6:
        md6 = md_arr[6]
    if num_cams > 7:
        md7 = md_arr[7]

    # Pre-allocated output buffers for _sorted_candidates_fast_out_nogil across threads
    _n_ftnr1_threads = np.empty((max_threads_alloc, n_sc), dtype=np.int32)
    _n_freq1_threads = np.empty((max_threads_alloc, n_sc), dtype=np.int32)
    _n_wc1_threads = np.empty((max_threads_alloc, n_sc, num_cams), dtype=np.int32)
    _n_ftnr2_threads = np.empty((max_threads_alloc, n_sc), dtype=np.int32)
    _n_freq2_threads = np.empty((max_threads_alloc, n_sc), dtype=np.int32)
    _n_wc2_threads = np.empty((max_threads_alloc, n_sc, num_cams), dtype=np.int32)

    ftnr_buf1_threads: cython.int[:, :] = _n_ftnr1_threads
    freq_buf1_threads: cython.int[:, :] = _n_freq1_threads
    wc_buf1_threads: cython.int[:, :, ::1] = _n_wc1_threads
    ftnr_buf2_threads: cython.int[:, :] = _n_ftnr2_threads
    freq_buf2_threads: cython.int[:, :] = _n_freq2_threads
    wc_buf2_threads: cython.int[:, :, ::1] = _n_wc2_threads

    _cpx_threads = np.empty((max_threads_alloc, num_cams), dtype=np.float64)
    _cpy_threads = np.empty((max_threads_alloc, num_cams), dtype=np.float64)
    _x2_cpx_threads = np.empty((max_threads_alloc, num_cams), dtype=np.float64)
    _x2_cpy_threads = np.empty((max_threads_alloc, num_cams), dtype=np.float64)
    _X_threads = np.zeros((max_threads_alloc, 6, 3), dtype=np.float64)
    _pp_threads = np.empty((max_threads_alloc, 2), dtype=np.float64)

    cpx_threads: cython.double[:, :] = _cpx_threads
    cpy_threads: cython.double[:, :] = _cpy_threads
    x2_cpx_threads: cython.double[:, :] = _x2_cpx_threads
    x2_cpy_threads: cython.double[:, :] = _x2_cpy_threads
    X_threads: cython.double[:, :, ::1] = _X_threads
    pp_threads: cython.double[:, :] = _pp_threads

    # Pre-allocated output buffers for assess_new_position_fast_nogil across threads
    _assess_targ_threads = np.full(
        (max_threads_alloc, num_cams, 2), COORD_UNUSED_K, dtype=np.float64
    )
    _assess_inds_threads = np.full(
        (max_threads_alloc, num_cams), PT_UNUSED, dtype=np.int32
    )
    _assess_pp_threads = np.empty((max_threads_alloc, 2), dtype=np.float64)
    _assess_targ2_threads = np.full(
        (max_threads_alloc, num_cams, 2), COORD_UNUSED_K, dtype=np.float64
    )
    _assess_inds2_threads = np.full(
        (max_threads_alloc, num_cams), PT_UNUSED, dtype=np.int32
    )

    assess_targ_threads: cython.double[:, :, ::1] = _assess_targ_threads
    assess_inds_threads: cython.int[:, :] = _assess_inds_threads
    assess_pp_threads: cython.double[:, :] = _assess_pp_threads
    assess_targ2_threads: cython.double[:, :, ::1] = _assess_targ2_threads
    assess_inds2_threads: cython.int[:, :] = _assess_inds2_threads

    # Pre-allocated output buffer for _point_position_out
    _pos_threads = np.empty((max_threads_alloc, 3), dtype=np.float64)
    pos_threads: cython.double[:, :] = _pos_threads
    _scratch_ray_threads = np.empty((max_threads_alloc, 6), dtype=np.float64)
    scratch_ray_threads: cython.double[:, :] = _scratch_ray_threads

    _pt_buf_threads = np.empty((max_threads_alloc, 3), dtype=np.float64)
    pt_buf_threads: cython.double[:, :] = _pt_buf_threads

    # Thread-local added particle buffers for safe post-addition
    max_cap3: cython.int = path_x_3.shape[0]
    max_cap2: cython.int = path_x_2.shape[0]

    _thread_added_count_3 = np.zeros(max_threads_alloc, dtype=np.int32)
    _thread_added_x_3 = np.empty((max_threads_alloc, max_cap3, 3), dtype=np.float64)
    _thread_added_cand_3 = np.empty(
        (max_threads_alloc, max_cap3, num_cams), dtype=np.int32
    )

    thread_added_count_3: cython.int[:] = _thread_added_count_3
    thread_added_x_3: cython.double[:, :, ::1] = _thread_added_x_3
    thread_added_cand_3: cython.int[:, :, ::1] = _thread_added_cand_3

    _thread_added_count_2 = np.zeros(max_threads_alloc, dtype=np.int32)
    _thread_added_h_2 = np.empty((max_threads_alloc, max_cap2), dtype=np.int32)
    _thread_added_x_2 = np.empty((max_threads_alloc, max_cap2, 3), dtype=np.float64)
    _thread_added_cand_2 = np.empty(
        (max_threads_alloc, max_cap2, num_cams), dtype=np.int32
    )
    _thread_added_rr_2 = np.empty((max_threads_alloc, max_cap2), dtype=np.float64)

    thread_added_count_2: cython.int[:] = _thread_added_count_2
    thread_added_h_2: cython.int[:, :] = _thread_added_h_2
    thread_added_x_2: cython.double[:, :, ::1] = _thread_added_x_2
    thread_added_cand_2: cython.int[:, :, ::1] = _thread_added_cand_2
    thread_added_rr_2: cython.double[:, :] = _thread_added_rr_2

    # Parallel loop over particles
    for h in prange(
        orig_parts_1, nogil=True, schedule="guided", num_threads=num_threads
    ):
        tid = threadid()

        _trackcorr_particle_fast(
            h,
            tid,
            X_threads,
            cpx_threads,
            cpy_threads,
            x2_cpx_threads,
            x2_cpy_threads,
            pp_threads,
            assess_targ_threads,
            assess_inds_threads,
            assess_pp_threads,
            assess_targ2_threads,
            assess_inds2_threads,
            pos_threads,
            ftnr_buf1_threads,
            freq_buf1_threads,
            wc_buf1_threads,
            ftnr_buf2_threads,
            freq_buf2_threads,
            wc_buf2_threads,
            scratch_ray_threads,
            pt_buf_threads,
            thread_added_count_3,
            thread_added_x_3,
            thread_added_cand_3,
            thread_added_count_2,
            thread_added_h_2,
            thread_added_x_2,
            thread_added_cand_2,
            thread_added_rr_2,
            md0,
            md1,
            md2,
            md3,
            md4,
            md5,
            md6,
            md7,
            path_inlist_1,
            path_x_1,
            path_prev_1,
            path_x_0,
            num_cams,
            mnr_arr,
            cal_arr,
            mo_arr,
            mnz_arr,
            mrw_arr,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            corres_p_1,
            targ_x_1,
            targ_y_1,
            targ_x_2,
            targ_y_2,
            targ_tnr_2,
            num_targets_2,
            dvxmin,
            dvxmax,
            dvymin,
            dvymax,
            dvzmin,
            dvzmax,
            imx,
            imy,
            path_x_2,
            targ_x_3,
            targ_y_3,
            targ_tnr_3,
            num_targets_3,
            path_x_3,
            dacc,
            dangle,
            lmax,
            path_decis_1,
            path_linkdecis_1,
            flatten_tol,
            pix_x,
            pix_y,
            X_lay_0,
            X_lay_1,
            ymin,
            ymax,
            Zmin_lay_0,
            Zmax_lay_1,
            add_flag,
        )

    # Sequential post-loop actual appending to global arrays to guarantee deterministic ordering and no race conditions
    for tid in range(max_threads_alloc):
        for idx_add in range(thread_added_count_3[tid]):
            np3 = num_parts_3[0]
            if np3 < path_x_3.shape[0]:
                path_x_3[np3, 0] = thread_added_x_3[tid, idx_add, 0]
                path_x_3[np3, 1] = thread_added_x_3[tid, idx_add, 1]
                path_x_3[np3, 2] = thread_added_x_3[tid, idx_add, 2]
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
                    cand_idx = thread_added_cand_3[tid, idx_add, ci]
                    if cand_idx != PT_UNUSED:
                        if 0 <= cand_idx < targ_tnr_3.shape[1]:
                            targ_tnr_3[ci, cand_idx] = np3
                            corres_p_3[np3, ci] = cand_idx
                num_parts_3[0] = np3 + 1
                num_added += 1

    for tid in range(max_threads_alloc):
        for idx_add in range(thread_added_count_2[tid]):
            h = thread_added_h_2[tid, idx_add]
            np2 = num_parts_2[0]
            if np2 < path_x_2.shape[0]:
                inlist = path_inlist_1[h]
                if inlist < POSI_K:
                    path_decis_1[h, inlist] = thread_added_rr_2[tid, idx_add]
                    path_linkdecis_1[h, inlist] = np2
                    path_inlist_1[h] = inlist + 1

                path_x_2[np2, 0] = thread_added_x_2[tid, idx_add, 0]
                path_x_2[np2, 1] = thread_added_x_2[tid, idx_add, 1]
                path_x_2[np2, 2] = thread_added_x_2[tid, idx_add, 2]
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
                    cand_idx = thread_added_cand_2[tid, idx_add, ci]
                    if cand_idx != PT_UNUSED:
                        if 0 <= cand_idx < targ_tnr_2.shape[1]:
                            targ_tnr_2[ci, cand_idx] = np2
                            corres_p_2[np2, ci] = cand_idx
                num_parts_2[0] = np2 + 1
                num_added += 1

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
                            path_decis_1[h, i + 1],
                            path_decis_1[h, i],
                        )
                        path_linkdecis_1[h, i], path_linkdecis_1[h, i + 1] = (
                            path_linkdecis_1[h, i + 1],
                            path_linkdecis_1[h, i],
                        )
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
    path_x_0: cython.double[:, ::1],
    # Frame 1 (current — read/write)
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:],
    path_decis_1: cython.double[:, ::1],
    path_linkdecis_1: cython.int[:, ::1],
    # Frame 2 (backward/prev in time — read/write)
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int[:],
    targ_x_2: cython.double[:, ::1],
    targ_y_2: cython.double[:, ::1],
    targ_tnr_2: cython.int[:, ::1],
    num_targets_2: cython.int[:],
    corres_p_2: cython.int[:, ::1],
    corres_nr_2: cython.int[:],
    path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:],
    path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, ::1],
    path_linkdecis_2: cython.int[:, ::1],
    # Frame 3 (further backward — read only, for extra angle check)
    path_x_3: cython.double[:, ::1],
    path_prev_3: cython.int[:],
    # Calibration — pre-flattened arrays
    cal_arr: cython.double[:, ::1],
    md_arr: object,
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    # Tracking params
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    dacc: cython.double,
    dangle: cython.double,
    add_flag: cython.int,
    lmax: cython.double,
    # Volume bounds
    X_lay_0: cython.double,
    X_lay_1: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    Zmin_lay_0: cython.double,
    Zmax_lay_1: cython.double,
    # Pixel params
    num_cams: cython.int,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
):
    """Backward tracking loop — compiled compiled.

    For each particle in buf[1] with next >= 0 and prev == -1,
    searches for candidates in buf[2] (backward in time).
    """
    count1: cython.int
    num_added: cython.int
    h: cython.int
    i: cython.int
    j: cython.int
    ki: cython.int
    ci: cython.int
    next_h: cython.int
    prev_h: cython.int
    ftnr_i: cython.int
    inlist: cython.int
    best_cand: cython.int
    has_mmlut: cython.int
    prev_of_cand: cython.int
    np2: cython.int
    in_volume: cython.int
    quali: cython.int
    ti: cython.int
    px: cython.double
    py: cython.double
    dp0: cython.double
    dp1: cython.double
    dp2: cython.double
    angle: cython.double
    acc: cython.double
    rr: cython.double
    d13: cython.double = 0.0
    d01: cython.double = 0.0
    dl: cython.double = 0.0
    idx: cython.int
    flag: cython.bint
    count1 = 0
    num_added = 0
    n_sc = num_cams * MAX_CANDS_K
    _n_ftnr = np.empty(n_sc, dtype=np.int32)
    _n_freq = np.empty(n_sc, dtype=np.int32)
    _n_wc = np.empty((n_sc, num_cams), dtype=np.int32)
    _ftnr_buf: cython.int[:] = _n_ftnr
    _freq_buf: cython.int[:] = _n_freq
    _wc_buf: cython.int[:, ::1] = _n_wc
    _cpx = np.empty(num_cams, dtype=np.float64)
    _cpy = np.empty(num_cams, dtype=np.float64)
    _X = np.zeros((6, 3), dtype=np.float64)
    cpx: cython.double[:] = _cpx
    cpy: cython.double[:] = _cpy
    X: cython.double[:, ::1] = _X
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp

    # Pre-allocated output buffers for assess_new_position_fast
    _assess_targ = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)
    _assess_pp = np.empty(2, dtype=np.float64)

    _pos_buf = np.zeros(3, dtype=np.float64)
    _pos_mv: cython.double[:] = _pos_buf
    _scratch_ray = np.zeros(6, dtype=np.float64)
    scratch_ray: cython.double[:] = _scratch_ray

    # cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr pre-flattened by caller
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
            has_mmlut = mnr_arr[j] > 0
            _point_to_pixel_out(
                X[2],
                cal_arr[j],
                md_arr[j],
                mo_arr[j],
                mnr_arr[j],
                mnz_arr[j],
                mrw_arr[j],
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            cpx[j] = _pp_mv[0]
            cpy[j] = _pp_mv[1]

        w_nc = _sorted_candidates_fast_out(
            X[2],
            cpx,
            cpy,
            num_cams,
            MAX_CANDS_K,
            cal_arr,
            md_arr,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
            targ_x_2,
            targ_y_2,
            targ_tnr_2,
            num_targets_2,
            dvxmin,
            dvxmax,
            dvymin,
            dvymax,
            dvzmin,
            dvzmax,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            imx,
            imy,
            TR_UNUSED_K,
            _ftnr_buf,
            _freq_buf,
            _wc_buf,
        )

        if w_nc > 0:
            i = 0
            while i < w_nc:
                ftnr_i = _ftnr_buf[i]
                X[3, 0] = path_x_2[ftnr_i, 0]
                X[3, 1] = path_x_2[ftnr_i, 1]
                X[3, 2] = path_x_2[ftnr_i, 2]

                dp0 = X[1, 0] - X[3, 0]
                dp1 = X[1, 1] - X[3, 1]
                dp2 = X[1, 2] - X[3, 2]

                if (
                    dvxmin < dp0 < dvxmax
                    and dvymin < dp1 < dvymax
                    and dvzmin < dp2 < dvzmax
                ):
                    _angle_acc_out(
                        X[1, 0],
                        X[1, 1],
                        X[1, 2],
                        X[2, 0],
                        X[2, 1],
                        X[2, 2],
                        X[3, 0],
                        X[3, 1],
                        X[3, 2],
                        _pp_mv,
                    )
                    angle = _pp_mv[0]
                    acc = _pp_mv[1]

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = c_sqrt(
                            (X[1, 0] - X[3, 0]) ** 2
                            + (X[1, 1] - X[3, 1]) ** 2
                            + (X[1, 2] - X[3, 2]) ** 2
                        )
                    d01 = c_sqrt(
                        (X[0, 0] - X[1, 0]) ** 2
                        + (X[0, 1] - X[1, 1]) ** 2
                        + (X[0, 2] - X[1, 2]) ** 2
                    )
                    dl = (d13 + d01) * 0.5
                    quali = _freq_buf[i]
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
                    X[2],
                    num_cams,
                    ADD_PART_K,
                    cal_arr,
                    md_arr,
                    mo_arr,
                    mnr_arr,
                    mnz_arr,
                    mrw_arr,
                    targ_x_2,
                    targ_y_2,
                    targ_tnr_2,
                    num_targets_2,
                    imx_half,
                    imy_half,
                    inv_pix_x,
                    inv_pix_y,
                    chfield,
                    int(imx),
                    int(imy),
                    pix_x,
                    pix_y,
                    flatten_tol,
                    TR_UNUSED_K,
                    COORD_UNUSED_K,
                    use_proj=True,
                    proj_x=cpx,
                    proj_y=cpy,
                    targ_pos_out=_assess_targ,
                    cand_inds_out=_assess_inds,
                    scratch=_assess_pp,
                )

                if quali >= 2:
                    in_volume = 0
                    _point_position_out(
                        targ_pos, num_cams, cal_arr, _pos_mv, scratch_ray
                    )
                    X[3, 0] = _pos_buf[0]
                    X[3, 1] = _pos_buf[1]
                    X[3, 2] = _pos_buf[2]

                    if (
                        X_lay_0 < X[3, 0] < X_lay_1
                        and ymin < X[3, 1] < ymax
                        and Zmin_lay_0 < X[3, 2] < Zmax_lay_1
                    ):
                        in_volume = 1

                    dp0 = X[1, 0] - X[3, 0]
                    dp1 = X[1, 1] - X[3, 1]
                    dp2 = X[1, 2] - X[3, 2]

                    if (
                        in_volume == 1
                        and dvxmin < dp0 < dvxmax
                        and dvymin < dp1 < dvymax
                        and dvzmin < dp2 < dvzmax
                    ):
                        _angle_acc_out(
                            X[1, 0],
                            X[1, 1],
                            X[1, 2],
                            X[2, 0],
                            X[2, 1],
                            X[2, 2],
                            X[3, 0],
                            X[3, 1],
                            X[3, 2],
                            _pp_mv,
                        )
                        angle = _pp_mv[0]
                        acc = _pp_mv[1]

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = c_sqrt(
                                (X[1, 0] - X[3, 0]) ** 2
                                + (X[1, 1] - X[3, 1]) ** 2
                                + (X[1, 2] - X[3, 2]) ** 2
                            )
                            d01 = c_sqrt(
                                (X[0, 0] - X[1, 0]) ** 2
                                + (X[0, 1] - X[1, 1]) ** 2
                                + (X[0, 2] - X[1, 2]) ** 2
                            )
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
                                    targ_tnr_2[ci, idx] = np2
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
                            path_decis_1[h, i + 1],
                            path_decis_1[h, i],
                        )
                        path_linkdecis_1[h, i], path_linkdecis_1[h, i + 1] = (
                            path_linkdecis_1[h, i + 1],
                            path_linkdecis_1[h, i],
                        )
                        flag = True

    # Link resolution — trackback style
    for h in range(num_parts_1):
        if path_inlist_1[h] > 0:
            best_cand = path_linkdecis_1[h, 0]

            # Case 1: candidate has no links at all
            if (
                path_prev_2[best_cand] == PREV_NONE_K
                and path_next_2[best_cand] == NEXT_NONE_K
            ):
                path_finaldecis_1[h] = path_decis_1[h, 0]
                path_prev_1[h] = best_cand
                path_next_2[best_cand] = h
                num_added += 1

            # Case 2: candidate has a prev but no next — extra angle check
            elif (
                path_prev_2[best_cand] != PREV_NONE_K
                and path_next_2[best_cand] == NEXT_NONE_K
            ):
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

                _angle_acc_out(
                    X[3, 0],
                    X[3, 1],
                    X[3, 2],
                    X[4, 0],
                    X[4, 1],
                    X[4, 2],
                    X[5, 0],
                    X[5, 1],
                    X[5, 2],
                    _pp_mv,
                )
                angle = _pp_mv[0]
                acc = _pp_mv[1]

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
def _find_closest_in_3d(
    path_x_2: cython.double[:, ::1],
    np2: cython.int,
    pred_x: cython.double,
    pred_y: cython.double,
    pred_z: cython.double,
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
    cand_inds: cython.int[:],
    cand_dists: cython.double[:],
):
    """Find up to max_cands closest candidates by distance within a 3D box.

    Maintains a running top-N by distance, matching candsearch_in_pix logic.
    Writes into pre-allocated cand_inds/cand_dists arrays.
    Returns the number of candidates found.
    """
    s: cython.int
    k: cython.int
    slot: cython.int
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    d: cython.double
    n_found = 0
    for s in range(max_cands):
        cand_inds[s] = -1
        cand_dists[s] = 1e20

    for k in range(np2):
        ddx = path_x_2[k, 0] - pred_x
        ddy = path_x_2[k, 1] - pred_y
        ddz = path_x_2[k, 2] - pred_z
        if abs(ddx) < dx and abs(ddy) < dy and abs(ddz) < dz:
            d = c_sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
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
    path_x_0: cython.double[:, ::1],
    path_prev_0: cython.int[:],
    num_parts_0: cython.int,
    # Frame 1 (curr) — read/write
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    num_parts_1: cython.int,
    # Frame 2 (next) — read/write
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int,
    # Tracking params
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
):
    """Full track3d loop (3 levels) — single compiled entry.

    Level 1: particles with previous links — predict from velocity.
    Level 2: no prev link — average velocity from neighbors.
    Level 3: no prev link, no neighbor info — use current position.

    Returns count1 (number of links established).
    """
    count1: cython.int
    np2: cython.int
    i: cython.int
    j: cython.int
    ci: cython.int
    prev_idx: cython.int
    pred_x: cython.double
    pred_y: cython.double
    pred_z: cython.double
    n_cands: cython.int
    n_decis: cython.int
    k: cython.int
    d0: cython.double
    d1: cython.double
    d2: cython.double
    acc: cython.double
    si: cython.int
    sj: cython.int
    vel_x: cython.double
    vel_y: cython.double
    vel_z: cython.double
    nvel: cython.int
    cx: cython.double
    cy: cython.double
    cz: cython.double
    pj: cython.int
    inv_nvel: cython.double
    count1 = 0
    np2 = num_parts_2
    _cand_inds = np.empty(max_cands, dtype=np.int32)
    _cand_dists = np.empty(max_cands, dtype=np.float64)
    _decis_vals = np.empty(max_cands, dtype=np.float64)
    _decis_inds = np.empty(max_cands, dtype=np.int32)

    cand_inds: cython.int[:] = _cand_inds
    cand_dists: cython.double[:] = _cand_dists
    decis_vals: cython.double[:] = _decis_vals
    decis_inds: cython.int[:] = _decis_inds

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

        n_cands = _find_closest_in_3d(
            path_x_2,
            np2,
            pred_x,
            pred_y,
            pred_z,
            dx,
            dy,
            dz,
            max_cands,
            cand_inds,
            cand_dists,
        )
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = path_x_1[i, 0] - 2.0 * path_x_2[k, 0] + path_x_0[prev_idx, 0]
            d1 = path_x_1[i, 1] - 2.0 * path_x_2[k, 1] + path_x_0[prev_idx, 1]
            d2 = path_x_1[i, 2] - 2.0 * path_x_2[k, 2] + path_x_0[prev_idx, 2]
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = (
                            decis_vals[sj],
                            decis_vals[sj - 1],
                        )
                        decis_inds[sj - 1], decis_inds[sj] = (
                            decis_inds[sj],
                            decis_inds[sj - 1],
                        )

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

        vel_x = 0.0
        vel_y = 0.0
        vel_z = 0.0
        nvel = 0
        cx = path_x_1[i, 0]
        cy = path_x_1[i, 1]
        cz = path_x_1[i, 2]

        for j in range(orig_parts):
            if j == i:
                continue
            if (
                abs(path_x_1[j, 0] - cx) < dx
                and abs(path_x_1[j, 1] - cy) < dy
                and abs(path_x_1[j, 2] - cz) < dz
                and path_prev_1[j] >= 0
            ):
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

        n_cands = _find_closest_in_3d(
            path_x_2,
            np2,
            pred_x,
            pred_y,
            pred_z,
            dx,
            dy,
            dz,
            max_cands,
            cand_inds,
            cand_dists,
        )
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = cx - 2.0 * path_x_2[k, 0] + pred_x
            d1 = cy - 2.0 * path_x_2[k, 1] + pred_y
            d2 = cz - 2.0 * path_x_2[k, 2] + pred_z
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = (
                            decis_vals[sj],
                            decis_vals[sj - 1],
                        )
                        decis_inds[sj - 1], decis_inds[sj] = (
                            decis_inds[sj],
                            decis_inds[sj - 1],
                        )

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

        n_cands = _find_closest_in_3d(
            path_x_2,
            np2,
            pred_x,
            pred_y,
            pred_z,
            dx,
            dy,
            dz,
            max_cands,
            cand_inds,
            cand_dists,
        )
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = pred_x - 2.0 * path_x_2[k, 0] + pred_x
            d1 = pred_y - 2.0 * path_x_2[k, 1] + pred_y
            d2 = pred_z - 2.0 * path_x_2[k, 2] + pred_z
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = (
                            decis_vals[sj],
                            decis_vals[sj - 1],
                        )
                        decis_inds[sj - 1], decis_inds[sj] = (
                            decis_inds[sj],
                            decis_inds[sj - 1],
                        )

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
