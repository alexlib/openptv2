"""Per-camera pixel-space math: projection, candidate search, multimedia refraction."""

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


@cython.ccall
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
    out_indices: cython.int[:],
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
    quader_buf: cython.double[:]
    xr: cython.double[:]
    xl: cython.double[:]
    yd: cython.double[:]
    yu: cython.double[:]
    with cython.gil:
        _quader_buf = np.zeros(24, dtype=np.float64)
        quader_buf = _quader_buf
        _xr_buf = np.zeros(8, dtype=np.float64)
        xr = _xr_buf
        _xl_buf = np.zeros(8, dtype=np.float64)
        xl = _xl_buf
        _yd_buf = np.zeros(8, dtype=np.float64)
        yd = _yd_buf
        _yu_buf = np.zeros(8, dtype=np.float64)
        yu = _yu_buf

    n = num_cams * max_cands

    # --- searchquader inlined ---
    px = center[0]
    py = center[1]
    pz = center[2]
    for pt in range(8):
        quader_buf[pt * 3 + 0] = px + (dvxmax if pt & 1 else dvxmin)
        quader_buf[pt * 3 + 1] = py + (dvymax if pt & 2 else dvymin)
        quader_buf[pt * 3 + 2] = pz + (dvzmax if pt & 4 else dvzmin)

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
    cands_buf: cython.int[:]
    with cython.gil:
        _cands_buf = np.zeros(4, dtype=np.int32)
        cands_buf = _cands_buf

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
                ftnr_out[base + ci] = targ_tnr[cam, idx]

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


@cython.ccall
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
