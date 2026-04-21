"""Numba-compiled kernels for the tracking hot path.

These are JIT-compiled versions of _point_to_pixel_packed and
multimed_r_nlay_iterative. When Numba is unavailable, plain Python
fallbacks are used transparently.
"""

import math
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

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


def pack_cal_array(cal, mm):
    """Pack calibration into a flat float64 array for Numba kernels."""
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


def pack_mmlut(cal):
    """Pack mmlut into Numba-friendly arrays.

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


@njit(cache=True)
def _multimed_r_nlay_1layer(pos_x, pos_y, pos_z,
                             ext_x0, ext_y0, ext_z0,
                             mm_n1, mm_n2_0, mm_n3, mm_d0):
    """Single-layer iterative radial shift — Numba-compiled."""
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


@njit(cache=True)
def point_to_pixel_jit(pos, cal, mmlut_data, mmlut_origin,
                       mmlut_nr, mmlut_nz, mmlut_rw,
                       imx_half, imy_half, inv_pix_x, inv_pix_y, chfield):
    """Project 3D position to pixel coordinates — Numba-compiled.

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

@njit(cache=True)
def candsearch_in_pix_jit(targ_x, targ_y, targ_tnr, num_targets,
                           cent_x, cent_y, dl, dr, du, dd,
                           imx, imy, tr_unused):
    """Find up to 4 closest candidates in pixel search area — Numba-compiled.

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


@njit(cache=True)
def candsearch_in_pix_rest_jit(targ_x, targ_y, targ_tnr, num_targets,
                                cent_x, cent_y, dl, dr, du, dd,
                                imx, imy, tr_unused):
    """Find closest unused candidate — Numba-compiled.

    Returns:
        (index, count) — index of closest candidate with tnr==TR_UNUSED, count (0 or 1).
    """
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


@njit(cache=True)
def searchquader_jit(point, quader, num_cams, cal_arrays, mmlut_datas,
                     mmlut_origins, mmlut_nrs, mmlut_nzs, mmlut_rws,
                     imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
                     imx, imy):
    """Compute search area for all cameras — Numba-compiled.

    Projects point + 8 corner points through all cameras in a single JIT call,
    eliminating per-projection Python→Numba dispatch overhead.

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

        cx, cy = point_to_pixel_jit(point, cal, md, mo, mnr, mnz, mrw,
                                    imx_half, imy_half, inv_pix_x, inv_pix_y, chfield)

        for pt in range(8):
            corner_x, corner_y = point_to_pixel_jit(
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


@njit(cache=True)
def sort_candidates_by_freq_jit(ftnr, freq, whichcam, n, num_cams, max_cands):
    """Sort candidates by frequency — Numba-compiled, matches C algorithm.

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


@njit(cache=True)
def angle_acc_jit(start_x, start_y, start_z, pred_x, pred_y, pred_z,
                  cand_x, cand_y, cand_z):
    """Compute angle and acceleration between predicted and candidate — Numba-compiled."""
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
