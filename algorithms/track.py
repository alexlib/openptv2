"""Tracking algorithms — Python translation of lib/src/track.c."""

import math
import numpy as np

from .constants import (
    MAX_CANDS, PT_UNUSED, TR_UNUSED, CORRES_NONE, PREV_NONE, NEXT_NONE,
    COORD_UNUSED, TR_BUFSPACE, TR_MAX_CAMS, ADD_PART,
)
from .tracking_frame_buf import register_link_candidate, reset_links
from .multimed import (
    multimed_nlay as _multimed_nlay,
    multimed_r_nlay_iterative as _multimed_r_nlay_iterative,
)
from .track_kernels import (
    pack_cal_array as _pack_cal_array,
    pack_mmlut as _pack_mmlut,
    point_to_pixel_jit as _point_to_pixel_jit,
    candsearch_in_pix_jit as _candsearch_in_pix_jit,
    candsearch_in_pix_rest_jit as _candsearch_in_pix_rest_jit,
    searchquader_jit as _searchquader_jit,
    sort_candidates_by_freq_jit as _sort_candidates_by_freq_jit,
    sorted_candidates_jit as _sorted_candidates_jit,
    point_position_jit as _point_position_jit,
    trackcorr_loop_jit as _trackcorr_loop_jit,
    trackback_loop_jit as _trackback_loop_jit,
    HAS_NUMBA,
)

Foundpix_dtype = np.dtype([
    ("ftnr", np.int32),
    ("freq", np.int32),
    ("whichcam", np.int32, (4,)),
])


def _vec3_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pack_cams_jit(cals, mm):
    """Pack all cameras for JIT: returns (cal_arrays, mmlut_tuples)."""
    cal_arrays = [_pack_cal_array(c, mm) for c in cals]
    mmlut_tuples = [_pack_mmlut(c) for c in cals]
    return cal_arrays, mmlut_tuples


def _pack_cams_jit_tuples(jit_cals, jit_mmluts):
    """Convert lists to tuples for Numba searchquader_jit."""
    return (
        tuple(jit_cals),
        tuple(m[0] for m in jit_mmluts),
        tuple(m[1] for m in jit_mmluts),
        tuple(m[2] for m in jit_mmluts),
        tuple(m[3] for m in jit_mmluts),
        tuple(m[4] for m in jit_mmluts),
    )


def _ptp_jit(pos, cal_arr, mmlut_tup, imx_half, imy_half, inv_pix_x, inv_pix_y, chfield):
    """Call the JIT kernel with pre-packed arrays."""
    return _point_to_pixel_jit(
        pos, cal_arr, mmlut_tup[0], mmlut_tup[1],
        mmlut_tup[2], mmlut_tup[3], mmlut_tup[4],
        imx_half, imy_half, inv_pix_x, inv_pix_y, chfield,
    )


def _pack_cal(cal, mm):
    """Pre-extract calibration fields into a tuple for fast access."""
    ext = cal.ext_par; ip = cal.int_par; gp = cal.glass_par; ap = cal.added_par
    gx = gp.vec_x; gy = gp.vec_y; gz = gp.vec_z
    dist_o_glas = math.sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog = 1.0 / dist_o_glas
    mmlut = cal.mmlut
    mmlut_data = mmlut.data
    return (
        ext.x0, ext.y0, ext.z0,
        ext.dm[0, 0], ext.dm[1, 0], ext.dm[2, 0],
        ext.dm[0, 1], ext.dm[1, 1], ext.dm[2, 1],
        ext.dm[0, 2], ext.dm[1, 2], ext.dm[2, 2],
        ip.cc, ip.xh, ip.yh,
        gx, gy, gz, dist_o_glas, inv_dog,
        mm.n1, mm.n2[0], mm.n3, mm.d[0],
        ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she,
        mmlut_data,
        mmlut.origin if mmlut_data is not None else None,
        mmlut.nr if mmlut_data is not None else 0,
        mmlut.nz if mmlut_data is not None else 0,
        mmlut.rw if mmlut_data is not None else 0,
    )


def _point_to_pixel_packed(pos, pc, imx_half, imy_half, inv_pix_x, inv_pix_y, chfield):
    """Project 3D position to pixel coordinates using pre-packed calibration."""
    pos0 = float(pos[0]); pos1 = float(pos[1]); pos2 = float(pos[2])

    (ext_x0, ext_y0, ext_z0,
     dm00, dm10, dm20, dm01, dm11, dm21, dm02, dm12, dm22,
     int_cc, xh, yh,
     gx, gy, gz, dist_o_glas, inv_dog,
     mm_n1, mm_n2_0, mm_n3, mm_d0,
     k1, k2, k3, p1, p2, scx, she,
     mmlut_data, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw) = pc

    dot_cam = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
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

    # === mmlut lookup + multimed_nlay (inlined) ===
    radial_shift = 1.0
    if mmlut_data is not None:
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
                mmf = (
                    mmlut_data[v0] * (1 - sr) * (1 - sz)
                    + mmlut_data[v0 + 1] * (1 - sr) * sz
                    + mmlut_data[v0 + mmlut_nz] * sr * (1 - sz)
                    + mmlut_data[v3] * sr * sz
                )
                if mmf > 0:
                    radial_shift = mmf
    if radial_shift == 1.0:
        radial_shift = _multimed_r_nlay_iterative(
            pos_t_0, 0.0, pos_t_2, 0.0, 0.0, ext_t_z0,
            mm_n1, mm_n2_0, mm_n3, mm_d0,
        )
    X_t = pos_t_0 * radial_shift

    # === back_trans_point (inlined) ===
    s_z = -pos_t_2 * inv_dog
    bx = ag_x - gx * s_z
    by = ag_y - gy * s_z
    bz = ag_z - gz * s_z

    if pos_t_0 > 0:
        s_x = -X_t / pos_t_0
        bx -= tmp_x * s_x
        by -= tmp_y * s_x
        bz -= tmp_z * s_x

    # === perspective projection ===
    dx = bx - ext_x0
    dy = by - ext_y0
    dz = bz - ext_z0

    deno = dm02 * dx + dm12 * dy + dm22 * dz
    x = -int_cc * (dm00 * dx + dm10 * dy + dm20 * dz) / deno
    y = -int_cc * (dm01 * dx + dm11 * dy + dm21 * dz) / deno

    # === flat_to_dist + distort_brown_affin (inlined) ===
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
        xd = x * radial_factor + p1 * (r2 + 2 * x * x) + 2 * p2 * x * y
        yd = y * radial_factor + p2 * (r2 + 2 * y * y) + 2 * p1 * x * y
        sin_she = math.sin(she)
        cos_she = math.cos(she)
        x_dist = scx * (xd - sin_she * yd)
        y_dist = scx * cos_she * yd

    # === metric_to_pixel (inlined) ===
    x_pixel = x_dist * inv_pix_x + imx_half
    y_pixel = imy_half - y_dist * inv_pix_y

    if chfield == 1:
        y_pixel = (y_pixel - 1.0) * 0.5
    elif chfield == 2:
        y_pixel = y_pixel * 0.5

    return x_pixel, y_pixel


def _point_to_pixel_fast(pos, cal, imx, imy, pix_x, pix_y, chfield, mm):
    """Project 3D position to pixel coordinates — convenience wrapper."""
    pc = _pack_cal(cal, mm)
    return _point_to_pixel_packed(pos, pc, imx * 0.5, imy * 0.5, 1.0 / pix_x, 1.0 / pix_y, chfield)


def predict(prev_pos, curr_pos, c):
    prev_pos = np.asarray(prev_pos)
    curr_pos = np.asarray(curr_pos)
    c[:] = curr_pos + (curr_pos - prev_pos)


def search_volume_center_moving(prev_pos, curr_pos):
    prev_pos = np.asarray(prev_pos)
    curr_pos = np.asarray(curr_pos)
    return curr_pos + (curr_pos - prev_pos)


def pos3d_in_bounds(pos, bounds):
    x, y, z = pos
    return bool(
        bounds.dvxmin < x < bounds.dvxmax and
        bounds.dvymin < y < bounds.dvymax and
        bounds.dvzmin < z < bounds.dvzmax
    )


def angle_acc(start, pred, cand):
    v0x = pred[0] - start[0]
    v0y = pred[1] - start[1]
    v0z = pred[2] - start[2]
    v1x = cand[0] - start[0]
    v1y = cand[1] - start[1]
    v1z = cand[2] - start[2]

    if v0x == -v1x and v0y == -v1y and v0z == -v1z:
        angle = 200.0
    elif v0x == v1x and v0y == v1y and v0z == v1z:
        angle = 0.0
    else:
        norm0 = math.sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
        norm1 = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if norm0 == 0 or norm1 == 0:
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


def candsearch_in_pix(next_targets, num_targets, cent_x, cent_y,
                      dl, dr, du, dd, cpar):
    p = [PT_UNUSED] * 4

    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > cpar.imy:
        ymax = cpar.imy

    p1 = p2 = p3 = p4 = PT_UNUSED
    dmin = 1e20
    d1 = d2 = d3 = d4 = dmin

    if not (0.0 <= cent_x <= cpar.imx and 0.0 <= cent_y <= cpar.imy):
        return p

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next_targets[j0].y < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        t = next_targets[j]
        if t.tnr != TR_UNUSED:
            if t.y > ymax:
                break
            if t.x > xmin and t.x < xmax and t.y > ymin and t.y < ymax:
                d = math.sqrt((cent_x - t.x) ** 2 + (cent_y - t.y) ** 2)

                if d < dmin:
                    dmin = d

                if d < d1:
                    p4, p3, p2, p1 = p3, p2, p1, j
                    d4, d3, d2, d1 = d3, d2, d1, d
                elif d1 < d < d2:
                    p4, p3, p2 = p3, p2, j
                    d4, d3, d2 = d3, d2, d
                elif d2 < d < d3:
                    p4, p3 = p3, j
                    d4, d3 = d3, d
                elif d3 < d < d4:
                    p4 = j
                    d4 = d

    p[0], p[1], p[2], p[3] = p1, p2, p3, p4
    return p


def candsearch_in_pix_rest(next_targets, num_targets, cent_x, cent_y,
                           dl, dr, du, dd, p, cpar):
    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > cpar.imx:
        xmax = cpar.imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > cpar.imy:
        ymax = cpar.imy

    p[0] = PT_UNUSED
    counter = 0
    dmin = 1e20

    if not (0.0 <= cent_x <= cpar.imx and 0.0 <= cent_y <= cpar.imy):
        return 0

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if next_targets[j0].y < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        t = next_targets[j]
        if t.tnr == TR_UNUSED:
            if t.y > ymax:
                break
            if t.x > xmin and t.x < xmax and t.y > ymin and t.y < ymax:
                d = math.sqrt((cent_x - t.x) ** 2 + (cent_y - t.y) ** 2)
                if d < dmin:
                    dmin = d
                    p[0] = j
                    counter = 1

    return counter


def reset_foundpix_array(arr, n, num_cams):
    for i in range(n):
        arr[i][0] = TR_UNUSED  # ftnr
        arr[i][1] = 0          # freq
        for j in range(num_cams):
            arr[i][2][j] = 0   # whichcam


def copy_foundpix_array(dest, src, n, num_cams):
    for i in range(n):
        dest[i][0] = src[i][0]
        dest[i][1] = src[i][1]
        for j in range(num_cams):
            dest[i][2][j] = src[i][2][j]


def _make_foundpix(num_cams):
    """Create a single foundpix entry as [ftnr, freq, whichcam_list]."""
    return [TR_UNUSED, 0, [0] * num_cams]


def _make_foundpix_array(n, num_cams):
    """Create array of n foundpix entries as plain Python lists."""
    return [_make_foundpix(num_cams) for _ in range(n)]


def sort_candidates_by_freq(items, num_cams):
    n = num_cams * MAX_CANDS

    for i in range(n):
        ftnr_i = items[i][0]
        for j in range(num_cams):
            for m in range(MAX_CANDS):
                if ftnr_i == items[4 * j + m][0]:
                    items[i][2][j] = 1

    for i in range(n):
        ftnr_i = items[i][0]
        if ftnr_i != TR_UNUSED:
            wc = items[i][2]
            for j in range(num_cams):
                if wc[j] == 1:
                    items[i][1] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if items[j - 1][1] < items[j][1]:
                items[j - 1], items[j] = items[j], items[j - 1]

    for i in range(n):
        ftnr_i = items[i][0]
        for j in range(i + 1, n):
            if items[j][0] == ftnr_i or items[j][1] < 2:
                items[j][1] = 0
                items[j][0] = TR_UNUSED

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if items[j - 1][1] < items[j][1]:
                items[j - 1], items[j] = items[j], items[j - 1]

    different = 0
    for i in range(n):
        if items[i][1] != 0:
            different += 1
    return different


def sort(n, a, b):
    """Bubble sort arrays a and b by ascending a values, in-place (matches C)."""
    flag = True
    while flag:
        flag = False
        for i in range(n - 1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                b[i], b[i + 1] = b[i + 1], b[i]
                flag = True


def point_to_pixel(point, cal, cpar):
    return _point_to_pixel_fast(
        point, cal, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield, cpar.mm,
    )


def searchquader(point, tpar, cpar, calib, _packed_cals=None, _pix_info=None,
                 _jit_tuples=None):
    num_cams = cpar.num_cams

    px, py, pz = point[0], point[1], point[2]
    dxmin, dymin, dzmin = tpar.dvxmin, tpar.dvymin, tpar.dvzmin
    dxmax, dymax, dzmax = tpar.dvxmax, tpar.dvymax, tpar.dvzmax

    quader = np.empty((8, 3))
    for pt in range(8):
        quader[pt, 0] = px + (dxmax if pt & 1 else dxmin)
        quader[pt, 1] = py + (dymax if pt & 2 else dymin)
        quader[pt, 2] = pz + (dzmax if pt & 4 else dzmin)

    if _pix_info is not None:
        c_imx, c_imy, imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield = _pix_info
    else:
        c_imx = cpar.imx; c_imy = cpar.imy
        imx_half = c_imx * 0.5; imy_half = c_imy * 0.5
        inv_pix_x = 1.0 / cpar.pix_x; inv_pix_y = 1.0 / cpar.pix_y
        c_chfield = cpar.chfield

    if _jit_tuples is not None:
        cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _jit_tuples
        pos_arr = np.asarray(point, dtype=np.float64)
        return _searchquader_jit(
            pos_arr, quader, num_cams, cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
            imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield, c_imx, c_imy)

    if _packed_cals is None:
        c_mm = cpar.mm
        _packed_cals = [_pack_cal(calib[i], c_mm) for i in range(num_cams)]

    xr = np.zeros(num_cams)
    xl = np.zeros(num_cams)
    yd = np.zeros(num_cams)
    yu = np.zeros(num_cams)

    for i in range(num_cams):
        pc = _packed_cals[i]
        xr_i = 0.0
        xl_i = float(c_imx)
        yd_i = 0.0
        yu_i = float(c_imy)

        cx, cy = _point_to_pixel_packed(point, pc, imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield)
        for pt in range(8):
            corner_x, corner_y = _point_to_pixel_packed(quader[pt], pc, imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield)
            if corner_x < xl_i: xl_i = corner_x
            if corner_y < yu_i: yu_i = corner_y
            if corner_x > xr_i: xr_i = corner_x
            if corner_y > yd_i: yd_i = corner_y

        if xl_i < 0: xl_i = 0
        if yu_i < 0: yu_i = 0
        if xr_i > c_imx: xr_i = c_imx
        if yd_i > c_imy: yd_i = c_imy

        xr[i] = xr_i - cx
        xl[i] = cx - xl_i
        yd[i] = yd_i - cy
        yu[i] = cy - yu_i

    return xr, xl, yd, yu


def register_closest_neighbs(targets, num_targets, cam, cent_x, cent_y,
                             dl, dr, du, dd, reg, cpar,
                             _targ_x=None, _targ_y=None, _targ_tnr=None):
    if HAS_NUMBA and _targ_x is not None:
        p0, p1, p2, p3 = _candsearch_in_pix_jit(
            _targ_x, _targ_y, _targ_tnr, num_targets,
            cent_x, cent_y, dl, dr, du, dd,
            cpar.imx, cpar.imy, TR_UNUSED)
        all_cands = [p0, p1, p2, p3]
        for cand in range(MAX_CANDS):
            if all_cands[cand] == PT_UNUSED:
                reg[cand][0] = TR_UNUSED
            else:
                reg[cand][2][cam] = 1
                reg[cand][0] = int(_targ_tnr[all_cands[cand]])
    else:
        all_cands = candsearch_in_pix(targets, num_targets, cent_x, cent_y,
                                      dl, dr, du, dd, cpar)
        for cand in range(MAX_CANDS):
            if all_cands[cand] == PT_UNUSED:
                reg[cand][0] = TR_UNUSED
            else:
                reg[cand][2][cam] = 1
                reg[cand][0] = targets[all_cands[cand]].tnr


def sorted_candidates_in_volume(center, center_proj, frm, run,
                                _packed_cals=None, _pix_info=None,
                                _jit_tuples=None):
    num_cams = frm.num_cams

    if HAS_NUMBA and _jit_tuples is not None:
        cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _jit_tuples
        if _pix_info is not None:
            c_imx, c_imy, imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield = _pix_info
        else:
            c_imx = run.cpar.imx; c_imy = run.cpar.imy
            imx_half = c_imx * 0.5; imy_half = c_imy * 0.5
            inv_pix_x = 1.0 / run.cpar.pix_x; inv_pix_y = 1.0 / run.cpar.pix_y
            c_chfield = run.cpar.chfield

        center_arr = np.asarray(center, dtype=np.float64)
        cpx = np.array([center_proj[j][0] for j in range(num_cams)], dtype=np.float64)
        cpy = np.array([center_proj[j][1] for j in range(num_cams)], dtype=np.float64)
        nt = np.array(frm.num_targets[:num_cams], dtype=np.int32)

        ftnr, freq, whichcam, num_cands = _sorted_candidates_jit(
            center_arr, cpx, cpy, num_cams, MAX_CANDS,
            cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
            tuple(frm.targ_x[:num_cams]), tuple(frm.targ_y[:num_cams]),
            tuple(frm.targ_tnr[:num_cams]), nt,
            run.tpar.dvxmin, run.tpar.dvxmax, run.tpar.dvymin,
            run.tpar.dvymax, run.tpar.dvzmin, run.tpar.dvzmax,
            imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield,
            c_imx, c_imy, TR_UNUSED,
        )
        if num_cands > 0:
            result = []
            for i in range(num_cands):
                result.append({'ftnr': int(ftnr[i]), 'freq': int(freq[i]),
                               'whichcam': list(whichcam[i])})
            result.append({'ftnr': TR_UNUSED, 'freq': 0, 'whichcam': [0]*num_cams})
            return result
        return None

    n = num_cams * MAX_CANDS
    points = _make_foundpix_array(n, num_cams)

    xr, xl, yd, yu = searchquader(center, run.tpar, run.cpar, run.cal,
                                  _packed_cals=_packed_cals, _pix_info=_pix_info,
                                  _jit_tuples=_jit_tuples)

    for cam in range(num_cams):
        cam_slice = points[cam * MAX_CANDS:(cam + 1) * MAX_CANDS]
        register_closest_neighbs(
            frm.targets[cam], frm.num_targets[cam], cam,
            center_proj[cam][0], center_proj[cam][1],
            xl[cam], xr[cam], yu[cam], yd[cam],
            cam_slice, run.cpar,
            _targ_x=frm.targ_x[cam], _targ_y=frm.targ_y[cam],
            _targ_tnr=frm.targ_tnr[cam])

    num_cands = sort_candidates_by_freq(points, num_cams)
    if num_cands > 0:
        result = []
        for i in range(num_cands):
            result.append({'ftnr': points[i][0], 'freq': points[i][1],
                           'whichcam': points[i][2][:] })
        result.append({'ftnr': TR_UNUSED, 'freq': 0, 'whichcam': [0]*num_cams})
        return result
    return None


def assess_new_position(pos, targ_pos, cand_inds, frm, run,
                        _jit_cals=None, _jit_mmluts=None, _pix_info=None):
    from .trafo import pixel_to_metric, dist_to_flat

    left = right = up = down = ADD_PART

    for cam in range(TR_MAX_CAMS):
        targ_pos[cam][0] = targ_pos[cam][1] = COORD_UNUSED

    c_imx = run.cpar.imx; c_imy = run.cpar.imy

    if _pix_info is not None:
        _, _, imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield = _pix_info
    else:
        imx_half = c_imx * 0.5; imy_half = c_imy * 0.5
        inv_pix_x = 1.0 / run.cpar.pix_x; inv_pix_y = 1.0 / run.cpar.pix_y
        c_chfield = run.cpar.chfield

    for cam in range(run.cpar.num_cams):
        if HAS_NUMBA and _jit_cals is not None:
            px, py = _ptp_jit(pos, _jit_cals[cam], _jit_mmluts[cam],
                              imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield)
        else:
            px, py = _point_to_pixel_fast(pos, run.cal[cam], c_imx, c_imy,
                                          run.cpar.pix_x, run.cpar.pix_y,
                                          c_chfield, run.cpar.mm)

        if HAS_NUMBA and hasattr(frm, 'targ_x'):
            best, num_cands = _candsearch_in_pix_rest_jit(
                frm.targ_x[cam], frm.targ_y[cam], frm.targ_tnr[cam],
                frm.num_targets[cam], px, py, left, right, up, down,
                c_imx, c_imy, TR_UNUSED)
            if num_cands > 0:
                cand_inds[cam][0] = best
        else:
            num_cands = candsearch_in_pix_rest(
                frm.targets[cam], frm.num_targets[cam],
                px, py, left, right, up, down,
                cand_inds[cam], run.cpar)

        if num_cands > 0:
            _ix = cand_inds[cam][0]
            targ_pos[cam][0] = frm.targ_x[cam][_ix] if hasattr(frm, 'targ_x') else frm.targets[cam][_ix].x
            targ_pos[cam][1] = frm.targ_y[cam][_ix] if hasattr(frm, 'targ_x') else frm.targets[cam][_ix].y

    valid_cams = 0
    for cam in range(run.cpar.num_cams):
        if (targ_pos[cam][0] != COORD_UNUSED and
                targ_pos[cam][1] != COORD_UNUSED):
            mx, my = pixel_to_metric(targ_pos[cam][0], targ_pos[cam][1], run.cpar)
            cal = run.cal[cam]
            fx, fy = dist_to_flat(
                mx, my,
                cal.int_par.xh, cal.int_par.yh,
                cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
                cal.added_par.p1, cal.added_par.p2,
                cal.added_par.scx, cal.added_par.she,
                run.flatten_tol)
            targ_pos[cam][0] = fx
            targ_pos[cam][1] = fy
            valid_cams += 1
    return valid_cams


def add_particle(frm, pos, cand_inds):
    num_parts = frm.num_parts
    ref_path_inf = frm.path_info[num_parts]
    ref_path_inf.x[:] = pos
    reset_links(ref_path_inf)

    ref_corres = frm.correspond[num_parts]
    for cam in range(frm.num_cams):
        ref_corres.p[cam] = CORRES_NONE
        if cand_inds[cam][0] != PT_UNUSED:
            _ix = cand_inds[cam][0]
            frm.targets[cam][_ix].tnr = num_parts
            frm.targ_tnr[cam][_ix] = num_parts
            ref_corres.p[cam] = _ix
            ref_corres.nr = num_parts
    frm.num_parts += 1


def track_forward_start(run):
    for step in range(run.seq_par.first, run.seq_par.first + TR_BUFSPACE - 1):
        run.fb.read_frame_at_end(step, read_links=False)
        run.fb.fb_next()
    run.fb.fb_prev()


def _sync_soa_to_aos(frm):
    """Fast SoA->AoS sync — only copies fields needed for file I/O."""
    for i in range(frm.num_parts):
        p = frm.path_info[i]
        p.x[:] = frm.path_x[i]
        p.prev = int(frm.path_prev[i])
        p.next = int(frm.path_next[i])
        p.prio = int(frm.path_prio[i])

        c = frm.correspond[i]
        c.nr = int(frm.corres_nr[i])
        c.p[:] = frm.corres_p[i]

    for cam in range(frm.num_cams):
        tnr_arr = frm.targ_tnr[cam]
        for j in range(frm.num_targets[cam]):
            frm.targets[cam][j].tnr = int(tnr_arr[j])


def trackcorr_c_loop(run_info, step):
    from .orientation import point_position

    fb = run_info.fb
    cal = run_info.cal
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar

    c_imx = cpar.imx; c_imy = cpar.imy
    imx_half = c_imx * 0.5; imy_half = c_imy * 0.5
    inv_pix_x = 1.0 / cpar.pix_x; inv_pix_y = 1.0 / cpar.pix_y
    c_chfield = cpar.chfield; c_mm = cpar.mm

    if HAS_NUMBA:
        jit_cals, jit_mmluts = _pack_cams_jit(cal, c_mm)
        _jt = _pack_cams_jit_tuples(jit_cals, jit_mmluts)
        cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _jt

        nc = fb.num_cams
        orig_parts = fb.buf[1].num_parts

        fb.buf[0]._sync_path_to_soa()
        fb.buf[1]._sync_path_to_soa()
        fb.buf[2]._sync_path_to_soa()
        fb.buf[3]._sync_path_to_soa()

        np2 = np.array([fb.buf[2].num_parts], dtype=np.int32)
        np3 = np.array([fb.buf[3].num_parts], dtype=np.int32)
        nt2 = np.array(fb.buf[2].num_targets[:nc], dtype=np.int32)
        nt3 = np.array(fb.buf[3].num_targets[:nc], dtype=np.int32)

        count1, num_added = _trackcorr_loop_jit(
            orig_parts,
            fb.buf[0].path_x,
            fb.buf[1].path_x, fb.buf[1].path_prev, fb.buf[1].path_next,
            fb.buf[1].path_inlist, fb.buf[1].path_finaldecis,
            fb.buf[1].path_decis, fb.buf[1].path_linkdecis,
            fb.buf[1].corres_p,
            tuple(fb.buf[1].targ_x[:nc]), tuple(fb.buf[1].targ_y[:nc]),
            fb.buf[2].path_x, fb.buf[2].path_prev, fb.buf[2].path_next,
            fb.buf[2].path_inlist, fb.buf[2].path_prio,
            fb.buf[2].path_finaldecis, fb.buf[2].path_decis,
            fb.buf[2].path_linkdecis, fb.buf[2].corres_p,
            fb.buf[2].corres_nr,
            tuple(fb.buf[2].targ_x[:nc]), tuple(fb.buf[2].targ_y[:nc]),
            tuple(fb.buf[2].targ_tnr[:nc]), nt2, np2,
            fb.buf[3].path_x, fb.buf[3].path_prev, fb.buf[3].path_next,
            fb.buf[3].path_inlist, fb.buf[3].path_prio,
            fb.buf[3].path_finaldecis, fb.buf[3].path_decis,
            fb.buf[3].path_linkdecis, fb.buf[3].corres_p,
            fb.buf[3].corres_nr,
            tuple(fb.buf[3].targ_x[:nc]), tuple(fb.buf[3].targ_y[:nc]),
            tuple(fb.buf[3].targ_tnr[:nc]), nt3, np3,
            cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
            tpar.dvxmin, tpar.dvxmax, tpar.dvymin, tpar.dvymax,
            tpar.dvzmin, tpar.dvzmax, tpar.dacc, tpar.dangle,
            int(tpar.add), run_info.lmax,
            vpar.X_lay[0], vpar.X_lay[1], run_info.ymin, run_info.ymax,
            vpar.Zmin_lay[0], vpar.Zmax_lay[1],
            nc, imx_half, imy_half, inv_pix_x, inv_pix_y,
            c_chfield, float(c_imx), float(c_imy),
            cpar.pix_x, cpar.pix_y, run_info.flatten_tol,
        )

        fb.buf[2].num_parts = int(np2[0])
        fb.buf[3].num_parts = int(np3[0])

        _sync_soa_to_aos(fb.buf[1])
        _sync_soa_to_aos(fb.buf[2])
        _sync_soa_to_aos(fb.buf[3])

        print(f"step: {step}, curr: {fb.buf[1].num_parts}, "
              f"next: {fb.buf[2].num_parts}, links: {count1}, "
              f"lost: {fb.buf[1].num_parts - count1}, add: {num_added}")

        run_info.npart = run_info.npart + fb.buf[1].num_parts
        run_info.nlinks = run_info.nlinks + count1

        fb.fb_next()
        fb.write_frame_from_start(step)
        if step < run_info.seq_par.last - 2:
            fb.read_frame_at_end(step + 3, read_links=False)
        else:
            fb.buf[fb.buf_len - 1].num_parts = 0
        return

    # ===== Python fallback (no Numba) =====
    curr_targets = fb.buf[1].targets
    packed_cals = [_pack_cal(cal[j], c_mm) for j in range(fb.num_cams)]
    _jt = None

    def _ptp(pos, j):
        return _point_to_pixel_packed(pos, packed_cals[j], imx_half, imy_half,
                                      inv_pix_x, inv_pix_y, c_chfield)

    count1 = 0
    num_added = 0
    orig_parts = fb.buf[1].num_parts

    for h in range(orig_parts):
        X = [np.zeros(3) for _ in range(6)]

        curr_path_inf = fb.buf[1].path_info[h]
        curr_corres = fb.buf[1].correspond[h]
        curr_path_inf.inlist = 0

        X[1][:] = curr_path_inf.x

        v1 = [[0.0, 0.0] for _ in range(fb.num_cams)]

        if curr_path_inf.prev >= 0:
            ref_path_inf = fb.buf[0].path_info[curr_path_inf.prev]
            X[0][:] = ref_path_inf.x
            X[2][:] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

            for j in range(fb.num_cams):
                v1[j] = list(_ptp(X[2], j))
        else:
            X[2][:] = X[1]
            for j in range(fb.num_cams):
                if curr_corres.p[j] == CORRES_NONE:
                    v1[j] = list(_ptp(X[2], j))
                else:
                    _ix = curr_corres.p[j]
                    v1[j][0] = curr_targets[j][_ix].x
                    v1[j][1] = curr_targets[j][_ix].y

        _pi = (c_imx, c_imy, imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield)
        w = sorted_candidates_in_volume(X[2], v1, fb.buf[2], run_info,
                                        _packed_cals=packed_cals, _pix_info=_pi,
                                        _jit_tuples=_jt)
        if w is None:
            continue

        count2 = 0
        mm = 0
        while w[mm]['ftnr'] != TR_UNUSED:
            ref_path_inf = fb.buf[2].path_info[w[mm]['ftnr']]
            X[3][:] = ref_path_inf.x

            if curr_path_inf.prev >= 0:
                for j in range(3):
                    X[5][j] = 0.5 * (5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])
            else:
                X[5][:] = search_volume_center_moving(X[1], X[3])

            for j in range(fb.num_cams):
                v1[j] = list(_ptp(X[5], j))

            wn = sorted_candidates_in_volume(X[5], v1, fb.buf[3], run_info,
                                             _packed_cals=packed_cals, _pix_info=_pi,
                                             _jit_tuples=_jt)
            if wn is not None:
                count3 = 0
                kk = 0
                while wn[kk]['ftnr'] != TR_UNUSED:
                    ref_path_inf = fb.buf[3].path_info[wn[kk]['ftnr']]
                    X[4][:] = ref_path_inf.x

                    diff_pos = X[4] - X[3]
                    if pos3d_in_bounds(diff_pos, tpar):
                        angle1, acc1 = angle_acc(X[3], X[4], X[5])
                        if curr_path_inf.prev >= 0:
                            angle0, acc0 = angle_acc(X[1], X[2], X[3])
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) / 2
                        angle = (angle0 + angle1) / 2
                        quali = wn[kk]['freq'] + w[mm]['freq']

                        if ((acc < tpar.dacc and angle < tpar.dangle) or
                                (acc < tpar.dacc / 10)):
                            dl = (_vec3_dist(X[1], X[3]) +
                                  _vec3_dist(X[4], X[3])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc +
                                  angle / tpar.dangle) / quali
                            register_link_candidate(curr_path_inf, rr, w[mm]['ftnr'])
                    kk += 1

            v2 = [[0.0, 0.0] for _ in range(TR_MAX_CAMS)]
            philf = [[PT_UNUSED] * MAX_CANDS for _ in range(TR_MAX_CAMS)]
            quali = assess_new_position(X[5], v2, philf, fb.buf[3], run_info,
                                        _pix_info=_pi)

            if quali >= 2:
                in_volume = 0
                v2_arr = np.array(v2[:cpar.num_cams], dtype=np.float64)
                X[4], dl = point_position(v2_arr, cpar.num_cams, cpar.mm, cal)

                if (vpar.X_lay[0] < X[4][0] < vpar.X_lay[1] and
                        run_info.ymin < X[4][1] < run_info.ymax and
                        vpar.Zmin_lay[0] < X[4][2] < vpar.Zmax_lay[1]):
                    in_volume = 1

                diff_pos = X[3] - X[4]
                if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[3], X[4], X[5])
                    if ((acc < tpar.dacc and angle < tpar.dangle) or
                            (acc < tpar.dacc / 10)):
                        dl = (_vec3_dist(X[1], X[3]) +
                              _vec3_dist(X[4], X[3])) / 2
                        rr = (dl / run_info.lmax + acc / tpar.dacc +
                              angle / tpar.dangle) / (quali + w[mm]['freq'])
                        register_link_candidate(curr_path_inf, rr, w[mm]['ftnr'])

                        if tpar.add:
                            add_particle(fb.buf[3], X[4], philf)
                            num_added += 1
                in_volume = 0
            quali = 0

            if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
                diff_pos = X[3] - X[1]
                if pos3d_in_bounds(diff_pos, tpar):
                    angle, acc = angle_acc(X[1], X[2], X[3])
                    if ((acc < tpar.dacc and angle < tpar.dangle) or
                            (acc < tpar.dacc / 10)):
                        quali = w[mm]['freq']
                        dl = (_vec3_dist(X[1], X[3]) +
                              _vec3_dist(X[0], X[1])) / 2
                        rr = (dl / run_info.lmax + acc / tpar.dacc +
                              angle / tpar.dangle) / quali
                        register_link_candidate(curr_path_inf, rr, w[mm]['ftnr'])

            mm += 1

        if tpar.add:
            if curr_path_inf.inlist == 0 and curr_path_inf.prev >= 0:
                v2 = [[0.0, 0.0] for _ in range(TR_MAX_CAMS)]
                philf = [[PT_UNUSED] * MAX_CANDS for _ in range(TR_MAX_CAMS)]
                quali = assess_new_position(X[2], v2, philf, fb.buf[2], run_info,
                                            _pix_info=_pi)

                if quali >= 2:
                    X[3][:] = X[2]
                    in_volume = 0

                    v2_arr = np.array(v2[:fb.num_cams], dtype=np.float64)
                    X[3], dl = point_position(v2_arr, fb.num_cams, cpar.mm, cal)

                    if (vpar.X_lay[0] < X[3][0] < vpar.X_lay[1] and
                            run_info.ymin < X[3][1] < run_info.ymax and
                            vpar.Zmin_lay[0] < X[3][2] < vpar.Zmax_lay[1]):
                        in_volume = 1

                    diff_pos = X[2] - X[3]
                    if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                        angle, acc = angle_acc(X[1], X[2], X[3])
                        if ((acc < tpar.dacc and angle < tpar.dangle) or
                                (acc < tpar.dacc / 10)):
                            dl = (_vec3_dist(X[1], X[3]) +
                                  _vec3_dist(X[0], X[1])) / 2
                            rr = (dl / run_info.lmax + acc / tpar.dacc +
                                  angle / tpar.dangle) / quali
                            register_link_candidate(
                                curr_path_inf, rr, fb.buf[2].num_parts)
                            add_particle(fb.buf[2], X[3], philf)
                            num_added += 1
                    in_volume = 0

    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]
        if curr_path_inf.inlist > 0:
            sort(curr_path_inf.inlist, curr_path_inf.decis,
                 curr_path_inf.linkdecis)
            curr_path_inf.finaldecis = curr_path_inf.decis[0]
            curr_path_inf.next = curr_path_inf.linkdecis[0]

    # Phase 2: Resolve conflicts (single-pass)
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]
        if curr_path_inf.inlist > 0:
            next_h = curr_path_inf.next
            ref_path_inf = fb.buf[2].path_info[next_h]
            if ref_path_inf.prev == PREV_NONE:
                ref_path_inf.prev = h
            else:
                prev_of_next = ref_path_inf.prev
                if fb.buf[1].path_info[prev_of_next].finaldecis > curr_path_inf.finaldecis:
                    fb.buf[1].path_info[prev_of_next].next = NEXT_NONE
                    ref_path_inf.prev = h
                else:
                    curr_path_inf.next = NEXT_NONE

    # Phase 3: Losers retry with fallback candidates (claim unclaimed only)
    for h in range(fb.buf[1].num_parts):
        curr_path_inf = fb.buf[1].path_info[h]
        if curr_path_inf.inlist > 1 and curr_path_inf.next == NEXT_NONE:
            for ti in range(1, curr_path_inf.inlist):
                cand = curr_path_inf.linkdecis[ti]
                if fb.buf[2].path_info[cand].prev == PREV_NONE:
                    curr_path_inf.next = cand
                    curr_path_inf.finaldecis = curr_path_inf.decis[ti]
                    fb.buf[2].path_info[cand].prev = h
                    break

    for h in range(fb.buf[1].num_parts):
        if fb.buf[1].path_info[h].next != NEXT_NONE:
            count1 += 1

    print(f"step: {step}, curr: {fb.buf[1].num_parts}, "
          f"next: {fb.buf[2].num_parts}, links: {count1}, "
          f"lost: {fb.buf[1].num_parts - count1}, add: {num_added}")

    run_info.npart = run_info.npart + fb.buf[1].num_parts
    run_info.nlinks = run_info.nlinks + count1

    fb.fb_next()
    fb.write_frame_from_start(step)
    if step < run_info.seq_par.last - 2:
        fb.read_frame_at_end(step + 3, read_links=False)
    else:
        fb.buf[fb.buf_len - 1].num_parts = 0


def trackcorr_c_finish(run_info, step):
    range_val = run_info.seq_par.last - run_info.seq_par.first
    npart = run_info.npart / range_val
    nlinks = run_info.nlinks / range_val
    print(f"Average over sequence, particles: {npart:5.1f}, "
          f"links: {nlinks:5.1f}, lost: {npart - nlinks:5.1f}")

    run_info.fb.fb_next()
    run_info.fb.write_frame_from_start(step)


def trackback_c(run_info):
    from .orientation import point_position

    cal = run_info.cal
    seq_par = run_info.seq_par
    tpar = run_info.tpar
    vpar = run_info.vpar
    cpar = run_info.cpar
    fb = run_info.fb

    c_imx = cpar.imx; c_imy = cpar.imy
    imx_half = c_imx * 0.5; imy_half = c_imy * 0.5
    inv_pix_x = 1.0 / cpar.pix_x; inv_pix_y = 1.0 / cpar.pix_y
    c_chfield = cpar.chfield; c_mm = cpar.mm
    packed_cals = [_pack_cal(cal[j], c_mm) for j in range(fb.num_cams)]
    if HAS_NUMBA:
        jit_cals, jit_mmluts = _pack_cams_jit(cal, c_mm)
        _jt = _pack_cams_jit_tuples(jit_cals, jit_mmluts)
    else:
        _jt = None

    def _ptp(pos, j):
        if HAS_NUMBA:
            return _ptp_jit(pos, jit_cals[j], jit_mmluts[j],
                            imx_half, imy_half, inv_pix_x, inv_pix_y, c_chfield)
        return _point_to_pixel_packed(pos, packed_cals[j], imx_half, imy_half,
                                      inv_pix_x, inv_pix_y, c_chfield)

    Ymin = 0.0
    Ymax = 0.0
    npart = 0.0
    nlinks = 0.0

    for step in range(seq_par.last, seq_par.last - 4, -1):
        fb.read_frame_at_end(step, read_links=True)
        fb.fb_next()
    fb.fb_prev()

    nc = fb.num_cams

    for step in range(seq_par.last - 1, seq_par.first, -1):

        if HAS_NUMBA:
            fb.buf[0]._sync_path_to_soa()
            fb.buf[1]._sync_path_to_soa()
            fb.buf[2]._sync_path_to_soa()
            fb.buf[3]._sync_path_to_soa()

            cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _jt
            num_parts_2 = np.array([fb.buf[2].num_parts], dtype=np.int32)

            count1, num_added = _trackback_loop_jit(
                fb.buf[1].num_parts,
                fb.buf[0].path_x,
                fb.buf[1].path_x, fb.buf[1].path_prev, fb.buf[1].path_next,
                fb.buf[1].path_inlist,
                fb.buf[1].path_finaldecis, fb.buf[1].path_decis,
                fb.buf[1].path_linkdecis,
                fb.buf[2].path_x, fb.buf[2].path_prev, fb.buf[2].path_next,
                num_parts_2,
                fb.buf[2].targ_x, fb.buf[2].targ_y, fb.buf[2].targ_tnr,
                fb.buf[2].num_targets,
                fb.buf[2].corres_p, fb.buf[2].corres_nr,
                fb.buf[2].path_inlist, fb.buf[2].path_prio,
                fb.buf[2].path_finaldecis,
                fb.buf[2].path_decis, fb.buf[2].path_linkdecis,
                fb.buf[3].path_x, fb.buf[3].path_prev,
                cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t,
                tpar.dvxmin, tpar.dvxmax, tpar.dvymin, tpar.dvymax,
                tpar.dvzmin, tpar.dvzmax,
                tpar.dacc, tpar.dangle, tpar.add, run_info.lmax,
                vpar.X_lay[0], vpar.X_lay[1], Ymin, Ymax,
                vpar.Zmin_lay[0], vpar.Zmax_lay[1],
                nc, imx_half, imy_half, inv_pix_x, inv_pix_y,
                c_chfield, c_imx, c_imy, cpar.pix_x, cpar.pix_y,
                run_info.flatten_tol,
            )

            fb.buf[2].num_parts = int(num_parts_2[0])

            _sync_soa_to_aos(fb.buf[1])
            _sync_soa_to_aos(fb.buf[2])

        else:
            for h in range(fb.buf[1].num_parts):
                curr_path_inf = fb.buf[1].path_info[h]

                if not ((curr_path_inf.next < 0) or (curr_path_inf.prev != -1)):
                    continue

                X = [np.zeros(3) for _ in range(6)]
                curr_path_inf.inlist = 0
                X[1][:] = curr_path_inf.x

                ref_path_inf = fb.buf[0].path_info[curr_path_inf.next]
                X[0][:] = ref_path_inf.x
                X[2][:] = search_volume_center_moving(ref_path_inf.x, curr_path_inf.x)

                n = [[0.0, 0.0] for _ in range(fb.num_cams)]
                for j in range(fb.num_cams):
                    n[j] = list(_ptp(X[2], j))

                _pi = (c_imx, c_imy, imx_half, imy_half, inv_pix_x, inv_pix_y,
                       c_chfield)
                w = sorted_candidates_in_volume(X[2], n, fb.buf[2], run_info,
                                                _packed_cals=packed_cals,
                                                _pix_info=_pi,
                                                _jit_tuples=_jt)

                if w is not None:
                    i = 0
                    while w[i]['ftnr'] != TR_UNUSED:
                        ref_path_inf = fb.buf[2].path_info[w[i]['ftnr']]
                        X[3][:] = ref_path_inf.x

                        diff_pos = X[1] - X[3]
                        if pos3d_in_bounds(diff_pos, tpar):
                            angle, acc = angle_acc(X[1], X[2], X[3])
                            if ((acc < tpar.dacc and angle < tpar.dangle) or
                                    (acc < tpar.dacc / 10)):
                                dl = (_vec3_dist(X[1], X[3]) +
                                      _vec3_dist(X[0], X[1])) / 2
                                quali = w[i]['freq']
                                rr = (dl / run_info.lmax + acc / tpar.dacc +
                                      angle / tpar.dangle) / quali
                                register_link_candidate(
                                    curr_path_inf, rr, w[i]['ftnr'])
                        i += 1

                if tpar.add:
                    if curr_path_inf.inlist == 0:
                        v2 = [[0.0, 0.0] for _ in range(TR_MAX_CAMS)]
                        philf = [[PT_UNUSED] * MAX_CANDS for _ in range(TR_MAX_CAMS)]
                        quali = assess_new_position(
                            X[2], v2, philf, fb.buf[2], run_info,
                            _jit_cals=jit_cals if HAS_NUMBA else None,
                            _jit_mmluts=jit_mmluts if HAS_NUMBA else None,
                            _pix_info=_pi)
                        if quali >= 2:
                            in_volume = 0
                            v2_arr = np.array(v2[:fb.num_cams], dtype=np.float64)
                            if HAS_NUMBA:
                                X[3], _dl = _point_position_jit(
                                    v2_arr, fb.num_cams, _jt[0])
                            else:
                                X[3], _dl = point_position(
                                    v2_arr, fb.num_cams, cpar.mm, cal)

                            if (vpar.X_lay[0] < X[3][0] < vpar.X_lay[1] and
                                    Ymin < X[3][1] < Ymax and
                                    vpar.Zmin_lay[0] < X[3][2] < vpar.Zmax_lay[1]):
                                in_volume = 1

                            diff_pos = X[1] - X[3]
                            if in_volume == 1 and pos3d_in_bounds(diff_pos, tpar):
                                angle, acc = angle_acc(X[1], X[2], X[3])
                                if ((acc < tpar.dacc and angle < tpar.dangle) or
                                        (acc < tpar.dacc / 10)):
                                    dl = (_vec3_dist(X[1], X[3]) +
                                          _vec3_dist(X[0], X[1])) / 2
                                    rr = (dl / run_info.lmax + acc / tpar.dacc +
                                          angle / tpar.dangle) / quali
                                    register_link_candidate(
                                        curr_path_inf, rr, fb.buf[2].num_parts)
                                    add_particle(fb.buf[2], X[3], philf)
                            in_volume = 0

            for h in range(fb.buf[1].num_parts):
                curr_path_inf = fb.buf[1].path_info[h]
                if curr_path_inf.inlist > 0:
                    sort(curr_path_inf.inlist, curr_path_inf.decis,
                         curr_path_inf.linkdecis)

            count1 = 0
            num_added = 0
            for h in range(fb.buf[1].num_parts):
                curr_path_inf = fb.buf[1].path_info[h]

                if curr_path_inf.inlist > 0:
                    ref_path_inf = fb.buf[2].path_info[
                        curr_path_inf.linkdecis[0]]

                    if (ref_path_inf.prev == PREV_NONE and
                            ref_path_inf.next == NEXT_NONE):
                        curr_path_inf.finaldecis = curr_path_inf.decis[0]
                        curr_path_inf.prev = curr_path_inf.linkdecis[0]
                        fb.buf[2].path_info[curr_path_inf.prev].next = h
                        num_added += 1

                    if (ref_path_inf.prev != PREV_NONE and
                            ref_path_inf.next == NEXT_NONE):
                        X = [np.zeros(3) for _ in range(6)]
                        X[0][:] = fb.buf[0].path_info[
                            curr_path_inf.next].x
                        X[1][:] = curr_path_inf.x
                        X[3][:] = ref_path_inf.x
                        X[4][:] = fb.buf[3].path_info[
                            ref_path_inf.prev].x
                        for j in range(3):
                            X[5][j] = 0.5 * (
                                5.0 * X[3][j] - 4.0 * X[1][j] + X[0][j])

                        angle, acc = angle_acc(X[3], X[4], X[5])
                        if ((acc < tpar.dacc and angle < tpar.dangle) or
                                (acc < tpar.dacc / 10)):
                            curr_path_inf.finaldecis = curr_path_inf.decis[0]
                            curr_path_inf.prev = curr_path_inf.linkdecis[0]
                            fb.buf[2].path_info[
                                curr_path_inf.prev].next = h
                            num_added += 1

                if curr_path_inf.prev != PREV_NONE:
                    count1 += 1

        print(f"step: {step}, curr: {fb.buf[1].num_parts}, "
              f"next: {fb.buf[2].num_parts}, links: {count1}, "
              f"lost: {fb.buf[1].num_parts - count1}, add: {num_added}")

        npart = npart + fb.buf[1].num_parts
        nlinks = nlinks + count1

        fb.fb_next()
        fb.write_frame_from_start(step)
        if step > seq_par.first + 2:
            fb.read_frame_at_end(step - 3, read_links=True)

    npart /= (seq_par.last - seq_par.first - 1)
    nlinks /= (seq_par.last - seq_par.first - 1)

    print(f"Average over sequence, particles: {npart:5.1f}, "
          f"links: {nlinks:5.1f}, lost: {npart - nlinks:5.1f}")

    fb.fb_next()
    fb.write_frame_from_start(seq_par.first)

    return nlinks
