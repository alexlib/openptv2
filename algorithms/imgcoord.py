"""Image coordinate projection from 3D world positions.

Translation of lib/src/imgcoord.c and lib/include/imgcoord.h.

Projects 3D world positions to 2D image coordinates, with or without
distortion modeling.
"""

import math
import numpy as np
from .trafo import flat_to_dist as _flat_to_dist
from .multimed import multimed_nlay as _multimed_nlay, get_mmf_from_mmlut as _get_mmf_from_mmlut


def flat_image_coord(
    pos,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    ext_dm,
    int_cc: float,
    glass_vec_x: float,
    glass_vec_y: float,
    glass_vec_z: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
    mmlut=None,
) -> tuple[float, float]:
    """Project 3D position to undistorted metric image coordinates.

    Inlines trans_cam_point + multimed_nlay + back_trans_point to avoid
    intermediate numpy array allocations in the hot path.
    """
    pos0, pos1, pos2 = float(pos[0]), float(pos[1]), float(pos[2])
    gx, gy, gz = glass_vec_x, glass_vec_y, glass_vec_z

    # === trans_cam_point (inlined) ===
    dist_o_glas = math.sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog = 1.0 / dist_o_glas

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

    # === mmlut lookup ===
    mmf = 1.0
    if mmlut is not None and mmlut.data is not None:
        mmf = _get_mmf_from_mmlut(
            (pos_t_0, 0.0, pos_t_2), mmlut.origin, mmlut.nr, mmlut.nz, mmlut.rw, mmlut.data,
        )
        if mmf <= 0:
            mmf = 1.0

    X_t, Y_t = _multimed_nlay(
        pos_t_0, 0.0, pos_t_2,
        0.0, 0.0, ext_t_z0, mm_n1, mm_n2_0, mm_n3, mm_d0,
        mmf=mmf,
    )

    # === back_trans_point (inlined) ===
    n_gl = dist_o_glas
    inv_ngl = inv_dog

    # ag_x/y/z already computed above (after_glass)
    # tmp_x/y/z = cross_p - after_glass (already computed)
    n_ve = math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)

    s_z = -pos_t_2 * inv_ngl
    bx = ag_x - gx * s_z
    by = ag_y - gy * s_z
    bz = ag_z - gz * s_z

    if n_ve > 0:
        s_x = -X_t / n_ve
        bx -= tmp_x * s_x
        by -= tmp_y * s_x
        bz -= tmp_z * s_x

    # === perspective projection ===
    dx = bx - ext_x0
    dy = by - ext_y0
    dz = bz - ext_z0

    dm00 = ext_dm[0, 0]; dm10 = ext_dm[1, 0]; dm20 = ext_dm[2, 0]
    dm01 = ext_dm[0, 1]; dm11 = ext_dm[1, 1]; dm21 = ext_dm[2, 1]
    dm02 = ext_dm[0, 2]; dm12 = ext_dm[1, 2]; dm22 = ext_dm[2, 2]

    deno = dm02 * dx + dm12 * dy + dm22 * dz

    x = -int_cc * (dm00 * dx + dm10 * dy + dm20 * dz) / deno
    y = -int_cc * (dm01 * dx + dm11 * dy + dm21 * dz) / deno

    return float(x), float(y)


def img_coord(
    pos,
    ext_x0_or_cal=None,
    ext_y0_or_mm=None,
    ext_z0=None,
    ext_dm=None,
    int_cc=None,
    int_xh=None,
    int_yh=None,
    glass_vec_x=None,
    glass_vec_y=None,
    glass_vec_z=None,
    mm_n1=None,
    mm_n2_0=None,
    mm_n3=None,
    mm_d0=None,
    k1=None,
    k2=None,
    k3=None,
    p1=None,
    p2=None,
    scx=None,
    she=None,
) -> tuple[float, float]:
    """Project 3D position to distorted metric image coordinates.

    Accepts either (pos, Calibration, mm_params) or all individual parameters.
    """
    if ext_z0 is None and hasattr(ext_x0_or_cal, 'ext_par'):
        cal = ext_x0_or_cal
        mm = ext_y0_or_mm
        mmlut = cal.mmlut if cal.mmlut.is_initialized else None
        return _img_coord_params(
            pos,
            cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
            cal.ext_par.dm, cal.int_par.cc,
            cal.int_par.xh, cal.int_par.yh,
            cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
            mm.n1, mm.n2[0], mm.n3, mm.d[0],
            cal.added_par.k1, cal.added_par.k2, cal.added_par.k3,
            cal.added_par.p1, cal.added_par.p2,
            cal.added_par.scx, cal.added_par.she,
            mmlut=mmlut,
        )
    return _img_coord_params(
        pos, ext_x0_or_cal, ext_y0_or_mm, ext_z0, ext_dm, int_cc,
        int_xh, int_yh, glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0, k1, k2, k3, p1, p2, scx, she,
    )


def _img_coord_params(
    pos, ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
    int_xh, int_yh, glass_vec_x, glass_vec_y, glass_vec_z,
    mm_n1, mm_n2_0, mm_n3, mm_d0, k1, k2, k3, p1, p2, scx, she,
    mmlut=None,
):
    x, y = flat_image_coord(
        pos,
        ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
        mmlut=mmlut,
    )
    return _flat_to_dist(x, y, int_xh, int_yh, k1, k2, k3, p1, p2, scx, she)
