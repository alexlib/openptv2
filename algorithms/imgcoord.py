"""Image coordinate projection from 3D world positions.

Translation of lib/src/imgcoord.c and lib/include/imgcoord.h.

Projects 3D world positions to 2D image coordinates, with or without
distortion modeling.
"""

import math
import cython
import numpy as np
from .trafo import flat_to_dist as _flat_to_dist
from .multimed import multimed_nlay as _multimed_nlay, get_mmf_from_mmlut as _get_mmf_from_mmlut


@cython.ccall
def flat_image_coord(
    pos: cython.double[:],
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    ext_dm: cython.double[:, :],
    int_cc: cython.double,
    glass_vec_x: cython.double,
    glass_vec_y: cython.double,
    glass_vec_z: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
    mmlut=None,
) -> tuple:
    """Project 3D position to undistorted metric image coordinates.

    Inlines trans_cam_point + multimed_nlay + back_trans_point to avoid
    intermediate numpy array allocations in the hot path.
    """
    pos0: cython.double = pos[0]
    pos1: cython.double = pos[1]
    pos2: cython.double = pos[2]
    gx: cython.double = glass_vec_x
    gy: cython.double = glass_vec_y
    gz: cython.double = glass_vec_z

    # === trans_cam_point (inlined) ===
    dist_o_glas: cython.double = math.sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog: cython.double = 1.0 / dist_o_glas

    dot_cam: cython.double = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_cam_glas: cython.double = dot_cam * inv_dog - dist_o_glas - mm_d0

    dot_pos: cython.double = pos0 * gx + pos1 * gy + pos2 * gz
    dist_point_glas: cython.double = dot_pos * inv_dog - dist_o_glas

    s_cam: cython.double = dist_cam_glas * inv_dog
    cc_x: cython.double = ext_x0 - gx * s_cam
    cc_y: cython.double = ext_y0 - gy * s_cam
    cc_z: cython.double = ext_z0 - gz * s_cam

    s_pt: cython.double = dist_point_glas * inv_dog
    cp_x: cython.double = pos0 - gx * s_pt
    cp_y: cython.double = pos1 - gy * s_pt
    cp_z: cython.double = pos2 - gz * s_pt

    ext_t_z0: cython.double = dist_cam_glas + mm_d0

    s_d: cython.double = mm_d0 * inv_dog
    ag_x: cython.double = cc_x - gx * s_d
    ag_y: cython.double = cc_y - gy * s_d
    ag_z: cython.double = cc_z - gz * s_d
    tmp_x: cython.double = cp_x - ag_x
    tmp_y: cython.double = cp_y - ag_y
    tmp_z: cython.double = cp_z - ag_z

    pos_t_0: cython.double = math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
    pos_t_2: cython.double = dist_point_glas

    # === mmlut lookup ===
    mmf: cython.double = 1.0
    if mmlut is not None and mmlut.data is not None:
        mmf = _get_mmf_from_mmlut(
            (pos_t_0, 0.0, pos_t_2), mmlut.origin, mmlut.nr, mmlut.nz, mmlut.rw, mmlut.data,
        )
        if mmf <= 0:
            mmf = 1.0

    X_t_val, Y_t_val = _multimed_nlay(
        pos_t_0, 0.0, pos_t_2,
        0.0, 0.0, ext_t_z0, mm_n1, mm_n2_0, mm_n3, mm_d0,
        mmf=mmf,
    )
    X_t: cython.double = X_t_val
    Y_t: cython.double = Y_t_val

    # === back_trans_point (inlined) ===
    n_gl: cython.double = dist_o_glas
    inv_ngl: cython.double = inv_dog

    # ag_x/y/z already computed above (after_glass)
    # tmp_x/y/z = cross_p - after_glass (already computed)
    n_ve: cython.double = math.sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)

    s_z: cython.double = -pos_t_2 * inv_ngl
    bx: cython.double = ag_x - gx * s_z
    by: cython.double = ag_y - gy * s_z
    bz: cython.double = ag_z - gz * s_z

    s_x: cython.double
    if n_ve > 0:
        s_x = -X_t / n_ve
        bx -= tmp_x * s_x
        by -= tmp_y * s_x
        bz -= tmp_z * s_x

    # === perspective projection ===
    dx: cython.double = bx - ext_x0
    dy: cython.double = by - ext_y0
    dz: cython.double = bz - ext_z0

    dm00: cython.double = ext_dm[0, 0]
    dm10: cython.double = ext_dm[1, 0]
    dm20: cython.double = ext_dm[2, 0]
    dm01: cython.double = ext_dm[0, 1]
    dm11: cython.double = ext_dm[1, 1]
    dm21: cython.double = ext_dm[2, 1]
    dm02: cython.double = ext_dm[0, 2]
    dm12: cython.double = ext_dm[1, 2]
    dm22: cython.double = ext_dm[2, 2]

    deno: cython.double = dm02 * dx + dm12 * dy + dm22 * dz

    x: cython.double = -int_cc * (dm00 * dx + dm10 * dy + dm20 * dz) / deno
    y: cython.double = -int_cc * (dm01 * dx + dm11 * dy + dm21 * dz) / deno

    return x, y


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
    pos = np.ascontiguousarray(pos, dtype=np.float64)
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


@cython.ccall
def _img_coord_params(
    pos: cython.double[:],
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    ext_dm: cython.double[:, :],
    int_cc: cython.double,
    int_xh: cython.double,
    int_yh: cython.double,
    glass_vec_x: cython.double,
    glass_vec_y: cython.double,
    glass_vec_z: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    mmlut=None,
) -> tuple:
    x: cython.double
    y: cython.double
    x, y = flat_image_coord(
        pos,
        ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
        mmlut=mmlut,
    )
    return _flat_to_dist(x, y, int_xh, int_yh, k1, k2, k3, p1, p2, scx, she)


def img_coord_batch(positions, cal, mm):
    """Project N 3D positions to distorted metric image coordinates.

    Uses Numba JIT when available for ~30x speedup over scalar loop.

    Args:
        positions: (N, 3) array of 3D positions.
        cal: Calibration object.
        mm: MmNp multimedia parameters.

    Returns:
        (N, 2) array of distorted metric coordinates.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    try:
        from .track_kernels import (
            HAS_NUMBA, img_coord_batch_jit, pack_cal_array, pack_mmlut,
        )
        if HAS_NUMBA:
            cal_arr = pack_cal_array(cal, mm)
            mmlut_data, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw = pack_mmlut(cal)
            return img_coord_batch_jit(
                positions, cal_arr, mmlut_data, mmlut_origin,
                mmlut_nr, mmlut_nz, mmlut_rw,
            )
    except ImportError:
        pass

    return _img_coord_batch_impl(positions, cal, mm)


@cython.ccall
def _img_coord_batch_impl(positions: cython.double[:, :], cal, mm) -> np.ndarray:
    n: cython.int = positions.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    res_mv: cython.double[:, :] = result

    # Extract params once
    ext_x0: cython.double = cal.ext_par.x0
    ext_y0: cython.double = cal.ext_par.y0
    ext_z0: cython.double = cal.ext_par.z0
    ext_dm: cython.double[:, :] = cal.ext_par.dm
    int_cc: cython.double = cal.int_par.cc
    int_xh: cython.double = cal.int_par.xh
    int_yh: cython.double = cal.int_par.yh
    glass_vec_x: cython.double = cal.glass_par.vec_x
    glass_vec_y: cython.double = cal.glass_par.vec_y
    glass_vec_z: cython.double = cal.glass_par.vec_z
    mm_n1: cython.double = mm.n1
    mm_n2_0: cython.double = mm.n2[0]
    mm_n3: cython.double = mm.n3
    mm_d0: cython.double = mm.d[0]
    k1: cython.double = cal.added_par.k1
    k2: cython.double = cal.added_par.k2
    k3: cython.double = cal.added_par.k3
    p1: cython.double = cal.added_par.p1
    p2: cython.double = cal.added_par.p2
    scx: cython.double = cal.added_par.scx
    she: cython.double = cal.added_par.she
    mmlut = cal.mmlut if cal.mmlut.is_initialized else None

    i: cython.int
    x: cython.double
    y: cython.double
    for i in range(n):
        x, y = _img_coord_params(
            positions[i, :],
            ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
            int_xh, int_yh, glass_vec_x, glass_vec_y, glass_vec_z,
            mm_n1, mm_n2_0, mm_n3, mm_d0, k1, k2, k3, p1, p2, scx, she,
            mmlut=mmlut,
        )
        res_mv[i, 0] = x
        res_mv[i, 1] = y
    return result


def flat_image_coord_batch(positions, cal, mm):
    """Project N 3D positions to flat metric image coordinates.

    Uses Numba JIT when available for ~24x speedup over scalar loop.

    Args:
        positions: (N, 3) array of 3D positions.
        cal: Calibration object.
        mm: MmNp multimedia parameters.

    Returns:
        (N, 2) array of flat (undistorted) metric coordinates.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    try:
        from .track_kernels import (
            HAS_NUMBA, flat_image_coord_batch_jit, pack_cal_array, pack_mmlut,
        )
        if HAS_NUMBA:
            cal_arr = pack_cal_array(cal, mm)
            mmlut_data, mmlut_origin, mmlut_nr, mmlut_nz, mmlut_rw = pack_mmlut(cal)
            return flat_image_coord_batch_jit(
                positions, cal_arr, mmlut_data, mmlut_origin,
                mmlut_nr, mmlut_nz, mmlut_rw,
            )
    except ImportError:
        pass

    return _flat_image_coord_batch_impl(positions, cal, mm)


@cython.ccall
def _flat_image_coord_batch_impl(positions: cython.double[:, :], cal, mm) -> np.ndarray:
    n: cython.int = positions.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    res_mv: cython.double[:, :] = result

    # Extract params once
    ext_x0: cython.double = cal.ext_par.x0
    ext_y0: cython.double = cal.ext_par.y0
    ext_z0: cython.double = cal.ext_par.z0
    ext_dm: cython.double[:, :] = cal.ext_par.dm
    int_cc: cython.double = cal.int_par.cc
    glass_vec_x: cython.double = cal.glass_par.vec_x
    glass_vec_y: cython.double = cal.glass_par.vec_y
    glass_vec_z: cython.double = cal.glass_par.vec_z
    mm_n1: cython.double = mm.n1
    mm_n2_0: cython.double = mm.n2[0]
    mm_n3: cython.double = mm.n3
    mm_d0: cython.double = mm.d[0]
    mmlut = cal.mmlut if cal.mmlut.is_initialized else None

    i: cython.int
    x: cython.double
    y: cython.double
    for i in range(n):
        x, y = flat_image_coord(
            positions[i, :],
            ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
            glass_vec_x, glass_vec_y, glass_vec_z,
            mm_n1, mm_n2_0, mm_n3, mm_d0,
            mmlut=mmlut,
        )
        res_mv[i, 0] = x
        res_mv[i, 1] = y
    return result


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
