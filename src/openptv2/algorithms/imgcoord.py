"""Image coordinate projection from 3D world positions.

Translation of lib/src/imgcoord.c and lib/include/imgcoord.h.

Projects 3D world positions to 2D image coordinates, with or without
distortion modeling.
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
from .multimed import (
    _multimed_nlay_core as _multimed_nlay_core,
)

_DUMMY_MMLUT_DATA = np.zeros(1, dtype=np.float64)


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def _flat_to_dist_core(
    flat_x: cython.double,
    flat_y: cython.double,
    xh: cython.double,
    yh: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
) -> tuple:
    x: cython.double = flat_x + xh
    y: cython.double = flat_y + yh

    r: cython.double = c_sqrt(x * x + y * y)
    if r < 1e-10:
        return 0.0, 0.0

    r2: cython.double = r * r
    r4: cython.double = r2 * r2
    r6: cython.double = r4 * r2
    radial_factor: cython.double = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_dist: cython.double = (
        x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
    )
    y_dist: cython.double = (
        y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
    )

    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)

    x1: cython.double = scx * (x_dist - sin_she * y_dist)
    y1: cython.double = scx * cos_she * y_dist

    return x1, y1


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def _get_mmf_from_mmlut_core(
    pos_x: cython.double,
    pos_y: cython.double,
    pos_z: cython.double,
    mmlut_origin_x: cython.double,
    mmlut_origin_y: cython.double,
    mmlut_origin_z: cython.double,
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
    mmlut_data: cython.double[:],
) -> cython.double:
    tx: cython.double = pos_x - mmlut_origin_x
    ty: cython.double = pos_y - mmlut_origin_y
    tz: cython.double = pos_z - mmlut_origin_z
    sz: cython.double = tz / mmlut_rw
    iz: cython.int = int(sz)
    sz -= iz

    R: cython.double = c_sqrt(tx * tx + ty * ty)
    sr: cython.double = R / mmlut_rw
    ir: cython.int = int(sr)
    sr -= ir

    # Bilinear interpolation needs the 4 cell corners
    # (ir,iz)..(ir+1,iz+1) to be valid grid indices, i.e. ir+1 <= nr-1 and
    # iz+1 <= nz-1. data has exactly nr*nz elements (indices 0..nr*nz-1);
    # the old check `v4 > nr*nz` let the exact far corner read one element
    # past the end (OOB under boundscheck(False)). Points outside the LUT
    # cells return 0.0 so the caller falls back to the iterative solve.
    # Keep in sync with multimed.get_mmf_from_mmlut.
    if ir < 0 or ir + 1 > mmlut_nr - 1:
        return 0.0
    if iz < 0 or iz + 1 > mmlut_nz - 1:
        return 0.0

    v4_0: cython.int = ir * mmlut_nz + iz
    v4_1: cython.int = ir * mmlut_nz + (iz + 1)
    v4_2: cython.int = (ir + 1) * mmlut_nz + iz
    v4_3: cython.int = (ir + 1) * mmlut_nz + (iz + 1)

    # Bilinear interpolation
    mmf: cython.double = (
        mmlut_data[v4_0] * (1.0 - sr) * (1.0 - sz)
        + mmlut_data[v4_1] * (1.0 - sr) * sz
        + mmlut_data[v4_2] * sr * (1.0 - sz)
        + mmlut_data[v4_3] * sr * sz
    )

    return mmf


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def _flat_image_coord_core(
    pos0: cython.double,
    pos1: cython.double,
    pos2: cython.double,
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
    has_mmlut: cython.bint,
    mmlut_origin_x: cython.double,
    mmlut_origin_y: cython.double,
    mmlut_origin_z: cython.double,
    mmlut_nr: cython.int,
    mmlut_nz: cython.int,
    mmlut_rw: cython.double,
    mmlut_data: cython.double[:],
) -> tuple:
    gx: cython.double = glass_vec_x
    gy: cython.double = glass_vec_y
    gz: cython.double = glass_vec_z

    # === trans_cam_point (inlined) ===
    dist_o_glas: cython.double = c_sqrt(gx * gx + gy * gy + gz * gz)

    # Zero glass vector → no glass interface.  Fall through to a simple
    # pinhole projection with no multimedia correction.
    if dist_o_glas < 1e-12:
        dx: cython.double = pos0 - ext_x0
        dy: cython.double = pos1 - ext_y0
        dz: cython.double = pos2 - ext_z0
        denom: cython.double = ext_dm[0, 2] * dx + ext_dm[1, 2] * dy + ext_dm[2, 2] * dz
        if denom == 0.0:
            return 0.0, 0.0
        x0: cython.double = (
            -int_cc
            * (ext_dm[0, 0] * dx + ext_dm[1, 0] * dy + ext_dm[2, 0] * dz)
            / denom
        )
        y0: cython.double = (
            -int_cc
            * (ext_dm[0, 1] * dx + ext_dm[1, 1] * dy + ext_dm[2, 1] * dz)
            / denom
        )
        return x0, y0

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

    pos_t_0: cython.double = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
    pos_t_2: cython.double = dist_point_glas

    # === mmlut lookup ===
    mmf: cython.double = 1.0
    if has_mmlut:
        mmf = _get_mmf_from_mmlut_core(
            pos_t_0,
            0.0,
            pos_t_2,
            mmlut_origin_x,
            mmlut_origin_y,
            mmlut_origin_z,
            mmlut_nr,
            mmlut_nz,
            mmlut_rw,
            mmlut_data,
        )
        if mmf <= 0.0:
            mmf = 1.0

    X_t_val, Y_t_val = _multimed_nlay_core(
        pos_t_0,
        pos_t_2,
        ext_t_z0,
        mm_n1,
        mm_n2_0,
        mm_n3,
        mm_d0,
        mmf,
    )
    X_t: cython.double = X_t_val

    # === back_trans_point (inlined) ===
    inv_ngl: cython.double = inv_dog

    # ag_x/y/z already computed above (after_glass)
    # tmp_x/y/z = cross_p - after_glass (already computed)
    n_ve: cython.double = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)

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

    deno: cython.double = ext_dm[0, 2] * dx + ext_dm[1, 2] * dy + ext_dm[2, 2] * dz

    x: cython.double = (
        -int_cc * (ext_dm[0, 0] * dx + ext_dm[1, 0] * dy + ext_dm[2, 0] * dz) / deno
    )
    y: cython.double = (
        -int_cc * (ext_dm[0, 1] * dx + ext_dm[1, 1] * dy + ext_dm[2, 1] * dz) / deno
    )

    return x, y


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
    """Project 3D position to undistorted metric image coordinates."""
    has_mmlut: cython.bint = False
    mmlut_origin_x: cython.double = 0.0
    mmlut_origin_y: cython.double = 0.0
    mmlut_origin_z: cython.double = 0.0
    mmlut_nr: cython.int = 0
    mmlut_nz: cython.int = 0
    mmlut_rw: cython.double = 0.0
    mmlut_data: cython.double[:] = _DUMMY_MMLUT_DATA

    if mmlut is not None and mmlut.data is not None:
        has_mmlut = True
        mmlut_origin_x = mmlut.origin[0]
        mmlut_origin_y = mmlut.origin[1]
        mmlut_origin_z = mmlut.origin[2]
        mmlut_nr = mmlut.nr
        mmlut_nz = mmlut.nz
        mmlut_rw = mmlut.rw
        mmlut_data = mmlut.data

    return _flat_image_coord_core(
        pos[0],
        pos[1],
        pos[2],
        ext_x0,
        ext_y0,
        ext_z0,
        ext_dm,
        int_cc,
        glass_vec_x,
        glass_vec_y,
        glass_vec_z,
        mm_n1,
        mm_n2_0,
        mm_n3,
        mm_d0,
        has_mmlut,
        mmlut_origin_x,
        mmlut_origin_y,
        mmlut_origin_z,
        mmlut_nr,
        mmlut_nz,
        mmlut_rw,
        mmlut_data,
    )


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
    if ext_z0 is None and hasattr(ext_x0_or_cal, "ext_par"):
        cal = ext_x0_or_cal
        mm = ext_y0_or_mm
        mmlut = cal.mmlut if cal.mmlut.is_initialized else None
        return _img_coord_params(
            pos,
            cal.ext_par.x0,
            cal.ext_par.y0,
            cal.ext_par.z0,
            cal.ext_par.dm,
            cal.int_par.cc,
            cal.int_par.xh,
            cal.int_par.yh,
            cal.glass_par.vec_x,
            cal.glass_par.vec_y,
            cal.glass_par.vec_z,
            mm.n1,
            mm.n2[0],
            mm.n3,
            mm.d[0],
            cal.added_par.k1,
            cal.added_par.k2,
            cal.added_par.k3,
            cal.added_par.p1,
            cal.added_par.p2,
            cal.added_par.scx,
            cal.added_par.she,
            mmlut=mmlut,
        )
    return _img_coord_params(
        pos,
        ext_x0_or_cal,
        ext_y0_or_mm,
        ext_z0,
        ext_dm,
        int_cc,
        int_xh,
        int_yh,
        glass_vec_x,
        glass_vec_y,
        glass_vec_z,
        mm_n1,
        mm_n2_0,
        mm_n3,
        mm_d0,
        k1,
        k2,
        k3,
        p1,
        p2,
        scx,
        she,
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
    has_mmlut: cython.bint = False
    mmlut_origin_x: cython.double = 0.0
    mmlut_origin_y: cython.double = 0.0
    mmlut_origin_z: cython.double = 0.0
    mmlut_nr: cython.int = 0
    mmlut_nz: cython.int = 0
    mmlut_rw: cython.double = 0.0
    mmlut_data: cython.double[:] = _DUMMY_MMLUT_DATA

    if mmlut is not None and mmlut.data is not None:
        has_mmlut = True
        mmlut_origin_x = mmlut.origin[0]
        mmlut_origin_y = mmlut.origin[1]
        mmlut_origin_z = mmlut.origin[2]
        mmlut_nr = mmlut.nr
        mmlut_nz = mmlut.nz
        mmlut_rw = mmlut.rw
        mmlut_data = mmlut.data

    x: cython.double
    y: cython.double
    x, y = _flat_image_coord_core(
        pos[0],
        pos[1],
        pos[2],
        ext_x0,
        ext_y0,
        ext_z0,
        ext_dm,
        int_cc,
        glass_vec_x,
        glass_vec_y,
        glass_vec_z,
        mm_n1,
        mm_n2_0,
        mm_n3,
        mm_d0,
        has_mmlut,
        mmlut_origin_x,
        mmlut_origin_y,
        mmlut_origin_z,
        mmlut_nr,
        mmlut_nz,
        mmlut_rw,
        mmlut_data,
    )
    return _flat_to_dist_core(x, y, int_xh, int_yh, k1, k2, k3, p1, p2, scx, she)


def img_coord_typed(pos: cython.double[:], cal, mm):
    """Fast typed path: pos must be a contiguous float64 array (no conversion).

    Same as img_coord but skips the np.ascontiguousarray check.
    Use in hot loops where the caller already has an ndarray view.
    """
    mmlut = cal.mmlut if cal.mmlut.is_initialized else None
    return _img_coord_params(
        pos,
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        cal.ext_par.dm,
        cal.int_par.cc,
        cal.int_par.xh,
        cal.int_par.yh,
        cal.glass_par.vec_x,
        cal.glass_par.vec_y,
        cal.glass_par.vec_z,
        mm.n1,
        mm.n2[0],
        mm.n3,
        mm.d[0],
        cal.added_par.k1,
        cal.added_par.k2,
        cal.added_par.k3,
        cal.added_par.p1,
        cal.added_par.p2,
        cal.added_par.scx,
        cal.added_par.she,
        mmlut=mmlut,
    )


def img_coord_batch(positions, cal, mm):
    """Project N 3D positions to distorted metric image coordinates.

    Args:
        positions: (N, 3) array-like of 3D positions (any object
            np.ascontiguousarray accepts, e.g. a plain list of lists — not
            just an ndarray already satisfying the buffer protocol).
        cal: Calibration object.
        mm: MmNp multimedia parameters.

    Returns:
        (N, 2) array of distorted metric coordinates.
    """
    # Coerce unconditionally, not just `if not cython.compiled`: the
    # compiled _img_coord_batch_impl's cython.double[:, :] parameter only
    # accepts buffer-protocol objects, so a plain list must be converted
    # here regardless of engine -- the old compiled-only skip left list
    # input raising a raw memoryview TypeError instead of working.
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    return _img_coord_batch_impl(positions, cal, mm)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def _img_coord_batch_impl(positions: cython.double[:, :], cal, mm) -> object:
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

    has_mmlut: cython.bint = False
    mmlut_origin_x: cython.double = 0.0
    mmlut_origin_y: cython.double = 0.0
    mmlut_origin_z: cython.double = 0.0
    mmlut_nr: cython.int = 0
    mmlut_nz: cython.int = 0
    mmlut_rw: cython.double = 0.0
    mmlut_data: cython.double[:] = _DUMMY_MMLUT_DATA

    if mmlut is not None and mmlut.data is not None:
        has_mmlut = True
        mmlut_origin_x = mmlut.origin[0]
        mmlut_origin_y = mmlut.origin[1]
        mmlut_origin_z = mmlut.origin[2]
        mmlut_nr = mmlut.nr
        mmlut_nz = mmlut.nz
        mmlut_rw = mmlut.rw
        mmlut_data = mmlut.data

    # Hoist point-independent calculations from flat_image_coord
    gx: cython.double = glass_vec_x
    gy: cython.double = glass_vec_y
    gz: cython.double = glass_vec_z

    dist_o_glas: cython.double = c_sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog: cython.double = 1.0 / dist_o_glas

    dot_cam: cython.double = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_cam_glas: cython.double = dot_cam * inv_dog - dist_o_glas - mm_d0

    s_cam: cython.double = dist_cam_glas * inv_dog
    cc_x: cython.double = ext_x0 - gx * s_cam
    cc_y: cython.double = ext_y0 - gy * s_cam
    cc_z: cython.double = ext_z0 - gz * s_cam

    ext_t_z0: cython.double = dist_cam_glas + mm_d0

    s_d: cython.double = mm_d0 * inv_dog
    ag_x: cython.double = cc_x - gx * s_d
    ag_y: cython.double = cc_y - gy * s_d
    ag_z: cython.double = cc_z - gz * s_d

    # Hoist point-independent calculations from flat_to_dist
    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)

    i: cython.int
    pos0: cython.double
    pos1: cython.double
    pos2: cython.double
    dot_pos: cython.double
    dist_point_glas: cython.double
    s_pt: cython.double
    cp_x: cython.double
    cp_y: cython.double
    cp_z: cython.double
    tmp_x: cython.double
    tmp_y: cython.double
    tmp_z: cython.double
    pos_t_0: cython.double
    pos_t_2: cython.double
    mmf: cython.double
    radial_shift: cython.double
    X_t: cython.double
    inv_ngl: cython.double
    n_ve: cython.double
    s_z: cython.double
    bx: cython.double
    by: cython.double
    bz: cython.double
    s_x: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    deno: cython.double
    flat_x: cython.double
    flat_y: cython.double

    x_pt: cython.double
    y_pt: cython.double
    r_pt: cython.double
    r2: cython.double
    r4: cython.double
    r6: cython.double
    radial_factor: cython.double
    x_dist: cython.double
    y_dist: cython.double

    it: cython.int
    beta1: cython.double
    sin_beta1: cython.double
    beta2: cython.double
    beta3: cython.double
    rbeta: cython.double
    rdiff: cython.double
    arg: cython.double
    r_val: cython.double
    rq: cython.double

    for i in range(n):
        pos0 = positions[i, 0]
        pos1 = positions[i, 1]
        pos2 = positions[i, 2]

        dot_pos = pos0 * gx + pos1 * gy + pos2 * gz
        dist_point_glas = dot_pos * inv_dog - dist_o_glas

        s_pt = dist_point_glas * inv_dog
        cp_x = pos0 - gx * s_pt
        cp_y = pos1 - gy * s_pt
        cp_z = pos2 - gz * s_pt

        tmp_x = cp_x - ag_x
        tmp_y = cp_y - ag_y
        tmp_z = cp_z - ag_z

        pos_t_0 = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
        pos_t_2 = dist_point_glas

        # === mmlut lookup ===
        mmf = 1.0
        if has_mmlut:
            mmf = _get_mmf_from_mmlut_core(
                pos_t_0,
                0.0,
                pos_t_2,
                mmlut_origin_x,
                mmlut_origin_y,
                mmlut_origin_z,
                mmlut_nr,
                mmlut_nz,
                mmlut_rw,
                mmlut_data,
            )
            if mmf <= 0.0:
                mmf = 1.0

        # === multimedia nlay ===
        radial_shift = 1.0
        if mmf > 0.0 and mmf != 1.0:
            radial_shift = mmf
        else:
            if mm_n1 == 1.0 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
                radial_shift = 1.0
            else:
                r_val = pos_t_0
                rq = r_val

                for it in range(40):
                    beta1 = c_atan(rq / (ext_t_z0 - pos_t_2))
                    sin_beta1 = c_sin(beta1)

                    arg = sin_beta1 * mm_n1 / mm_n2_0
                    if arg > 1.0:
                        arg = 1.0
                    elif arg < -1.0:
                        arg = -1.0
                    beta2 = c_asin(arg)

                    arg = sin_beta1 * mm_n1 / mm_n3
                    if arg > 1.0:
                        arg = 1.0
                    elif arg < -1.0:
                        arg = -1.0
                    beta3 = c_asin(arg)

                    rbeta = (
                        (ext_t_z0 - mm_d0) * c_tan(beta1)
                        - pos_t_2 * c_tan(beta3)
                        + mm_d0 * c_tan(beta2)
                    )
                    rdiff = r_val - rbeta
                    rq += rdiff

                    if abs(rdiff) < 0.001:
                        break
                else:
                    rq = r_val

                if r_val != 0.0:
                    radial_shift = rq / r_val
                else:
                    radial_shift = 1.0

        X_t = pos_t_0 * radial_shift

        # === back_trans_point ===
        inv_ngl = inv_dog
        n_ve = pos_t_0

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

        deno = ext_dm[0, 2] * dx + ext_dm[1, 2] * dy + ext_dm[2, 2] * dz

        flat_x = (
            -int_cc * (ext_dm[0, 0] * dx + ext_dm[1, 0] * dy + ext_dm[2, 0] * dz) / deno
        )
        flat_y = (
            -int_cc * (ext_dm[0, 1] * dx + ext_dm[1, 1] * dy + ext_dm[2, 1] * dz) / deno
        )

        # === flat to dist ===
        x_pt = flat_x + int_xh
        y_pt = flat_y + int_yh

        r_pt = c_sqrt(x_pt * x_pt + y_pt * y_pt)
        if r_pt < 1e-10:
            res_mv[i, 0] = 0.0
            res_mv[i, 1] = 0.0
        else:
            r2 = r_pt * r_pt
            r4 = r2 * r2
            r6 = r4 * r2
            radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

            x_dist = (
                x_pt * radial_factor
                + p1 * (r2 + 2.0 * x_pt * x_pt)
                + 2.0 * p2 * x_pt * y_pt
            )
            y_dist = (
                y_pt * radial_factor
                + p2 * (r2 + 2.0 * y_pt * y_pt)
                + 2.0 * p1 * x_pt * y_pt
            )

            res_mv[i, 0] = scx * (x_dist - sin_she * y_dist)
            res_mv[i, 1] = scx * cos_she * y_dist

    return result


def flat_image_coord_batch(positions, cal, mm):
    """Project N 3D positions to flat metric image coordinates.

    Args:
        positions: (N, 3) array-like of 3D positions (any object
            np.ascontiguousarray accepts, e.g. a plain list of lists — not
            just an ndarray already satisfying the buffer protocol).
        cal: Calibration object.
        mm: MmNp multimedia parameters.

    Returns:
        (N, 2) array of flat (undistorted) metric coordinates.
    """
    # See img_coord_batch's comment: coerce unconditionally so list input
    # works compiled too, not just in pure-Python mode.
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    return _flat_image_coord_batch_impl(positions, cal, mm)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def _flat_image_coord_batch_impl(positions: cython.double[:, :], cal, mm) -> object:
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

    has_mmlut: cython.bint = False
    mmlut_origin_x: cython.double = 0.0
    mmlut_origin_y: cython.double = 0.0
    mmlut_origin_z: cython.double = 0.0
    mmlut_nr: cython.int = 0
    mmlut_nz: cython.int = 0
    mmlut_rw: cython.double = 0.0
    mmlut_data: cython.double[:] = _DUMMY_MMLUT_DATA

    if mmlut is not None and mmlut.data is not None:
        has_mmlut = True
        mmlut_origin_x = mmlut.origin[0]
        mmlut_origin_y = mmlut.origin[1]
        mmlut_origin_z = mmlut.origin[2]
        mmlut_nr = mmlut.nr
        mmlut_nz = mmlut.nz
        mmlut_rw = mmlut.rw
        mmlut_data = mmlut.data

    # Hoist point-independent calculations from flat_image_coord
    gx: cython.double = glass_vec_x
    gy: cython.double = glass_vec_y
    gz: cython.double = glass_vec_z

    dist_o_glas: cython.double = c_sqrt(gx * gx + gy * gy + gz * gz)
    inv_dog: cython.double = 1.0 / dist_o_glas

    dot_cam: cython.double = ext_x0 * gx + ext_y0 * gy + ext_z0 * gz
    dist_cam_glas: cython.double = dot_cam * inv_dog - dist_o_glas - mm_d0

    s_cam: cython.double = dist_cam_glas * inv_dog
    cc_x: cython.double = ext_x0 - gx * s_cam
    cc_y: cython.double = ext_y0 - gy * s_cam
    cc_z: cython.double = ext_z0 - gz * s_cam

    ext_t_z0: cython.double = dist_cam_glas + mm_d0

    s_d: cython.double = mm_d0 * inv_dog
    ag_x: cython.double = cc_x - gx * s_d
    ag_y: cython.double = cc_y - gy * s_d
    ag_z: cython.double = cc_z - gz * s_d

    i: cython.int
    pos0: cython.double
    pos1: cython.double
    pos2: cython.double
    dot_pos: cython.double
    dist_point_glas: cython.double
    s_pt: cython.double
    cp_x: cython.double
    cp_y: cython.double
    cp_z: cython.double
    tmp_x: cython.double
    tmp_y: cython.double
    tmp_z: cython.double
    pos_t_0: cython.double
    pos_t_2: cython.double
    mmf: cython.double
    radial_shift: cython.double
    X_t: cython.double
    inv_ngl: cython.double
    n_ve: cython.double
    s_z: cython.double
    bx: cython.double
    by: cython.double
    bz: cython.double
    s_x: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    deno: cython.double

    it: cython.int
    beta1: cython.double
    sin_beta1: cython.double
    beta2: cython.double
    beta3: cython.double
    rbeta: cython.double
    rdiff: cython.double
    arg: cython.double
    r_val: cython.double
    rq: cython.double

    for i in range(n):
        pos0 = positions[i, 0]
        pos1 = positions[i, 1]
        pos2 = positions[i, 2]

        dot_pos = pos0 * gx + pos1 * gy + pos2 * gz
        dist_point_glas = dot_pos * inv_dog - dist_o_glas

        s_pt = dist_point_glas * inv_dog
        cp_x = pos0 - gx * s_pt
        cp_y = pos1 - gy * s_pt
        cp_z = pos2 - gz * s_pt

        tmp_x = cp_x - ag_x
        tmp_y = cp_y - ag_y
        tmp_z = cp_z - ag_z

        pos_t_0 = c_sqrt(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z)
        pos_t_2 = dist_point_glas

        # === mmlut lookup ===
        mmf = 1.0
        if has_mmlut:
            mmf = _get_mmf_from_mmlut_core(
                pos_t_0,
                0.0,
                pos_t_2,
                mmlut_origin_x,
                mmlut_origin_y,
                mmlut_origin_z,
                mmlut_nr,
                mmlut_nz,
                mmlut_rw,
                mmlut_data,
            )
            if mmf <= 0.0:
                mmf = 1.0

        # === multimedia nlay ===
        radial_shift = 1.0
        if mmf > 0.0 and mmf != 1.0:
            radial_shift = mmf
        else:
            if mm_n1 == 1.0 and mm_n2_0 == 1.0 and mm_n3 == 1.0:
                radial_shift = 1.0
            else:
                r_val = pos_t_0
                rq = r_val

                for it in range(40):
                    beta1 = c_atan(rq / (ext_t_z0 - pos_t_2))
                    sin_beta1 = c_sin(beta1)

                    arg = sin_beta1 * mm_n1 / mm_n2_0
                    if arg > 1.0:
                        arg = 1.0
                    elif arg < -1.0:
                        arg = -1.0
                    beta2 = c_asin(arg)

                    arg = sin_beta1 * mm_n1 / mm_n3
                    if arg > 1.0:
                        arg = 1.0
                    elif arg < -1.0:
                        arg = -1.0
                    beta3 = c_asin(arg)

                    rbeta = (
                        (ext_t_z0 - mm_d0) * c_tan(beta1)
                        - pos_t_2 * c_tan(beta3)
                        + mm_d0 * c_tan(beta2)
                    )
                    rdiff = r_val - rbeta
                    rq += rdiff

                    if abs(rdiff) < 0.001:
                        break
                else:
                    rq = r_val

                if r_val != 0.0:
                    radial_shift = rq / r_val
                else:
                    radial_shift = 1.0

        X_t = pos_t_0 * radial_shift

        # === back_trans_point ===
        inv_ngl = inv_dog
        n_ve = pos_t_0

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

        deno = ext_dm[0, 2] * dx + ext_dm[1, 2] * dy + ext_dm[2, 2] * dz

        res_mv[i, 0] = (
            -int_cc * (ext_dm[0, 0] * dx + ext_dm[1, 0] * dy + ext_dm[2, 0] * dz) / deno
        )
        res_mv[i, 1] = (
            -int_cc * (ext_dm[0, 1] * dx + ext_dm[1, 1] * dy + ext_dm[2, 1] * dz) / deno
        )

    return result


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
