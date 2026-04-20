"""Image coordinate projection from 3D world positions.

Translation of lib/src/imgcoord.c and lib/include/imgcoord.h.

Projects 3D world positions to 2D image coordinates, with or without
distortion modeling.
"""

import numpy as np
from .trafo import flat_to_dist as _flat_to_dist


def flat_image_coord(
    pos: np.ndarray,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    ext_dm: np.ndarray,
    int_cc: float,
    glass_vec_x: float,
    glass_vec_y: float,
    glass_vec_z: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
) -> tuple[float, float]:
    """Project 3D position to undistorted metric image coordinates.

    Calculates projection from world space to metric coordinates
    in image space without distortions.

    Args:
        pos: 3D world position (x, y, z).
        ext_x0, ext_y0, ext_z0: camera center.
        ext_dm: 3x3 rotation matrix.
        int_cc: camera constant (focal length).
        glass_vec_x, glass_vec_y, glass_vec_z: glass normal.
        mm_n1, mm_n2_0, mm_n3, mm_d0: multimedia parameters.

    Returns:
        (x, y) undistorted metric coordinates.
    """
    from .multimed import trans_cam_point, back_trans_point, multimed_nlay

    # Transform through multimedia interface
    pos_t, cross_p, cross_c, ext_t_z0 = trans_cam_point(
        pos, ext_x0, ext_y0, ext_z0,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    # Get radial shift using transformed camera center (0, 0, ext_t_z0)
    X_t, Y_t = multimed_nlay(
        pos_t[0], pos_t[1], pos_t[2],
        0.0, 0.0, ext_t_z0, mm_n1, mm_n2_0, mm_n3, mm_d0,
    )
    pos_t = np.array([X_t, Y_t, pos_t[2]])

    # Transform back
    pos = back_trans_point(
        pos_t, cross_p, cross_c,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    # Perspective projection
    dx = pos[0] - ext_x0
    dy = pos[1] - ext_y0
    dz = pos[2] - ext_z0

    deno = ext_dm[0, 2] * dx + ext_dm[1, 2] * dy + ext_dm[2, 2] * dz

    x = -int_cc * (ext_dm[0, 0] * dx + ext_dm[1, 0] * dy + ext_dm[2, 0] * dz) / deno
    y = -int_cc * (ext_dm[0, 1] * dx + ext_dm[1, 1] * dy + ext_dm[2, 1] * dz) / deno

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
):
    x, y = flat_image_coord(
        pos,
        ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )
    return _flat_to_dist(x, y, int_xh, int_yh, k1, k2, k3, p1, p2, scx, she)
