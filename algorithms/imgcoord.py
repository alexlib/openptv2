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
    pos_t, cross_p, cross_c = trans_cam_point(
        pos, ext_x0, ext_y0, ext_z0,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    # Get radial shift
    X_t, Y_t = multimed_nlay(
        pos_t[0], pos_t[1], pos_t[2],
        ext_x0, ext_y0, mm_n1, mm_n2_0, mm_n3, mm_d0,
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
    pos: np.ndarray,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    ext_dm: np.ndarray,
    int_cc: float,
    int_xh: float,
    int_yh: float,
    glass_vec_x: float,
    glass_vec_y: float,
    glass_vec_z: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    scx: float,
    she: float,
) -> tuple[float, float]:
    """Project 3D position to distorted metric image coordinates.

    Uses flat_image_coord then applies Brown distortion model.

    Args:
        pos: 3D world position (x, y, z).
        ext_x0, ext_y0, ext_z0: camera center.
        ext_dm: 3x3 rotation matrix.
        int_cc: camera constant.
        int_xh, int_yh: principal point.
        glass_vec_x, glass_vec_y, glass_vec_z: glass normal.
        mm_n1, mm_n2_0, mm_n3, mm_d0: multimedia parameters.
        k1, k2, k3, p1, p2, scx, she: distortion parameters.

    Returns:
        (x, y) distorted metric coordinates.
    """
    x, y = flat_image_coord(
        pos,
        ext_x0, ext_y0, ext_z0, ext_dm, int_cc,
        glass_vec_x, glass_vec_y, glass_vec_z,
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    # Apply distortion
    return _flat_to_dist(x, y, int_xh, int_yh, k1, k2, k3, p1, p2, scx, she)
