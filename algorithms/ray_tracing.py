"""Ray tracing through multi-media interfaces.

Translation of lib/src/ray_tracing.c and lib/include/ray_tracing.h.

Traces optical rays through multi-media interfaces (air-glass-water)
using Snell's law and returns the crossing point and direction vector.
"""

import math
import cython
import numpy as np


@cython.ccall
@cython.locals(
    norm_tmp1=cython.double, tx=cython.double, ty=cython.double, tz=cython.double,
    start_dir_x=cython.double, start_dir_y=cython.double, start_dir_z=cython.double,
    norm_glass=cython.double, glass_dir_x=cython.double, glass_dir_y=cython.double, glass_dir_z=cython.double,
    c=cython.double, dist_cam_glass=cython.double, dot_glass_start=cython.double, d1=cython.double,
    Xb_x=cython.double, Xb_y=cython.double, Xb_z=cython.double,
    n=cython.double, bp_x=cython.double, bp_y=cython.double, bp_z=cython.double, norm_bp=cython.double,
    p=cython.double, n_glass=cython.double,
    a2_x=cython.double, a2_y=cython.double, a2_z=cython.double,
    dot_glass_a2=cython.double, d2=cython.double,
    X_x=cython.double, X_y=cython.double, X_z=cython.double,
    n_a2=cython.double, n_final=cython.double,
    out_x=cython.double, out_y=cython.double, out_z=cython.double
)
def ray_tracing(
    x: cython.double,
    y: cython.double,
    ext_dm: cython.double[:, :],
    ext_x0: cython.double,
    ext_y0: cython.double,
    ext_z0: cython.double,
    int_cc: cython.double,
    glass_vec_x: cython.double,
    glass_vec_y: cython.double,
    glass_vec_z: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
) -> tuple:
    """Trace a ray through multi-media interface.

    Traces the optical ray through layers (typically air-glass-water)
    and returns the position of the ray crossing point and the direction
    vector in the final medium.

    Args:
        x, y: metric position in image space.
        ext_dm: 3x3 rotation matrix of camera.
        ext_x0, ext_y0, ext_z0: camera center position.
        int_cc: camera constant (focal length).
        glass_vec_x, glass_vec_y, glass_vec_z: glass normal vector.
        mm_n1: refractive index of first medium (air).
        mm_n2_0: refractive index of second medium (glass).
        mm_n3: refractive index of third medium (water).
        mm_d0: thickness of glass layer.

    Returns:
        (X, out) where X is crossing point position (3,) and out is
        direction vector in final medium (3,).
    """
    # Initial ray direction in camera coordinate system
    norm_tmp1 = math.sqrt(x * x + y * y + int_cc * int_cc)
    tx = x / norm_tmp1
    ty = y / norm_tmp1
    tz = -int_cc / norm_tmp1

    # Transform to global coordinate system (ext_dm @ tmp1)
    start_dir_x = ext_dm[0, 0] * tx + ext_dm[0, 1] * ty + ext_dm[0, 2] * tz
    start_dir_y = ext_dm[1, 0] * tx + ext_dm[1, 1] * ty + ext_dm[1, 2] * tz
    start_dir_z = ext_dm[2, 0] * tx + ext_dm[2, 1] * ty + ext_dm[2, 2] * tz

    # Glass normal (unit vector)
    norm_glass = math.sqrt(glass_vec_x * glass_vec_x + glass_vec_y * glass_vec_y + glass_vec_z * glass_vec_z)
    glass_dir_x = glass_vec_x / norm_glass
    glass_dir_y = glass_vec_y / norm_glass
    glass_dir_z = glass_vec_z / norm_glass
    c = norm_glass + mm_d0

    # Project start ray on glass vector to find n1/n2 interface
    dist_cam_glass = (glass_dir_x * ext_x0 + glass_dir_y * ext_y0 + glass_dir_z * ext_z0) - c
    dot_glass_start = glass_dir_x * start_dir_x + glass_dir_y * start_dir_y + glass_dir_z * start_dir_z
    d1 = -dist_cam_glass / dot_glass_start

    # Xb = primary_point + start_dir * d1
    Xb_x = ext_x0 + start_dir_x * d1
    Xb_y = ext_y0 + start_dir_y * d1
    Xb_z = ext_z0 + start_dir_z * d1

    # Decompose ray into glass-normal and glass-parallel components
    n = start_dir_x * glass_dir_x + start_dir_y * glass_dir_y + start_dir_z * glass_dir_z
    bp_x = start_dir_x - glass_dir_x * n
    bp_y = start_dir_y - glass_dir_y * n
    bp_z = start_dir_z - glass_dir_z * n
    norm_bp = math.sqrt(bp_x * bp_x + bp_y * bp_y + bp_z * bp_z)
    if norm_bp > 0:
        bp_x /= norm_bp
        bp_y /= norm_bp
        bp_z /= norm_bp

    # Transform direction inside glass using Snell's law
    p = math.sqrt(1.0 - n * n) * mm_n1 / mm_n2_0  # glass parallel
    n_glass = -math.sqrt(1.0 - p * p)  # glass normal

    # Propagation length in glass
    a2_x = bp_x * p + glass_dir_x * n_glass
    a2_y = bp_y * p + glass_dir_y * n_glass
    a2_z = bp_z * p + glass_dir_z * n_glass
    
    dot_glass_a2 = glass_dir_x * a2_x + glass_dir_y * a2_y + glass_dir_z * a2_z
    d2 = mm_d0 / abs(dot_glass_a2)

    # Point X on horizontal plane between n2, n3
    X_x = Xb_x + a2_x * d2
    X_y = Xb_y + a2_y * d2
    X_z = Xb_z + a2_z * d2

    # Direction in next medium
    n_a2 = a2_x * glass_dir_x + a2_y * glass_dir_y + a2_z * glass_dir_z
    bp_x = a2_x - glass_dir_x * n_glass
    bp_y = a2_y - glass_dir_y * n_glass
    bp_z = a2_z - glass_dir_z * n_glass
    norm_bp = math.sqrt(bp_x * bp_x + bp_y * bp_y + bp_z * bp_z)
    if norm_bp > 0:
        bp_x /= norm_bp
        bp_y /= norm_bp
        bp_z /= norm_bp

    p = math.sqrt(1.0 - n_a2 * n_a2)
    p = p * mm_n2_0 / mm_n3
    n_final = -math.sqrt(1.0 - p * p)

    out_x = bp_x * p + glass_dir_x * n_final
    out_y = bp_y * p + glass_dir_y * n_final
    out_z = bp_z * p + glass_dir_z * n_final

    return (
        np.array([X_x, X_y, X_z], dtype=np.float64),
        np.array([out_x, out_y, out_z], dtype=np.float64)
    )


def ray_tracing_batch(xy, cal, mm):
    """Trace N rays through multi-media interface.

    Args:
        xy: (N, 2) array of metric image coordinates.
        cal: Calibration object.
        mm: MmNp multimedia parameters.

    Returns:
        (positions, directions) — each (N, 3) float64 arrays.
    """
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    n = xy.shape[0]
    positions = np.empty((n, 3), dtype=np.float64)
    directions = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        pos, d = ray_tracing(
            xy[i, 0], xy[i, 1],
            cal.ext_par.dm,
            cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
            cal.int_par.cc,
            cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
            mm.n1, mm.n2[0], mm.n3, mm.d[0],
        )
        positions[i] = pos
        directions[i] = d
    return positions, directions


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
