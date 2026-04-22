"""Ray tracing through multi-media interfaces.

Translation of lib/src/ray_tracing.c and lib/include/ray_tracing.h.

Traces optical rays through multi-media interfaces (air-glass-water)
using Snell's law and returns the crossing point and direction vector.
"""

import math
import numpy as np
from .vec_utils import vec_set, unit_vector, vec_dot, vec_norm, vec_scalar_mul, vec_add, vec_subt


def ray_tracing(
    x: float,
    y: float,
    ext_dm: np.ndarray,
    ext_x0: float,
    ext_y0: float,
    ext_z0: float,
    int_cc: float,
    glass_vec_x: float,
    glass_vec_y: float,
    glass_vec_z: float,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    tmp1 = np.array([x, y, -int_cc], dtype=np.float64)
    tmp1 = unit_vector(tmp1)

    # Transform to global coordinate system
    start_dir = ext_dm @ tmp1

    # Primary point (camera center)
    primary_point = np.array([ext_x0, ext_y0, ext_z0], dtype=np.float64)

    # Glass normal (unit vector)
    glass_vec = np.array([glass_vec_x, glass_vec_y, glass_vec_z], dtype=np.float64)
    glass_dir = unit_vector(glass_vec)
    c = vec_norm(glass_vec) + mm_d0

    # Project start ray on glass vector to find n1/n2 interface
    dist_cam_glass = vec_dot(glass_dir, primary_point) - c
    d1 = -dist_cam_glass / vec_dot(glass_dir, start_dir)

    tmp1 = vec_scalar_mul(start_dir, d1)
    Xb = vec_add(primary_point, tmp1)

    # Decompose ray into glass-normal and glass-parallel components
    n = vec_dot(start_dir, glass_dir)
    tmp1 = vec_scalar_mul(glass_dir, n)
    tmp2 = vec_subt(start_dir, tmp1)
    bp = unit_vector(tmp2)

    # Transform direction inside glass using Snell's law
    p = math.sqrt(1 - n * n) * mm_n1 / mm_n2_0  # glass parallel
    n_glass = -math.sqrt(1 - p * p)  # glass normal

    # Propagation length in glass
    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n_glass)
    a2 = vec_add(tmp1, tmp2)
    d2 = mm_d0 / abs(vec_dot(glass_dir, a2))

    # Point on horizontal plane between n2, n3
    tmp1 = vec_scalar_mul(a2, d2)
    X = vec_add(Xb, tmp1)

    # Direction in next medium
    n_a2 = vec_dot(a2, glass_dir)
    tmp2_for_sub = vec_scalar_mul(glass_dir, n_glass)  # reuse n_glass component
    tmp2 = vec_subt(a2, tmp2_for_sub)
    bp = unit_vector(tmp2)

    p = math.sqrt(1 - n_a2 * n_a2)
    p = p * mm_n2_0 / mm_n3
    n_final = -math.sqrt(1 - p * p)

    tmp1 = vec_scalar_mul(bp, p)
    tmp2 = vec_scalar_mul(glass_dir, n_final)
    out = vec_add(tmp1, tmp2)

    return X, out


def ray_tracing_batch(xy, cal, mm):
    """Trace N rays through multi-media interface.

    Uses Numba JIT when available for parallel acceleration.

    Args:
        xy: (N, 2) array of metric image coordinates.
        cal: Calibration object.
        mm: MmNp multimedia parameters.

    Returns:
        (positions, directions) — each (N, 3) float64 arrays.
    """
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    try:
        from .track_kernels import (
            HAS_NUMBA, ray_tracing_batch_jit, pack_cal_array,
        )
        if HAS_NUMBA:
            cal_arr = pack_cal_array(cal, mm)
            return ray_tracing_batch_jit(xy, cal_arr)
    except ImportError:
        pass
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
