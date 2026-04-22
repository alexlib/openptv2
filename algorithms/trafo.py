"""Coordinate transformations between pixel, metric, and flat-image systems.

Translation of lib/src/trafo.c and lib/include/trafo.h.

Transforms between:
- Pixel coordinates: raw image coordinates (0..imx, 0..imy)
- Metric coordinates: physical coordinates on sensor (mm), centered
- Flat-image coordinates: undistorted metric coordinates (ideal pinhole)
- Distorted coordinates: metric coordinates with Brown distortion applied
"""

import math
import numpy as np
from dataclasses import dataclass


# Y-remap mode constants (for interlaced cameras)
NO_REMAP = 0
DOUBLED_PLUS_ONE = 1
DOUBLED = 2


def old_pixel_to_metric(
    x_pixel: float,
    y_pixel: float,
    im_size_x: int,
    im_size_y: int,
    pix_size_x: float,
    pix_size_y: float,
    y_remap_mode: int = NO_REMAP,
) -> tuple[float, float]:
    """Convert pixel coordinates to metric coordinates.

    Args:
        x_pixel, y_pixel: input pixel coordinates.
        im_size_x, im_size_y: image dimensions in pixels.
        pix_size_x, pix_size_y: pixel size in mm.
        y_remap_mode: 0=normal, 1=odd lines, 2=even lines (interlaced).

    Returns:
        (x_metric, y_metric) tuple.
    """
    # Apply y remapping
    if y_remap_mode == DOUBLED_PLUS_ONE:
        y_pixel = 2.0 * y_pixel + 1.0
    elif y_remap_mode == DOUBLED:
        y_pixel = 2.0 * y_pixel

    x_metric = (x_pixel - im_size_x / 2.0) * pix_size_x
    y_metric = (im_size_y / 2.0 - y_pixel) * pix_size_y

    return x_metric, y_metric


def old_metric_to_pixel(
    x_metric: float,
    y_metric: float,
    im_size_x: int,
    im_size_y: int,
    pix_size_x: float,
    pix_size_y: float,
    y_remap_mode: int = NO_REMAP,
) -> tuple[float, float]:
    """Convert metric coordinates to pixel coordinates.

    Args:
        x_metric, y_metric: input metric coordinates.
        im_size_x, im_size_y: image dimensions in pixels.
        pix_size_x, pix_size_y: pixel size in mm.
        y_remap_mode: 0=normal, 1=odd lines, 2=even lines (interlaced).

    Returns:
        (x_pixel, y_pixel) tuple.
    """
    x_pixel = x_metric / pix_size_x + im_size_x / 2.0
    y_pixel = im_size_y / 2.0 - y_metric / pix_size_y

    # Apply y remapping (inverse)
    if y_remap_mode == DOUBLED_PLUS_ONE:
        y_pixel = (y_pixel - 1.0) / 2.0
    elif y_remap_mode == DOUBLED:
        y_pixel = y_pixel / 2.0

    return x_pixel, y_pixel


def pixel_to_metric(
    x_pixel,
    y_pixel=None,
    imx_or_cpar=None,
    imy=None,
    pix_x=None,
    pix_y=None,
    chfield=NO_REMAP,
) -> tuple[float, float]:
    """Convert pixel to metric coordinates.

    Accepts either (x, y, cpar) or (x, y, imx, imy, pix_x, pix_y, chfield).
    """
    if imy is None and hasattr(imx_or_cpar, 'imx'):
        cpar = imx_or_cpar
        return old_pixel_to_metric(x_pixel, y_pixel,
                                   cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield)
    return old_pixel_to_metric(x_pixel, y_pixel, imx_or_cpar, imy, pix_x, pix_y, chfield)


def metric_to_pixel(
    x_metric,
    y_metric=None,
    imx_or_cpar=None,
    imy=None,
    pix_x=None,
    pix_y=None,
    chfield=NO_REMAP,
) -> tuple[float, float]:
    """Convert metric to pixel coordinates.

    Accepts either (x, y, cpar) or (x, y, imx, imy, pix_x, pix_y, chfield).
    """
    if imy is None and hasattr(imx_or_cpar, 'imx'):
        cpar = imx_or_cpar
        return old_metric_to_pixel(x_metric, y_metric,
                                   cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield)
    return old_metric_to_pixel(x_metric, y_metric, imx_or_cpar, imy, pix_x, pix_y, chfield)


def pixel_to_metric_batch(xy, cpar):
    """Convert N pixel coordinates to metric.

    Uses Numba JIT when available for parallel acceleration.

    Args:
        xy: (N, 2) array of pixel coordinates.
        cpar: ControlPar with imx, imy, pix_x, pix_y, chfield.

    Returns:
        (N, 2) array of metric coordinates.
    """
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    try:
        from .track_kernels import HAS_NUMBA, pixel_to_metric_batch_jit
        if HAS_NUMBA:
            return pixel_to_metric_batch_jit(
                xy, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield,
            )
    except ImportError:
        pass
    n = xy.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = old_pixel_to_metric(
            xy[i, 0], xy[i, 1], cpar.imx, cpar.imy,
            cpar.pix_x, cpar.pix_y, cpar.chfield,
        )
    return result


def metric_to_pixel_batch(xy, cpar):
    """Convert N metric coordinates to pixel.

    Uses Numba JIT when available for parallel acceleration.

    Args:
        xy: (N, 2) array of metric coordinates.
        cpar: ControlPar with imx, imy, pix_x, pix_y, chfield.

    Returns:
        (N, 2) array of pixel coordinates.
    """
    xy = np.ascontiguousarray(xy, dtype=np.float64)
    try:
        from .track_kernels import HAS_NUMBA, metric_to_pixel_batch_jit
        if HAS_NUMBA:
            return metric_to_pixel_batch_jit(
                xy, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield,
            )
    except ImportError:
        pass
    n = xy.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        result[i, 0], result[i, 1] = old_metric_to_pixel(
            xy[i, 0], xy[i, 1], cpar.imx, cpar.imy,
            cpar.pix_x, cpar.pix_y, cpar.chfield,
        )
    return result


def distort_brown_affin(
    x: float,
    y: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    scx: float,
    she: float,
) -> tuple[float, float]:
    """Apply Brown distortion to undistorted metric coordinates.

    Transforms ideal pinhole coordinates to real distorted image coordinates.

    Args:
        x, y: undistorted metric coordinates.
        k1, k2, k3: radial distortion coefficients.
        p1, p2: decentering distortion coefficients.
        scx: scale factor.
        she: shear angle.

    Returns:
        (x_distorted, y_distorted) tuple.
    """
    r = math.sqrt(x * x + y * y)

    if r < 1e-10:
        return 0.0, 0.0

    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_dist = x * radial_factor + p1 * (r2 + 2 * x * x) + 2 * p2 * x * y
    y_dist = y * radial_factor + p2 * (r2 + 2 * y * y) + 2 * p1 * x * y

    sin_she = math.sin(she)
    cos_she = math.cos(she)

    x1 = scx * (x_dist - sin_she * y_dist)
    y1 = scx * cos_she * y_dist

    return float(x1), float(y1)


def correct_brown_affin(
    x: float,
    y: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    scx: float,
    she: float,
) -> tuple[float, float]:
    """Inverse Brown distortion (single iteration, for backward compatibility).

    Args:
        x, y: distorted metric coordinates.
        k1, k2, k3: radial distortion coefficients.
        p1, p2: decentering distortion coefficients.
        scx: scale factor.
        she: shear angle.

    Returns:
        (x_flat, y_flat) undistorted coordinates.
    """
    sin_she = math.sin(she)
    cos_she = math.cos(she)
    inv_scx = 1.0 / scx

    # Initial guess: inverse affine transformation
    xq = x * inv_scx
    yq = y * inv_scx / cos_she
    xq += yq * sin_she

    max_iter = 20
    damping = 0.7
    tol = 1e-8

    for _ in range(max_iter):
        xq_old, yq_old = xq, yq

        # Forward distort current guess
        xt, yt = distort_brown_affin(xq, yq, k1, k2, k3, p1, p2, scx, she)

        # Error
        dx = (x - xt) * inv_scx
        dy = (y - yt) * inv_scx

        # Update with damping
        xq += dx * damping
        yq += dy * damping

        # Check convergence
        change = math.sqrt((xq - xq_old) ** 2 + (yq - yq_old) ** 2)
        pos_magnitude = math.sqrt(xq * xq + yq * yq)
        if pos_magnitude > 1e-10 and change / pos_magnitude < tol:
            break

    return xq, yq


def correct_brown_affine_exact(
    x: float,
    y: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    scx: float,
    she: float,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Iteratively solve inverse Brown distortion with full convergence.

    Args:
        x, y: distorted metric coordinates.
        k1, k2, k3: radial distortion coefficients.
        p1, p2: decentering distortion coefficients.
        scx: scale factor.
        she: shear angle.
        tol: convergence tolerance.

    Returns:
        (x_flat, y_flat) undistorted coordinates.
    """
    r_init = math.sqrt(x * x + y * y)

    if r_init < 1e-10:
        return 0.0, 0.0

    sin_she = math.sin(she)
    cos_she = math.cos(she)
    inv_scx = 1.0 / scx

    # Initial guess: inverse affine transformation
    xq = (x + y * sin_she) * inv_scx
    yq = y / cos_she

    max_iter = 50
    damping = 0.5

    for _ in range(max_iter):
        r2 = xq * xq + yq * yq
        r4 = r2 * r2
        r6 = r4 * r2

        radial_factor = k1 * r2 + k2 * r4 + k3 * r6

        dx = xq * radial_factor + p1 * (r2 + 2 * xq * xq) + 2 * p2 * xq * yq
        dy = yq * radial_factor + p2 * (r2 + 2 * yq * yq) + 2 * p1 * xq * yq

        xq_new = (x + y * sin_she) * inv_scx - dx
        yq_new = y / cos_she - dy

        dx_change = xq_new - xq
        dy_change = yq_new - yq

        xq += damping * dx_change
        yq += damping * dy_change

        if math.sqrt(dx_change ** 2 + dy_change ** 2) < tol:
            break

    return xq, yq


def flat_to_dist(
    flat_x: float,
    flat_y: float,
    xh: float,
    yh: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    scx: float,
    she: float,
) -> tuple[float, float]:
    """Convert flat-image to distorted metric coordinates.

    Args:
        flat_x, flat_y: flat-image (undistorted, centered) coordinates.
        xh, yh: principal point (sensor shift).
        k1, k2, k3, p1, p2, scx, she: distortion parameters.

    Returns:
        (dist_x, dist_y) distorted metric coordinates.
    """
    # Make coordinates relative to sensor center
    flat_x += xh
    flat_y += yh

    return distort_brown_affin(flat_x, flat_y, k1, k2, k3, p1, p2, scx, she)


def dist_to_flat(
    dist_x: float,
    dist_y: float,
    xh: float,
    yh: float,
    k1: float,
    k2: float,
    k3: float,
    p1: float,
    p2: float,
    scx: float,
    she: float,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Convert distorted metric to flat-image coordinates.

    Args:
        dist_x, dist_y: distorted metric coordinates.
        xh, yh: principal point (sensor shift).
        k1, k2, k3, p1, p2, scx, she: distortion parameters.
        tol: convergence tolerance.

    Returns:
        (flat_x, flat_y) flat-image coordinates.
    """
    flat_x, flat_y = correct_brown_affine_exact(
        dist_x, dist_y, k1, k2, k3, p1, p2, scx, she, tol
    )
    return flat_x - xh, flat_y - yh
