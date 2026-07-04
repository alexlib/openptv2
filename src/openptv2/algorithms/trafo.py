"""Coordinate transformations between pixel, metric, and flat-image systems.

Translation of lib/src/trafo.c and lib/include/trafo.h.

Transforms between:
- Pixel coordinates: raw image coordinates (0..imx, 0..imy)
- Metric coordinates: physical coordinates on sensor (mm), centered
- Flat-image coordinates: undistorted metric coordinates (ideal pinhole)
- Distorted coordinates: metric coordinates with Brown distortion applied
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt, sin as c_sin, cos as c_cos
else:
    from math import sqrt as c_sqrt, sin as c_sin, cos as c_cos

# Y-remap mode constants (for interlaced cameras)
NO_REMAP: cython.int = 0
DOUBLED_PLUS_ONE: cython.int = 1
DOUBLED: cython.int = 2


@cython.cfunc
@cython.inline
def _old_pixel_to_metric_out(
    x_pixel: cython.double,
    y_pixel: cython.double,
    im_size_x: cython.int,
    im_size_y: cython.int,
    pix_size_x: cython.double,
    pix_size_y: cython.double,
    y_remap_mode: cython.int,
    out: cython.double[:],
):
    """Convert pixel to metric coordinates — _out variant."""
    yp: cython.double = y_pixel
    if y_remap_mode == DOUBLED_PLUS_ONE:
        yp = 2.0 * yp + 1.0
    elif y_remap_mode == DOUBLED:
        yp = 2.0 * yp
    out[0] = (x_pixel - im_size_x * 0.5) * pix_size_x
    out[1] = (im_size_y * 0.5 - yp) * pix_size_y


@cython.ccall
@cython.cdivision(True)
@cython.profile(False)
def old_pixel_to_metric(
    x_pixel: cython.double,
    y_pixel: cython.double,
    im_size_x: cython.int,
    im_size_y: cython.int,
    pix_size_x: cython.double,
    pix_size_y: cython.double,
    y_remap_mode: cython.int = 0,
) -> tuple:
    """Convert pixel coordinates to metric coordinates.

    Args:
        x_pixel, y_pixel: input pixel coordinates.
        im_size_x, im_size_y: image dimensions in pixels.
        pix_size_x, pix_size_y: pixel size in mm.
        y_remap_mode: 0=normal, 1=odd lines, 2=even lines (interlaced).

    Returns:
        (x_metric, y_metric) tuple.
    """
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    _old_pixel_to_metric_out(
        x_pixel,
        y_pixel,
        im_size_x,
        im_size_y,
        pix_size_x,
        pix_size_y,
        y_remap_mode,
        _out_mv,
    )
    return _out_mv[0], _out_mv[1]


@cython.cfunc
@cython.inline
def _old_metric_to_pixel_out(
    x_metric: cython.double,
    y_metric: cython.double,
    im_size_x: cython.int,
    im_size_y: cython.int,
    pix_size_x: cython.double,
    pix_size_y: cython.double,
    y_remap_mode: cython.int,
    out: cython.double[:],
):
    """Convert metric to pixel coordinates — _out variant."""
    xp: cython.double = x_metric / pix_size_x + im_size_x * 0.5
    yp: cython.double = im_size_y * 0.5 - y_metric / pix_size_y
    if y_remap_mode == DOUBLED_PLUS_ONE:
        yp = (yp - 1.0) * 0.5
    elif y_remap_mode == DOUBLED:
        yp = yp * 0.5
    out[0] = xp
    out[1] = yp


@cython.ccall
def old_metric_to_pixel(
    x_metric: cython.double,
    y_metric: cython.double,
    im_size_x: cython.int,
    im_size_y: cython.int,
    pix_size_x: cython.double,
    pix_size_y: cython.double,
    y_remap_mode: cython.int = 0,
) -> tuple:
    """Convert metric coordinates to pixel coordinates.

    Args:
        x_metric, y_metric: input metric coordinates.
        im_size_x, im_size_y: image dimensions in pixels.
        pix_size_x, pix_size_y: pixel size in mm.
        y_remap_mode: 0=normal, 1=odd lines, 2=even lines (interlaced).

    Returns:
        (x_pixel, y_pixel) tuple.
    """
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    _old_metric_to_pixel_out(
        x_metric,
        y_metric,
        im_size_x,
        im_size_y,
        pix_size_x,
        pix_size_y,
        y_remap_mode,
        _out_mv,
    )
    return _out_mv[0], _out_mv[1]


@cython.ccall
def pixel_to_metric(
    x_pixel: cython.double,
    y_pixel: cython.double,
    imx_or_cpar,
    imy: cython.int = 0,
    pix_x: cython.double = 0.0,
    pix_y: cython.double = 0.0,
    chfield: cython.int = 0,
) -> tuple:
    """Convert pixel to metric coordinates.

    Accepts either (x, y, cpar) or (x, y, imx, imy, pix_x, pix_y, chfield).
    """
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    if hasattr(imx_or_cpar, "imx"):
        _old_pixel_to_metric_out(
            x_pixel,
            y_pixel,
            imx_or_cpar.imx,
            imx_or_cpar.imy,
            imx_or_cpar.pix_x,
            imx_or_cpar.pix_y,
            imx_or_cpar.chfield,
            _out_mv,
        )
    else:
        _old_pixel_to_metric_out(
            x_pixel,
            y_pixel,
            int(imx_or_cpar),
            imy,
            pix_x,
            pix_y,
            chfield,
            _out_mv,
        )
    return _out_mv[0], _out_mv[1]


@cython.ccall
def metric_to_pixel(
    x_metric: cython.double,
    y_metric: cython.double,
    imx_or_cpar,
    imy: cython.int = 0,
    pix_x: cython.double = 0.0,
    pix_y: cython.double = 0.0,
    chfield: cython.int = 0,
) -> tuple:
    """Convert metric to pixel coordinates.

    Accepts either (x, y, cpar) or (x, y, imx, imy, pix_x, pix_y, chfield).
    """
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    if hasattr(imx_or_cpar, "imx"):
        _old_metric_to_pixel_out(
            x_metric,
            y_metric,
            imx_or_cpar.imx,
            imx_or_cpar.imy,
            imx_or_cpar.pix_x,
            imx_or_cpar.pix_y,
            imx_or_cpar.chfield,
            _out_mv,
        )
    else:
        _old_metric_to_pixel_out(
            x_metric,
            y_metric,
            int(imx_or_cpar),
            imy,
            pix_x,
            pix_y,
            chfield,
            _out_mv,
        )
    return _out_mv[0], _out_mv[1]


@cython.ccall
def pixel_to_metric_batch(xy, cpar) -> object:
    """Convert N pixel coordinates to metric.

    Args:
        xy: (N, 2) array of pixel coordinates.
        cpar: ControlPar with imx, imy, pix_x, pix_y, chfield.

    Returns:
        (N, 2) array of metric coordinates.
    """
    xy_arr = np.asarray(xy, dtype=np.float64)

    imx: cython.double = cpar.imx
    imy: cython.double = cpar.imy
    pix_x: cython.double = cpar.pix_x
    pix_y: cython.double = cpar.pix_y
    chfield: cython.int = cpar.chfield

    result = np.empty_like(xy_arr)
    result[:, 0] = (xy_arr[:, 0] - imx / 2.0) * pix_x

    if chfield == DOUBLED_PLUS_ONE:
        y_pixel = 2.0 * xy_arr[:, 1] + 1.0
    elif chfield == DOUBLED:
        y_pixel = 2.0 * xy_arr[:, 1]
    else:
        y_pixel = xy_arr[:, 1]

    result[:, 1] = (imy / 2.0 - y_pixel) * pix_y
    return result


@cython.ccall
def metric_to_pixel_batch(xy, cpar) -> object:
    """Convert N metric coordinates to pixel.

    Args:
        xy: (N, 2) array of metric coordinates.
        cpar: ControlPar with imx, imy, pix_x, pix_y, chfield.

    Returns:
        (N, 2) array of pixel coordinates.
    """
    xy_arr = np.asarray(xy, dtype=np.float64)

    imx: cython.double = cpar.imx
    imy: cython.double = cpar.imy
    pix_x: cython.double = cpar.pix_x
    pix_y: cython.double = cpar.pix_y
    chfield: cython.int = cpar.chfield

    result = np.empty_like(xy_arr)
    result[:, 0] = xy_arr[:, 0] / pix_x + imx / 2.0
    y_pixel = imy / 2.0 - xy_arr[:, 1] / pix_y

    if chfield == DOUBLED_PLUS_ONE:
        y_pixel = (y_pixel - 1.0) / 2.0
    elif chfield == DOUBLED:
        y_pixel = y_pixel / 2.0

    result[:, 1] = y_pixel
    return result


@cython.cfunc
@cython.inline
def _distort_brown_affin_core_out(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    sin_she: cython.double,
    cos_she: cython.double,
    out: cython.double[:],
):
    """Brown distortion with precomputed trig values — _out variant."""
    r: cython.double = c_sqrt(x * x + y * y)

    if r < 1e-10:
        out[0] = 0.0
        out[1] = 0.0
        return

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

    out[0] = scx * (x_dist - sin_she * y_dist)
    out[1] = scx * cos_she * y_dist


@cython.cfunc
@cython.inline
def _distort_brown_affin_core(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    sin_she: cython.double,
    cos_she: cython.double,
) -> tuple:
    """Brown distortion with precomputed trig values."""
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    _distort_brown_affin_core_out(
        x, y, k1, k2, k3, p1, p2, scx, sin_she, cos_she, _out_mv
    )
    return _out_mv[0], _out_mv[1]


@cython.cfunc
@cython.inline
def distort_brown_affin_out(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    out: cython.double[:],
):
    """Apply Brown distortion — _out variant."""
    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)
    _distort_brown_affin_core_out(x, y, k1, k2, k3, p1, p2, scx, sin_she, cos_she, out)


@cython.ccall
def distort_brown_affin(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
) -> tuple:
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
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    distort_brown_affin_out(x, y, k1, k2, k3, p1, p2, scx, she, _out_mv)
    return _out_mv[0], _out_mv[1]


@cython.cfunc
@cython.inline
def _correct_brown_affin_out(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    out: cython.double[:],
    _scratch: cython.double[:],
):
    """Inverse Brown distortion — _out variant (no tuple allocation in loop).

    Args:
        x, y: distorted metric coordinates.
        k1, k2, k3, p1, p2, scx, she: distortion parameters.
        out: [x_flat, y_flat] output array.
        _scratch: scratch [2] array for _distort_brown_affin_core_out.
    """
    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)
    inv_scx: cython.double = 1.0 / scx

    # Initial guess: inverse affine transformation
    xq: cython.double = x * inv_scx
    yq: cython.double = y * inv_scx / cos_she
    xq += yq * sin_she

    max_iter: cython.int = 20
    damping: cython.double = 0.7
    tol: cython.double = 1e-8

    xq_old: cython.double
    yq_old: cython.double
    dx: cython.double
    dy: cython.double
    change: cython.double
    pos_magnitude: cython.double
    _: cython.int

    for _ in range(max_iter):
        xq_old, yq_old = xq, yq

        # Forward distort current guess (no tuple creation)
        _distort_brown_affin_core_out(
            xq, yq, k1, k2, k3, p1, p2, scx, sin_she, cos_she, _scratch
        )

        # Error
        dx = (x - _scratch[0]) * inv_scx
        dy = (y - _scratch[1]) * inv_scx

        # Update with damping
        xq += dx * damping
        yq += dy * damping

        # Check convergence
        change = c_sqrt((xq - xq_old) ** 2 + (yq - yq_old) ** 2)
        pos_magnitude = c_sqrt(xq * xq + yq * yq)
        if pos_magnitude > 1e-10 and change / pos_magnitude < tol:
            break

    out[0] = xq
    out[1] = yq


@cython.ccall
def correct_brown_affin(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
) -> tuple:
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
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    _scratch = np.empty(2, dtype=np.float64)
    _scratch_mv: cython.double[:] = _scratch
    _correct_brown_affin_out(x, y, k1, k2, k3, p1, p2, scx, she, _out_mv, _scratch_mv)
    return _out_mv[0], _out_mv[1]


@cython.cfunc
@cython.inline
def _correct_brown_affine_exact_out(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    tol: cython.double,
    out: cython.double[:],
):
    """Iteratively solve inverse Brown distortion — _out variant.

    Args:
        x, y: distorted metric coordinates.
        k1, k2, k3, p1, p2, scx, she, tol: params.
        out: [x_flat, y_flat] output array.
    """
    r_init: cython.double = c_sqrt(x * x + y * y)

    if r_init < 1e-10:
        out[0] = 0.0
        out[1] = 0.0
        return

    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)
    inv_scx: cython.double = 1.0 / scx

    # Initial guess: inverse affine transformation
    xq: cython.double = (x + y * sin_she) * inv_scx
    yq: cython.double = y / cos_she

    max_iter: cython.int = 50
    damping: cython.double = 0.5

    r2: cython.double
    r4: cython.double
    r6: cython.double
    radial_factor: cython.double
    dx: cython.double
    dy: cython.double
    xq_new: cython.double
    yq_new: cython.double
    dx_change: cython.double
    dy_change: cython.double
    _: cython.int

    for _ in range(max_iter):
        r2 = xq * xq + yq * yq
        r4 = r2 * r2
        r6 = r4 * r2

        radial_factor = k1 * r2 + k2 * r4 + k3 * r6

        dx = xq * radial_factor + p1 * (r2 + 2.0 * xq * xq) + 2.0 * p2 * xq * yq
        dy = yq * radial_factor + p2 * (r2 + 2.0 * yq * yq) + 2.0 * p1 * xq * yq

        xq_new = (x + y * sin_she) * inv_scx - dx
        yq_new = y / cos_she - dy

        dx_change = xq_new - xq
        dy_change = yq_new - yq

        xq += damping * dx_change
        yq += damping * dy_change

        if c_sqrt(dx_change**2 + dy_change**2) < tol:
            break

    out[0] = xq
    out[1] = yq


@cython.cfunc
def correct_brown_affine_exact(
    x: cython.double,
    y: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    tol: cython.double = 1e-8,
) -> tuple:
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
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    _correct_brown_affine_exact_out(x, y, k1, k2, k3, p1, p2, scx, she, tol, _out_mv)
    return _out_mv[0], _out_mv[1]


@cython.ccall
def flat_to_dist_out(
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
    out: cython.double[:],
):
    """Convert flat-image to distorted metric coordinates — _out variant."""
    distort_brown_affin_out(flat_x + xh, flat_y + yh, k1, k2, k3, p1, p2, scx, she, out)


@cython.ccall
def flat_to_dist(
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
    """Convert flat-image to distorted metric coordinates.

    Args:
        flat_x, flat_y: flat-image (undistorted, centered) coordinates.
        xh, yh: principal point (sensor shift).
        k1, k2, k3, p1, p2, scx, she: distortion parameters.

    Returns:
        (dist_x, dist_y) distorted metric coordinates.
    """
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    flat_to_dist_out(flat_x, flat_y, xh, yh, k1, k2, k3, p1, p2, scx, she, _out_mv)
    return _out_mv[0], _out_mv[1]


@cython.ccall
def dist_to_flat_out(
    dist_x: cython.double,
    dist_y: cython.double,
    xh: cython.double,
    yh: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    tol: cython.double,
    out: cython.double[:],
):
    """Convert distorted metric to flat-image coordinates — _out variant."""
    _scratch = np.empty(2, dtype=np.float64)
    _scratch_mv: cython.double[:] = _scratch
    _correct_brown_affine_exact_out(
        dist_x, dist_y, k1, k2, k3, p1, p2, scx, she, tol, _scratch_mv
    )
    out[0] = _scratch_mv[0] - xh
    out[1] = _scratch_mv[1] - yh


@cython.ccall
def dist_to_flat(
    dist_x: cython.double,
    dist_y: cython.double,
    xh: cython.double,
    yh: cython.double,
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    tol: cython.double = 1e-8,
) -> tuple:
    """Convert distorted metric to flat-image coordinates.

    Args:
        dist_x, dist_y: distorted metric coordinates.
        xh, yh: principal point (sensor shift).
        k1, k2, k3, p1, p2, scx, she: distortion parameters.
        tol: convergence tolerance.

    Returns:
        (flat_x, flat_y) flat-image coordinates.
    """
    _out = np.empty(2, dtype=np.float64)
    _out_mv: cython.double[:] = _out
    dist_to_flat_out(dist_x, dist_y, xh, yh, k1, k2, k3, p1, p2, scx, she, tol, _out_mv)
    return _out_mv[0], _out_mv[1]


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled


@cython.ccall
@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def correct_brown_affine_batch(
    xy: cython.double[:, ::1],
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
    out: object = None,
) -> object:
    n: cython.Py_ssize_t = xy.shape[0]
    if out is None:
        result: np.ndarray = np.empty((n, 2), dtype=np.float64)
    else:
        result = np.asarray(out, dtype=np.float64)
    result_view: cython.double[:, :] = result

    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)
    inv_scx: cython.double = 1.0 / scx
    damping: cython.double = 0.7
    tol: cython.double = 1e-8
    max_iter: cython.int = 20

    i: cython.Py_ssize_t
    _: cython.int
    x: cython.double
    y: cython.double
    xq: cython.double
    yq: cython.double
    xq_old: cython.double
    yq_old: cython.double
    r: cython.double
    r2: cython.double
    r4: cython.double
    r6: cython.double
    radial_factor: cython.double
    x_dist: cython.double
    y_dist: cython.double
    xt: cython.double
    yt: cython.double
    dx: cython.double
    dy: cython.double
    change: cython.double
    pos_magnitude: cython.double

    for i in range(n):
        x = xy[i, 0]
        y = xy[i, 1]

        # Initial guess: inverse affine transformation
        xq = x * inv_scx
        yq = y * inv_scx / cos_she
        xq += yq * sin_she

        for _ in range(max_iter):
            xq_old = xq
            yq_old = yq

            # Inlined distort_brown_affin
            r = c_sqrt(xq * xq + yq * yq)
            if r < 1e-10:
                xt = 0.0
                yt = 0.0
            else:
                r2 = r * r
                r4 = r2 * r2
                r6 = r4 * r2
                radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
                x_dist = (
                    xq * radial_factor + p1 * (r2 + 2.0 * xq * xq) + 2.0 * p2 * xq * yq
                )
                y_dist = (
                    yq * radial_factor + p2 * (r2 + 2.0 * yq * yq) + 2.0 * p1 * xq * yq
                )
                xt = scx * (x_dist - sin_she * y_dist)
                yt = scx * cos_she * y_dist

            dx = (x - xt) * inv_scx
            dy = (y - yt) * inv_scx

            xq += dx * damping
            yq += dy * damping

            change = c_sqrt((xq - xq_old) ** 2 + (yq - yq_old) ** 2)
            pos_magnitude = c_sqrt(xq * xq + yq * yq)
            if pos_magnitude > 1e-10 and change / pos_magnitude < tol:
                break

        result_view[i, 0] = xq
        result_view[i, 1] = yq

    return result


@cython.ccall
@cython.profile(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def distort_brown_affine_batch(
    xy: cython.double[:, ::1],
    k1: cython.double,
    k2: cython.double,
    k3: cython.double,
    p1: cython.double,
    p2: cython.double,
    scx: cython.double,
    she: cython.double,
) -> object:
    n: cython.Py_ssize_t = xy.shape[0]
    result: np.ndarray = np.empty((n, 2), dtype=np.float64)
    result_view: cython.double[:, ::1] = result

    sin_she: cython.double = c_sin(she)
    cos_she: cython.double = c_cos(she)

    i: cython.Py_ssize_t
    x: cython.double
    y: cython.double
    r: cython.double
    r2: cython.double
    r4: cython.double
    r6: cython.double
    radial_factor: cython.double
    x_dist: cython.double
    y_dist: cython.double

    for i in range(n):
        x = xy[i, 0]
        y = xy[i, 1]
        r = c_sqrt(x * x + y * y)
        if r < 1e-10:
            result_view[i, 0] = 0.0
            result_view[i, 1] = 0.0
        else:
            r2 = r * r
            r4 = r2 * r2
            r6 = r4 * r2
            radial_factor = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            x_dist = x * radial_factor + p1 * (r2 + 2.0 * x * x) + 2.0 * p2 * x * y
            y_dist = y * radial_factor + p2 * (r2 + 2.0 * y * y) + 2.0 * p1 * x * y
            result_view[i, 0] = scx * (x_dist - sin_she * y_dist)
            result_view[i, 1] = scx * cos_she * y_dist

    return result
