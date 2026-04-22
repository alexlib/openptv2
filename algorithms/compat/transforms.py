"""
Transform compatibility wrappers providing optv-like batch API.
"""

import numpy as np
from algorithms.trafo import (
    pixel_to_metric_batch,
    metric_to_pixel_batch,
    correct_brown_affin,
    distort_brown_affin,
    correct_brown_affine_exact,
)


def convert_arr_pixel_to_metric(input_arr, cpar, out=None):
    """
    Convert pixel coordinates to metric coordinates (batch operation).

    Args:
        input_arr: ndarray[n, 2] of pixel coordinates
        cpar: ControlParams instance
        out: Optional output array

    Returns:
        ndarray[n, 2] of metric coordinates
    """
    result = pixel_to_metric_batch(input_arr, cpar._cpar)
    if out is not None:
        out[:] = result
        return out
    return result


def convert_arr_metric_to_pixel(input_arr, cpar, out=None):
    """
    Convert metric coordinates to pixel coordinates (batch operation).

    Args:
        input_arr: ndarray[n, 2] of metric coordinates
        cpar: ControlParams instance
        out: Optional output array

    Returns:
        ndarray[n, 2] of pixel coordinates
    """
    result = metric_to_pixel_batch(input_arr, cpar._cpar)
    if out is not None:
        out[:] = result
        return out
    return result


def correct_arr_brown_affine(input_arr, cal, out=None):
    """
    Apply Brown-Affine distortion correction (batch operation).

    Args:
        input_arr: ndarray[n, 2] of distorted coordinates
        cal: Calibration instance
        out: Optional output array

    Returns:
        ndarray[n, 2] of corrected coordinates
    """
    result = np.empty_like(input_arr)
    k1 = cal._cal.added_par.k1
    k2 = cal._cal.added_par.k2
    k3 = cal._cal.added_par.k3
    p1 = cal._cal.added_par.p1
    p2 = cal._cal.added_par.p2
    scx = cal._cal.added_par.scx
    she = cal._cal.added_par.she

    for i in range(len(input_arr)):
        x, y = input_arr[i]
        result[i] = correct_brown_affin(x, y, k1, k2, k3, p1, p2, scx, she)

    if out is not None:
        out[:] = result
        return out
    return result


def distort_arr_brown_affine(input_arr, cal, out=None):
    """
    Apply Brown-Affine distortion (batch operation).

    Args:
        input_arr: ndarray[n, 2] of corrected coordinates
        cal: Calibration instance
        out: Optional output array

    Returns:
        ndarray[n, 2] of distorted coordinates
    """
    result = np.empty_like(input_arr)
    k1 = cal._cal.added_par.k1
    k2 = cal._cal.added_par.k2
    k3 = cal._cal.added_par.k3
    p1 = cal._cal.added_par.p1
    p2 = cal._cal.added_par.p2
    scx = cal._cal.added_par.scx
    she = cal._cal.added_par.she

    for i in range(len(input_arr)):
        x, y = input_arr[i]
        result[i] = distort_brown_affin(x, y, k1, k2, k3, p1, p2, scx, she)

    if out is not None:
        out[:] = result
        return out
    return result


def distorted_to_flat(input_arr, cal, out=None, tol=0.00001):
    """
    Remove distortion with iterative solver (batch operation).

    Args:
        input_arr: ndarray[n, 2] of distorted coordinates
        cal: Calibration instance
        out: Optional output array
        tol: Convergence tolerance

    Returns:
        ndarray[n, 2] of flat (corrected) coordinates
    """
    result = np.empty_like(input_arr)
    k1 = cal._cal.added_par.k1
    k2 = cal._cal.added_par.k2
    k3 = cal._cal.added_par.k3
    p1 = cal._cal.added_par.p1
    p2 = cal._cal.added_par.p2
    scx = cal._cal.added_par.scx
    she = cal._cal.added_par.she

    for i in range(len(input_arr)):
        x, y = input_arr[i]
        result[i] = correct_brown_affine_exact(x, y, k1, k2, k3, p1, p2, scx, she, tol)

    if out is not None:
        out[:] = result
        return out
    return result
