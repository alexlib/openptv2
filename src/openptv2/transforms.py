"""Streamlined batch transformations and distortion corrections."""

import numpy as np
from openptv2.algorithms.trafo import (
    pixel_to_metric_batch,
    metric_to_pixel_batch,
    correct_brown_affine_batch,
    distort_brown_affine_batch,
)

def _unwrap_cpar(cpar):
    if hasattr(cpar, '_cpar'):
        return cpar._cpar
    if hasattr(cpar, 'control_par'):
        return cpar.control_par
    if hasattr(cpar, 'get_pixel_size') and hasattr(cpar, 'get_image_size'):
        from openptv2.algorithms.parameters import ControlPar, MmNp
        px, py = cpar.get_pixel_size()
        imx, imy = cpar.get_image_size()
        return ControlPar(
            num_cams=cpar.get_num_cams(),
            imx=imx, imy=imy,
            pix_x=px, pix_y=py,
            hp_flag=cpar.get_hp_flag(),
            all_cam_flag=cpar.get_allCam_flag(),
            tiff_flag=cpar.get_tiff_flag(),
            chfield=cpar.get_chfield(),
            mm=MmNp(n1=1.0)
        )
    return cpar


def convert_arr_pixel_to_metric(input_arr, cpar, out=None):
    """
    Convert pixel coordinates to metric coordinates (batch operation).

    Args:
        input_arr: ndarray[n, 2] of pixel coordinates
        cpar: ControlPar instance
        out: Optional output array

    Returns:
        ndarray[n, 2] of metric coordinates
    """
    raw_cpar = _unwrap_cpar(cpar)
    result = pixel_to_metric_batch(input_arr, raw_cpar)
    if out is not None:
        out[:] = result
        return out
    return result


def convert_arr_metric_to_pixel(input_arr, cpar, out=None):
    """
    Convert metric coordinates to pixel coordinates (batch operation).

    Args:
        input_arr: ndarray[n, 2] of metric coordinates
        cpar: ControlPar instance
        out: Optional output array

    Returns:
        ndarray[n, 2] of pixel coordinates
    """
    raw_cpar = _unwrap_cpar(cpar)
    result = metric_to_pixel_batch(input_arr, raw_cpar)
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
    raw_cal = cal._cal if hasattr(cal, '_cal') else cal
    input_arr = np.ascontiguousarray(input_arr, dtype=np.float64)
    result = correct_brown_affine_batch(
        input_arr,
        raw_cal.added_par.k1,
        raw_cal.added_par.k2,
        raw_cal.added_par.k3,
        raw_cal.added_par.p1,
        raw_cal.added_par.p2,
        raw_cal.added_par.scx,
        raw_cal.added_par.she,
    )
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
    raw_cal = cal._cal if hasattr(cal, '_cal') else cal
    input_arr = np.ascontiguousarray(input_arr, dtype=np.float64)
    result = distort_brown_affine_batch(
        input_arr,
        raw_cal.added_par.k1,
        raw_cal.added_par.k2,
        raw_cal.added_par.k3,
        raw_cal.added_par.p1,
        raw_cal.added_par.p2,
        raw_cal.added_par.scx,
        raw_cal.added_par.she,
    )
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
    from openptv2.algorithms.trafo import dist_to_flat

    raw_cal = cal._cal if hasattr(cal, '_cal') else cal
    ap = raw_cal.added_par
    ip = raw_cal.int_par

    input_arr = np.asarray(input_arr, dtype=np.float64)
    result = np.empty_like(input_arr)

    for i in range(len(input_arr)):
        # dist_to_flat removes the full Brown-affine distortion model
        # (k1,k2,k3,p1,p2,scx,she) AND subtracts the principal point (xh, yh).
        # The principal-point subtraction is essential for off-center
        # calibrations (e.g. image-splitter cameras); omitting it silently
        # shifts every coordinate by (xh, yh) and breaks correspondence.
        result[i, 0], result[i, 1] = dist_to_flat(
            input_arr[i, 0], input_arr[i, 1],
            ip.xh, ip.yh,
            ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she,
            tol,
        )

    if out is not None:
        out[:] = result
        return out
    return result


__all__ = [
    "convert_arr_metric_to_pixel",
    "convert_arr_pixel_to_metric",
    "correct_arr_brown_affine",
    "distort_arr_brown_affine",
    "distorted_to_flat",
]
