"""
Image coordinate compatibility wrappers providing optv-like batch API.
"""

import numpy as np
from algorithms.imgcoord import img_coord_batch, flat_image_coord_batch


def image_coordinates(input_arr, cal, mm, output=None):
    """
    Project 3D positions to 2D image coordinates (batch operation).

    Args:
        input_arr: ndarray[n, 3] of 3D positions
        cal: Calibration instance
        mm: MultimediaParams instance or MmNp object
        output: Optional output array

    Returns:
        ndarray[n, 2] of image coordinates
    """
    # Unwrap multimedia params if needed
    if hasattr(mm, '_mm'):
        mm_obj = mm._mm
    else:
        mm_obj = mm

    result = img_coord_batch(input_arr, cal._cal, mm_obj)

    if output is not None:
        output[:] = result
        return output
    return result


def flat_image_coordinates(input_arr, cal, mm, output=None):
    """
    Project 3D positions to flat (undistorted) image coordinates (batch operation).

    Args:
        input_arr: ndarray[n, 3] of 3D positions
        cal: Calibration instance
        mm: MultimediaParams instance or MmNp object
        output: Optional output array

    Returns:
        ndarray[n, 2] of flat image coordinates
    """
    # Unwrap multimedia params if needed
    if hasattr(mm, '_mm'):
        mm_obj = mm._mm
    else:
        mm_obj = mm

    result = flat_image_coord_batch(input_arr, cal._cal, mm_obj)

    if output is not None:
        output[:] = result
        return output
    return result
