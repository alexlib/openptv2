"""Streamlined image coordinates projection API."""

from openptv2.algorithms.imgcoord import flat_image_coord_batch, img_coord_batch


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
    raw_cal = cal
    raw_mm = mm

    result = img_coord_batch(input_arr, raw_cal, raw_mm)

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
    raw_cal = cal
    raw_mm = mm

    result = flat_image_coord_batch(input_arr, raw_cal, raw_mm)

    if output is not None:
        output[:] = result
        return output
    return result


__all__ = ["flat_image_coordinates", "image_coordinates"]
