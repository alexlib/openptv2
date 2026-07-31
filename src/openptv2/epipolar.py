"""Streamlined epipolar geometry API."""

from openptv2.algorithms.epi import epipolar_curve as _epipolar_curve


def epipolar_curve(point, cal1, cal2, num_points, cpar, vpar):
    """
    Calculate epipolar curve between two cameras.

    Args:
        point: Image point tuple/array (x, y) in camera 1
        cal1: Calibration instance for origin camera
        cal2: Calibration instance for projection camera
        num_points: Number of points to generate on curve
        cpar: ControlPar instance
        vpar: VolumePar instance

    Returns:
        ndarray[num_points, 2]: Points on epipolar curve
    """
    raw_cal1 = cal1._cal if hasattr(cal1, '_cal') else cal1
    raw_cal2 = cal2._cal if hasattr(cal2, '_cal') else cal2
    raw_cpar = cpar._cpar if hasattr(cpar, '_cpar') else cpar
    raw_vpar = vpar._vpar if hasattr(vpar, '_vpar') else vpar

    return _epipolar_curve(
        point,
        raw_cal1,
        raw_cal2,
        num_points,
        raw_cpar,
        raw_vpar
    )

__all__ = ["epipolar_curve"]
