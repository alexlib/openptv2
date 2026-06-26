"""
Epipolar geometry compatibility wrapper providing optv-like API.
"""

from openptv2.algorithms.epi import epipolar_curve as _epipolar_curve


def epipolar_curve(point, cal1, cal2, num_points, cpar, vpar):
    """
    Calculate epipolar curve between two cameras.

    Args:
        point: Image point tuple/array (x, y) in camera 1
        cal1: Calibration instance for origin camera (compat wrapper)
        cal2: Calibration instance for projection camera (compat wrapper)
        num_points: Number of points to generate on curve
        cpar: ControlParams instance (compat wrapper)
        vpar: VolumeParams instance (compat wrapper)

    Returns:
        ndarray[num_points, 2]: Points on epipolar curve
    """
    return _epipolar_curve(
        point,
        cal1._cal,
        cal2._cal,
        num_points,
        cpar._cpar,
        vpar._vpar
    )
