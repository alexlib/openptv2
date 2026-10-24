"""Line 2 of an .ori file holds omega/phi/kappa (radians), not a reference point.

The rotation matrix on lines 4-6 is the matrix of exactly those angles, so the
reader must return the file's angles and a matrix equal to the file's one.
"""

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from tests._support import find_test_data_root

CAL_DIR = find_test_data_root() / "test_cavity" / "cal"


@pytest.mark.parametrize("cam", [1, 2, 3, 4])
def test_ori_line2_is_angles(cam):
    ori = CAL_DIR / f"cam{cam}.tif.ori"
    values = [float(v) for v in ori.read_text().split()]
    angles, matrix = values[3:6], np.array(values[6:15]).reshape(3, 3)

    cal = Calibration.from_file(str(ori), str(ori.with_suffix(".addpar")))

    np.testing.assert_allclose(cal.get_angles(), angles)
    # The file stores the matrix rounded to 7 decimals.
    np.testing.assert_allclose(cal.get_rotation_matrix(), matrix, atol=1e-6)
