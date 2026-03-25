"""
Engine comparison tests for epipolar module - using subprocess isolation.

Tests epipolar_curve function.
Tolerance: 1e-7 (complex geometry calculations)
"""

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path
from .conftest import get_tolerance

TOLERANCE = get_tolerance("epipolar")

FIXTURES = Path(__file__).parent.parent.parent / "testing_fodder"


class TestEpipolar:
    """Compare epipolar functions between optv and python engines."""

    def test_epipolar_curve_from_files(self):
        """Test epipolar_curve with parameters from files (like original optv test).

        Runs in isolated subprocess to avoid pytest/C extension memory issues.
        """
        test_code = """
import sys
import numpy as np
from pathlib import Path

FIXTURES = Path("tests/testing_fodder")

from optv.epipolar import epipolar_curve as optv_func
from optv.calibration import Calibration as OptvCal
from optv.parameters import ControlParams as OptvCParam, VolumeParams as OptvVParam

from algorithms.epi import epipolar_curve as python_func
from algorithms.calibration import Calibration as PythonCal
from algorithms.parameters import ControlPar as PythonCParam
from algorithms.parameters_adapter import VolumeParams as PythonVParam
from algorithms.parameters import MultimediaPar

ori_tmpl = str(FIXTURES / "calibration/sym_cam{cam_num}.tif.ori")
add_file = str(FIXTURES / "calibration/cam1.tif.addpar")

optv_orig_cal = OptvCal()
optv_orig_cal.from_file(ori_tmpl.format(cam_num=1), add_file)
optv_proj_cal = OptvCal()
optv_proj_cal.from_file(ori_tmpl.format(cam_num=3), add_file)

optv_orig_cal.set_angles(np.r_[0.0, -np.pi / 4.0, 0.0])
optv_proj_cal.set_angles(np.r_[0.0, 3 * np.pi / 4.0, 0.0])

optv_cpar = OptvCParam(4)
optv_cpar.read_control_par(str(FIXTURES / "corresp/control.par"))
sens_size = optv_cpar.get_image_size()

optv_vpar = OptvVParam()
optv_vpar.read_volume_par(str(FIXTURES / "corresp/criteria.par"))
optv_vpar.set_Zmin_lay([-10, -10])
optv_vpar.set_Zmax_lay([10, 10])

mult_params = optv_cpar.get_multimedia_params()
mult_params.set_n1(1.0)
mult_params.set_layers(np.array([1.0]), np.array([1.0]))
mult_params.set_n3(1.0)

python_orig_cal = PythonCal()
python_orig_cal.from_file(ori_tmpl.format(cam_num=1), add_file)
python_orig_cal.set_angles(np.array([0.0, -np.pi / 4.0, 0.0]))

python_proj_cal = PythonCal()
python_proj_cal.from_file(ori_tmpl.format(cam_num=3), add_file)
python_proj_cal.set_angles(np.array([0.0, 3 * np.pi / 4.0, 0.0]))

python_cpar = PythonCParam()
python_cpar.imx = 1280
python_cpar.imy = 1024
python_cpar.pix_x = 0.017
python_cpar.pix_y = 0.017
python_cpar.chfield = 0

python_vpar = PythonVParam(xmin=-250, xmax=100, ymin=-100, ymax=100, zmin=-10, zmax=10)
python_vpar.z_min_lay = [-10, -10]
python_vpar.z_max_lay = [10, 10]
python_vpar.x_lay = [-250, 100]

python_mmp = MultimediaPar()
python_mmp.n1 = 1.0
python_mmp.n2 = [1.0]
python_mmp.n3 = 1.0
python_mmp.d = [0.0]
python_cpar.mm = python_mmp

mid = np.array(sens_size) / 2.0
optv_result = optv_func(mid, optv_orig_cal, optv_proj_cal, 5, optv_cpar, optv_vpar)
python_result = python_func(mid, python_orig_cal, python_proj_cal, 5, python_cpar, python_vpar)

np.testing.assert_allclose(optv_result, python_result, rtol=1e-7, atol=1e-7)
print("TEST_PASSED")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),
        )

        if result.returncode != 0:
            pytest.fail(
                f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        if "TEST_PASSED" not in result.stdout:
            pytest.fail(
                f"Test did not pass:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
