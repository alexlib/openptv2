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
from .conftest import get_tolerance, FIXTURES

TOLERANCE = get_tolerance("epipolar")


class TestEpipolar:
    """Compare epipolar functions between optv and python engines."""

    def test_epipolar_curve_from_files(self):
        """Test epipolar_curve with parameters from files (like original optv test).

        Runs in isolated subprocess to avoid pytest/C extension memory issues.
        """
        test_code = f"""
import sys
import numpy as np
from pathlib import Path

FIXTURES = Path({str(FIXTURES)!r})

from optv.epipolar import epipolar_curve as optv_func
from optv.calibration import Calibration as OptvCal
from optv.parameters import ControlParams as OptvCParam, VolumeParams as OptvVParam

from algorithms.epi import epipolar_curve as python_func
from algorithms.calibration import Calibration as PythonCal
from algorithms.parameters import ControlPar as PythonCParam
from algorithms.parameters_adapter import VolumeParams as PythonVParam
from algorithms.parameters import MultimediaPar

ori_tmpl = str(FIXTURES / "calibration/sym_cam{{cam_num}}.tif.ori")
add_file = str(FIXTURES / "calibration/cam1.tif.addpar")

# Load shared parameters from files once, then use for both engines
optv_cpar = OptvCParam(4)
optv_cpar.read_control_par(str(FIXTURES / "corresp/control.par"))
sens_size = optv_cpar.get_image_size()
imx, imy = sens_size
pix_x, pix_y = optv_cpar.get_pixel_size()
chfield = optv_cpar.get_chfield()

optv_vpar = OptvVParam()
optv_vpar.read_volume_par(str(FIXTURES / "corresp/criteria.par"))
optv_vpar.set_Zmin_lay([-10, -10])
optv_vpar.set_Zmax_lay([10, 10])

mult_params = optv_cpar.get_multimedia_params()
mult_params.set_n1(1.0)
mult_params.set_layers(np.array([1.0]), np.array([1.0]))
mult_params.set_n3(1.0)

# Calibrations (same files for both engines)
optv_orig_cal = OptvCal()
optv_orig_cal.from_file(ori_tmpl.format(cam_num=1), add_file)
optv_orig_cal.set_angles(np.r_[0.0, -np.pi / 4.0, 0.0])
optv_proj_cal = OptvCal()
optv_proj_cal.from_file(ori_tmpl.format(cam_num=3), add_file)
optv_proj_cal.set_angles(np.r_[0.0, 3 * np.pi / 4.0, 0.0])

python_orig_cal = PythonCal()
python_orig_cal.from_file(ori_tmpl.format(cam_num=1), add_file)
python_orig_cal.set_angles(np.array([0.0, -np.pi / 4.0, 0.0]))
python_proj_cal = PythonCal()
python_proj_cal.from_file(ori_tmpl.format(cam_num=3), add_file)
python_proj_cal.set_angles(np.array([0.0, 3 * np.pi / 4.0, 0.0]))

# Python ControlPar with SAME values as optv file
python_cpar = PythonCParam()
python_cpar.num_cams = 4
python_cpar.imx = imx
python_cpar.imy = imy
python_cpar.pix_x = pix_x
python_cpar.pix_y = pix_y
python_cpar.chfield = chfield
python_cpar.mm = MultimediaPar(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)

# Python VolumePar with SAME values as optv file
x_lay = list(optv_vpar.get_X_lay())
z_min_lay = list(optv_vpar.get_Zmin_lay())
z_max_lay = list(optv_vpar.get_Zmax_lay())
python_vpar = PythonVParam(
    xmin=x_lay[0], xmax=x_lay[1],
    ymin=-100, ymax=100,
    zmin=z_min_lay[0], zmax=z_max_lay[0],
)
python_vpar.x_lay = x_lay
python_vpar.z_min_lay = z_min_lay
python_vpar.z_max_lay = z_max_lay

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
