"""
Engine comparison tests for epipolar module.

Tests epipolar_curve function. Each engine reads ALL parameters from the
same files through its own reader, ensuring both reader parity and
algorithm parity are tested simultaneously.

Tolerance: 1e-7 (complex geometry calculations)
"""

import numpy as np
import pytest
from pathlib import Path
from ..conftest import get_tolerance, FIXTURES

TOLERANCE = get_tolerance("epipolar")

optv = pytest.importorskip("optv")

from optv.epipolar import epipolar_curve as optv_func
from optv.calibration import Calibration as OptvCal
from optv.parameters import (
    ControlParams as OptvCParam,
    VolumeParams as OptvVParam,
)

from algorithms.epi import epipolar_curve as python_func
from algorithms.calibration import Calibration as PythonCal
from algorithms.parameters import read_control_par, read_volume_par
from algorithms.parameters_adapter import VolumeParams as PythonVParam


class TestEpipolar:
    """Compare epipolar functions between optv and python engines."""

    def test_epipolar_curve_from_files(self):
        """Test epipolar_curve with parameters from files.

        Each engine reads the SAME files through its own reader:
        - control.par -> ControlParams (optv) / ControlPar (python)
        - criteria.par -> VolumeParams (optv) / VolumePar (python)
        - sym_cam*.tif.ori + cam1.tif.addpar -> Calibration (both)
        """
        ori_tmpl = str(FIXTURES / "calibration/sym_cam{cam_num}.tif.ori")
        add_file = str(FIXTURES / "calibration/cam1.tif.addpar")
        control_par_file = str(FIXTURES / "corresp/control.par")
        volume_par_file = str(FIXTURES / "corresp/criteria.par")

        # ---- optv engine reads from files ----
        optv_cpar = OptvCParam(4)
        optv_cpar.read_control_par(control_par_file)
        sens_size = optv_cpar.get_image_size()

        optv_vpar = OptvVParam()
        optv_vpar.read_volume_par(volume_par_file)
        optv_vpar.set_Zmin_lay([-10, -10])
        optv_vpar.set_Zmax_lay([10, 10])

        mult_params = optv_cpar.get_multimedia_params()
        mult_params.set_n1(1.0)
        mult_params.set_layers(np.array([1.0]), np.array([1.0]))
        mult_params.set_n3(1.0)

        optv_orig_cal = OptvCal()
        optv_orig_cal.from_file(ori_tmpl.format(cam_num=1), add_file)
        optv_orig_cal.set_angles(np.r_[0.0, -np.pi / 4.0, 0.0])
        optv_proj_cal = OptvCal()
        optv_proj_cal.from_file(ori_tmpl.format(cam_num=3), add_file)
        optv_proj_cal.set_angles(np.r_[0.0, 3 * np.pi / 4.0, 0.0])

        # ---- python engine reads from the SAME files ----
        python_cpar = read_control_par(Path(control_par_file))
        python_cpar.mm.set_n1(1.0)
        python_cpar.mm.set_layers([1.0], [0.0])
        python_cpar.mm.set_n3(1.0)

        python_vpar = read_volume_par(Path(volume_par_file))
        python_vpar.z_min_lay = [-10.0, -10.0]
        python_vpar.z_max_lay = [10.0, 10.0]

        python_orig_cal = PythonCal()
        python_orig_cal.from_file(ori_tmpl.format(cam_num=1), add_file)
        python_orig_cal.set_angles(np.array([0.0, -np.pi / 4.0, 0.0]))
        python_proj_cal = PythonCal()
        python_proj_cal.from_file(ori_tmpl.format(cam_num=3), add_file)
        python_proj_cal.set_angles(np.array([0.0, 3 * np.pi / 4.0, 0.0]))

        python_vpar_adapter = PythonVParam(
            xmin=python_vpar.x_lay[0],
            xmax=python_vpar.x_lay[1],
            ymin=-100,
            ymax=100,
            zmin=python_vpar.z_min_lay[0],
            zmax=python_vpar.z_max_lay[0],
        )
        python_vpar_adapter.x_lay = list(python_vpar.x_lay)
        python_vpar_adapter.z_min_lay = list(python_vpar.z_min_lay)
        python_vpar_adapter.z_max_lay = list(python_vpar.z_max_lay)

        mid = np.array(sens_size) / 2.0
        optv_result = optv_func(
            mid, optv_orig_cal, optv_proj_cal, 5, optv_cpar, optv_vpar
        )
        python_result = python_func(
            mid, python_orig_cal, python_proj_cal, 5, python_cpar, python_vpar_adapter
        )

        np.testing.assert_allclose(
            optv_result, python_result, rtol=1e-7, atol=1e-7
        )
