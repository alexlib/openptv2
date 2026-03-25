"""
Engine comparison tests for multimed module.

Tests multimedia refraction functions.
Tolerance: 1e-7 (refraction calculations)
"""

import numpy as np
import pytest
from .conftest import get_tolerance

TOLERANCE = get_tolerance("multimed")


class TestMultimed:
    """Compare multimed functions between optv and python engines."""

    def test_multimed_r_nlay_basic(self):
        """Test multimed_r_nlay with basic data."""
        from optv.multimed import multimed_r_nlay as optv_func
        from optv.calibration import Calibration
        from optv.parameters import ControlParams

        from algorithms.calibration import Calibration as PythonCal
        from algorithms.parameters import MultimediaPar

        pos = np.array([0.0, 0.0, 100.0])
        angles = np.array([0.0, 0.0, 0.0])

        optv_cal = Calibration(pos=pos, angs=angles)
        python_cal = PythonCal()
        python_cal.set_pos(pos)
        python_cal.set_angles(angles)

        mmp = MultimediaPar()
        mmp.nlay = 1

        test_point = np.array([10.0, 20.0, 30.0])

        try:
            optv_result = optv_func(optv_cal, mmp, test_point)
        except Exception as e:
            pytest.fail(f"optv multimed_r_nlay failed: {e}")

        try:
            from algorithms.multimed import multimed_r_nlay as python_func

            python_result = python_func(python_cal, mmp, test_point)

            assert abs(optv_result - python_result) < TOLERANCE
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_trans_cam_point(self):
        """Test trans_cam_point function."""
        from algorithms.multimed import trans_cam_point

        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()
        python_cal.set_pos(np.array([0.0, 0.0, 100.0]))
        python_cal.set_angles(np.array([0.0, 0.0, 0.0]))

        test_point = np.array([50.0, 50.0, 50.0])

        try:
            python_result = trans_cam_point(test_point, python_cal)
        except Exception as e:
            pytest.fail(f"python trans_cam_point failed: {e}")

        assert python_result is not None
        assert len(python_result) == 3

    def test_back_trans_point(self):
        """Test back_trans_point function."""
        from algorithms.multimed import back_trans_point

        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()
        python_cal.set_pos(np.array([0.0, 0.0, 100.0]))
        python_cal.set_angles(np.array([0.0, 0.0, 0.0]))

        test_point = np.array([50.0, 50.0, 50.0])

        try:
            python_result = back_trans_point(test_point, python_cal)
        except Exception as e:
            pytest.fail(f"python back_trans_point failed: {e}")

        assert python_result is not None

    def test_move_along_ray(self):
        """Test move_along_ray function."""
        from algorithms.multimed import move_along_ray

        vertex = np.array([0.0, 0.0, 0.0])
        direct = np.array([0.0, 0.0, 1.0])
        glob_z = 50.0

        try:
            python_result = move_along_ray(glob_z, vertex, direct)
        except Exception as e:
            pytest.fail(f"python move_along_ray failed: {e}")

        assert python_result is not None
        np.testing.assert_allclose(
            python_result, np.array([0.0, 0.0, 50.0]), rtol=1e-10
        )

    def test_multimed_nlay_with_multiple_layers(self):
        """Test multimed functions with multiple layers."""
        from algorithms.multimed import multimed_nlay
        from algorithms.parameters import MultimediaPar
        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()
        python_cal.set_pos(np.array([0.0, 0.0, 100.0]))

        mmp = MultimediaPar()
        mmp.nlay = 2

        test_points = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ]
        )

        try:
            results = []
            for pt in test_points:
                r = multimed_nlay(mmp, pt)
                results.append(r)
        except Exception as e:
            pytest.fail(f"python multimed_nlay failed: {e}")

        assert len(results) == 2

    def test_init_mmlut(self):
        """Test init_mmlut function."""
        from algorithms.multimed import init_mmlut
        from algorithms.parameters import ControlPar, VolumePar
        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()

        cpar = ControlPar()
        cpar.imx = 1024
        cpar.imy = 1024

        vpar = VolumePar()
        vpar.Zmin = 0.0
        vpar.Zmax = 100.0

        try:
            result_cal = init_mmlut(vpar, cpar, python_cal)
        except Exception as e:
            pytest.fail(f"python init_mmlut failed: {e}")

        assert result_cal is not None
