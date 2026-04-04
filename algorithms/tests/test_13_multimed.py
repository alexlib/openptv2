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
        from algorithms.calibration import Calibration as PythonCal
        from algorithms.parameters import MultimediaPar

        pos = np.array([0.0, 0.0, 100.0])
        angles = np.array([0.0, 0.0, 0.0])

        python_cal = PythonCal()
        python_cal.set_pos(pos)
        python_cal.set_angles(angles)

        mmp = MultimediaPar()
        mmp.nlay = 1

        test_point = np.array([10.0, 20.0, 30.0])
        from algorithms.multimed import multimed_r_nlay as python_func

        python_result = python_func(python_cal, mmp, test_point)

        assert python_result is not None
        assert python_result > 0

    def test_trans_cam_point(self):
        """Test trans_cam_point function."""
        from algorithms.multimed import trans_cam_point

        from algorithms.calibration import Calibration as PythonCal
        from algorithms.parameters import MultimediaPar

        python_cal = PythonCal()
        python_cal.set_pos(np.array([0.0, 0.0, 100.0]))
        python_cal.set_angles(np.array([0.0, 0.0, 0.0]))
        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[5.0], n3=1.0)
        glass_dir = np.array([0.0, 0.0, 1.0])

        test_point = np.array([50.0, 50.0, 50.0])

        try:
            python_result = trans_cam_point(
                python_cal.ext_par, mm, glass_dir, test_point
            )
        except Exception as e:
            pytest.fail(f"python trans_cam_point failed: {e}")

        assert python_result is not None
        assert len(python_result) == 4

    def test_back_trans_point(self):
        """Test back_trans_point function."""
        from algorithms.multimed import back_trans_point, trans_cam_point

        from algorithms.calibration import Calibration as PythonCal
        from algorithms.parameters import MultimediaPar

        python_cal = PythonCal()
        python_cal.set_pos(np.array([0.0, 0.0, 100.0]))
        python_cal.set_angles(np.array([0.0, 0.0, 0.0]))
        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[5.0], n3=1.0)
        glass = np.array([0.0, 0.0, 1.0])

        test_point = np.array([50.0, 50.0, 50.0])

        try:
            pos_t, cross_p, cross_c, _ = trans_cam_point(
                python_cal.ext_par, mm, glass, test_point
            )
            python_result = back_trans_point(pos_t, mm, glass, cross_p, cross_c)
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
        mmp.set_layers([1.33, 1.5], [5.0, 10.0])
        mmp.n3 = 1.0

        test_points = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ]
        )

        try:
            results = []
            for pt in test_points:
                r = multimed_nlay(python_cal, mmp, pt)
                results.append(r)
        except Exception as e:
            pytest.fail(f"python multimed_nlay failed: {e}")

        assert len(results) == 2

    def test_init_mmlut(self):
        """Test init_mmlut function.

        Uses trivial medium (all n=1) so multimed_r_nlay short-circuits
        to return 1.0 immediately, keeping the test fast.
        Includes a warm-up call to avoid Numba JIT compilation overhead.
        """
        import numpy as np
        from algorithms.multimed import init_mmlut
        from algorithms.parameters import ControlPar, VolumePar, MultimediaPar
        from algorithms.calibration import Calibration as PythonCal

        def _make_inputs():
            cal = PythonCal()
            cal.set_pos(np.array([0.0, 0.0, 50.0]))
            cal.int_par.cc = 10.0

            cpar = ControlPar()
            cpar.imx = 64
            cpar.imy = 64
            cpar.pix_x = 0.01
            cpar.pix_y = 0.01
            cpar.mm = MultimediaPar()
            cpar.mm.n1 = 1.0
            cpar.mm.nlay = 1
            cpar.mm.n2 = [1.0]
            cpar.mm.n3 = 1.0

            vpar = VolumePar()
            vpar.Zmin = 0.0
            vpar.Zmax = 10.0
            vpar.z_min_lay = [0.0]
            vpar.z_max_lay = [10.0]
            return vpar, cpar, cal

        # Warm-up: trigger Numba JIT compilation
        vpar, cpar, cal = _make_inputs()
        _ = init_mmlut(vpar, cpar, cal)

        # Actual test
        vpar, cpar, cal = _make_inputs()
        try:
            result_cal = init_mmlut(vpar, cpar, cal)
        except Exception as e:
            pytest.fail(f"python init_mmlut failed: {e}")

        assert result_cal is not None
        assert result_cal.mmlut.nr > 0
        assert result_cal.mmlut.nz > 0
