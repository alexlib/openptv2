"""
Engine comparison tests for multimed module.

Tests multimedia refraction functions.
Tolerance: 1e-7 (refraction calculations)
"""

import numpy as np
import pytest
from ..conftest import get_tolerance

TOLERANCE = get_tolerance("multimed")


class TestMultimed:
    """Compare multimed functions between optv and python engines."""

    def test_fast_get_mmf_from_mmlut(self):
        """Test the compiled MMLUT lookup helper directly."""
        from algorithms.multimed import get_mmf_from_mmlut, fast_get_mmf_from_mmlut
        from algorithms.calibration import Calibration as PythonCal

        cal = PythonCal()
        cal.mmlut.rw = 10
        cal.mmlut.origin = np.array([0.0, 0.0, 0.0])
        cal.mmlut.nr = 2
        cal.mmlut.nz = 2
        cal.mmlut_data = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float64,
        )

        pos = np.array([5.0, 0.0, 5.0], dtype=np.float64)

        result = get_mmf_from_mmlut(cal, pos)
        compiled = fast_get_mmf_from_mmlut(
            cal.mmlut.rw,
            cal.mmlut.origin,
            cal.mmlut_data.flatten(),
            cal.mmlut.nz,
            cal.mmlut.nr,
            pos,
        )

        assert fast_get_mmf_from_mmlut.nopython_signatures is not None
        assert len(fast_get_mmf_from_mmlut.nopython_signatures) > 0
        np.testing.assert_allclose(result, compiled, rtol=1e-12, atol=1e-12)

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

        # Default MultimediaPar has n1=n2=n3=1 → trivial single-medium, shift = 1.0
        assert abs(python_result - 1.0) < 1e-10

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

        pos_t, cross_p, cross_c, z0 = trans_cam_point(
            python_cal.ext_par, mm, glass_dir, test_point
        )

        # glass_par=[0,0,1] (magnitude 1, pointing in z), cam at z=100, d=5:
        # dist_cam_glas = dot([0,0,100],[0,0,1])/1 - 1 - 5 = 94; z0 = 94+5 = 99
        assert abs(z0 - 99.0) < 1e-10
        assert cross_c.shape == (3,)
        assert cross_p.shape == (3,)
        assert pos_t.shape == (3,)

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

        pos_t, cross_p, cross_c, _ = trans_cam_point(
            python_cal.ext_par, mm, glass, test_point
        )
        recovered = back_trans_point(pos_t, mm, glass, cross_p, cross_c)

        # back_trans_point is the inverse of trans_cam_point
        np.testing.assert_allclose(recovered, test_point, atol=1e-6)

    def test_move_along_ray(self):
        """Test move_along_ray function."""
        from algorithms.multimed import move_along_ray

        vertex = np.array([0.0, 0.0, 0.0])
        direct = np.array([0.0, 0.0, 1.0])
        glob_z = 50.0

        python_result = move_along_ray(glob_z, vertex, direct)

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

        results = []
        for pt in test_points:
            Xq, Yq = multimed_nlay(python_cal, mmp, pt)
            results.append((Xq, Yq))

        assert len(results) == 2
        # Point (0,0,z) is on the camera axis → Xq=0, Yq=0 regardless of shift
        Xq_axis, Yq_axis = multimed_nlay(
            python_cal, mmp, np.array([0.0, 0.0, 30.0])
        )
        assert abs(Xq_axis) < 1e-10
        assert abs(Yq_axis) < 1e-10

    def test_epi_mm_batch_parity(self):
        """Test the batched epipolar helper matches repeated scalar calls."""
        from algorithms.calibration import Calibration as PythonCal
        from algorithms.epi import (
            epi_mm,
            epi_mm_batch,
            epi_mm_batch_inputs,
            _epi_mm_batch_restore_inputs,
            _epi_mm_batch_row,
        )
        from algorithms.parameters import MultimediaPar, VolumePar

        cal1 = PythonCal()
        cal1.set_pos(np.array([0.0, 0.0, 100.0]))
        cal1.set_angles(np.array([0.0, 0.0, 0.0]))

        cal2 = PythonCal()
        cal2.set_pos(np.array([25.0, 0.0, 100.0]))
        cal2.set_angles(np.array([0.0, 0.0, 0.0]))

        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[5.0], n3=1.0)
        vpar = VolumePar(
            x_lay=[0.0, 100.0],
            z_min_lay=[0.0, 0.0],
            z_max_lay=[50.0, 50.0],
        )

        xl = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        yl = np.array([5.0, 15.0, 25.0], dtype=np.float64)

        batch = epi_mm_batch(xl, yl, cal1, cal2, mm, vpar)
        scalar = np.array([epi_mm(xl[i], yl[i], cal1, cal2, mm, vpar) for i in range(3)])

        np.testing.assert_allclose(batch, scalar, rtol=1e-12, atol=1e-12)

        row = _epi_mm_batch_row(float(xl[0]), float(yl[0]), cal1, cal2, mm, vpar)
        np.testing.assert_allclose(row, scalar[0], rtol=1e-12, atol=1e-12)

        inputs = epi_mm_batch_inputs(cal1, cal2, mm, vpar)
        assert inputs.cal1_pos.shape == (3,)
        assert inputs.cal1_dm.shape == (3, 3)
        assert inputs.cal1_glass.shape == (3,)
        assert inputs.cal1_cc == cal1.int_par.cc
        assert inputs.cal2_pos.shape == (3,)
        assert inputs.cal2_dm.shape == (3, 3)
        assert inputs.cal2_glass.shape == (3,)
        assert inputs.cal2_cc == cal2.int_par.cc
        assert inputs.mm_n1 == mm.n1
        assert inputs.mm_d.shape == (1,)
        assert inputs.mm_n2.shape == (1,)
        assert inputs.mm_n3 == mm.n3
        assert inputs.mmlut_origin.shape == (3,)
        assert inputs.mmlut_data.ndim == 1

        restored_cal1, restored_cal2, restored_mm = _epi_mm_batch_restore_inputs(inputs)
        np.testing.assert_allclose(restored_cal1.get_pos(), cal1.get_pos(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(restored_cal2.get_pos(), cal2.get_pos(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(restored_mm.n2, mm.n2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(restored_mm.d, mm.d, rtol=1e-12, atol=1e-12)
        assert restored_mm.n1 == mm.n1
        assert restored_mm.n3 == mm.n3

    def test_epi_mm_batch_arrays_alias(self):
        """Test that the arrays alias matches the main batch entry point."""
        from algorithms.calibration import Calibration as PythonCal
        from algorithms.epi import epi_mm_batch, epi_mm_batch_arrays
        from algorithms.parameters import MultimediaPar, VolumePar

        cal1 = PythonCal()
        cal1.set_pos(np.array([0.0, 0.0, 100.0]))
        cal1.set_angles(np.array([0.0, 0.0, 0.0]))

        cal2 = PythonCal()
        cal2.set_pos(np.array([25.0, 0.0, 100.0]))
        cal2.set_angles(np.array([0.0, 0.0, 0.0]))

        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[5.0], n3=1.0)
        vpar = VolumePar(
            x_lay=[0.0, 100.0],
            z_min_lay=[0.0, 0.0],
            z_max_lay=[50.0, 50.0],
        )

        xl = np.array([10.0, 20.0], dtype=np.float64)
        yl = np.array([5.0, 15.0], dtype=np.float64)

        expected = epi_mm_batch(xl, yl, cal1, cal2, mm, vpar)
        actual = epi_mm_batch_arrays(xl, yl, cal1, cal2, mm, vpar)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

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
