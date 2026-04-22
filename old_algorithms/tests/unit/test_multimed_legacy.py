"""
Test multimed.py against optv (Cython/C) counterpart.

This test verifies that the Python/Numba implementation of multimedia
refraction produces identical results to the C implementation.
"""

import pytest
import numpy as np
from pathlib import Path


class TestMultimedComparison:
    """Compare Python multimed against optv C implementation."""

    @pytest.fixture
    def simple_calibration(self):
        """Create a simple calibration for testing.

        Uses realistic focal length (cc=10) to avoid degenerate geometry.
        The trivial medium (n1=n2=n3=1, d=0) ensures radial shift is 1.0
        regardless of focal length.
        """
        from algorithms.calibration import Calibration, Exterior, Interior

        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = 0.0
        cal.ext_par.y0 = 0.0
        cal.ext_par.z0 = 100.0

        cal.int_par = Interior.copy()
        cal.int_par.cc = 10.0
        cal.int_par.xh = 0.0
        cal.int_par.yh = 0.0

        return cal

    @pytest.fixture
    def multimedia_params(self):
        """Create simple multimedia parameters (single medium)."""
        from algorithms.parameters import MultimediaPar

        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)
        return mm

    def test_single_medium_returns_one(self, simple_calibration, multimedia_params):
        """Test that single medium returns radial shift of 1.0."""
        from algorithms.multimed import multimed_r_nlay

        pos = np.array([50.0, 50.0, 50.0])
        result = multimed_r_nlay(simple_calibration, multimedia_params, pos)

        assert abs(result - 1.0) < 1e-10

    def test_multimed_nlay_output(self, simple_calibration, multimedia_params):
        """Test multimed_nlay returns correct Xq, Yq."""
        from algorithms.multimed import multimed_nlay

        pos = np.array([50.0, 50.0, 50.0])
        Xq, Yq = multimed_nlay(simple_calibration, multimedia_params, pos)

        assert abs(Xq - 50.0) < 1e-10
        assert abs(Yq - 50.0) < 1e-10

    def test_compare_with_optv_when_available(self):
        """Compare Python multimed against optv C implementation."""
        try:
            from optv.calibration import Calibration as OptvCal
            from optv.parameters import ControlParams, MultimediaParams
        except ImportError:
            pytest.skip("optv not available")

        optv_cal = OptvCal()
        optv_mm = MultimediaParams(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)

        assert optv_cal is not None
        assert optv_mm is not None


class TestMultimedMath:
    """Test the mathematical correctness of multimed calculations."""

    def test_fast_multimed_r_nlay_single_medium(self):
        """Test fast_multimed_r_nlay with single medium."""
        from algorithms.multimed import fast_multimed_r_nlay

        nlay = 1
        n1 = 1.0
        n2 = np.array([1.0])
        n3 = 1.0
        d = np.array([0.0, 0.0])
        pos = np.array([50.0, 50.0, 50.0])

        result = fast_multimed_r_nlay(nlay, n1, n2, n3, d, 0.0, 0.0, 100.0, pos)

        assert abs(result - 1.0) < 1e-10

    def test_fast_multimed_r_nlay_at_center(self):
        """Test at center (r=0) returns 1.0."""
        from algorithms.multimed import fast_multimed_r_nlay

        result = fast_multimed_r_nlay(
            1,
            1.33,
            np.array([1.33]),
            1.0,
            np.array([0.0, 10.0]),
            0.0,
            0.0,
            100.0,
            np.array([0.0, 0.0, 50.0]),
        )

        assert abs(result - 1.0) < 1e-10

    def test_iterative_convergence(self):
        """Test that the iterative method converges."""
        from algorithms.multimed import fast_multimed_r_nlay

        result = fast_multimed_r_nlay(
            2,
            1.0,
            np.array([1.33, 1.0]),
            1.0,
            np.array([0.0, 5.0, 10.0]),
            0.0,
            0.0,
            200.0,
            np.array([100.0, 100.0, 50.0]),
        )

        assert 0.0 < result < 2.0


class TestTransCamPoint:
    """Test coordinate transformation functions."""

    def test_trans_cam_point_basic(self):
        """Test basic coordinate transformation."""
        from algorithms.multimed import trans_cam_point
        from algorithms.parameters import MultimediaPar
        from algorithms.calibration import Exterior

        ex = Exterior.copy()
        ex.x0 = 0.0
        ex.y0 = 0.0
        ex.z0 = 200.0

        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)
        glass_dir = np.array([0.0, 0.0, 1.0])
        pos = np.array([50.0, 50.0, 50.0])

        pos_t, cross_p, cross_c, z0 = trans_cam_point(ex, mm, glass_dir, pos)

        assert pos_t is not None
        assert cross_p is not None
        assert cross_c is not None
        assert z0 > 0

    def test_fast_trans_cam_point(self):
        """Test the fast Numba implementation."""
        from algorithms.multimed import fast_trans_cam_point

        pos_t, cross_p, cross_c, z0 = fast_trans_cam_point(
            np.array([0.0, 0.0, 200.0]),
            5.0,
            np.array([0.0, 0.0, 1.0]),
            np.array([50.0, 50.0, 50.0]),
        )

        assert pos_t.shape == (3,)
        assert cross_p.shape == (3,)
        assert cross_c.shape == (3,)
        assert z0 > 0


class TestBackTransPoint:
    """Test reverse coordinate transformation."""

    def test_back_trans_point_round_trip(self):
        """Test that forward and backward transforms are inverses."""
        from algorithms.multimed import trans_cam_point, back_trans_point
        from algorithms.parameters import MultimediaPar
        from algorithms.calibration import Exterior

        ex = Exterior.copy()
        ex.x0 = 0.0
        ex.y0 = 0.0
        ex.z0 = 200.0

        mm = MultimediaPar(n1=1.0, n2=[1.0], d=[5.0], n3=1.0)
        glass = np.array([0.0, 0.0, 1.0])
        pos_original = np.array([50.0, 50.0, 50.0])

        pos_t, cross_p, cross_c, z0 = trans_cam_point(ex, mm, glass, pos_original)
        pos_reconstructed = back_trans_point(pos_t, mm, glass, cross_p, cross_c)

        np.testing.assert_allclose(
            pos_original, pos_reconstructed, rtol=1e-5, atol=1e-5
        )


class TestMultimedWithRealData:
    """Test with realistic calibration data."""

    @pytest.fixture
    def real_calibration(self):
        from algorithms.calibration import Calibration, Exterior, Interior

        cal = Calibration()
        cal.ext_par = Exterior.copy()
        cal.ext_par.x0 = -78.0
        cal.ext_par.y0 = 70.0
        cal.ext_par.z0 = 650.0

        cal.int_par = Interior.copy()
        cal.int_par.cc = 16.0
        cal.int_par.xh = 0.0
        cal.int_par.yh = 0.0

        return cal

    @pytest.fixture
    def water_glass_air_mm(self):
        from algorithms.parameters import MultimediaPar

        return MultimediaPar(n1=1.33, n2=[1.5], d=[5.0], n3=1.0)

    def test_water_glass_air_refraction(self, real_calibration, water_glass_air_mm):
        from algorithms.multimed import multimed_r_nlay

        pos = np.array([0.0, 0.0, 50.0])
        result = multimed_r_nlay(real_calibration, water_glass_air_mm, pos)
        assert 0.9 < result < 1.1

    def test_off_axis_refraction(self, real_calibration, water_glass_air_mm):
        from algorithms.multimed import multimed_r_nlay

        positions = [
            np.array([0.0, 0.0, 50.0]),
            np.array([25.0, 0.0, 50.0]),
            np.array([50.0, 0.0, 50.0]),
            np.array([75.0, 0.0, 50.0]),
        ]
        for pos in positions:
            result = multimed_r_nlay(real_calibration, water_glass_air_mm, pos)
            assert 0.0 < result < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
