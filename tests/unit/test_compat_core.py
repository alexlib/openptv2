"""
Tests for compatibility layer core objects (Phase 1).
"""

from pathlib import Path

import numpy as np
import pytest

from openptv2.calibration import Calibration
from openptv2.parameters import (
    ControlParams,
    MultimediaParams,
    SequenceParams,
    TargetParams,
    TrackingParams,
    VolumeParams,
)
from openptv2.tracking_framebuf import CORRES_NONE, Frame, Target, TargetArray

# Test data paths
TEST_DATA = Path(__file__).parent.parent.parent / "test_data" / "synthetic"
CALIB_PATH = TEST_DATA / "cal" / "cam1.tif"


class TestCalibrationCompat:
    """Test Calibration wrapper API."""

    def test_from_file(self):
        """Test reading calibration from file (instance method, like optv)."""
        cal = Calibration()
        cal.from_file(
            str(CALIB_PATH) + ".ori",
            str(CALIB_PATH) + ".addpar"
        )
        assert isinstance(cal, Calibration)

        # Test getters
        pos = cal.get_pos()
        assert pos.shape == (3,)
        assert isinstance(pos, np.ndarray)

        angles = cal.get_angles()
        assert angles.shape == (3,)

        dm = cal.get_rotation_matrix()
        assert dm.shape == (3, 3)

    def test_setters(self):
        """Test calibration setters."""
        cal = Calibration()

        # Test position
        pos = np.array([1.0, 2.0, 3.0])
        cal.set_pos(pos)
        np.testing.assert_array_equal(cal.get_pos(), pos)

        # Test angles
        angles = np.array([0.1, 0.2, 0.3])
        cal.set_angles(angles)
        retrieved = cal.get_angles()
        np.testing.assert_allclose(retrieved, angles, rtol=1e-6)

        # Test primary point (3 elements: xh, yh, cc)
        pp = np.array([5.0, 6.0, 100.0])
        cal.set_primary_point(pp)
        np.testing.assert_array_equal(cal.get_primary_point(), pp)

        # Test radial distortion
        dist = np.array([0.001, 0.002, 0.003])
        cal.set_radial_distortion(dist)
        np.testing.assert_array_equal(cal.get_radial_distortion(), dist)

        # Test decentering
        decent = np.array([0.0001, 0.0002])
        cal.set_decentering(decent)
        np.testing.assert_array_equal(cal.get_decentering(), decent)

        # Test affine
        affine = np.array([1.0, 0.0])
        cal.set_affine_trans(affine)
        np.testing.assert_array_equal(cal.get_affine(), affine)

        # Test glass vector
        gvec = np.array([0.0, 0.0, 1.0])
        cal.set_glass_vec(gvec)
        np.testing.assert_array_equal(cal.get_glass_vec(), gvec)


class TestParametersCompat:
    """Test parameter wrapper APIs."""

    def test_control_params(self):
        """Test ControlParams wrapper."""
        cpar = ControlParams(num_cams=4)
        assert cpar.get_num_cams() == 4

        # Test image size
        cpar.set_image_size((1280, 1024))
        assert cpar.get_image_size() == (1280, 1024)

        # Test pixel size
        cpar.set_pixel_size((0.012, 0.012))
        assert cpar.get_pixel_size() == (0.012, 0.012)

        # Test flags
        cpar.set_hp_flag(1)
        assert cpar.get_hp_flag() == 1

        cpar.set_chfield(0)
        assert cpar.get_chfield() == 0

    def test_volume_params(self):
        """Test VolumeParams wrapper."""
        vpar = VolumeParams()

        X_lay = np.array([-100.0, 100.0])
        vpar.set_X_lay(X_lay)
        np.testing.assert_array_equal(vpar.get_X_lay(), X_lay)

        Zmin = np.array([-50.0, -50.0])
        vpar.set_Zmin_lay(Zmin)
        np.testing.assert_array_equal(vpar.get_Zmin_lay(), Zmin)

        vpar.set_cn(0.5)
        assert vpar.get_cn() == 0.5

    def test_tracking_params(self):
        """Test TrackingParams wrapper."""
        tpar = TrackingParams()

        tpar.set_dvxmin(-10.0)
        assert tpar.get_dvxmin() == -10.0

        tpar.set_dvxmax(10.0)
        assert tpar.get_dvxmax() == 10.0

        tpar.set_dangle(0.5)
        assert tpar.get_dangle() == 0.5

        tpar.set_dacc(1.0)
        assert tpar.get_dacc() == 1.0

        tpar.set_add(1)
        assert tpar.get_add() == 1

    def test_sequence_params(self):
        """Test SequenceParams wrapper."""
        spar = SequenceParams(num_cams=4)

        spar.set_first(10001)
        assert spar.get_first() == 10001

        spar.set_last(10005)
        assert spar.get_last() == 10005

    def test_target_params(self):
        """Test TargetParams wrapper."""
        tpar = TargetParams()

        # Test grey thresholds
        gvthres = np.array([20, 20, 20, 20], dtype=np.int32)
        tpar.set_grey_thresholds(gvthres)
        np.testing.assert_array_equal(tpar.get_grey_thresholds(), gvthres)

        # Test discontinuity
        tpar.set_max_discontinuity(5)
        assert tpar.get_max_discontinuity() == 5

        # Test pixel count bounds
        tpar.set_pixel_count_bounds((5, 1000))
        assert tpar.get_pixel_count_bounds() == (5, 1000)

        # Test size bounds
        tpar.set_xsize_bounds((1, 100))
        assert tpar.get_xsize_bounds() == (1, 100)

        tpar.set_ysize_bounds((1, 100))
        assert tpar.get_ysize_bounds() == (1, 100)

        # Test min sum grey
        tpar.set_min_sum_grey(100)
        assert tpar.get_min_sum_grey() == 100

        # Test cross size
        tpar.set_cross_size(2)
        assert tpar.get_cross_size() == 2

    def test_multimedia_params(self):
        """Test MultimediaParams wrapper."""
        mm = MultimediaParams(n1=1.0, n3=1.49)

        assert mm.get_n1() == 1.0
        assert mm.get_n3() == 1.49
        assert mm.get_nlay() >= 1

        mm.set_n1(1.0003)
        assert mm.get_n1() == 1.0003


class TestTrackingFrameBufCompat:
    """Test tracking frame buffer wrapper APIs."""

    def test_target_creation(self):
        """Test Target wrapper creation."""
        targ = Target(pnr=5, x=100.0, y=200.0, n=50, nx=10, ny=10, sumg=1000, tnr=CORRES_NONE)

        assert targ.pnr() == 5
        assert targ.x() == 100.0
        assert targ.y() == 200.0
        assert targ.tnr() == CORRES_NONE

        pos = targ.pos()
        np.testing.assert_array_equal(pos, [100.0, 200.0])

        counts = targ.count_pixels()
        assert counts == (50, 10, 10)

        assert targ.sum_grey_value() == 1000

    def test_target_setters(self):
        """Test Target setters."""
        targ = Target()

        targ.set_pnr(42)
        assert targ.pnr() == 42

        targ.set_pos([123.4, 567.8])
        pos = targ.pos()
        np.testing.assert_allclose(pos, [123.4, 567.8])

        targ.set_tnr(7)
        assert targ.tnr() == 7

        targ.set_pixel_counts(60, 12, 11)
        assert targ.count_pixels() == (60, 12, 11)

        targ.set_sum_grey_value(2000)
        assert targ.sum_grey_value() == 2000

    def test_target_array_creation(self):
        """Test TargetArray creation and indexing."""
        ta = TargetArray(5)
        assert len(ta) == 5

        # Test indexing
        targ = ta[0]
        assert isinstance(targ, Target)

        # Test setting
        from openptv2.algorithms.tracking_frame_buf import Target as AlgoTarget
        new_target = AlgoTarget(pnr=10, x=50.0, y=60.0, n=20, nx=5, ny=5, sumg=500, tnr=-1)
        ta[1] = new_target
        assert ta[1].pnr() == 10

    def test_target_array_sort(self):
        """Test TargetArray sorting."""
        from openptv2.algorithms.tracking_frame_buf import Target as AlgoTarget

        targets = [
            AlgoTarget(pnr=i, x=float(i), y=100.0 - i*10, n=10, nx=3, ny=3, sumg=100, tnr=-1)
            for i in range(5)
        ]
        ta = TargetArray(targets)

        # Sort by Y
        ta.sort_y()

        # Check sorted order (Y values should be ascending)
        y_vals = [ta[i].y() for i in range(5)]
        assert y_vals == sorted(y_vals)

    def test_frame_creation(self):
        """Test Frame wrapper creation."""
        frame = Frame(num_cams=4)
        assert frame._num_cams == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
