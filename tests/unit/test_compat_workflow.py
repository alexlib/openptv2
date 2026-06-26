"""
Tests for compatibility layer workflow (Phase 3: Correspondences & Tracker).
"""

import pytest
import numpy as np
from pathlib import Path

from openptv2.algorithms.compat.calibration import Calibration
from openptv2.algorithms.compat.parameters import (
    ControlParams, VolumeParams, TrackingParams, SequenceParams, TargetParams
)
from openptv2.algorithms.compat.tracking_framebuf import TargetArray, Target
from openptv2.algorithms.compat.correspondences import MatchedCoords, correspondences
from openptv2.algorithms.compat.tracker import Tracker, default_naming
from openptv2.algorithms.tracking_frame_buf import Target as AlgoTarget


# Test data paths
TEST_DATA = Path(__file__).parent.parent.parent / "test_data" / "synthetic"


def _load_cal(ori_file, add_file):
    """Load calibration using instance method pattern (matching optv API)."""
    cal = Calibration()
    cal.from_file(ori_file, add_file)
    return cal


class TestMatchedCoordsCompat:
    """Test MatchedCoords wrapper."""

    def test_matched_coords_creation(self):
        """Test MatchedCoords initialization."""
        # Create some targets
        targets = [
            AlgoTarget(pnr=0, x=100.0, y=200.0, n=10, nx=3, ny=3, sumg=500, tnr=-1),
            AlgoTarget(pnr=1, x=150.0, y=250.0, n=12, nx=4, ny=3, sumg=600, tnr=-1),
        ]
        ta = TargetArray(targets)

        # Create calibration and control params
        cal = Calibration()
        cal.set_radial_distortion(np.array([0.0, 0.0, 0.0]))

        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        # Create MatchedCoords
        mc = MatchedCoords(ta, cpar, cal, tol=1e-6)

        # Should have same number of corrected coords
        assert len(mc._corrected) == 2

    def test_matched_coords_as_arrays(self):
        """Test as_arrays method."""
        targets = [
            AlgoTarget(pnr=5, x=100.0, y=200.0, n=10, nx=3, ny=3, sumg=500, tnr=-1),
            AlgoTarget(pnr=7, x=150.0, y=250.0, n=12, nx=4, ny=3, sumg=600, tnr=-1),
        ]
        ta = TargetArray(targets)

        cal = Calibration()
        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        mc = MatchedCoords(ta, cpar, cal, reset_numbers=False)
        pos, pnr = mc.as_arrays()

        assert pos.shape == (2, 2)
        assert pnr.shape == (2,)
        assert pnr[0] == 5
        assert pnr[1] == 7

    def test_matched_coords_get_by_pnrs(self):
        """Test get_by_pnrs filtering."""
        targets = [
            AlgoTarget(pnr=i, x=100.0 + i*10, y=200.0, n=10, nx=3, ny=3, sumg=500, tnr=-1)
            for i in range(5)
        ]
        ta = TargetArray(targets)

        cal = Calibration()
        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        mc = MatchedCoords(ta, cpar, cal)
        filtered = mc.get_by_pnrs([1, 3])

        assert filtered.shape == (2, 2)

    def test_matched_coords_empty(self):
        """Test MatchedCoords with empty target list."""
        ta = TargetArray(0)

        cal = Calibration()
        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        mc = MatchedCoords(ta, cpar, cal)
        pos, pnr = mc.as_arrays()

        assert pos.shape == (0, 2)
        assert pnr.shape == (0,)


class TestCorrespondencesCompat:
    """Test correspondences wrapper."""

    def test_correspondences_no_targets(self):
        """Test correspondences with no targets."""
        num_cams = 4

        # Empty target arrays
        img_pts = [TargetArray(0) for _ in range(num_cams)]

        # Load calibrations
        cals = [
            _load_cal(
                str(TEST_DATA / "cal" / f"cam{i+1}.tif.ori"),
                str(TEST_DATA / "cal" / f"cam{i+1}.tif.addpar")
            )
            for i in range(num_cams)
        ]

        cpar = ControlParams(num_cams=num_cams)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        # Create MatchedCoords (empty)
        flat_coords = [MatchedCoords(ta, cpar, cals[i]) for i, ta in enumerate(img_pts)]

        vpar = VolumeParams()
        vpar.set_X_lay(np.array([-100.0, 100.0]))
        vpar.set_Zmin_lay(np.array([-50.0, -50.0]))
        vpar.set_Zmax_lay(np.array([50.0, 50.0]))

        # Run correspondences
        sorted_pos, sorted_corresp, num_targs = correspondences(
            img_pts, flat_coords, cals, vpar, cpar
        )

        # Should have no correspondences
        assert all(arr.shape[1] == 0 for arr in sorted_pos)
        assert len(num_targs) == num_cams


class TestTrackerCompat:
    """Test Tracker wrapper."""

    def test_tracker_creation(self):
        """Test Tracker initialization."""
        num_cams = 4

        cals = [
            _load_cal(
                str(TEST_DATA / "cal" / f"cam{i+1}.tif.ori"),
                str(TEST_DATA / "cal" / f"cam{i+1}.tif.addpar")
            )
            for i in range(num_cams)
        ]

        cpar = ControlParams(num_cams=num_cams)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        vpar = VolumeParams()
        vpar.set_X_lay(np.array([-100.0, 100.0]))
        vpar.set_Zmin_lay(np.array([-50.0, -50.0]))
        vpar.set_Zmax_lay(np.array([50.0, 50.0]))

        tpar = TrackingParams()
        tpar.set_dvxmin(-10.0)
        tpar.set_dvxmax(10.0)
        tpar.set_dvymin(-10.0)
        tpar.set_dvymax(10.0)
        tpar.set_dvzmin(-10.0)
        tpar.set_dvzmax(10.0)
        tpar.set_dangle(0.5)
        tpar.set_dacc(5.0)

        spar = SequenceParams(num_cams=num_cams)
        spar.set_first(10001)
        spar.set_last(10005)

        tracker = Tracker(cpar, vpar, tpar, spar, cals)

        assert tracker is not None
        assert tracker.current_step() == -1  # Not initialized yet

    def test_tracker_restart(self):
        """Test Tracker restart (initialization)."""
        num_cams = 1

        cals = [
            _load_cal(
                str(TEST_DATA / "cal" / "cam1.tif.ori"),
                str(TEST_DATA / "cal" / "cam1.tif.addpar")
            )
        ]

        cpar = ControlParams(num_cams=num_cams)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        vpar = VolumeParams()
        vpar.set_X_lay(np.array([-100.0, 100.0]))
        vpar.set_Zmin_lay(np.array([-50.0, -50.0]))
        vpar.set_Zmax_lay(np.array([50.0, 50.0]))

        tpar = TrackingParams()
        tpar.set_dvxmin(-10.0)
        tpar.set_dvxmax(10.0)

        spar = SequenceParams(num_cams=num_cams)
        spar.set_first(10001)
        spar.set_last(10003)

        tracker = Tracker(cpar, vpar, tpar, spar, cals)
        tracker.restart()

        assert tracker.current_step() == 10001
        assert tracker._is_initialized

    def test_tracker_default_naming(self):
        """Test default file naming."""
        assert 'corres' in default_naming
        assert 'linkage' in default_naming
        assert 'prio' in default_naming

    def test_tracker_not_initialized_error(self):
        """Test error when calling methods before initialization."""
        num_cams = 1

        cals = [Calibration()]
        cpar = ControlParams(num_cams=num_cams)
        vpar = VolumeParams()
        tpar = TrackingParams()
        spar = SequenceParams(num_cams=num_cams)
        spar.set_first(10001)
        spar.set_last(10003)

        tracker = Tracker(cpar, vpar, tpar, spar, cals)

        # Should raise error before restart()
        with pytest.raises(RuntimeError, match="not initialized"):
            tracker.step_forward()

        with pytest.raises(RuntimeError, match="not initialized"):
            tracker.finalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
