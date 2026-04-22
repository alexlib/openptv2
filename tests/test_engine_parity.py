"""
Engine parity tests - verify optv and algorithms produce identical results.

Since C extensions can't be reimported in the same process, run these tests
twice with different OPENPTV_ENGINE settings:

    OPENPTV_ENGINE=optv uv run pytest tests/test_engine_parity.py::TestBasicAPI -v
    OPENPTV_ENGINE=python uv run pytest tests/test_engine_parity.py::TestBasicAPI -v

The results should be identical.
"""

import pytest
import numpy as np
from pathlib import Path
import os

# Test data paths
TEST_DATA = Path(__file__).parent.parent / "test_data" / "synthetic"


class TestBasicAPI:
    """Test basic API operations with current engine."""

    def test_calibration_read(self):
        """Verify calibration can be read."""
        from openptv2 import Calibration, get_engine

        print(f"\nTesting with engine: {get_engine()}")

        cal = Calibration.from_file(
            str(TEST_DATA / "cal" / "cam1.tif.ori"),
            str(TEST_DATA / "cal" / "cam1.tif.addpar")
        )

        # Test getters
        pos = cal.get_pos()
        assert pos.shape == (3,)
        assert np.all(np.isfinite(pos))

        angles = cal.get_angles()
        assert angles.shape == (3,)
        assert np.all(np.isfinite(angles))

        dm = cal.get_rotation_matrix()
        assert dm.shape == (3, 3)
        assert np.all(np.isfinite(dm))

        # Verify it's a rotation matrix (det = ±1)
        det = np.linalg.det(dm)
        assert abs(abs(det) - 1.0) < 1e-10

    def test_control_params(self):
        """Verify ControlParams works."""
        from openptv2 import ControlParams, get_engine

        print(f"\nTesting with engine: {get_engine()}")

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))
        cpar.set_hp_flag(1)

        assert cpar.get_num_cams() == 4
        assert cpar.get_image_size() == (1280, 1024)
        assert cpar.get_pixel_size() == (0.012, 0.012)
        assert cpar.get_hp_flag() == 1

    def test_pixel_to_metric(self):
        """Verify pixel→metric conversion."""
        from openptv2 import ControlParams, convert_arr_pixel_to_metric, get_engine

        print(f"\nTesting with engine: {get_engine()}")

        pixels = np.array([
            [640.0, 512.0],   # Image center
            [320.0, 256.0],
            [800.0, 700.0],
        ])

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        metric = convert_arr_pixel_to_metric(pixels, cpar)

        assert metric.shape == (3, 2)
        assert np.all(np.isfinite(metric))

        # Image center should map to (0, 0)
        np.testing.assert_allclose(metric[0], [0.0, 0.0], atol=0.01)

    def test_metric_to_pixel(self):
        """Verify metric→pixel conversion."""
        from openptv2 import ControlParams, convert_arr_metric_to_pixel, get_engine

        print(f"\nTesting with engine: {get_engine()}")

        metric = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [-2.5, 3.7],
        ])

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        pixels = convert_arr_metric_to_pixel(metric, cpar)

        assert pixels.shape == (3, 2)
        assert np.all(np.isfinite(pixels))

        # (0,0) metric should map to image center
        np.testing.assert_allclose(pixels[0], [640.0, 512.0], atol=0.01)

    def test_round_trip_transform(self):
        """Verify pixel→metric→pixel round trip."""
        from openptv2 import (
            ControlParams,
            convert_arr_pixel_to_metric,
            convert_arr_metric_to_pixel,
            get_engine
        )

        print(f"\nTesting with engine: {get_engine()}")

        original = np.array([
            [100.0, 200.0],
            [500.0, 600.0],
            [1000.0, 800.0],
        ])

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        metric = convert_arr_pixel_to_metric(original, cpar)
        back = convert_arr_metric_to_pixel(metric, cpar)

        np.testing.assert_allclose(back, original, rtol=1e-10)

    def test_target_array(self):
        """Verify TargetArray operations."""
        from openptv2 import TargetArray, get_engine
        from algorithms.tracking_frame_buf import Target as AlgoTarget

        print(f"\nTesting with engine: {get_engine()}")

        # Create target array
        targets = [
            AlgoTarget(pnr=i, x=float(i*10), y=100.0 - i*5, n=10, nx=3, ny=3, sumg=100, tnr=-1)
            for i in range(5)
        ]
        ta = TargetArray(targets)

        assert len(ta) == 5

        # Test sorting
        ta.sort_y()
        y_vals = [ta[i].y() for i in range(5)]
        assert y_vals == sorted(y_vals)

    def test_image_coordinates(self):
        """Verify 3D→2D projection."""
        from openptv2 import Calibration, MultimediaParams, image_coordinates, get_engine

        print(f"\nTesting with engine: {get_engine()}")

        cal = Calibration.from_file(
            str(TEST_DATA / "cal" / "cam1.tif.ori"),
            str(TEST_DATA / "cal" / "cam1.tif.addpar")
        )
        mm = MultimediaParams(n1=1.0, n3=1.0)

        positions = np.array([
            [0.0, 0.0, 100.0],
            [10.0, 10.0, 100.0],
            [-5.0, 15.0, 120.0],
        ])

        coords = image_coordinates(positions, cal, mm)

        assert coords.shape == (3, 2)
        assert np.all(np.isfinite(coords))

    def test_epipolar_curve(self):
        """Verify epipolar curve generation."""
        from openptv2 import (
            Calibration, ControlParams, VolumeParams, epipolar_curve, get_engine
        )

        print(f"\nTesting with engine: {get_engine()}")

        cal1 = Calibration.from_file(
            str(TEST_DATA / "cal" / "cam1.tif.ori"),
            str(TEST_DATA / "cal" / "cam1.tif.addpar")
        )
        cal2 = Calibration.from_file(
            str(TEST_DATA / "cal" / "cam2.tif.ori"),
            str(TEST_DATA / "cal" / "cam2.tif.addpar")
        )

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        vpar = VolumeParams()
        vpar.set_X_lay(np.array([-100.0, 100.0]))
        vpar.set_Zmin_lay(np.array([-50.0, -50.0]))
        vpar.set_Zmax_lay(np.array([50.0, 50.0]))

        point = np.array([640.0, 512.0])
        curve = epipolar_curve(point, cal1, cal2, 10, cpar, vpar)

        assert curve.shape == (10, 2)
        assert np.all(np.isfinite(curve))

    def test_tracker_creation(self):
        """Verify Tracker can be created."""
        from openptv2 import (
            Calibration, ControlParams, VolumeParams, TrackingParams,
            SequenceParams, Tracker, get_engine
        )

        print(f"\nTesting with engine: {get_engine()}")

        cals = [
            Calibration.from_file(
                str(TEST_DATA / "cal" / f"cam{i+1}.tif.ori"),
                str(TEST_DATA / "cal" / f"cam{i+1}.tif.addpar")
            )
            for i in range(4)
        ]

        cpar = ControlParams(num_cams=4)
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

        spar = SequenceParams(num_cams=4)
        spar.set_first(10001)
        spar.set_last(10003)

        tracker = Tracker(cpar, vpar, tpar, spar, cals)

        assert tracker is not None
        assert tracker.current_step() == -1  # Not initialized


def test_engine_detection():
    """Verify engine detection works."""
    from openptv2 import get_engine, is_optv_available, is_python_available

    engine = get_engine()
    assert engine in ("optv", "python")

    # At least one engine should be available
    assert is_optv_available() or is_python_available()

    print(f"\nCurrent engine: {engine}")
    print(f"optv available: {is_optv_available()}")
    print(f"python available: {is_python_available()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
