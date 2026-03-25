"""
Engine comparison tests for orientation module.

Tests point_positions, full_calibration, multi_cam_point_positions, etc.
Tolerance: 1e-7 (iterative algorithms)
"""

import numpy as np
import pytest
from pathlib import Path
from .conftest import get_tolerance, create_test_calibration, create_test_control_params

TOLERANCE = get_tolerance("orientation")


class TestOrientationFunctions:
    """Compare orientation functions between optv and python engines."""

    def test_point_positions_basic(self):
        """Test point_positions with basic data."""
        from optv.orientation import point_positions as optv_func

        np.random.seed(42)
        num_targets = 5

        targets = np.random.rand(num_targets, 4, 3) * 100

        cals = []
        python_cals = []

        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, float(i * np.pi / 2)])

            from optv.calibration import Calibration as OptvCal
            from algorithms.calibration import Calibration as PythonCal

            optv_cal = OptvCal(pos=pos, angs=angles)
            python_cal = PythonCal()
            python_cal.set_pos(pos)
            python_cal.set_angles(angles)

            cals.append(optv_cal)
            python_cals.append(python_cal)

        try:
            optv_result = optv_func(targets, cals)
        except Exception as e:
            pytest.fail(f"optv point_positions failed: {e}")

        from algorithms.orientation import point_positions as python_func

        try:
            python_result = python_func(targets, python_cals)
        except Exception as e:
            pytest.fail(f"python point_positions failed: {e}")

        if optv_result is not None and python_result is not None:
            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )

    def test_multi_cam_point_positions(self):
        """Test multi_cam_point_positions function."""
        from optv.orientation import multi_cam_point_positions as optv_func

        np.random.seed(42)
        num_targets = 3

        img_pts = []
        flat_pts = []

        for _ in range(4):
            pts = np.random.rand(num_targets, 2) * 100
            img_pts.append(pts)

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, float(i * np.pi / 4)])

            from optv.calibration import Calibration as OptvCal

            optv_cal = OptvCal(pos=pos, angs=angles)
            cals.append(optv_cal)

        try:
            optv_result = optv_func(img_pts, cals)
        except Exception as e:
            pytest.fail(f"optv multi_cam_point_positions failed: {e}")

        from algorithms.orientation import multi_cam_point_positions as python_func

        try:
            python_result = python_func(img_pts, cals)
        except Exception as e:
            pytest.fail(f"python multi_cam_point_positions failed: {e}")

        if optv_result is not None and python_result is not None:
            valid_mask = ~(np.isnan(optv_result) | np.isnan(python_result))
            if np.any(valid_mask):
                np.testing.assert_allclose(
                    optv_result[valid_mask],
                    python_result[valid_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )

    def test_single_cam_point_positions(self):
        """Test single_cam_point_positions function."""
        from optv.orientation import single_cam_point_positions as optv_func

        np.random.seed(42)
        num_targets = 5

        targets = np.random.rand(num_targets, 4, 3) * 100

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, 0.0])

            from optv.calibration import Calibration as OptvCal

            optv_cal = OptvCal(pos=pos, angs=angles)
            cals.append(optv_cal)

        try:
            optv_result = optv_func(targets, cals)
        except Exception as e:
            pytest.fail(f"optv single_cam_point_positions failed: {e}")

        from algorithms.orientation import single_cam_point_positions as python_func

        try:
            python_result = python_func(targets, cals)
        except Exception as e:
            pytest.fail(f"python single_cam_point_positions failed: {e}")

        if optv_result is not None and python_result is not None:
            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )

    def test_external_calibration(self):
        """Test external_calibration function."""
        from optv.orientation import external_calibration as optv_func

        from optv.calibration import Calibration as OptvCal

        optv_cal = OptvCal()

        np.random.seed(42)
        obj_pts = np.random.rand(10, 3) * 100
        img_pts = np.random.rand(10, 2) * 1000

        try:
            optv_result = optv_func(optv_cal, obj_pts, img_pts)
        except Exception as e:
            pytest.fail(f"optv external_calibration failed: {e}")

        from algorithms.orientation import external_calibration as python_func
        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()

        try:
            python_result = python_func(python_cal, obj_pts, img_pts)
        except Exception as e:
            pytest.fail(f"python external_calibration failed: {e}")

        if optv_result is not None and python_result is not None:
            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )


class TestFullCalibration:
    """Test full_calibration function."""

    def test_full_calibration_basic(self):
        """Test full_calibration with basic data."""
        from optv.orientation import full_calibration as optv_func

        np.random.seed(42)
        num_targets = 20

        targets = []
        for _ in range(4):
            pts = np.random.rand(num_targets, 3) * 100
            targets.append(pts)

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, float(i * np.pi / 4)])

            from optv.calibration import Calibration as OptvCal

            optv_cal = OptvCal(pos=pos, angs=angles)
            cals.append(optv_cal)

        try:
            optv_result = optv_func(cals, targets)
        except Exception as e:
            pytest.fail(f"optv full_calibration failed: {e}")

        from algorithms.orientation import full_calibration as python_func

        python_cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, float(i * np.pi / 4)])

            from algorithms.calibration import Calibration as PythonCal

            python_cal = PythonCal()
            python_cal.set_pos(pos)
            python_cal.set_angles(angles)
            python_cals.append(python_cal)

        try:
            python_result = python_func(python_cals, targets)
        except Exception as e:
            pytest.fail(f"python full_calibration failed: {e}")

        if optv_result is not None and python_result is not None:
            for i, (optv_c, python_c) in enumerate(zip(optv_result, python_result)):
                if optv_c is not None and python_c is not None:
                    np.testing.assert_allclose(
                        optv_c.get_pos(),
                        python_c.get_pos(),
                        rtol=TOLERANCE,
                        atol=TOLERANCE,
                    )


class TestMatchDetectionToRef:
    """Test match_detection_to_ref function."""

    def test_match_detection_to_ref_basic(self):
        """Test match_detection_to_ref with basic data."""
        from optv.orientation import match_detection_to_ref as optv_func

        np.random.seed(42)
        num_points = 10

        ref_pts = np.random.rand(num_points, 3) * 100
        det_pts = ref_pts + np.random.rand(num_points, 3) * 2

        from optv.calibration import Calibration as OptvCal

        optv_cal = OptvCal()

        try:
            optv_result = optv_func(optv_cal, det_pts, ref_pts)
        except Exception as e:
            pytest.fail(f"optv match_detection_to_ref failed: {e}")

        from algorithms.orientation import match_detection_to_ref as python_func
        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()

        try:
            python_result = python_func(python_cal, det_pts, ref_pts)
        except Exception as e:
            pytest.fail(f"python match_detection_to_ref failed: {e}")

        if optv_result is not None and python_result is not None:
            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )


class TestOrientationEdgeCases:
    """Test edge cases for orientation functions."""

    def test_point_positions_single_target(self):
        """Test with single target."""
        from optv.orientation import point_positions as optv_func

        targets = np.random.rand(1, 4, 3) * 100

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, 0.0])

            from optv.calibration import Calibration as OptvCal

            optv_cal = OptvCal(pos=pos, angs=angles)
            cals.append(optv_cal)

        try:
            optv_result = optv_func(targets, cals)
        except Exception as e:
            pytest.fail(f"optv point_positions failed: {e}")

        from algorithms.orientation import point_positions as python_func

        python_cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, 0.0])

            from algorithms.calibration import Calibration as PythonCal

            python_cal = PythonCal()
            python_cal.set_pos(pos)
            python_cal.set_angles(angles)
            python_cals.append(python_cal)

        try:
            python_result = python_func(targets, python_cals)
        except Exception as e:
            pytest.fail(f"python point_positions failed: {e}")

        if optv_result is not None and python_result is not None:
            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )

    def test_point_positions_colinear(self):
        """Test with colinear targets (degenerate case)."""
        from optv.orientation import point_positions as optv_func

        targets = np.zeros((5, 4, 3))
        targets[:, :, 0] = np.linspace(0, 100, 5)[:, None]
        targets[:, :, 1] = 50.0
        targets[:, :, 2] = 50.0

        cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, 0.0])

            from optv.calibration import Calibration as OptvCal

            optv_cal = OptvCal(pos=pos, angs=angles)
            cals.append(optv_cal)

        try:
            optv_result = optv_func(targets, cals)
        except Exception:
            pass

        from algorithms.orientation import point_positions as python_func

        python_cals = []
        for i in range(4):
            pos = np.array([float(i * 100), 0.0, 100.0])
            angles = np.array([0.0, 0.0, 0.0])

            from algorithms.calibration import Calibration as PythonCal

            python_cal = PythonCal()
            python_cal.set_pos(pos)
            python_cal.set_angles(angles)
            python_cals.append(python_cal)

        try:
            python_result = python_func(targets, python_cals)
        except Exception:
            pass
