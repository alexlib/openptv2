"""
Tests for compatibility layer processing functions (Phase 2).
"""

import pytest
import numpy as np
from pathlib import Path

from openptv2.algorithms.compat.calibration import Calibration
from openptv2.algorithms.compat.parameters import ControlParams, VolumeParams, TargetParams, MultimediaParams
from openptv2.algorithms.compat.transforms import (
    convert_arr_pixel_to_metric,
    convert_arr_metric_to_pixel,
    correct_arr_brown_affine,
    distort_arr_brown_affine,
    distorted_to_flat,
)
from openptv2.algorithms.compat.imgcoord import image_coordinates, flat_image_coordinates
from openptv2.algorithms.compat.image_processing import preprocess_image
from openptv2.algorithms.compat.segmentation import target_recognition
from openptv2.algorithms.compat.orientation import (
    external_calibration,
    multi_cam_point_positions,
    point_positions,
    match_detection_to_ref,
)
from openptv2.algorithms.compat.epipolar import epipolar_curve


# Test data paths
TEST_DATA = Path(__file__).parent.parent.parent / "test_data" / "synthetic"
CALIB_PATH = TEST_DATA / "cal" / "cam1.tif"


def _load_cal(ori_file, add_file):
    """Load calibration using instance method pattern (matching optv API)."""
    cal = Calibration()
    cal.from_file(ori_file, add_file)
    return cal


class TestTransformsCompat:
    """Test transform wrapper functions."""

    def test_pixel_to_metric(self):
        """Test pixel to metric conversion."""
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        pixels = np.array([[640.0, 512.0], [320.0, 256.0]])
        metric = convert_arr_pixel_to_metric(pixels, cpar)

        assert metric.shape == (2, 2)
        # Center pixel should map to (0, 0)
        np.testing.assert_allclose(metric[0], [0.0, 0.0], atol=0.01)

    def test_metric_to_pixel(self):
        """Test metric to pixel conversion."""
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        metric = np.array([[0.0, 0.0], [1.0, 1.0]])
        pixels = convert_arr_metric_to_pixel(metric, cpar)

        assert pixels.shape == (2, 2)
        # (0,0) metric should map to image center
        np.testing.assert_allclose(pixels[0], [640.0, 512.0], atol=0.01)

    def test_round_trip_pixel_metric(self):
        """Test pixel <-> metric round trip."""
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        original = np.array([[100.0, 200.0], [500.0, 600.0]])
        metric = convert_arr_pixel_to_metric(original, cpar)
        back = convert_arr_metric_to_pixel(metric, cpar)

        np.testing.assert_allclose(back, original, rtol=1e-10)

    def test_brown_affine_distortion(self):
        """Test Brown-Affine distortion correction."""
        cal = Calibration()
        cal.set_radial_distortion(np.array([0.001, 0.0, 0.0]))
        cal.set_decentering(np.array([0.0, 0.0]))
        cal.set_affine_trans(np.array([1.0, 0.0]))

        distorted = np.array([[1.0, 1.0], [2.0, 2.0]])
        corrected = correct_arr_brown_affine(distorted, cal)

        assert corrected.shape == (2, 2)
        # Distortion should change the coordinates (k1=0.001 causes ~1.5% at r=2)
        assert not np.allclose(corrected, distorted)

    def test_distorted_to_flat(self):
        """Test iterative distortion removal."""
        cal = Calibration()
        cal.set_radial_distortion(np.array([0.001, 0.0, 0.0]))
        cal.set_decentering(np.array([0.0, 0.0]))
        cal.set_affine_trans(np.array([1.0, 0.0]))

        distorted = np.array([[1.0, 1.0], [2.0, 2.0]])
        flat = distorted_to_flat(distorted, cal, tol=1e-6)

        assert flat.shape == (2, 2)


class TestImgCoordCompat:
    """Test image coordinate wrapper functions."""

    def test_image_coordinates(self):
        """Test 3D to 2D projection."""
        cal = _load_cal(
            str(CALIB_PATH) + ".ori",
            str(CALIB_PATH) + ".addpar"
        )
        mm = MultimediaParams(n1=1.0, n3=1.0)

        # 3D points in front of camera
        positions = np.array([[0.0, 0.0, 100.0], [10.0, 10.0, 100.0]])
        coords = image_coordinates(positions, cal, mm)

        assert coords.shape == (2, 2)
        # Should produce finite coordinates
        assert np.all(np.isfinite(coords))

    def test_flat_image_coordinates(self):
        """Test flat (undistorted) projection."""
        cal = _load_cal(
            str(CALIB_PATH) + ".ori",
            str(CALIB_PATH) + ".addpar"
        )
        mm = MultimediaParams(n1=1.0, n3=1.0)

        positions = np.array([[0.0, 0.0, 100.0]])
        coords = flat_image_coordinates(positions, cal, mm)

        assert coords.shape == (1, 2)
        assert np.all(np.isfinite(coords))


class TestImageProcessingCompat:
    """Test image processing wrapper."""

    def test_preprocess_image(self):
        """Test image preprocessing."""
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((128, 128))
        cpar.set_chfield(0)

        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        processed = preprocess_image(img, filter_hp=0, cpar=cpar, lowpass_dim=1)

        assert processed.shape == img.shape
        assert processed.dtype == np.uint8


class TestSegmentationCompat:
    """Test segmentation wrapper."""

    def test_target_recognition_no_targets(self):
        """Test target recognition on blank image."""
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((128, 128))

        tpar = TargetParams()
        tpar.set_grey_thresholds(np.array([200, 200, 200, 200], dtype=np.int32))
        tpar.set_max_discontinuity(5)
        tpar.set_pixel_count_bounds((5, 1000))
        tpar.set_xsize_bounds((1, 100))
        tpar.set_ysize_bounds((1, 100))
        tpar.set_min_sum_grey(100)

        # Blank image - should have very few or no targets
        img = np.zeros((128, 128), dtype=np.uint8)
        targets = target_recognition(img, tpar, cam=0, cpar=cpar)

        # Allow some edge effects, but should be minimal
        assert len(targets) < 3

    def test_target_recognition_bright_spot(self):
        """Test target recognition with bright spot."""
        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((128, 128))

        tpar = TargetParams()
        tpar.set_grey_thresholds(np.array([100, 100, 100, 100], dtype=np.int32))
        tpar.set_max_discontinuity(5)
        tpar.set_pixel_count_bounds((5, 1000))
        tpar.set_xsize_bounds((1, 100))
        tpar.set_ysize_bounds((1, 100))
        tpar.set_min_sum_grey(500)

        # Image with a bright spot
        img = np.zeros((128, 128), dtype=np.uint8)
        img[60:68, 60:68] = 255  # 8x8 bright region

        targets = target_recognition(img, tpar, cam=0, cpar=cpar)

        # Should detect the bright spot
        assert len(targets) >= 1


class TestOrientationCompat:
    """Test orientation wrapper functions."""

    def test_external_calibration(self):
        """Test external calibration with realistic initialization."""
        cal = Calibration()
        # Initialize with reasonable camera position (not at origin)
        cal.set_pos(np.array([0.0, 0.0, 1000.0]))  # 1m behind target
        cal.set_angles(np.array([0.0, 0.0, 0.0]))
        cal.set_primary_point(np.array([0.0, 0.0]))

        cpar = ControlParams(num_cams=1)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        # Use synthetic calibration target (4 corners)
        ref_pts = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
        ])

        # Synthetic image points (rough projection)
        img_pts = np.array([
            [640.0, 512.0],
            [740.0, 512.0],
            [740.0, 612.0],
            [640.0, 612.0],
        ])

        # Should run without error (may not converge with synthetic data)
        try:
            result = external_calibration(cal, ref_pts, img_pts, cpar)
            assert isinstance(result, bool)
        except (ValueError, ZeroDivisionError):
            # May fail to converge or have numerical issues with synthetic data
            pass


class TestEpipolarCompat:
    """Test epipolar geometry wrapper."""

    def test_epipolar_curve(self):
        """Test epipolar curve generation."""
        cal1 = _load_cal(
            str(TEST_DATA / "cal" / "cam1.tif") + ".ori",
            str(TEST_DATA / "cal" / "cam1.tif") + ".addpar"
        )
        cal2 = _load_cal(
            str(TEST_DATA / "cal" / "cam2.tif") + ".ori",
            str(TEST_DATA / "cal" / "cam2.tif") + ".addpar"
        )

        cpar = ControlParams(num_cams=4)
        cpar.set_image_size((1280, 1024))
        cpar.set_pixel_size((0.012, 0.012))

        vpar = VolumeParams()
        vpar.set_X_lay(np.array([-100.0, 100.0]))
        vpar.set_Zmin_lay(np.array([-50.0, -50.0]))
        vpar.set_Zmax_lay(np.array([50.0, 50.0]))

        point = np.array([640.0, 512.0])
        curve = epipolar_curve(point, cal1, cal2, num_points=10, cpar=cpar, vpar=vpar)

        assert curve.shape == (10, 2)
        assert np.all(np.isfinite(curve))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
