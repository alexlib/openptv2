"""
Engine comparison tests for coordinate transforms (trafo/transforms).

Tests pixel <-> metric coordinate conversions and brown-affine transforms.
Tolerance: 1e-10 (precise mathematical transformations)
"""

import numpy as np
import pytest
from .conftest import get_tolerance, create_test_control_params, compare_arrays

TOLERANCE = get_tolerance("trafo")


class TestTransforms:
    """Compare coordinate transform results between optv and python engines."""

    def test_pixel_to_metric_synthetic(self, synthetic_pixel_coords):
        """Test pixel_to_metric with synthetic pixel coordinates."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None, "optv ControlParams not available"
        assert python_cpar is not None, "python ControlPar not available"

        from optv.transforms import convert_arr_pixel_to_metric as optv_func
        from algorithms.trafo import arr_pixel_to_metric as python_func

        pixel_coords = synthetic_pixel_coords.copy()

        optv_result = optv_func(pixel_coords, optv_cpar)
        python_result = python_func(
            pixel_coords.astype(np.int32),
            python_cpar.imx,
            python_cpar.imy,
            python_cpar.pix_x,
            python_cpar.pix_y,
        )

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_metric_to_pixel_synthetic(self, synthetic_2d_metric_coords):
        """Test metric_to_pixel with synthetic metric coordinates."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None
        assert python_cpar is not None

        from optv.transforms import convert_arr_metric_to_pixel as optv_func

        metric_coords = synthetic_2d_metric_coords.copy()

        optv_result = optv_func(metric_coords, optv_cpar)

        from algorithms.trafo import arr_metric_to_pixel as python_func

        python_result = python_func(metric_coords, python_cpar)

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_pixel_to_metric_single_point(self):
        """Test pixel_to_metric with a single point."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None
        assert python_cpar is not None

        from optv.transforms import convert_arr_pixel_to_metric as optv_func

        pixel = np.array([[512.0, 512.0]], dtype=np.float64)
        optv_result = optv_func(pixel, optv_cpar)

        from algorithms.trafo import pixel_to_metric as python_func

        x, y = python_func(pixel[0, 0], pixel[0, 1], python_cpar)
        python_result = np.array([[x, y]])

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_metric_to_pixel_single_point(self):
        """Test metric_to_pixel with a single point."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None
        assert python_cpar is not None

        from optv.transforms import convert_arr_metric_to_pixel as optv_func

        metric = np.array([[1.0, 2.0]], dtype=np.float64)
        optv_result = optv_func(metric, optv_cpar)

        from algorithms.trafo import metric_to_pixel as python_func

        x, y = python_func(metric[0, 0], metric[0, 1], python_cpar)
        python_result = np.array([[x, y]])

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_correct_brown_affine_synthetic(self, synthetic_2d_metric_coords):
        """Test correct_brown_affine (distortion correction)."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        coords = synthetic_2d_metric_coords.copy()

        from optv.transforms import correct_arr_brown_affine as optv_func

        optv_result = optv_func(coords, optv_cal)

        from algorithms.trafo import correct_arr_brown_affine as python_func

        python_result = python_func(coords, python_cal)

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_distort_brown_affine_synthetic(self, synthetic_2d_metric_coords):
        """Test distort_brown_affine (apply distortion)."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        coords = synthetic_2d_metric_coords.copy()

        from optv.transforms import distort_arr_brown_affine as optv_func

        optv_result = optv_func(coords, optv_cal)

        from algorithms.trafo import distort_arr_brown_affine as python_func

        python_result = python_func(coords, python_cal)

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_distorted_to_flat_synthetic(self, synthetic_2d_metric_coords):
        """Test distorted_to_flat conversion."""
        from optv.calibration import Calibration as OptvCal
        from algorithms.calibration import Calibration as PythonCal

        optv_cal = OptvCal()
        python_cal = PythonCal()

        coords = synthetic_2d_metric_coords.copy()

        from optv.transforms import distorted_to_flat as optv_func

        optv_result = optv_func(coords, optv_cal)

        from algorithms.trafo import distorted_to_flat as python_func

        python_result = python_func(coords, python_cal)

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_round_trip_pixel_metric(self):
        """Test pixel -> metric -> pixel round trip."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None
        assert python_cpar is not None

        original = np.array([[512.0, 384.0], [256.0, 768.0]], dtype=np.float64)

        from optv.transforms import (
            convert_arr_pixel_to_metric as optv_p2m,
            convert_arr_metric_to_pixel as optv_m2p,
        )
        from algorithms.trafo import (
            arr_pixel_to_metric as python_p2m,
            arr_metric_to_pixel as python_m2p,
        )

        optv_metric = optv_p2m(original, optv_cpar)
        optv_back = optv_m2p(optv_metric, optv_cpar)

        python_metric = python_p2m(
            original.astype(np.int32),
            python_cpar.imx,
            python_cpar.imy,
            python_cpar.pix_x,
            python_cpar.pix_y,
        )
        python_back = python_m2p(python_metric, python_cpar)

        np.testing.assert_allclose(
            optv_back, python_back, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_edge_case_zeros(self):
        """Test with zeros (image center in pixel coords)."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None
        assert python_cpar is not None

        zeros = np.array([[0.0, 0.0]], dtype=np.float64)

        from optv.transforms import convert_arr_pixel_to_metric as optv_func
        from algorithms.trafo import pixel_to_metric as python_func

        optv_result = optv_func(zeros, optv_cpar)
        x, y = python_func(zeros[0, 0], zeros[0, 1], python_cpar)
        python_result = np.array([[x, y]])

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_edge_case_corner(self):
        """Test with image corner coordinates."""
        optv_cpar, python_cpar = create_test_control_params()
        assert optv_cpar is not None
        assert python_cpar is not None

        corner = np.array([[1024.0, 1024.0]], dtype=np.float64)

        from optv.transforms import convert_arr_pixel_to_metric as optv_func
        from algorithms.trafo import pixel_to_metric as python_func

        optv_result = optv_func(corner, optv_cpar)
        x, y = python_func(corner[0, 0], corner[0, 1], python_cpar)
        python_result = np.array([[x, y]])

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )
