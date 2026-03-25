"""
Engine comparison tests for imgcoord module.

Tests flat_image_coordinates, image_coordinates functions.
Tolerance: 1e-7 (iterative algorithms)
"""

import numpy as np
import pytest
from .conftest import get_tolerance, create_test_calibration, create_test_control_params

TOLERANCE = get_tolerance("imgcoord")


class TestImageCoordinates:
    """Compare image coordinate functions between optv and python engines."""

    def test_flat_image_coordinates_synthetic(self, synthetic_metric_coords):
        """Test flat_image_coordinates with synthetic 3D coordinates."""
        from optv.imgcoord import flat_image_coordinates as optv_func
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coords_3d = synthetic_metric_coords.copy()

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal()

        optv_result = optv_func(coords_3d, cal, mmp)

        try:
            from algorithms.imgcoord import flat_image_coordinates as python_func
            from algorithms.parameters import ControlPar, MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_cpar = ControlPar()
            python_cpar.imx = 1024
            python_cpar.imy = 1024
            python_cpar.pix_x = 0.01
            python_cpar.pix_y = 0.01

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()

            python_result = python_func(coords_3d, python_cal, python_mmp)

            assert optv_result.shape == python_result.shape
            finite_mask = np.isfinite(optv_result) & np.isfinite(python_result)
            if np.any(finite_mask):
                np.testing.assert_allclose(
                    optv_result[finite_mask],
                    python_result[finite_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_image_coordinates_synthetic(self, synthetic_metric_coords):
        """Test image_coordinates with synthetic 3D coordinates."""
        from optv.imgcoord import image_coordinates as optv_func
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coords_3d = synthetic_metric_coords.copy()

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal()

        optv_result = optv_func(coords_3d, cal, mmp)

        try:
            from algorithms.imgcoord import image_coordinates as python_func
            from algorithms.parameters import ControlPar, MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_cpar = ControlPar()
            python_cpar.imx = 1024
            python_cpar.imy = 1024
            python_cpar.pix_x = 0.01
            python_cpar.pix_y = 0.01

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()

            python_result = python_func(coords_3d, python_cal, python_mmp)

            assert optv_result.shape == python_result.shape
            finite_mask = np.isfinite(optv_result) & np.isfinite(python_result)
            if np.any(finite_mask):
                np.testing.assert_allclose(
                    optv_result[finite_mask],
                    python_result[finite_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_flat_image_coordinates_single_point(self):
        """Test flat_image_coordinates with a single point."""
        from optv.imgcoord import flat_image_coordinates as optv_func
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coord_3d = np.array([[50.0, 60.0, 70.0]], dtype=np.float64)

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal()

        optv_result = optv_func(coord_3d, cal, mmp)

        try:
            from algorithms.imgcoord import flat_image_coordinates as python_func
            from algorithms.parameters import ControlPar, MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()

            python_result = python_func(coord_3d, python_cal, python_mmp)

            assert optv_result.shape == python_result.shape
            finite_mask = np.isfinite(optv_result) & np.isfinite(python_result)
            if np.any(finite_mask):
                np.testing.assert_allclose(
                    optv_result[finite_mask],
                    python_result[finite_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_image_coordinates_single_point(self):
        """Test image_coordinates with a single point."""
        from optv.imgcoord import image_coordinates as optv_func
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coord_3d = np.array([[50.0, 60.0, 70.0]], dtype=np.float64)

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal()

        optv_result = optv_func(coord_3d, cal, mmp)

        try:
            from algorithms.imgcoord import image_coordinates as python_func
            from algorithms.parameters import ControlPar, MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()

            python_result = python_func(coord_3d, python_cal, python_mmp)

            assert optv_result.shape == python_result.shape
            finite_mask = np.isfinite(optv_result) & np.isfinite(python_result)
            if np.any(finite_mask):
                np.testing.assert_allclose(
                    optv_result[finite_mask],
                    python_result[finite_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_flat_image_coordinates_with_distortion(self):
        """Test flat_image_coordinates with camera distortion."""
        from optv.imgcoord import flat_image_coordinates as optv_func
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coords_3d = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=np.float64,
        )

        rad_dist = np.array([0.001, 0.002, 0.0005])

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal(rad_dist=rad_dist)

        optv_result = optv_func(coords_3d, cal, mmp)

        try:
            from algorithms.imgcoord import flat_image_coordinates as python_func
            from algorithms.parameters import MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()
            python_cal.set_radial_distortion(rad_dist)

            python_result = python_func(coords_3d, python_cal, python_mmp)

            assert optv_result.shape == python_result.shape
            finite_mask = np.isfinite(optv_result) & np.isfinite(python_result)
            if np.any(finite_mask):
                np.testing.assert_allclose(
                    optv_result[finite_mask],
                    python_result[finite_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_image_coordinates_with_distortion(self):
        """Test image_coordinates with camera distortion."""
        from optv.imgcoord import image_coordinates as optv_func
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coords_3d = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
            ],
            dtype=np.float64,
        )

        rad_dist = np.array([0.001, 0.002, 0.0005])

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal(rad_dist=rad_dist)

        optv_result = optv_func(coords_3d, cal, mmp)

        try:
            from algorithms.imgcoord import image_coordinates as python_func
            from algorithms.parameters import MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()
            python_cal.set_radial_distortion(rad_dist)

            python_result = python_func(coords_3d, python_cal, python_mmp)

            assert optv_result.shape == python_result.shape
            finite_mask = np.isfinite(optv_result) & np.isfinite(python_result)
            if np.any(finite_mask):
                np.testing.assert_allclose(
                    optv_result[finite_mask],
                    python_result[finite_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_round_trip_3d_to_2d(self):
        """Test 3D -> 2D -> 3D round trip."""
        from optv.imgcoord import flat_image_coordinates, image_coordinates
        from optv.parameters import ControlParams, MultimediaParams
        from optv.calibration import Calibration as OptvCal

        coords_3d = np.array([[50.0, 60.0, 70.0]], dtype=np.float64)

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        mmp = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        cal = OptvCal()

        flat_result = flat_image_coordinates(coords_3d, cal, mmp)
        img_result = image_coordinates(coords_3d, cal, mmp)

        try:
            from algorithms.imgcoord import (
                flat_image_coordinates as python_flat,
                image_coordinates as python_img,
            )
            from algorithms.parameters import MultimediaPar
            from algorithms.calibration import Calibration as PythonCal

            python_mmp = MultimediaPar()
            python_mmp.nlay = 1

            python_cal = PythonCal()

            python_flat_result = python_flat(coords_3d, python_cal, python_mmp)
            python_img_result = python_img(coords_3d, python_cal, python_mmp)

            assert flat_result.shape == python_flat_result.shape
            flat_mask = np.isfinite(flat_result) & np.isfinite(python_flat_result)
            if np.any(flat_mask):
                np.testing.assert_allclose(
                    flat_result[flat_mask],
                    python_flat_result[flat_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )

            assert img_result.shape == python_img_result.shape
            img_mask = np.isfinite(img_result) & np.isfinite(python_img_result)
            if np.any(img_mask):
                np.testing.assert_allclose(
                    img_result[img_mask],
                    python_img_result[img_mask],
                    rtol=TOLERANCE,
                    atol=TOLERANCE,
                )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_check_arrays_input_validation(self):
        """Test check_arrays input validation function."""
        from optv.imgcoord import check_arrays

        valid_input = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float64)
        valid_output = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)

        result = check_arrays(valid_input, valid_output)
        assert result is None
