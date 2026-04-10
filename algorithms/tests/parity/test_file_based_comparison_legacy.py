"""
File-based engine comparison tests.

Both engines read the SAME source files through their own readers:
- optv engine: Cython/C file readers
- python engine: Python file readers

This tests TWO things simultaneously:
1. Reader parity — same files are parsed identically by both engines
2. Algorithm parity — same inputs produce same outputs (where both engines implement the algorithm)

If a test fails, it could mean:
- The algorithms diverge (different computation)
- The readers diverge (different file parsing)
- Both

Usage:
    pytest tests/test_file_based_comparison.py -v
"""

import numpy as np
import pytest

from ..conftest import get_tolerance

TOLERANCE = get_tolerance("parameters")


class TestFileBasedParameterParity:
    """Verify both engines read the same parameter files identically."""

    def test_control_params_from_file(self, file_control_params):
        """Both engines read the same control.par file."""
        optv_cpar, python_cpar = file_control_params
        assert optv_cpar is not None
        assert python_cpar is not None

        assert optv_cpar.get_num_cams() == python_cpar.num_cams
        assert optv_cpar.get_image_size() == (python_cpar.imx, python_cpar.imy)
        np.testing.assert_allclose(
            optv_cpar.get_pixel_size(),
            (python_cpar.pix_x, python_cpar.pix_y),
            rtol=TOLERANCE,
        )
        assert optv_cpar.get_chfield() == python_cpar.chfield

    def test_volume_params_from_file(self, file_volume_params):
        """Both engines read the same volume.par file."""
        optv_vpar, python_vpar = file_volume_params
        assert optv_vpar is not None
        assert python_vpar is not None

        x_lay = list(optv_vpar.get_X_lay())
        z_min = list(optv_vpar.get_Zmin_lay())
        z_max = list(optv_vpar.get_Zmax_lay())

        np.testing.assert_allclose(x_lay, python_vpar.x_lay, rtol=TOLERANCE)
        np.testing.assert_allclose(z_min, python_vpar.z_min_lay, rtol=TOLERANCE)
        np.testing.assert_allclose(z_max, python_vpar.z_max_lay, rtol=TOLERANCE)

    def test_calibration_exterior_from_file(self, file_calibration_cam1):
        """Both engines read the same calibration .ori file — exterior params."""
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cal is not None
        assert python_cal is not None

        optv_pos = optv_cal.get_pos()
        python_pos = python_cal.get_pos()
        np.testing.assert_allclose(optv_pos, python_pos, rtol=TOLERANCE)

        optv_angles = optv_cal.get_angles()
        python_angles = python_cal.get_angles()
        np.testing.assert_allclose(optv_angles, python_angles, rtol=TOLERANCE)

    def test_calibration_interior_from_file(self, file_calibration_cam1):
        """Both engines read the same calibration .ori file — interior params."""
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cal is not None
        assert python_cal is not None

        optv_pp = optv_cal.get_primary_point()
        python_pp = python_cal.get_primary_point()
        np.testing.assert_allclose(optv_pp, python_pp, rtol=TOLERANCE)

    def test_calibration_rotation_from_file(self, file_calibration_cam1):
        """Both engines read the same calibration .ori file — rotation matrix."""
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cal is not None
        assert python_cal is not None

        optv_dm = optv_cal.get_rotation_matrix()
        python_dm = python_cal.get_rotation_matrix()
        np.testing.assert_allclose(optv_dm, python_dm, rtol=TOLERANCE)

    def test_calibration_distortion_from_file(self, file_calibration_cam1):
        """Both engines read the same calibration .addpar file — distortion params."""
        optv_cal, python_cal = file_calibration_cam1
        assert optv_cal is not None
        assert python_cal is not None

        optv_rd = optv_cal.get_radial_distortion()
        python_rd = python_cal.get_radial_distortion()
        np.testing.assert_allclose(optv_rd, python_rd, rtol=TOLERANCE)

    def test_calibration_4cam_from_files(self, file_calibration_4cam):
        """Both engines read all 4 camera calibration files identically."""
        optv_cals, python_cals = file_calibration_4cam
        for i, (o, p) in enumerate(zip(optv_cals, python_cals)):
            assert o is not None, f"optv cam{i + 1} calibration failed to load"
            assert p is not None, f"python cam{i + 1} calibration failed to load"

            np.testing.assert_allclose(
                o.get_pos(),
                p.get_pos(),
                rtol=TOLERANCE,
                err_msg=f"cam{i + 1} position mismatch",
            )
            np.testing.assert_allclose(
                o.get_angles(),
                p.get_angles(),
                rtol=TOLERANCE,
                err_msg=f"cam{i + 1} angles mismatch",
            )
            np.testing.assert_allclose(
                o.get_rotation_matrix(),
                p.get_rotation_matrix(),
                rtol=TOLERANCE,
                err_msg=f"cam{i + 1} rotation mismatch",
            )


class TestFileBasedAlgorithmParity:
    """Verify algorithms produce identical results when fed file-based params."""

    def test_pixel_to_metric_from_file_params(
        self, file_control_params, synthetic_pixel_coords
    ):
        """pixel_to_metric with parameters read from the same file."""
        from algorithms.trafo import arr_pixel_to_metric as python_func

        optv_cpar, python_cpar = file_control_params
        assert optv_cpar is not None
        assert python_cpar is not None

        imx, imy = optv_cpar.get_image_size()
        pix_x, pix_y = optv_cpar.get_pixel_size()

        # Verify both engines read the same values from the file
        assert imx == python_cpar.imx
        assert imy == python_cpar.imy
        np.testing.assert_allclose(pix_x, python_cpar.pix_x, rtol=TOLERANCE)
        np.testing.assert_allclose(pix_y, python_cpar.pix_y, rtol=TOLERANCE)

        # arr_pixel_to_metric requires int32 input array (numba signature)
        pixel_int = synthetic_pixel_coords.astype(np.int32)
        python_result = python_func(
            pixel_int,
            int(python_cpar.imx),
            int(python_cpar.imy),
            float(python_cpar.pix_x),
            float(python_cpar.pix_y),
        )
        assert python_result.shape == pixel_int.shape

    def test_flat_image_coordinates_from_file(
        self, file_calibration_cam1, synthetic_metric_coords
    ):
        """flat_image_coordinates with calibration read from the same file."""
        from optv.imgcoord import flat_image_coordinates as optv_func
        from algorithms.imgcoord import flat_image_coordinates as python_func
        from optv.parameters import MultimediaParams
        from algorithms.parameters import MultimediaPar

        optv_cal, python_cal = file_calibration_cam1
        assert optv_cal is not None
        assert python_cal is not None

        # Same trivial multimedia params for both
        optv_mm = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)
        python_mm = MultimediaPar(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)

        optv_result = optv_func(synthetic_metric_coords, optv_cal, optv_mm)
        python_result = python_func(synthetic_metric_coords, python_cal, python_mm)

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_image_coordinates_from_file(
        self, file_calibration_cam1, synthetic_metric_coords
    ):
        """image_coordinates with calibration read from the same file."""
        from optv.imgcoord import image_coordinates as optv_func
        from algorithms.imgcoord import image_coordinates as python_func
        from optv.parameters import ControlParams, MultimediaParams
        from algorithms.parameters import ControlPar, MultimediaPar

        optv_cal, python_cal = file_calibration_cam1
        assert optv_cal is not None
        assert python_cal is not None

        optv_cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        optv_mm = MultimediaParams(n1=1.0, n2=None, n3=1.0, d=None)

        python_cpar = ControlPar()
        python_cpar.imx = 1024
        python_cpar.imy = 1024
        python_cpar.pix_x = 0.01
        python_cpar.pix_y = 0.01
        python_mm = MultimediaPar(n1=1.0, n2=[1.0], d=[0.0], n3=1.0)

        optv_result = optv_func(synthetic_metric_coords, optv_cal, optv_mm)
        python_result = python_func(synthetic_metric_coords, python_cal, python_mm)

        np.testing.assert_allclose(
            optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
        )

    def test_target_recognition_from_file_params(
        self, file_control_params, file_target_params
    ):
        """Target recognition with parameters read from the same files."""
        from optv.segmentation import target_recognition as optv_func
        from algorithms.segmentation import target_recognition as python_func

        optv_cpar, python_cpar = file_control_params
        optv_tpar, python_tpar = file_target_params
        assert optv_cpar is not None
        assert python_cpar is not None
        assert optv_tpar is not None
        assert python_tpar is not None

        # Image size must match what's in the control.par file
        imx, imy = optv_cpar.get_image_size()
        img = np.zeros((imy, imx), dtype=np.uint8)
        # Add targets well within the image bounds
        img[300:305, 400:405] = 200
        img[600:605, 700:705] = 180

        # Both engines use the same image and file-read parameters
        optv_result = optv_func(img, optv_tpar, 0, optv_cpar)
        python_result = python_func(img, python_tpar, 0, python_cpar)

        # Compare number of targets found
        assert len(optv_result) == len(python_result), (
            f"Target count mismatch: optv={len(optv_result)}, python={len(python_result)}"
        )
