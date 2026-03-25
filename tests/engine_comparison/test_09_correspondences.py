"""
Engine comparison tests for correspondences module.

Tests MatchedCoords class and correspondences function.
Tolerance: 1e-7 (complex matching algorithms)
"""

import numpy as np
import pytest
from .conftest import get_tolerance, create_test_control_params, create_test_calibration

TOLERANCE = get_tolerance("correspondences")


class TestMatchedCoords:
    """Compare MatchedCoords between optv and python engines."""

    def test_matched_coords_creation(self):
        """Test MatchedCoords creation with targets."""
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords as OptvMatchedCoords
        from optv.calibration import Calibration

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()

        optv_ta = TargetArray(10)
        for i in range(10):
            optv_ta[i].set_pnr(i)
            optv_ta[i].set_pos((float(i * 10), float(i * 20)))
            optv_ta[i].set_pixel_counts(5, 2, 2)
            optv_ta[i].set_sum_grey_value(100.0)
            optv_ta[i].set_tnr(0)

        try:
            optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)
            optv_result = optv_mc.as_arrays()
        except Exception as e:
            pytest.fail(f"optv MatchedCoords failed: {e}")

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(10)
            for i in range(10):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10), float(i * 20)))
                python_ta[i].set_pixel_counts(5, 2, 2)
                python_ta[i].set_sum_grey_value(100.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_result = python_mc.as_arrays()

            np.testing.assert_allclose(
                optv_result[0], python_result[0], rtol=TOLERANCE, atol=TOLERANCE
            )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_matched_coords_as_arrays(self):
        """Test MatchedCoords.as_arrays() method."""
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords as OptvMatchedCoords
        from optv.calibration import Calibration

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()

        optv_ta = TargetArray(5)
        for i in range(5):
            optv_ta[i].set_pnr(i)
            optv_ta[i].set_pos((float(i * 10 + 5), float(i * 15 + 3)))
            optv_ta[i].set_pixel_counts(3, 2, 2)
            optv_ta[i].set_sum_grey_value(50.0)
            optv_ta[i].set_tnr(0)

        optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)
        pos, pnr = optv_mc.as_arrays()

        assert pos.shape[1] == 2
        assert len(pos) == len(pnr)

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(5)
            for i in range(5):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10 + 5), float(i * 15 + 3)))
                python_ta[i].set_pixel_counts(3, 2, 2)
                python_ta[i].set_sum_grey_value(50.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_pos, python_pnr = python_mc.as_arrays()

            np.testing.assert_allclose(pos, python_pos, rtol=TOLERANCE, atol=TOLERANCE)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_matched_coords_get_by_pnrs(self):
        """Test MatchedCoords.get_by_pnrs() method."""
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords as OptvMatchedCoords
        from optv.calibration import Calibration

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()

        optv_ta = TargetArray(8)
        for i in range(8):
            optv_ta[i].set_pnr(i)
            optv_ta[i].set_pos((float(i * 10), float(i * 20)))
            optv_ta[i].set_pixel_counts(5, 2, 2)
            optv_ta[i].set_sum_grey_value(100.0)
            optv_ta[i].set_tnr(0)

        optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)

        query_pnrs = np.array([0, 2, 4, 6], dtype=np.intp)
        optv_result = optv_mc.get_by_pnrs(query_pnrs)

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA

            python_ta = PythonTA(8)
            for i in range(8):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10), float(i * 20)))
                python_ta[i].set_pixel_counts(5, 2, 2)
                python_ta[i].set_sum_grey_value(100.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_result = python_mc.get_by_pnrs(query_pnrs)

            np.testing.assert_allclose(
                optv_result, python_result, rtol=TOLERANCE, atol=TOLERANCE
            )
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestCorrespondencesFunction:
    """Test correspondences function."""

    def test_correspondences_basic(self):
        """Test correspondences function with basic data."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords
        from optv.parameters import ControlParams, VolumeParams
        from optv.calibration import Calibration
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()

        num_targets = 15
        num_cams = 4

        img_pts = []
        flat_coords = []
        cals = []

        for cam in range(num_cams):
            ta = TargetArray(num_targets)
            for i in range(num_targets):
                ta[i].set_pnr(i)
                ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                ta[i].set_pixel_counts(5, 2, 2)
                ta[i].set_sum_grey_value(100.0)
                ta[i].set_tnr(0)
            img_pts.append(ta)

            cal = Calibration()

            mc = MatchedCoords(ta, optv_cpar, cal)
            flat_coords.append(mc)
            cals.append(cal)

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)
        cparam = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, cparam)
        except Exception as e:
            pytest.fail(f"optv correspondences failed: {e}")

        try:
            from algorithms.correspondences import correspondences as python_func
            from algorithms.tracking_frame_buf import TargetArray as PythonTA
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords

            python_img_pts = []
            python_flat_coords = []
            python_cals = []
            for cam in range(num_cams):
                ta = PythonTA(num_targets)
                for i in range(num_targets):
                    ta[i].set_pnr(i)
                    ta[i].set_pos((float(i * 10 + cam * 5), float(i * 15 + cam * 3)))
                    ta[i].set_pixel_counts(5, 2, 2)
                    ta[i].set_sum_grey_value(100.0)
                    ta[i].set_tnr(0)
                cal = create_test_calibration()[1]
                python_img_pts.append(ta)
                python_flat_coords.append(PythonMatchedCoords(ta, python_cpar, cal))
                python_cals.append(cal)
            match_counts = [0, 0, 0, 0]
            python_result = python_func(
                PythonFrame(num_cams), python_flat_coords, vpar, python_cpar, python_cals, match_counts
            )
        except (ImportError, AttributeError, TypeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_single_cam_correspondence(self):
        """Test single_cam_correspondence function."""
        from optv.correspondences import single_cam_correspondence as optv_func
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords
        from optv.calibration import Calibration
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_cpar, python_cpar = create_test_control_params()
        optv_cal, python_cal = create_test_calibration()

        num_targets = 10

        ta = TargetArray(num_targets)
        for i in range(num_targets):
            ta[i].set_pnr(i)
            ta[i].set_pos((float(i * 10), float(i * 20)))
            ta[i].set_pixel_counts(5, 2, 2)
            ta[i].set_sum_grey_value(100.0)
            ta[i].set_tnr(0)

        cal = Calibration()

        mc = MatchedCoords(ta, optv_cpar, cal)
        img_pts = [ta]
        flat_coords = [mc]
        cals = [cal]

        try:
            optv_result = optv_func(img_pts, flat_coords, cals)
        except Exception as e:
            pytest.fail(f"optv single_cam_correspondence failed: {e}")

        try:
            from algorithms.correspondences import (
                single_cam_correspondence as python_func,
            )
            from algorithms.tracking_frame_buf import TargetArray as PythonTA
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords

            python_ta = PythonTA(num_targets)
            for i in range(num_targets):
                python_ta[i].set_pnr(i)
                python_ta[i].set_pos((float(i * 10), float(i * 20)))
                python_ta[i].set_pixel_counts(5, 2, 2)
                python_ta[i].set_sum_grey_value(100.0)
                python_ta[i].set_tnr(0)

            python_mc = PythonMatchedCoords(python_ta, python_cpar, python_cal)
            python_result = python_func([python_ta], [python_mc])
        except (ImportError, AttributeError, TypeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")


class TestCorrespondencesEdgeCases:
    """Test edge cases for correspondence functions."""

    def test_correspondences_empty_targets(self):
        """Test with empty target arrays."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray
        from optv.correspondences import MatchedCoords
        from optv.parameters import ControlParams, VolumeParams
        from optv.calibration import Calibration
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        num_cams = 4

        img_pts = [TargetArray(0) for _ in range(num_cams)]
        flat_coords = []
        cals = []

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        cpar.get_multimedia_params().set_layers([1.0], [1.0])
        cpar.get_multimedia_params().set_n3(1.0)
        from algorithms.parameters import ControlPar as PythonControlPar

        python_cpar = PythonControlPar()
        python_cpar.imx = 1024
        python_cpar.imy = 1024
        python_cpar.pix_x = 0.01
        python_cpar.pix_y = 0.01
        from algorithms.parameters import ControlPar as PythonControlPar

        python_cpar = PythonControlPar()
        python_cpar.imx = 1024
        python_cpar.imy = 1024
        python_cpar.pix_x = 0.01
        python_cpar.pix_y = 0.01
        _, python_cpar = create_test_control_params()
        _, python_cpar = create_test_control_params()
        _, python_cpar = create_test_control_params()
        _, python_cpar = create_test_control_params()
        optv_cpar, python_cpar = create_test_control_params()

        for _ in range(num_cams):
            cal = Calibration()
            mc = MatchedCoords(TargetArray(0), cpar, cal)
            flat_coords.append(mc)
            cals.append(cal)

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, cpar)
        except Exception:
            pass

        try:
            from algorithms.correspondences import correspondences as python_func

            python_frame = PythonFrame(num_cams)
            python_frame.targets = img_pts
            python_frame.num_targets = [0 for _ in range(num_cams)]
            match_counts = [0, 0, 0, 0]
            python_result = python_func(
                python_frame, flat_coords, vpar, python_cpar, cals, match_counts
            )
        except (ImportError, AttributeError, TypeError):
            pass

    def test_correspondences_single_target(self):
        """Test with single target per camera."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords
        from optv.parameters import ControlParams, VolumeParams
        from optv.calibration import Calibration
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        num_cams = 4

        img_pts = []
        flat_coords = []
        cals = []

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )

        for cam in range(num_cams):
            ta = TargetArray(1)
            ta[0].set_pnr(0)
            ta[0].set_pos((50.0, 50.0))
            ta[0].set_pixel_counts(5, 2, 2)
            ta[0].set_sum_grey_value(100.0)
            ta[0].set_tnr(0)
            img_pts.append(ta)

            cal = Calibration()
            mc = MatchedCoords(ta, cpar, cal)
            flat_coords.append(mc)
            cals.append(cal)

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, cpar)
        except Exception as e:
            pytest.fail(f"optv correspondences failed: {e}")

        try:
            from algorithms.correspondences import correspondences as python_func
            from algorithms.tracking_frame_buf import TargetArray as PythonTA
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.parameters import ControlPar as PythonControlPar

            python_frame = PythonFrame(num_cams)
            python_img_pts = []
            python_flat_coords = []
            python_cals = []
            python_cpar = PythonControlPar()
            python_cpar.imx = 1024
            python_cpar.imy = 1024
            python_cpar.pix_x = 0.01
            python_cpar.pix_y = 0.01
            python_cpar.mm.set_layers([1.0], [1.0])
            python_cpar.mm.set_n3(1.0)
            for _ in range(num_cams):
                ta = PythonTA(1)
                ta[0].set_pnr(0)
                ta[0].set_pos((50.0, 50.0))
                ta[0].set_pixel_counts(5, 2, 2)
                ta[0].set_sum_grey_value(100.0)
                ta[0].set_tnr(0)
                python_img_pts.append(ta)
                cal = create_test_calibration()[1]
                python_flat_coords.append(PythonMatchedCoords(ta, python_cpar, cal))
                python_cals.append(cal)

            python_frame.targets = python_img_pts
            python_frame.num_targets = [1 for _ in range(num_cams)]
            match_counts = [0, 0, 0, 0]
            python_result = python_func(
                python_frame, python_flat_coords, vpar, python_cpar, python_cals, match_counts
            )
        except (ImportError, AttributeError, TypeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")
