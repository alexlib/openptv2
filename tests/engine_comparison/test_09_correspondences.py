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
        from optv.parameters import ControlParams
        from optv.calibration import Calibration

        optv_cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        optv_cal = Calibration()

        optv_ta = TargetArray()
        for i in range(10):
            t = Target(
                pnr=i,
                x=float(i * 10),
                y=float(i * 20),
                n=5,
                nx=2,
                ny=2,
                sumg=100.0,
                tnr=0,
            )
            optv_ta.append(t)

        try:
            optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)
            optv_result = optv_mc.as_arrays()
        except Exception as e:
            pytest.fail(f"optv MatchedCoords failed: {e}")

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords
            from algorithms.tracking_frame_buf import TargetArray as PythonTA
            from algorithms.tracking_frame_buf import Target as PythonTarget

            python_ta = PythonTA()
            for i in range(10):
                t = PythonTarget(
                    pnr=i,
                    x=float(i * 10),
                    y=float(i * 20),
                    n=5,
                    nx=2,
                    ny=2,
                    sumg=100.0,
                    tnr=0,
                )
                python_ta.append(t)

            python_mc = PythonMatchedCoords(python_ta, optv_cpar, optv_cal)
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
        from optv.parameters import ControlParams
        from optv.calibration import Calibration

        optv_cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        optv_cal = Calibration()

        optv_ta = TargetArray()
        for i in range(5):
            t = Target(
                pnr=i,
                x=float(i * 10 + 5),
                y=float(i * 15 + 3),
                n=3,
                nx=2,
                ny=2,
                sumg=50.0,
                tnr=0,
            )
            optv_ta.append(t)

        optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)
        pos, pnr = optv_mc.as_arrays()

        assert pos.shape[1] == 2
        assert len(pos) == len(pnr)

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords

            python_mc = PythonMatchedCoords(optv_ta, optv_cpar, optv_cal)
            python_pos, python_pnr = python_mc.as_arrays()

            np.testing.assert_allclose(pos, python_pos, rtol=TOLERANCE, atol=TOLERANCE)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_matched_coords_get_by_pnrs(self):
        """Test MatchedCoords.get_by_pnrs() method."""
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords as OptvMatchedCoords
        from optv.parameters import ControlParams
        from optv.calibration import Calibration

        optv_cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        optv_cal = Calibration()

        optv_ta = TargetArray()
        for i in range(8):
            t = Target(
                pnr=i,
                x=float(i * 10),
                y=float(i * 20),
                n=5,
                nx=2,
                ny=2,
                sumg=100.0,
                tnr=0,
            )
            optv_ta.append(t)

        optv_mc = OptvMatchedCoords(optv_ta, optv_cpar, optv_cal)

        query_pnrs = np.array([0, 2, 4, 6], dtype=np.intp)
        optv_result = optv_mc.get_by_pnrs(query_pnrs)

        try:
            from algorithms.correspondences import MatchedCoords as PythonMatchedCoords

            python_mc = PythonMatchedCoords(optv_ta, optv_cpar, optv_cal)
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

        num_targets = 15
        num_cams = 4

        img_pts = []
        flat_coords = []
        cals = []

        for cam in range(num_cams):
            ta = TargetArray()
            for i in range(num_targets):
                t = Target(
                    pnr=i,
                    x=float(i * 10 + cam * 5),
                    y=float(i * 15 + cam * 3),
                    n=5,
                    nx=2,
                    ny=2,
                    sumg=100.0,
                    tnr=0,
                )
                ta.append(t)
            img_pts.append(ta)

            cpar = ControlParams(
                num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
            )
            cal = Calibration()

            mc = MatchedCoords(ta, cpar, cal)
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

            python_result = python_func(img_pts, flat_coords, cals, vpar, cparam)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")

    def test_single_cam_correspondence(self):
        """Test single_cam_correspondence function."""
        from optv.correspondences import single_cam_correspondence as optv_func
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords
        from optv.parameters import ControlParams
        from optv.calibration import Calibration

        num_targets = 10

        ta = TargetArray()
        for i in range(num_targets):
            t = Target(
                pnr=i,
                x=float(i * 10),
                y=float(i * 20),
                n=5,
                nx=2,
                ny=2,
                sumg=100.0,
                tnr=0,
            )
            ta.append(t)

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )
        cal = Calibration()

        mc = MatchedCoords(ta, cpar, cal)
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

            python_result = python_func(img_pts, flat_coords, cals)
        except (ImportError, AttributeError) as e:
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

        num_cams = 4

        img_pts = [TargetArray() for _ in range(num_cams)]
        flat_coords = []
        cals = []

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )

        for _ in range(num_cams):
            cal = Calibration()
            mc = MatchedCoords(TargetArray(), cpar, cal)
            flat_coords.append(mc)
            cals.append(cal)

        vpar = VolumeParams(xmin=0, xmax=100, ymin=0, ymax=100, zmin=0, zmax=50)

        try:
            optv_result = optv_func(img_pts, flat_coords, cals, vpar, cpar)
        except Exception:
            pass

        try:
            from algorithms.correspondences import correspondences as python_func

            python_result = python_func(img_pts, flat_coords, cals, vpar, cpar)
        except (ImportError, AttributeError):
            pass

    def test_correspondences_single_target(self):
        """Test with single target per camera."""
        from optv.correspondences import correspondences as optv_func
        from optv.tracking_framebuf import TargetArray, Target
        from optv.correspondences import MatchedCoords
        from optv.parameters import ControlParams, VolumeParams
        from optv.calibration import Calibration

        num_cams = 4

        img_pts = []
        flat_coords = []
        cals = []

        cpar = ControlParams(
            num_cams=4, image_size=(1024, 1024), pixel_size=(0.01, 0.01)
        )

        for cam in range(num_cams):
            ta = TargetArray()
            t = Target(pnr=0, x=50.0, y=50.0, n=5, nx=2, ny=2, sumg=100.0, tnr=0)
            ta.append(t)
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

            python_result = python_func(img_pts, flat_coords, cals, vpar, cpar)
        except (ImportError, AttributeError) as e:
            pytest.fail(f"Python implementation missing or incomplete: {e}")
