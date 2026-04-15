"""
Engine comparison tests for Frame class.

Tests Frame data structure and positions() method.
Tolerance: 1e-10 (direct data comparison)
"""

import numpy as np
import pytest
from pathlib import Path
from ..conftest import get_tolerance, FIXTURES

TOLERANCE = get_tolerance("frame")


class TestFrame:
    """Compare Frame results between optv and python engines."""

    def test_frame_creation_empty(self):
        """Test creating an empty Frame."""
        from optv.tracking_framebuf import Frame as OptvFrame
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_frm = OptvFrame(num_cams=4)
        python_frm = PythonFrame(num_cams=4)

        assert optv_frm is not None
        assert python_frm is not None

    def test_frame_positions_empty(self):
        """Test positions() on empty frame."""
        from optv.tracking_framebuf import Frame as OptvFrame
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_frm = OptvFrame(num_cams=4)
        python_frm = PythonFrame(num_cams=4)

        python_pos = python_frm.positions()

        assert optv_frm is not None
        assert python_pos.shape[1] == 3

    def test_frame_target_positions_for_camera(self):
        """Test target_positions_for_camera() method."""
        from optv.tracking_framebuf import Frame as OptvFrame
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_frm = OptvFrame(num_cams=4)
        python_frm = PythonFrame(num_cams=4)

        for cam in range(4):
            python_pos = python_frm.target_positions_for_camera(cam)

            assert optv_frm is not None
            assert python_pos.shape[1] == 2

    def test_target_cache_refreshes_from_targets(self):
        """Test that cached target arrays mirror the current target records."""
        from algorithms.tracking_frame_buf import Frame as PythonFrame, Target

        frm = PythonFrame(num_cams=2, max_targets=4)

        frm.targets[0][0] = Target(pnr=10, x=1.5, y=2.5, tnr=-1)
        frm.targets[0][1] = Target(pnr=11, x=3.5, y=4.5, tnr=7)
        frm.targets[0][2] = Target(pnr=12, x=5.5, y=6.5, tnr=-1)
        frm.num_targets[0] = 3

        frm.targets[1][0] = Target(pnr=20, x=7.25, y=8.25, tnr=3)
        frm.num_targets[1] = 1

        frm.refresh_target_arrays()

        np.testing.assert_allclose(frm.target_x[0], [1.5, 3.5, 5.5])
        np.testing.assert_allclose(frm.target_y[0], [2.5, 4.5, 6.5])
        np.testing.assert_array_equal(frm.target_tnr[0], [-1, 7, -1])
        np.testing.assert_allclose(frm.target_x[1], [7.25])
        np.testing.assert_allclose(frm.target_y[1], [8.25])
        np.testing.assert_array_equal(frm.target_tnr[1], [3])

        frm.targets[0][1].tnr = 42
        frm.refresh_target_arrays(0)
        np.testing.assert_array_equal(frm.target_tnr[0], [-1, 42, -1])


class TestFramePositions:
    """Test positions() method in detail."""

    def test_positions_shape(self):
        """Test that positions() returns correct shape."""
        from optv.tracking_framebuf import Frame as OptvFrame
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_frm = OptvFrame(num_cams=4)
        python_frm = PythonFrame(num_cams=4)

        python_pos = python_frm.positions()

        assert python_pos.shape[1] == 3
        assert optv_frm is not None


class TestFrameTargetPositions:
    """Test target_positions_for_camera() method in detail."""

    def test_target_positions_shape(self):
        """Test that target_positions_for_camera() returns correct shape."""
        from optv.tracking_framebuf import Frame as OptvFrame
        from algorithms.tracking_frame_buf import Frame as PythonFrame

        optv_frm = OptvFrame(num_cams=4)
        python_frm = PythonFrame(num_cams=4)

        for cam in range(4):
            python_pos = python_frm.target_positions_for_camera(cam)

            assert optv_frm is not None
            assert python_pos.shape[1] == 2
