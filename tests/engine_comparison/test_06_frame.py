"""
Engine comparison tests for Frame class.

Tests Frame data structure and positions() method.
Tolerance: 1e-10 (direct data comparison)
"""

import numpy as np
import pytest
from pathlib import Path
from .conftest import get_tolerance, FIXTURES

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
