"""
Engine comparison tests for TargetArray class.

Tests TargetArray data structure and all methods.
Tolerance: 1e-10 (direct data comparison)
"""

import numpy as np
import pytest
from .conftest import get_tolerance, create_test_target, create_test_target_list

TOLERANCE = get_tolerance("target_array")


class TestTargetArray:
    """Compare TargetArray results between optv and python engines."""

    def test_target_array_creation_empty(self):
        """Test creating an empty TargetArray."""
        from optv.tracking_framebuf import TargetArray as OptvTA
        from algorithms.tracking_frame_buf import TargetArray as PythonTA

        optv_ta = OptvTA()
        python_ta = PythonTA()

        assert len(optv_ta) == len(python_ta)

    def test_target_array_creation_with_size(self):
        """Test creating a TargetArray with specified size."""
        from optv.tracking_framebuf import TargetArray as OptvTA
        from algorithms.tracking_frame_buf import TargetArray as PythonTA

        optv_ta = OptvTA(size=10)
        python_ta = PythonTA(size=10)

        assert len(optv_ta) == len(python_ta)
        assert len(optv_ta) == 10

    def test_target_array_sort_y(self):
        """Test sorting TargetArray by Y coordinate."""
        from optv.tracking_framebuf import TargetArray as OptvTA, Target as OptvT
        from algorithms.tracking_frame_buf import TargetArray as PythonTA

        size = 4
        optv_ta = OptvTA(size=size)
        python_ta = PythonTA(size=size)

        for i in range(size):
            t = optv_ta[i]
            t.set_pos((float(i), float(size - 1 - i)))
            python_ta[i].set_pos((float(i), float(size - 1 - i)))

        optv_ta.sort_y()
        python_ta.sort_y()

        for i in range(size):
            optv_pos = optv_ta[i].pos()
            python_pos = python_ta[i].pos()
            np.testing.assert_allclose(
                [optv_pos[0], optv_pos[1]],
                [python_pos[0], python_pos[1]],
                rtol=TOLERANCE,
                atol=TOLERANCE,
            )

    def test_target_array_len(self):
        """Test len() function on TargetArray."""
        from optv.tracking_framebuf import TargetArray as OptvTA
        from algorithms.tracking_frame_buf import TargetArray as PythonTA

        for size in [5, 10, 15]:
            optv_ta = OptvTA(size=size)
            python_ta = PythonTA(size=size)

            assert len(optv_ta) == len(python_ta)
            assert len(optv_ta) == size


class TestTargetArrayEdgeCases:
    """Test edge cases for TargetArray."""

    def test_target_array_single_element(self):
        """Test TargetArray with single element."""
        from optv.tracking_framebuf import TargetArray as OptvTA
        from algorithms.tracking_frame_buf import TargetArray as PythonTA

        optv_ta = OptvTA(size=1)
        python_ta = PythonTA(size=1)

        assert len(optv_ta) == 1
        assert len(python_ta) == 1

    def test_target_array_large_size(self):
        """Test TargetArray with large size."""
        from optv.tracking_framebuf import TargetArray as OptvTA
        from algorithms.tracking_frame_buf import TargetArray as PythonTA

        size = 1000
        optv_ta = OptvTA(size=size)
        python_ta = PythonTA(size=size)

        assert len(optv_ta) == size
        assert len(python_ta) == size
