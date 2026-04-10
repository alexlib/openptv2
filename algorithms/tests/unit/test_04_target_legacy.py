"""
Engine comparison tests for Target class.

Tests Target data structure and all get/set methods.
Tolerance: 1e-10 (direct data comparison)
"""

import numpy as np
import pytest
from ..conftest import get_tolerance, create_test_target

TOLERANCE = get_tolerance("target")


class TestTarget:
    """Compare Target results between optv and python engines."""

    def test_target_creation_with_kwargs(self):
        """Test Target creation with keyword arguments."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        target_data = create_test_target(
            pnr=5, x=100.5, y=200.3, n=10, nx=3, ny=3, sumg=500.0, tnr=2
        )

        optv_t = OptvTarget(**target_data)
        python_t = PythonTarget(**target_data)

        assert optv_t.pnr() == python_t.pnr
        assert optv_t.tnr() == python_t.tnr

        optv_pos = optv_t.pos()
        python_pos = python_t.pos()
        assert abs(optv_pos[0] - python_pos[0]) < TOLERANCE
        assert abs(optv_pos[1] - python_pos[1]) < TOLERANCE

    def test_pnr_get_set(self):
        """Test pnr (particle number) get and set methods."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        optv_t = OptvTarget(pnr=0, x=0, y=0, n=0, nx=0, ny=0, sumg=0, tnr=0)
        python_t = PythonTarget()

        test_pnr = 42
        optv_t.set_pnr(test_pnr)
        python_t.set_pnr(test_pnr)

        assert optv_t.pnr() == python_t.pnr
        assert optv_t.pnr() == test_pnr

    def test_tnr_get_set(self):
        """Test tnr (track number) get and set methods."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        optv_t = OptvTarget(pnr=0, x=0, y=0, n=0, nx=0, ny=0, sumg=0, tnr=0)
        python_t = PythonTarget()

        test_tnr = 7
        optv_t.set_tnr(test_tnr)
        python_t.set_tnr(test_tnr)

        assert optv_t.tnr() == python_t.tnr
        assert optv_t.tnr() == test_tnr

    def test_pos_get_set(self):
        """Test position get and set methods."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        optv_t = OptvTarget(pnr=0, x=0, y=0, n=0, nx=0, ny=0, sumg=0, tnr=0)
        python_t = PythonTarget()

        test_pos = (150.5, 250.75)
        optv_t.set_pos(test_pos)
        python_t.set_pos(test_pos)

        optv_result = optv_t.pos()
        python_result = python_t.pos()

        assert abs(optv_result[0] - python_result[0]) < TOLERANCE
        assert abs(optv_result[1] - python_result[1]) < TOLERANCE
        assert abs(optv_result[0] - test_pos[0]) < TOLERANCE
        assert abs(optv_result[1] - test_pos[1]) < TOLERANCE

    def test_count_pixels_get_set(self):
        """Test count_pixels get and set methods."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        optv_t = OptvTarget(pnr=0, x=0, y=0, n=0, nx=0, ny=0, sumg=0, tnr=0)
        python_t = PythonTarget()

        test_n, test_nx, test_ny = 25, 5, 5
        optv_t.set_pixel_counts(test_n, test_nx, test_ny)
        python_t.set_pixel_counts(test_n, test_nx, test_ny)

        optv_result = optv_t.count_pixels()
        python_result = python_t.count_pixels()

        assert optv_result[0] == python_result[0]
        assert optv_result[1] == python_result[1]
        assert optv_result[2] == python_result[2]

    def test_sum_grey_value_get_set(self):
        """Test sum_grey_value get and set methods."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        optv_t = OptvTarget(pnr=0, x=0, y=0, n=0, nx=0, ny=0, sumg=0, tnr=0)
        python_t = PythonTarget()

        test_sumg = 1234
        optv_t.set_sum_grey_value(test_sumg)
        python_t.set_sum_grey_value(float(test_sumg))

        assert abs(optv_t.sum_grey_value() - python_t.sum_grey_value()) < TOLERANCE

    def test_target_default_creation(self):
        """Test Target creation with default (empty) constructor."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        target_data = create_test_target(
            pnr=0, x=0.0, y=0.0, n=0, nx=0, ny=0, sumg=0.0, tnr=0
        )

        optv_t = OptvTarget(**target_data)
        python_t = PythonTarget(**target_data)

        assert optv_t.pnr() == python_t.pnr

    def test_target_multiple_targets(self):
        """Test creating and comparing multiple targets."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        for i in range(10):
            target_data = create_test_target(
                pnr=i,
                x=float(i * 10),
                y=float(i * 20),
                n=i + 1,
                nx=(i % 3) + 1,
                ny=(i % 3) + 1,
                sumg=float(i * 100),
                tnr=i % 2,
            )

            optv_t = OptvTarget(**target_data)
            python_t = PythonTarget(**target_data)

            assert optv_t.pnr() == python_t.pnr
            assert optv_t.tnr() == python_t.tnr

            optv_pos = optv_t.pos()
            python_pos = python_t.pos()
            np.testing.assert_allclose(
                [optv_pos[0], optv_pos[1]],
                [python_pos[0], python_pos[1]],
                rtol=TOLERANCE,
                atol=TOLERANCE,
            )

    def test_target_edge_case_zeros(self):
        """Test Target with all zero values."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        target_data = create_test_target(
            pnr=0, x=0.0, y=0.0, n=0, nx=0, ny=0, sumg=0.0, tnr=0
        )

        optv_t = OptvTarget(**target_data)
        python_t = PythonTarget(**target_data)

        assert optv_t.pnr() == python_t.pnr
        assert optv_t.tnr() == python_t.tnr

        optv_pos = optv_t.pos()
        python_pos = python_t.pos()
        assert abs(optv_pos[0] - python_pos[0]) < TOLERANCE
        assert abs(optv_pos[1] - python_pos[1]) < TOLERANCE

    def test_target_large_values(self):
        """Test Target with large coordinate values."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        target_data = create_test_target(
            pnr=9999,
            x=10000.5,
            y=20000.3,
            n=1000,
            nx=50,
            ny=50,
            sumg=999999.0,
            tnr=100,
        )

        optv_t = OptvTarget(**target_data)
        python_t = PythonTarget(**target_data)

        optv_pos = optv_t.pos()
        python_pos = python_t.pos()
        np.testing.assert_allclose(
            [optv_pos[0], optv_pos[1]],
            [python_pos[0], python_pos[1]],
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    def test_target_float_precision(self):
        """Test Target with high-precision float values."""
        from optv.tracking_framebuf import Target as OptvTarget
        from algorithms.tracking_frame_buf import Target as PythonTarget

        target_data = create_test_target(
            pnr=1,
            x=123.456789012345,
            y=789.012345678901,
            n=5,
            nx=2,
            ny=2,
            sumg=123.456789,
            tnr=0,
        )

        optv_t = OptvTarget(**target_data)
        python_t = PythonTarget(**target_data)

        optv_pos = optv_t.pos()
        python_pos = python_t.pos()
        np.testing.assert_allclose(
            [optv_pos[0], optv_pos[1]],
            [python_pos[0], python_pos[1]],
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )
