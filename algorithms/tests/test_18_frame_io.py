"""
Engine comparison tests for frame I/O operations.

Tests read_targets, write_targets, and Frame.read() using actual test data files.
Mirrors bindings/tests/test_framebuf.py to ensure identical file I/O behavior.

Tolerance: 1e-10 (direct data comparison)
"""

import os
import numpy as np
import pytest
from pathlib import Path
from .conftest import get_tolerance, FIXTURES

TOLERANCE = get_tolerance("frame")


class TestReadTargets:
    """Test read_targets function, mirroring bindings test_read_targets."""

    def test_read_targets_sample(self):
        """Reading a targets file from Python, mirroring bindings test."""
        from algorithms.tracking_frame_buf import read_targets

        targs = read_targets("test_data/sample.", 42)

        assert len(targs) == 2
        assert [t.tnr for t in targs] == [1, 0]
        assert abs(targs[0].x - 1127.0) < TOLERANCE
        assert abs(targs[0].y - 796.0) < TOLERANCE
        assert abs(targs[1].x - 796.0) < TOLERANCE
        assert abs(targs[1].y - 809.0) < TOLERANCE

    def test_read_targets_frame(self):
        """Reading frame targets file."""
        from algorithms.tracking_frame_buf import read_targets

        targs = read_targets("test_data/frame/cam1.", 333)

        assert len(targs) == 13
        assert abs(targs[0].x - 900.0) < TOLERANCE
        assert abs(targs[0].y - 123.0) < TOLERANCE

    def test_read_targets_all_cameras(self):
        """Reading targets from all 4 cameras."""
        from algorithms.tracking_frame_buf import read_targets

        for cam in range(1, 5):
            targs = read_targets(f"test_data/frame/cam{cam}.", 333)
            assert len(targs) > 0


class TestWriteTargets:
    """Test write_targets function, mirroring bindings test_write_targets."""

    def test_round_trip_targets(self):
        """Round-trip test of writing targets, mirroring bindings test."""
        from algorithms.tracking_frame_buf import TargetArray, read_targets

        targs_list = read_targets("test_data/sample.", 42)

        # Convert list to TargetArray for write() method
        targs = TargetArray(len(targs_list))
        for i, t in enumerate(targs_list):
            targs[i].set_pnr(t.pnr)
            targs[i].set_pos((t.x, t.y))
            targs[i].set_pixel_counts(t.n, t.nx, t.ny)
            targs[i].set_sum_grey_value(t.sumg)
            targs[i].set_tnr(t.tnr)

        targs.write("test_data/alg_round_trip.", 1)
        tback = read_targets("test_data/alg_round_trip.", 1)

        assert len(targs_list) == len(tback)
        assert [t.tnr for t in targs_list] == [t.tnr for t in tback]
        for orig, back in zip(targs_list, tback):
            assert abs(orig.x - back.x) < TOLERANCE
            assert abs(orig.y - back.y) < TOLERANCE

    def test_write_targets_format(self):
        """Test that written targets have correct format."""
        from algorithms.tracking_frame_buf import Target, TargetArray, read_targets

        targs = TargetArray(2)
        targs[0].set_pnr(0)
        targs[0].set_pos((100.5, 200.3))
        targs[0].set_pixel_counts(10, 3, 3)
        targs[0].set_sum_grey_value(500)
        targs[0].set_tnr(1)
        targs[1].set_pnr(1)
        targs[1].set_pos((300.1, 400.7))
        targs[1].set_pixel_counts(15, 4, 4)
        targs[1].set_sum_grey_value(750)
        targs[1].set_tnr(2)

        targs.write("test_data/alg_fmt_test.", 99)
        back = read_targets("test_data/alg_fmt_test.", 99)

        assert len(back) == 2
        for i in range(2):
            assert abs(targs[i].pos()[0] - back[i].x) < TOLERANCE
            assert abs(targs[i].pos()[1] - back[i].y) < TOLERANCE

    def tearDown_cleanup(self):
        """Clean up written test files."""
        for f in [
            "test_data/alg_round_trip.0001_targets",
            "test_data/alg_fmt_test.0099_targets",
        ]:
            if os.path.exists(f):
                os.remove(f)


class TestTargetSortY:
    """Test TargetArray.sort_y, mirroring bindings test_sort_y."""

    def test_sort_y(self):
        """Sorting on the Y coordinate in place."""
        from algorithms.tracking_frame_buf import read_targets, TargetArray

        targs_list = read_targets("test_data/frame/cam1.", 333)
        revs_list = read_targets("test_data/frame/cam1_reversed.", 333)

        # Convert to TargetArray for sort_y
        revs = TargetArray(len(revs_list))
        for i, t in enumerate(revs_list):
            revs[i].set_pnr(t.pnr)
            revs[i].set_pos((t.x, t.y))
            revs[i].set_pixel_counts(t.n, t.nx, t.ny)
            revs[i].set_sum_grey_value(t.sumg)
            revs[i].set_tnr(t.tnr)

        revs.sort_y()

        for targ, rev in zip(targs_list, revs):
            assert abs(targ.x - rev.pos()[0]) < TOLERANCE
            assert abs(targ.y - rev.pos()[1]) < TOLERANCE


class TestFrameRead:
    """Test Frame.read(), mirroring bindings test_read_frame."""

    def test_read_frame(self):
        """Reading a frame from disk."""
        from algorithms.tracking_frame_buf import Frame

        target_bases = [f"test_data/frame/cam{c}." for c in range(1, 5)]

        frm = Frame(num_cams=4)
        success = frm.read(
            corres_file_base="test_data/frame/rt_is",
            linkage_file_base="test_data/frame/ptv_is",
            prio_file_base="",
            target_file_base=target_bases,
            frame_num=333,
        )

        assert success

        pos = frm.positions()
        assert pos.shape == (10, 3)

        targs = frm.target_positions_for_camera(3)
        assert targs.shape == (10, 2)

        targs_correct = np.array(
            [
                [426.0, 199.0],
                [429.0, 60.0],
                [431.0, 327.0],
                [509.0, 315.0],
                [345.0, 222.0],
                [465.0, 139.0],
                [487.0, 403.0],
                [241.0, 178.0],
                [607.0, 209.0],
                [563.0, 238.0],
            ]
        )
        np.testing.assert_array_equal(targs, targs_correct)


@pytest.fixture(autouse=True)
def cleanup_written_files():
    """Cleanup any files written during tests."""
    yield
    for f in [
        "test_data/alg_round_trip.0001_targets",
        "test_data/alg_fmt_test.0099_targets",
    ]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
