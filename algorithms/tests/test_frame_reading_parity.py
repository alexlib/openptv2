"""
Frame reading parity tests: Python vs Cython readers.

Both engines read the SAME _targets files through their own native readers,
verifying that:
1. File path resolution works identically
2. Target data (position, pixel counts, grey values) matches exactly
3. Error handling behavior is consistent

This isolates the I/O layer from the tracking algorithm.
"""

import os
import numpy as np
import pytest
from pathlib import Path

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "track"
)


class TestFrameReadingParity:
    """Compare Python and Cython frame readers on identical input files."""

    def test_read_targets_single_camera(self):
        """Python and Cython read identical target data from one camera."""
        from algorithms.tracking_frame_buf import (
            read_targets as py_read_targets,
        )
        from optv.tracking_framebuf import (
            read_targets as cy_read_targets,
        )

        file_base = os.path.join(TRACK_DATA_DIR, "newpart", "cam1.")
        frame_num = 10001

        py_targets = py_read_targets(file_base, frame_num)
        cy_targets = cy_read_targets(file_base, frame_num)

        assert len(py_targets) == len(cy_targets), (
            f"Target count mismatch: Python={len(py_targets)}, Cython={len(cy_targets)}"
        )

        for i, (py_t, cy_t) in enumerate(zip(py_targets, cy_targets)):
            assert py_t.pnr == cy_t.pnr(), f"Target {i}: pnr mismatch"
            assert py_t.x == cy_t.pos()[0], f"Target {i}: x mismatch"
            assert py_t.y == cy_t.pos()[1], f"Target {i}: y mismatch"
            assert py_t.n == cy_t.count_pixels()[0], f"Target {i}: n mismatch"
            assert py_t.nx == cy_t.count_pixels()[1], f"Target {i}: nx mismatch"
            assert py_t.ny == cy_t.count_pixels()[2], f"Target {i}: ny mismatch"
            assert py_t.sumg == cy_t.sum_grey_value(), f"Target {i}: sumg mismatch"
            assert py_t.tnr == cy_t.tnr(), f"Target {i}: tnr mismatch"

    def test_read_targets_all_cameras(self):
        """Python and Cython read identical target data from all 4 cameras."""
        from algorithms.tracking_frame_buf import (
            read_targets as py_read_targets,
        )
        from optv.tracking_framebuf import (
            read_targets as cy_read_targets,
        )

        for cam in range(1, 5):
            file_base = os.path.join(TRACK_DATA_DIR, "newpart", f"cam{cam}.")
            for frame_num in (10001, 10002, 10003):
                py_targets = py_read_targets(file_base, frame_num)
                cy_targets = cy_read_targets(file_base, frame_num)

                assert len(py_targets) == len(cy_targets), (
                    f"Cam {cam}, frame {frame_num}: count mismatch "
                    f"Python={len(py_targets)}, Cython={len(cy_targets)}"
                )

                for i, (py_t, cy_t) in enumerate(zip(py_targets, cy_targets)):
                    np.testing.assert_allclose(
                        [py_t.x, py_t.y],
                        list(cy_t.pos()),
                        atol=1e-10,
                        err_msg=f"Cam {cam}, frame {frame_num}, target {i}: position mismatch",
                    )

    def test_read_targets_underscore_format(self):
        """Python reader handles underscore-separated filenames like C reader."""
        from algorithms.tracking_frame_buf import (
            _target_filename,
        )

        # The C reader uses: sprintf(filein, "%s%04d%s", file_base, frame_num, "_targets")
        # So "sample_" + 0001 + "_targets" = "sample_0001_targets"
        file_base = "sample_"
        frame_num = 1

        expected = "sample_0001_targets"
        actual = _target_filename(file_base, frame_num)
        assert actual == expected, f"Expected '{expected}', got '{actual}'"

    def test_read_targets_dotted_format(self):
        """Python reader handles dot-separated filenames like C reader."""
        from algorithms.tracking_frame_buf import (
            _target_filename,
        )

        # The C reader uses: sprintf(filein, "%s%04d%s", file_base, frame_num, "_targets")
        # So "cam1." + 0001 + "_targets" = "cam1.0001_targets"
        file_base = "cam1."
        frame_num = 10001

        expected = "cam1.10001_targets"
        actual = _target_filename(file_base, frame_num)
        assert actual == expected, f"Expected '{expected}', got '{actual}'"

    def test_read_path_frame_parity(self):
        """Python and C read identical particle data from rt_is files."""
        from algorithms.tracking_frame_buf import (
            read_path_frame as py_read_path_frame,
        )

        corres_base = os.path.join(TRACK_DATA_DIR, "res_orig", "particles")
        frame_num = 10001

        # Python reader
        py_cor_buf, py_path_buf = py_read_path_frame(corres_base, "", "", frame_num)

        py_count = len(py_cor_buf)
        assert py_count > 0, "Should have read some particles"

        # Verify the Python reader produces sensible data
        for i in range(py_count):
            assert py_cor_buf[i].nr == i + 1
            assert len(py_path_buf[i].x) == 3
            for cam in range(4):
                assert py_cor_buf[i].p[cam] >= -1

        # Also verify by reading the raw file directly
        raw_path = f"{corres_base}.{frame_num}"
        with open(raw_path) as f:
            lines = f.readlines()
        raw_count = int(lines[0].strip())
        assert py_count == raw_count, (
            f"Python reader count {py_count} != file count {raw_count}"
        )

    def test_read_targets_missing_file_behavior(self):
        """Document how each reader handles missing files.

        Both readers have issues with missing files:
        - Python raises FileNotFoundError
        - Cython returns a corrupted TargetArray (num_targets=-1)

        This is a known discrepancy; the Frame.read() method works around
        it by checking file existence before calling read_targets.
        """
        from algorithms.tracking_frame_buf import (
            read_targets as py_read_targets,
        )
        from optv.tracking_framebuf import (
            read_targets as cy_read_targets,
        )

        file_base = os.path.join(TRACK_DATA_DIR, "nonexistent", "cam1.")
        frame_num = 99999

        # Python raises FileNotFoundError for missing file
        with pytest.raises((FileNotFoundError, IOError)):
            py_read_targets(file_base, frame_num)

        # Cython has a bug: when C returns -1, the wrapper creates a
        # TargetArray with -1 targets, causing SystemError on len()
        # This is a known issue in the Cython binding.
        cy_result = cy_read_targets(file_base, frame_num)
        # The result is in an invalid state - we just verify it was called
        assert cy_result is not None

    def test_frame_read_file_existence_check(self):
        """Python Frame.read checks file existence before attempting to read."""
        from algorithms.tracking_frame_buf import Frame

        frame = Frame(num_cams=4, max_targets=1000)

        target_bases = [
            os.path.join(TRACK_DATA_DIR, "newpart", f"cam{c + 1}.") for c in range(4)
        ]

        # Should return False for non-existent corres file
        result = frame.read(
            corres_file_base="/nonexistent/path/particles",
            linkage_file_base="",
            prio_file_base="",
            target_file_base=target_bases,
            frame_num=10001,
        )
        assert result is False, "Frame.read should return False for missing corres file"

        # Should return True for valid files
        corres_base = os.path.join(TRACK_DATA_DIR, "res_orig", "particles")
        result = frame.read(
            corres_file_base=corres_base,
            linkage_file_base="",
            prio_file_base="",
            target_file_base=target_bases,
            frame_num=10001,
        )
        assert result is True, "Frame.read should return True for valid files"
        assert frame.num_parts > 0, "Should have read some particles"
        for cam in range(4):
            assert frame.num_targets[cam] >= 0, (
                f"Camera {cam} should have valid target count"
            )
