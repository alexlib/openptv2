"""
Engine comparison tests for Frame class.

These tests verify that both engines (optv and python) produce identical results
for Frame.read(), positions(), and target_positions_for_camera().

Usage:
    pytest tests/engine_comparison/test_frame_parity.py -v
    pytest tests/engine_comparison/test_frame_parity.py -v --validate-engine
"""

import os
import pytest
import numpy as np

# Test data location - use absolute path from project root
TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "test_data",
    "frame",
)


class TestFrameParity:
    """Test that Frame.read() produces identical results in both engines."""

    def test_frame_read_cython(self):
        """Test reading frame with Cython engine."""
        from optv.tracking_framebuf import Frame

        # Setup file paths (matching test_framebuf.py)
        targ_files = [os.path.join(TEST_DATA_DIR, f"cam{c}.") for c in range(1, 5)]

        frm = Frame(
            4,
            corres_file_base=os.path.join(TEST_DATA_DIR, "rt_is").encode(),
            linkage_file_base=os.path.join(TEST_DATA_DIR, "ptv_is").encode(),
            target_file_base=[f.encode() for f in targ_files],
            frame_num=333,
        )

        pos = frm.positions()

        # Verify shape
        assert pos.shape == (10, 3), f"Expected (10, 3), got {pos.shape}"

        # Get target positions for camera 3
        targs = frm.target_positions_for_camera(3)

        # Expected values from Cython test
        targs_expected = np.array(
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

        np.testing.assert_array_equal(targs, targs_expected)

    def test_frame_read_python(self):
        """Test reading frame with Python engine."""
        from algorithms.frame_adapter import Frame

        # Setup file paths
        targ_files = [os.path.join(TEST_DATA_DIR, f"cam{c}.") for c in range(1, 5)]

        frm = Frame(
            4,
            corres_file_base=os.path.join(TEST_DATA_DIR, "rt_is"),
            linkage_file_base=os.path.join(TEST_DATA_DIR, "ptv_is"),
            target_file_base=targ_files,
            frame_num=333,
        )

        pos = frm.positions()

        # Verify shape
        assert pos.shape == (10, 3), f"Expected (10, 3), got {pos.shape}"

        # Get target positions for camera 3
        targs = frm.target_positions_for_camera(3)

        # Expected values from Cython test
        targs_expected = np.array(
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

        np.testing.assert_array_equal(targs, targs_expected)

    def test_frame_parity(self):
        """Compare Frame results between Cython and Python engines."""
        from optv.tracking_framebuf import Frame as CythonFrame
        from algorithms.frame_adapter import Frame as PythonFrame

        TOLERANCE = 1e-7

        # Setup file paths
        targ_files = [os.path.join(TEST_DATA_DIR, f"cam{c}.") for c in range(1, 5)]

        # Read with Cython
        cython_frame = CythonFrame(
            4,
            corres_file_base=os.path.join(TEST_DATA_DIR, "rt_is").encode(),
            linkage_file_base=os.path.join(TEST_DATA_DIR, "ptv_is").encode(),
            target_file_base=[f.encode() for f in targ_files],
            frame_num=333,
        )

        # Read with Python
        python_frame = PythonFrame(
            4,
            corres_file_base=os.path.join(TEST_DATA_DIR, "rt_is"),
            linkage_file_base=os.path.join(TEST_DATA_DIR, "ptv_is"),
            target_file_base=targ_files,
            frame_num=333,
        )

        # Compare positions()
        cython_pos = cython_frame.positions()
        python_pos = python_frame.positions()

        assert cython_pos.shape == python_pos.shape, (
            f"Shape mismatch: cython={cython_pos.shape}, python={python_pos.shape}"
        )

        np.testing.assert_allclose(
            cython_pos,
            python_pos,
            rtol=TOLERANCE,
            atol=TOLERANCE,
            err_msg="positions() mismatch between engines",
        )

        # Compare target_positions_for_camera() for each camera
        for cam in range(4):
            cython_targs = cython_frame.target_positions_for_camera(cam)
            python_targs = python_frame.target_positions_for_camera(cam)

            assert cython_targs.shape == python_targs.shape, (
                f"Shape mismatch for cam {cam}: cython={cython_targs.shape}, python={python_targs.shape}"
            )

            # Handle NaN comparison
            nan_mask = ~np.isnan(cython_targs)
            np.testing.assert_allclose(
                cython_targs[nan_mask],
                python_targs[nan_mask],
                rtol=TOLERANCE,
                atol=TOLERANCE,
                err_msg=f"target_positions_for_camera({cam}) mismatch",
            )

        # Also verify num_parts matches
        # Cython Frame doesn't have num_parts attribute, use positions().shape[0]
        cython_num_parts = cython_frame.positions().shape[0]
        python_num_parts = python_frame.num_parts
        assert cython_num_parts == python_num_parts, (
            f"num_parts mismatch: cython={cython_num_parts}, python={python_num_parts}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
