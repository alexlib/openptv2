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
import subprocess
import sys

from ..conftest import FIXTURES

# Test data location - use the discovered repository-local test_data root.
TEST_DATA_DIR = os.path.join(FIXTURES, "frame")


class TestFrameParity:
    """Test that Frame.read() produces identical results in both engines."""

    @pytest.mark.slow
    def test_frame_parity(self):
        """Test frame read parity between Cython and Python engines.

        Note: Marked as slow because it runs an isolated subprocess to avoid
        pytest/C extension memory issues.
        """
        test_code = f"""
import os
import numpy as np
from optv.tracking_framebuf import Frame as CythonFrame
from algorithms.frame_adapter import Frame as PythonFrame

test_data_dir = {TEST_DATA_DIR!r}
NUM_CAMS = 4
targ_files = [os.path.join(test_data_dir, f\"cam{{c}}.\") for c in range(1, NUM_CAMS + 1)]

cython_frame = CythonFrame(
    NUM_CAMS,
    corres_file_base=os.path.join(test_data_dir, \"rt_is\").encode(),
    linkage_file_base=os.path.join(test_data_dir, \"ptv_is\").encode(),
    target_file_base=[f.encode() for f in targ_files],
    frame_num=333,
)

python_frame = PythonFrame(
    NUM_CAMS,
    corres_file_base=os.path.join(test_data_dir, \"rt_is\"),
    linkage_file_base=os.path.join(test_data_dir, \"ptv_is\"),
    target_file_base=targ_files,
    frame_num=333,
)

TOLERANCE = 1e-7

cython_pos = cython_frame.positions()
python_pos = python_frame.positions()
assert cython_pos.shape == python_pos.shape
np.testing.assert_allclose(cython_pos, python_pos, rtol=TOLERANCE, atol=TOLERANCE)

for cam in range(NUM_CAMS):
    cython_targs = cython_frame.target_positions_for_camera(cam)
    python_targs = python_frame.target_positions_for_camera(cam)
    assert cython_targs.shape == python_targs.shape
    nan_mask = ~np.isnan(cython_targs)
    np.testing.assert_allclose(
        cython_targs[nan_mask], python_targs[nan_mask], rtol=TOLERANCE, atol=TOLERANCE
    )

assert cython_frame.positions().shape[0] == python_frame.num_parts
print(\"TEST_PASSED\")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        )

        if result.returncode != 0:
            pytest.fail(
                f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        assert "TEST_PASSED" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
