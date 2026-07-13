"""Tests for openptv2.batch.pyptv_batch_plugins.run_batch, covering all three
modes (sequence, tracking, both) with the built-in splitter plugins.

Runs against a scratch copy of test_data/test_splitter: the splitter plugins
write detection (cam*_targets) and tracking (rt_is.*/ptv_is.*) output files
in place, so operating on the checked-in fixture directly would leave it
dirty after every run.
"""

import shutil
from pathlib import Path

import pytest

from openptv2.batch.pyptv_batch_plugins import run_batch


@pytest.fixture
def splitter_copy(tmp_path):
    src_exp_path = Path(__file__).parent.parent.parent / "test_data" / "test_splitter"
    assert src_exp_path.exists(), f"Fixture not found: {src_exp_path}"

    test_exp_path = tmp_path / "test_splitter"
    shutil.copytree(src_exp_path, test_exp_path)
    return test_exp_path / "parameters_Run1.yaml"


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["sequence", "tracking", "both"])
def test_batch_plugins_runs(splitter_copy, mode):
    """Each mode runs end-to-end via the built-in splitter plugins and exits
    cleanly (run_batch raises on plugin failure, so no exception == success).
    """
    run_batch(
        yaml_file=splitter_copy,
        seq_first=1000001,
        seq_last=1000002,
        tracking_plugin="ext_tracker_splitter",
        sequence_plugin="ext_sequence_splitter",
        mode=mode,
    )

    res_dir = splitter_copy.parent / "res"
    if mode in ("sequence", "both"):
        assert (res_dir / "rt_is.1000001").exists()
    if mode in ("tracking", "both"):
        assert (res_dir / "ptv_is.1000001").exists()
