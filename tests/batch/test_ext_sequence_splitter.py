"""Pytest version of ext_sequence_splitter plugin test (simplified)"""

import shutil
from pathlib import Path

import pytest

from openptv2.batch.pyptv_batch_plugins import run_batch


@pytest.mark.integration
def test_ext_sequence_splitter_plugin(tmp_path):
    """Test that ext_sequence_splitter plugin runs without errors using direct call.

    Runs against a scratch copy of the fixture: the plugin writes detection
    (cam*_targets) and tracking (rt_is.*) output files in place, so operating
    on the real test_data/test_splitter would leave the checked-in fixture
    dirty after every run.
    """
    src_exp_path = Path(__file__).parent.parent.parent / "test_data" / "test_splitter"
    assert src_exp_path.exists(), f"Fixture not found: {src_exp_path}"

    test_exp_path = tmp_path / "test_splitter"
    shutil.copytree(src_exp_path, test_exp_path)
    yaml_file = test_exp_path / "parameters_Run1.yaml"
    assert yaml_file.exists(), f"YAML file not found: {yaml_file}"

    # Frame range and plugin names
    start_frame = 1000001
    end_frame = 1000002
    sequence_plugin = "ext_sequence_splitter"
    tracking_plugin = "ext_tracker_splitter"  # Not used, but required by signature

    run_batch(
        yaml_file=yaml_file,
        seq_first=start_frame,
        seq_last=end_frame,
        tracking_plugin=tracking_plugin,
        sequence_plugin=sequence_plugin,
        mode="sequence",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
