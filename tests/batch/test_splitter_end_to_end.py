"""End-to-end proof of the YAML-driven splitter pipeline.

A YAML saved with splitter mode and the plugin selection must run
unmodified through both batch runners with NO CLI plugin flags, and the
serial and parallel runs must produce identical rt_is outputs. The split
per-camera views stay in memory: detection and stereo matching happen in
the same process and only target/rt_is files are written.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from openptv2.storage import RunStore, resolve_store_path

FIXTURE = Path(__file__).parent.parent.parent / "test_data" / "test_splitter"
FIRST, LAST = 1000001, 1000004


def _prepare_experiment(tmp_path: Path, name: str) -> Path:
    """Scratch copy of the fixture with the splitter plugins selected in the
    YAML and no stale results."""
    exp_path = tmp_path / name
    shutil.copytree(FIXTURE, exp_path)

    res_path = exp_path / "res"
    if res_path.exists():
        shutil.rmtree(res_path)

    yaml_file = exp_path / "parameters_Run1.yaml"
    data = yaml.safe_load(yaml_file.read_text())
    data["plugins"] = {
        "selected_sequence": "splitter_sequence",
        "selected_tracking": "splitter_tracking",
    }
    yaml_file.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    )
    return yaml_file


def _read_correspondences(exp_path: Path) -> dict:
    store = RunStore(resolve_store_path(exp_path / "res"), mode="r")
    out = {}
    for frame in range(FIRST, LAST + 1):
        assert store.has_correspondences(frame), (
            f"missing correspondences for frame {frame}"
        )
        out[frame] = store.read_correspondences(frame)
    return out


@pytest.mark.integration
def test_serial_and_parallel_splitter_runs_match(tmp_path):
    from openptv2.batch.pyptv_batch import run_batch
    from openptv2.batch.pyptv_batch_parallel import main as parallel_main

    serial_yaml = _prepare_experiment(tmp_path, "serial")
    parallel_yaml = _prepare_experiment(tmp_path, "parallel")

    images_before = sorted(p.name for p in (serial_yaml.parent / "img").glob("*.tif"))

    # No plugin flags anywhere: the YAML plugins.selected_* must drive both.
    run_batch(serial_yaml, FIRST, LAST, mode="sequence")
    parallel_main(parallel_yaml, FIRST, LAST, n_processes=2, mode="sequence")

    serial_out = _read_correspondences(serial_yaml.parent)
    parallel_out = _read_correspondences(parallel_yaml.parent)

    found_any = False
    for frame in serial_out:
        serial_pos, serial_cam_ids = serial_out[frame]
        parallel_pos, parallel_cam_ids = parallel_out[frame]
        found_any = found_any or len(serial_pos) > 0
        assert np.array_equal(serial_pos, parallel_pos), (
            f"serial and parallel positions differ for frame {frame}"
        )
        assert np.array_equal(serial_cam_ids, parallel_cam_ids), (
            f"serial and parallel cam ids differ for frame {frame}"
        )
    assert found_any, "no correspondences found in any frame"

    # In-memory splitting: no split-view images may appear on disk.
    images_after = sorted(p.name for p in (serial_yaml.parent / "img").glob("*.tif"))
    assert images_after == images_before


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
