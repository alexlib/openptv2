"""Guard test: the batch processing runtime must work from YAML alone.

We copy test_cavity to a temp dir, delete every legacy ``.par``/``.dat`` file
under ``parameters/``, keep only ``parameters_Run1.yaml``, and run the batch
sequence. If it still reconstructs particles, the runtime is genuinely
YAML-only and ``.par`` files are backward-compat scaffolding, not a dependency.
"""
import shutil
from pathlib import Path

import pytest

from openptv2.batch import pyptv_batch

DATASET = Path("test_data/test_cavity")


def _read_count(rt_is: Path) -> int:
    return int(rt_is.read_text().splitlines()[0])


@pytest.mark.integration
@pytest.mark.skipif(not DATASET.exists(), reason="test_cavity not present")
def test_batch_runs_without_any_par_files(tmp_path):
    ds = tmp_path / "test_cavity"
    shutil.copytree(DATASET, ds)

    # Strip ALL legacy parameter files — leave only the YAML + cal/img data.
    removed = 0
    for pat in ("parameters/*.par", "parameters/*.dat"):
        for f in ds.glob(pat):
            f.unlink()
            removed += 1
    assert removed > 0, "expected legacy .par/.dat files to remove"
    assert (ds / "parameters_Run1.yaml").exists()
    assert not list(ds.glob("parameters/*.par")), "no .par should remain"

    # Run the batch sequence purely from YAML.
    pyptv_batch.main(str(ds / "parameters_Run1.yaml"), 10000, 10001, mode="sequence")

    for frame in (10000, 10001):
        rt = ds / "res" / f"rt_is.{frame}"
        assert rt.exists(), f"rt_is.{frame} not produced from YAML-only run"
        assert _read_count(rt) > 0, f"rt_is.{frame} empty — YAML-only run degenerate"
