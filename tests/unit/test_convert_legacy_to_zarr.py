"""Unit tests for converting legacy ASCII run files to a unified Zarr store."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.storage.legacy import convert_ascii_to_zarr, import_run, main
from openptv2.storage.run_store import RunStore

pytestmark = pytest.mark.ci


@pytest.fixture
def temp_run_folder(tmp_path):
    """Create a temporary PTV run folder with synthetic legacy ASCII files."""
    img_dir = tmp_path / "img"
    res_dir = tmp_path / "res"
    img_dir.mkdir(parents=True)
    res_dir.mkdir(parents=True)

    # Frame 10001 targets for 4 cams
    for cam in range(1, 5):
        t_file = img_dir / f"cam{cam}.10001_targets"
        with open(t_file, "w") as f:
            f.write("2\n")
            f.write("   0  100.5000  200.5000     5     3     3   150    -1\n")
            f.write("   1  300.2500  400.2500     6     4     4   180    -1\n")

    # Frame 10001 rt_is
    with open(res_dir / "rt_is.10001", "w") as f:
        f.write("2\n")
        f.write("   1    10.000    20.000    30.000    0    0    0    0\n")
        f.write("   2    40.000    50.000    60.000    1    1    1    1\n")

    # Frame 10001 ptv_is
    with open(res_dir / "ptv_is.10001", "w") as f:
        f.write("2\n")
        f.write("  -1    1    10.000    20.000    30.000\n")
        f.write("  -1    2    40.000    50.000    60.000\n")

    # Frame 10001 added
    with open(res_dir / "added.10001", "w") as f:
        f.write("2\n")
        f.write("  -1    1    10.000    20.000    30.000 0\n")
        f.write("  -1    2    40.000    50.000    60.000 1\n")

    return tmp_path


def test_convert_ascii_to_zarr_basic(temp_run_folder):
    store = convert_ascii_to_zarr(temp_run_folder)
    assert isinstance(store, RunStore)
    assert 10001 in store.frames()

    # Verify targets
    targs = store.read_targets(0, 10001)
    assert len(targs) == 2
    assert pytest.approx(targs[0].pos()[0]) == 100.5

    # Verify correspondences
    pos, ids = store.read_correspondences(10001)
    assert len(pos) == 2
    assert np.allclose(pos[0], [10.0, 20.0, 30.0])

    # Verify linkage
    prev, nxt, p_pos = store.read_linkage(10001, "ptv_is")
    assert len(prev) == 2
    assert prev[0] == -1
    assert nxt[0] == 1


def test_convert_ascii_to_zarr_with_remove_ascii(temp_run_folder):
    target_file = temp_run_folder / "img" / "cam1.10001_targets"
    rt_file = temp_run_folder / "res" / "rt_is.10001"
    assert target_file.exists()
    assert rt_file.exists()

    store = convert_ascii_to_zarr(temp_run_folder, remove_ascii=True)
    assert isinstance(store, RunStore)

    # Verify ASCII files were deleted
    assert not target_file.exists()
    assert not rt_file.exists()
    assert not (temp_run_folder / "res" / "ptv_is.10001").exists()
    assert not (temp_run_folder / "res" / "added.10001").exists()

    # Verify Zarr store has the data
    assert store.has_targets(0, 10001)
    assert store.has_correspondences(10001)


def test_convert_cli_main_entrypoint(temp_run_folder):
    ret = main([str(temp_run_folder), "--remove-ascii"])
    assert ret == 0
    zarr_dir = temp_run_folder / "res" / "run.zarr"
    assert zarr_dir.exists()
    assert not (temp_run_folder / "img" / "cam1.10001_targets").exists()
