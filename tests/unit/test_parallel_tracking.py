import os
import shutil
from pathlib import Path

import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.tracking_run import TrackingRun
from openptv2.storage import RunStore

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE_STORE = REPO_ROOT / "test_data" / "test_cavity" / "run.zarr"


def read_all_calibration(num_cams, base_path="."):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


@pytest.fixture
def temp_cavity_dir(tmp_path):
    """Fixture copying the test_cavity dataset plus its Zarr store to tmp.

    Zarr is the database of record (see docs/plans/
    2026-08-15-zarr-only-transition-plan.md): the store carries the per-frame
    targets and correspondences these tests track from; no ASCII is read.
    """
    original_dir = REPO_ROOT / "test_data" / "test_cavity"
    if not FIXTURE_STORE.exists():
        pytest.skip("test_cavity run.zarr fixture not found")

    dest_dir = tmp_path / "test_cavity"
    shutil.copytree(original_dir, dest_dir, ignore=shutil.ignore_patterns("res*"))

    old_cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        yield dest_dir
    finally:
        os.chdir(old_cwd)


def run_tracking(num_threads, add_flag, work_dir):
    """Track the fixture store's frames with the given thread count."""
    store = RunStore(work_dir / "run.zarr", mode="a")

    cpar = ControlPar.from_yaml("parameters.yaml")
    calib = read_all_calibration(cpar.num_cams, base_path=".")

    max_targets = 20000
    tpar = convert_track_par_to_tuple(TrackPar.from_yaml("parameters.yaml"))._replace(
        add=add_flag
    )
    run = TrackingRun(
        seq_par=SequencePar.from_yaml("parameters.yaml"),
        tpar=tpar,
        vpar=VolumePar.from_yaml("parameters.yaml"),
        cpar=cpar,
        buf_len=4,
        max_targets=max_targets,
        corres_file_base="res/rt_is",
        linkage_file_base="res/ptv_is",
        prio_file_base="res/added",
        cal=calib,
        flatten_tol=0.0001,
        store=store,
    )

    track_forward_start(run)
    for step in range(run.seq_par.first, run.seq_par.last):
        trackcorr_c_loop(run, step, num_threads=num_threads)
    trackcorr_c_finish(run, run.seq_par.last)

    return run


def test_tracking_determinism(temp_cavity_dir):
    """Verify tracking correctness and determinism (add=0)."""
    run_ref = run_tracking(num_threads=1, add_flag=0, work_dir=temp_cavity_dir)
    assert run_ref.npart > 0
    assert run_ref.nlinks > 0

    # Ensure backward compatibility when num_threads is explicitly passed
    run_compat = run_tracking(num_threads=4, add_flag=0, work_dir=temp_cavity_dir)
    assert run_compat.npart == run_ref.npart
    assert run_compat.nlinks == run_ref.nlinks


def test_tracking_with_add(temp_cavity_dir):
    """Verify tracking with particle addition (add=1)."""
    run_ref = run_tracking(num_threads=1, add_flag=1, work_dir=temp_cavity_dir)
    assert run_ref.npart > 0
    assert run_ref.nlinks > 0

    run_compat = run_tracking(num_threads=4, add_flag=1, work_dir=temp_cavity_dir)
    assert run_compat.npart == run_ref.npart
    assert run_compat.nlinks == run_ref.nlinks
