import os
import shutil
import time
from pathlib import Path

import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.tracking_run import tr_new


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
    """Fixture to copy the test_cavity dataset to a temporary directory.

    This ensures that our tests are completely non-destructive and do not
    leave any generated tracking files in the active git workspace.
    """
    original_dir = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not original_dir.exists():
        pytest.skip("test_cavity directory not found")

    dest_dir = tmp_path / "test_cavity"
    shutil.copytree(original_dir, dest_dir)

    old_cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        yield dest_dir
    finally:
        os.chdir(old_cwd)


def run_tracking(num_threads, add_flag):
    """Orchestrates a standard tracking run with the given thread count and add particle flag."""
    if os.path.exists("res"):
        shutil.rmtree("res")
    shutil.copytree("res_orig", "res")

    cpar = ControlPar.from_yaml("parameters.yaml")
    calib = read_all_calibration(cpar.num_cams, base_path=".")

    run = tr_new(
        SequencePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        VolumePar.from_yaml("parameters.yaml"),
        ControlPar.from_yaml("parameters.yaml"),
        4,
        20000,
        "res/rt_is",
        "res/ptv_is",
        "res/added",
        calib,
        0.0001,
    )
    run.tpar = run.tpar._replace(add=add_flag)

    track_forward_start(run)
    for step in range(run.seq_par.first, run.seq_par.last):
        trackcorr_c_loop(run, step, num_threads=num_threads)
    trackcorr_c_finish(run, run.seq_par.last)

    return run


def test_tracking_determinism(temp_cavity_dir):
    """Verify tracking correctness and determinism (add=0)."""
    run_ref = run_tracking(num_threads=1, add_flag=0)
    assert run_ref.npart > 0
    assert run_ref.nlinks > 0

    # Ensure backward compatibility when num_threads is explicitly passed
    run_compat = run_tracking(num_threads=4, add_flag=0)
    assert run_compat.npart == run_ref.npart
    assert run_compat.nlinks == run_ref.nlinks


def test_tracking_with_add(temp_cavity_dir):
    """Verify tracking with particle addition (add=1)."""
    run_ref = run_tracking(num_threads=1, add_flag=1)
    assert run_ref.npart > 0
    assert run_ref.nlinks > 0

    run_compat = run_tracking(num_threads=4, add_flag=1)
    assert run_compat.npart == run_ref.npart
    assert run_compat.nlinks == run_ref.nlinks
