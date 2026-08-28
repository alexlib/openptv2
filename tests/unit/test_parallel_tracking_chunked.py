"""Unit and integration tests for parallel chunked tracking & trajectory stitching (Task 4)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2 import (
    Calibration,
    ControlParams,
    RunStore,
    SequenceParams,
    Tracker,
    TrackingParams,
    VolumeParams,
    partition_tracking_chunks,
    track_sequence_chunked_parallel,
)
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from tests._support import find_test_data_root

TEST_DATA_ROOT = find_test_data_root()
CAVITY_DIR = TEST_DATA_ROOT / "test_cavity"


@pytest.fixture
def cavity_test_env(tmp_path):
    """Isolated environment with test_cavity fixtures."""
    if not (CAVITY_DIR / "res_orig").exists():
        pytest.skip("test_cavity res_orig fixtures missing")

    dst = tmp_path / "run"
    dst.mkdir()
    shutil.copytree(CAVITY_DIR / "res_orig", dst / "res")
    shutil.copytree(CAVITY_DIR / "cal", dst / "cal")
    shutil.copy(CAVITY_DIR / "parameters.yaml", dst / "parameters.yaml")

    old_cwd = os.getcwd()
    os.chdir(dst)
    try:
        yield dst
    finally:
        os.chdir(old_cwd)


def _load_cavity_cals(cpar, base_path: Path) -> list[Calibration]:
    return [
        Calibration.from_file(
            str(base_path / f"cal/cam{cam + 1}.tif.ori"),
            str(base_path / f"cal/cam{cam + 1}.tif.addpar"),
        )
        for cam in range(cpar.num_cams)
    ]


def test_partition_tracking_chunks_math():
    """Verify temporal chunk partitioning with overlap."""
    # 1 worker -> single chunk
    chunks = partition_tracking_chunks(first=1000, last=1100, n_workers=1, overlap=4)
    assert chunks == [(1000, 1100, 1000, 1100)]

    # 4 workers over 101 frames (1000..1100)
    chunks = partition_tracking_chunks(first=1000, last=1100, n_workers=4, overlap=4)
    assert len(chunks) == 4

    # Check continuity of valid ranges
    for i in range(len(chunks) - 1):
        assert chunks[i][3] + 1 == chunks[i + 1][2]  # valid_end + 1 == next valid_start

    # Check first and last valid frames match sequence bounds
    assert chunks[0][2] == 1000
    assert chunks[-1][3] == 1100

    # Check overlap extension
    for i in range(1, len(chunks)):
        # chunk start should be extended backward
        assert chunks[i][0] <= chunks[i][2] - 4
    for i in range(len(chunks) - 1):
        # chunk end should be extended forward
        assert chunks[i][1] >= chunks[i][3] + 4


def test_chunked_tracking_cavity_parity_store(cavity_test_env):
    """Test chunked parallel tracking against serial baseline with RunStore."""
    cpar = ControlPar.from_yaml("parameters.yaml")
    vpar = VolumePar.from_yaml("parameters.yaml")
    tpar = TrackPar.from_yaml("parameters.yaml")
    spar = SequencePar.from_yaml("parameters.yaml")
    cals = _load_cavity_cals(cpar, cavity_test_env)

    # 1. Serial Baseline Tracking
    serial_store = RunStore("res/serial_run.zarr", mode="w")
    serial_tracker = Tracker(
        cpar, vpar, tpar, spar, cals, naming={"linkage": "res/ptv_is", "corres": "res/rt_is", "prio": "res/added"}, store=serial_store
    )
    serial_tracker.full_forward_3d()
    serial_npart = serial_tracker.npart
    serial_nlinks = serial_tracker.nlinks
    assert serial_npart == 2082
    assert serial_nlinks == 1518

    # 2. Parallel Chunked Tracking (2 workers on 4 frames)
    parallel_store = RunStore("res/parallel_run.zarr", mode="w")
    # Populate parallel store with correspondences
    for f in range(spar.get_first(), spar.get_last() + 1):
        pos, ids = serial_store.read_correspondences(f)
        parallel_store.write_correspondences(f, pos, ids)

    par_tracker = Tracker(
        cpar, vpar, tpar, spar, cals, naming={"linkage": "res/ptv_is", "corres": "res/rt_is", "prio": "res/added"}, store=parallel_store
    )
    par_npart, par_nlinks = par_tracker.full_forward_chunked_parallel(
        n_workers=2, overlap=1, mode="3d", postprocess=False
    )

    # 3. Verify linkage frame parity across every frame
    for f in range(spar.get_first(), spar.get_last() + 1):
        assert parallel_store.has_linkage(f, "ptv_is")
        s_prev, s_next, s_pos = serial_store.read_linkage(f, "ptv_is")
        p_prev, p_next, p_pos = parallel_store.read_linkage(f, "ptv_is")

        assert np.array_equal(p_pos, s_pos)
        # Link arrays must match
        assert np.array_equal(p_prev, s_prev)
        assert np.array_equal(p_next, s_next)

    assert par_npart == serial_npart
    assert par_nlinks == serial_nlinks


def test_chunked_tracking_4be_and_corr_modes(cavity_test_env):
    """Test chunked tracking execution in 4BE and standard correlation modes."""
    cpar = ControlPar.from_yaml("parameters.yaml")
    vpar = VolumePar.from_yaml("parameters.yaml")
    tpar = TrackPar.from_yaml("parameters.yaml")
    spar = SequencePar.from_yaml("parameters.yaml")
    cals = _load_cavity_cals(cpar, cavity_test_env)

    store_4be = RunStore("res/run_4be.zarr", mode="w")
    npart_4be, nlinks_4be = track_sequence_chunked_parallel(
        cpar=cpar,
        vpar=vpar,
        tpar=tpar,
        spar=spar,
        cals=cals,
        store=store_4be,
        n_workers=2,
        overlap=1,
        mode="4be",
        postprocess=False,
    )
    assert npart_4be > 0
    assert nlinks_4be > 0
    assert store_4be.sealed


def test_chunked_tracking_synthetic_trajectory_continuity(tmp_path, monkeypatch):
    """Verify that multi-frame trajectories are stitched across chunk boundaries without breakage."""
    work_dir = tmp_path / "synthetic_run"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    # 1. Generate 20 frames of 10 continuous particle trajectories
    n_frames = 20
    n_particles = 10
    first_frame = 1000
    last_frame = first_frame + n_frames - 1

    # Base coordinates + linear motion
    np.random.seed(42)
    base_pos = np.random.uniform(low=[-20.0, -20.0, 10.0], high=[20.0, 20.0, 50.0], size=(n_particles, 3))
    velocities = np.random.uniform(low=[-0.5, -0.5, -0.5], high=[0.5, 0.5, 0.5], size=(n_particles, 3))

    store = RunStore("res/run.zarr", mode="w")
    for step in range(n_frames):
        f = first_frame + step
        pos = base_pos + step * velocities
        ids = np.ones((n_particles, 4), dtype=np.int32)
        store.write_correspondences(f, pos, ids)

    # Tracking parameters
    cpar = ControlPar(num_cams=4)
    vpar = VolumePar(
        X_lay=[-100.0, 100.0],
        Zmin_lay=[-100.0, 100.0],
        Zmax_lay=[-100.0, 100.0],
    )
    tpar = TrackPar(
        dvxmin=-2.0, dvxmax=2.0,
        dvymin=-2.0, dvymax=2.0,
        dvzmin=-2.0, dvzmax=2.0,
        dangle=100.0, dacc=2.0,
        add=0,
    )
    spar = SequencePar(
        img_base_name=[""] * 4,
        first=first_frame,
        last=last_frame,
    )
    cals = [Calibration() for _ in range(4)]
    from openptv2.algorithms.multimed import init_mmlut
    for c in cals:
        init_mmlut(vpar, cpar, c)

    # 2. Run chunked tracking with 4 workers (5 frames per chunk, overlap=2)
    npart, nlinks = track_sequence_chunked_parallel(
        cpar=cpar,
        vpar=vpar,
        tpar=tpar,
        spar=spar,
        cals=cals,
        store=store,
        n_workers=4,
        overlap=2,
        mode="3d",
        postprocess=False,
    )

    # 3. Verify sealed trajectories
    assert store.sealed
    traj_idx = store.traj_index()
    # All 10 synthetic particles should have continuous trajectories spanning all 20 frames
    assert len(traj_idx["trajid"]) == n_particles
    for i in range(n_particles):
        assert traj_idx["first"][i] == first_frame
        assert traj_idx["last"][i] == last_frame
        assert traj_idx["length"][i] == n_frames


def test_chunked_tracking_with_postprocessing(cavity_test_env):
    """Test chunked tracking with cold start, gap relinking, and reciprocity postprocessing."""
    cpar = ControlPar.from_yaml("parameters.yaml")
    vpar = VolumePar.from_yaml("parameters.yaml")
    tpar = TrackPar.from_yaml("parameters.yaml")
    spar = SequencePar.from_yaml("parameters.yaml")
    cals = _load_cavity_cals(cpar, cavity_test_env)

    store = RunStore("res/run_post.zarr", mode="w")
    # Copy correspondences from ASCII fixture
    tracker_base = Tracker(cpar, vpar, tpar, spar, cals, store=store)
    npart, nlinks = tracker_base.full_forward_chunked_parallel(
        n_workers=2, overlap=2, mode="3d", postprocess=True
    )
    assert npart > 0
    assert nlinks > 0
    assert store.sealed
    assert len(store.traj_index()["trajid"]) > 0



