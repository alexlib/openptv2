"""Unit tests for parallel multi-camera stereo correspondences."""

import os
from pathlib import Path

import numpy as np
import pytest

from openptv2.correspondences import (
    MatchedCoords,
    correspondences,
    match_correspondences_batch_parallel,
    match_frame_correspondences,
)
from openptv2.gui.experiment import Experiment
from openptv2.gui.ptv import py_start_proc_c
from openptv2.orientation import point_positions
from openptv2.storage import RunStore
from openptv2.tracking_framebuf import TargetArray

REPO_ROOT = Path(__file__).parent.parent.parent
TEST_DATA = REPO_ROOT / "test_data" / "test_cavity"


@pytest.fixture
def cavity_params():
    """Load calibration, control, and volume parameters from test_cavity."""
    yaml_file = TEST_DATA / "parameters_Run1.yaml"
    if not yaml_file.exists():
        pytest.skip("test_cavity fixture not present")

    exp = Experiment()
    cwd0 = os.getcwd()
    os.chdir(TEST_DATA)
    try:
        exp.pm.from_yaml(yaml_file)
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(exp.pm)
        return exp, cpar, spar, vpar, cals
    finally:
        os.chdir(cwd0)


@pytest.fixture
def sample_targets(cavity_params):
    """Load targets from test_cavity for frames 10001..10004."""
    exp, cpar, spar, vpar, cals = cavity_params
    num_cams = len(cals)
    frames = [10001, 10002, 10003, 10004]
    targets_per_frame = {}

    from openptv2.algorithms.tracking_frame_buf import (
        read_targets as read_targets_ascii,
    )

    for frame in frames:
        frame_targs = []
        for cam in range(num_cams):
            file_base = str(TEST_DATA / "img" / f"cam{cam + 1}.")
            targs = read_targets_ascii(file_base, frame, cam_idx=cam)
            tarr = TargetArray(len(targs))
            for i, t in enumerate(targs):
                tarr[i].set_pnr(t.pnr)
                tarr[i].set_pos((t.x, t.y))
                tarr[i].set_pixel_counts(t.n, t.nx, t.ny)
                tarr[i].set_sum_grey_value(t.sumg)
                tarr[i].set_tnr(t.tnr)
            frame_targs.append(tarr)
        targets_per_frame[frame] = frame_targs

    return targets_per_frame, frames


def test_match_frame_correspondences_parity(cavity_params, sample_targets):
    """Test that match_frame_correspondences produces bit-exact parity with sequential code."""
    exp, cpar, spar, vpar, cals = cavity_params
    targets_per_frame, frames = sample_targets
    num_cams = len(cals)

    for frame in frames:
        detections = targets_per_frame[frame]

        # 1. Reference sequential calculation
        corrected = []
        for i_cam in range(num_cams):
            targs = detections[i_cam]
            if len(targs) > 0:
                targs.sort_y()
            mc = MatchedCoords(targs, cpar, cals[i_cam])
            corrected.append(mc)

        sorted_pos, sorted_corresp, _ = correspondences(
            detections, corrected, cals, vpar, cpar
        )
        sorted_pos = np.concatenate(sorted_pos, axis=1)
        sorted_corresp = np.concatenate(sorted_corresp, axis=1)
        flat = np.array(
            [
                corr.get_by_pnrs(corresp)
                for corr, corresp in zip(corrected, sorted_corresp)
            ]
        )
        ref_pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)
        ref_cam_ids = sorted_corresp.astype(np.int32).T

        # 2. match_frame_correspondences calculation
        pos_3d, cam_target_ids = match_frame_correspondences(detections, cpar, cals, vpar)

        # 3. Assert bit-exact matching
        assert len(pos_3d) == len(ref_pos)
        assert np.allclose(pos_3d, ref_pos, atol=1e-12)
        assert np.array_equal(cam_target_ids, ref_cam_ids)


def test_batch_parallel_in_memory(cavity_params, sample_targets):
    """Test match_correspondences_batch_parallel across 1 vs 4 workers in memory."""
    exp, cpar, spar, vpar, cals = cavity_params
    targets_per_frame, frames = sample_targets

    # Run sequential (n_workers=1)
    seq_results = match_correspondences_batch_parallel(
        frames=frames,
        cpar=cpar,
        cals=cals,
        vpar=vpar,
        targets=targets_per_frame,
        n_workers=1,
    )

    # Run parallel (n_workers=4)
    par_results = match_correspondences_batch_parallel(
        frames=frames,
        cpar=cpar,
        cals=cals,
        vpar=vpar,
        targets=targets_per_frame,
        n_workers=4,
    )

    assert set(seq_results.keys()) == set(par_results.keys()) == set(frames)

    for frame in frames:
        seq_pos, seq_ids = seq_results[frame]
        par_pos, par_ids = par_results[frame]

        assert len(seq_pos) > 0, f"Frame {frame} had 0 correspondences"
        assert np.allclose(seq_pos, par_pos, atol=1e-12)
        assert np.array_equal(seq_ids, par_ids)


def test_batch_parallel_zarr_store(cavity_params, sample_targets, tmp_path):
    """Test match_correspondences_batch_parallel reading and writing to Zarr store."""
    exp, cpar, spar, vpar, cals = cavity_params
    targets_per_frame, frames = sample_targets
    num_cams = len(cals)

    zarr_dir = tmp_path / "run.zarr"
    store = RunStore(zarr_dir, mode="w")

    # Populate targets in Zarr store
    for frame in frames:
        for cam in range(num_cams):
            store.write_targets(cam, frame, targets_per_frame[frame][cam])

    # Run parallel correspondences reading from and writing to Zarr store
    par_results = match_correspondences_batch_parallel(
        frames=frames,
        cpar=cpar,
        cals=cals,
        vpar=vpar,
        zarr_store_path=str(zarr_dir),
        n_workers=2,
        write_to_store=True,
    )

    # Verify store content
    store_read = RunStore(zarr_dir, mode="r")
    for frame in frames:
        assert store_read.has_correspondences(frame)
        stored_pos, stored_ids = store_read.read_correspondences(frame)
        ret_pos, ret_ids = par_results[frame]

        assert np.allclose(stored_pos, ret_pos, atol=1e-12)
        assert np.array_equal(stored_ids, ret_ids)


def test_match_frame_empty_targets(cavity_params):
    """Test handling of empty / zero-target frames."""
    exp, cpar, spar, vpar, cals = cavity_params
    num_cams = len(cals)
    empty_detections = [TargetArray(0) for _ in range(num_cams)]

    pos, ids = match_frame_correspondences(empty_detections, cpar, cals, vpar)
    assert pos.shape == (0, 3)
    assert ids.shape == (0, max(4, num_cams))
