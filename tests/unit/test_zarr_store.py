"""Unit tests for ZarrFrameStore and Flowtracks HDF5 export."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from openptv2.algorithms.tracking_frame_buf import TargetArray
from openptv2.storage import ZarrFrameStore


def test_zarr_store_targets_roundtrip(tmp_path):
    """Test target array write and read roundtrip in ZarrFrameStore."""
    zarr_path = tmp_path / "test_run.zarr"
    store = ZarrFrameStore(zarr_path, mode="w")

    # Create dummy target array
    tarr = TargetArray(2)
    tarr[0].set_pnr(10)
    tarr[0].set_pos((100.5, 200.5))
    tarr[0].set_pixel_counts(5, 2, 3)
    tarr[0].set_sum_grey_value(120)
    tarr[0].set_tnr(1)

    tarr[1].set_pnr(11)
    tarr[1].set_pos((150.2, 250.2))
    tarr[1].set_pixel_counts(8, 3, 3)
    tarr[1].set_sum_grey_value(200)
    tarr[1].set_tnr(2)

    # Write targets
    store.write_targets(cam_idx=0, frame=10000, targets=tarr)

    assert store.has_targets(cam_idx=0, frame=10000)
    assert not store.has_targets(cam_idx=0, frame=10001)

    # Read targets back
    read_tarr = store.read_targets(cam_idx=0, frame=10000)
    assert len(read_tarr) == 2
    assert read_tarr[0].pnr() == 10
    assert np.isclose(read_tarr[0].pos()[0], 100.5)
    assert np.isclose(read_tarr[0].pos()[1], 200.5)
    assert read_tarr[0].count_pixels() == (5, 2, 3)
    assert read_tarr[0].sum_grey_value() == 120
    assert read_tarr[0].tnr() == 1


def test_zarr_store_correspondences_roundtrip(tmp_path):
    """Test 3D correspondences write and read roundtrip in ZarrFrameStore."""
    zarr_path = tmp_path / "test_run.zarr"
    store = ZarrFrameStore(zarr_path, mode="w")

    pos_3d = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    cam_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)

    store.write_correspondences(frame=10000, pos_3d=pos_3d, cam_target_ids=cam_ids)

    read_pos, read_cams = store.read_correspondences(frame=10000)
    assert np.allclose(read_pos, pos_3d)
    assert np.array_equal(read_cams, cam_ids)


def test_zarr_store_to_flowtracks_h5(tmp_path):
    """Test exporting Zarr trajectory store to Flowtracks HDF5 format."""
    zarr_path = tmp_path / "test_run.zarr"
    h5_path = tmp_path / "flowtracks_out.h5"

    store = ZarrFrameStore(zarr_path, mode="w")

    pos = np.array([[1000.0, 2000.0, 3000.0]], dtype=np.float64)  # mm
    vel = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)  # mm/s
    frames = np.array([10000], dtype=np.int32)
    traj_ids = np.array([1], dtype=np.int32)

    store.write_trajectories(pos=pos, vel=vel, frames=frames, traj_ids=traj_ids)
    store.to_flowtracks_h5(h5_path)

    assert h5_path.exists()

    with h5py.File(h5_path, "r") as h5f:
        assert "pos" in h5f
        assert np.isclose(h5f["pos"][0, 0], 1.0)  # 1000 mm -> 1.0 m for Flowtracks
        assert np.isclose(h5f["vel"][0, 0], 0.01)  # 10 mm/s -> 0.01 m/s
        assert h5f["frame"][0] == 10000
        assert h5f["trajid"][0] == 1


def test_parallel_detection_with_zarr_store(tmp_path):
    """Verify parallel preprocessing writes targets directly to Zarr store."""
    import os
    import shutil
    from unittest.mock import Mock

    from openptv2.gui import ptv
    from openptv2.gui.experiment import Experiment

    cavity_src = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not cavity_src.exists():
        pytest.skip("test_cavity directory not found")

    temp_dir = tmp_path / "test_cavity"
    shutil.copytree(cavity_src, temp_dir)

    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yaml_file = temp_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.pm.from_yaml(yaml_file)
        exp.target_filenames = exp.pm.get_target_filenames()

        cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(exp.pm)
        exp_mock = Mock()
        exp_mock.pm = exp.pm
        exp_mock.num_cams = exp.pm.num_cams
        exp_mock.cpar = cpar
        exp_mock.spar = spar
        exp_mock.vpar = vpar
        exp_mock.track_par = track_par
        exp_mock.tpar = tpar
        exp_mock.cals = cals
        exp_mock.target_filenames = exp.target_filenames

        spar.set_first(10000)
        spar.set_last(10001)

        zarr_path = temp_dir / "targets_test.zarr"
        ptv.preprocess_and_detect_all_parallel(
            exp_mock, num_workers=2, zarr_store_path=str(zarr_path)
        )

        store = ZarrFrameStore(zarr_path, mode="r")
        assert store.has_targets(cam_idx=0, frame=10000)
        tarr = store.read_targets(cam_idx=0, frame=10000)
        assert len(tarr) > 0
    finally:
        os.chdir(old_cwd)


def test_zarr_store_export_frame_text(tmp_path):
    """Test formatting binary Zarr frame data into human-readable ASCII text."""
    zarr_path = tmp_path / "test_inspect.zarr"
    store = ZarrFrameStore(zarr_path, mode="w")

    tarr = TargetArray(1)
    tarr[0].set_pnr(1)
    tarr[0].set_pos((12.3456, 78.9012))
    tarr[0].set_pixel_counts(10, 3, 3)
    tarr[0].set_sum_grey_value(255)
    tarr[0].set_tnr(0)

    store.write_targets(cam_idx=0, frame=10000, targets=tarr)

    text_out = store.export_frame_text(frame=10000, dataset_type="targets", cam_idx=0)
    assert "1" in text_out  # count line
    assert "12.3456" in text_out
    assert "78.9012" in text_out


def test_py_sequence_loop_writes_through_run_store(tmp_path):
    """py_sequence_loop unconditionally dual-writes into the unified RunStore
    (Phase B: OPENPTV_STORAGE was retired in favor of an explicit store
    parameter threaded from _open_run_store(exp); no env var toggles this
    anymore)."""
    import os
    import shutil

    from openptv2.gui import ptv
    from openptv2.gui.experiment import Experiment
    from openptv2.storage import RunStore

    cavity_src = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not cavity_src.exists():
        pytest.skip("test_cavity directory not found")

    temp_dir = tmp_path / "test_cavity"
    shutil.copytree(cavity_src, temp_dir)

    old_cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yaml_file = temp_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.exp_path = str(temp_dir)
        exp.pm.from_yaml(yaml_file)
        exp.target_filenames = exp.pm.get_target_filenames()

        cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(exp.pm)
        exp.cpar = cpar
        exp.spar = spar
        exp.vpar = vpar
        exp.track_par = track_par
        exp.tpar = tpar
        exp.cals = cals

        spar.set_first(10000)
        spar.set_last(10001)

        ptv.py_sequence_loop(exp)

        # RunStore.open prefers an existing store (the committed fixture
        # store at the dataset root) over creating a fresh res/run.zarr --
        # read back whatever store the pipeline actually used.
        store = RunStore.open(temp_dir, mode="r")
        assert store.has_targets(cam=0, frame=10000)

        pos_3d, cam_ids = store.read_correspondences(10000)
        assert len(pos_3d) > 0
    finally:
        os.chdir(old_cwd)


def test_read_zarr_trajectories(tmp_path):
    """Test read_zarr_trajectories on both linkage and trajectories groups."""
    from openptv2.storage import ZarrFrameStore, read_zarr_trajectories

    zarr_path = tmp_path / "test_traj.zarr"
    store = ZarrFrameStore(zarr_path, mode="w")

    # Write linkage for two frames
    # Frame 1: 2 particles
    p1_pos = np.array([[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]])  # mm
    prev1 = np.array([-1, -1])
    next1 = np.array([0, 1])
    store.write_linkage(frame=1, prev_ids=prev1, next_ids=next1, pos_3d=p1_pos)

    # Frame 2: 2 particles
    p2_pos = np.array([[12.0, 22.0, 32.0], [102.0, 202.0, 302.0]])  # mm
    prev2 = np.array([0, 1])
    next2 = np.array([-1, -1])
    store.write_linkage(frame=2, prev_ids=prev2, next_ids=next2, pos_3d=p2_pos)

    trajs = read_zarr_trajectories(zarr_path)
    assert len(trajs) == 2
    # Check positions in flowtracks Trajectory objects are in meters
    p0 = trajs[0].pos()
    np.testing.assert_allclose(p0[0], [0.010, 0.020, 0.030])
    np.testing.assert_allclose(p0[1], [0.012, 0.022, 0.032])
