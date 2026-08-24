"""Unit tests for detect_targets_batch_parallel."""

import os
import shutil
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from openptv2 import detect_targets_batch_parallel, target_recognition
from openptv2.algorithms.segmentation import targ_rec
from openptv2.gui import ptv
from openptv2.gui.experiment import Experiment
from openptv2.storage import ZarrFrameStore


@pytest.fixture
def cavity_images(tmp_path):
    """Fixture providing sample cavity dataset images and parameters."""
    cavity_src = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not cavity_src.exists():
        pytest.skip("test_cavity dataset not found")

    temp_dir = tmp_path / "test_cavity"
    shutil.copytree(cavity_src, temp_dir)

    yaml_file = temp_dir / "parameters_Run1.yaml"
    exp = Experiment()
    exp.pm.from_yaml(yaml_file)
    cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(exp.pm)

    # Collect image paths for 4 cameras across 5 frames (20 images total)
    img_paths = []
    cam_indices = []
    frame_indices = []
    for frame in range(10000, 10005):
        for i_cam in range(exp.pm.num_cams):
            imname = temp_dir / (spar.get_img_base_name(i_cam) % frame)
            if imname.exists():
                img_paths.append(str(imname))
                cam_indices.append(i_cam)
                frame_indices.append(frame)

    return {
        "dir": temp_dir,
        "img_paths": img_paths,
        "cam_indices": cam_indices,
        "frame_indices": frame_indices,
        "tpar": tpar,
        "cpar": cpar,
    }


def test_batch_parallel_parity_with_image_paths(cavity_images):
    """Verify bit-exact target detection parity between serial and parallel runs with image paths."""
    img_paths = cavity_images["img_paths"]
    cam_indices = cavity_images["cam_indices"]
    tpar = cavity_images["tpar"]

    # 1. Serial execution baseline (n_workers=1)
    res_serial = detect_targets_batch_parallel(
        img_paths,
        tpar,
        n_workers=1,
        cam_indices=cam_indices,
        return_type="arrays",
    )

    # 2. Parallel execution (n_workers=4)
    res_parallel = detect_targets_batch_parallel(
        img_paths,
        tpar,
        n_workers=4,
        cam_indices=cam_indices,
        return_type="arrays",
    )

    assert len(res_serial) == len(res_parallel)
    for i in range(len(res_serial)):
        s = res_serial[i]
        p = res_parallel[i]
        assert s["n_found"] == p["n_found"]
        if s["n_found"] > 0:
            np.testing.assert_allclose(s["x"], p["x"])
            np.testing.assert_allclose(s["y"], p["y"])
            np.testing.assert_array_equal(s["n"], p["n"])
            np.testing.assert_array_equal(s["nx"], p["nx"])
            np.testing.assert_array_equal(s["ny"], p["ny"])
            np.testing.assert_array_equal(s["sumg"], p["sumg"])


def test_batch_parallel_with_3d_shared_memory():
    """Verify 3D array detection with multiprocessing SharedMemory."""
    rng = np.random.RandomState(42)
    n_frames = 12
    height, width = 128, 128
    images = np.zeros((n_frames, height, width), dtype=np.uint8)

    # Inject discrete bright spots
    for f in range(n_frames):
        for _ in range(5):
            cx = rng.randint(20, width - 20)
            cy = rng.randint(20, height - 20)
            images[f, cy - 1 : cy + 2, cx - 1 : cx + 2] = 200

    params = {
        "gvthres": 50,
        "discont": 20,
        "nnmin": 1,
        "nnmax": 50,
        "nxmin": 1,
        "nxmax": 20,
        "nymin": 1,
        "nymax": 20,
        "sumg_min": 100,
    }

    # Serial
    targets_seq = detect_targets_batch_parallel(
        images,
        params,
        n_workers=1,
        use_shared_memory=False,
        return_type="targets",
    )

    # Parallel with SharedMemory
    targets_shm = detect_targets_batch_parallel(
        images,
        params,
        n_workers=2,
        use_shared_memory=True,
        return_type="targets",
    )

    assert len(targets_seq) == len(targets_shm) == n_frames
    for f in range(n_frames):
        assert len(targets_seq[f]) == len(targets_shm[f])
        for t_s, t_p in zip(targets_seq[f], targets_shm[f]):
            assert np.isclose(t_s.x, t_p.x)
            assert np.isclose(t_s.y, t_p.y)
            assert t_s.sumg == t_p.sumg


def test_batch_parallel_empty_images():
    """Verify handling of completely black images (no peaks)."""
    empty_imgs = np.zeros((4, 64, 64), dtype=np.uint8)
    params = {"gvthres": 50}

    counts = detect_targets_batch_parallel(
        empty_imgs, params, n_workers=2, return_type="counts"
    )
    assert counts == [0, 0, 0, 0]

    targets = detect_targets_batch_parallel(
        empty_imgs, params, n_workers=2, return_type="targets"
    )
    assert len(targets) == 4
    # Empty default target returned
    for tlist in targets:
        assert len(tlist) == 1
        assert tlist[0].pnr == 1


def test_batch_parallel_zarr_output(tmp_path):
    """Verify writing directly to a ZarrFrameStore."""
    n_frames = 4
    height, width = 64, 64
    images = np.zeros((n_frames, height, width), dtype=np.uint8)
    images[:, 30:33, 30:33] = 220

    zarr_path = tmp_path / "targets_test.zarr"
    params = {"gvthres": 50, "nnmin": 1, "nnmax": 50, "sumg_min": 100}

    detect_targets_batch_parallel(
        images,
        params,
        n_workers=2,
        zarr_store_path=str(zarr_path),
        cam_indices=[0, 1, 0, 1],
        frame_indices=[100, 100, 101, 101],
    )

    store = ZarrFrameStore(zarr_path, mode="r")
    assert store.has_targets(cam_idx=0, frame=100)
    assert store.has_targets(cam_idx=1, frame=100)
    assert store.has_targets(cam_idx=0, frame=101)
    assert store.has_targets(cam_idx=1, frame=101)

    t0_100 = store.read_targets(cam_idx=0, frame=100)
    assert len(t0_100) == 1
    assert np.isclose(t0_100[0].x, 31.5)
    assert np.isclose(t0_100[0].y, 31.5)
