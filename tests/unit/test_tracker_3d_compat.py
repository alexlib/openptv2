import os
import shutil
from pathlib import Path

import numpy as np

from openptv2.calibration import Calibration
from openptv2.parameters import (
    ControlParams,
    SequenceParams,
    TrackingParams,
    VolumeParams,
)
from openptv2.storage import RunStore
from openptv2.tracker import Tracker


def _seed_track_store(store: RunStore, base_dir: Path, num_cams: int) -> None:
    for rt_file in sorted((base_dir / "res_orig").glob("rt_is.*")):
        frame = int(rt_file.suffix.lstrip("."))
        lines = [ln.strip() for ln in rt_file.read_text().splitlines() if ln.strip()]
        n = int(lines[0]) if lines else 0
        if n > 0:
            rows = np.loadtxt(lines[1 : n + 1], dtype=np.float64).reshape(n, 8)
            pos_3d = rows[:, 1:4]
            cam_ids = rows[:, 4:8].astype(np.int32)
        else:
            pos_3d = np.empty((0, 3), dtype=np.float64)
            cam_ids = np.empty((0, 4), dtype=np.int32)
        store.write_correspondences(frame=frame, pos_3d=pos_3d, cam_target_ids=cam_ids)

    for cam in range(num_cams):
        for tfile in sorted((base_dir / "img_orig").glob(f"cam{cam + 1}.*_targets")):
            frame = int(tfile.name.split(".")[1].split("_")[0])
            lines = [ln.strip() for ln in tfile.read_text().splitlines() if ln.strip()]
            n = int(lines[0]) if lines else 0
            if n > 0:
                arr = np.loadtxt(lines[1 : n + 1], dtype=np.float64).reshape(n, 8)
            else:
                arr = np.empty((0, 8), dtype=np.float64)
            store.write_targets(cam, frame, arr)


def read_all_calibration(num_cams, base_path="test_data/track"):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration()
        cal.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


def test_tracker_3d_compat_loop(tmp_path, monkeypatch):
    test_dir = Path(__file__).resolve().parent / "../../test_data/track"
    work_dir = tmp_path / "track"
    shutil.copytree(test_dir, work_dir)
    monkeypatch.chdir(work_dir)

    # Read control parameters to find correct number of cameras
    cpar = ControlParams(num_cams=2)
    cpar.read_control_par("parameters/ptv.par")
    num_cams = cpar.get_num_cams()

    vpar = VolumeParams()
    vpar.read_volume_par("parameters/criteria.par")

    tpar = TrackingParams()
    tpar.read_track_par("parameters/track.par")
    tpar.set_track_mode(1)
    tpar.set_add(0)

    spar = SequenceParams(num_cams=num_cams)
    spar.read_sequence_par("parameters/sequence.par", num_cams)

    cals = read_all_calibration(num_cams, base_path=".")
    store = RunStore(work_dir / "res" / "run.zarr", mode="w")
    _seed_track_store(store, work_dir, num_cams)

    tracker = Tracker(cpar, vpar, tpar, spar, cals, store=store)
    tracker.full_forward_3d()

    assert tracker._run is not None
    assert tracker._run.npart > 0
    assert tracker._run.nlinks > 0


def test_tracker_3d_compat_step_forward(tmp_path, monkeypatch):
    test_dir = Path(__file__).resolve().parent / "../../test_data/track"
    work_dir = tmp_path / "track"
    shutil.copytree(test_dir, work_dir)
    monkeypatch.chdir(work_dir)

    cpar = ControlParams(num_cams=2)
    cpar.read_control_par("parameters/ptv.par")
    num_cams = cpar.get_num_cams()

    vpar = VolumeParams()
    vpar.read_volume_par("parameters/criteria.par")

    tpar = TrackingParams()
    tpar.read_track_par("parameters/track.par")
    tpar.set_track_mode(1)
    tpar.set_add(0)

    spar = SequenceParams(num_cams=num_cams)
    spar.read_sequence_par("parameters/sequence.par", num_cams)

    cals = read_all_calibration(num_cams, base_path=".")
    store = RunStore(work_dir / "res" / "run.zarr", mode="w")
    _seed_track_store(store, work_dir, num_cams)

    tracker = Tracker(cpar, vpar, tpar, spar, cals, store=store)
    tracker.restart()

    has_more = True
    steps = 0
    while has_more:
        has_more = tracker.step_forward_3d()
        steps += 1

    assert steps > 0
    assert tracker._run.npart > 0
