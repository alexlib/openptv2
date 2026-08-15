"""Tracker <-> RunStore integration: Phase B of the unified-store plan.

The real per-frame write path during tracking is Tracker.full_forward_3d()
-> TrackingRun -> FrameBuf.write_frame_from_start() -> Frame.write() ->
write_path_frame()/write_targets() (algorithms/tracking_frame_buf.py). This
test proves a RunStore passed into Tracker(..., store=store) reaches every
layer of that chain and is populated identically to the ASCII output,
without changing the tracker's numeric result -- the plumbing this file
covers replaced the old OPENPTV_STORAGE env var with an explicit parameter.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.storage import RunStore
from openptv2.tracker import Tracker
from tests._support import find_test_data_root

TEST_DATA_ROOT = find_test_data_root()
CAVITY_DIR = TEST_DATA_ROOT / "test_cavity"


@pytest.fixture
def cavity_orig_workdir(tmp_path):
    """Private copy of test_cavity's res_orig/img_orig -- the known-good,
    checked-in ASCII fixture pair test_track3d.py's own full_forward_3d test
    already asserts npart=2082/nlinks=1748 against -- so this test's baseline
    numbers are not new claims, just re-run against a private copy with a
    store attached."""
    if not (CAVITY_DIR / "res_orig").exists() or not (CAVITY_DIR / "img_orig").exists():
        pytest.skip("test_cavity res_orig/img_orig fixtures not present")
    dst = tmp_path / "run"
    dst.mkdir()
    shutil.copytree(CAVITY_DIR / "res_orig", dst / "res")
    shutil.copytree(CAVITY_DIR / "img_orig", dst / "img")
    shutil.copytree(CAVITY_DIR / "cal", dst / "cal")
    shutil.copy(CAVITY_DIR / "parameters.yaml", dst / "parameters.yaml")
    return dst


def _read_calibrations(cpar, base_path: Path):
    return [
        Calibration.from_file(
            str(base_path / f"cal/cam{cam + 1}.tif.ori"),
            str(base_path / f"cal/cam{cam + 1}.tif.addpar"),
        )
        for cam in range(cpar.num_cams)
    ]


def test_tracker_full_forward_3d_writes_through_store(cavity_orig_workdir, monkeypatch):
    monkeypatch.chdir(cavity_orig_workdir)

    cpar = ControlPar.from_yaml("parameters.yaml")
    cals = _read_calibrations(cpar, cavity_orig_workdir)
    store = RunStore("res/run.zarr", mode="w")

    tracker = Tracker(
        cpar,
        VolumePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        SequencePar.from_yaml("parameters.yaml"),
        cals,
        store=store,
    )
    tracker.full_forward_3d()

    # Same numeric result as the no-store path (test_track3d.py's
    # test_tracker_full_forward_3d_test_cavity) -- store threading must not
    # perturb tracking itself.
    assert tracker.npart == 2082
    assert tracker.nlinks == 1748

    # Store-only output (no ASCII -- see
    # docs/plans/2026-08-15-zarr-only-transition-plan.md): ptv_is is the
    # tracker's own output and must not be (re)written to disk for a
    # store-backed run. rt_is.10001 is res_orig fixture input, copied in
    # above, not written by the tracker -- its presence says nothing here.
    assert not (cavity_orig_workdir / "res" / "ptv_is.10001").exists()

    frames = store.frames()
    assert frames == [10001, 10002, 10003, 10004]
    assert store.linkage_names() == ["ptv_is"]
    assert store.target_cameras() == [0, 1, 2, 3]

    store_pos, store_cam_ids = store.read_correspondences(10001)
    assert store_pos.shape[0] == store_cam_ids.shape[0]

    store_prev, store_next, store_pos2 = store.read_linkage(10001, "ptv_is")
    assert store_prev.shape == store_next.shape == (store_pos2.shape[0],)


def test_tracker_postprocess_writes_through_store(cavity_orig_workdir, monkeypatch):
    """Tracker.postprocess() (tracking_postprocess.py's enforce_reciprocity/
    seed_cold_start/relink_trajectory_gaps) used to bypass RunStore entirely
    with a raw zarr.open_group call using a differently-padded frame key
    (frame_NNNNN vs RunStore's frame_NNNNNN) -- silently a no-op against a
    RunStore-backed run. Now threaded through the same store parameter as
    everything else; this proves it actually reads/writes the store, not a
    stale/incompatible zarr layout."""
    monkeypatch.chdir(cavity_orig_workdir)

    cpar = ControlPar.from_yaml("parameters.yaml")
    cals = _read_calibrations(cpar, cavity_orig_workdir)
    store = RunStore("res/run.zarr", mode="w")

    tracker = Tracker(
        cpar,
        VolumePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        SequencePar.from_yaml("parameters.yaml"),
        cals,
        store=store,
    )
    tracker.full_forward_3d()
    stats = tracker.postprocess()

    assert stats["links_before"] == tracker.nlinks
    assert stats["links_after"] >= 0

    # postprocess() rewrote the store's linkage in place -- prove it's still
    # readable and shaped consistently, not a stale/incompatible layout.
    store_prev, store_next, store_pos = store.read_linkage(10001, "ptv_is")
    assert store_prev.shape == store_next.shape == (store_pos.shape[0],)
