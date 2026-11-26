"""Determinism tests: identical inputs must produce identical outputs.

Runs the tracker twice on the same synthetic scene and asserts identical
linkage output. Also verifies RunStore write/read round-trip determinism.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.calibration import Calibration
from openptv2.storage import RunStore
from openptv2.tracker import Tracker


def _make_minimal_cals(num_cams: int = 4) -> list[Calibration]:
    cals: list[Calibration] = []
    for i in range(num_cams):
        cal = Calibration()
        cal.set_pos([float(i * 100), 0.0, 500.0])
        cal.set_angles([0.1, -0.1, 0.5])
        cal.set_primary_point([0.00655 * 640, 0.00655 * 512, 10.0])
        cal.set_radial_distortion([0.0, 0.0, 0.0])
        cal.set_decentering([0.0, 0.0])
        cal.set_affine_trans([1.0, 0.0])
        cal.set_glass_vec([0.0, 0.0, 0.0])
        cals.append(cal)
    return cals


def _write_synthetic_scene(store_root: Path, n_frames: int = 5, n_particles: int = 10):
    """Write targets + correspondences into a RunStore, return the store path."""
    store_root = Path(store_root)
    store_path = store_root / "res" / "run.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = RunStore(store_path, mode="w")

    rng = np.random.default_rng(7)
    positions = rng.uniform(-20, 20, size=(n_particles, 3)).astype(float)

    for f in range(10001, 10001 + n_frames):
        # targets per camera
        for cam in range(4):
            arr = np.zeros((n_particles, 8), dtype=float)
            arr[:, 0] = np.arange(n_particles)  # pnr
            arr[:, 1] = positions[:, 0] * 0.1 + cam  # x pixel-ish
            arr[:, 2] = positions[:, 1] * 0.1 + cam  # y pixel-ish
            arr[:, 3] = 5
            arr[:, 4] = 5
            arr[:, 5] = 25
            arr[:, 6] = 1000
            arr[:, 7] = -1  # tnr
            store.write_targets(cam, f, arr)
        # correspondences
        cam_ids = np.tile(np.arange(n_particles, dtype=np.int32)[:, None], (1, 4))
        store.write_correspondences(frame=f, pos_3d=positions, cam_target_ids=cam_ids)

    return store_path


def _make_tracker(store: RunStore, spar: SequencePar, cpar: ControlPar,
                  tpar: TrackPar, vpar: VolumePar, cals: list[Calibration],
                  naming: dict) -> Tracker:
    return Tracker(
        cpar=cpar,
        vpar=vpar,
        tpar=tpar,
        spar=spar,
        cals=cals,
        naming=naming,
        store=store,
    )


@pytest.mark.slow
def test_tracker_deterministic_forward(tmp_path):
    """Two identical tracker runs produce identical linkage output."""
    store_path = _write_synthetic_scene(tmp_path, n_frames=5, n_particles=8)
    naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}

    spar = SequencePar(num_cams=4, first=10001, last=10005)
    cpar = ControlPar(num_cams=4, imx=1024, imy=1024, pix_x=0.01, pix_y=0.01)
    tpar = TrackPar(
        dvxmin=-15.5, dvxmax=15.5, dvymin=-15.5, dvymax=15.5,
        dvzmin=-15.5, dvzmax=15.5, dangle=120.0, dacc=5.5, add=0, track_mode=1,
    )
    vpar = VolumePar(
        X_lay=[-100.0, 100.0], Zmin_lay=[-100.0, -100.0], Zmax_lay=[100.0, 100.0],
        cnx=0.5, cny=0.5, cn=0.5, csumg=0.0, corrmin=0.0, eps0=0.15,
    )
    cals = _make_minimal_cals()

    results = []
    for _ in range(2):
        store = RunStore(store_path, mode="a")
        tracker = _make_tracker(store, spar, cpar, tpar, vpar, cals, naming)
        tracker.full_forward()
        frames = []
        for f in range(10001, 10005):
            prev, nxt, _x = store.read_linkage(f, "ptv_is")
            frames.append((prev.copy(), nxt.copy()))
        results.append(frames)

    for f_idx in range(len(results[0])):
        for slot in range(len(results[0][f_idx][0])):
            assert results[0][f_idx][0][slot] == results[1][f_idx][0][slot]
            assert results[0][f_idx][1][slot] == results[1][f_idx][1][slot]


@pytest.mark.slow
def test_runstore_roundtrip_deterministic(tmp_path):
    """Write then read correspondence data is byte-identical across runs."""
    store_path = _write_synthetic_scene(tmp_path, n_frames=3, n_particles=8)

    snapshots = []
    for _ in range(2):
        store = RunStore(store_path, mode="a")
        clouds = []
        for f in range(10001, 10004):
            if store.has_correspondences(f):
                pos, ids = store.read_correspondences(f)
                clouds.append((pos.copy(), ids.copy()))
        snapshots.append(clouds)

    assert len(snapshots[0]) == len(snapshots[1])
    for (p0, i0), (p1, i1) in zip(snapshots[0], snapshots[1]):
        assert p0.shape == p1.shape
        assert i0.shape == i1.shape
        assert np.allclose(p0, p1)
        assert np.array_equal(i0, i1)
