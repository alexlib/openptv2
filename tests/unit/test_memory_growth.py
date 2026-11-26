"""Memory regression test: RSS growth must be bounded across frames.

Tracks a long synthetic sequence and asserts memory does not grow
linearly with frame count (no per-frame leak).
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


def _write_synthetic_scene(store_root: Path, n_frames: int, n_particles: int = 20):
    store_root = Path(store_root)
    store_path = store_root / "res" / "run.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = RunStore(store_path, mode="w")

    rng = np.random.default_rng(5)
    positions = rng.uniform(-20, 20, size=(n_particles, 3)).astype(float)

    for f in range(10001, 10001 + n_frames):
        for cam in range(4):
            arr = np.zeros((n_particles, 8), dtype=float)
            arr[:, 0] = np.arange(n_particles)
            arr[:, 1] = positions[:, 0] * 0.1 + cam
            arr[:, 2] = positions[:, 1] * 0.1 + cam
            arr[:, 3] = 5
            arr[:, 4] = 5
            arr[:, 5] = 25
            arr[:, 6] = 1000
            arr[:, 7] = -1
            store.write_targets(cam, f, arr)
        cam_ids = np.tile(np.arange(n_particles, dtype=np.int32)[:, None], (1, 4))
        store.write_correspondences(frame=f, pos_3d=positions, cam_target_ids=cam_ids)

    return store_path


def _rss_mb() -> float:
    """Resident set size in MB, best-effort."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


@pytest.mark.slow
def test_memory_growth_bounded(tmp_path):
    """Tracking 200 frames must not leak memory linearly vs 50 frames."""
    store_path_short = _write_synthetic_scene(tmp_path / "short", n_frames=50)
    store_path_long = _write_synthetic_scene(tmp_path / "long", n_frames=200)

    naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}

    def _run_tracker(store_path: Path, n_frames: int) -> float:
        spar = SequencePar(num_cams=4, first=10001, last=10001 + n_frames - 1)
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
        store = RunStore(store_path, mode="a")
        tracker = Tracker(cpar, vpar, tpar, spar, cals, naming=naming, store=store)
        tracker.full_forward()
        return _rss_mb()

    rss_short = _run_tracker(store_path_short, 50)
    rss_long = _run_tracker(store_path_long, 200)

    # Memory per frame should not grow: long/short ratio of frames is 4x,
    # so RSS ratio must be well under 4x (allowing baseline overhead).
    if rss_short > 0 and rss_long > 0:
        ratio = rss_long / rss_short
        assert ratio < 3.0, f"RSS grew {ratio:.2f}x for 4x frames — possible leak"
