"""Module interface contract tests.

Verifies the data shapes and schemas at module boundaries:
- RunStore write/read round-trip preserves array shapes/dtypes
- Tracker output linkage has expected per-frame slot count
- Preset config dicts contain required keys
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


def _write_synthetic_scene(store_root: Path, n_frames: int = 4, n_particles: int = 6):
    store_root = Path(store_root)
    store_path = store_root / "res" / "run.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store = RunStore(store_path, mode="w")

    rng = np.random.default_rng(3)
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


class TestRunStoreContract:
    def test_targets_write_read_preserves_shape(self, tmp_path):
        store_path = tmp_path / "res" / "run.zarr"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store = RunStore(store_path, mode="w")
        n, ncols = 7, 8
        arr = np.zeros((n, ncols), dtype=float)
        arr[:, 0] = np.arange(n)
        store.write_targets(0, 10001, arr)
        assert store.has_targets(0, 10001)
        back = store.read_targets(0, 10001)
        assert len(back) == n
        # TargetArray is a list-like of per-target records.

    def test_correspondences_write_read_preserves_shape(self, tmp_path):
        store_path = tmp_path / "res" / "run.zarr"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store = RunStore(store_path, mode="w")
        pos = np.zeros((5, 3), dtype=float)
        cam_ids = np.zeros((5, 4), dtype=np.int32)
        store.write_correspondences(frame=10001, pos_3d=pos, cam_target_ids=cam_ids)
        assert store.has_correspondences(10001)
        back_pos, back_ids = store.read_correspondences(10001)
        assert back_pos.shape == (5, 3)
        assert back_ids.shape == (5, 4)

    def test_empty_targets_roundtrip(self, tmp_path):
        store_path = tmp_path / "res" / "run.zarr"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store = RunStore(store_path, mode="w")
        empty = np.empty((0, 8), dtype=float)
        store.write_targets(0, 10001, empty)
        back = store.read_targets(0, 10001)
        assert len(back) == 0


class TestTrackerContract:
    @pytest.mark.slow
    def test_linkage_output_matches_particle_count(self, tmp_path):
        store_path = _write_synthetic_scene(tmp_path, n_frames=4, n_particles=6)
        naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}

        spar = SequencePar(num_cams=4, first=10001, last=10004)
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

        for f in range(10001, 10004):
            assert store.has_linkage(f, "ptv_is")
            prev, nxt, _x = store.read_linkage(f, "ptv_is")
            assert len(prev) == 6
            assert len(nxt) == 6
            assert prev.dtype == np.int32 or prev.dtype == np.int64
            assert nxt.dtype == np.int32 or nxt.dtype == np.int64


class TestPresetConfigContract:
    def test_preset_configs_have_required_keys(self):
        from openptv2.tracking_presets import PRESET_CONFIGS
        required = {"track_mode", "flagNewParticles", "postprocess", "selected_tracking"}
        for name, cfg in PRESET_CONFIGS.items():
            assert required.issubset(cfg.keys()), f"preset {name} missing keys"

    def test_preset_config_values_sane(self):
        from openptv2.tracking_presets import PRESET_CONFIGS
        for name, cfg in PRESET_CONFIGS.items():
            assert cfg["track_mode"] in (0, 1), f"preset {name} bad track_mode"
            assert cfg["flagNewParticles"] in (True, False)
            assert cfg["postprocess"] in (True, False)
            assert isinstance(cfg["selected_tracking"], str)
