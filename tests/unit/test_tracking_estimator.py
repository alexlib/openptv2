"""Unit tests for openptv2.tracking_estimator (Step 1b)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openptv2.storage.run_store import RunStore
from openptv2.tracking_estimator import estimate_params, make_synthetic_exp


def _write_synthetic_correspondences(
    store: RunStore,
    n_frames: int,
    *,
    n_real: int = 40,
    n_ghosts: int = 0,
    step_xyz: tuple[float, float, float] = (2.0, 0.0, 0.0),
    seed: int = 0,
    first_frame: int = 1,
) -> None:
    rng = np.random.default_rng(seed)
    # Place real particles well separated on a loose 3D grid + small jitter so
    # inter-particle spacing (~20 mm) >> step_xyz.
    base = rng.uniform(-100.0, 100.0, size=(n_real, 3))
    step = np.asarray(step_xyz, dtype=float)
    num_cams = 4

    for offset in range(n_frames):
        fn = first_frame + offset
        real_pos = base + offset * step
        if n_ghosts > 0:
            # Place ghosts far from real particles so they have no successor
            # within v_max.
            ghost_pos = rng.uniform(500.0, 1000.0, size=(n_ghosts, 3))
            pos = np.vstack([real_pos, ghost_pos])
        else:
            pos = real_pos
        cam_ids = np.zeros((len(pos), num_cams), dtype=np.int32)
        store.write_correspondences(fn, pos, cam_ids)


def test_estimate_params_synthetic_displacement_density_and_ghost_rate(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.zarr", mode="w")
    # 10 frames, 40 real particles moving 2.0 mm/frame + 10 ghosts/frame (20%).
    _write_synthetic_correspondences(
        store,
        n_frames=10,
        n_real=40,
        n_ghosts=10,
        step_xyz=(2.0, 0.0, 0.0),
        seed=42,
        first_frame=1,
    )
    exp = make_synthetic_exp(store, first=1, last=10)

    res = estimate_params(exp)

    assert res["density"] == pytest.approx(50.0)
    assert res["density_max"] == 50
    assert res["displacement"] == pytest.approx(2.0, rel=0.05)
    assert res["v_max"] == pytest.approx(2.0 * 1.5, rel=0.05)
    # 10 out of 50 particles per frame are ghosts with no successor within v_max.
    assert res["ghost_rate"] == pytest.approx(0.20, abs=0.05)
    assert res["noise_floor_z"] is None
    assert res["conditioning_ratio"] is None
    assert res["slow_flow"] is False
    assert res["windows_sampled"] == [(1, 10)]


def test_estimate_params_samples_multiple_windows_on_long_sequence(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "long.zarr", mode="w")
    _write_synthetic_correspondences(
        store,
        n_frames=200,
        n_real=25,
        n_ghosts=0,
        step_xyz=(1.5, 0.0, 0.0),
        seed=7,
        first_frame=1,
    )
    exp = make_synthetic_exp(store, first=1, last=200)

    res = estimate_params(exp)

    assert len(res["windows_sampled"]) == 5
    assert res["density"] == pytest.approx(25.0)
    assert res["v_max"] == pytest.approx(1.5 * 1.5, rel=0.05)
    assert res["ghost_rate"] == pytest.approx(0.0, abs=0.02)


def test_estimate_params_caps_v_max_when_rho_exceeds_one(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "dense_fast.zarr", mode="w")
    rng = np.random.default_rng(123)
    # Dense cluster with small inter-particle distance (~2-3 mm) but fast
    # displacement (15 mm/frame) -> rho = p95_disp / mean_ipd > 1.0.
    base = rng.uniform(0.0, 10.0, size=(50, 3))
    for fn in range(1, 6):
        pos = base + (fn - 1) * np.array([15.0, 0.0, 0.0])
        store.write_correspondences(fn, pos, np.zeros((50, 4), dtype=np.int32))

    exp = make_synthetic_exp(store, first=1, last=5)
    res = estimate_params(exp)

    assert res["rho"] is not None and res["rho"] > 1.0
    assert res["mean_interparticle_distance"] is not None
    assert res["v_max"] <= res["mean_interparticle_distance"] + 1e-9


def test_estimate_params_slow_flow_high_noise_conditioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "slow.zarr", mode="w")
    _write_synthetic_correspondences(
        store,
        n_frames=6,
        n_real=30,
        n_ghosts=0,
        step_xyz=(0.25, 0.0, 0.0),
        seed=9,
        first_frame=1,
    )
    exp = make_synthetic_exp(
        store,
        first=1,
        last=6,
        cals=[object()],
        cpar=SimpleNamespace(),
    )

    monkeypatch.setattr(
        "openptv2.tracking_estimator.z_noise_floor_mm",
        lambda *a, **k: 0.5,
    )
    res = estimate_params(exp)

    assert res["noise_floor_z"] == pytest.approx(0.5)
    assert res["conditioning_ratio"] == pytest.approx(0.25 / 0.5, rel=0.05)
    assert res["conditioning_ratio"] < 1.0
    assert res["slow_flow"] is True

    # Low noise floor -> well-conditioned -> slow_flow=False
    monkeypatch.setattr(
        "openptv2.tracking_estimator.z_noise_floor_mm",
        lambda *a, **k: 0.1,
    )
    res_fast = estimate_params(exp)
    assert res_fast["conditioning_ratio"] > 1.0
    assert res_fast["slow_flow"] is False


def test_estimate_params_user_overrides_win_over_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "override.zarr", mode="w")
    _write_synthetic_correspondences(
        store,
        n_frames=6,
        n_real=30,
        n_ghosts=0,
        step_xyz=(0.2, 0.0, 0.0),
        seed=11,
        first_frame=1,
    )
    exp = make_synthetic_exp(
        store,
        first=1,
        last=6,
        cals=[object()],
        cpar=SimpleNamespace(),
    )
    monkeypatch.setattr(
        "openptv2.tracking_estimator.z_noise_floor_mm",
        lambda *a, **k: 1.0,
    )

    # Explicit overrides win over auto (auto would give v_max~0.3, slow_flow=True, density=30).
    user_cfg = {
        "v_max": 7.5,
        "slow_flow": False,
        "density_estimate": 420,
    }
    user_cfg_snapshot = dict(user_cfg)
    res = estimate_params(exp, track_cfg=user_cfg)

    assert res["v_max"] == pytest.approx(7.5)
    assert res["slow_flow"] is False
    assert res["density"] == pytest.approx(420.0)
    # Pure / read-only: caller's dict was not mutated.
    assert user_cfg == user_cfg_snapshot

    # "auto" / None / density_estimate=0 do NOT override measured estimates.
    res_auto = estimate_params(
        exp,
        track_cfg={"v_max": "auto", "slow_flow": None, "density_estimate": 0},
    )
    assert res_auto["v_max"] == pytest.approx(0.2 * 1.5, rel=0.05)
    assert res_auto["slow_flow"] is True
    assert res_auto["density"] == pytest.approx(30.0)


def test_estimate_params_empty_experiment_is_safe() -> None:
    res = estimate_params(None)
    assert res["v_max"] is None
    assert res["density"] == 0.0
    assert res["slow_flow"] is False
    assert res["ghost_rate"] is None
    assert res["windows_sampled"] == []
