"""Unit tests for openptv2.plugins.unified_tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openptv2.plugins.unified_tracking import Tracking
from openptv2.storage.run_store import RunStore
from openptv2.tracking_decision import TrackingDecision
from openptv2.tracking_estimator import make_synthetic_exp


def test_auto_regime_defaults():
    plugin = Tracking()
    regime = plugin._auto_regime(None, {})
    assert regime == {
        "bidirectional": True,
        "use_2d": True,
        "slow_flow": False,
        "delta_t": 1,
        "confirm": None,
    }


def test_auto_regime_density_thresholds():
    plugin = Tracking()
    # Moderate-low density (<500): use_2d off, bidirectional on
    r_low = plugin._auto_regime(None, {"density_estimate": 200})
    assert r_low["bidirectional"] is True
    assert r_low["use_2d"] is False

    # Dense (>=500, <=5000): both on
    r_mid = plugin._auto_regime(None, {"density_estimate": 1000})
    assert r_mid["bidirectional"] is True
    assert r_mid["use_2d"] is True

    # Extreme density (>5000): bidirectional off, use_2d on
    r_high = plugin._auto_regime(None, {"density_estimate": 6000})
    assert r_high["bidirectional"] is False
    assert r_high["use_2d"] is True


def test_auto_regime_slow_flow_and_overrides():
    plugin = Tracking()
    r_slow = plugin._auto_regime(None, {"slow_flow": True})
    assert r_slow["slow_flow"] is True
    assert r_slow["delta_t"] == 5

    r_override = plugin._auto_regime(
        None,
        {
            "density_estimate": 1000,
            "slow_flow": True,
            "bidirectional": False,
            "use_2d": "false",
            "delta_t": 2,
            "confirm_link": 0.5,
        },
    )
    assert r_override["bidirectional"] is False
    assert r_override["use_2d"] is False
    assert r_override["delta_t"] == 2
    assert r_override["confirm"] == 0.5


def test_sample_probe_windows_short_and_long():
    assert Tracking.sample_probe_windows(100, 120, n_windows=5, window=30) == [(100, 120)]

    wins = Tracking.sample_probe_windows(1, 1000, n_windows=5, window=30)
    assert len(wins) == 5
    for s, e in wins:
        assert 1 <= s < e <= 1000
        assert (e - s + 1) == 30
    assert len(set(wins)) == len(wins)


def test_aggregate_probe_metrics():
    assert Tracking.aggregate_probe_metrics([]) == {}

    agg = Tracking.aggregate_probe_metrics(
        [
            {"v_max": 2.0, "density": 100, "jump_rate": 0.01, "yield": 0.8},
            {"v_max": 4.0, "density": 300, "jump_rate": 0.05, "yield": 0.9},
            {"v_max": 3.0, "density": 200, "jump_rate": 0.02, "yield": 0.85},
        ]
    )
    assert agg["v_max"] == 4.0
    assert agg["density"] == 300
    assert agg["jump_rate"] == 0.05
    assert agg["yield"] == 0.85
    assert agg["regime_drift"] is True


def _write_correspondences(
    store: RunStore,
    n_frames: int,
    *,
    n_particles: int = 30,
    step_xyz: tuple[float, float, float] = (2.0, 0.0, 0.0),
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    base = rng.uniform(-50.0, 50.0, size=(n_particles, 3))
    step = np.asarray(step_xyz, dtype=float)
    for fn in range(1, n_frames + 1):
        pos = base + (fn - 1) * step
        store.write_correspondences(fn, pos, np.zeros((n_particles, 4), dtype=np.int32))


def test_auto_regime_with_runstore_and_explicit_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "regime.zarr", mode="w")
    _write_correspondences(store, n_frames=6, n_particles=30, step_xyz=(0.2, 0.0, 0.0), seed=5)
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

    plugin = Tracking()
    regime = plugin._auto_regime(exp, {})

    # Density=30 (<500) -> use_2d False; conditioning_ratio ~ 0.4 < 1.0 -> slow_flow True -> delta_t=5
    assert regime["slow_flow"] is True
    assert regime["delta_t"] == 5
    assert regime["use_2d"] is False
    assert regime["bidirectional"] is True
    assert plugin._resolved["v_max"] == pytest.approx(0.3, rel=0.05)
    assert plugin._resolved["conditioning_ratio"] == pytest.approx(0.4, rel=0.05)
    assert plugin._resolved["density"] == pytest.approx(30.0)

    # Explicit user overrides win over estimator output
    user_cfg = {
        "v_max": 9.0,
        "slow_flow": False,
        "delta_t": 2,
        "use_2d": True,
        "bidirectional": False,
        "confirm_link": 0.75,
    }
    user_snapshot = dict(user_cfg)
    regime_override = plugin._auto_regime(exp, user_cfg)

    assert regime_override == {
        "bidirectional": False,
        "use_2d": True,
        "slow_flow": False,
        "delta_t": 2,
        "confirm": 0.75,
    }
    assert plugin._resolved["v_max"] == pytest.approx(9.0)
    assert user_cfg == user_snapshot


def _make_mock_ptv_and_tracker(store: RunStore) -> tuple[SimpleNamespace, SimpleNamespace]:
    calls: list[str] = []

    def _full_forward() -> None:
        calls.append("full_forward")

    def _full_forward_3d() -> None:
        calls.append("full_forward_3d")

    def _full_backward() -> None:
        calls.append("full_backward")

    def _postprocess(max_gap: int = 2) -> dict[str, int]:
        calls.append("postprocess")
        return {"links_before": 0, "links_after": 0}

    tracker = SimpleNamespace(
        _store=store,
        _naming={"linkage": ""},
        calls=calls,
        full_forward=_full_forward,
        full_forward_3d=_full_forward_3d,
        full_backward=_full_backward,
        postprocess=_postprocess,
    )
    ptv = SimpleNamespace(
        store=store,
        py_trackcorr_init=lambda exp: tracker,
    )
    return ptv, tracker


def test_conditioning_gate_below_0_5_warns_sets_flags_and_saves_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = RunStore(tmp_path / "cond_05.zarr", mode="w")
    _write_correspondences(store, n_frames=6, n_particles=30, step_xyz=(0.3, 0.0, 0.0), seed=12)
    exp = make_synthetic_exp(
        store,
        first=1,
        last=6,
        cals=[object()],
        cpar=SimpleNamespace(),
        track_cfg={"ghost_filter": False, "postprocess": False},
    )
    exp.track_par = SimpleNamespace(
        dvxmin=-10.0,
        dvxmax=10.0,
        dvymin=-10.0,
        dvymax=10.0,
        dvzmin=-10.0,
        dvzmax=10.0,
    )
    # conditioning_ratio = 0.3 / 1.0 = 0.3 (< 0.5, >= 0.2)
    monkeypatch.setattr("openptv2.tracking_estimator.z_noise_floor_mm", lambda *a, **k: 1.0)
    monkeypatch.setattr("openptv2.track_assisted.run_configured_corrective_pass", lambda *a, **k: None)

    hybrid_calls: list[dict] = []

    class _FakeHybridTracking:
        def __init__(self, ptv=None, exp=None):
            self.exp = exp

        def do_tracking(self) -> None:
            hybrid_calls.append(dict(self.exp.pm.parameters["track"]))

    monkeypatch.setattr("openptv2.plugins.hybrid_deltat_3d.Tracking", _FakeHybridTracking)

    ptv, _tracker = _make_mock_ptv_and_tracker(store)
    plugin = Tracking(ptv=ptv, exp=exp)

    with caplog.at_level(logging.WARNING, logger="openptv2.unified_tracking"):
        plugin.do_tracking()

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "0.300" in captured.out
    assert any("WARNING" in r.message and "0.300" in r.message for r in caplog.records)
    assert not any("STRONG WARNING" in r.message for r in caplog.records)

    assert len(hybrid_calls) == 1
    assert hybrid_calls[0]["stride"] == 5
    assert hybrid_calls[0]["dvxmax"] == pytest.approx(0.45, rel=0.05)
    # On-disk / user track_cfg not mutated
    assert "dvxmax" not in exp.pm.parameters["track"]

    decision = TrackingDecision.load(store)
    assert decision.conditioning_ratio == pytest.approx(0.3, rel=0.05)
    assert decision.regime_flags["slow_flow"] is True
    assert decision.regime_flags["delta_t"] == 5
    assert decision.regime_flags["use_2d"] is True
    assert decision.chosen_config["slow_flow"] is True
    assert decision.chosen_config["delta_t"] == 5
    assert decision.chosen_config["use_2d"] is True


def test_conditioning_gate_below_0_2_strong_warning_and_proceeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = RunStore(tmp_path / "cond_02.zarr", mode="w")
    _write_correspondences(store, n_frames=6, n_particles=30, step_xyz=(0.1, 0.0, 0.0), seed=19)
    # Explicit use_2d=False and slow_flow=False to verify overrides are respected
    # while STRONG WARNING still logs/prints and tracking proceeds.
    exp = make_synthetic_exp(
        store,
        first=1,
        last=6,
        cals=[object()],
        cpar=SimpleNamespace(),
        track_cfg={
            "ghost_filter": False,
            "postprocess": True,
            "use_2d": False,
            "slow_flow": False,
        },
    )
    exp.track_par = SimpleNamespace(
        dvxmin=-10.0,
        dvxmax=10.0,
        dvymin=-10.0,
        dvymax=10.0,
        dvzmin=-10.0,
        dvzmax=10.0,
    )
    # conditioning_ratio = 0.1 / 1.0 = 0.1 (< 0.2)
    monkeypatch.setattr("openptv2.tracking_estimator.z_noise_floor_mm", lambda *a, **k: 1.0)
    monkeypatch.setattr("openptv2.track_assisted.run_configured_corrective_pass", lambda *a, **k: None)

    ptv, tracker = _make_mock_ptv_and_tracker(store)
    plugin = Tracking(ptv=ptv, exp=exp)

    with caplog.at_level(logging.WARNING, logger="openptv2.unified_tracking"):
        plugin.do_tracking()

    captured = capsys.readouterr()
    assert "STRONG WARNING" in captured.out
    assert "0.100" in captured.out
    assert any("STRONG WARNING" in r.message and "calibration" in r.message.lower() for r in caplog.records)

    # Proceeded via trackcorr with use_2d=False (full_forward_3d) and v_max applied to tpar
    assert tracker.calls == ["full_forward_3d", "full_backward", "postprocess"]
    assert exp.track_par.dvxmax == pytest.approx(0.15, rel=0.05)

    decision = TrackingDecision.load(store)
    assert decision.conditioning_ratio == pytest.approx(0.1, rel=0.05)
    assert decision.regime_flags["use_2d"] is False
    assert decision.chosen_config["v_max"] == pytest.approx(0.15, rel=0.05)
    assert isinstance(decision.report_metrics, dict)
    assert "reciprocity_rate" in decision.report_metrics
