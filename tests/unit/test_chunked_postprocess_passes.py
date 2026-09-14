"""run_postprocess_passes must report failures instead of hiding them."""

import pytest

import openptv2.tracking_postprocess as tp
from openptv2.algorithms.parameters import TrackPar
from openptv2.tracking_chunked import run_postprocess_passes


def test_a_failing_pass_warns_and_the_others_still_run(monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append("seed")
        raise RuntimeError("store is read-only")

    monkeypatch.setattr(tp, "seed_cold_start", boom)
    monkeypatch.setattr(tp, "relink_trajectory_gaps", lambda *a, **k: calls.append("relink") or {"bridged_gaps": 3})
    monkeypatch.setattr(tp, "enforce_reciprocity", lambda *a, **k: calls.append("recip") or {"severed_next": 0})

    with pytest.warns(RuntimeWarning, match="seed_cold_start failed"):
        stats = run_postprocess_passes("res/ptv_is", 1, 10, TrackPar(dvxmax=1.0, dacc=0.4))

    assert calls == ["seed", "relink", "recip"]
    assert "error" in stats["seed_cold_start"]
    assert stats["relink_trajectory_gaps"] == {"bridged_gaps": 3}


def test_passes_receive_the_tracking_limits(monkeypatch):
    seen = {}
    monkeypatch.setattr(tp, "seed_cold_start", lambda base, first, last, dv, store=None: seen.update(dv=dv) or {})
    monkeypatch.setattr(
        tp, "relink_trajectory_gaps",
        lambda base, first, last, max_gap, max_accel_err, store=None: seen.update(acc=max_accel_err) or {},
    )
    monkeypatch.setattr(tp, "enforce_reciprocity", lambda *a, **k: {})

    run_postprocess_passes("res/ptv_is", 1, 10, TrackPar(dvxmax=1.4, dacc=0.4))

    assert seen == {"dv": 1.4, "acc": 0.4}
