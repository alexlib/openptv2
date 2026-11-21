"""End to end: the holistic loop brings back a particle the matcher dropped because it hid in a neighbour's blob."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.holistic import run_holistic
from openptv2.loco import project_pixels
from openptv2.merge_post import chains_from_linkage
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore

pytestmark = pytest.mark.unit
YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"
N_FR = 20
MERGED = range(8, 13)


def _scene(tmp_path):
    cpar, vpar, cals, _ = load_parameters(YAML)
    rng = np.random.default_rng(4)
    starts = rng.uniform(-6, 6, (8, 3))
    vel = rng.uniform(-0.04, 0.04, (8, 3))
    starts[1] = starts[0] + [0.0, 0.0, 0.0]
    probe = project_pixels(np.array([starts[0], starts[0] + [1.0, 0, 0]]), cals[0], cpar)
    starts[1][0] += 2.0 / np.linalg.norm(probe[1] - probe[0])        # B: 2 px from A in camera 0
    vel[1] = vel[0]
    bright = np.full(8, 200.0)
    bright[0], bright[1] = 300.0, 150.0
    store = RunStore(tmp_path / "res" / "run.zarr", mode="w")
    truth = {}
    for f in range(1, N_FR + 1):
        p = starts + vel * f
        truth[f] = p
        px = np.stack([project_pixels(p, c, cpar) for c in cals], axis=1)       # (8, 4, 2)
        ids = np.tile(np.arange(8)[:, None], (1, 4))
        for c in range(4):
            rows = []
            for k in range(8):
                rows.append([k, px[k, c, 0], px[k, c, 1], 9, 3, 3, bright[k], -1])
            rows = np.array(rows, float)
            if c == 0 and f in MERGED:                                          # A and B are one blob
                w = bright[:2] / bright[:2].sum()
                rows[0, 1:3] = w @ px[:2, 0]
                rows[0, 6] = bright[:2].sum()
                rows[0, 3] = 14
                rows = np.delete(rows, 1, axis=0)
                rows[:, 0] = np.arange(len(rows))
                ids[:, 0] = np.r_[0, -1, np.arange(1, 7)]
            store.write_targets(c, f, rows)
        keep = np.ones(8, bool)
        if f in MERGED:
            keep[1] = False                                                      # the matcher dropped B
        store.write_correspondences(f, p[keep], ids[keep])
    return cpar, vpar, cals, store, truth


def test_hidden_particle_is_born_and_linked_through_the_overlap(tmp_path):
    from openptv2.plugins.two_phase_tracking import track_store

    cpar, vpar, cals, store, truth = _scene(tmp_path)
    frames = list(range(1, N_FR + 1))
    cfg = {"v_max": 0.5, "q_seed": None, "q_young": 0, "confirm_tol": None, "confirm_auto": None}

    def relink():
        return track_store(store, frames, cfg, cals, cpar)

    n0 = {f: len(store.read_correspondences(f)[0]) for f in frames}
    stats = run_holistic(store, frames, cpar, cals, vpar, relink, rounds=1, partner_min_len=0, log=lambda s: None)
    assert stats["rounds"][0]["births"] >= len(MERGED)
    for f in frames:                                   # no unlinked birth is left in the output
        pos, _ = store.read_correspondences(f)
        prev, nxt, _x = store.read_linkage(f, "ptv_is")
        extra = np.arange(len(pos)) >= n0[f]
        assert not np.any(extra & (prev < 0) & (nxt < 0))
    for f in MERGED:
        pos, _ = store.read_correspondences(f)
        d = np.linalg.norm(pos - truth[f][1], axis=1)
        assert d.min() < 0.05, f"B not recovered in frame {f}"
    # one chain follows B through the overlap
    prev = {f: store.read_linkage(f, "ptv_is")[0] for f in frames}
    nxt = {f: store.read_linkage(f, "ptv_is")[1] for f in frames}
    tid = chains_from_linkage(prev, nxt)
    chain_of_b = set()
    for f in frames:
        pos, _ = store.read_correspondences(f)
        j = int(np.argmin(np.linalg.norm(pos - truth[f][1], axis=1)))
        chain_of_b.add(int(tid[f][j]))
    assert len(chain_of_b) <= 2                       # at most one break
    obs = store.read_observations(MERGED[0])
    assert obs is not None and obs.shape[1:] == (4, 4)
    assert (obs[:, 0, 3] >= 2).sum() >= 2             # the merged blob is marked as shared by two points


def test_linker_writing_through_another_store_object_keeps_rows_aligned(tmp_path):
    """The pipeline's linker opens its own store; the loop must not read a stale linkage afterwards."""
    from openptv2.plugins.two_phase_tracking import track_store

    cpar, vpar, cals, store, truth = _scene(tmp_path)
    frames = list(range(1, N_FR + 1))
    cfg = {"v_max": 0.5, "q_seed": None, "q_young": 0, "confirm_tol": None, "confirm_auto": None}

    def relink_elsewhere():
        return track_store(RunStore(store.store_path, mode="a"), frames, cfg, cals, cpar)

    stats = run_holistic(store, frames, cpar, cals, vpar, relink_elsewhere, rounds=2, partner_min_len=0,
                         log=lambda s: None)
    assert len(stats["rounds"]) == 2


def test_unified_delegates_to_holistic_when_asked(monkeypatch):
    from types import SimpleNamespace

    from openptv2.plugins import holistic_tracking, unified_tracking

    called = []
    monkeypatch.setattr(holistic_tracking.Tracking, "do_tracking", lambda self: called.append(1))
    t = unified_tracking.Tracking(exp=SimpleNamespace())
    monkeypatch.setattr(t, "_track_cfg", lambda exp: {"holistic": True})
    monkeypatch.setattr(t, "_auto_regime", lambda *a, **k: {})
    monkeypatch.setattr(t, "_apply_conditioning_gate", lambda *a, **k: None)
    monkeypatch.setattr(t, "_resolve_store_and_range", lambda exp: (None, None, 1, 2))
    t.do_tracking()
    assert called == [1]


def test_trajectories_cache_round_trip(tmp_path):
    import numpy as np

    from openptv2.holistic import write_trajectories_cache
    from openptv2.storage import RunStore
    from openptv2.tracking_postprocess import write_linkage

    st = RunStore(tmp_path / "run.zarr", mode="a")
    frames = [1, 2, 3]
    n_cams = 2
    for f in frames:
        pos = np.array([[0.0 + f, 0, 0], [10.0, 5.0 + 2 * f, 1.0]])
        st.write_correspondences(f, pos, np.full((2, n_cams), 0, dtype=np.int64))
        for c in range(n_cams):
            st.write_targets(c, f, np.zeros((1, 8)))
        prev = np.array([-1 if f == 1 else 0, -1 if f == 1 else 1])
        nxt = np.array([-2 if f == 3 else 0, -2 if f == 3 else 1])
        write_linkage("ptv_is", f, prev, nxt, np.zeros(2), store=st)
    n = write_trajectories_cache(RunStore(tmp_path / "run.zarr", mode="a"), frames, n_cams)
    g = RunStore(tmp_path / "run.zarr", mode="r").root["trajectories"]
    assert n == 2 and g["pos"].shape == (6, 3)
    vel = np.asarray(g["vel"][:])
    assert np.isclose(np.abs(vel).max(), 2e-3)                    # the fast track: 2 mm per frame, in metres


def _stub_two_phase(monkeypatch, cfg):
    """A two_phase plugin on a stub experiment whose store has frames 1..3; records which path ran."""
    from types import SimpleNamespace

    from openptv2.plugins import two_phase_tracking as tp

    store = SimpleNamespace(frames=lambda: [1, 2, 3])
    exp = SimpleNamespace(pm=SimpleNamespace(parameters={"track": cfg}), _store=store, cals=[], cpar=None,
                          spar=None)
    ran = []
    monkeypatch.setattr(tp, "track_store", lambda *a, **k: ran.append("linker"))
    import openptv2.holistic as hol

    monkeypatch.setattr(hol, "run_holistic_experiment", lambda e, s, f, c, rounds: ran.append(("holistic", rounds)))
    return tp, exp, ran


def test_two_phase_is_plain_by_default(monkeypatch):
    tp, exp, ran = _stub_two_phase(monkeypatch, {})
    tp.Tracking(exp=exp).do_tracking()
    assert ran == ["linker"]


def test_two_phase_runs_the_loop_when_rounds_are_set(monkeypatch):
    tp, exp, ran = _stub_two_phase(monkeypatch, {"holistic_rounds": 3})
    tp.Tracking(exp=exp).do_tracking()
    assert ran == [("holistic", 3)]


def test_holistic_name_is_two_phase_with_two_rounds(monkeypatch):
    from openptv2.plugins import holistic_tracking

    tp, exp, ran = _stub_two_phase(monkeypatch, {})
    holistic_tracking.Tracking(exp=exp).do_tracking()
    assert ran == [("holistic", 2)]
    tp, exp, ran = _stub_two_phase(monkeypatch, {"holistic_rounds": 0})
    holistic_tracking.Tracking(exp=exp).do_tracking()
    assert ran == ["linker"]


def test_retired_trackers_are_hidden_but_resolve():
    from openptv2.plugins.loader import (
        discover_available_plugins,
        resolve_plugin_module,
    )
    from openptv2.tracking_setup import TRACKERS, resolve_tracking_setup

    shown = set(discover_available_plugins(None)["available_tracking"])
    assert not shown & {"unified", "4be", "four_be", "hybrid_deltat_3d", "holistic", "splitter_tracking"}
    assert {"two_phase", "fast_3d", "trackcorr"} <= shown
    assert TRACKERS == ("two_phase", "trackcorr", "fast_3d")
    assert resolve_tracking_setup({}, {"selected_tracking": "unified"}).tracker == "two_phase"
    assert resolve_tracking_setup({}, {"selected_tracking": "holistic"}).tracker == "two_phase"
    for name in ("unified", "4be", "hybrid_deltat_3d", "holistic"):          # old YAMLs keep running
        from openptv2.plugins.loader import BUILTIN_TRACKING_PLUGINS

        assert resolve_plugin_module(name, BUILTIN_TRACKING_PLUGINS) is not None
