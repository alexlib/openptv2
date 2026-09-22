"""Unit tests for the stateful two-phase tracker (velocity + gaps).

Pure array-level tests: no work dir, no calibration (identity project_fn).
"""

import numpy as np

from openptv2.plugins.two_phase_tracking import (
    TwoPhaseTracker,
    TwoPhaseTrackerConfig,
)


def ident(P):
    """Fake projection: leaves == 3D coords (first 2 dims used as pixels)."""
    P = np.asarray(P, dtype=np.float64)
    return P[:, :2]


def cfg(**kw):
    base = {"v_max": 5.0, "leaf_weight": 1.0, "use_velocity": True,
            "cost_mode": "projected", "max_gap": 2}
    base.update(kw)
    return TwoPhaseTrackerConfig(**base)


def leaves_of(frames):
    return [ident(P) for P in frames]


def test_crossing_needs_velocity():
    """Head-on X-cross: raw bounces (swap), velocity crosses (truth)."""
    frames = [
        np.array([[0.0, 0, 0], [2.0, 0, 0]]),
        np.array([[0.9, 0, 0], [1.1, 0, 0]]),  # near-coincident mid-frame
        np.array([[2.0, 0, 0], [0.0, 0, 0]]),
    ]
    raw = TwoPhaseTracker(cfg(use_velocity=False))
    got = raw.track_frames(frames, leaves_of(frames), project_fn=ident)
    # bounce: row0 -> row1 and row1 -> row0 at the crossing step
    assert (1, 0, 2, 1) in got and (1, 1, 2, 0) in got

    vel = TwoPhaseTracker(cfg())
    got = vel.track_frames(frames, leaves_of(frames), project_fn=ident)
    assert (1, 0, 2, 0) in got and (1, 1, 2, 1) in got


def test_gap_bridging():
    """A particle missing one frame is re-caught (max_gap=2)."""
    frames = [
        np.array([[0.0, 0, 0], [9.0, 9, 9]]),
        np.array([[1.0, 0, 0], [9.0, 9, 9]]),
        np.array([[9.0, 9, 9]]),  # pid0 occluded
        np.array([[3.0, 0, 0], [9.0, 9, 9]]),
    ]
    tr = TwoPhaseTracker(cfg())
    got = tr.track_frames(frames, leaves_of(frames), project_fn=ident)
    # gap-spanning link (1,0) -> (3,0): last seen row 0 to row 0
    assert (1, 0, 3, 0) in got


def test_gap_retires():
    """A particle missing longer than max_gap is not re-caught."""
    frames = [
        np.array([[0.0, 0, 0]]),
        np.array([[9.0, 9, 9]]),
        np.array([[9.0, 9, 9]]),
        np.array([[9.0, 9, 9]]),
        np.array([[4.0, 0, 0]]),
    ]
    tr = TwoPhaseTracker(cfg(max_gap=2))
    got = tr.track_frames(frames, leaves_of(frames), project_fn=ident)
    assert all(t1 != 4 for _, _, t1, _ in got)  # never re-caught


def test_cold_start_zero_velocity():
    """First link from standstill links nearest neighbour."""
    frames = [np.array([[0.0, 0, 0]]), np.array([[0.4, 0, 0]])]
    tr = TwoPhaseTracker(cfg())
    assert tr.track_frames(frames, leaves_of(frames),
                           project_fn=ident) == [(0, 0, 1, 0)]


def test_cost_3d_without_project_fn():
    """No project_fn -> 3D costs, velocity still crosses."""
    frames = [
        np.array([[0.0, 0, 0], [2.0, 0, 0]]),
        np.array([[0.9, 0, 0], [1.1, 0, 0]]),
        np.array([[2.0, 0, 0], [0.0, 0, 0]]),
    ]
    tr = TwoPhaseTracker(cfg(cost_mode="3d"))
    got = tr.track_frames(frames, leaves_of(frames), project_fn=None)
    assert (1, 0, 2, 0) in got and (1, 1, 2, 1) in got


def test_cascade_merge_additive():
    """Merge extends free ends, never steals, accepts gap links."""
    import importlib.util

    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "cascade_track",
        _Path(__file__).resolve().parent.parent.parent
        / "scripts" / "cascade_track.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    merge_links = mod.merge_links
    corr_prev = [[-1, -1], [0, -1]]  # frame1: row0<-row0, row1 head
    corr_next = [[0, -2], [-1, -1]]  # row0->row0; row1 tail (-2 = dropped)
    nrows = [2, 2]
    # (0,1)->(1,1): both ends free -> accepted (extends tail+head)
    # (0,0)->(1,1): target taken after first -> rejected (no steal)
    # (0,1)->(1,0): source free but target taken by base -> rejected
    # gap link (0,1)->(1,1) counted once
    tp_links = [(0, 1, 1, 1), (0, 0, 1, 1), (0, 1, 1, 0)]
    mp, mn, n, added = merge_links(corr_prev, corr_next, tp_links, nrows)
    assert n == 1 and added == [(0, 1, 1, 1)]
    assert list(mn[0]) == [0, 1] and list(mp[1]) == [0, 1]
    # base links untouched
    assert mn[0][0] == 0 and mp[1][0] == 0


def test_bidirectional_tracking():
    """Bidirectional tracking resolves crossings and returns clean 1-to-1 chains."""
    frames = [
        np.array([[0.0, 0, 0], [2.0, 0, 0]]),
        np.array([[0.9, 0, 0], [1.1, 0, 0]]),
        np.array([[2.0, 0, 0], [0.0, 0, 0]]),
    ]
    tr = TwoPhaseTracker(cfg(bidirectional=True))
    got = tr.track_frames(frames, leaves_of(frames), project_fn=ident)
    assert (1, 0, 2, 0) in got and (1, 1, 2, 1) in got

    # Test return_chains=True with bidirectional=True
    links, chains = tr.track_frames(
        frames, leaves_of(frames), project_fn=ident, return_chains=True
    )
    assert len(links) == len(got)
    assert len(chains) == 2
    assert all(len(c["frames"]) == 3 for c in chains)
