"""Unit tests for the stateful two-phase tracker (velocity + gaps).

Pure array-level tests: no work dir, no calibration (identity project_fn).
"""

import numpy as np

from openptv2.plugins.two_phase_tracking import (
    Tracking,
    TwoPhaseTracker,
    TwoPhaseTrackerConfig,
    _links_to_linkage,
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


def test_linkage_drops_gap_links_no_phantom():
    """Gap links never reach ptv_is: two of them with coinciding rows made a
    reciprocal phantom link (964:1161->966:1191 + 965:1161->967:1191 wrote
    965:1161 <-> 966:1191, an 82 mm one-frame step on a real 4-camera run)."""
    frames = [964, 965, 966, 967]
    sizes = [2, 2, 2, 2]
    links = [(0, 1, 2, 1), (1, 1, 3, 1), (0, 0, 1, 0)]  # two gaps + one real link
    lk = _links_to_linkage(links, frames, sizes)
    (p964, n964), (p965, n965), (p966, n966), (p967, n967) = lk
    assert n964.tolist() == [0, -1] and p965.tolist() == [0, -1]  # consecutive kept
    assert n965[1] == -1 and p966[1] == -1  # no phantom 965:1 <-> 966:1
    assert p967[1] == -1 and n964[1] == -1  # gap ends left unlinked


def test_linkage_skips_missing_frame_numbers():
    """Consecutive list positions but non-consecutive frame numbers = a gap."""
    lk = _links_to_linkage([(0, 0, 1, 0)], [10, 12], [1, 1])
    assert lk[0][1][0] == -1 and lk[1][0][0] == -1


def test_project_fn_batched_matches_pointwise():
    """Bug 3 (plan step 2): batched project_fn == old point-by-point loop."""
    from types import SimpleNamespace

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar

    base = (
        __import__("pathlib").Path(__file__).resolve().parent.parent.parent
        / "test_data"
        / "test_cavity"
    )
    cals = [
        Calibration.from_file(
            str(base / f"cal/cam{c}.tif.ori"),
            str(base / f"cal/cam{c}.tif.addpar"),
        )
        for c in (1, 2)
    ]
    cpar = ControlPar()
    cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y = 1280, 1024, 0.012, 0.012
    fn = Tracking(ptv=None,
                  exp=SimpleNamespace(cals=cals, cpar=cpar))._build_project_fn()
    assert fn is not None
    rng = np.random.default_rng(1)
    pts = rng.uniform([-60, -30, -20], [50, 50, 20], size=(25, 3))
    got = fn(pts)
    want = np.full_like(got, np.nan)
    for i in range(len(pts)):
        for ci in range(2):
            m = img_coord_batch(pts[i : i + 1], cals[ci], cpar.mm)[0]
            want[i, 2 * ci] = m[0] / 0.012 + 640.0
            want[i, 2 * ci + 1] = 512.0 - m[1] / 0.012
    np.testing.assert_allclose(got, np.nan_to_num(want), rtol=1e-12, atol=1e-12)
    assert fn(np.zeros((0, 3))).shape == (0, 4)
