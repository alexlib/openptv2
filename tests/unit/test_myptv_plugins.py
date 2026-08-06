"""Unit tests for MyPTV 3D and 2D tracking plugins."""

import numpy as np
import pytest

from openptv2.plugins.loader import BUILTIN_TRACKING_PLUGINS, resolve_plugin_module
from openptv2.plugins.myptv_2d_tracking import MyPTV2DTracker
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker


def test_myptv_3d_tracker_synthetic_linear_motion():
    tracker = MyPTV3DTracker(v_max=5.0, a_max=10.0, max_gap=1, dt=0.1)

    # 10 frames of linear particle motion: p(t) = p0 + v * t
    v1 = np.array([2.0, 1.0, 0.0])
    v2 = np.array([-1.0, 3.0, 0.5])

    frames = []
    for f in range(10):
        t = f * 0.1
        p1 = t * v1
        p2 = np.array([10.0, 0.0, 0.0]) + t * v2
        frames.append(np.array([p1, p2]))

    results = tracker.track_frames(frames)

    # Should find 2 continuous trajectories of length 10
    assert len(results) == 2
    lens = sorted([len(tr["pos"]) for tr in results])
    assert lens == [10, 10]


def test_myptv_3d_tracker_with_cost_weights():
    """Verify MyPTV3DTracker works cleanly with custom CostWeights."""
    from openptv2.tracking_cost import CostWeights

    weights = CostWeights(
        w_distance=1.0, w_velocity=0.5, w_acceleration=0.2, w_intensity=0.0
    )
    tracker = MyPTV3DTracker(
        v_max=5.0, a_max=10.0, max_gap=1, dt=0.1, cost_weights=weights
    )

    frames = [
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.array([[0.2, 0.1, 0.0], [9.9, 0.3, 0.05]]),
        np.array([[0.4, 0.2, 0.0], [9.8, 0.6, 0.10]]),
    ]

    results = tracker.track_frames(frames)
    assert len(results) == 2
    assert len(results[0]["pos"]) == 3
    assert len(results[1]["pos"]) == 3


def test_myptv_2d_tracker_synthetic_pixel_motion():
    tracker = MyPTV2DTracker(max_pixel_disp=15.0, max_gap=1)

    # 2D pixel trajectory
    frames_2d = []
    for f in range(8):
        pt1 = np.array([100.0 + f * 5.0, 200.0 + f * 2.0])
        pt2 = np.array([300.0 - f * 3.0, 400.0 + f * 4.0])
        frames_2d.append(np.array([pt1, pt2]))

    results = tracker.track_2d_blobs(frames_2d)

    assert len(results) == 2
    lens = sorted([len(tr["pos_2d"]) for tr in results])
    assert lens == [8, 8]


def _reference_match(pred, cands, radius):
    """Dense big-M assignment — the definition match_within_radius optimises."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    dists = cdist(pred, cands)
    in_radius = dists <= np.broadcast_to(radius, (len(pred),))[:, None]
    if not in_radius.any():
        return set()
    big = dists[in_radius].sum() + 1.0
    rows, cols = linear_sum_assignment(np.where(in_radius, dists, big))
    keep = in_radius[rows, cols]
    return set(zip(rows[keep].tolist(), cols[keep].tolist()))


@pytest.mark.parametrize(
    "n_pred,n_cand,radius",
    [
        (30, 30, 2.0),  # dense path, balanced
        (30, 5, 2.0),  # dense path, far more tracks than candidates
        (5, 30, 2.0),  # dense path, far more candidates than tracks
        (40, 40, 0.05),  # dense path, radius so tight most rows have no edge
        (450, 450, 1.5),  # over DENSE_CUTOFF -> component-decomposed path
        (450, 450, 12.0),  # over cutoff, radius wide enough to fuse components
    ],
)
def test_match_within_radius_equals_dense_reference(n_pred, n_cand, radius):
    """The KD-tree/component path must agree with the dense formulation."""
    from openptv2.plugins._assignment import match_within_radius

    rng = np.random.default_rng(7)
    pred = rng.uniform(-20, 20, size=(n_pred, 3))
    cands = rng.uniform(-20, 20, size=(n_cand, 3))

    rows, cols = match_within_radius(pred, cands, radius)
    got = set(zip(rows.tolist(), cols.tolist()))
    expected = _reference_match(pred, cands, radius)

    # Same number of links, and the same total displacement. Distinct pairings
    # of equal cost are acceptable; a different link count is not.
    assert len(got) == len(expected)
    assert len(set(r for r, _ in got)) == len(got), "a track matched twice"
    assert len(set(c for _, c in got)) == len(got), "a candidate matched twice"

    def total(pairs):
        return sum(float(np.linalg.norm(pred[r] - cands[c])) for r, c in pairs)

    assert total(got) == pytest.approx(total(expected))


def test_match_within_radius_per_track_radius():
    """A per-prediction radius must gate each row independently."""
    from openptv2.plugins._assignment import match_within_radius

    pred = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    cands = np.array([[1.0, 0.0, 0.0], [12.0, 0.0, 0.0]])

    rows, cols = match_within_radius(pred, cands, np.array([5.0, 0.5]))
    assert set(zip(rows.tolist(), cols.tolist())) == {(0, 0)}


def _crowded_frames_3d(n_particles=60, n_frames=15, seed=20260726):
    """Deterministic crowded field: rotation + noise + 5% turnover per frame.

    Crowded enough that the cost matrix actually has competing candidates,
    which is what the vectorized cost-matrix build has to get right.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-20, 20, size=(n_particles, 3))
    frames = []
    for _ in range(n_frames):
        vel = np.stack([-pos[:, 1], pos[:, 0], np.zeros(len(pos))], axis=1) * 0.02
        pos = pos + vel + rng.normal(0, 0.02, pos.shape)
        pos = pos[rng.random(len(pos)) > 0.05]
        n_new = n_particles - len(pos)
        if n_new > 0:
            pos = np.vstack([pos, rng.uniform(-20, 20, size=(n_new, 3))])
        frames.append(pos.copy())
    return frames


def test_myptv_3d_tracker_crowded_field_golden():
    """Pins tracker output on a crowded field.

    Golden values predate the vectorized (cdist) cost-matrix build and were
    verified bit-identical against the previous per-track loop.
    """
    frames = _crowded_frames_3d()
    tracker = MyPTV3DTracker(v_max=2.0, a_max=2.0, max_gap=2, dt=0.1)
    results = tracker.track_frames(frames)

    assert len(results) == 276
    assert sum(len(tr["pos"]) for tr in results) == 867
    assert max(len(tr["pos"]) for tr in results) == 15
    np.testing.assert_allclose(
        results[0]["pos"][0], [8.059652, 14.914145, 3.423682], atol=1e-6
    )


def test_myptv_2d_tracker_crowded_field_golden():
    frames_3d = _crowded_frames_3d()
    frames_2d = [
        np.stack([f[:, 0] * 30 + 512, f[:, 1] * 30 + 512], axis=1) for f in frames_3d
    ]
    results = MyPTV2DTracker(max_pixel_disp=60.0, max_gap=2).track_2d_blobs(frames_2d)

    assert len(results) == 91
    assert sum(len(tr["pos_2d"]) for tr in results) == 892
    assert max(len(tr["pos_2d"]) for tr in results) == 15


@pytest.mark.parametrize(
    "frames",
    [
        pytest.param([np.zeros((0, 3))] * 4, id="all_empty"),
        pytest.param([np.zeros((1, 3))], id="one_frame"),
        pytest.param(
            [np.zeros((3, 3)), np.zeros((0, 3)), np.ones((3, 3))], id="empty_middle"
        ),
    ],
)
def test_myptv_3d_tracker_degenerate_frames(frames):
    """Vectorized cost matrix must not be reached with a zero-length axis."""
    MyPTV3DTracker(v_max=2.0, a_max=2.0, max_gap=2, dt=0.1).track_frames(frames)


def test_myptv_plugins_plugin_loader_resolution():
    mod_3d = resolve_plugin_module("myptv_3d_tracking", BUILTIN_TRACKING_PLUGINS)
    assert hasattr(mod_3d, "Tracking")

    mod_2d = resolve_plugin_module("myptv_2d_tracking", BUILTIN_TRACKING_PLUGINS)
    assert hasattr(mod_2d, "Tracking")
