"""Tests for ghost_ilp module."""

import numpy as np

from openptv2.ghost_ilp import (
    ilp_uniqueness_frame,
    resolve_ilp_config,
    solve_component_ilp,
)


def test_empty():
    pos = np.zeros((0, 3))
    cam = np.zeros((0, 2), dtype=int)
    keep = ilp_uniqueness_frame(pos, cam)
    assert keep.shape == (0,)
    assert keep.dtype == bool


def test_single_point_kept():
    pos = np.array([[1.0, 2.0, 3.0]])
    cam = np.array([[5, 7]], dtype=int)
    keep = ilp_uniqueness_frame(pos, cam)
    assert keep.tolist() == [True]


def test_disjoint_points_all_kept():
    pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    cam = np.array([[1, 2], [3, 4]], dtype=int)
    keep = ilp_uniqueness_frame(pos, cam)
    assert keep.tolist() == [True, True]


def test_conflict_keeps_higher_conf():
    pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    cam = np.array([[5, 2], [5, 9]], dtype=int)  # share target 5
    conf = np.array([0.3, 0.9])
    keep = ilp_uniqueness_frame(pos, cam, conf=conf)
    assert keep.tolist() == [False, True]


def test_conflict_equal_conf_keeps_first():
    pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    cam = np.array([[5, 2], [5, 9]], dtype=int)
    conf = np.array([0.5, 0.5])
    keep = ilp_uniqueness_frame(pos, cam, conf=conf)
    # tie -> first kept
    assert keep.tolist() == [True, False]


def test_three_way_ilp_uniqueness():
    # 3 points, all share target 0 in cam0; keep highest conf only
    pos = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    cam = np.array([[0, 1], [0, 2], [0, 3]], dtype=int)
    conf = np.array([0.2, 0.9, 0.5])
    keep = ilp_uniqueness_frame(pos, cam, conf=conf)
    assert keep.tolist() == [False, True, False]


def test_three_points_no_conflict_all_kept():
    pos = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    cam = np.array([[1, 2], [3, 4], [5, 6]], dtype=int)
    keep = ilp_uniqueness_frame(pos, cam)
    assert keep.tolist() == [True, True, True]


def test_resolve_ilp_config_explicit():
    assert resolve_ilp_config({"ilp_uniqueness": True}, {}) is True
    assert resolve_ilp_config({"ilp_uniqueness": "false"}, {}) is False
    assert resolve_ilp_config({"ilp_uniqueness": "true"}, {}) is True


def test_resolve_ilp_config_auto():
    assert resolve_ilp_config({"ilp_uniqueness": "auto"}, {"ghost_rate": 0.5}) is True
    assert resolve_ilp_config({"ilp_uniqueness": "auto"}, {"ghost_rate": 0.05, "density": 100}) is False
    assert resolve_ilp_config({"ilp_uniqueness": "auto"}, {"density": 2500}) is True


def test_solve_component_ilp_overflow_fallback():
    # component larger than max_size -> greedy fallback, still returns subset
    n = 25
    pos = np.arange(n)[:, None] * np.ones((1, 3))
    cam = np.column_stack([np.zeros(n, dtype=int), np.arange(n, dtype=int)])
    # all share target 0 in cam0 -> only one can be kept
    conf = np.random.default_rng(0).random(n)
    sel = solve_component_ilp(pos, cam, list(range(n)), conf, max_size=5)
    assert len(sel) == 1
    assert sel[0] == int(np.argmax(conf))


def test_trajectory_ilp_keeps_non_conflicting():
    from openptv2.ghost_ilp import trajectory_ilp_uniqueness

    trajs = [
        [(1, 0, 10), (2, 0, 11)],
        [(1, 0, 20), (2, 0, 21)],
    ]
    keep = trajectory_ilp_uniqueness(trajs, [2.0, 1.5])
    assert keep.tolist() == [True, True]


def test_trajectory_ilp_prefers_two_clean_over_one_spanning_ghost():
    from openptv2.ghost_ilp import trajectory_ilp_uniqueness

    # Trajectory 1 spans targets from both Trajectory 0 and Trajectory 2.
    # sum(score[0] + score[2]) = 3.0 + 3.0 = 6.0 > score[1] = 4.0 -> keep {0, 2}.
    trajs = [
        [(1, 0, 1), (2, 0, 2)],
        [(1, 0, 1), (3, 0, 3)],
        [(3, 0, 3), (4, 0, 4)],
    ]
    keep = trajectory_ilp_uniqueness(trajs, [3.0, 4.0, 3.0])
    assert keep.tolist() == [True, False, True]


def test_trajectory_ilp_overflow_fallback():
    from openptv2.ghost_ilp import trajectory_ilp_uniqueness

    trajs = [[(1, 0, 99), (2, 0, i)] for i in range(10)]
    scores = [float(i) for i in range(10)]
    keep = trajectory_ilp_uniqueness(trajs, scores, max_size=3)
    assert keep.sum() == 1
    assert bool(keep[9]) is True
