"""A10: ray-convergence quality mark and its use in the two_phase tracker."""

import numpy as np

from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig
from openptv2.point_quality import fit_scale, ghost_probability


def test_fit_scale_recovers_line_in_centre_distance():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-40, 40, size=(5000, 3))
    d = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
    rcm = (0.03 + 0.0004 * d) * rng.lognormal(0, 0.15, len(d))
    centre, a, b = fit_scale(pos, rcm, np.full(len(d), 4))
    assert abs(a - 0.03) < 0.004 and abs(b - 0.0004) < 0.0001


def test_ghost_probability_rises_with_rcm_and_is_higher_for_three_cameras():
    pos = np.zeros((4, 3))
    scale = (np.zeros(3), 0.04, 0.0)
    rcm = np.array([0.03, 0.04, 0.08, 0.14])
    p4 = ghost_probability(pos, rcm, np.full(4, 4), scale)
    p3 = ghost_probability(pos, rcm, np.full(4, 3), scale)
    assert np.all(np.diff(p4) >= 0) and np.all(np.diff(p3) >= 0)
    assert np.all(p3 >= p4) and p4[0] < 0.05 and p3[-1] > 0.5


def _frames():
    # one particle moving straight; frame 2 also holds a stranger just off it
    f0 = np.array([[0.0, 0, 0]])
    f1 = np.array([[0.05, 0, 0]])
    f2 = np.array([[0.10, 0, 0], [0.30, 0, 0]])
    return [f0, f1, f2]


def _run(**kw):
    cfg = TwoPhaseTrackerConfig(
        v_max=1.0, use_velocity=True, cost_mode="3d", max_gap=1, **kw
    )
    ghost = [np.zeros(1), np.zeros(1), np.array([0.0, 0.9])]
    return TwoPhaseTracker(cfg).track_frames(_frames(), None, frame_ghost=ghost)


def test_q_seed_stops_doubtful_point_from_starting_a_trajectory():
    # the stranger (row 1, frame 2) has ghost probability 0.9: it must not seed
    # a track, so a frame 3 point near it finds nobody to link to
    frames = _frames() + [np.array([[0.15, 0, 0], [0.31, 0, 0]])]
    ghost = [np.zeros(1), np.zeros(1), np.array([0.0, 0.9]), np.zeros(2)]

    def links(q_seed):
        cfg = TwoPhaseTrackerConfig(
            v_max=1.0, cost_mode="3d", max_gap=1, q_seed=q_seed
        )
        return set(TwoPhaseTracker(cfg).track_frames(frames, None, frame_ghost=ghost))

    assert (2, 1, 3, 1) in links(None)
    assert (2, 1, 3, 1) not in links(0.5)
    assert (2, 0, 3, 0) in links(0.5)  # the real trajectory is untouched


def test_q_young_blocks_young_trajectory_but_not_established_one():
    base = {(0, 0, 1, 0), (1, 0, 2, 0)}
    assert base <= set(_run())
    # a track with 1 point (young) may not step onto the doubtful point (0.9)
    frames = [np.array([[0.0, 0, 0]]), np.array([[0.05, 0, 0]])]
    ghost = [np.zeros(1), np.array([0.9])]
    cfg = TwoPhaseTrackerConfig(
        v_max=1.0, cost_mode="3d", max_gap=1, q_seed=0.5, q_young=3
    )
    assert TwoPhaseTracker(cfg).track_frames(frames, None, frame_ghost=ghost) == []
    cfg_old = TwoPhaseTrackerConfig(v_max=1.0, cost_mode="3d", max_gap=1)
    assert TwoPhaseTracker(cfg_old).track_frames(frames, None) == [(0, 0, 1, 0)]
    # established: 3 points, then a doubtful 4th is still linked
    f3 = [np.array([[0.0, 0, 0]]), np.array([[0.05, 0, 0]]),
          np.array([[0.10, 0, 0]]), np.array([[0.15, 0, 0]])]
    g3 = [np.zeros(1), np.zeros(1), np.zeros(1), np.array([0.9])]
    got = TwoPhaseTracker(cfg).track_frames(f3, None, frame_ghost=g3)
    assert (2, 0, 3, 0) in got
