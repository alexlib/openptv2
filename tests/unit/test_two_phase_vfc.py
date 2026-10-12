"""Unit tests for Spatial Neighborhood Consensus (Vector Field Consensus / VFC)
in TwoPhaseTracker (Option A: Pre-Hungarian Gate/Cost, Option B: Post-Assignment Fast-VFC).
"""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.plugins.two_phase_tracking import (
    TwoPhaseTracker,
    TwoPhaseTrackerConfig,
    compute_local_consensus_field,
    filter_links_by_spatial_consensus,
)


def ident(p: np.ndarray) -> np.ndarray:
    """Mock projection: returns first 2 dimensions as pixel coordinates."""
    p = np.asarray(p, dtype=np.float64)
    return p[:, :2]


def test_compute_local_consensus_field_basic():
    """Consensus field correctly extracts local median velocity and dispersion."""
    # 4 points moving at (1.0, 0.0, 0.0) and 1 outlier moving at (0.0, 5.0, 0.0)
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],  # near center
        ]
    )
    vel = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],  # outlier
        ]
    )
    query = np.array([[0.5, 0.5, 0.0]])

    # Radius covers all 5 points
    cons_v, cons_disp, valid = compute_local_consensus_field(
        pos, vel, query, radius=2.0, min_neighbors=3
    )
    assert valid[0]
    # Median along x should be 1.0, along y should be 0.0
    np.testing.assert_allclose(cons_v[0], [1.0, 0.0, 0.0], atol=1e-6)


def test_compute_local_consensus_field_insufficient_neighbors():
    """Points without enough neighbors report valid=False and NaN velocity."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    query = np.array([[0.0, 0.0, 0.0]])

    cons_v, cons_disp, valid = compute_local_consensus_field(
        pos, vel, query, radius=2.0, min_neighbors=3
    )
    assert not valid[0]
    assert np.isnan(cons_v[0]).all()


def test_filter_links_by_spatial_consensus_option_b():
    """Option B prunes links whose displacement deviates strongly from neighbor consensus."""
    # 4 coherent links with velocity (1.0, 0.0, 0.0), 1 outlier link with velocity (1.0, 8.0, 0.0)
    links = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
    tids = [0, 1, 2, 3, 4]
    tracks = {
        0: {"pos": np.array([0.0, 0.0, 0.0]), "last_t": 0},
        1: {"pos": np.array([1.0, 0.0, 0.0]), "last_t": 0},
        2: {"pos": np.array([0.0, 1.0, 0.0]), "last_t": 0},
        3: {"pos": np.array([1.0, 1.0, 0.0]), "last_t": 0},
        4: {"pos": np.array([0.5, 0.5, 0.0]), "last_t": 0},
    }
    pts1 = np.array(
        [
            [1.0, 0.0, 0.0],  # disp = (1, 0, 0)
            [2.0, 0.0, 0.0],  # disp = (1, 0, 0)
            [1.0, 1.0, 0.0],  # disp = (1, 0, 0)
            [2.0, 1.0, 0.0],  # disp = (1, 0, 0)
            [1.5, 8.5, 0.0],  # disp = (1, 8, 0) -> severe outlier
        ]
    )

    kept, rejected = filter_links_by_spatial_consensus(
        links,
        tids,
        tracks,
        pts1,
        dt=1.0,
        current_t=1,
        radius=2.0,
        min_neighbors=3,
        post_k=3.0,
        min_tol=0.5,
    )

    assert (4, 4) not in kept
    assert 4 in rejected
    assert len(kept) == 4
    for i in range(4):
        assert (i, i) in kept


def test_option_a_vfc_gate_in_two_phase_tracker():
    """Option A (vfc_mode='pre', vfc_gate) prunes ghost candidate edges before Hungarian."""
    # 4 parallel particles moving coherently at v=(1, 0, 0) across frames 0, 1, 2.
    # At frame 2, particle 0 encounters two candidate points:
    # - True particle at (2.0, 0.0, 0.0) -> disp = (1.0, 0.0, 0.0)
    # - Ghost particle at (1.0, 2.5, 0.0) -> disp = (0.0, 2.5, 0.0)
    #
    # We deliberately give the ghost a 0.0 pixel distance in 2D leaves (xy), so without VFC
    # leaf_weight would strongly prefer the ghost!
    frames = [
        # Frame 0: 4 particles
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        ),
        # Frame 1: all 4 moved by +1 in x
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        ),
        # Frame 2: particles 1, 2, 3 moved by +1 in x (now at x=2)
        # particle 0 has true candidate at (2, 0, 0) and ghost at (1.2, 2.5, 0.0)
        np.array(
            [
                [2.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 3.0, 0.0],
                [2.0, 0.0, 0.0],  # true match for particle 0 (index 3)
                [1.2, 2.5, 0.0],  # ghost match for particle 0 (index 4)
            ]
        ),
    ]

    # Leaves: make ghost (index 4) have a perfect 2D match to particle 0's prediction
    leaves_f0 = ident(frames[0])
    leaves_f1 = ident(frames[1])
    leaves_f2 = ident(frames[2]).copy()
    # Predicted pos for particle 0 is (2.0, 0.0)
    # Manipulate leaf of ghost so it looks closer in 2D appearance than true particle:
    leaves_f2[4] = [2.0, 0.0]  # fake 2D match for ghost
    leaves_f2[3] = [2.2, 0.2]  # slightly noisy 2D match for true particle
    frame_leaves = [leaves_f0, leaves_f1, leaves_f2]

    # Without VFC: 2D leaf cost favors ghost (index 4) over true particle (index 3)
    cfg_off = TwoPhaseTrackerConfig(
        v_max=5.0,
        leaf_weight=10.0,
        use_velocity=True,
        vfc_mode="off",
    )
    tracker_off = TwoPhaseTracker(cfg_off)
    links_off = tracker_off.track_frames(frames, frame_leaves, project_fn=ident)
    # Check what particle 0 (row 0 in frame 1) linked to in frame 2:
    p0_link_off = [l for l in links_off if l[0] == 1 and l[1] == 0]
    assert len(p0_link_off) == 1
    assert p0_link_off[0][3] == 4  # tricked by ghost!

    # With Option A (vfc_mode='pre', vfc_gate=1.5):
    # Candidate 4 deviates strongly from neighbor velocity consensus of (1, 0, 0)
    # and is gated out before Hungarian matching!
    cfg_pre = TwoPhaseTrackerConfig(
        v_max=5.0,
        leaf_weight=10.0,
        use_velocity=True,
        vfc_mode="pre",
        vfc_radius=5.0,
        vfc_min_neighbors=3,
        vfc_gate=1.5,
    )
    tracker_pre = TwoPhaseTracker(cfg_pre)
    links_pre = tracker_pre.track_frames(frames, frame_leaves, project_fn=ident)
    p0_link_pre = [l for l in links_pre if l[0] == 1 and l[1] == 0]
    assert len(p0_link_pre) == 1
    assert p0_link_pre[0][3] == 3  # correctly linked to true candidate!


def test_option_b_post_vfc_severs_outlier_in_tracker():
    """Option B (vfc_mode='post') severs spurious link that Hungarian accepted."""
    # 4 particles in a cluster. 3 move at (1, 0, 0).
    # 4th particle has no true candidate, only a ghost at (0, 3, 0) which Hungarian accepts.
    frames = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],  # particle 3 jumps/deviates
            ]
        ),
    ]
    leaves = [ident(f) for f in frames]

    # Without VFC post: link is formed
    cfg_off = TwoPhaseTrackerConfig(
        v_max=5.0,
        cost_mode="3d",
        use_velocity=False,
        vfc_mode="off",
    )
    links_off = TwoPhaseTracker(cfg_off).track_frames(frames, leaves, project_fn=ident)
    p3_link_off = [l for l in links_off if l[1] == 3]
    assert len(p3_link_off) == 1

    # With Option B post-VFC: displacement of particle 3 is (0, 0, 0) while neighbors are (1, 0, 0)
    cfg_post = TwoPhaseTrackerConfig(
        v_max=5.0,
        cost_mode="3d",
        use_velocity=False,
        vfc_mode="post",
        vfc_radius=5.0,
        vfc_min_neighbors=3,
        vfc_post_min_tol=0.4,
    )
    links_post = TwoPhaseTracker(cfg_post).track_frames(
        frames, leaves, project_fn=ident
    )
    # Particle 3's link is severed by post-VFC
    p3_link_post = [l for l in links_post if l[1] == 3]
    assert len(p3_link_post) == 0
    # Inlier particles 0, 1, 2 links are preserved
    assert len(links_post) == 3


def test_vfc_mode_both_and_sparse_fallback():
    """Mode 'both' operates seamlessly, and sparse particle count (<min_neighbors) gracefully falls back."""
    # Only 2 particles (below min_neighbors=3)
    frames = [
        np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [6.0, 5.0, 0.0]]),
    ]
    leaves = [ident(f) for f in frames]
    cfg = TwoPhaseTrackerConfig(
        v_max=5.0,
        cost_mode="3d",
        vfc_mode="both",
        vfc_min_neighbors=3,
    )
    tracker = TwoPhaseTracker(cfg)
    links = tracker.track_frames(frames, leaves, project_fn=ident)
    assert len(links) == 2
