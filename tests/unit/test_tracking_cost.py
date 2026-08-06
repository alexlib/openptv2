"""Unit tests for openptv2.tracking_cost."""

import numpy as np
import pytest

from openptv2.tracking_cost import (
    CostWeights,
    compute_multi_term_cost_matrix,
    compute_velocity_aligned_search_radius,
)


def test_cost_weights_normalize():
    """Verify weights normalization logic."""
    cw = CostWeights(
        w_distance=2.0, w_velocity=2.0, w_acceleration=0.0, w_intensity=0.0
    )
    norm = cw.normalize()
    assert norm.w_distance == pytest.approx(0.5)
    assert norm.w_velocity == pytest.approx(0.5)
    assert norm.w_acceleration == pytest.approx(0.0)
    assert norm.w_intensity == pytest.approx(0.0)


def test_velocity_aligned_search_radius():
    """Verify adaptive search radius calculation along velocity direction."""
    velocities = np.array(
        [
            [0.0, 0.0, 0.0],  # unseeded / zero speed
            [10.0, 0.0, 0.0],  # moving particle
        ]
    )
    r_long, r_trans = compute_velocity_aligned_search_radius(
        velocities, v_max=5.0, a_max=2.0, aspect_ratio=2.5
    )

    assert r_long[0] == pytest.approx(5.0)  # zero speed -> v_max
    assert r_trans[0] == pytest.approx(5.0)

    assert r_long[1] == pytest.approx(5.0)  # 2.0 * 2.5
    assert r_trans[1] == pytest.approx(2.0)


def test_multi_term_cost_matrix_distance_only():
    """Verify distance cost computation matches Euclidean distance."""
    pred_pos = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    cand_pos = np.array([[3.0, 4.0, 0.0], [10.0, 0.0, 0.0]])

    cost = compute_multi_term_cost_matrix(
        pred_pos, cand_pos, weights=CostWeights(1.0, 0.0, 0.0, 0.0)
    )

    assert cost.shape == (2, 2)
    assert cost[0, 0] == pytest.approx(5.0)  # hypot(3, 4)
    assert cost[1, 1] == pytest.approx(0.0)


def test_multi_term_cost_matrix_with_intensity():
    """Verify multi-term cost matrix incorporates intensity similarity."""
    pred_pos = np.array([[0.0, 0.0, 0.0]])
    cand_pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    pred_intensity = np.array([100.0])
    cand_intensity = np.array([100.0, 200.0])

    weights = CostWeights(
        w_distance=0.5, w_velocity=0.0, w_acceleration=0.0, w_intensity=0.5
    )

    cost = compute_multi_term_cost_matrix(
        pred_pos,
        cand_pos,
        pred_intensity=pred_intensity,
        cand_intensity=cand_intensity,
        weights=weights,
    )

    # Candidate 0: dist=0, intensity_diff=0 -> cost=0
    assert cost[0, 0] == pytest.approx(0.0)
    # Candidate 1: dist=0, intensity_diff=100 -> cost=0.5 * 100 = 50.0
    assert cost[0, 1] == pytest.approx(50.0)
