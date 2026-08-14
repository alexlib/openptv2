"""estimate_level1_dist_weight must tell a slow/dense flow (motion << particle
spacing) from a fast/sparse one (motion ~ particle spacing) apart, and must
fall back safely on too little data."""

import numpy as np

from openptv2.algorithms.track3d import estimate_level1_dist_weight


def test_slow_dense_flow_gets_a_higher_weight_than_fast_sparse_flow():
    rng = np.random.default_rng(0)

    # Motion tiny compared to particle spacing (~test_cavity's regime).
    slow = rng.uniform(-50, 50, (500, 3))
    slow_next = slow + rng.normal(0, 0.15, (500, 3))

    # Motion comparable to particle spacing.
    fast = rng.uniform(-10, 10, (60, 3))
    fast_next = fast + rng.normal([2.0, 0.0, 0.0], 0.3, (60, 3))

    w_slow = estimate_level1_dist_weight(slow, slow_next)
    w_fast = estimate_level1_dist_weight(fast, fast_next)

    assert w_slow > w_fast


def test_falls_back_to_default_with_too_few_points():
    a = np.zeros((0, 3))
    b = np.zeros((0, 3))
    assert estimate_level1_dist_weight(a, b) == 1.0
