"""Real differentiable-pipeline sensitivity objective (auto-research follow-up).

Verifies the ground-truth-vs-reconstructed Delta K_a objective used to
Sobol-rank which Stage 1-5 parameters matter most for Lagrangian physics
fidelity -- see openptv2.autoresearch.pipeline_objective.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openptv2.autoresearch.pipeline_objective import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    _round_sg_params,
    pipeline_delta_kurtosis,
)


def test_round_sg_params_enforces_valid_window_and_poly():
    # window rounds to nearest odd >= 3
    assert _round_sg_params(4.0, 2.0)[0] % 2 == 1
    assert _round_sg_params(2.0, 2.0)[0] >= 3
    # poly_order clipped into [2, window - 1]
    window, poly = _round_sg_params(3.0, 4.0)
    assert 2 <= poly <= window - 1


def test_pipeline_delta_kurtosis_single_and_batch():
    single = pipeline_delta_kurtosis(np.array([0.2, 10.0, 7.0, 2.0]))
    batch = pipeline_delta_kurtosis(np.array([[0.2, 10.0, 7.0, 2.0], [0.2, 10.0, 7.0, 2.0]]))
    assert np.isscalar(single) or np.ndim(single) == 0
    assert batch.shape == (2,)
    assert batch[0] == pytest.approx(batch[1])  # same params -> same fixed ground truth -> equal
    assert batch[0] == pytest.approx(float(single))


def test_aggressive_threshold_causes_far_more_bias_than_near_zero():
    """The whitepaper's core claim, quantified: a harsher Stage-1 gate must
    bias the Stage-5 kurtosis far more than a near-inert one."""
    gentle = pipeline_delta_kurtosis(np.array([0.001, 5.0, 7.0, 2.0]))
    aggressive = pipeline_delta_kurtosis(np.array([0.6, 25.0, 7.0, 2.0]))
    assert float(aggressive) > 10 * float(gentle)


def test_param_names_match_bounds_length():
    assert len(PARAM_NAMES) == len(PARAM_BOUNDS)
