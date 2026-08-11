import sys
from pathlib import Path

# Add scripts directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import pytest
from demo_hybrid_strategies import (
    run_strategy_single_pass,
    run_strategy_1_forward_fast_backward_kalman,
    run_strategy_2_two_scale_velocity_cascading,
    compare_all_strategies,
)
import benchmark_utils as bu


@pytest.fixture
def sample_dataset():
    """Returns the synthetic turbulent test dataset path."""
    p = Path("test_data/synthetic_turbulent")
    assert p.exists(), "Synthetic test dataset missing!"
    return p


def test_strategy_single_pass(sample_dataset):
    """Test baseline single pass tracking strategy."""
    tracks, dt = run_strategy_single_pass(src=sample_dataset, first=10001, n_frames=5)
    assert isinstance(tracks, dict)
    assert len(tracks) > 0
    assert dt > 0.0


def test_strategy_1_forward_fast_backward_kalman(sample_dataset):
    """Test Strategy 1: Forward-Fast / Backward-Kalman Hybrid."""
    tracks, dt = run_strategy_1_forward_fast_backward_kalman(src=sample_dataset, first=10001, n_frames=5)
    assert isinstance(tracks, dict)
    assert len(tracks) > 0
    assert dt > 0.0


def test_strategy_2_two_scale_velocity_cascading(sample_dataset):
    """Test Strategy 2: Two-Scale Velocity Cascading."""
    tracks, dt = run_strategy_2_two_scale_velocity_cascading(src=sample_dataset, first=10001, n_frames=5)
    assert isinstance(tracks, dict)
    assert len(tracks) > 0
    assert dt > 0.0


def test_compare_all_strategies(sample_dataset):
    """Test full strategies comparison workflow."""
    rows = compare_all_strategies(src=sample_dataset, first=10001, n_frames=5)
    assert len(rows) == 3
    for r in rows:
        assert "strategy" in r
        assert r["precision"] > 0.80
        assert r["yield_recall"] > 0.10
        assert r["ms_per_frame"] > 0.0
