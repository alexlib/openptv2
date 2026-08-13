"""Sobol sensitivity + optimization agent + CLI smoke tests (Phase 4)."""

import json

import numpy as np
import pytest

from openptv2.autoresearch.agent import AutoResearchAgent
from openptv2.autoresearch.cli_autotune import build_parser, demo_objective
from openptv2.autoresearch.sensitivity import sobol_indices


def test_sobol_indices_match_analytic_linear_function():
    """y = x1 + 2*x2, independent uniform[0,1]: analytic S1 = [0.2, 0.8], no
    interaction so ST == S1."""

    def linear(x):
        return x[:, 0] + 2.0 * x[:, 1]

    result = sobol_indices(linear, bounds=[(0.0, 1.0), (0.0, 1.0)], n_base=4096, seed=0)
    assert result["S1"] == pytest.approx([0.2, 0.8], abs=0.02)
    assert result["ST"] == pytest.approx([0.2, 0.8], abs=0.02)


def test_agent_sensitivity_matches_sobol_indices():
    def linear(x):
        return x[:, 0] + 2.0 * x[:, 1]

    agent = AutoResearchAgent()
    result = agent.sensitivity(linear, bounds=[(0.0, 1.0), (0.0, 1.0)], n_base=1024, seed=1)
    assert result["S1"][1] > result["S1"][0]  # x2 has 4x the variance contribution


def test_optimize_blackbox_finds_demo_minimum():
    agent = AutoResearchAgent()
    result = agent.optimize_blackbox(
        lambda p: float(demo_objective(p)), bounds=[(-5.0, 5.0), (-5.0, 5.0)], seed=0, maxiter=50
    )
    # The bowl's global minimum is near (3, -1); allow slack for the mild
    # sinusoidal perturbation shifting it slightly.
    assert result.best_params[0] == pytest.approx(3.0, abs=0.5)
    assert result.best_params[1] == pytest.approx(-1.0, abs=0.5)


def test_optimize_gradient_converges_on_quadratic():
    torch = pytest.importorskip("torch")

    def quadratic(params):
        return ((params - torch.tensor([2.0, -3.0], dtype=torch.float64)) ** 2).sum()

    agent = AutoResearchAgent()
    result = agent.optimize_gradient(quadratic, init_params=[0.0, 0.0], n_steps=300, lr=0.1)
    assert result.best_params[0] == pytest.approx(2.0, abs=0.05)
    assert result.best_params[1] == pytest.approx(-3.0, abs=0.05)
    assert result.history[-1] < result.history[0]


def test_cli_sensitivity_demo(capsys, tmp_path):
    parser = build_parser()
    out_json = tmp_path / "sens.json"
    args = parser.parse_args(["sensitivity", "--demo", "--n-base", "64", "--json", str(out_json)])
    args.func(args)
    captured = capsys.readouterr()
    assert "S1" in captured.out
    data = json.loads(out_json.read_text())
    assert set(data) == {"S1", "ST"}


def test_cli_optimize_demo_blackbox(capsys):
    parser = build_parser()
    args = parser.parse_args(["optimize", "--demo", "--method", "blackbox", "--n-iters", "30"])
    args.func(args)
    captured = capsys.readouterr()
    assert "best_params" in captured.out


def test_cli_pareto_demo(capsys):
    parser = build_parser()
    args = parser.parse_args(["pareto", "--demo", "--n-samples", "10"])
    args.func(args)
    captured = capsys.readouterr()
    assert "Pareto front" in captured.out


def test_demo_objective_batched_and_single():
    single = demo_objective(np.array([3.0, -1.0]))
    batched = demo_objective(np.array([[3.0, -1.0], [0.0, 0.0]]))
    assert np.isscalar(single) or single.ndim == 0
    assert batched.shape == (2,)
    assert batched[0] == pytest.approx(float(single))
