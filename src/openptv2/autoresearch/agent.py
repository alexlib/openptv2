"""Auto-Research optimization agent (Phase 4).

Tunes pipeline hyperparameters two ways:

* :meth:`AutoResearchAgent.optimize_gradient` -- Adam over a differentiable
  PyTorch objective (the Phase 2/3 ``openptv2.differentiable`` runtime).
* :meth:`AutoResearchAgent.optimize_blackbox` -- ``scipy.optimize``'s
  differential evolution (population-based, gradient-free) for the legacy
  C/Cython runtime, where no autograd path exists.

:meth:`AutoResearchAgent.sensitivity` wraps
:mod:`openptv2.autoresearch.sensitivity` for Sobol variance decomposition
ahead of either optimization -- which parameters are worth tuning at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from openptv2.autoresearch.sensitivity import sobol_indices


@dataclass
class OptimizationResult:
    """Result of either optimization path."""

    best_params: np.ndarray
    best_loss: float
    history: list[float] = field(default_factory=list)


class AutoResearchAgent:
    """Global sensitivity analysis + optimization over pipeline parameters."""

    def sensitivity(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        bounds: list[tuple[float, float]],
        n_base: int = 256,
        seed: int | None = 0,
    ) -> dict[str, np.ndarray]:
        """Sobol first/total-order indices for ``objective`` over ``bounds``."""
        return sobol_indices(objective, bounds, n_base=n_base, seed=seed)

    def optimize_gradient(
        self,
        loss_fn: Callable,
        init_params,
        n_steps: int = 200,
        lr: float = 0.05,
    ) -> OptimizationResult:
        """Adam optimization over a differentiable ``loss_fn(params) -> scalar Tensor``.

        Parameters
        ----------
        loss_fn : callable
            Takes a ``torch.Tensor`` of shape ``(k,)`` and returns a scalar
            differentiable loss.
        init_params : array-like
            Starting parameter values, shape ``(k,)``.
        """
        import torch

        params = torch.as_tensor(init_params, dtype=torch.float64).clone().requires_grad_(True)
        opt = torch.optim.Adam([params], lr=lr)
        history: list[float] = []
        for _ in range(n_steps):
            opt.zero_grad()
            loss = loss_fn(params)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        return OptimizationResult(
            best_params=params.detach().numpy(), best_loss=history[-1], history=history
        )

    def optimize_blackbox(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        seed: int | None = 0,
        maxiter: int = 100,
    ) -> OptimizationResult:
        """Gradient-free optimization via ``scipy.optimize.differential_evolution``.

        For the legacy Cython runtime, or any objective with no autograd path.
        """
        from scipy.optimize import differential_evolution

        result = differential_evolution(objective, bounds, seed=seed, maxiter=maxiter, polish=True)
        return OptimizationResult(best_params=result.x, best_loss=float(result.fun), history=[])


__all__ = ["AutoResearchAgent", "OptimizationResult"]
