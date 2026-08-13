"""Differentiable soft stereo correspondence via Sinkhorn optimal transport
(Phase 2, Stage 3).

Prototyped and gradient-verified (``torch.autograd.gradcheck``) in a live
marimo notebook before landing here.
"""

from __future__ import annotations

import torch


def sinkhorn_soft_assign(cost: torch.Tensor, epsilon: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    """Entropy-regularized optimal transport (log-domain Sinkhorn).

    Parameters
    ----------
    cost : Tensor (N, M)
        Cost matrix (e.g. epipolar-miss distance between camera-1 and
        camera-2 candidate points). Uniform marginals.
    epsilon : float
        Entropy regularization strength; smaller values sharpen the
        assignment toward a hard permutation.
    n_iters : int
        Number of Sinkhorn iterations.

    Returns
    -------
    Tensor (N, M)
        Soft assignment / transport plan.
    """
    n, m = cost.shape
    log_mu = -torch.log(torch.full((n,), float(n), dtype=cost.dtype, device=cost.device))
    log_nu = -torch.log(torch.full((m,), float(m), dtype=cost.dtype, device=cost.device))
    u = torch.zeros(n, dtype=cost.dtype, device=cost.device)
    v = torch.zeros(m, dtype=cost.dtype, device=cost.device)
    K = -cost / epsilon
    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(K + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(K + u[:, None], dim=0)
    return torch.exp(K + u[:, None] + v[None, :])


class SoftSinkhornMatcher(torch.nn.Module):
    """Differentiable soft stereo correspondence via entropy-regularized OT."""

    def __init__(self, epsilon: float = 0.1, n_iters: int = 50):
        super().__init__()
        self.epsilon = epsilon
        self.n_iters = n_iters

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        return sinkhorn_soft_assign(cost, epsilon=self.epsilon, n_iters=self.n_iters)


__all__ = ["sinkhorn_soft_assign", "SoftSinkhornMatcher"]
