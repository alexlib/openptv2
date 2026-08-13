"""Differentiable frame-to-frame linkage + smooth velocity/acceleration
(Phase 2, Stage 4->5).

Prototyped and gradient-verified (``torch.autograd.gradcheck``) in a live
marimo notebook before landing here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from openptv2.differentiable.matching import SoftSinkhornMatcher


class DifferentiableSegmentTracker(torch.nn.Module):
    """Soft frame-to-frame particle linkage.

    Builds a cost matrix from Euclidean distance between (optionally
    velocity-predicted) positions at frame ``t`` and observed positions at
    frame ``t+1``, then reuses :class:`~openptv2.differentiable.matching.SoftSinkhornMatcher`
    (the same entropy-regularized OT as the Stage-3 stereo matcher) to produce
    a differentiable soft link matrix instead of a discrete nearest-neighbor
    search.
    """

    def __init__(self, epsilon: float = 0.1, n_iters: int = 50):
        super().__init__()
        self.matcher = SoftSinkhornMatcher(epsilon=epsilon, n_iters=n_iters)

    def forward(
        self,
        pos_t: torch.Tensor,
        pos_t1: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pos_t : Tensor (N, D)
            Positions at frame t.
        pos_t1 : Tensor (M, D)
            Positions at frame t+1.
        velocity : Tensor (N, D), optional
            Predicted per-particle velocity used to advance ``pos_t`` before
            costing (constant-velocity gating).

        Returns
        -------
        Tensor (N, M)
            Soft link / transport plan.
        """
        predicted = pos_t + velocity if velocity is not None else pos_t
        cost = torch.cdist(predicted, pos_t1)
        return self.matcher(cost)


def savitzky_golay_kernels(
    window: int = 5, poly_order: int = 2, dt: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute Savitzky-Golay convolution kernels (position, velocity,
    acceleration) from a local least-squares polynomial fit.

    Kernel construction is a fixed (non-learned) linear solve; the resulting
    filter is differentiable because it is applied as a plain ``conv1d`` over
    the (differentiable) trajectory.
    """
    if poly_order < 2:
        raise ValueError(
            f"poly_order must be >= 2 to estimate acceleration (2nd derivative), got {poly_order}"
        )
    half = window // 2
    t = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(t, poly_order + 1, increasing=True)
    pinv = np.linalg.pinv(A)  # (poly_order + 1, window)
    pos_kernel = pinv[0]
    vel_kernel = pinv[1] / dt
    acc_kernel = 2.0 * pinv[2] / dt**2
    return (
        torch.tensor(pos_kernel, dtype=torch.float64),
        torch.tensor(vel_kernel, dtype=torch.float64),
        torch.tensor(acc_kernel, dtype=torch.float64),
    )


def differentiable_savitzky_golay(
    positions: torch.Tensor, window: int = 5, poly_order: int = 2, dt: float = 1.0
) -> dict[str, torch.Tensor]:
    """Smooth position and derive velocity/acceleration via a differentiable
    Savitzky-Golay filter.

    Parameters
    ----------
    positions : Tensor (N, T, D)
        Per-particle position sequences (D=3 for x, y, z).

    Returns
    -------
    dict
        ``position``, ``velocity``, ``acceleration``, each (N, T', D) with
        ``T' = T - window + 1`` (valid-mode convolution).
    """
    pos_k, vel_k, acc_k = savitzky_golay_kernels(window, poly_order, dt)
    pos_k = pos_k.to(positions.dtype).flip(0).view(1, 1, -1)
    vel_k = vel_k.to(positions.dtype).flip(0).view(1, 1, -1)
    acc_k = acc_k.to(positions.dtype).flip(0).view(1, 1, -1)

    n, t_len, d = positions.shape
    x = positions.permute(0, 2, 1).reshape(n * d, 1, t_len)
    smoothed = F.conv1d(x, pos_k).reshape(n, d, -1).permute(0, 2, 1)
    velocity = F.conv1d(x, vel_k).reshape(n, d, -1).permute(0, 2, 1)
    acceleration = F.conv1d(x, acc_k).reshape(n, d, -1).permute(0, 2, 1)
    return {"position": smoothed, "velocity": velocity, "acceleration": acceleration}


class DifferentiableSavitzkyGolay(torch.nn.Module):
    """Module wrapper around :func:`differentiable_savitzky_golay`."""

    def __init__(self, window: int = 5, poly_order: int = 2, dt: float = 1.0):
        super().__init__()
        self.window = window
        self.poly_order = poly_order
        self.dt = dt

    def forward(self, positions: torch.Tensor) -> dict[str, torch.Tensor]:
        return differentiable_savitzky_golay(positions, self.window, self.poly_order, self.dt)


__all__ = [
    "DifferentiableSegmentTracker",
    "savitzky_golay_kernels",
    "differentiable_savitzky_golay",
    "DifferentiableSavitzkyGolay",
]
