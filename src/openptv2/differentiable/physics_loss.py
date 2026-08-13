"""Differentiable Lagrangian turbulence physics loss (Phase 3).

.. math::
    \\mathcal{L}_{\\text{total}} = w_1 |\\Delta K_a| + w_2 |\\Delta E_L(\\omega)|
        - w_3 \\min(1, \\langle T \\rangle / (20 \\Delta t))
        + w_4 \\text{ReprojectionError} + w_5 \\text{GhostPenalty}

Prototyped and gradient-verified end-to-end (Stage 1 ``soft_threshold``
through Stage 5 acceleration to this loss, ``d L / d I_threshold != 0``) in a
live marimo notebook before landing here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def kurtosis(accel: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Flatness/kurtosis of the acceleration distribution: :math:`K_a = E[a^4] / E[a^2]^2`.

    For a zero-mean Gaussian this equals 3; real turbulent acceleration PDFs
    are strongly non-Gaussian (:math:`K_a \\gg 3`), which is exactly the
    signal a false tracking "crossing swap" artificially inflates.
    """
    a = accel.reshape(-1)
    m2 = (a**2).mean()
    m4 = (a**4).mean()
    return m4 / (m2**2).clamp_min(eps)


def delta_kurtosis_loss(pred_accel: torch.Tensor, target_ka: torch.Tensor | float) -> torch.Tensor:
    """:math:`|\\Delta K_a|` between predicted acceleration and a target kurtosis."""
    return (kurtosis(pred_accel) - target_ka).abs()


def velocity_power_spectrum(velocity: torch.Tensor) -> torch.Tensor:
    """Lagrangian velocity power spectral density :math:`E_L(\\omega)` via FFT.

    Parameters
    ----------
    velocity : Tensor (..., T)
        Time series along the last dimension (one per particle).

    Returns
    -------
    Tensor (..., T // 2 + 1)
    """
    v = velocity - velocity.mean(dim=-1, keepdim=True)
    spec = torch.fft.rfft(v, dim=-1)
    return (spec.abs() ** 2) / v.shape[-1]


def spectral_loss(pred_velocity: torch.Tensor, target_psd: torch.Tensor) -> torch.Tensor:
    """:math:`|\\Delta E_L(\\omega)|`-style loss: MSE between predicted and target PSD.

    ``pred_velocity`` is averaged over all leading (particle) dimensions
    before comparison; ``target_psd`` is the already-averaged reference.
    """
    pred_psd = velocity_power_spectrum(pred_velocity)
    pred_psd = pred_psd.mean(dim=tuple(range(pred_velocity.dim() - 1))) if pred_velocity.dim() > 1 else pred_psd
    n = min(pred_psd.shape[-1], target_psd.shape[-1])
    return (pred_psd[..., :n] - target_psd[..., :n]).pow(2).mean()


def track_length_term(track_lengths: torch.Tensor, dt: float, tau_l: float = 20.0) -> torch.Tensor:
    """:math:`\\min(1, \\langle T \\rangle / (\\tau_L \\Delta t))` -- rewards longer tracks."""
    mean_t = track_lengths.to(torch.float64).mean()
    return torch.clamp(mean_t / (tau_l * dt), max=1.0)


def reprojection_error(observed_px: torch.Tensor, projected_px: torch.Tensor) -> torch.Tensor:
    """RMS pixel reprojection error."""
    return (observed_px - projected_px).pow(2).sum(-1).mean().sqrt()


def ghost_penalty(assignment_plan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean row entropy of a soft assignment plan.

    A peaked (low-entropy) row is a confident, unambiguous link; high entropy
    signals the kind of ambiguous assignment that lets a ghost particle get
    linked into a real track.
    """
    p = assignment_plan.clamp_min(eps)
    row_entropy = -(p * p.log()).sum(-1)
    return row_entropy.mean()


@dataclass
class PhysicsLossWeights:
    """Weights for :func:`total_physics_loss`."""

    w1: float = 1.0
    w2: float = 1.0
    w3: float = 1.0
    w4: float = 1.0
    w5: float = 1.0
    tau_l: float = 20.0


def total_physics_loss(
    pred_accel: torch.Tensor,
    target_ka: torch.Tensor | float,
    pred_velocity: torch.Tensor,
    target_psd: torch.Tensor,
    track_lengths: torch.Tensor,
    dt: float,
    observed_px: torch.Tensor,
    projected_px: torch.Tensor,
    assignment_plan: torch.Tensor,
    weights: PhysicsLossWeights = PhysicsLossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine all five loss terms into the total Lagrangian physics loss.

    Returns
    -------
    total : Tensor (scalar)
    components : dict[str, Tensor]
        Each term before weighting, for logging/dashboards.
    """
    d_ka = delta_kurtosis_loss(pred_accel, target_ka)
    d_e = spectral_loss(pred_velocity, target_psd)
    t_term = track_length_term(track_lengths, dt, weights.tau_l)
    reproj = reprojection_error(observed_px, projected_px)
    ghost = ghost_penalty(assignment_plan)

    total = (
        weights.w1 * d_ka
        + weights.w2 * d_e
        - weights.w3 * t_term
        + weights.w4 * reproj
        + weights.w5 * ghost
    )
    components = {
        "delta_kurtosis": d_ka,
        "delta_spectral": d_e,
        "track_length_term": t_term,
        "reprojection_error": reproj,
        "ghost_penalty": ghost,
    }
    return total, components


__all__ = [
    "kurtosis",
    "delta_kurtosis_loss",
    "velocity_power_spectrum",
    "spectral_loss",
    "track_length_term",
    "reprojection_error",
    "ghost_penalty",
    "PhysicsLossWeights",
    "total_physics_loss",
]
