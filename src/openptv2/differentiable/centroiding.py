"""Differentiable subpixel 2D target centroiding (Phase 2, Stage 1->2).

Prototyped and gradient-verified (``torch.autograd.gradcheck``) in a live
marimo notebook before landing here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_threshold(
    image: torch.Tensor, i_threshold: torch.Tensor | float, sharpness: float = 15.0
) -> torch.Tensor:
    """Differentiable Stage-1 soft-thresholding.

    A sigmoid gate around ``i_threshold`` replaces the discrete
    ``image > threshold`` mask, so gradients flow into ``i_threshold``.

    Parameters
    ----------
    image : Tensor
        Raw intensity image or patch.
    i_threshold : Tensor or float
        Intensity threshold.
    sharpness : float
        Gate steepness; higher values approach a hard cutoff.
    """
    gate = torch.sigmoid(sharpness * (image - i_threshold))
    return image * gate


def soft_argmax_2d(patches: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Subpixel centroid via softmax-weighted coordinates.

    Parameters
    ----------
    patches : Tensor (N, H, W)
        Intensity patches.
    temperature : float
        Softmax temperature; lower values sharpen the weighting toward the
        peak (approaching a hard argmax as temperature -> 0).

    Returns
    -------
    Tensor (N, 2)
        Patch-local (x, y) subpixel centroid.
    """
    n, h, w = patches.shape
    flat = (patches / temperature).reshape(n, h * w)
    weights = F.softmax(flat, dim=-1).reshape(n, h, w)
    ys = torch.arange(h, dtype=patches.dtype, device=patches.device)
    xs = torch.arange(w, dtype=patches.dtype, device=patches.device)
    cy = (weights.sum(dim=2) * ys).sum(dim=1)
    cx = (weights.sum(dim=1) * xs).sum(dim=1)
    return torch.stack([cx, cy], dim=-1)


class SoftArgmax2D(torch.nn.Module):
    """Module wrapper around :func:`soft_argmax_2d`."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return soft_argmax_2d(patches, temperature=self.temperature)


def gaussian_moment_fit(patches: torch.Tensor) -> dict[str, torch.Tensor]:
    """Closed-form differentiable 2D Gaussian fit via image moments.

    Avoids an iterative (non-cleanly-differentiable) optimizer: the centroid
    and spread come directly from the intensity-weighted first and second
    moments, which are differentiable everywhere the patch mass is nonzero.

    Parameters
    ----------
    patches : Tensor (N, H, W)
        Non-negative intensity patches.

    Returns
    -------
    dict
        ``centroid`` (N, 2) (x, y), ``sigma`` (N, 2) (x, y std), ``amplitude``
        (N,) total patch mass.
    """
    n, h, w = patches.shape
    ys = torch.arange(h, dtype=patches.dtype, device=patches.device)
    xs = torch.arange(w, dtype=patches.dtype, device=patches.device)
    mass = patches.sum(dim=(1, 2)).clamp_min(1e-8)
    cy = (patches.sum(dim=2) * ys).sum(dim=1) / mass
    cx = (patches.sum(dim=1) * xs).sum(dim=1) / mass
    dy = ys[None, :] - cy[:, None]
    dx = xs[None, :] - cx[:, None]
    var_y = (patches.sum(dim=2) * dy**2).sum(dim=1) / mass
    var_x = (patches.sum(dim=1) * dx**2).sum(dim=1) / mass
    sigma = torch.stack([var_x.clamp_min(1e-8).sqrt(), var_y.clamp_min(1e-8).sqrt()], dim=-1)
    return {"centroid": torch.stack([cx, cy], dim=-1), "sigma": sigma, "amplitude": mass}


class DifferentiableGaussianFit(torch.nn.Module):
    """Module wrapper around :func:`gaussian_moment_fit`."""

    def forward(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        return gaussian_moment_fit(patches)


__all__ = [
    "soft_threshold",
    "soft_argmax_2d",
    "SoftArgmax2D",
    "gaussian_moment_fit",
    "DifferentiableGaussianFit",
]
