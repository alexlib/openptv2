"""Differentiable pinhole camera model + epipolar ray intersection (Phase 2, Stage 2->3).

Prototyped and gradient-verified (``torch.autograd.gradcheck``) in a live
marimo notebook before landing here.
"""

from __future__ import annotations

import torch


def project_pinhole(
    points3d: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    cc: torch.Tensor,
    xh: torch.Tensor,
    yh: torch.Tensor,
    k1: torch.Tensor | float = 0.0,
    k2: torch.Tensor | float = 0.0,
    k3: torch.Tensor | float = 0.0,
    p1: torch.Tensor | float = 0.0,
    p2: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Differentiable Brown-Conrady pinhole projection.

    Parameters
    ----------
    points3d : Tensor (N, 3)
        World points.
    R : Tensor (3, 3)
        Rotation (world -> camera).
    t : Tensor (3,)
        Translation (world -> camera).
    cc : Tensor
        Camera constant (focal length).
    xh, yh : Tensor
        Principal point offset.
    k1, k2, k3, p1, p2 : Tensor or float
        Radial (k) and tangential (p) distortion coefficients.

    Returns
    -------
    Tensor (N, 2)
        Distorted metric image-plane coordinates.
    """
    cam = points3d @ R.T + t
    xn = cam[:, 0] / cam[:, 2]
    yn = cam[:, 1] / cam[:, 2]
    r2 = xn**2 + yn**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
    yd = yn * radial + p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
    x_img = xh + cc * xd
    y_img = yh + cc * yd
    return torch.stack([x_img, y_img], dim=-1)


def closest_point_between_rays(
    o1: torch.Tensor,
    d1: torch.Tensor,
    o2: torch.Tensor,
    d2: torch.Tensor,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable epipolar ray intersection.

    The midpoint of the shortest segment between two skew rays (standard
    closed-form least-squares solution), used to triangulate a 3D point from
    two camera rays without a discrete nearest-neighbor search.

    Parameters
    ----------
    o1, o2 : Tensor (..., 3)
        Ray origins (camera centers).
    d1, d2 : Tensor (..., 3)
        Ray directions (need not be normalized).

    Returns
    -------
    midpoint : Tensor (..., 3)
    miss_distance : Tensor (...)
        Distance between the two closest points (0 for perfectly
        intersecting rays; the "epipolar miss" in the whitepaper).
    """
    d1n = d1 / d1.norm(dim=-1, keepdim=True).clamp_min(eps)
    d2n = d2 / d2.norm(dim=-1, keepdim=True).clamp_min(eps)
    r = o1 - o2
    a = (d1n * d1n).sum(-1)
    b = (d1n * d2n).sum(-1)
    c = (d2n * d2n).sum(-1)
    d = (d1n * r).sum(-1)
    e = (d2n * r).sum(-1)
    denom = (a * c - b * b).clamp_min(eps)
    s = (b * e - c * d) / denom
    t_ = (a * e - b * d) / denom
    p1 = o1 + s.unsqueeze(-1) * d1n
    p2 = o2 + t_.unsqueeze(-1) * d2n
    mid = 0.5 * (p1 + p2)
    miss = (p1 - p2).norm(dim=-1)
    return mid, miss


__all__ = ["project_pinhole", "closest_point_between_rays"]
