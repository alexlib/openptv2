"""Real differentiable-pipeline objective for Sobol sensitivity analysis.

Wires ``openptv2.differentiable``'s Stage 1-5 core to a genuine ground-truth
comparison: how much does each upstream processing parameter bias the
downstream acceleration kurtosis (:math:`\\Delta K_a`), the whitepaper's
central claim -- a Stage-1 micro-parameter propagating through to distort
Stage-5 Lagrangian physics. Answers "which pipeline parameters should we
improve to get better Lagrangian turbulence statistics" with data instead
of guesswork, replacing ``cli_autotune``'s placeholder demo objective.

The same ground-truth trajectory batch (fixed seed) is reused across every
Sobol sample -- otherwise trajectory randomness would swamp the parameter
effect being measured.
"""

from __future__ import annotations

import numpy as np
import torch

from openptv2.benchmarking.jhtdb_client import synthetic_hit_trajectories
from openptv2.differentiable.centroiding import gaussian_moment_fit, soft_threshold
from openptv2.differentiable.geometry import closest_point_between_rays
from openptv2.differentiable.physics_loss import kurtosis
from openptv2.differentiable.tracking import differentiable_savitzky_golay

PARAM_NAMES = ["i_threshold", "threshold_sharpness", "sg_window", "sg_poly_order"]
PARAM_BOUNDS = [
    (0.0, 0.7),  # i_threshold: Stage 1 soft-threshold level
    (2.0, 30.0),  # threshold_sharpness: how hard the Stage-1 gate is
    (3.0, 11.0),  # sg_window: Stage 5 Savitzky-Golay window (rounded to odd int)
    (2.0, 4.0),  # sg_poly_order: Stage 5 poly order (>=2 to estimate acceleration)
]

_N_PARTICLES = 25
_N_FRAMES = 24
_SEED = 0
_CC = 100.0
_PATCH_HW = 9
_PSF_SIGMA = 1.2
_REFERENCE_WINDOW = 7
_REFERENCE_POLY = 2


def _round_sg_params(window_raw: float, poly_raw: float) -> tuple[int, int]:
    window = int(round(window_raw))
    if window % 2 == 0:
        window += 1
    window = max(3, window)
    poly = int(round(poly_raw))
    poly = max(2, min(poly, window - 1))  # need >= poly+1 points to fit
    return window, poly


def _true_trajectories(dtype=torch.float64) -> torch.Tensor:
    """Fixed ground-truth batch, shared across every Sobol sample."""
    traj_np = synthetic_hit_trajectories(_N_PARTICLES, _N_FRAMES, dt=1.0, domain=80.0, seed=_SEED)
    traj = torch.tensor(traj_np, dtype=dtype)
    traj[..., 2] += 200.0  # shift in front of the two toy cameras
    return traj


def _reconstruct(true_pts: torch.Tensor, i_threshold: float, sharpness: float) -> torch.Tensor:
    """Stage 1-3: render -> soft-threshold -> centroid -> triangulate."""
    dtype = true_pts.dtype
    o1 = torch.tensor([-50.0, 0.0, -500.0], dtype=dtype)
    o2 = torch.tensor([50.0, 0.0, -500.0], dtype=dtype)
    n_particles, n_frames, _ = true_pts.shape
    recon = torch.empty_like(true_pts)
    half = _PATCH_HW // 2
    yy, xx = torch.meshgrid(
        torch.arange(_PATCH_HW, dtype=dtype), torch.arange(_PATCH_HW, dtype=dtype), indexing="ij"
    )

    def render(frac):
        px = (frac[:, 0] + half).unsqueeze(-1).unsqueeze(-1)
        py = (frac[:, 1] + half).unsqueeze(-1).unsqueeze(-1)
        return torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * _PSF_SIGMA**2))

    for f in range(n_frames):
        pt = true_pts[:, f, :]
        cam1 = pt - o1
        img1 = torch.stack([cam1[:, 0] / cam1[:, 2] * _CC, cam1[:, 1] / cam1[:, 2] * _CC], dim=-1)
        cam2 = pt - o2
        img2 = torch.stack([cam2[:, 0] / cam2[:, 2] * _CC, cam2[:, 1] / cam2[:, 2] * _CC], dim=-1)
        frac1, frac2 = img1 - img1.round(), img2 - img2.round()

        patch1 = soft_threshold(render(frac1), i_threshold, sharpness)
        patch2 = soft_threshold(render(frac2), i_threshold, sharpness)
        c1 = gaussian_moment_fit(patch1)["centroid"]
        c2 = gaussian_moment_fit(patch2)["centroid"]
        est1 = (img1.round() - half) + c1
        est2 = (img2.round() - half) + c2
        ones = torch.ones(n_particles, dtype=dtype)
        d1 = torch.stack([est1[:, 0] / _CC, est1[:, 1] / _CC, ones], dim=-1)
        d2 = torch.stack([est2[:, 0] / _CC, est2[:, 1] / _CC, ones], dim=-1)
        mid, _miss = closest_point_between_rays(o1, d1, o2, d2)
        recon[:, f, :] = mid
    return recon


def pipeline_delta_kurtosis(params: np.ndarray) -> np.ndarray:
    """Sobol-ready objective: :math:`|\\Delta K_a|` for a batch of
    ``(i_threshold, threshold_sharpness, sg_window, sg_poly_order)`` samples.

    Ground truth :math:`K_a` is computed once (fixed reference SG settings,
    on the noiseless true trajectory) so the comparison isolates each
    parameter's effect rather than re-randomizing the trajectory per sample.
    """
    p = np.atleast_2d(params)
    true_pts = _true_trajectories()
    true_out = differentiable_savitzky_golay(
        true_pts, window=_REFERENCE_WINDOW, poly_order=_REFERENCE_POLY, dt=1.0
    )
    true_ka = float(kurtosis(true_out["acceleration"]))

    out = np.empty(len(p))
    for i, row in enumerate(p):
        i_threshold, sharpness, window_raw, poly_raw = row
        window, poly = _round_sg_params(window_raw, poly_raw)
        recon = _reconstruct(true_pts, float(i_threshold), float(sharpness))
        sg_out = differentiable_savitzky_golay(recon, window=window, poly_order=poly, dt=1.0)
        recon_ka = float(kurtosis(sg_out["acceleration"]))
        out[i] = abs(recon_ka - true_ka)
    return out if params.ndim > 1 else out[0]


__all__ = ["PARAM_NAMES", "PARAM_BOUNDS", "pipeline_delta_kurtosis"]
