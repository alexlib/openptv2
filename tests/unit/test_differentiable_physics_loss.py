"""Physics loss unit checks + end-to-end gradient flow (Phase 3).

Verification strategy per docs/plans/differentiable_ptv_nextgen_plan.md:
validate end-to-end gradient flow, verify
d(L_total) / d(I_threshold) != 0 -- i.e. a Stage-1 micro-parameter 100 steps
upstream must influence the Stage-5 Lagrangian physics loss.
"""

import pytest

torch = pytest.importorskip("torch")

from openptv2.differentiable.centroiding import gaussian_moment_fit, soft_threshold
from openptv2.differentiable.geometry import closest_point_between_rays
from openptv2.differentiable.physics_loss import (
    PhysicsLossWeights,
    ghost_penalty,
    kurtosis,
    reprojection_error,
    spectral_loss,
    total_physics_loss,
    track_length_term,
    velocity_power_spectrum,
)
from openptv2.differentiable.tracking import differentiable_savitzky_golay


def test_kurtosis_of_gaussian_is_three():
    g = torch.Generator().manual_seed(0)
    gauss = torch.randn(200_000, generator=g, dtype=torch.float64)
    assert kurtosis(gauss).item() == pytest.approx(3.0, abs=0.05)


def test_spectral_loss_self_comparison_is_zero():
    v = torch.randn(5, 64, dtype=torch.float64)
    target = velocity_power_spectrum(v).mean(dim=0)
    assert spectral_loss(v, target).item() == pytest.approx(0.0, abs=1e-10)


def test_track_length_term_saturates_at_one():
    short = torch.tensor([2.0, 3.0])
    long = torch.tensor([100.0, 200.0])
    assert track_length_term(short, dt=1.0, tau_l=20.0).item() < 0.2
    assert track_length_term(long, dt=1.0, tau_l=20.0).item() == pytest.approx(1.0)


def test_ghost_penalty_peaked_vs_uniform():
    peaked = torch.tensor([[0.98, 0.01, 0.01]], dtype=torch.float64)
    uniform = torch.full((1, 3), 1 / 3, dtype=torch.float64)
    assert ghost_penalty(peaked).item() < ghost_penalty(uniform).item()


def test_total_physics_loss_combines_all_terms():
    accel = torch.randn(10, dtype=torch.float64)
    vel = torch.randn(3, 20, dtype=torch.float64)
    total, components = total_physics_loss(
        pred_accel=accel,
        target_ka=3.0,
        pred_velocity=vel,
        target_psd=velocity_power_spectrum(vel).mean(dim=0),
        track_lengths=torch.tensor([50.0, 60.0]),
        dt=1.0,
        observed_px=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
        projected_px=torch.tensor([[1.1, 1.9]], dtype=torch.float64),
        assignment_plan=torch.tensor([[0.9, 0.1]], dtype=torch.float64),
        weights=PhysicsLossWeights(),
    )
    assert set(components) == {
        "delta_kurtosis",
        "delta_spectral",
        "track_length_term",
        "reprojection_error",
        "ghost_penalty",
    }
    assert torch.isfinite(total)


def _render_gaussian_patch(peak_xy, patch_hw=9, sigma=1.2, dtype=torch.float64):
    h = w = patch_hw
    yy, xx = torch.meshgrid(torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij")
    px, py = peak_xy[..., 0], peak_xy[..., 1]
    return torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma**2))


def _pinhole_project(point3d, origin, cc):
    cam = point3d - origin
    return torch.stack([cam[..., 0] / cam[..., 2] * cc, cam[..., 1] / cam[..., 2] * cc], dim=-1)


def _pinhole_ray_dir(img_xy, cc):
    x, y = img_xy[..., 0], img_xy[..., 1]
    return torch.stack([x / cc, y / cc, torch.ones_like(x)], dim=-1)


def test_end_to_end_gradient_flows_from_threshold_to_physics_loss():
    """The whitepaper's central claim: a Stage-1 intensity threshold must
    influence the Stage-5 physics loss through centroiding -> triangulation
    -> Savitzky-Golay acceleration -> kurtosis.
    """
    dtype = torch.float64
    cc = 100.0
    patch_hw = 9
    n_frames = 12
    o1 = torch.tensor([-50.0, 0.0, -500.0], dtype=dtype)
    o2 = torch.tensor([50.0, 0.0, -500.0], dtype=dtype)

    t = torch.arange(n_frames, dtype=dtype)
    true_v, true_a = 2.0, 0.4
    x = true_v * t + 0.5 * true_a * t**2
    true_pts = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
    true_pts = true_pts + torch.tensor([0.0, 0.0, 200.0], dtype=dtype)

    def pipeline(i_threshold):
        recon = []
        for f in range(n_frames):
            pt = true_pts[f]
            img1 = _pinhole_project(pt, o1, cc)
            img2 = _pinhole_project(pt, o2, cc)
            frac1 = img1 - img1.round()
            frac2 = img2 - img2.round()
            patch1 = _render_gaussian_patch(frac1 + patch_hw // 2, patch_hw, dtype=dtype)
            patch2 = _render_gaussian_patch(frac2 + patch_hw // 2, patch_hw, dtype=dtype)
            patch1 = soft_threshold(patch1, i_threshold, sharpness=15.0)
            patch2 = soft_threshold(patch2, i_threshold, sharpness=15.0)
            c1 = gaussian_moment_fit(patch1.unsqueeze(0))["centroid"][0]
            c2 = gaussian_moment_fit(patch2.unsqueeze(0))["centroid"][0]
            est_img1 = (img1.round() - patch_hw // 2) + c1
            est_img2 = (img2.round() - patch_hw // 2) + c2
            d1 = _pinhole_ray_dir(est_img1, cc)
            d2 = _pinhole_ray_dir(est_img2, cc)
            mid, _miss = closest_point_between_rays(o1, d1, o2, d2)
            recon.append(mid)
        traj = torch.stack(recon, dim=0).unsqueeze(0)
        out = differentiable_savitzky_golay(traj, window=5, poly_order=2, dt=1.0)
        return kurtosis(out["acceleration"])

    i_threshold = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    loss = pipeline(i_threshold)
    loss.backward()

    assert i_threshold.grad is not None
    assert i_threshold.grad.item() != pytest.approx(0.0, abs=1e-8)
