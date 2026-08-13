"""Gradient checks + basic correctness for openptv2.differentiable (Phase 2).

Verification strategy per docs/plans/differentiable_ptv_nextgen_plan.md:
PyTorch operator gradient checks via torch.autograd.gradcheck.
"""

import pytest

torch = pytest.importorskip("torch")

from openptv2.differentiable.centroiding import gaussian_moment_fit, soft_argmax_2d
from openptv2.differentiable.geometry import closest_point_between_rays, project_pinhole
from openptv2.differentiable.matching import sinkhorn_soft_assign
from openptv2.differentiable.tracking import (
    DifferentiableSegmentTracker,
    differentiable_savitzky_golay,
    savitzky_golay_kernels,
)


def _gaussian_patch(h=9, w=9, cx=4.3, cy=4.1, sigma=1.2):
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij"
    )
    patch = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return patch.unsqueeze(0)


def test_soft_argmax_gradcheck():
    # Soft-argmax is a coarse, differentiable estimator: at low temperature it
    # collapses toward the single peak pixel rather than the true subpixel
    # location, so only gradcheck (not tight accuracy) belongs here --
    # gaussian_moment_fit is the accurate subpixel estimator (see below).
    patch = _gaussian_patch().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p: soft_argmax_2d(p, temperature=0.05), (patch,), fast_mode=True
    )


def test_gaussian_moment_fit_gradcheck_and_accuracy():
    patch = _gaussian_patch(cx=4.3, cy=4.1, sigma=1.2).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p: gaussian_moment_fit(p)["centroid"], (patch,), fast_mode=True
    )
    out = gaussian_moment_fit(patch.detach())
    assert out["centroid"][0, 0] == pytest.approx(4.3, abs=0.01)
    assert out["centroid"][0, 1] == pytest.approx(4.1, abs=0.01)
    assert out["sigma"][0, 0] == pytest.approx(1.2, abs=0.01)


def test_project_pinhole_gradcheck():
    R = torch.eye(3, dtype=torch.float64)
    t = torch.tensor([0.0, 0.0, -400.0], dtype=torch.float64)
    pts = torch.tensor([[0.0, 0.0, 0.0], [10.0, -5.0, 3.0]], dtype=torch.float64, requires_grad=True)
    cc = torch.tensor(100.0, dtype=torch.float64, requires_grad=True)
    k1 = torch.tensor(0.001, dtype=torch.float64, requires_grad=True)
    xh = torch.tensor(0.0, dtype=torch.float64)
    yh = torch.tensor(0.0, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda p, cc, k1: project_pinhole(p, R, t, cc, xh, yh, k1=k1),
        (pts, cc, k1),
        fast_mode=True,
    )


def test_closest_point_between_rays_recovers_known_point():
    true_pt = torch.tensor([5.0, -2.0, 3.0], dtype=torch.float64)
    o1 = torch.tensor([0.0, 0.0, -100.0], dtype=torch.float64, requires_grad=True)
    o2 = torch.tensor([50.0, 0.0, -100.0], dtype=torch.float64, requires_grad=True)
    d1 = (true_pt - o1).detach()
    d2 = (true_pt - o2).detach()

    mid, miss = closest_point_between_rays(o1, d1, o2, d2)
    assert torch.allclose(mid, true_pt, atol=1e-8)
    assert float(miss.detach()) == pytest.approx(0.0, abs=1e-8)
    assert torch.autograd.gradcheck(
        lambda o1, o2: closest_point_between_rays(o1, d1, o2, d2)[0], (o1, o2), fast_mode=True
    )


def test_sinkhorn_soft_assign_converges_to_diagonal_and_gradcheck():
    # Three well-separated clusters: the cheap assignment is the identity.
    cost = torch.tensor(
        [[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    plan = sinkhorn_soft_assign(cost, epsilon=0.05, n_iters=100)
    assert torch.isfinite(plan).all()
    assert bool((plan.diag() > 0.9 / 3).all())
    assert torch.autograd.gradcheck(
        lambda c: sinkhorn_soft_assign(c, epsilon=0.2, n_iters=50), (cost,), fast_mode=True
    )


def test_segment_tracker_links_nearest_particles():
    tracker = DifferentiableSegmentTracker(epsilon=0.05, n_iters=100)
    pos_t = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float64)
    pos_t1 = torch.tensor([[0.2, 0.1], [10.1, -0.1]], dtype=torch.float64)
    plan = tracker(pos_t, pos_t1)
    linked = plan.argmax(dim=1)
    assert linked.tolist() == [0, 1]


def test_differentiable_savitzky_golay_recovers_constant_acceleration():
    t = torch.arange(20, dtype=torch.float64)
    true_v, true_a = 2.0, 0.3
    x = true_v * t + 0.5 * true_a * t**2
    pos = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1).unsqueeze(0)
    pos.requires_grad_(True)

    out = differentiable_savitzky_golay(pos, window=5, poly_order=2, dt=1.0)
    assert out["acceleration"][0, 5, 0].item() == pytest.approx(true_a, abs=1e-8)
    assert torch.autograd.gradcheck(
        lambda p: differentiable_savitzky_golay(p, window=5, poly_order=2)["acceleration"],
        (pos,),
        fast_mode=True,
    )


def test_savitzky_golay_kernels_rejects_poly_order_below_2():
    """poly_order=1 (linear fit) has no 2nd-derivative term -- acceleration
    can't be estimated, so this must fail loudly, not with an IndexError."""
    with pytest.raises(ValueError, match="poly_order must be >= 2"):
        savitzky_golay_kernels(window=5, poly_order=1)
