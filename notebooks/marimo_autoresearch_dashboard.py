# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "torch",
#     "matplotlib",
#     "wigglystuff",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # OpenPTV³ Auto-Research Dashboard

    Live demonstration of the OpenPTV³ differentiable pipeline
    (`docs/plans/differentiable_ptv_nextgen_plan.md`): a Stage-1 intensity
    threshold, dragged with the slider below, backpropagates through
    soft-argmax centroiding, epipolar triangulation, and a differentiable
    Savitzky-Golay filter, all the way to the Stage-5 Lagrangian physics loss
    (acceleration kurtosis $K_a$).

    Self-contained (no `openptv2` install needed) so it runs zero-setup via
    `molab.marimo.io` with a free cloud GPU attached. The differentiable core
    mirrors `openptv2.differentiable`; ground truth is a synthetic
    Ornstein-Uhlenbeck "turbulent" particle field (the same offline fallback
    `openptv2.benchmarking.jhtdb_client` uses when the JHTDB DNS service is
    unreachable) rather than a live JHTDB stream, since JHTDB requires a
    registered auth token this dashboard doesn't have.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return F, device, np, torch


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(device, mo):
    mo.md(f"**Compute device:** `{device}`" + (" (cloud GPU attached)" if device.type == "cuda" else " (CPU -- attach a GPU on molab for full throughput)"))
    return


@app.cell
def _(mo):
    n_particles_slider = mo.ui.slider(4, 200, value=30, step=1, label="particles")
    n_frames_slider = mo.ui.slider(10, 60, value=20, step=1, label="frames")
    i_threshold_slider = mo.ui.slider(0.0, 0.9, value=0.3, step=0.01, label="I_threshold (Stage 1)")
    seed_slider = mo.ui.slider(0, 100, value=0, step=1, label="seed")
    mo.hstack([n_particles_slider, n_frames_slider, i_threshold_slider, seed_slider])
    return i_threshold_slider, n_frames_slider, n_particles_slider, seed_slider


@app.cell
def _(mo):
    mo.md("## Stage 1-5: differentiable core (mirrors `openptv2.differentiable`)")
    return


@app.cell
def _(F, torch):
    def soft_threshold(image, i_threshold, sharpness=15.0):
        """Stage 1: sigmoid gate replacing the discrete image > threshold mask."""
        gate = torch.sigmoid(sharpness * (image - i_threshold))
        return image * gate

    def gaussian_moment_fit(patches):
        """Stage 2: subpixel centroid via intensity-weighted image moments."""
        n, h, w = patches.shape
        ys = torch.arange(h, dtype=patches.dtype, device=patches.device)
        xs = torch.arange(w, dtype=patches.dtype, device=patches.device)
        mass = patches.sum(dim=(1, 2)).clamp_min(1e-8)
        cy = (patches.sum(dim=2) * ys).sum(dim=1) / mass
        cx = (patches.sum(dim=1) * xs).sum(dim=1) / mass
        return torch.stack([cx, cy], dim=-1)

    def render_gaussian_patch(peak_xy, patch_hw=9, sigma=1.2, dtype=torch.float64):
        """peak_xy: (..., 2) -> patch(es) (..., patch_hw, patch_hw), batched
        over any leading dims (e.g. one patch per particle) via broadcasting."""
        h = w = patch_hw
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=dtype), torch.arange(w, dtype=dtype), indexing="ij"
        )
        px = peak_xy[..., 0].unsqueeze(-1).unsqueeze(-1)
        py = peak_xy[..., 1].unsqueeze(-1).unsqueeze(-1)
        return torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma**2))

    def pinhole_project(point3d, origin, cc):
        cam = point3d - origin
        return torch.stack([cam[..., 0] / cam[..., 2] * cc, cam[..., 1] / cam[..., 2] * cc], dim=-1)

    def pinhole_ray_dir(img_xy, cc):
        x, y = img_xy[..., 0], img_xy[..., 1]
        return torch.stack([x / cc, y / cc, torch.ones_like(x)], dim=-1)

    def closest_point_between_rays(o1, d1, o2, d2, eps=1e-9):
        """Stage 3: differentiable epipolar intersection (midpoint of the
        shortest segment between two skew rays)."""
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
        return 0.5 * (p1 + p2)

    def savitzky_golay_kernels(window=5, poly_order=2, dt=1.0):
        import numpy as np

        half = window // 2
        t = np.arange(-half, half + 1, dtype=np.float64)
        A = np.vander(t, poly_order + 1, increasing=True)
        pinv = np.linalg.pinv(A)
        return (
            torch.tensor(pinv[0], dtype=torch.float64),
            torch.tensor(pinv[1] / dt, dtype=torch.float64),
            torch.tensor(2.0 * pinv[2] / dt**2, dtype=torch.float64),
        )

    def differentiable_savitzky_golay(positions, window=5, poly_order=2, dt=1.0):
        """Stage 5: smooth position + derive velocity/acceleration via a
        differentiable Savitzky-Golay filter (conv1d, fixed least-squares kernel)."""
        pos_k, vel_k, acc_k = savitzky_golay_kernels(window, poly_order, dt)
        pos_k = pos_k.to(positions.dtype).flip(0).view(1, 1, -1)
        vel_k = vel_k.to(positions.dtype).flip(0).view(1, 1, -1)
        acc_k = acc_k.to(positions.dtype).flip(0).view(1, 1, -1)
        n, t_len, d = positions.shape
        x = positions.permute(0, 2, 1).reshape(n * d, 1, t_len)
        vel = F.conv1d(x, vel_k).reshape(n, d, -1).permute(0, 2, 1)
        acc = F.conv1d(x, acc_k).reshape(n, d, -1).permute(0, 2, 1)
        return {"velocity": vel, "acceleration": acc}

    def kurtosis(accel, eps=1e-12):
        """Lagrangian physics loss (Stage 5 -> whitepaper's K_a)."""
        a = accel.reshape(-1)
        m2 = (a**2).mean()
        m4 = (a**4).mean()
        return m4 / (m2**2).clamp_min(eps)

    def velocity_power_spectrum(velocity):
        v = velocity - velocity.mean(dim=-1, keepdim=True)
        spec = torch.fft.rfft(v, dim=-1)
        return (spec.abs() ** 2) / v.shape[-1]

    return (
        closest_point_between_rays,
        differentiable_savitzky_golay,
        gaussian_moment_fit,
        kurtosis,
        pinhole_project,
        pinhole_ray_dir,
        render_gaussian_patch,
        soft_threshold,
        velocity_power_spectrum,
    )


@app.cell
def _(mo):
    mo.md("## Synthetic ground truth (offline HIT stand-in) + end-to-end pipeline")
    return


@app.cell
def _(
    closest_point_between_rays,
    differentiable_savitzky_golay,
    gaussian_moment_fit,
    i_threshold_slider,
    kurtosis,
    n_frames_slider,
    n_particles_slider,
    pinhole_project,
    pinhole_ray_dir,
    render_gaussian_patch,
    seed_slider,
    soft_threshold,
    torch,
    velocity_power_spectrum,
):
    def synthetic_hit_trajectories(n_particles, n_frames, seed):
        """Ornstein-Uhlenbeck 'turbulent' velocity walk -- offline stand-in
        for JHTDB Lagrangian trajectories (same construction as
        openptv2.benchmarking.jhtdb_client.synthetic_hit_trajectories)."""
        g = torch.Generator().manual_seed(seed)
        pos = torch.zeros(n_particles, 3, dtype=torch.float64)
        pos[:, 2] = 200.0  # in front of the cameras
        vel = torch.zeros(n_particles, 3, dtype=torch.float64)
        traj = torch.empty(n_particles, n_frames, 3, dtype=torch.float64)
        for f in range(n_frames):
            vel = 0.9 * vel + torch.randn(n_particles, 3, generator=g, dtype=torch.float64) * 0.3
            pos = pos + vel
            traj[:, f, :] = pos
        return traj

    def run_pipeline(i_threshold, n_particles, n_frames, seed, cc=100.0, patch_hw=9):
        dtype = torch.float64
        o1 = torch.tensor([-50.0, 0.0, -500.0], dtype=dtype)
        o2 = torch.tensor([50.0, 0.0, -500.0], dtype=dtype)
        true_pts = synthetic_hit_trajectories(n_particles, n_frames, seed)  # (N, T, 3)

        recon = torch.empty(n_particles, n_frames, 3, dtype=dtype)
        for f in range(n_frames):
            pt = true_pts[:, f, :]  # (N, 3)
            img1 = pinhole_project(pt, o1, cc)
            img2 = pinhole_project(pt, o2, cc)
            frac1 = img1 - img1.round()
            frac2 = img2 - img2.round()
            patch1 = render_gaussian_patch(frac1 + patch_hw // 2, patch_hw, dtype=dtype)
            patch2 = render_gaussian_patch(frac2 + patch_hw // 2, patch_hw, dtype=dtype)
            patch1 = soft_threshold(patch1, i_threshold, sharpness=15.0)
            patch2 = soft_threshold(patch2, i_threshold, sharpness=15.0)
            c1 = gaussian_moment_fit(patch1)
            c2 = gaussian_moment_fit(patch2)
            est_img1 = (img1.round() - patch_hw // 2) + c1
            est_img2 = (img2.round() - patch_hw // 2) + c2
            d1 = pinhole_ray_dir(est_img1, cc)
            d2 = pinhole_ray_dir(est_img2, cc)
            recon[:, f, :] = closest_point_between_rays(o1, d1, o2, d2)

        out = differentiable_savitzky_golay(recon, window=5, poly_order=2, dt=1.0)
        ka = kurtosis(out["acceleration"])
        psd = velocity_power_spectrum(out["velocity"]).mean(dim=0)
        return recon, out, ka, psd

    i_thresh = torch.tensor(i_threshold_slider.value, dtype=torch.float64, requires_grad=True)
    trajectory, sg_out, ka_loss, psd = run_pipeline(
        i_thresh, n_particles_slider.value, n_frames_slider.value, seed_slider.value
    )
    ka_loss.backward()
    grad_i_threshold = float(i_thresh.grad)
    return grad_i_threshold, ka_loss, psd, sg_out, trajectory


@app.cell
def _(grad_i_threshold, ka_loss, mo):
    mo.md(
        f"""
    ## Live physics: gradient flow from Stage 1 to Stage 5

    $K_a$ (acceleration kurtosis) = **{ka_loss.item():.4f}**
    &nbsp;&nbsp;|&nbsp;&nbsp;
    $\\partial K_a / \\partial I_{{\\text{{threshold}}}}$ = **{grad_i_threshold:.4f}**

    Drag the `I_threshold` slider above -- both numbers update reactively,
    demonstrating that a Stage-1 micro-parameter measurably moves the Stage-5
    Lagrangian physics loss (the whitepaper's central claim).
    """
    )
    return


@app.cell
def _(mo):
    mo.md("## Acceleration PDF and Lagrangian velocity power spectrum")
    return


@app.cell
def _(plt, sg_out):
    accel_flat = sg_out["acceleration"].detach().numpy().reshape(-1)

    fig_pdf, ax_pdf = plt.subplots(figsize=(6, 4))
    ax_pdf.hist(accel_flat, bins=40, density=True, color="steelblue", alpha=0.8)
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlabel("acceleration a")
    ax_pdf.set_ylabel("PDF (log scale)")
    ax_pdf.set_title("Acceleration PDF (heavy tails -> high $K_a$)")
    fig_pdf.tight_layout()
    fig_pdf
    return


@app.cell
def _(np, plt, psd):
    psd_np = psd.detach().numpy()
    omega = np.arange(1, len(psd_np) + 1)

    fig_psd, ax_psd = plt.subplots(figsize=(6, 4))
    ax_psd.loglog(omega, psd_np, color="darkorange")
    ax_psd.set_xlabel("frequency index")
    ax_psd.set_ylabel(r"$E_L(\omega)$")
    ax_psd.set_title("Lagrangian velocity power spectrum")
    fig_psd.tight_layout()
    fig_psd
    return


@app.cell
def _(mo):
    mo.md("## Interactive 3D trajectories")
    return


@app.cell
def _(mo, trajectory):
    from wigglystuff import ThreeWidget

    _traj = trajectory.detach().numpy()
    _n_particles = _traj.shape[0]
    _palette = [
        "#ef4444", "#f97316", "#eab308", "#22c55e",
        "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899",
    ]
    trajectory_points = [
        {
            "x": float(_traj[p, f, 0]),
            "y": float(_traj[p, f, 1]),
            "z": float(_traj[p, f, 2]),
            "color": _palette[p % len(_palette)],
            "size": 0.15,
        }
        for p in range(_n_particles)
        for f in range(_traj.shape[1])
    ]
    three_widget = mo.ui.anywidget(
        ThreeWidget(
            data=trajectory_points,
            width=800,
            height=500,
            show_grid=True,
            show_axes=True,
            axis_labels=["x", "y", "z"],
            auto_rotate=True,
        )
    )
    three_widget
    return


if __name__ == "__main__":
    app.run()
