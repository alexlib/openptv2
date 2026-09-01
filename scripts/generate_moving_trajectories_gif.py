"""Generate exotic Lagrangian GIF where particles MOVE along trajectories (trails grow)."""

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(11)
n_traj = 28
T = 72
trajectories = []
for i in range(n_traj):
    z0 = rng.uniform(-22, 22)
    r = rng.uniform(7, 16)
    pitch = rng.uniform(0.9, 2.0)
    omega = rng.uniform(0.28, 0.42)
    v_z = rng.uniform(1.4, 2.2) * (1 if rng.random() > 0.5 else -1)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, 9, T)
    # helical base + gentle swirl
    x = r * np.cos(omega * t * 5 + phase) + 1.8 * np.sin(t * 0.8 + i * 0.6)
    y = r * np.sin(omega * t * 5 + phase) + 1.8 * np.cos(t * 1.0 + i * 0.4)
    z = z0 + v_z * t * 3.2 + 0.7 * np.sin(t * 1.6 + i)
    # slight radial breathing for exotic feel
    r_mod = 1 + 0.12 * np.sin(t * 2.2 + i)
    x *= r_mod
    y *= r_mod
    pts = np.stack([x, y, z], axis=1) + rng.normal(0, 0.28, (T, 3))
    trajectories.append(pts)

trajectories = [t for t in trajectories if len(t) == T]
# color by mean speed in xy
vels = np.array([np.linalg.norm(t[-1] - t[0]) / T for t in trajectories])
norm = plt.Normalize(vmin=np.percentile(vels, 8), vmax=np.percentile(vels, 92))
cmap = plt.cm.turbo

out_dir = Path("docs/images")
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "exotic_helical_moving.gif"
png_path = out_dir / "exotic_helical_moving.png"

# fixed view - NO rotation
elev, azim = 24, 38
trail = 18  # frames of tail

fig = plt.figure(figsize=(8.2, 6.2), dpi=130)
ax = fig.add_subplot(111, projection="3d")
# set limits once from full data
all_pts = np.concatenate(trajectories, axis=0)
xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
zmin, zmax = all_pts[:, 2].min(), all_pts[:, 2].max()
pad = 2.5
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(ymin - pad, ymax + pad)
ax.set_zlim(zmin - pad, zmax + pad)
ax.set_xlabel("X [mm]", fontsize=9, fontweight="bold")
ax.set_ylabel("Y [mm]", fontsize=9, fontweight="bold")
ax.set_zlabel("Z [mm]", fontsize=9, fontweight="bold")
ax.set_title(
    "Lagrangian particles moving — exotic helices",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax.grid(True, linestyle="--", alpha=0.22)
ax.view_init(elev=elev, azim=azim)
# colorbar once
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.55, aspect=18, pad=0.07)
cbar.set_label("Mean speed [mm/frame]", fontsize=9)
fig.tight_layout()

frames = []
for f in range(T):
    ax.clear()
    # keep same limits/view each frame
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_zlim(zmin - pad, zmax + pad)
    ax.set_xlabel("X [mm]", fontsize=9, fontweight="bold")
    ax.set_ylabel("Y [mm]", fontsize=9, fontweight="bold")
    ax.set_zlabel("Z [mm]", fontsize=9, fontweight="bold")
    ax.set_title(
        "Lagrangian particles moving — exotic helices",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.view_init(elev=elev, azim=azim)
    # draw fading trails + moving head
    for traj, v in zip(trajectories, vels):
        c = cmap(norm(v))
        t0 = max(0, f - trail)
        seg = traj[t0 : f + 1]
        if len(seg) > 1:
            # fading tail: older segments more transparent
            for j in range(len(seg) - 1):
                alpha = 0.18 + 0.72 * (j + 1) / len(seg)
                lw = 0.7 + 1.0 * (j + 1) / len(seg)
                ax.plot(
                    seg[j : j + 2, 0],
                    seg[j : j + 2, 1],
                    seg[j : j + 2, 2],
                    color=c,
                    alpha=alpha,
                    lw=lw,
                )
        # head
        ax.scatter(
            traj[f, 0],
            traj[f, 1],
            traj[f, 2],
            color=c,
            s=26,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.98,
            zorder=5,
        )
    # small frame counter
    ax.text2D(
        0.02,
        0.96,
        f"frame {f + 1:02d}/{T}",
        transform=ax.transAxes,
        fontsize=9,
        color="#9aa3b2",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f1115", ec="#222631", alpha=0.85),
    )
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames.append(rgba[..., :3])

plt.close()

# save preview PNG = last frame (imageio.v3)
iio.imwrite(png_path, frames[-1])
print(f"Saved PNG {png_path} {png_path.stat().st_size / 1024:.0f} KB")
# stack frames for GIF (imageio.v3 expects (N,H,W,C) array)
iio.imwrite(gif_path, np.stack(frames), duration=60, loop=0)
print(
    f"Saved GIF {gif_path} {gif_path.stat().st_size / 1024:.0f} KB frames={len(frames)} shape={frames[0].shape}"
)
im = iio.imread(gif_path, index=None)
print(f"Verified {im.shape[0] if im.ndim == 4 else 1} frames (shape {im.shape})")
