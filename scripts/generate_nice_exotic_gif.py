"""Generate a nice exotic Lagrangian GIF — helical Burgers-style spirals."""

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

# --- synthetic exotic trajectories: helical vortex + turbulent wiggle ---
rng = np.random.default_rng(7)
n_traj = 36
trajectories = []
vels = []
for i in range(n_traj):
    # each trajectory: helix with different radius/pitch + small turbulent noise
    z0 = rng.uniform(-20, 20)
    r = rng.uniform(6, 18)
    pitch = rng.uniform(0.8, 2.2)
    omega = rng.uniform(0.25, 0.45)  # angular speed
    v_z = rng.uniform(1.2, 2.5) * (1 if rng.random() > 0.5 else -1)
    T = 60
    t = np.linspace(0, 8, T)
    # base helix
    x = r * np.cos(omega * t * 6 + rng.uniform(0, 2 * np.pi))
    y = r * np.sin(omega * t * 6 + rng.uniform(0, 2 * np.pi))
    z = z0 + v_z * t * 4 + 0.9 * np.sin(t * 1.7)
    # add center drift + turbulent wiggle
    x += 2 * np.sin(t * 0.9 + i)
    y += 2 * np.cos(t * 1.1 + i * 0.7)
    # small 3D noise
    pts = np.stack([x, y, z], axis=1) + rng.normal(0, 0.35, (T, 3))
    # keep only segment that stays in volume, add slight curvature
    trajectories.append(pts)
    vels.append(float(np.linalg.norm(pts[-1] - pts[0]) / T))

trajectories = [t for t in trajectories if len(t) > 10]
vels = np.array(vels[: len(trajectories)])
norm = plt.Normalize(vmin=np.percentile(vels, 5), vmax=np.percentile(vels, 95))
cmap = plt.cm.turbo  # more exotic than viridis

# --- render rotating GIF ---
fig = plt.figure(figsize=(8, 6), dpi=120)
ax = fig.add_subplot(111, projection="3d")

for traj, v in zip(trajectories, vels):
    c = cmap(norm(v))
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, lw=1.4, alpha=0.85)
    # head marker
    ax.scatter(
        traj[-1, 0],
        traj[-1, 1],
        traj[-1, 2],
        color=c,
        s=18,
        edgecolors="white",
        linewidths=0.4,
        alpha=0.95,
    )
    # faint trail tail
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=c, s=6, alpha=0.25)

ax.set_xlabel("X [mm]", fontsize=9, fontweight="bold")
ax.set_ylabel("Y [mm]", fontsize=9, fontweight="bold")
ax.set_zlabel("Z [mm]", fontsize=9, fontweight="bold")
ax.set_title(
    "Exotic Lagrangian helices — Burgers vortex style",
    fontsize=11,
    fontweight="bold",
    pad=10,
)
ax.grid(True, linestyle="--", alpha=0.25)
# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.55, aspect=18, pad=0.08)
cbar.set_label("Mean speed [mm/frame]", fontsize=9)
# equal-ish view
ax.set_box_aspect((1, 1, 1))

out_dir = Path("docs/images")
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "exotic_helical_trajectories.gif"
png_path = out_dir / "exotic_helical_trajectories.png"

# save static PNG from first angle
ax.view_init(elev=22, azim=38)
fig.tight_layout()
plt.savefig(png_path, dpi=140)
print(f"Saved PNG {png_path} {png_path.stat().st_size / 1024:.1f} KB")

# animate: 36 frames, 12 deg steps, copy buffer!
frames = []
for azim in range(0, 360, 12):
    ax.view_init(elev=22, azim=azim)
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames.append(rgba[..., :3])  # drop alpha for GIF

plt.close()
imageio.mimsave(gif_path, frames, duration=0.07, loop=0, palettesize=128)
print(
    f"Saved GIF {gif_path} {gif_path.stat().st_size / 1024:.0f} KB frames={len(frames)} shape={frames[0].shape}"
)
# verify
import imageio.v2 as imageio2

im = imageio2.mimread(str(gif_path))
print(f"Verified frames read back: {len(im)}")
