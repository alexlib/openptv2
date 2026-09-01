"""Generate smooth Burgers-style Lagrangian GIF — particles move, trails smooth."""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from pathlib import Path
from scipy.signal import savgol_filter

rng = np.random.default_rng(42)
n_traj = 26
T = 80
trajectories_raw = []
for i in range(n_traj):
    # Burgers-like: core radius ~8mm, outer ~16mm, swirl + axial jet
    # inner core: solid rotation, outer: free vortex decay
    r_core = 8.0
    r = rng.uniform(4, 18)
    # swirl strength ~ Gamma/(2πr) * (1-exp(-r²/r_core²)) — simplified
    gamma = rng.uniform(22, 38)
    vt_factor = (1 - np.exp(-(r**2)/(r_core**2 + 1e-6))) / max(r, 2)
    omega = 0.18 + 0.35 * vt_factor  # swirl
    v_z = rng.uniform(1.0, 2.4) * (1 if i % 2 == 0 else -1) * (0.6 + 0.5*np.exp(-r/12))
    z0 = rng.uniform(-18, 18)
    phase = rng.uniform(0, 2*np.pi)
    t = np.linspace(0, 10, T)
    # helical + slow radial breathing (exotic)
    x = r * np.cos(omega * t * 5 + phase) + 0.9*np.sin(t*0.7 + i*0.5)
    y = r * np.sin(omega * t * 5 + phase) + 0.9*np.cos(t*0.9 + i*0.3)
    z = z0 + v_z * t * 3.0 + 0.5*np.sin(t*1.4 + i*0.8)
    pts = np.stack([x, y, z], axis=1)
    # add tiny measurement noise then smooth
    pts_noisy = pts + rng.normal(0, 0.18, pts.shape)
    trajectories_raw.append(pts_noisy)

# smooth each trajectory with Savitzky-Golay (window 9, poly 3) for nice Lagrangian curves
trajectories = []
for pts in trajectories_raw:
    # savgol needs window odd and < T
    win = 9 if T >= 9 else (T//2*2+1)
    try:
        smooth = np.stack([savgol_filter(pts[:, k], window_length=win, polyorder=3, mode="interp") for k in range(3)], axis=1)
    except Exception:
        smooth = pts
    trajectories.append(smooth)

vels = np.array([np.linalg.norm(t[-1]-t[0])/T for t in trajectories])
norm = plt.Normalize(vmin=np.percentile(vels, 10), vmax=np.percentile(vels, 90))
cmap = plt.cm.turbo

out_dir = Path("docs/images")
out_dir.mkdir(parents=True, exist_ok=True)
gif_path = out_dir / "burgers_smooth_moving.gif"
png_path = out_dir / "burgers_smooth_moving.png"

elev, azim = 26, 42
trail = 20

fig = plt.figure(figsize=(8.4, 6.4), dpi=130)
ax = fig.add_subplot(111, projection="3d")
all_pts = np.concatenate(trajectories, axis=0)
xmin, xmax = all_pts[:,0].min(), all_pts[:,0].max()
ymin, ymax = all_pts[:,1].min(), all_pts[:,1].max()
zmin, zmax = all_pts[:,2].min(), all_pts[:,2].max()
pad = 3.0
ax.set_xlim(xmin-pad, xmax+pad)
ax.set_ylim(ymin-pad, ymax+pad)
ax.set_zlim(zmin-pad, zmax+pad)
ax.set_xlabel("X [mm]", fontsize=9, fontweight="bold")
ax.set_ylabel("Y [mm]", fontsize=9, fontweight="bold")
ax.set_zlabel("Z [mm]", fontsize=9, fontweight="bold")
ax.set_title("Burgers vortex — smooth Lagrangian spirals", fontsize=11, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.22)
ax.view_init(elev=elev, azim=azim)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.55, aspect=18, pad=0.07)
cbar.set_label("Mean speed [mm/frame]", fontsize=9)
fig.tight_layout()

frames = []
for f in range(T):
    ax.clear()
    ax.set_xlim(xmin-pad, xmax+pad)
    ax.set_ylim(ymin-pad, ymax+pad)
    ax.set_zlim(zmin-pad, zmax+pad)
    ax.set_xlabel("X [mm]", fontsize=9, fontweight="bold")
    ax.set_ylabel("Y [mm]", fontsize=9, fontweight="bold")
    ax.set_zlabel("Z [mm]", fontsize=9, fontweight="bold")
    ax.set_title("Burgers vortex — smooth Lagrangian spirals", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.view_init(elev=elev, azim=azim)
    for traj, v in zip(trajectories, vels):
        c = cmap(norm(v))
        t0 = max(0, f - trail)
        seg = traj[t0:f+1]
        if len(seg) > 1:
            for j in range(len(seg)-1):
                alpha = 0.20 + 0.75 * (j+1)/len(seg)
                lw = 0.8 + 1.1 * (j+1)/len(seg)
                ax.plot(seg[j:j+2,0], seg[j:j+2,1], seg[j:j+2,2], color=c, alpha=alpha, lw=lw)
        ax.scatter(traj[f,0], traj[f,1], traj[f,2], color=c, s=28, edgecolors="white", linewidths=0.5, alpha=0.98, zorder=5)
    ax.text2D(0.02, 0.96, f"frame {f+1:02d}/{T}", transform=ax.transAxes, fontsize=9, color="#9aa3b2",
              bbox=dict(boxstyle="round,pad=0.25", fc="#0f1115", ec="#222631", alpha=0.85))
    fig.canvas.draw()
    rgba = np.array(fig.canvas.buffer_rgba(), copy=True)
    frames.append(rgba[...,:3])

plt.close()
# use imageio.v3
iio.imwrite(png_path, frames[-1])
print(f"Saved PNG {png_path} {png_path.stat().st_size/1024:.0f} KB")
iio.imwrite(gif_path, np.stack(frames), duration=55, loop=0)
print(f"Saved GIF {gif_path} {gif_path.stat().st_size/1024:.0f} KB frames={len(frames)} shape={frames[0].shape}")
im = iio.imread(gif_path, index=None)
print(f"Verified {im.shape[0]} frames shape {im.shape}")
