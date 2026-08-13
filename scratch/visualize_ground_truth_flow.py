import os, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

origin_hdf5 = r"C:\Users\alex\Github\proPTV\data\500_25\origin\tracks_origin.hdf5"

if not os.path.exists(origin_hdf5):
    print(f"Error: ground truth file {origin_hdf5} not found.")
    exit(1)

print("Loading ground truth trajectories from HDF5...")
gt_tracks = []
with h5py.File(origin_hdf5, 'r') as f:
    for k in f.keys():
        gt_tracks.append(f[k][:])

print(f"Loaded {len(gt_tracks)} ground truth tracks.")

# Calculate velocity magnitudes for color coding
all_velocities = []
all_positions = []
for tr in gt_tracks:
    # tr shape: (N_steps, cols) -> col 0=t, 1..3=X,Y,Z, 4..6=U,V,W
    pos = tr[:, 1:4]
    vel = tr[:, 4:7] if tr.shape[1] >= 7 else np.diff(pos, axis=0, prepend=pos[:1])
    speed = np.linalg.norm(vel, axis=1)
    all_velocities.append(speed)
    all_positions.append(pos)

min_speed = min(np.min(s) for s in all_velocities)
max_speed = max(np.max(s) for s in all_velocities)
mean_speed = np.mean([np.mean(s) for s in all_velocities])

print(f"Flow Speed Stats: Min={min_speed:.6f}, Mean={mean_speed:.6f}, Max={max_speed:.6f}")

# Create multi-panel figure
fig = plt.figure(figsize=(18, 8))

# Panel 1: 3D Trajectories colored by velocity magnitude
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
cmap = plt.cm.plasma

norm = matplotlib.colors.Normalize(vmin=min_speed, vmax=max_speed)

for pos, speed in zip(all_positions, all_velocities):
    for i in range(len(pos) - 1):
        c = cmap(norm(speed[i]))
        ax1.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2], color=c, linewidth=1.5, alpha=0.8)
    # Start point dot
    ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color='black', s=2, alpha=0.5)

ax1.set_title("3D Particle Flow Trajectories (Rayleigh-Bénard Convection)", fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel("X (Normalized)", labelpad=8)
ax1.set_ylabel("Y (Normalized)", labelpad=8)
ax1.set_zlabel("Z (Normalized)", labelpad=8)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)

# Colorbar for 3D plot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label("Speed |U| (mm/frame or normalized units)", fontsize=10)

# Panel 2: 2D Projection (XY, XZ, YZ Plane Projections)
ax2 = fig.add_subplot(1, 2, 2)

# Subplot inside panel 2: XZ Plane (Side view showing thermal plumes)
for pos, speed in zip(all_positions[:250], all_velocities[:250]):
    ax2.plot(pos[:, 0], pos[:, 2], color='teal', alpha=0.4, linewidth=1.0)
    # Direction arrow
    if len(pos) >= 2:
        ax2.annotate('', xy=(pos[-1, 0], pos[-1, 2]), xytext=(pos[0, 0], pos[0, 2]),
                     arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8, alpha=0.6))

ax2.set_title("2D Projection (XZ Plane - Thermal Plumes Circulation)", fontsize=12, fontweight='bold')
ax2.set_xlabel("X Position")
ax2.set_ylabel("Z Position (Height)")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True, linestyle="--", alpha=0.5)

# Text Box with Flow Summary
flow_summary = (
    f"Dataset Flow Characterization:\n"
    f"• Total Particle Count: 500\n"
    f"• Time Steps: 5 (t = 0..4)\n"
    f"• Min Speed: {min_speed:.5f}\n"
    f"• Mean Speed: {mean_speed:.5f}\n"
    f"• Max Speed: {max_speed:.5f}\n"
    f"• Spatial Domain: [0, 1]³\n"
    f"• Flow Type: Thermal Convection Roll"
)
ax2.text(0.03, 0.72, flow_summary, transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.9))

plt.tight_layout()
output_img = r"C:\Users\alex\projects\openptv2\scratch\flow_trajectories_500_25.png"
plt.savefig(output_img, dpi=200)
print(f"Flow trajectory visualization saved to: {output_img}")
