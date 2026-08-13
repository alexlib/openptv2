import os, glob, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openptv2.plugins.proptv import ProPTVConfig
from openptv2.plugins.proptv_tracking import ProPTVTracker

# Source data directory from proPTV dataset
data_dir = r"C:\Users\alex\Github\proPTV\data\500_25"
origin_files = [f"{data_dir}/origin/origin_{str(t).zfill(5)}.txt" for t in range(5)]
origin_hdf5 = f"{data_dir}/origin/tracks_origin.hdf5"

print("1. Loading 3D particle clouds per frame from proPTV origin dataset...")
frame_particles = []
for t, filepath in enumerate(origin_files):
    if not os.path.exists(filepath):
        print(f"Error: missing {filepath}")
        exit(1)
    # File columns: pnr, x, y, z, u, v, w, T, p
    data = np.loadtxt(filepath)
    pts_3d = data[:, 1:4] # extract X, Y, Z
    frame_particles.append(pts_3d)
    print(f"  Frame t={t}: {len(pts_3d)} 3D particles loaded.")

# 2. Configure openptv2 ProPTVTracker
print("\n2. Initializing openptv2 ProPTVTracker plugin...")
config = ProPTVConfig(
    t_init=3,
    maxvel=0.015,
    angle=60.0,
    dt=1.0,
    Vmin=[0.0, 0.0, 0.0],
    Vmax=[1.0, 1.0, 1.0],
    NN=[3, 3, 3]
)
tracker = ProPTVTracker(config)

# 3. Execute tracking
print("\n3. Running openptv2 ProPTVTracker on 500_25 dataset...")
tracks = tracker.track_frames(frame_particles)
print(f"  Finished! Total openptv2 reconstructed tracks: {len(tracks)}")

# 4. Load Ground Truth Tracks for comparison
gt_tracks = []
if os.path.exists(origin_hdf5):
    with h5py.File(origin_hdf5, 'r') as f:
        for k in f.keys():
            gt_tracks.append(f[k][:])

print(f"  Ground truth tracks count: {len(gt_tracks)}")

# Extract initial 3D positions at t=0
rec_p0 = np.array([tr["pos"][0] for tr in tracks]) if tracks else np.empty((0, 3))
gt_p0 = np.array([tr[0, 1:4] for tr in gt_tracks]) if gt_tracks else np.empty((0, 3))

# Calculate matching metrics
matched_count = 0
min_errors = []
if len(gt_p0) > 0 and len(rec_p0) > 0:
    for p in rec_p0:
        dists = np.linalg.norm(gt_p0 - p, axis=1)
        min_dist = np.min(dists)
        min_errors.append(min_dist)
        if min_dist < 0.02:
            matched_count += 1

min_errors = np.array(min_errors)
pmp = (matched_count / len(gt_p0)) * 100 if len(gt_p0) > 0 else 0.0
mean_error = np.mean(min_errors) if len(min_errors) > 0 else 0.0

print("\n================ OPENPTV2 PERFORMANCE METRICS ================")
print(f"Total Ground Truth Particles : {len(gt_p0)}")
print(f"Total Reconstructed Particles: {len(rec_p0)}")
print(f"Percentage Matched Particles : {pmp:.2f}%")
print(f"Mean 3D Error (Euclidean)    : {mean_error:.6f} [domain 0..1]")
print("=============================================================\n")

# Plot comparison
fig = plt.figure(figsize=(16, 6))

# Subplot 1: 3D Trajectories
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
for tr in gt_tracks[:150]:
    ax1.plot(tr[:, 1], tr[:, 2], tr[:, 3], color='blue', alpha=0.3, linewidth=0.8)
ax1.scatter(gt_p0[:150, 0], gt_p0[:150, 1], gt_p0[:150, 2], color='navy', s=3, label='Ground Truth')

for tr in tracks[:150]:
    pos = tr["pos"]
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='green', alpha=0.7, linewidth=1.2)
if len(rec_p0) > 0:
    ax1.scatter(rec_p0[:150, 0], rec_p0[:150, 1], rec_p0[:150, 2], color='darkgreen', s=6, label='openptv2 (proPTV plugin)')

ax1.set_title("3D Particle Tracks Comparison (openptv2 vs Ground Truth)", fontsize=11, fontweight='bold')
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend(loc='upper right')

# Subplot 2: Histogram & Metrics
ax2 = fig.add_subplot(1, 2, 2)
if len(min_errors) > 0:
    ax2.hist(min_errors, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean Error = {mean_error:.5f}')
    ax2.set_title("openptv2 Reconstruction Error Distribution", fontsize=11, fontweight='bold')
    ax2.set_xlabel("3D Euclidean Error")
    ax2.set_ylabel("Particle Count")
    ax2.legend()

    summary_text = f"openptv2 Performance:\n• GT Particles: {len(gt_p0)}\n• Tracked: {len(rec_p0)}\n• PMP: {pmp:.2f}%\n• Mean Error: {mean_error:.5f}"
    ax2.text(0.60, 0.65, summary_text, transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="honeydew", alpha=0.8))

plt.tight_layout()
output_plot = r"C:\Users\alex\projects\openptv2\scratch\openptv2_tracking_performance.png"
plt.savefig(output_plot, dpi=200)
print(f"Plot saved to: {output_plot}")
