import os, time, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openptv2.plugins.proptv import ProPTVConfig
from openptv2.plugins.proptv_tracking import ProPTVTracker
from openptv2.plugins.quality_3d_tracking import Quality3DTracker
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.plugins.fast_3d_smooth_tracking import Fast3DSmoothTracker

data_dir = r"C:\Users\alex\Github\proPTV\data\500_30"
origin_files = [f"{data_dir}/origin/origin_{str(t).zfill(5)}.txt" for t in range(30)]
origin_hdf5 = f"{data_dir}/origin/tracks_origin.hdf5"

print("=============================================================")
print("  OPENPTV2 MULTI-TRACKER BENCHMARK (500_30 DATASET - 30 FRAMES)")
print("=============================================================")

# 1. Load particle coordinates for 30 frames
frame_particles = []
for t, filepath in enumerate(origin_files):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing {filepath}")
    data = np.loadtxt(filepath, skiprows=1 if t == 0 else 1)
    # File columns: ID, X, Y, Z, U, V, W, T, P...
    pts_3d = data[:, 1:4] # extract X, Y, Z
    frame_particles.append(pts_3d)

print(f"Loaded {len(frame_particles)} frames of 3D particles ({len(frame_particles[0])} particles/frame).")

# 2. Load Ground Truth Tracks (30 frames long)
gt_tracks = []
if os.path.exists(origin_hdf5):
    with h5py.File(origin_hdf5, 'r') as f:
        for k in f.keys():
            gt_tracks.append(f[k][:])

gt_p0 = np.array([tr[0, 1:4] for tr in gt_tracks]) if gt_tracks else np.empty((0, 3))

# Benchmark configuration dictionary
trackers = {
    "proPTV (Predictive GMM)": ProPTVTracker(ProPTVConfig(
        t_init=3, maxvel=0.015, angle=60.0, dt=1.0, Vmin=[0,0,0], Vmax=[1,1,1]
    )),
    "3D Kalman-Hungarian": Quality3DTracker(
        v_max=0.015, a_max=0.010, dt=1.0
    ),
    "MyPTV 3D Kinematic": MyPTV3DTracker(
        v_max=0.015, a_max=0.010, max_gap=2, dt=1.0
    ),
    "Fast 3D Smooth (SG)": Fast3DSmoothTracker(
        v_max=0.015, dacc=0.010, smooth_window=5, dt=1.0
    ),
}

results = {}

for name, tracker_inst in trackers.items():
    t_start = time.perf_counter()
    reconstructed_tracks = tracker_inst.track_frames(frame_particles)
    t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
    
    num_tracks = len(reconstructed_tracks)
    
    # Matching metrics against Ground Truth
    if num_tracks > 0 and len(gt_p0) > 0:
        rec_p0 = np.array([tr["pos"][0] for tr in reconstructed_tracks])
        # Find nearest ground truth particle
        dists = np.linalg.norm(rec_p0[:, None, :] - gt_p0[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        matched_mask = min_dists < 1e-3
        pmp = (np.sum(matched_mask) / len(gt_p0)) * 100.0
        mean_err = np.mean(min_dists[matched_mask]) if np.any(matched_mask) else np.nan
    else:
        pmp, mean_err = 0.0, np.nan
        
    results[name] = {
        "tracks": num_tracks,
        "pmp": pmp,
        "mean_err": mean_err,
        "time_ms": t_elapsed,
        "reconstructed_tracks": reconstructed_tracks,
    }
    
    print(f"  • {name:<26}: {num_tracks} tracks | PMP: {pmp:.2f}% | Mean Error: {mean_err:.6f} | Time: {t_elapsed:.2f} ms")

print("\n=============================================================")
print(f"{'Tracker':<28} | {'Tracks':<8} | {'PMP %':<8} | {'Mean 3D Error':<13} | {'Time (ms)':<9}")
print("-" * 75)
for name, res in results.items():
    print(f"{name:<28} | {res['tracks']:<8} | {res['pmp']:<8.2f} | {res['mean_err']:<13.6f} | {res['time_ms']:<9.2f}")
print("=============================================================\n")

# Save 30-frame comparative plot
fig = plt.figure(figsize=(16, 12))

# Subplot 1: 3D Trajectories for proPTV
ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title("proPTV (Predictive GMM) - 30 Frames", fontsize=12, fontweight='bold')
for tr in results["proPTV (Predictive GMM)"]["reconstructed_tracks"][:100]:
    pos = np.array(tr["pos"])
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.7, linewidth=0.8)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)

# Subplot 2: 3D Trajectories for Fast 3D Smooth
ax2 = fig.add_subplot(222, projection='3d')
ax2.set_title("Fast 3D Smooth (SG) - 30 Frames", fontsize=12, fontweight='bold')
for tr in results["Fast 3D Smooth (SG)"]["reconstructed_tracks"][:100]:
    pos = np.array(tr["pos"])
    ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.7, linewidth=0.8)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_zlim(0, 1)

# Subplot 3: 3D Trajectories for 3D Kalman-Hungarian
ax3 = fig.add_subplot(223, projection='3d')
ax3.set_title("3D Kalman-Hungarian - 30 Frames", fontsize=12, fontweight='bold')
for tr in results["3D Kalman-Hungarian"]["reconstructed_tracks"][:100]:
    pos = np.array(tr["pos"])
    ax3.plot(pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.7, linewidth=0.8)
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)

# Subplot 4: Performance Comparison Bar Chart
ax4 = fig.add_subplot(224)
names = [n.replace(" (Predictive GMM)", "").replace(" (SG)", "") for n in results.keys()]
times = [results[n]["time_ms"] for n in results.keys()]
bars = ax4.bar(names, times, color=['#2b5c8f', '#e06d53', '#2ca02c', '#9467bd'])
ax4.set_ylabel("Execution Time (ms)", fontsize=11)
ax4.set_title("Execution Time Comparison (30 Frames, 500 Particles)", fontsize=12, fontweight='bold')
ax4.set_yscale('log')
plt.xticks(rotation=15)

for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0, yval * 1.1, f"{yval:.1f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plot_path = r"C:\Users\alex\projects\openptv2\scratch\benchmark_30_frames.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"30-frame benchmark figure saved to: {plot_path}")
