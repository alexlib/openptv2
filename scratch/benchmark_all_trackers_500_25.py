import os, time, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import OpenPTV2 Tracker Plugins
from openptv2.plugins.proptv import ProPTVConfig
from openptv2.plugins.proptv_tracking import ProPTVTracker
from openptv2.plugins.quality_3d_tracking import Quality3DTracker
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.plugins.fast_3d_smooth_tracking import Fast3DSmoothTracker

# Data paths
data_dir = r"C:\Users\alex\Github\proPTV\data\500_25"
origin_files = [f"{data_dir}/origin/origin_{str(t).zfill(5)}.txt" for t in range(5)]
origin_hdf5 = f"{data_dir}/origin/tracks_origin.hdf5"

print("=============================================================")
print("  OPENPTV2 MULTI-TRACKER BENCHMARK (500_25 DATASET)")
print("=============================================================\n")

print("1. Loading 3D Particle Coordinates (500 particles x 5 frames)...")
frame_particles = []
for t, filepath in enumerate(origin_files):
    if not os.path.exists(filepath):
        print(f"Error: Missing dataset file {filepath}")
        exit(1)
    pts = np.loadtxt(filepath)[:, 1:4] # X, Y, Z
    frame_particles.append(pts)
    print(f"  Frame {t}: {len(pts)} 3D particles")

# Load Ground Truth
gt_tracks = []
if os.path.exists(origin_hdf5):
    with h5py.File(origin_hdf5, 'r') as f:
        for k in f.keys():
            gt_tracks.append(f[k][:])

gt_p0 = np.array([tr[0, 1:4] for tr in gt_tracks]) if gt_tracks else frame_particles[0]
total_gt = len(gt_p0)

# Define Trackers
trackers_dict = {
    "proPTV (Predictive GMM)": ProPTVTracker(ProPTVConfig(t_init=3, maxvel=0.015, angle=60.0, dt=1.0)),
    "3D Kalman-Hungarian": Quality3DTracker(v_max=0.015, a_max=0.010, dt=1.0),
    "MyPTV 3D Kinematic": MyPTV3DTracker(v_max=0.015, a_max=0.010, dt=1.0),
    "Fast 3D Smooth (SG)": Fast3DSmoothTracker(v_max=0.015, dacc=0.010, dt=1.0)
}

results = []

print("\n2. Executing Multi-Tracker Benchmark...")
for name, tracker_obj in trackers_dict.items():
    t0 = time.perf_counter()
    tracks = tracker_obj.track_frames(frame_particles)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Calculate metrics
    rec_count = len(tracks)
    rec_p0 = np.array([tr["pos"][0] for tr in tracks]) if rec_count > 0 else np.empty((0, 3))

    matched = 0
    errors = []
    if len(rec_p0) > 0 and len(gt_p0) > 0:
        for p in rec_p0:
            dists = np.linalg.norm(gt_p0 - p, axis=1)
            min_d = np.min(dists)
            errors.append(min_d)
            if min_d < 0.02:
                matched += 1

    pmp = (matched / total_gt) * 100.0 if total_gt > 0 else 0.0
    mean_err = np.mean(errors) if errors else 0.0

    res_entry = {
        "name": name,
        "count": rec_count,
        "matched": matched,
        "pmp": pmp,
        "mean_err": mean_err,
        "time_ms": elapsed_ms,
        "tracks": tracks
    }
    results.append(res_entry)

    print(f"  • {name:<26}: {rec_count} tracks | PMP: {pmp:.2f}% | Mean Error: {mean_err:.6f} | Time: {elapsed_ms:.2f} ms")

print("\n=============================================================")
print(f"{'Tracker':<28} | {'Tracks':<8} | {'PMP %':<8} | {'Mean 3D Error':<12} | {'Time (ms)':<10}")
print("-" * 75)
for r in results:
    print(f"{r['name']:<28} | {r['count']:<8} | {r['pmp']:<8.2f} | {r['mean_err']:<12.6f} | {r['time_ms']:<10.2f}")
print("=============================================================\n")

# Plot Comparative Bar Chart & 3D Tracks
fig = plt.figure(figsize=(18, 7))

# Subplot 1: PMP % & Time comparison
ax1 = fig.add_subplot(1, 2, 1)
names = [r["name"] for r in results]
pmps = [r["pmp"] for r in results]
times = [r["time_ms"] for r in results]

x = np.arange(len(names))
width = 0.35

rects1 = ax1.bar(x - width/2, pmps, width, label='Percentage Matched Particles (PMP %)', color='seagreen')
ax1_twin = ax1.twinx()
rects2 = ax1_twin.bar(x + width/2, times, width, label='Runtime (ms)', color='mediumpurple', alpha=0.8)

ax1.set_ylabel('PMP (%)', color='seagreen', fontweight='bold')
ax1_twin.set_ylabel('Execution Time (ms)', color='mediumpurple', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha='right', fontsize=9, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.set_title("OpenPTV2 Trackers Accuracy & Speed Comparison (500_25 Dataset)", fontsize=11, fontweight='bold')

# Subplot 2: 3D Trajectory Overlay for top trackers
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
colors = ['red', 'dodgerblue', 'darkorange', 'green']

# Plot GT in gray background
for tr in gt_tracks[:100]:
    ax2.plot(tr[:, 1], tr[:, 2], tr[:, 3], color='lightgray', alpha=0.4, linewidth=0.8)

for idx, r in enumerate(results):
    for tr in r["tracks"][:40]:
        pos = np.asarray(tr["pos"])
        ax2.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[idx % len(colors)], alpha=0.6, linewidth=1.0)
    # Dummy plot for legend
    ax2.plot([], [], [], color=colors[idx % len(colors)], label=r["name"])

ax2.set_title("3D Trajectories Overlay (Sample 40 tracks per tracker)", fontsize=11, fontweight='bold')
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
output_img = r"C:\Users\alex\projects\openptv2\scratch\all_trackers_benchmark_500_25.png"
plt.savefig(output_img, dpi=200)
print(f"Benchmark comparative figure saved to: {output_img}")
