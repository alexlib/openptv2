import os, time, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openptv2.plugins.cython_3d_tracking import Cython3DTracker
from openptv2.plugins.fast_3d_smooth_tracking import Fast3DSmoothTracker
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.plugins.quality_3d_tracking import Quality3DTracker
from openptv2.plugins.proptv_tracking import ProPTVTracker, ProPTVConfig

data_dir_30 = r"C:\Users\alex\Github\proPTV\data\500_30"
origin_files_30 = [f"{data_dir_30}/origin/origin_{str(t).zfill(5)}.txt" for t in range(30)]
origin_hdf5_30 = f"{data_dir_30}/origin/tracks_origin.hdf5"

print("==========================================================================")
print("  OPENPTV2 ALL MODULAR 3D TRACKING PLUGINS BENCHMARK (30 FRAMES - 500 PARTICLES)")
print("==========================================================================")

# Load 30 frames
frame_particles = []
for t, filepath in enumerate(origin_files_30):
    data = np.loadtxt(filepath, skiprows=1)
    frame_particles.append(data[:, 1:4])

# Load GT tracks
gt_tracks = []
if os.path.exists(origin_hdf5_30):
    with h5py.File(origin_hdf5_30, 'r') as f:
        for k in f.keys():
            gt_tracks.append(f[k][:])

gt_p0 = np.array([tr[0, 1:4] for tr in gt_tracks])

modular_trackers = {
    "Cython 3D Segment Priority": Cython3DTracker(v_max=0.015, a_max=0.010, dt=1.0),
    "Fast 3D Smooth (SG)": Fast3DSmoothTracker(v_max=0.015, dacc=0.010, smooth_window=5, dt=1.0),
    "MyPTV / Nearest Hungarian": MyPTV3DTracker(v_max=0.015, a_max=0.010, max_gap=2, dt=1.0),
    "3D Kalman-Hungarian": Quality3DTracker(v_max=0.015, a_max=0.010, dt=1.0),
    "proPTV (Predictive GMM)": ProPTVTracker(ProPTVConfig(t_init=3, maxvel=0.015, angle=60.0, dt=1.0, Vmin=[0,0,0], Vmax=[1,1,1])),
}

results = {}

for name, tracker in modular_trackers.items():
    t0 = time.perf_counter()
    reconstructed_tracks = tracker.track_frames(frame_particles)
    t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
    
    num_tracks = len(reconstructed_tracks)
    if num_tracks > 0 and len(gt_p0) > 0:
        rec_p0 = np.array([tr["pos"][0] for tr in reconstructed_tracks])
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
    
    print(f"  • {name:<28}: {num_tracks} tracks | PMP: {pmp:.2f}% | Mean Err: {mean_err:.6f} | Time: {t_elapsed:.2f} ms")

print("\n==========================================================================")
print(f"{'Modular Plugin Name':<30} | {'Tracks':<8} | {'PMP %':<8} | {'Mean 3D Error':<13} | {'Time (ms)':<9}")
print("-" * 77)
for name, res in results.items():
    print(f"{name:<30} | {res['tracks']:<8} | {res['pmp']:<8.2f} | {res['mean_err']:<13.6f} | {res['time_ms']:<9.2f}")
print("==========================================================================\n")

# Plot figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(results.keys())
pmps = [results[n]["pmp"] for n in names]
times = [results[n]["time_ms"] for n in names]

colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

axes[0].barh(names, pmps, color=colors)
axes[0].set_xlim(0, 105)
axes[0].set_xlabel('Perfect Matching Percentage (PMP %)', fontsize=12, fontweight='bold')
axes[0].set_title('Tracking Accuracy (30 Frames - 500 Particles)', fontsize=13, fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].barh(names, times, color=colors)
axes[1].set_xscale('log')
axes[1].set_xlabel('Execution Time (ms - Log Scale)', fontsize=12, fontweight='bold')
axes[1].set_title('Tracking Speed Benchmark', fontsize=13, fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('scratch/all_modular_plugins_30_frames.png', dpi=200)
print("Saved benchmark plot to scratch/all_modular_plugins_30_frames.png")
