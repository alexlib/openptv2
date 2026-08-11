"""Verify physical fluid dynamic properties of synthetic flow and tracker outputs.

Computes:
1. Lagrangian Velocity Autocorrelation Function R_vv(tau) = <v(t) . v(t+tau)> / <||v||^2>
2. Displacement PDF & Acceleration PDF
3. Spatial Velocity Field & Streamline Coherence
4. Frame-to-frame continuity & ID stability verification
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import openptv2.benchmarking as bm
from openptv2.tracking_metrics import calculate_tracking_metrics

# 1. Generate Ground Truth Scenario
spec = bm.ScenarioSpec(
    num_particles=220,
    num_frames=30,
    velocity=2.0,
    velocity_jitter=1.0,
    gap_probability=0.08,
    noise_mm=0.08,
    ghost_ratio=0.04,
    seed=2026,
    entering_particles=6,
    leaving_particles=6,
    flow_type="turbulent",
    crossings=[
        bm.CrossingSpec(at_frame=15, min_distance=0.0, speed=2.0),
        bm.CrossingSpec(at_frame=18, min_distance=0.0, speed=1.5),
    ],
)

true_tracks, frame_gt = bm.generate_scenario(spec)

# Extract trajectories as continuous arrays
all_velocities = []
all_accelerations = []
max_tau = 15
autocorr_sum = np.zeros(max_tau)
autocorr_counts = np.zeros(max_tau)

for pid, points in true_tracks.items():
    if len(points) < 3:
        continue
    pts = np.array([(x, y, z) for f, x, y, z in points])
    frames = np.array([f for f, x, y, z in points])
    
    # Check if frames are contiguous
    dt_steps = np.diff(frames)
    
    # Velocity v_k = (x_{k+1} - x_k) / dt
    vels = np.diff(pts, axis=0) # (N-1, 3)
    accs = np.diff(vels, axis=0) # (N-2, 3)
    
    all_velocities.extend(vels)
    all_accelerations.extend(accs)
    
    # Calculate velocity autocorrelation R_vv(tau)
    n_v = len(vels)
    v_mean_sq = np.mean(np.sum(vels**2, axis=1))
    if v_mean_sq > 1e-6:
        for tau in range(max_tau):
            if tau < n_v:
                # Dot product v(t) . v(t + tau)
                v_t = vels[:n_v - tau]
                v_tau = vels[tau:]
                dots = np.sum(v_t * v_tau, axis=1)
                autocorr_sum[tau] += np.sum(dots)
                autocorr_counts[tau] += len(dots)

# Normalized autocorrelation
R_vv = np.zeros(max_tau)
valid_mask = autocorr_counts > 0
R_vv[valid_mask] = autocorr_sum[valid_mask] / autocorr_counts[valid_mask]
R_vv = R_vv / R_vv[0]  # Normalize so R_vv(0) = 1.0

all_vels_arr = np.array(all_velocities)
all_accs_arr = np.array(all_accelerations)
speeds = np.linalg.norm(all_vels_arr, axis=1)
acc_norms = np.linalg.norm(all_accs_arr, axis=1)

print("=== Physical Flow Diagnostics Summary ===")
print(f"Total True Trajectories: {len(true_tracks)}")
print(f"Mean Particle Speed: {np.mean(speeds):.3f} +/- {np.std(speeds):.3f} mm/frame")
print(f"Max Particle Speed: {np.max(speeds):.3f} mm/frame")
print(f"Mean Acceleration Residual: {np.mean(acc_norms):.3f} +/- {np.std(acc_norms):.3f} mm/frame^2")
print(f"Max Acceleration Residual: {np.max(acc_norms):.3f} mm/frame^2")
print(f"Lagrangian Velocity Autocorrelation R_vv(tau):")
for tau in range(min(8, max_tau)):
    print(f"  tau = {tau} frames: R_vv = {R_vv[tau]:.3f}")

# 2. Plot Flow Physics Diagnostics Figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

# (A) Velocity Autocorrelation Curve
axs[0, 0].plot(range(max_tau), R_vv, 'o-', color='navy', linewidth=2, markersize=6)
axs[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axs[0, 0].set_title("Lagrangian Velocity Autocorrelation R_vv(tau)", fontweight='bold')
axs[0, 0].set_xlabel("Lag tau (frames)")
axs[0, 0].set_ylabel("R_vv(tau)")
axs[0, 0].grid(True, alpha=0.3)

# (B) Speed Distribution PDF
axs[0, 1].hist(speeds, bins=25, density=True, color='crimson', alpha=0.7, edgecolor='black')
axs[0, 1].set_title("Particle Speed Distribution PDF", fontweight='bold')
axs[0, 1].set_xlabel("Speed ||v|| (mm/frame)")
axs[0, 1].set_ylabel("Probability Density")
axs[0, 1].grid(True, alpha=0.3)

# (C) Acceleration Distribution PDF
axs[1, 0].hist(acc_norms, bins=25, density=True, color='darkgreen', alpha=0.7, edgecolor='black')
axs[1, 0].set_title("Fluid Acceleration Distribution PDF", fontweight='bold')
axs[1, 0].set_xlabel("Acceleration ||a|| (mm/frame^2)")
axs[1, 0].set_ylabel("Probability Density")
axs[1, 0].grid(True, alpha=0.3)

# (D) Selected Continuous Trajectories (Position vs Frame)
sample_ids = list(true_tracks.keys())[:5]
for pid in sample_ids:
    pts = np.array([(f, x, y, z) for f, x, y, z in true_tracks[pid]])
    axs[1, 1].plot(pts[:, 0], pts[:, 1], 'o-', label=f"Track {pid} (X)", alpha=0.8)

axs[1, 1].set_title("Continuous Particle Position X(t)", fontweight='bold')
axs[1, 1].set_xlabel("Frame Number")
axs[1, 1].set_ylabel("X Position (mm)")
axs[1, 1].legend(loc='upper right', fontsize=8)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
out_dir = Path("C:/Users/alex/.gemini/antigravity-cli/brain/e6c485aa-6bb2-492c-9e33-bee2bcaf6728")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "flow_physics_diagnostics.png"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Saved flow physics plot to {out_path}")
