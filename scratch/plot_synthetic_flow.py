import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import openptv2.benchmarking as bm

# Generate the default benchmark scenario
spec = bm.ScenarioSpec(
    num_particles=50,
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

fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# Color map for trajectories
cmap = plt.colormaps['plasma']
colors = [cmap(i) for i in np.linspace(0, 1, len(true_tracks))]

for idx, (pid, points) in enumerate(true_tracks.items()):
    pts = np.array([(x, y, z) for f, x, y, z in points])
    if len(pts) > 0:
        # Plot trajectory line
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.7, linewidth=1.5, color=colors[idx % len(colors)])
        # Mark start point
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color='green', s=15, alpha=0.9, zorder=5)
        # Mark end point
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='red', s=15, alpha=0.9, zorder=5)

ax.set_title("Synthetic Benchmark Flow (Turbulent OU-Inertia Model)", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("X (mm)", labelpad=10)
ax.set_ylabel("Y (mm)", labelpad=10)
ax.set_zlabel("Z (mm)", labelpad=10)

# Set domain bounds
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-40, 40)

# Clean grid styling
ax.grid(True, linestyle='--', alpha=0.3)
ax.view_init(elev=25, azim=45)

out_dir = Path("C:/Users/alex/.gemini/antigravity-cli/brain/e6c485aa-6bb2-492c-9e33-bee2bcaf6728")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "synthetic_flow_trajectories.png"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Saved plot to {out_path}")
