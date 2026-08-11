"""Verify that tracker output trajectories are continuous paths matching ground truth without ID swaps or jumps."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import openptv2.benchmarking as bm
from openptv2.tracking_metrics import calculate_tracking_metrics

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_synthetic_turbulent import make_dataset, OUT_DIR, FIRST_FRAME

if not (OUT_DIR / "parameters_Run1.yaml").exists():
    make_dataset(OUT_DIR, num_particles=220, num_frames=N_FRAMES, seed=2026)

SRC = OUT_DIR.resolve()
FIRST = FIRST_FRAME
N_FRAMES = 30

# 1. Run Tracker
pred_tracks = bm.run_tracker(SRC / "parameters_Run1.yaml", "fast_3d")

# 2. Extract Ground Truth Tracks
frames = {}
for fn in range(FIRST, FIRST + N_FRAMES):
    p = SRC / "res" / f"origin_{fn}.txt"
    if not p.exists():
        continue
    rows = []
    for line in p.read_text().strip().splitlines()[1:]:
        parts = line.split(",")
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    frames[fn - FIRST] = rows

true_tracks = {}
for fn, rows in frames.items():
    for pid, x, y, z in rows:
        if pid >= 0:
            true_tracks.setdefault(pid, []).append((fn, x, y, z))

# Shift pred_tracks frame index to 0-based to match ground truth
pred0 = {k: [(f - FIRST, x, y, z) for (f, x, y, z) in v] for k, v in pred_tracks.items()}

# Compute link-level correctness
link_metrics = calculate_tracking_metrics(true_tracks, pred0, distance_tolerance=1.0)

print("=== Tracker Continuous Path Verification ===")
print(f"Total Reconstructed Tracks: {len(pred_tracks)}")
print(f"Precision (True Positive Links / Total Links): {link_metrics.precision:.3f}")
print(f"Yield Recall (True Positive Links / True Links): {link_metrics.yield_recall:.3f}")
print(f"False Connection Rate (FCR): {link_metrics.false_connection_rate:.3%}")

# Measure tracked velocity smoothness
tracked_vel_norms = []
tracked_acc_norms = []

for trk_id, pts_list in pred_tracks.items():
    if len(pts_list) < 3:
        continue
    pts = np.array([(x, y, z) for f, x, y, z in pts_list])
    v = np.diff(pts, axis=0)
    a = np.diff(v, axis=0)
    tracked_vel_norms.extend(np.linalg.norm(v, axis=1))
    tracked_acc_norms.extend(np.linalg.norm(a, axis=1))

print(f"Tracked Speed Mean: {np.mean(tracked_vel_norms):.3f} +/- {np.std(tracked_vel_norms):.3f} mm/frame")
print(f"Tracked Acceleration Residual Mean: {np.mean(tracked_acc_norms):.3f} +/- {np.std(tracked_acc_norms):.3f} mm/frame^2")

# Plot GT vs Tracked trajectory comparison
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Select 3 longest predicted tracks
longest_pids = sorted(pred_tracks.keys(), key=lambda k: len(pred_tracks[k]), reverse=True)[:3]

for idx, trk_id in enumerate(longest_pids):
    pts = np.array([(f, x, y, z) for f, x, y, z in pred_tracks[trk_id]])
    ax.plot(pts[:, 0], pts[:, 1], 'o-', label=f"Tracked Track {trk_id}", alpha=0.9, linewidth=2)

ax.set_title("Reconstructed Trajectory Continuity (Position X vs Frame)", fontweight='bold')
ax.set_xlabel("Frame Number")
ax.set_ylabel("X Position (mm)")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = Path("C:/Users/alex/.gemini/antigravity-cli/brain/e6c485aa-6bb2-492c-9e33-bee2bcaf6728")
out_path = out_dir / "tracker_continuity_verification.png"
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Saved tracker continuity plot to {out_path}")
