import glob
from pathlib import Path

import numpy as np

from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

wp1_dir = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1")
wp1_res = wp1_dir / "res"


def load_rt_is(filepath):
    lines = filepath.read_text().strip().splitlines()
    if not lines:
        return np.zeros((0, 3))
    pts = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts)


frame_files = sorted(
    glob.glob(str(wp1_res / "rt_is.*")), key=lambda x: int(Path(x).suffix[1:])
)[:10]
frames = [load_rt_is(Path(f)) for f in frame_files]

print(f"Loaded {len(frames)} frames for empirical parameter extraction.")

# Run MyPTV 3D tracking with wide bounds to capture full motion range
tracker = MyPTV3DTracker(v_max=10.0, a_max=50.0, max_gap=1, dt=1.0)
trajectories = tracker.track_frames(frames)

print(f"MyPTV 3D produced {len(trajectories)} trajectories.")

# Extract frame-to-frame displacements (velocities) and accelerations
all_dx = []
all_dy = []
all_dz = []
all_mags = []
all_accs = []

for tr in trajectories:
    pos = tr["pos"]
    if len(pos) < 2:
        continue
    # Displacements
    disps = np.diff(pos, axis=0)  # (N-1, 3)
    for d in disps:
        all_dx.append(d[0])
        all_dy.append(d[1])
        all_dz.append(d[2])
        all_mags.append(np.linalg.norm(d))

    # Accelerations (second differences)
    if len(pos) >= 3:
        accs = np.diff(disps, axis=0)  # (N-2, 3)
        for a in accs:
            all_accs.append(np.linalg.norm(a))

dx_arr = np.array(all_dx)
dy_arr = np.array(all_dy)
dz_arr = np.array(all_dz)
mag_arr = np.array(all_mags)
acc_arr = np.array(all_accs)

print("\n=== Empirical Motion Statistics from MyPTV Trajectories ===")
print(f"Total frame-to-frame displacements analyzed: {len(mag_arr)}")
print(f"Total accelerations analyzed: {len(acc_arr)}")

print("\nVelocity dX (mm/frame):")
print(f"  min={dx_arr.min():.3f}, max={dx_arr.max():.3f}, mean={dx_arr.mean():.3f}")
print(f"  p1={np.percentile(dx_arr, 1):.3f}, p99={np.percentile(dx_arr, 99):.3f}")
print(
    f"  p0.5={np.percentile(dx_arr, 0.5):.3f}, p99.5={np.percentile(dx_arr, 99.5):.3f}"
)

print("\nVelocity dY (mm/frame):")
print(f"  min={dy_arr.min():.3f}, max={dy_arr.max():.3f}, mean={dy_arr.mean():.3f}")
print(f"  p1={np.percentile(dy_arr, 1):.3f}, p99={np.percentile(dy_arr, 99):.3f}")
print(
    f"  p0.5={np.percentile(dy_arr, 0.5):.3f}, p99.5={np.percentile(dy_arr, 99.5):.3f}"
)

print("\nVelocity dZ (mm/frame):")
print(f"  min={dz_arr.min():.3f}, max={dz_arr.max():.3f}, mean={dz_arr.mean():.3f}")
print(f"  p1={np.percentile(dz_arr, 1):.3f}, p99={np.percentile(dz_arr, 99):.3f}")
print(
    f"  p0.5={np.percentile(dz_arr, 0.5):.3f}, p99.5={np.percentile(dz_arr, 99.5):.3f}"
)

print("\n3D Displacement Magnitude (mm/frame):")
print(
    f"  median={np.median(mag_arr):.3f}, p95={np.percentile(mag_arr, 95):.3f}, p99={np.percentile(mag_arr, 99):.3f}, max={mag_arr.max():.3f}"
)

if len(acc_arr) > 0:
    print("\n3D Acceleration Magnitude dacc (mm/frame^2):")
    print(
        f"  median={np.median(acc_arr):.3f}, p95={np.percentile(acc_arr, 95):.3f}, p99={np.percentile(acc_arr, 99):.3f}, max={acc_arr.max():.3f}"
    )

# Recommend tightest bounds encompassing 99.5% of real particles
dvxmin = float(np.floor(np.percentile(dx_arr, 0.5) * 10) / 10.0)
dvxmax = float(np.ceil(np.percentile(dx_arr, 99.5) * 10) / 10.0)

dvymin = float(np.floor(np.percentile(dy_arr, 0.5) * 10) / 10.0)
dvymax = float(np.ceil(np.percentile(dy_arr, 99.5) * 10) / 10.0)

dvzmin = float(np.floor(np.percentile(dz_arr, 0.5) * 10) / 10.0)
dvzmax = float(np.ceil(np.percentile(dz_arr, 99.5) * 10) / 10.0)

dacc_rec = (
    float(np.ceil(np.percentile(acc_arr, 99.5) * 10) / 10.0)
    if len(acc_arr) > 0
    else 5.0
)

print("\n=== Recommended Empirical Parameters for track_3d / fast_3d ===")
print(f"  dvxmin: {dvxmin:.1f}")
print(f"  dvxmax: {dvxmax:.1f}")
print(f"  dvymin: {dvymin:.1f}")
print(f"  dvymax: {dvymax:.1f}")
print(f"  dvzmin: {dvzmin:.1f}")
print(f"  dvzmax: {dvzmax:.1f}")
print(f"  dacc:   {dacc_rec:.1f}")
