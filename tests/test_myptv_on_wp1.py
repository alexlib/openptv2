import glob
import numpy as np
from pathlib import Path
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

wp1_res = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\res")

def load_rt_is(filepath):
    lines = filepath.read_text().strip().splitlines()
    if not lines:
        return np.zeros((0, 3))
    n_pts = int(lines[0])
    pts = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts)

frame_files = sorted(glob.glob(str(wp1_res / "rt_is.*")), key=lambda x: int(Path(x).suffix[1:]))[:10]
frames = [load_rt_is(Path(f)) for f in frame_files]

print(f"Loaded {len(frames)} frames of 3D positions.")
for i, f in enumerate(frames):
    print(f"  Frame {i+1}: {len(f)} 3D particles")

for v_max in [3.0, 5.0, 8.0, 10.0]:
    for a_max in [10.0, 25.0, 50.0]:
        tracker = MyPTV3DTracker(v_max=v_max, a_max=a_max, max_gap=2, dt=0.1)
        trajs = tracker.track_frames(frames)
        long_trajs = [t for t in trajs if len(t['pos']) >= 3]
        total_linked_pts = sum(len(t['pos']) for t in trajs)
        print(f"MyPTV3D (v_max={v_max:4.1f}, a_max={a_max:4.1f}): {len(trajs)} total trajectories, {len(long_trajs)} long (>=3 frames), {total_linked_pts} total points tracked")
