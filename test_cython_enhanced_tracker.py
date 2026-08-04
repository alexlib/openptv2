import glob
import time
import numpy as np
from pathlib import Path
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.plugins._assignment import match_within_radius

wp1_res = Path(r"C:\Users\alex\Downloads\hidimaging_test\TT13_aorta\wp1\res")

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

frame_files = sorted(glob.glob(str(wp1_res / "rt_is.*")), key=lambda x: int(Path(x).suffix[1:]))[:10]
frames = [load_rt_is(Path(f)) for f in frame_files]

print("=== Benchmarking Tracking Algorithms on 10 Frames ===")
print(f"Total particles across 10 frames: {sum(len(f) for f in frames)}")

# 1. MyPTV 3D Python + SciPy
t0 = time.perf_counter()
tracker_myptv = MyPTV3DTracker(v_max=8.0, a_max=50.0, max_gap=1, dt=1.0)
trajs_myptv = tracker_myptv.track_frames(frames)
t_myptv = (time.perf_counter() - t0) * 1000
pts_myptv = sum(len(t['pos']) for t in trajs_myptv)

print(f"\n1. MyPTV 3D (Python/SciPy):")
print(f"   Time: {t_myptv:.2f} ms")
print(f"   Tracked points: {pts_myptv} / {sum(len(f) for f in frames)} ({pts_myptv/sum(len(f) for f in frames)*100:.1f}%)")
print(f"   Total trajectories: {len(trajs_myptv)}")

# 2. Prototype Enhanced Cython-style Fast Tracker in Python first to verify logic
def fast_cython_enhanced_tracker(frames, v_max=8.0, a_max=25.0):
    num_frames = len(frames)
    if num_frames < 2:
        return []
    
    # Active tracks: dict of pos, time, vel, id
    active_tracks = []
    completed_tracks = []
    next_id = 1
    
    for p in frames[0]:
        active_tracks.append({"id": next_id, "pos": [p], "time": [0], "vel": [np.zeros(3)], "gap": 0})
        next_id += 1
        
    for f in range(1, num_frames):
        cand_pts = frames[f]
        if len(active_tracks) == 0 or len(cand_pts) == 0:
            completed_tracks.extend(active_tracks)
            active_tracks = []
            for p in cand_pts:
                active_tracks.append({"id": next_id, "pos": [p], "time": [f], "vel": [np.zeros(3)], "gap": 0})
                next_id += 1
            continue
            
        # Predictions
        last_p = np.array([tr["pos"][-1] for tr in active_tracks])
        last_v = np.array([tr["vel"][-1] for tr in active_tracks])
        seeded = np.fromiter((len(tr["pos"]) > 1 for tr in active_tracks), dtype=bool)
        
        pred = np.where(seeded[:, None], last_p + last_v, last_p)
        radius = np.where(seeded, a_max, v_max)
        
        # Fast Component / Hungarian Matcher
        rows, cols = match_within_radius(pred, cand_pts, radius)
        
        matched_cands = set(cols)
        matched_tracks = set(rows)
        
        for r, c in zip(rows, cols):
            tr = active_tracks[r]
            new_p = cand_pts[c]
            v_new = new_p - tr["pos"][-1]
            tr["pos"].append(new_p)
            tr["time"].append(f)
            tr["vel"].append(v_new)
            tr["gap"] = 0
            
        new_active = []
        for i, tr in enumerate(active_tracks):
            if i not in matched_tracks:
                tr["gap"] += 1
                if tr["gap"] <= 1:
                    new_active.append(tr)
                else:
                    completed_tracks.append(tr)
            else:
                new_active.append(tr)
                
        for c in range(len(cand_pts)):
            if c not in matched_cands:
                new_active.append({"id": next_id, "pos": [cand_pts[c]], "time": [f], "vel": [np.zeros(3)], "gap": 0})
                next_id += 1
                
        active_tracks = new_active
        
    completed_tracks.extend(active_tracks)
    return [t for t in completed_tracks if len(t["pos"]) >= 2]

t0 = time.perf_counter()
trajs_proto = fast_cython_enhanced_tracker(frames, v_max=8.0, a_max=25.0)
t_proto = (time.perf_counter() - t0) * 1000
pts_proto = sum(len(t['pos']) for t in trajs_proto)

print(f"\n2. Enhanced Tracker Prototype:")
print(f"   Time: {t_proto:.2f} ms")
print(f"   Tracked points: {pts_proto} / {sum(len(f) for f in frames)} ({pts_proto/sum(len(f) for f in frames)*100:.1f}%)")
print(f"   Total trajectories: {len(trajs_proto)}")
