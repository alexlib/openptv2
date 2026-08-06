import glob
import numpy as np
from pathlib import Path
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

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

total_pts = sum(len(f) for f in frames)
print(f"Total frame particles across 10 frames: {total_pts}")

# 1. MyPTV with max_gap = 0 (No gap recovery - strictly frame-to-frame)
tracker_gap0 = MyPTV3DTracker(v_max=20.0, a_max=30.0, max_gap=0, dt=1.0)
trajs_gap0 = tracker_gap0.track_frames(frames)
pts_gap0 = sum(len(t['pos']) for t in trajs_gap0)
links_gap0 = sum(len(t['pos']) - 1 for t in trajs_gap0)
avg_links_gap0 = links_gap0 / 9.0

# 2. MyPTV with max_gap = 1 (With 1-frame gap recovery)
tracker_gap1 = MyPTV3DTracker(v_max=20.0, a_max=30.0, max_gap=1, dt=1.0)
trajs_gap1 = tracker_gap1.track_frames(frames)
pts_gap1 = sum(len(t['pos']) for t in trajs_gap1)
links_gap1 = sum(len(t['pos']) - 1 for t in trajs_gap1)
avg_links_gap1 = links_gap1 / 9.0

print("\n=== Quantifying the 13% Gap Causes ===")
print(f"1. MyPTV without Gap Recovery (max_gap=0):")
print(f"   Average Links / Step: {avg_links_gap0:.1f} / {385.4} ({avg_links_gap0/385.4*100:.1f}% retention)")
print(f"   Tracked points: {pts_gap0} / {total_pts} ({pts_gap0/total_pts*100:.1f}%)")

print(f"\n2. MyPTV with Gap Recovery (max_gap=1):")
print(f"   Average Links / Step: {avg_links_gap1:.1f} / {385.4} ({avg_links_gap1/385.4*100:.1f}% retention)")
print(f"   Tracked points: {pts_gap1} / {total_pts} ({pts_gap1/total_pts*100:.1f}%)")

diff_links = avg_links_gap1 - avg_links_gap0
print(f"\nContribution of Gap Recovery alone: +{diff_links:.1f} links/step (+{diff_links/385.4*100:.1f}% retention!)")
