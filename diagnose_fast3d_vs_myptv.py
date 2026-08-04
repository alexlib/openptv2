import glob
import numpy as np
from pathlib import Path
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

# Let's inspect Step 1 (Frame 1 -> Frame 2)
p1 = frames[0]
p2 = frames[1]

# In Step 1, all 390 particles in Frame 1 are NEW (no velocity history), so Level 3 / static prediction is used!
# Position prediction = p1
# Search radius = v_max = 20.0 mm

print(f"Frame 1 particles: {len(p1)}, Frame 2 particles: {len(p2)}")

# MyPTV matching for Step 1:
r_rows, r_cols = match_within_radius(p1, p2, 20.0)
print(f"MyPTV Step 1 matched: {len(r_rows)} / {len(p1)} particles ({len(r_rows)/len(p1)*100:.1f}%)")

# How many closest candidates are found within the box in fast_3d?
# In fast_3d: _find_closest_in_3d uses MAX_CANDS = 4.
# It searches for up to 4 closest candidates within the box [dx, dy, dz] = [20.0, 18.0, 23.0]
# BUT in fast_3d, Level 1 / Level 2 / Level 3 logic in track3d_loop_fast:

# Let's check how many particles in Frame 1 have a candidate in Frame 2 in fast_3d:
box_matches = 0
for i in range(len(p1)):
    cx, cy, cz = p1[i]
    # Check candidates in p2
    cands = []
    for k in range(len(p2)):
        ddx = abs(p2[k, 0] - cx)
        ddy = abs(p2[k, 1] - cy)
        ddz = abs(p2[k, 2] - cz)
        if ddx < 18.0 and ddy < 18.0 and ddz < 23.0:
            cands.append(k)
    if cands:
        box_matches += 1

print(f"fast_3d candidates found in box: {box_matches} / {len(p1)}")
