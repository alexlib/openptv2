# ruff: noqa: E402
import os
from pathlib import Path

import numpy as np

os.chdir("/home/user/Documents/GitHub/openptv2")

from algorithms.tracking_frame_buf import read_targets

TEST_DATA_DIR = Path("test_data/test_cavity")

# Load 10002 targets for Cam 1 (index 0)
t2 = read_targets(str(TEST_DATA_DIR / "img/cam1.%d"), 10002)

target_x = 1204.73
target_y = 181.65

closest_dist = 1e9
closest_idx = -1

for i, t in enumerate(t2):
    dist = np.sqrt((t.x - target_x) ** 2 + (t.y - target_y) ** 2)
    if dist < closest_dist:
        closest_dist = dist
        closest_idx = i

print(
    f"Target in 10002 closest to (1204.73, 181.65): index={closest_idx}, pos={t2[closest_idx].x:.2f},{t2[closest_idx].y:.2f}, dist={closest_dist:.2f}"
)
print(f"Target 113 in 10002 pos: {t2[113].x:.2f},{t2[113].y:.2f}")
