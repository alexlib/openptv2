"""
Quick probe script: Estimate interparticle distance and inter-frame displacement
using the Burgers dataset and the project's rt_is file reader.
"""
import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../pyptv"))
from gui.pyptv.ptv import read_rt_is_file

data_dir = os.path.join(os.path.dirname(__file__), "../../test_data/burgers/res_orig")

# Load frames
frames = []
for i in range(10001, 10006):
    fname = os.path.join(data_dir, f"rt_is.{i:04d}")
    arr = read_rt_is_file(fname)
    arr = np.array(arr)  # shape (N, 7)
    frames.append(arr[:, :3])  # only x, y, z
print(f"Loaded {len(frames)} frames, each with {[f.shape[0] for f in frames]} particles")

# Per-frame interparticle distance
for idx, arr in enumerate(frames):
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    vol = np.prod(maxs - mins)
    n = arr.shape[0]
    ipd = (vol / n) ** (1/3) if n > 0 else np.nan
    print(f"Frame {idx}: Volume={vol:.2f} mm^3, N={n}, Interparticle dist~{ipd:.2f} mm")

# Merge 4 frames for fictitious density
merged = np.vstack(frames[:4])
mins = merged.min(axis=0)
maxs = merged.max(axis=0)
vol = np.prod(maxs - mins)
n = merged.shape[0]
ipd_merged = (vol / n) ** (1/3) if n > 0 else np.nan
print(f"Merged 4 frames: Volume={vol:.2f} mm^3, N={n}, Interparticle dist~{ipd_merged:.2f} mm")

# Inter-frame displacement
all_disp = []
for i in range(4):
    arr1, arr2 = frames[i], frames[i+1]
    if arr1.shape == arr2.shape:
        d = np.linalg.norm(arr2 - arr1, axis=1)
        all_disp.append(d)
    else:
        print(f"Frame {i} and {i+1} have different particle counts; skipping displacement calc.")
if all_disp:
    all_disp = np.concatenate(all_disp)
    print(f"Inter-frame displacement: min={all_disp.min():.2f}, max={all_disp.max():.2f}, mean={all_disp.mean():.2f} mm")
else:
    print("Inter-frame displacement: not computed (mismatched particle counts)")
