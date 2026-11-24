# ruff: noqa
"""Truth-free counterpart of loss_in_episodes.py: tracker breaks and missing views against streak flags.

    uv run python bench/limits/loss_streaks_real.py <label> <run dir (res/run.zarr, parameters_Run1.yaml)> <f0> <f1> [K]

(1) 3-camera points: does the missing view land on an existing blob (1.5 px / 3 px)? -> a shared (or mis-matched) blob
    instead of no blob.  On syn1 this is checked against truth in loss_in_episodes.py.
(2) Track ends and starts (not at the window edge) against mid-track points: share with a flagged blob in any camera.
    flagged = streak mass ratio > 1.3 or another streak ended / started within 3 px in the 6 frames before / after.
(3) For the flagged blobs at track ends: is the same blob unflagged (clean) on both sides within K frames on its streak?
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import streak_lib as sl
from openptv2.loco import project_pixels
from openptv2.merge_post import arrays_from_store
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore

label, root, F0, F1 = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
K = int(sys.argv[5]) if len(sys.argv) > 5 else 6
st = RunStore.open(root, mode="r")
cpar, vpar, cals, _ = load_parameters(root / "parameters_Run1.yaml")
a = arrays_from_store(st, "ptv_is", F0, F1, 4)
frame, blob, tid, pos = a["frame"], a["blob"], a["tid"], a["pos"]
B = {(c, f): np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for f in range(F0, F1 + 1) for c in range(4)}
D = np.vstack([sl.node_features(B, c, F0, F1) for c in range(4)])
idx = {(int(c), int(f), int(j)): i for i, (c, f, j) in enumerate(D[:, :3])}
flag = (np.nan_to_num(D[:, 5], nan=1.0) > 1.3) | (D[:, 6] < sl.R_TOPO) | (D[:, 7] < sl.R_TOPO)
unit_ok = ~((np.nan_to_num(D[:, 5], nan=1.0) > 1.15) | (D[:, 6] < sl.R_TOPO) | (D[:, 7] < sl.R_TOPO))
ncam = (blob >= 0).sum(1)
print(f"[{label}] frames {F0}-{F1}: {len(frame)} points, cameras per point: " + ", ".join(f"{k}: {100 * (ncam == k).mean():.0f}%" for k in (4, 3, 2)))

# (1) the missing view of 3-camera points
inner = (frame >= F0 + K) & (frame <= F1 - K)
r3 = np.flatnonzero((ncam == 3) & inner)
res = {"blob within 1.5 px": 0, "blob within 1.5-3 px": 0, "no blob within 3 px": 0}
for c in range(4):
    sel = r3[blob[r3, c] < 0]
    for f in np.unique(frame[sel]):
        s2 = sel[frame[sel] == f]
        if len(B[c, int(f)]) == 0:
            res["no blob within 3 px"] += len(s2); continue
        pr = project_pixels(pos[s2], cals[c], cpar)
        d = cKDTree(B[c, int(f)][:, 1:3]).query(pr)[0]
        res["blob within 1.5 px"] += int((d <= 1.5).sum()); res["blob within 1.5-3 px"] += int(((d > 1.5) & (d <= 3)).sum()); res["no blob within 3 px"] += int((d > 3).sum())
tot = sum(res.values())
print(f"  (1) missing view of 3-camera points ({tot}): " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in res.items()))

# (2) track ends / starts / mid-track points
by = defaultdict(list)
for i in np.flatnonzero(inner | (frame >= F0)):
    by[int(tid[i])].append(i)
def anyflag(i):
    return any(flag[idx[c, int(frame[i]), int(blob[i, c])]] for c in range(4) if blob[i, c] >= 0 and (c, int(frame[i]), int(blob[i, c])) in idx)
ends, starts, mids = [], [], []
for t, rows in by.items():
    rows = sorted(rows, key=lambda i: frame[i])
    if len(rows) < 6: continue
    if frame[rows[-1]] < F1 - K and frame[rows[-1]] > F0 + K: ends.append(rows[-1])
    if frame[rows[0]] > F0 + K and frame[rows[0]] < F1 - K: starts.append(rows[0])
    for i in rows[2:-2]:
        if F0 + K <= frame[i] <= F1 - K: mids.append(i)
mids = np.array(mids)[::7]
rate = lambda rows: np.mean([anyflag(i) for i in rows]) if len(rows) else np.nan
print(f"  (2) points with a flagged blob in some camera: track ends {100 * rate(ends):.1f}% (n={len(ends)}), track starts {100 * rate(starts):.1f}% (n={len(starts)}), mid-track {100 * rate(mids):.1f}% (n={len(mids)})")
def nflag(i): return sum(flag[idx[c, int(frame[i]), int(blob[i, c])]] for c in range(4) if blob[i, c] >= 0 and (c, int(frame[i]), int(blob[i, c])) in idx)
for nm, rows in (("ends", ends), ("starts", starts), ("mid-track", mids)):
    v = np.array([nflag(i) for i in rows])
    print(f"      {nm:10s} flagged cameras 0/1/2/3+: " + "/".join(f"{100 * (v == k).mean() if k < 3 else 100 * (v >= 3).mean():.0f}%" for k in range(4)))
# 3 vs 4 cameras at the ends
e = np.array(ends); s = np.array(starts)
print(f"      cameras used: ends with 4 cameras {100 * (ncam[e] == 4).mean():.0f}%, starts {100 * (ncam[s] == 4).mean():.0f}%, mid-track {100 * (ncam[mids] == 4).mean():.0f}%")

# (3) flagged end blobs: clean on both sides within K on the same streak?
tids_cam = {c: sl.link_cam(B, c, F0, F1) for c in range(4)}
by_tr = {c: defaultdict(list) for c in range(4)}
for c in range(4):
    for (f, j), t in tids_cam[c].items(): by_tr[c][t].append((f, j))
n_f = n_both = n_one = 0
for i in ends + starts:
    f = int(frame[i])
    for c in range(4):
        j = int(blob[i, c])
        if j < 0 or (c, f, j) not in idx or not flag[idx[c, f, j]]: continue
        nodes = by_tr[c][tids_cam[c][f, j]]
        left = any(g < f and f - g <= K and unit_ok[idx[c, g, jj]] for g, jj in nodes)
        right = any(g > f and g - f <= K and unit_ok[idx[c, g, jj]] for g, jj in nodes)
        n_f += 1; n_both += left and right; n_one += (left != right)
print(f"  (3) flagged blobs at track ends/starts: {n_f}; unflagged frame on both sides within {K} frames on the streak: {100 * n_both / n_f:.0f}%, one side only {100 * n_one / n_f:.0f}%, none {100 * (n_f - n_both - n_one) / n_f:.0f}%")
