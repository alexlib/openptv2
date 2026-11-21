"""Where is yield lost? Truth points grouped by how many cameras see them alone in a blob (clean), with the share found
by the matcher cloud and by a finished run.  Usage: joint_yield_breakdown.py <case> <run tag, e.g. holistic_d1>."""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from pathlib import Path

import synth_bench as sb

case, tag = sys.argv[1], sys.argv[2]
cd = sb.WORK / "cases" / case
tr = np.load(cd / "truth.npz")
pred = np.load(sb.WORK / "runs" / case / tag / "pred.npz")
tid, fr, pos, bt = tr["particle_id"], tr["frame"], tr["pos_mm"], tr["blob_tid"]
by = {}
from scipy.spatial import cKDTree

for f in np.unique(fr):
    m = fr == f
    by[int(f)] = (pos[m], tid[m], cKDTree(pos[m]))
g0, _, off = sb._match(by, pred["frame"], pred["pos"], 0.6)
bias = np.median(off[g0 >= 0], axis=0)
gid, _, _ = sb._match(by, pred["frame"], pred["pos"] - bias, 0.4)
def key(i, f):
    return i.astype(np.int64) * 100000 + f
have = np.isin(key(tid, fr), np.unique(key(gid[gid >= 0], pred["frame"][gid >= 0])))
# track length of the pred trajectory holding the match
tl = {}
tj = pred["trajid"]
ln = np.bincount(np.unique(tj, return_inverse=True)[1])
inv = np.unique(tj, return_inverse=True)[1]
plen = ln[inv]
kk = key(gid, pred["frame"])
mp = {k: l for k, l, g in zip(kk, plen, gid) if g >= 0}
tlen = np.array([mp.get(k, 0) for k in key(tid, fr)])
# clean cams
share = np.zeros(bt.shape, int)
for c in range(bt.shape[1]):
    k2 = fr.astype(np.int64) * 10_000_000 + np.where(bt[:, c] >= 0, bt[:, c], -1 - np.arange(len(fr)))
    u, inv2, cnt = np.unique(k2, return_inverse=True, return_counts=True)
    share[:, c] = cnt[inv2]
has = bt >= 0
ncam = has.sum(1)
nclean = (has & (share == 1)).sum(1)
print("truth points", len(tid), "found", have.mean().round(3), " found in track>=10:", (tlen >= 10).mean().round(3))
print("cams with blob: ", {k: round(float((ncam == k).mean()), 3) for k in range(5)})
print("category: share of truth | found | found in track>=10")
for name, m in [("<2 blob cams (unreachable)", ncam < 2),
                (">=2 clean cams", nclean >= 2), ("1 clean cam", (nclean == 1) & (ncam >= 2)),
                ("0 clean cams (all merged)", (nclean == 0) & (ncam >= 2))]:
    print(f"{name:30s} {m.mean():.3f} | {have[m].mean():.3f} | {(tlen[m] >= 10).mean():.3f}")

# matcher cloud (input to the tracker)
from openptv2.storage import RunStore

st = RunStore(cd / "res" / "run.zarr", mode="r")
F, P = [], []
for f in sorted(st.frames()):
    p = np.asarray(st.read_correspondences(f)[0], float)
    F.append(np.full(len(p), f))
    P.append(p)
F = np.concatenate(F)
P = np.concatenate(P)
g, _, off2 = sb._match(by, F, P, 0.6)
b2 = np.median(off2[g >= 0], axis=0)
g, _, _ = sb._match(by, F, P - b2, 0.4)
inp = np.isin(key(tid, fr), np.unique(key(g[g >= 0], F[g >= 0])))
print("matcher cloud: points", len(F), "per frame", len(F)//len(np.unique(F)), "truth recall", inp.mean().round(3), "ghost share", (g < 0).mean().round(3))
for name, m in [(">=2 clean", nclean >= 2), ("  4 clean", nclean == 4), ("  3 clean", nclean == 3), ("  2 clean", nclean == 2),
                ("1 clean", (nclean == 1) & (ncam >= 2)), ("0 clean", (nclean == 0) & (ncam >= 2))]:
    print(f"{name:12s} share {m.mean():.3f} | in matcher cloud {inp[m].mean():.3f} | final output {have[m].mean():.3f} | final in track>=10 {(tlen[m]>=10).mean():.3f}")
