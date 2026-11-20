"""Precision of H3 births: share of births within 0.4 mm of a true particle that the old cloud lacked.
uv run python scripts/joint_birth_quality.py <case> [suffix]
"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
import synth_bench as sb
from scipy.spatial import cKDTree

from openptv2.storage import RunStore

case = sys.argv[1]
suffix = sys.argv[2] if len(sys.argv) > 2 else ""
base = RunStore(sb.WORK / "cases" / case / "res" / "run.zarr", mode="r")
aug = RunStore(sb.WORK / "cases" / f"{case}_H3{suffix}" / "res" / "run.zarr", mode="r")
tr = np.load(sb.WORK / "cases" / case / "truth.npz")
n_out = 200
offs, bp, bf, bi = [], [], [], []
for f in range(1, n_out + 1):
    p0, _ = base.read_correspondences(f)
    p1, i1 = aug.read_correspondences(f)
    bp.append(p1[len(p0) :])
    bf.append(np.full(len(p1) - len(p0), f))
    bi.append(np.asarray(i1)[len(p0) :, :4])
    offs.append(p1[: len(p0)] - 0)
bp, bf, bi = np.vstack(bp), np.concatenate(bf), np.vstack(bi)
# bias from old points
d0 = []
for f in range(1, n_out + 1, 10):
    p0, _ = base.read_correspondences(f)
    t = tr["pos_mm"][tr["frame"] == f]
    d, j = cKDTree(t).query(p0)
    d0.append((p0 - t[j])[d < 0.6])
bias = np.median(np.vstack(d0), axis=0)
ok = np.zeros(len(bp), bool)
dist = np.full(len(bp), np.nan)
new = np.zeros(len(bp), bool)
for f in range(1, n_out + 1):
    r = np.flatnonzero(bf == f)
    if not len(r):
        continue
    t = np.flatnonzero(tr["frame"] == f)
    d, j = cKDTree(tr["pos_mm"][t]).query(bp[r] - bias)
    ok[r] = d < 0.4
    dist[r] = d
    # is the matched truth particle already represented by an old point?
    p0, _ = base.read_correspondences(f)
    d2, _ = cKDTree(p0 - bias).query(tr["pos_mm"][t][j])
    new[r] = ok[r] & (d2 > 0.4)
print(
    "births:",
    len(bp),
    " within 0.4 mm of a true particle: %.3f ; of a particle missing from the old cloud: %.3f"
    % (ok.mean(), new.mean()),
)
print(
    "  distance to nearest truth: median %.3f mm, 90%% %.3f"
    % (np.median(dist), np.percentile(dist, 90))
)
print("  cameras per birth:", np.bincount((bi >= 0).sum(1)))
