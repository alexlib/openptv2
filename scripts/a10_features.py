"""Per-trajectory ghost features (ray convergence rcm, 4-camera share, jitter, speed) on a
synthetic bench case, with AUCs for ghost-only vs real trajectories.

    uv run python scripts/a10_features.py CASE [TRACKER]      # default two_phase

Also writes post-filter variants (a10_rcm_L*_t*) as runs for `synth_bench.py eval`.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import WORK, _experiment, _match

from openptv2.algorithms.orientation import COORD_UNUSED, multi_cam_point_positions
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
from openptv2.storage import RunStore

case = sys.argv[1]
trk = sys.argv[2] if len(sys.argv) > 2 else "two_phase"
cd = WORK / "cases" / case
info = json.loads((cd / "case.json").read_text())
n_out = info["n_out"]
exp = _experiment(cd, cd / "parameters_Run1.yaml", 1, n_out)
cpar, cals = exp.cpar, exp.cals
nC = len(cals)
s = RunStore.open(cd, mode="r")
corr, rcm = {}, {}
for f in range(1, n_out + 1):
    q, c = s.read_correspondences(f)
    corr[f] = (q, c)
    T = np.full((len(q), nC, 2), COORD_UNUSED)
    for cam in range(nC):
        tg = np.asarray(s.root[f"targets/cam_{cam}/frame_{f:06d}"])
        v = c[:, cam] >= 0
        for i in np.flatnonzero(v):
            mx, my = pixel_to_metric(tg[c[i, cam], 1], tg[c[i, cam], 2], cpar)
            cl = cals[cam]
            a = cl.added_par
            T[i, cam] = dist_to_flat(
                mx,
                my,
                cl.int_par.xh,
                cl.int_par.yh,
                a.k1,
                a.k2,
                a.k3,
                a.p1,
                a.p2,
                a.scx,
                a.she,
            )
    _, r = multi_cam_point_positions(T, cpar, cals)
    rcm[f] = np.asarray(r)
tr = np.load(cd / "truth.npz")
bf = {}
for f in np.unique(tr["frame"]):
    m = tr["frame"] == f
    bf[int(f)] = (tr["pos_mm"][m], tr["particle_id"][m], cKDTree(tr["pos_mm"][m]))
p = np.load(WORK / "runs" / case / trk / "pred.npz")
tid, fr, pos = p["trajid"], p["frame"], p["pos"]
nc = np.zeros(len(fr), int)
rc = np.zeros(len(fr))
for f in np.unique(fr):
    m = fr == f
    q, c = corr[int(f)]
    d, j = cKDTree(q).query(pos[m])
    ok = d < 1e-6
    nc[m] = np.where(ok, (c[j] >= 0).sum(1), 0)
    rc[m] = np.where(ok, rcm[int(f)][j], np.nan)
g0, _, off = _match(bf, fr, pos, 0.6)
bias = np.median(off[g0 >= 0], axis=0)
gid, _, _ = _match(bf, fr, pos - bias, 0.4)
print("point-level rcm median real / ghost, by cams:")
for k in (3, 4):
    m = nc == k
    print(
        f"  {k}-cam: real {np.nanmedian(rc[m & (gid >= 0)]):.4f}  ghost {np.nanmedian(rc[m & (gid < 0)]):.4f}  (n {np.sum(m & (gid >= 0))}/{np.sum(m & (gid < 0))})"
    )
o = np.lexsort((fr, tid))
tid, fr, pos, nc, rc, gid = tid[o], fr[o], pos[o], nc[o], rc[o], gid[o]
cuts = np.flatnonzero(np.diff(tid)) + 1
rows = []
for idx in np.split(np.arange(len(tid)), cuts):
    L = len(idx)
    if L < 6:
        continue
    x = pos[idx]
    d2 = np.diff(x, 2, axis=0)
    rows.append(
        [
            L,
            np.mean(nc[idx] == 4),
            np.nanmean(rc[idx]),
            np.nanmedian(rc[idx]),
            np.nanpercentile(rc[idx], 90),
            np.sqrt((d2[:, 2] ** 2).mean()),
            np.linalg.norm(np.diff(x, axis=0), axis=1).mean(),
            (gid[idx] < 0).all(),
            (gid[idx] >= 0).all(),
            idx[0],
        ]
    )
R = np.array(rows, float)
np.save(f"/private/tmp/claude-501/a10_{case}.npy", R)
names = ["len", "share4", "rcm_mean", "rcm_med", "rcm_p90", "z_d2", "speed"]
ghost = R[:, 7] == 1
real = R[:, 8] == 1


def auc(x, pos_m, neg_m):
    a = x[pos_m]
    b = x[neg_m]
    r = np.argsort(np.argsort(np.r_[a, b])) + 1
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


print(
    f"trajs len>=6: real {real.sum()}, ghost-only {ghost.sum()}   (AUC: P(feature higher for ghost than real))"
)
for i, n in enumerate(names):
    print(
        f"  {n:9s} real p10/50/90 {np.round(np.nanpercentile(R[real, i], [10, 50, 90]), 4)}  ghost {np.round(np.nanpercentile(R[ghost, i], [10, 50, 90]), 4)}  AUC {auc(R[:, i], ghost, real):.3f}"
    )

# ---- rule: drop trajectories (len>=minlen) whose mean rcm > t
ghostpt = gid < 0
for minlen, t in [(6, 0.060), (6, 0.050), (6, 0.045), (11, 0.050), (3, 0.050)]:
    drop = np.zeros(len(tid), bool)
    nr = ng = 0
    for idx in np.split(np.arange(len(tid)), cuts):
        if len(idx) >= minlen and np.nanmean(rc[idx]) > t:
            drop[idx] = True
            if (gid[idx] >= 0).all():
                nr += 1
            elif (gid[idx] < 0).all():
                ng += 1
    label = f"a10_rcm_L{minlen}_t{t}"
    run = WORK / "runs" / case / label
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "pred.npz", trajid=tid[~drop], frame=fr[~drop], pos=pos[~drop]
    )
    (run / "run.json").write_text(
        json.dumps({"tracker": label, "case": case, "time_s": 0, "status": "ok"})
    )
    print(
        f"{label}: ghost pts removed {np.sum(drop & ghostpt) / ghostpt.sum():.0%}, real pts removed {np.sum(drop & ~ghostpt) / (~ghostpt).sum():.1%}, real trajs {nr}, ghost-only trajs {ng}"
    )
