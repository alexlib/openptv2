"""Appearance cue: grey sum / pixel count of a blob relative to the particle's own typical value (rendered Synthetic-CompleteTest frames, openptv2 detections, truth from the render). docs/plans/2026-11-19-joint-tracker.md section 15.2."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from openptv2.storage import RunStore

SYN = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"
st = RunStore.open(SYN / "syn1" / "test", mode="r")
tr = np.load(SYN / "truth/truth.npz")
obs = []                                   # (frame, cam, pid, blob row values, n_members)
for f in range(100, 141):
    m = np.flatnonzero(tr["frame"] == f)
    for cam in range(4):
        rows = np.asarray(st.root[f"targets/cam_{cam}/frame_{f:06d}"])
        px = tr["pixel_xy_per_cam"][m][:, 2 * cam: 2 * cam + 2]
        ok = ~np.isnan(px[:, 0]); pid = tr["particle_id"][m][ok]; px = px[ok]
        d, j = cKDTree(rows[:, 1:3]).query(px, distance_upper_bound=3.0)
        has = np.isfinite(d)
        cnt = np.bincount(j[has], minlength=len(rows))
        for p, jj in zip(pid[has], j[has]):
            obs.append((f, cam, p, rows[jj, 3], rows[jj, 6], cnt[jj]))
obs = np.array(obs)
f_, c_, p_, n_, s_, k_ = obs.T
# typical single-particle grey sum / pixel count per (particle, camera): median over frames where it is alone in its blob
typ = defaultdict(list)
for i in np.flatnonzero(k_ == 1):
    typ[(int(p_[i]), int(c_[i]))].append((n_[i], s_[i]))
tm = {k: np.median(np.array(v), axis=0) for k, v in typ.items() if len(v) >= 3}
sel = np.array([(int(p_[i]), int(c_[i])) in tm for i in range(len(p_))])
y = (k_[sel] >= 2).astype(int)
n_rel = np.array([n_[i] / tm[(int(p_[i]), int(c_[i]))][0] for i in np.flatnonzero(sel)])
s_rel = np.array([s_[i] / tm[(int(p_[i]), int(c_[i]))][1] for i in np.flatnonzero(sel)])
def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print("observations with a known typical appearance:", len(y), " merged share %.3f" % y.mean())
print("alone (no context):   AUC by n %.3f, by sum grey %.3f" % (auc(n_[sel], y), auc(s_[sel], y)))
print("with the particle's own typical appearance (from other frames): AUC by n/typical %.3f, by sumgrey/typical %.3f" % (auc(n_rel, y), auc(s_rel, y)))
for thr in (1.25, 1.5, 2.0):
    pr = s_rel > thr
    print("  sumgrey > %.2f x typical: finds %.0f %% of merged, %.0f %% of alarms right, false alarms on %.1f %% of single blobs" % (thr, 100 * (pr & (y == 1)).sum() / y.sum(), 100 * (pr & (y == 1)).sum() / max(pr.sum(), 1), 100 * (pr & (y == 0)).sum() / (y == 0).sum()))
print("  spread of sum grey / typical for single blobs: median %.2f, 5-95%% %.2f-%.2f" % (np.median(s_rel[y == 0]), *np.percentile(s_rel[y == 0], [5, 95])))
print("  for merged blobs: median %.2f, 5-95%% %.2f-%.2f" % (np.median(s_rel[y == 1]), *np.percentile(s_rel[y == 1], [5, 95])))
