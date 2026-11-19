"""Shape cue: intensity-weighted second moments of a blob's pixels (high-passed render), static, with context and over lags of 1/3/6 frames, single vs merged. docs/plans/2026-11-19-joint-tracker.md section 15.2."""
import os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
SYN = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"
RUN = SYN / "syn1" / "test"
os.chdir(RUN)
from openptv2.gui.parameter_manager import ParameterManager
from openptv2.gui.ptv import (DEFAULT_HIGHPASS_FILTER_SIZE, py_start_proc_c, read_frame_images, simple_highpass)
from openptv2.storage import RunStore
pm = ParameterManager(); pm.from_yaml(RUN / "parameters_Run1.yaml")
cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
base = pm.get_parameter("sequence")["base_name"]; ptv = pm.get_parameter("ptv") or {}
st = RunStore.open(RUN, mode="r"); tr = np.load(SYN / "truth/truth.npz")
F0, F1 = 100, 130
obs = {}
def moments(a, x, y, nx, ny):
    h = int(max(nx, ny) // 2) + 2
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(yi - h, 0), min(yi + h + 1, a.shape[0]); x0, x1 = max(xi - h, 0), min(xi + h + 1, a.shape[1])
    if y1 - y0 < 3 or x1 - x0 < 3: return None
    sub = a[y0:y1, x0:x1].astype(float)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    w = np.clip(sub, 0, None)
    if w.sum() <= 0: return None
    mx, my = (w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()
    cxx = (w * (xs - mx) ** 2).sum() / w.sum(); cyy = (w * (ys - my) ** 2).sum() / w.sum(); cxy = (w * (xs - mx) * (ys - my)).sum() / w.sum()
    ev = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    peak = sub.max() / w.sum()
    return np.array([ev[1], ev[0], ev[1] / max(ev[0], 1e-6), peak, ang])
for f in range(F0, F1 + 1):
    imgs = read_frame_images(pm, base, 4, f)
    m = np.flatnonzero(tr["frame"] == f)
    for cam in range(4):
        hp = simple_highpass(imgs[cam], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
        rows = np.asarray(st.root[f"targets/cam_{cam}/frame_{f:06d}"])
        px = tr["pixel_xy_per_cam"][m][:, 2 * cam: 2 * cam + 2]
        ok = ~np.isnan(px[:, 0]); pid = tr["particle_id"][m][ok]; px = px[ok]
        d, j = cKDTree(rows[:, 1:3]).query(px, distance_upper_bound=3.0)
        has = np.isfinite(d); cnt = np.bincount(j[has], minlength=len(rows))
        cache = {}
        for p, jj in zip(pid[has], j[has]):
            if jj not in cache:
                r = rows[jj]; cache[jj] = moments(hp, r[1], r[2], r[4], r[5])
            if cache[jj] is not None:
                obs[(f, cam, int(p))] = (cache[jj], cnt[jj], rows[jj, 6])
def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1; n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
keys = list(obs); X = np.array([obs[k][0] for k in keys]); cnt = np.array([obs[k][1] for k in keys])
y = (cnt >= 2).astype(int)
names = ["major variance", "minor variance", "elongation", "peak/sum (peakiness)"]
print("STATIC, blobs with >=1 particle: %d obs, merged share %.3f" % (len(y), y.mean()))
for k, nm in enumerate(names):
    print("  AUC merged vs single by %-22s %.3f" % (nm, auc(X[:, k], y)))
# dynamics at several lags: same particle, frames f and f+L, both single vs both merged (merged at both ends)
for L in (1, 3, 6):
    S, M = [], []
    for (f, cam, p), (v, c, sg) in obs.items():
        o2 = obs.get((f + L, cam, p))
        if o2 is None: continue
        dv = np.abs(o2[0][:4] - v[:4]) / np.maximum(np.abs(v[:4]), 1e-9)
        dang = abs(((o2[0][4] - v[4]) + np.pi / 2) % np.pi - np.pi / 2)
        row = np.r_[dv, dang, abs(o2[2] - sg) / max(sg, 1)]
        if c == 1 and o2[1] == 1: S.append(row)
        elif c >= 2 and o2[1] >= 2: M.append(row)
    S, M = np.array(S), np.array(M); yy = np.r_[np.zeros(len(S)), np.ones(len(M))]
    print("LAG %d frames: single-in-both %d, merged-in-both %d; median change single/merged, AUC" % (L, len(S), len(M)))
    for k, nm in enumerate(["major variance", "minor variance", "elongation", "peakiness", "orientation (rad)", "grey sum"]):
        print("   %-18s %.3f / %.3f   AUC %.3f" % (nm, np.median(S[:, k]), np.median(M[:, k]), auc(np.r_[S[:, k], M[:, k]], yy)))
# context: relative to the particle's own typical shape (median over its single-particle frames)
from collections import defaultdict
typ = defaultdict(list)
for (f, cam, p), (v, c, sg) in obs.items():
    if c == 1: typ[(cam, p)].append(v[:4])
tm = {k: np.median(np.array(v), axis=0) for k, v in typ.items() if len(v) >= 3}
idx = [i for i, (f, cam, p) in enumerate(keys) if (cam, p) in tm]
rel = np.array([X[i, :4] / np.maximum(tm[(keys[i][1], keys[i][2])], 1e-9) for i in idx]); yc = y[idx]
print("WITH CONTEXT (shape relative to the particle's own typical shape), %d obs" % len(idx))
for k, nm in enumerate(names):
    print("  AUC by %-22s relative to typical: %.3f" % (nm, auc(rel[:, k], yc)))
sgrel = np.array([obs[keys[i]][2] / 1.0 for i in idx])
