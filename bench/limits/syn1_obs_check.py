# ruff: noqa
"""Observation-level check of the pixel fit inside refine_merged on syn1: for shared views, the pixel position the refine
step uses (centre split vs pixel fit) against the truth projection. Needs a tracked scratch run (syn1_pixel_fit.py base).

    uv run python bench/limits/syn1_obs_check.py <work>/base <n_frames>
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb

run, n_frames = Path(sys.argv[1]), int(sys.argv[2])
exp = sb._experiment(run, run / "parameters_Run1.yaml", 1, n_frames)
os.chdir(run)
from openptv2.holistic import pixel_fitter_for
from openptv2.merge_post import arrays_from_store, estimate_signatures, extend_membership
from openptv2.merge_refine import refine_merged, shared_blobs
from openptv2.storage import RunStore

st = RunStore(str(run / "res" / "run.zarr"), mode="r")
cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
arr = arrays_from_store(st, "ptv_is", 1, n_frames, 4)
blob, centre, sumg, _ = extend_membership(arr["frame"], arr["pos"], arr["blob"], arr["centre"], arr["sumg"], arr["blobs"], cpar, cals, radius=2.0)
sig = estimate_signatures(arr["tid"], arr["frame"], blob, sumg)
fitter = pixel_fitter_for(exp)
fitter.min_prior_px = float(os.environ.get('MIN_PRIOR', 0.15)); fitter.var_scale = float(os.environ.get('VAR_SCALE', 2.5))
kw = dict(signatures=sig, rounds=3, x_init=arr["pos"], shared_only=True, partner_min_len=10)
r0 = refine_merged(arr["frame"], arr["tid"], blob, centre, sumg, cpar, cals, vpar, **kw)
r1 = refine_merged(arr["frame"], arr["tid"], blob, centre, sumg, cpar, cals, vpar, pixel_fit=fitter, **kw)
print("refine info split:", r0.info[-1]); print("refine info fit  :", r1.info[-1])

z = np.load(sb.SRC / "truth" / "truth.npz")
frame, pos = arr["frame"], np.asarray(arr["pos"], float)
share = shared_blobs(frame, blob)
tpx = z["pixel_xy_per_cam"]
rows = {c: [] for c in ("raw", "split", "fit")}
sel_all = []
for f in range(1, n_frames + 1):
    m = z["frame"] == f
    tp, tpix = z["pos_mm"][m], tpx[m]
    rr = np.flatnonzero(frame == f)
    d, j = cKDTree(tp).query(pos[rr])
    for k, (r, dd, jj) in enumerate(zip(rr, d, j)):
        if dd > 0.3:
            continue
        for c in range(4):
            if share[r, c] >= 2 and np.isfinite(tpix[jj, 2 * c]) and np.isfinite(r0.obs_px[r, c, 0]) and np.isfinite(r1.obs_px[r, c, 0]):
                t = tpix[jj, 2 * c: 2 * c + 2]
                rows["raw"].append(np.hypot(*(centre[r, c] - t)))
                rows["split"].append(np.hypot(*(r0.obs_px[r, c] - t)))
                rows["fit"].append(np.hypot(*(r1.obs_px[r, c] - t)))
                sel_all.append((r1.obs_var[r, c], r0.obs_var[r, c], r1.obs_px[r, c, 0] != r0.obs_px[r, c, 0]))
print(f"shared views matched to truth: {len(rows['raw'])}")
for k, v in rows.items():
    v = np.array(v); print(f"  {k:6s} median {np.median(v):.3f}  mean {v.mean():.3f}  p90 {np.percentile(v,90):.3f}  >1px {np.mean(v>1):.3f}")
s = np.array(sel_all, float)
print(f"obs variance (px^2): split median {np.median(s[:,1]):.4f}  fit median {np.median(s[:,0]):.4f};  share of views changed by the fit {s[:,2].mean():.3f}")
print("fitter stats:", {k: (v if k != 'noise' else round(float(np.median(v)), 2)) for k, v in fitter.stats.items()})

# 3D position error of the points that touch a shared blob
touch = (share >= 2).any(1)
errs = {"stored": [], "split": [], "fit": []}
for f in range(1, n_frames + 1):
    m = z["frame"] == f
    tp = z["pos_mm"][m]
    rr = np.flatnonzero((frame == f) & touch)
    d, j = cKDTree(tp).query(pos[rr])
    for r, dd, jj in zip(rr, d, j):
        if dd < 0.3 and np.isfinite(r0.x[r, 0]) and np.isfinite(r1.x[r, 0]):
            errs["stored"].append(np.linalg.norm(pos[r] - tp[jj])); errs["split"].append(np.linalg.norm(r0.x[r] - tp[jj])); errs["fit"].append(np.linalg.norm(r1.x[r] - tp[jj]))
print(f"points touching a shared blob, 3D error vs truth (mm), n={len(errs['stored'])}")
for k, v in errs.items():
    v = np.array(v); print(f"  {k:7s} median {np.median(v):.4f}  rms {np.sqrt(np.mean(v**2)):.4f}  p90 {np.percentile(v,90):.4f}")
