# ruff: noqa
"""Where does the tracker's remaining error on syn1 come from? Per-point error against the truth triangulated with the
processing calibration (calibration offset removed), by cameras used and by shared blob. Needs a tracked scratch run.

    uv run python bench/limits/error_sources.py <work>/corr <n_frames>
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb

run, n = Path(sys.argv[1]), int(sys.argv[2])
exp = sb._experiment(run, run / "parameters_Run1.yaml", 1, n)
os.chdir(run)
from openptv2.loco import pixels_to_flat
from openptv2.merge_post import arrays_from_store
from openptv2.merge_refine import shared_blobs
from openptv2.orientation import point_positions
from openptv2.storage import RunStore

cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
z = np.load(sb.SRC / "truth" / "truth.npz"); k = z["frame"] <= n
fr, px = z["frame"][k], z["pixel_xy_per_cam"][k].reshape(-1, 4, 2)
flat = np.full((len(px), 4, 2), -1e10)
for c in range(4):
    v = np.isfinite(px[:, c, 0]); flat[v, c] = pixels_to_flat(px[v, c], cals[c], cpar)
xt = np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)
a = arrays_from_store(RunStore(str(run / "res" / "run.zarr"), mode="r"), "ptv_is", 1, n, 4)
frame, pos, blob = a["frame"], np.asarray(a["pos"], float), a["blob"]
share = shared_blobs(frame, blob)
err, ncam, touch = [], [], []
for f in range(1, n + 1):
    m = (fr == f) & np.isfinite(xt[:, 0]); tree = cKDTree(xt[m])
    rr = np.flatnonzero(frame == f); d, j = tree.query(pos[rr])
    ok = d < 0.5
    err += list(d[ok]); ncam += list((blob[rr[ok]] >= 0).sum(1)); touch += list((share[rr[ok]] >= 2).any(1))
err, ncam, touch = np.array(err), np.array(ncam), np.array(touch)
print(f"points matched {len(err)}; error vs calibrated truth (mm): median {np.median(err):.4f} rms {np.sqrt(np.mean(err**2)):.4f} p90 {np.percentile(err,90):.4f}")
tot = np.sum(err ** 2)
for nc in (2, 3, 4):
    for t in (False, True):
        s = (ncam == nc) & (touch == t)
        if s.any():
            print(f"  cams {nc} {'shared  ' if t else 'unshared'} share of points {s.mean():5.3f}  median {np.median(err[s]):.4f}  rms {np.sqrt(np.mean(err[s]**2)):.4f}  share of squared error {np.sum(err[s]**2)/tot:5.3f}")
