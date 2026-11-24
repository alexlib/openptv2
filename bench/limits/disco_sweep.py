# ruff: noqa
"""What does the detection's discontinuity value (targ_rec.disco) do, against truth? (rendered case syn1)

    uv run python bench/limits/disco_sweep.py RUN_DIR TRUTH_NPZ [--frames 3 60] [--values 0 2 5 10 20 50 100 255]

Per true particle in a camera view: own blob (the nearest blob holds only it, within 1 px), merged (the blob holds >= 2 true
particles), over-split (>= 2 blobs within 2 px of it), missed (no blob within 2.5 px). Blobs with no true particle are
fragments/noise. Plus the centroid error of the own blobs.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("truth")
ap.add_argument("--frames", type=int, nargs="+", default=[3, 60])
ap.add_argument("--values", type=int, nargs="+", default=[0, 2, 5, 10, 20, 50, 100, 255])
a = ap.parse_args()
run = Path(a.run_dir).resolve(); os.chdir(run)
from openptv2.gui.parameter_manager import ParameterManager
from openptv2.gui.ptv import DEFAULT_HIGHPASS_FILTER_SIZE, py_start_proc_c, read_frame_images, simple_highpass
from openptv2.segmentation import target_recognition

t = np.load(a.truth)
pm = ParameterManager(); pm.from_yaml(run / "parameters_Run1.yaml")
cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
ncam = len(cals); base = pm.get_parameter("sequence")["base_name"]; ptv = pm.get_parameter("ptv") or {}
hps, trs = [], []
for f in a.frames:
    raw = read_frame_images(pm, base, ncam, f)
    pxy = t["pixel_xy_per_cam"][t["frame"] == f]
    for c in range(ncam):
        hps.append((f, c, simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))))
        xy = pxy[:, 2 * c: 2 * c + 2]; trs.append(xy[np.isfinite(xy).all(1)])
print(f"{'disco':>6} {'blobs':>7} {'own%':>6} {'merged%':>8} {'split%':>7} {'missed%':>8} {'fragm%':>7} {'own_err_med':>11} {'own_err_p90':>11}")
for v in a.values:
    tpar.set_max_discontinuity(v)
    n_t = n_own = n_mer = n_spl = n_mis = n_blob = n_frag = 0
    errs = []
    for (f, c, hp), tr in zip(hps, trs):
        ts = target_recognition(hp, tpar, c, cpar)
        det = np.array([[x.pos()[0], x.pos()[1]] for x in ts]) if len(ts) else np.zeros((0, 2))
        n_blob += len(det); n_t += len(tr)
        if len(det) == 0:
            n_mis += len(tr); continue
        bt, tt = cKDTree(det), cKDTree(tr)
        d, ib = bt.query(tr)
        cnt = np.array([len(tt.query_ball_point(p, 2.5)) for p in det])
        n_frag += int((cnt == 0).sum())
        nb2 = np.array([len(bt.query_ball_point(p, 2.0)) for p in tr])
        for k in range(len(tr)):
            if d[k] > 2.5: n_mis += 1; continue
            if nb2[k] >= 2: n_spl += 1
            if cnt[ib[k]] >= 2: n_mer += 1
            elif d[k] < 1.0 and nb2[k] == 1: n_own += 1; errs.append(d[k])
    e = np.array(errs) if errs else np.array([np.nan])
    print(f"{v:6d} {n_blob:7d} {100*n_own/n_t:6.1f} {100*n_mer/n_t:8.1f} {100*n_spl/n_t:7.1f} {100*n_mis/n_t:8.1f} {100*n_frag/max(n_blob,1):7.1f} {np.median(e):11.3f} {np.percentile(e,90):11.3f}")
