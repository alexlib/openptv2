# ruff: noqa
"""Why does self-calibration lose matches on lv_multi? Parameter-group sweep on a detected+matched run (redetect_selfcal.py output).

    uv run python bench/calibration_lab/lv_why.py <run dir (parameters_Run1.yaml, res/run.zarr)> <first> <last> [label]

For each variant of the fit (which parameter groups are free, priors, anchor) the cameras are refitted on the clean 4-camera points of
the odd frames and the matcher is run again on ALL frames with them (nothing is written): 4-camera matches against the old cameras,
and how far each group of parameters moved.
"""
import copy, sys, time
from pathlib import Path

import numpy as np

from openptv2.calibration_consistency import _triangulate, collect_clean_points, inlier_mask, matching_yield
from openptv2.recorrespond import load_parameters
from openptv2.selfcal import joint_self_calibrate, known_body_from_calibration
from openptv2.storage import RunStore

run, f0, f1 = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else run.name
frames = list(range(f0, f1 + 1))
cpar, vpar, cals, _ = load_parameters(run / "parameters_Run1.yaml"); cals = list(cals)
st = RunStore(str(run / "res" / "run.zarr"), mode="r")
d = collect_clean_points(st, [f for f in frames if f % 2 == 1], 4)
ctr = d["centre"]; x0 = _triangulate(ctr, cals, cpar, vpar, [0, 1, 2, 3]); ok = np.isfinite(x0[:, 0])
keep, thr = inlier_mask(ctr[ok], x0[ok], cals, cpar); idx = np.flatnonzero(ok)[keep]
ctr, x0 = ctr[idx], x0[idx]
print(f"[{label}] {len(x0)} fit points (inlier threshold {thr:.2f} px); old cameras: matching all {len(frames)} frames ...", flush=True)
q0, t0 = matching_yield(st, frames, cals, cpar, vpar)
print(f"  old cameras: {q0} 4-camera matches of {t0}")
body = None
try:
    body = known_body_from_calibration(run, cals, cpar)
except Exception as e:
    print("  (no anchor:", e, ")")


def moved(a, b):
    out = dict(pos=0.0, ang=0.0, pp=0.0, cc=0.0, k=0.0, p=0.0)
    for x, y in zip(a, b):
        out["pos"] = max(out["pos"], float(np.linalg.norm(np.asarray(x.get_pos()) - np.asarray(y.get_pos()))))
        out["ang"] = max(out["ang"], float(np.max(np.abs(np.asarray(x.get_angles()) - np.asarray(y.get_angles()))) * 1e3))
        out["pp"] = max(out["pp"], float(np.hypot(x.int_par.xh - y.int_par.xh, x.int_par.yh - y.int_par.yh)))
        out["cc"] = max(out["cc"], abs(x.int_par.cc - y.int_par.cc))
        out["k"] = max(out["k"], abs(x.added_par.k1 - y.added_par.k1) * 1e3, abs(x.added_par.k2 - y.added_par.k2) * 1e3)
        out["p"] = max(out["p"], abs(x.added_par.p1 - y.added_par.p1) * 1e3, abs(x.added_par.p2 - y.added_par.p2) * 1e3)
    return out


V = {
    "ext only": dict(free_pp=False, free_radial=(), free_decentering=False),
    "ext + pp": dict(free_pp=True, free_radial=(), free_decentering=False),
    "ext + k1k2": dict(free_pp=False, free_radial=("k1", "k2"), free_decentering=False),
    "ext + p1p2": dict(free_pp=False, free_radial=(), free_decentering=True),
    "ext + pp + k1k2": dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=False),
    "all (default)": dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True),
    "all + focal": dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True, free_cc=True),
    "all, tight priors": dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True, ext_sigma_scale=0.3, pp_sigma_mm=0.01, k_sigma=0.005, p_sigma=0.0005),
}
import os
if os.environ.get("EDGE") == "1":
    V = {"all, loose (old default)": V["all (default)"]}
    for e in (0.25, 0.5, 1.0, 2.0, 5.0, 20.0):
        V[f"all, edge_px {e:g}"] = dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True, edge_px=e)
    V["ext + pp + p1p2, edge_px 1"] = dict(free_pp=True, free_radial=(), free_decentering=True, edge_px=1.0)
elif body is not None:
    V["all + anchor"] = dict(free_pp=True, free_radial=("k1", "k2"), free_decentering=True, body=body)
print(f"  {'variant':20} {'fit rms px':>12} {'matches vs old':>15} {'pos mm':>7} {'ang mrad':>9} {'pp mm':>7} {'cc':>6} {'k*1e3':>7} {'p*1e3':>7}", flush=True)
for name, kw in V.items():
    kw = dict(kw)
    r = joint_self_calibrate(ctr, x0, cals, cpar, max_nfev=200, ext_sigma_scale=kw.pop("ext_sigma_scale", 3.0), vpar=vpar, **kw)
    q, t = matching_yield(st, frames, r.cals, cpar, vpar)
    m = moved(cals, r.cals)
    print(f"  {name:20} {r.rms_before_px:5.3f}->{r.rms_after_px:5.3f} {100 * q / q0:14.1f}% {m['pos']:7.2f} {m['ang']:9.2f} {m['pp']:7.3f} {m['cc']:6.2f} {m['k']:7.3f} {m['p']:7.3f}", flush=True)
