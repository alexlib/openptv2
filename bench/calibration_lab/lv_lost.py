# ruff: noqa
"""Where do the matches that self-calibration loses lie? Radius on the sensor, per camera, old vs new cameras.

    uv run python bench/calibration_lab/lv_lost.py <run dir> <first> <last> [variant]     variant: default | ext | p1p2
"""
import sys
from pathlib import Path

import numpy as np

from openptv2.calibration_consistency import _triangulate, collect_clean_points, inlier_mask
from openptv2.correspondences import match_correspondences_batch_parallel
from openptv2.recorrespond import load_parameters
from openptv2.selfcal import joint_self_calibrate
from openptv2.storage import RunStore

run, f0, f1 = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
variant = sys.argv[4] if len(sys.argv) > 4 else "default"
frames = list(range(f0, f1 + 1))
cpar, vpar, cals, _ = load_parameters(run / "parameters_Run1.yaml"); cals = list(cals)
st = RunStore(str(run / "res" / "run.zarr"), mode="r")
d = collect_clean_points(st, [f for f in frames if f % 2 == 1], 4)
ctr = d["centre"]; x0 = _triangulate(ctr, cals, cpar, vpar, [0, 1, 2, 3]); ok = np.isfinite(x0[:, 0])
keep, _ = inlier_mask(ctr[ok], x0[ok], cals, cpar); idx = np.flatnonzero(ok)[keep]
kw = dict(default=dict(), ext=dict(free_pp=False, free_radial=(), free_decentering=False), p1p2=dict(free_pp=False, free_radial=(), free_decentering=True))[variant]
new = joint_self_calibrate(ctr[idx], x0[idx], cals, cpar, max_nfev=200, ext_sigma_scale=3.0, vpar=vpar, **kw).cals


def quads(cs):
    res = match_correspondences_batch_parallel(frames=frames, cpar=cpar, cals=cs, vpar=vpar, zarr_store_path=str(run / "res" / "run.zarr"), n_workers=1, write_to_store=False)
    out = {}
    for f, (pos, ids) in res.items():
        ids = np.asarray(ids)[:, :4]
        for row in ids[(ids >= 0).all(axis=1)]:
            out[(f,) + tuple(int(v) for v in row)] = None
    return set(out)


A, B = quads(cals), quads(new)
lost, gained, both = A - B, B - A, A & B
print(f"[{variant}] 4-camera matches: old {len(A)}  new {len(B)}  kept {len(both)}  lost {len(lost)}  gained {len(gained)}  (lost {100 * len(lost) / len(A):.1f} %, gained {100 * len(gained) / len(A):.1f} %)")
tg = {(f, c): np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for f in frames for c in range(4)}
cx, cy = cpar.imx / 2, cpar.imy / 2


def radii(keys):
    r = [[] for _ in range(4)]
    for k in keys:
        f = k[0]
        for c in range(4):
            x, y = tg[f, c][k[1 + c], 1:3]
            r[c].append(np.hypot((x - cx) * cpar.pix_x, (y - cy) * cpar.pix_y))
    return [np.array(v) for v in r]


bins = [0, 3, 5, 6.5, 8, 10]
print("  fraction of matches by radius on the sensor (mm), averaged over the 4 cameras; share of the old matches that were lost in the bin")
ra, rl, rg = radii(list(A)), radii(list(lost)), radii(list(gained))
print(f"  {'radius mm':>10} {'old':>7} {'lost':>7} {'gained':>7} {'lost / old in bin':>18}")
for lo, hi in zip(bins[:-1], bins[1:]):
    na = np.mean([np.mean((r >= lo) & (r < hi)) for r in ra]); nl = np.mean([np.mean((r >= lo) & (r < hi)) for r in rl]) if len(lost) else 0
    ng = np.mean([np.mean((r >= lo) & (r < hi)) for r in rg]) if len(gained) else 0
    print(f"  {lo:>4.1f}-{hi:<5.1f} {100 * na:6.1f}% {100 * nl:6.1f}% {100 * ng:6.1f}% {100 * (nl * len(lost)) / max(na * len(A), 1):17.1f}%")
print("  largest radius of any camera in the quad: median old %.2f lost %.2f" % (np.median([max(r[i] for r in ra) for i in range(len(A))]), np.median([max(r[i] for r in rl) for i in range(len(lost))]) if len(lost) else float("nan")))
