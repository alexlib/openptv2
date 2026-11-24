# ruff: noqa
"""The error that calibration alone puts into syn1 (no detection, no merging, no linking).

Triangulate the truth pixel positions (rendered with a perturbed calibration) with the processing calibration, with
perfect identities, and score with the benchmark's evaluator. Control: project the truth 3D through the processing
calibration instead (must score ~0).

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/calib_floor.py <work>/base <work>/truth_case <n_frames>
"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb

run, cd, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
exp = sb._experiment(run, run / "parameters_Run1.yaml", 1, n)
from openptv2.loco import pixels_to_flat, project_pixels
from openptv2.orientation import point_positions

cpar, cals, vpar = exp.cpar, list(exp.cals), exp.vpar
z = np.load(sb.SRC / "truth" / "truth.npz")
keep = z["frame"] <= n
pid, fr, pos, px = z["particle_id"][keep], z["frame"][keep], z["pos_mm"][keep], z["pixel_xy_per_cam"][keep]
rng = np.random.default_rng(0)

def triangulate(pxs):
    flat = np.full((len(pxs), 4, 2), -1e10)
    for c in range(4):
        v = np.isfinite(pxs[:, c, 0])
        if v.any():
            flat[v, c] = pixels_to_flat(pxs[v, c], cals[c], cpar)
    return np.asarray(point_positions(flat, cpar, cals, vpar, method="lsq")[0], float)

flow = sb.Flow()
for name, pxs in (
    ("truth pixels (render calibration), exact", px.reshape(-1, 4, 2)),
    ("truth pixels + 0.05 px noise", px.reshape(-1, 4, 2) + rng.normal(0, 0.05, (len(px), 4, 2))),
    ("control: truth 3D through the processing calibration", np.stack([project_pixels(pos, c, cpar) for c in cals], axis=1)),
):
    ok = np.isfinite(pxs[:, :, 0]).sum(1) >= 2
    x = triangulate(pxs[ok])
    good = np.isfinite(x[:, 0])
    pred = {"trajid": pid[ok][good], "frame": fr[ok][good], "pos": x[good]}
    ev = sb.evaluate_pred(cd, pred, flow)
    d = np.linalg.norm(x[good] - pos[ok][good], axis=1)
    print(f"{name:55s} vel41 {ev['by_window'][41]['vel']:.4f}  acc41 {ev['by_window'][41]['acc']:.2f}  3D err median {np.median(d):.4f} mm  (n={good.sum()})", flush=True)
