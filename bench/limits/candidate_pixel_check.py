# ruff: noqa
"""Can the raw image tell the ghosts from the real among the points a relaxed second pass adds?  (syn1, truth)

    uv run python bench/limits/candidate_pixel_check.py <base dir> <relaxed dir> <n_frames>

New 3-camera points: label = truth (ghost if no truth particle within 0.4 mm). Cues:
  blob   brightness of the views (min and max/min of the grey sums, per-camera normalised), own views
  pixel  light in the missing camera: peak of the raw image within 1.5 px of the projected point, over the
         local background, in units of the local noise (MAD)
"""
import os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
import synth_bench as sb
import streak_lib as sl
from openptv2.loco import project_pixels
from openptv2.merge_post import arrays_from_store
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore
from openptv2.gui.ptv import read_frame_images

base, rel, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
sb0 = RunStore(str(base / "res" / "run.zarr"), mode="r")
n_base = {f: len(sb0.read_correspondences(f)[0]) for f in range(1, N + 1)}
st = RunStore(str(rel / "res" / "run.zarr"), mode="r")
a = arrays_from_store(st, "ptv_is", 1, N, 4)
tr = np.load(sb.SRC / "truth" / "truth.npz"); k = tr["frame"] <= N
by = {int(f): (tr["pos_mm"][k][tr["frame"][k] == f], tr["particle_id"][k][tr["frame"][k] == f], cKDTree(tr["pos_mm"][k][tr["frame"][k] == f])) for f in np.unique(tr["frame"][k])}
g0, _, off = sb._match(by, a["frame"], a["pos"], 0.6); bias = np.median(off[g0 >= 0], axis=0)
gid, _, _ = sb._match(by, a["frame"], a["pos"] - bias, 0.4)
new = a["row"] >= np.array([n_base[int(f)] for f in a["frame"]])
ncam = (a["blob"] >= 0).sum(1)
cpar, vpar, cals, _ = load_parameters(rel / "parameters_Run1.yaml")
exp = sb._experiment(rel, rel / "parameters_Run1.yaml", 1, N)
pm = exp.pm; bname = pm.get_parameter("sequence")["base_name"]
os.chdir(rel)

def evidence(img, p, r_in=1.5, r_win=7):
    x, y = p
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 - r_win < 0 or y0 - r_win < 0 or x0 + r_win >= img.shape[1] or y0 + r_win >= img.shape[0]: return np.nan
    win = img[y0 - r_win:y0 + r_win + 1, x0 - r_win:x0 + r_win + 1].astype(float)
    yy, xx = np.mgrid[-r_win:r_win + 1, -r_win:r_win + 1]
    d = np.hypot(xx + x0 + .5 - x, yy + y0 + .5 - y)
    ring = win[d > 4.0]
    bg = np.median(ring); sd = 1.4826 * np.median(np.abs(ring - bg)) + 1.0
    return (win[d <= r_in].max() - bg) / sd

rows = []     # per new/ordinary 3-camera point: label, evidence, blob cues
gain = {}
for f in range(1, N + 1):
    imgs = read_frame_images(pm, bname, 4, f)
    sel = np.flatnonzero((a["frame"] == f) & (ncam == 3))
    for c in range(4):
        gain[f, c] = np.nanmedian(a["sumg"][a["frame"] == f][:, c]) if (a["frame"] == f).any() else 1.0
    for i in sel:
        c = int(np.flatnonzero(a["blob"][i] < 0)[0])
        p = project_pixels(a["pos"][i][None], cals[c], cpar)[0]
        z = evidence(imgs[c], p)
        sg = np.array([a["sumg"][i, cc] / gain[f, cc] for cc in range(4) if a["blob"][i, cc] >= 0])
        rows.append((int(new[i]), int(gid[i] < 0), z, np.log(sg.min()), np.log(sg.max() / sg.min()), f))
R = np.array(rows); R = R[np.isfinite(R[:, 2])]
for nm, m in (("ordinary 3-camera points", R[:, 0] == 0), ("new 3-camera points", R[:, 0] == 1)):
    r = R[m]
    print(f"{nm:26s} n={len(r):6d}  ghost {100 * r[:, 1].mean():4.1f}%   missing-view light z: real median {np.median(r[r[:, 1] == 0, 2]):6.1f}  ghost median {np.median(r[r[:, 1] == 1, 2]):6.1f}")
r = R[R[:, 0] == 1]; y = r[:, 1]; grp = (r[:, 5] // 2).astype(int)
def score(cols):
    X = np.column_stack(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = sl.cross_val_logit(X, y, grp)
    thr = np.percentile(p[y == 0], 90)
    return sl.auc(p, y.astype(int)), (p[y == 1] > thr).mean(), (p[y == 0] > thr).mean()
for nm, cols in (("blob cues (view brightness)", [r[:, 3], r[:, 4]]), ("pixel cue (light in the missing camera)", [np.log1p(np.clip(r[:, 2], 0, None))]), ("both", [r[:, 3], r[:, 4], np.log1p(np.clip(r[:, 2], 0, None))])):
    auc, rec, fa = score(cols)
    print(f"  ghost vs real among new points, {nm:42s} AUC {auc:.3f}  ghosts removed {100 * rec:.0f}% at {100 * fa:.0f}% of the real lost")
