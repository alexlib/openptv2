# ruff: noqa
"""Light in the missing view of 3-camera points (raw image), real against synthetic. Truth-free.

    uv run python bench/limits/light_check_real.py <label> <run dir> <f0> <f1>
"""
import os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb
from openptv2.gui.ptv import read_frame_images
from openptv2.loco import project_pixels
from openptv2.merge_post import arrays_from_store
from openptv2.storage import RunStore

label, root, F0, F1 = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
exp = sb._experiment(root, root / "parameters_Run1.yaml", F0, F1)
pm = exp.pm; bname = pm.get_parameter("sequence")["base_name"]
st = RunStore(str(root / "res" / "run.zarr"), mode="r")
a = arrays_from_store(st, "ptv_is", F0, F1, 4)
ncam = (a["blob"] >= 0).sum(1)

def light_z(img, p, r_in=1.5, r_win=7):
    x, y = p; x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 - r_win < 0 or y0 - r_win < 0 or x0 + r_win >= img.shape[1] or y0 + r_win >= img.shape[0]: return np.nan
    win = img[y0 - r_win:y0 + r_win + 1, x0 - r_win:x0 + r_win + 1].astype(float)
    yy, xx = np.mgrid[-r_win:r_win + 1, -r_win:r_win + 1]
    d = np.hypot(xx + x0 + .5 - x, yy + y0 + .5 - y); ring = win[d > 4.0]
    bg = np.median(ring); sd = 1.4826 * np.median(np.abs(ring - bg)) + 1.0
    return (win[d <= r_in].max() - bg) / sd

os.chdir(root)
from collections import Counter
ln = Counter(a['tid'].tolist()); plen = np.array([ln[t] for t in a['tid']])
z3, z4, p3 = [], [], []
for f in range(F0, F1 + 1):
    imgs = read_frame_images(pm, bname, 4, f)
    for i in np.flatnonzero((a["frame"] == f) & (ncam == 3)):
        c = int(np.flatnonzero(a["blob"][i] < 0)[0])
        z3.append(light_z(imgs[c], project_pixels(a["pos"][i][None], exp.cals[c], exp.cpar)[0])); p3.append(plen[i])
    rows = np.flatnonzero((a["frame"] == f) & (ncam == 4))[::6]
    for i in rows:                      # a held-out view of a 4-camera point: where light must be
        c = i % 4
        z4.append(light_z(imgs[c], project_pixels(a["pos"][i][None], exp.cals[c], exp.cpar)[0]))
z3, z4, p3 = np.array(z3), np.array(z4), np.array(p3); p3 = p3[np.isfinite(z3)]; z3, z4 = z3[np.isfinite(z3)], z4[np.isfinite(z4)]
q = lambda z: " ".join(f"{np.percentile(z, p):6.1f}" for p in (5, 10, 25, 50))
print(f"[{label}] frames {F0}-{F1}: 3-camera points {len(z3)}; their missing view: z percentiles 5/10/25/50 = {q(z3)}")
print(f"[{label}] reference, a view of 4-camera points:               z percentiles 5/10/25/50 = {q(z4)}")
for t in (6, 10, 20):
    print(f"[{label}]   3-camera points with z < {t:2d}: {100 * (z3 < t).mean():4.1f}%   (4-camera views below: {100 * (z4 < t).mean():.1f}%)")

print(f"[{label}] 3-camera points by light in the missing view: share in tracks of 1 point / of 2-4 points / of 5+ points (track length within these {F1 - F0 + 1} frames)")
for nm, m in (("z < 10 ", z3 < 10), ("z 10-30", (z3 >= 10) & (z3 < 30)), ("z >= 30", z3 >= 30)):
    print(f"[{label}]   {nm}: n={m.sum():5d}   {100 * (p3[m] == 1).mean():4.1f}% / {100 * ((p3[m] >= 2) & (p3[m] <= 4)).mean():4.1f}% / {100 * (p3[m] >= 5).mean():4.1f}%")
