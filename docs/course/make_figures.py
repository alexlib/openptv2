# ruff: noqa
"""Figures of the course, made from the bundled data (test_data/test_cavity_small). Run from the repository root:

    uv run python docs/course/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openptv2.recorrespond import load_parameters

root = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small"
out = Path(__file__).resolve().parent / "figures"
cpar, vpar, cals, _ = load_parameters(root / "parameters_Run1.yaml")

# four camera images of one frame
fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
for i, a in enumerate(ax):
    im = np.asarray(Image.open(root / "img" / f"cam{i + 1}.10000"))
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(f"camera {i + 1}, frame 10000", fontsize=9)
    a.set_xticks([0, 128, 255]); a.set_yticks([0, 128, 255])
fig.tight_layout(); fig.savefig(out / "four_images.png", dpi=110); plt.close(fig)

# where the cameras are (top view x-z) and what they look at
fig, a = plt.subplots(figsize=(9, 3.6))
for i, c in enumerate(cals):
    p = np.asarray(c.get_pos(), float)
    m = np.asarray(c.get_rotation_matrix(), float)
    look = m[:, 2] if abs(m[2, 2]) > 0.5 else m[2, :]
    a.plot(p[2], p[0], "ks")
    a.annotate(f"camera {i + 1}", (p[2], p[0]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    a.arrow(p[2], p[0], -np.sign(p[2]) * 260, 0, head_width=12, color="#888", length_includes_head=True)
a.axvspan(-125, 125, color="#d7ecf7", alpha=0.7)
a.axvspan(-131, -125, color="#7fb4e0"); a.axvspan(125, 131, color="#7fb4e0")
a.add_patch(plt.Rectangle((-20, -10), 45, 20, fill=False, ec="#b23a48", lw=2))
a.text(2, 18, "measurement\nvolume", ha="center", color="#b23a48", fontsize=9)
a.set_xlabel("z (mm)"); a.set_ylabel("x (mm)"); a.set_aspect("equal"); a.set_xlim(-650, 650); a.set_ylim(-170, 170)
a.set_title("top view: four cameras, two on each side of a water cell", fontsize=10)
fig.tight_layout(); fig.savefig(out / "camera_layout.png", dpi=110); plt.close(fig)
print("figures written to", out)


# ======================================================================================================================
# Pages 4-7: run the pipeline on a scratch copy of the example (the bundled data is never written to) and draw the results
import shutil
import subprocess
import sys
import tempfile

from scipy import ndimage

from openptv2.epipolar import epipolar_curve
from openptv2.storage.run_store import RunStore

work = Path(tempfile.mkdtemp(prefix="course_fig_")) / "my_run"
shutil.copytree(root, work, ignore=shutil.ignore_patterns("tmp*.txt", "res"))
batch = [sys.executable, "-c", "import sys; from openptv2.batch.pyptv_batch import main_cli; sys.argv[0]='openptv2-batch'; main_cli()"]
for mode in ("sequence", "tracking"):
    subprocess.run(batch + ["--workdir", str(work), "--first", "10000", "--last", "10004", "--mode", mode],
                   check=True, capture_output=True)
store = RunStore(work / "res" / "run.zarr", mode="r")
FR = list(range(10000, 10005))
cpar, vpar, cals, params = load_parameters(work / "parameters_Run1.yaml")
eps_px = float(params["criteria"]["eps0"]) / cpar.pix_x
img = lambda cam, f: np.asarray(Image.open(work / "img" / f"cam{cam + 1}.{f}")).astype(float)
tg = lambda cam, f: np.array([[t.x, t.y] for t in store.read_targets(cam, f)])

# 4. detection: all targets of camera 1, and a zoom
f0 = 10000
im, xy = img(0, f0), tg(0, f0)
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[1, 1]))
ax[0].imshow(im, cmap="gray", vmin=0, vmax=160)
ax[0].plot(xy[:, 0], xy[:, 1], "o", mfc="none", mec="#ff5a4d", ms=7, mew=0.9)
ax[0].add_patch(plt.Rectangle((140, 140), 64, 64, fill=False, ec="#4da3ff", lw=1.6))
ax[0].set_title(f"camera 1, frame {f0}: {len(xy)} targets found (red rings)", fontsize=10)
ax[1].imshow(im, cmap="gray", vmin=0, vmax=160, interpolation="nearest")
sel = (xy[:, 0] > 140) & (xy[:, 0] < 204) & (xy[:, 1] > 140) & (xy[:, 1] < 204)
ax[1].plot(xy[sel, 0], xy[sel, 1], "+", color="#ff5a4d", ms=14, mew=1.6)
ax[1].set_xlim(140, 204); ax[1].set_ylim(204, 140)
ax[1].set_title("zoom on the blue box: one + per target, at its centre\n(the centre is found to a fraction of a pixel)", fontsize=10)
fig.tight_layout(); fig.savefig(out / "detect_overlay.png", dpi=110); plt.close(fig)

# 4b. the threshold: which pixels count as "bright"
fig, ax = plt.subplots(1, 4, figsize=(12, 3.5))
crop = im[140:204, 140:204]
ax[0].imshow(crop, cmap="gray", vmin=0, vmax=160, interpolation="nearest"); ax[0].set_title("the image", fontsize=10)
flat = im - ndimage.uniform_filter(im, 15)                 # a simple background removal (the program uses a high-pass filter)
flat = flat[140:204, 140:204]
for a, thr in zip(ax[1:], (4, 15, 40)):
    mask = flat > thr
    _, n = ndimage.label(mask)
    a.imshow(mask.astype(float), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.set_title(f"brighter than {thr}\n{n} groups", fontsize=9)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(out / "detect_threshold.png", dpi=110); plt.close(fig)

# 5. the epipolar line of one camera-1 target in camera 2, and the match
pos, cid = store.read_correspondences(f0)
q = int(np.where((cid >= 0).all(1))[0][0])
t0 = store.read_targets(0, f0)[int(cid[q, 0])]
t1 = store.read_targets(1, f0)[int(cid[q, 1])]
line = epipolar_curve((t0.x, t0.y), cals[0], cals[1], 60, cpar, vpar)
xy2 = tg(1, f0)
# distance of every camera-2 target to the polyline (px)
def dist_to_line(p, L):
    d = []
    for a, b in zip(L[:-1], L[1:]):
        ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1); d.append(np.linalg.norm(p - (a + t * ab), axis=-1))
    return np.min(d, axis=0)
d2 = np.array([dist_to_line(p, line) for p in xy2])
near = d2 < eps_px
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
ax[0].imshow(img(0, f0), cmap="gray", vmin=0, vmax=160)
ax[0].plot(t0.x, t0.y, "o", mfc="none", mec="#ff5a4d", ms=16, mew=2)
ax[0].set_title("camera 1: pick one target (red ring)", fontsize=10)
ax[1].imshow(img(1, f0), cmap="gray", vmin=0, vmax=160)
ax[1].plot(line[:, 0], line[:, 1], "-", color="#4da3ff", lw=1.2)
nrm = np.gradient(line, axis=0); nrm = np.column_stack([-nrm[:, 1], nrm[:, 0]]) / np.linalg.norm(nrm, axis=1, keepdims=True)
ax[1].fill(np.r_[line[:, 0] + eps_px * nrm[:, 0], (line[::-1, 0] - eps_px * nrm[::-1, 0])],
           np.r_[line[:, 1] + eps_px * nrm[:, 1], (line[::-1, 1] - eps_px * nrm[::-1, 1])], color="#4da3ff", alpha=0.25, lw=0)
ax[1].plot(xy2[~near, 0], xy2[~near, 1], ".", color="#bbbbbb", ms=4)
ax[1].plot(xy2[near, 0], xy2[near, 1], "o", mfc="none", mec="#ffd24d", ms=9, mew=1.4)
ax[1].plot(t1.x, t1.y, "o", mfc="none", mec="#ff5a4d", ms=16, mew=2)
ax[1].set_title(f"camera 2: its epipolar line (blue) and the +-{eps_px:.0f} px band.\n{int(near.sum())} targets in the band (yellow); the match is the red ring", fontsize=10)
for a in ax:
    a.set_xlim(0, 255); a.set_ylim(255, 0)
fig.tight_layout(); fig.savefig(out / "epipolar.png", dpi=110); plt.close(fig)

# 5b. what matching found: counts per frame and the 3D points of one frame
cnt = {k: [] for k in (3, 4)}
for f in FR:
    p, c = store.read_correspondences(f)
    s = (c >= 0).sum(1)
    for k in cnt:
        cnt[k].append(int((s == k).sum()))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1, 1.4]))
ax[0].bar([str(f) for f in FR], cnt[4], color="#b23a48", label="seen by 4 cameras")
ax[0].bar([str(f) for f in FR], cnt[3], bottom=cnt[4], color="#e0a458", label="seen by 3 cameras")
ax[0].set_xlabel("frame"); ax[0].set_ylabel("matched 3D points"); ax[0].legend(fontsize=8); ax[0].set_title("matches per frame", fontsize=10)
s = (cid >= 0).sum(1)
for k, col in ((3, "#e0a458"), (4, "#b23a48")):
    ax[1].plot(pos[s == k, 2], pos[s == k, 0], "o", color=col, ms=5, label=f"{k} cameras")
x0, x1 = params["criteria"]["X_lay"]; z0, z1 = params["criteria"]["Zmin_lay"][0], params["criteria"]["Zmax_lay"][0]
ax[1].add_patch(plt.Rectangle((z0, x0), z1 - z0, x1 - x0, fill=False, ec="#555", ls="--"))
ax[1].set_xlabel("z (mm)"); ax[1].set_ylabel("x (mm)"); ax[1].legend(fontsize=8)
ax[1].set_title(f"frame {f0}: the matched 3D points (top view; dashed = measurement volume)", fontsize=10)
fig.tight_layout(); fig.savefig(out / "matches.png", dpi=110); plt.close(fig)

# 6. tracks
tr = store.trajectories()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for tid in np.unique(tr["trajid"]):
    m = tr["trajid"] == tid
    o = np.argsort(tr["time"][m]); p = 1000.0 * tr["pos"][m][o]          # the store keeps metres
    for a, (i, j) in zip(ax, ((2, 0), (2, 1))):
        a.plot(p[:, i], p[:, j], "-", lw=1.4)
        a.plot(p[0, i], p[0, j], "o", ms=5, color="#222"); a.plot(p[-1, i], p[-1, j], ">", ms=6, color="#b23a48")
ax[0].set_xlabel("z (mm)"); ax[0].set_ylabel("x (mm)"); ax[0].set_title("top view (z, x)", fontsize=10)
ax[1].set_xlabel("z (mm)"); ax[1].set_ylabel("y (mm)"); ax[1].set_title("side view (z, y)", fontsize=10)
fig.suptitle(f"{len(np.unique(tr['trajid']))} trajectories over 5 frames (dot = first frame, triangle = last)", fontsize=10)
fig.tight_layout(); fig.savefig(out / "tracks.png", dpi=110); plt.close(fig)
print("course figures written to", out)


# ======================================================================================================================
# Pages 8-9: the larger example (test_data/test_cavity: same experiment, full images), on a scratch copy
from openptv2.calibration_consistency import consistency_report

big = Path(tempfile.mkdtemp(prefix="course_fig_big_")) / "big_run"
shutil.copytree(root.parent / "test_cavity", big, ignore=shutil.ignore_patterns("res"))
for mode in ("sequence", "tracking"):
    subprocess.run(batch + ["--workdir", str(big), "--first", "10000", "--last", "10004", "--mode", mode], check=True, capture_output=True)
bstore = RunStore(big / "res" / "run.zarr", mode="r")
bcpar, bvpar, bcals, _ = load_parameters(big / "parameters_Run1.yaml")
rep = consistency_report(bstore, FR, bcals, bcpar, bvpar)
um_px = rep.mm_per_px * 1000.0
fig, a = plt.subplots(figsize=(8.5, 3.4))
a.axvspan(0, 0.05 * um_px, color="#2e8b57", alpha=0.18); a.axvspan(0.05 * um_px, 0.30 * um_px, color="#e0a458", alpha=0.25)
a.axvspan(0.30 * um_px, 0.45 * um_px, color="#b23a48", alpha=0.18)
a.barh([f"camera {i + 1}" for i in range(4)], rep.jump_um, xerr=[3 * s for s in rep.se_um], color="#4f7cac", capsize=3)
a.text(0.025 * um_px, 3.55, "consistent", ha="center", fontsize=8); a.text(0.175 * um_px, 3.55, "marginal", ha="center", fontsize=8)
a.text(0.375 * um_px, 3.55, "inconsistent", ha="center", fontsize=8)
a.set_xlim(0, 0.45 * um_px); a.invert_yaxis()
a.set_xlabel("position jump when this camera is missing (micrometres; bar = value, whisker = 3 standard errors)")
a.set_title(f"camera-switching jump, {rep.n_points} four-camera points in 5 frames: verdict {rep.verdict.upper()}", fontsize=10)
fig.tight_layout(); fig.savefig(out / "consistency_jump.png", dpi=110); plt.close(fig)

# page 9: the trajectories of the larger example, coloured by speed
tr = bstore.trajectories()
fig, a = plt.subplots(figsize=(7.5, 4.6))
segs, spd = [], []
for tid in np.unique(tr["trajid"]):
    m = tr["trajid"] == tid
    o = np.argsort(tr["time"][m]); p = 1000.0 * tr["pos"][m][o]
    for p0, p1 in zip(p[:-1], p[1:]):
        segs.append([(p0[0], p0[1]), (p1[0], p1[1])]); spd.append(np.linalg.norm(p1 - p0))
from matplotlib.collections import LineCollection
lc = LineCollection(segs, array=np.array(spd), cmap="viridis", linewidths=1.6)
a.add_collection(lc); a.autoscale(); a.set_aspect("equal")
fig.colorbar(lc, ax=a, label="step between two frames (mm)")
a.set_xlabel("x (mm)"); a.set_ylabel("y (mm)"); a.set_title(f"{len(np.unique(tr['trajid']))} trajectories of the larger example, seen along z", fontsize=10)
fig.tight_layout(); fig.savefig(out / "tracks_big.png", dpi=110); plt.close(fig)
print("page 8-9 figures written")
