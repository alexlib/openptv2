# ruff: noqa
"""align_4cam on a real recording (no truth): size, stability and the camera-switching jump along tracks.

    uv run python bench/limits/align4cam_real.py <label> <run dir (parameters_Run1.yaml, cal, img, res/run.zarr)> <f0> <f1> <work> [tracker]

Copies the store (the images are symlinked), then
  (1) corrections: |x_on - x_off| for the partial-camera points by camera count and by distance to the nearest 4-camera point
  (2) stability: the same corrections when the field is fitted on the first half of the frames only (the frames of that half
      are compared): how much does the result depend on the frame range?
  (3) tracking with ``tracker`` (default two_phase) on both stores: track statistics, and the jump at camera-set changes:
      for a point seen by 3 cameras between two 4-camera points of the same track, s = x_t - (x_(t-1) + x_(t+1)) / 2;
      |mean s| per missing camera is the systematic jump (random scatter averages out), rms(s) the total.
"""
import os, shutil, sys, time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

label, root, F0, F1, work = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
tracker = sys.argv[6] if len(sys.argv) > 6 else "two_phase"
frames = list(range(F0, F1 + 1))
mid = F0 + (F1 - F0) // 2
work.mkdir(parents=True, exist_ok=True)


def make(name):
    d = work / name
    if d.exists():
        shutil.rmtree(d)
    (d / "res").mkdir(parents=True)
    shutil.copytree(root / "cal", d / "cal")
    shutil.copytree(root / "res" / "run.zarr", d / "res" / "run.zarr")
    os.symlink(root / "img", d / "img")
    raw = yaml.safe_load((root / "parameters_Run1.yaml").read_text())
    raw["sequence"]["first"], raw["sequence"]["last"] = F0, F1
    raw.setdefault("track", {})["align_4cam"] = False        # the explicit alignment below, not the loader's
    (d / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return d


from openptv2.merge_post import arrays_from_store
from openptv2.merge_refine import align_store_correspondences_4cam
from openptv2.plugins import run_tracking_plugin
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore

off, on, half = make("off"), make("on"), make("half")
cpar, vpar, cals, _ = load_parameters(off / "parameters_Run1.yaml")
sto = RunStore(str(off / "res" / "run.zarr"), mode="r")
t0 = time.perf_counter()
n_on = align_store_correspondences_4cam(RunStore(str(on / "res" / "run.zarr"), mode="a"), frames, list(cals), cpar, vpar)
t_al = time.perf_counter() - t0
n_half = align_store_correspondences_4cam(RunStore(str(half / "res" / "run.zarr"), mode="a"), list(range(F0, mid + 1)), list(cals), cpar, vpar)
print(f"[{label}] frames {F0}-{F1}: aligned {n_on} partial points in {t_al:.2f}s")

stn, sth = RunStore(str(on / "res" / "run.zarr"), mode="r"), RunStore(str(half / "res" / "run.zarr"), mode="r")
P0, P1, PH, NC, FR = [], [], [], [], []
for f in frames:
    pos, ids = sto.read_correspondences(f)
    P0.append(np.asarray(pos, float)); P1.append(np.asarray(stn.read_correspondences(f)[0], float))
    PH.append(np.asarray(sth.read_correspondences(f)[0], float))
    NC.append((np.asarray(ids)[:, :4] >= 0).sum(1)); FR.append(np.full(len(pos), f))
P0, P1, PH, NC, FR = map(np.concatenate, (P0, P1, PH, NC, FR))
mv = np.linalg.norm(P1 - P0, axis=1)

print("\n(1) size of the correction |x_on - x_off| (mm); particle displacement per frame is about 0.1 mm")
for nc in (4, 3, 2):
    s = NC == nc
    if s.any():
        print(f"    {nc} cams: n {s.sum():>7}  median {np.median(mv[s]):.4f}  p90 {np.percentile(mv[s], 90):.4f}  p99 {np.percentile(mv[s], 99):.4f}  max {mv[s].max():.4f}")
tree4 = cKDTree(P0[NC == 4])
part = (NC < 4) & np.isfinite(P0[:, 0])
dn, _ = tree4.query(P0[part])
print(f"    {'distance to nearest 4-cam point':>32} {'n':>7} {'|corr| median':>14} {'p99':>8} {'max':>8}")
for lo, hi in ((0, .25), (.25, .5), (.5, 1), (1, 2), (2, 1e9)):
    s = (dn >= lo) & (dn < hi)
    if s.sum() >= 10:
        print(f"    {lo:>14g}-{min(hi, 99):<5g} mm {s.sum():>7} {np.median(mv[part][s]):>14.4f} {np.percentile(mv[part][s], 99):>8.4f} {mv[part][s].max():>8.4f}")

print("\n(2) stability: field fitted on the first half only vs all frames (compared on the first half, partial points)")
fh = part & (FR <= mid)
dh = np.linalg.norm(PH - P1, axis=1)[fh]
print(f"    |x_half - x_all|: median {np.median(dh):.4f}  p90 {np.percentile(dh, 90):.4f}  p99 {np.percentile(dh, 99):.4f} mm   (correction size on the same points: median {np.median(mv[fh]):.4f})")

print(f"\n(3) tracking with {tracker}")
res = {}
for lab, d in (("off", off), ("on", on)):
    exp = None
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import synth_bench as sb
    exp = sb._experiment(d, d / "parameters_Run1.yaml", F0, F1)
    cwd = os.getcwd(); os.chdir(d)
    try:
        t1 = time.perf_counter(); run_tracking_plugin(tracker, exp); tt = time.perf_counter() - t1
    finally:
        os.chdir(cwd)
    a = arrays_from_store(RunStore(str(d / "res" / "run.zarr"), mode="r"), "ptv_is", F0, F1, 4)
    frame, tid, pos, blob = a["frame"], a["tid"], np.asarray(a["pos"], float), a["blob"]
    nc = (blob >= 0).sum(1)
    o = np.lexsort((frame, tid)); frame, tid, pos, blob, nc = frame[o], tid[o], pos[o], blob[o], nc[o]
    cuts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True]); lens = np.diff(cuts)
    stat = dict(points=len(tid), tracks=len(lens), mean_len=lens.mean(), tracks50=int((lens >= 50).sum()), pts_in_20=int(lens[lens >= 20].sum()), t=tt)
    ok = (tid[2:] == tid[:-2]) & (frame[2:] == frame[:-2] + 2)
    mid_i = np.flatnonzero(ok) + 1
    s2 = pos[mid_i] - 0.5 * (pos[mid_i - 1] + pos[mid_i + 1])
    c4 = (nc[mid_i - 1] == 4) & (nc[mid_i + 1] == 4)
    rows = {}
    for c in range(4):
        sel = c4 & (nc[mid_i] == 3) & (blob[mid_i, c] < 0)
        rows[f"cam {c + 1} missing"] = (int(sel.sum()), np.linalg.norm(s2[sel].mean(0)) if sel.sum() > 20 else np.nan, np.sqrt(np.mean(np.sum(s2[sel] ** 2, 1))) if sel.sum() > 20 else np.nan)
    sel = (nc[mid_i] == 4) & (nc[mid_i - 1] == 4) & (nc[mid_i + 1] == 4)
    rows["all 4 cams"] = (int(sel.sum()), np.linalg.norm(s2[sel].mean(0)), np.sqrt(np.mean(np.sum(s2[sel] ** 2, 1))))
    res[lab] = (stat, rows)
for k in ("points", "tracks", "mean_len", "tracks50", "pts_in_20", "t"):
    print(f"    {k:10} off {res['off'][0][k]:>10.1f}   on {res['on'][0][k]:>10.1f}")
print(f"    jump at camera-set changes (second difference of the track, mm): |mean s| (systematic) and rms")
print(f"    {'':16} {'n':>6} {'|mean| off':>11} {'|mean| on':>10}   {'rms off':>8} {'rms on':>7}")
for k in res["off"][1]:
    (n0, m0, r0), (n1, m1, r1) = res["off"][1][k], res["on"][1][k]
    print(f"    {k:16} {n0:>6} {m0:>11.4f} {m1:>10.4f}   {r0:>8.4f} {r1:>7.4f}")
