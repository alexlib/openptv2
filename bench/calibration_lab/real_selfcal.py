# ruff: noqa
"""Real recording (no truth): consistency report, self-calibration, alignment, and what tracking makes of each.

    POSTPTV_DIR=<clean postptv> uv run python bench/calibration_lab/real_selfcal.py <label> <run dir> <f0> <f1> <work> [tracker] [variants]

<run dir> holds parameters_Run1.yaml, cal/, img/ and res/run.zarr (matched, with targets). Nothing in it is modified: every
variant works on a copy (the images are symlinked). Variants (default off,align,selfcal):
  off      as recorded
  align    align_partial_points (explicit stopgap)
  selfcal  self_calibrate_from_store -> new cameras written over the copy's cal files -> frames matched again
Printed: the report of the original cameras, the self-calibration result (judged on held-out frames), and per variant track
statistics plus the jump at camera-set changes along tracks (second difference of a track through a 3-camera point between two
4-camera points, per missing camera: |mean| = systematic, rms = total) and the track noise at all-4-camera points.
"""
import importlib, os, shutil, sys, time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "limits"))
import syn1_pixel_fit as sp
sb = sp.sb
oc = importlib.import_module("openptv2.correspondences")
from openptv2.calibration_consistency import (
    align_partial_points, consistency_report, self_calibrate_from_store, write_calibration)
from openptv2.merge_post import arrays_from_store
from openptv2.plugins import run_tracking_plugin
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore

label, root, F0, F1, work = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
tracker = sys.argv[6] if len(sys.argv) > 6 else "two_phase"
variants = (sys.argv[7] if len(sys.argv) > 7 else "off,align,selfcal").split(",")
frames = list(range(F0, F1 + 1))
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
    raw.setdefault("track", {})["align_4cam"] = False      # the explicit variants below, never an implicit one
    (d / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return d


dirs = {v: make(v) for v in variants}
cpar, vpar, cals, _ = load_parameters(dirs["off"] / "parameters_Run1.yaml")
cals = list(cals)
print(f"[{label}] frames {F0}-{F1}, tracker {tracker}")
st_off = RunStore(str(dirs["off"] / "res" / "run.zarr"), mode="r")
rep0 = consistency_report(st_off, frames, cals, cpar, vpar)
print("\noriginal cameras\n" + rep0.text())

if "align" in variants:
    r = align_partial_points(RunStore(str(dirs["align"] / "res" / "run.zarr"), mode="a"), frames, cals, cpar, vpar, min_verdict="consistent")
    print("\n" + r.text())
if "selfcal" in variants:
    t0 = time.perf_counter()
    res = self_calibrate_from_store(st_off, frames, cals, cpar, vpar)
    print(f"\n{res.text()}   ({time.perf_counter() - t0:.0f}s)")
    write_calibration(res.cals, dirs["selfcal"] / "cal", prefix="cam_", suffix=".tif")
    cpar2, vpar2, cals2, _ = load_parameters(dirs["selfcal"] / "parameters_Run1.yaml")
    t0 = time.perf_counter()
    oc.match_correspondences_batch_parallel(frames=frames, cpar=cpar2, cals=list(cals2), vpar=vpar2,
        zarr_store_path=str(dirs["selfcal"] / "res" / "run.zarr"), n_workers=1, write_to_store=True)
    print(f"    matched {len(frames)} frames again in {time.perf_counter() - t0:.0f}s")
    rep1 = consistency_report(RunStore(str(dirs["selfcal"] / "res" / "run.zarr"), mode="r"), frames, list(cals2), cpar2, vpar2)
    print("  new cameras, all frames, re-matched\n" + rep1.text().replace("\n", "\n  "))

print(f"\ntracking with {tracker}")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
stats = {}
for v in variants:
    d = dirs[v]
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
    ok = (tid[2:] == tid[:-2]) & (frame[2:] == frame[:-2] + 2)
    mid = np.flatnonzero(ok) + 1
    s2 = pos[mid] - 0.5 * (pos[mid - 1] + pos[mid + 1])
    c4 = (nc[mid - 1] == 4) & (nc[mid + 1] == 4)
    jump, rms = [], []
    for c in range(4):
        sel = c4 & (nc[mid] == 3) & (blob[mid, c] < 0)
        jump.append(np.linalg.norm(s2[sel].mean(0)) * 1e3 if sel.sum() > 20 else np.nan)
        rms.append(np.sqrt(np.mean(np.sum(s2[sel] ** 2, 1))) * 1e3 if sel.sum() > 20 else np.nan)
    sel = (nc[mid] == 4) & (nc[mid - 1] == 4) & (nc[mid + 1] == 4)
    stats[v] = dict(points=len(tid), tracks=len(lens), mean_len=lens.mean(), tracks50=int((lens >= 50).sum()),
                    pts_in_20=int(lens[lens >= 20].sum()), jump=jump, rms=rms, rms4=np.sqrt(np.mean(np.sum(s2[sel] ** 2, 1))) * 1e3, t=tt)
print(f"    {'variant':8} {'points':>8} {'tracks':>8} {'mean len':>9} {'tracks>=50':>11} {'pts in >=20':>12} {'track noise (4 cams, um)':>25}   jump um per missing camera: systematic | rms")
for v in variants:
    s = stats[v]
    print(f"    {v:8} {s['points']:>8} {s['tracks']:>8} {s['mean_len']:>9.2f} {s['tracks50']:>11} {s['pts_in_20']:>12} {s['rms4']:>25.1f}   "
          + " ".join(f"{j:5.1f}" for j in s["jump"]) + " | " + " ".join(f"{r:5.0f}" for r in s["rms"]))
