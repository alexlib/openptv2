# ruff: noqa
"""align_4cam on syn1 (rendered images, truth, a real calibration error), scored against truth.

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/align4cam_study.py <work> <n_frames> [trackers] [cal]

cal: as_is (default; processing calibration differs from the rendering one) | perfect (the rendering calibration,
reconstructed by synth_bench._render_cals: the control where there is nothing to correct) | x2 (error doubled).
trackers: comma list of plugin names (default two_phase,trackcorr,fast_3d).

Matches once, copies the store, applies align_store_correspondences_4cam to one copy ("on"), and prints for both:
  (1) point error against truth by number of cameras (bias vector, rms) -- the camera-switching jump is bias(3 cam) - bias(4 cam)
  (2) size of the correction (mm) by number of cameras and the largest values
  (3) tracking scores per tracker (velocity error on the common set, yield, ghosts, long-track completeness, link precision)
"""
import importlib, os, shutil, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import syn1_pixel_fit as sp
sb = sp.sb
oc = importlib.import_module("openptv2.correspondences")

work, N = Path(sys.argv[1]), int(sys.argv[2])
trackers = [t for t in (sys.argv[3] if len(sys.argv) > 3 else "two_phase,trackcorr,fast_3d").split(",") if t and t != "none"]
cal_mode = sys.argv[4] if len(sys.argv) > 4 else "as_is"
work.mkdir(parents=True, exist_ok=True)
cd = sp.case_dir(work, N)
flow = sb.Flow()

base = work / "base"
if not (base / "res" / "run.zarr").exists():
    yaml_path = sp.prepare(base, N, {})
else:
    yaml_path = base / "parameters_Run1.yaml"
exp = sb._experiment(base, yaml_path, 1, N)
cals, cpar, vpar = exp.cals, exp.cpar, exp.vpar
if cal_mode != "as_is" and not (base / "cal_mode.txt").exists():
    # perfect: processing calibration + offsets = the calibration the images were rendered with; x2: minus the offsets
    offs = np.load(sb.WORK / "render_cal_offsets.npy")
    sign = {"perfect": 1.0, "x2": -1.0}[cal_mode]
    for c, (cal, x) in enumerate(zip(cals, offs)):
        cal.set_pos(np.asarray(cal.get_pos(), float) + sign * x[:3])
        cal.set_angles(np.asarray(cal.get_angles(), float) + sign * x[3:] * 1e-3)
        ori = base / "cal" / f"cam_{c + 1}.tif.ori"
        cal.write(str(ori), str(ori).replace(".ori", ".addpar"))
    (base / "cal_mode.txt").write_text(cal_mode)
from openptv2.storage import RunStore

def has_corr(d):
    try:
        st = RunStore(str(d / "res" / "run.zarr"), mode="r")
        return st.has_correspondences(1)
    except Exception:
        return False

if cal_mode != "as_is" or not has_corr(base):   # a copied store holds points triangulated with the processing calibration: re-match
    t0 = time.perf_counter()
    oc.match_correspondences_batch_parallel(frames=range(1, N + 1), cpar=cpar, cals=cals, vpar=vpar,
        zarr_store_path=str(base / "res" / "run.zarr"), n_workers=1, write_to_store=True)
    print(f"matched {N} frames in {time.perf_counter() - t0:.0f}s", flush=True)

def clone(src, name):
    dst = work / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    return dst

off_dir, on_dir = clone(base, "off"), clone(base, "on")
from openptv2.merge_refine import align_store_correspondences_4cam
st_on = RunStore(str(on_dir / "res" / "run.zarr"), mode="a")
st_off = RunStore(str(off_dir / "res" / "run.zarr"), mode="r")
t0 = time.perf_counter()
n_up = align_store_correspondences_4cam(st_on, list(range(1, N + 1)), list(cals), cpar, vpar)
print(f"aligned {n_up} partial points in {time.perf_counter() - t0:.2f}s", flush=True)

tr = np.load(sb.SRC / "truth" / "truth.npz")
k = tr["frame"] <= N
by = {int(f): cKDTree(tr["pos_mm"][k][tr["frame"][k] == f]) for f in np.unique(tr["frame"][k])}
tpos = {int(f): tr["pos_mm"][k][tr["frame"][k] == f] for f in by}

def point_table(st):
    rows = []
    for f in range(1, N + 1):
        pos, ids = st.read_correspondences(f)
        pos, ids = np.asarray(pos, float), np.asarray(ids)[:, :4]
        d, j = by[f].query(pos)
        ok = d < 0.6
        rows.append(np.column_stack([np.full(len(pos), f), (ids >= 0).sum(1), ok, pos - np.where(ok[:, None], tpos[f][j], np.nan), pos]))
    return np.vstack(rows)

A, B = point_table(st_off), point_table(st_on)
print("\n(1) point error against truth (mm; matched within 0.6 mm), bias = mean offset vector")
print(f"    {'cams':>4} {'n':>7} {'':>3}| {'bias x,y,z (off)':>28} {'rms (off)':>10} | {'bias x,y,z (on)':>28} {'rms (on)':>10}")
for nc in (4, 3, 2):
    s = (A[:, 1] == nc) & (A[:, 2] == 1)
    if s.sum() < 20:
        continue
    for lab, M in (("off", A), ("on", B)):
        pass
    ba, bb = np.nanmean(A[s, 3:6], 0), np.nanmean(B[s, 3:6], 0)
    ra, rb = np.sqrt(np.nanmean(np.sum(A[s, 3:6] ** 2, 1))), np.sqrt(np.nanmean(np.sum(B[s, 3:6] ** 2, 1)))
    print(f"    {nc:>4} {s.sum():>7}    | {np.array2string(ba, precision=4):>28} {ra:>10.4f} | {np.array2string(bb, precision=4):>28} {rb:>10.4f}")
s3, s4 = (A[:, 1] == 3) & (A[:, 2] == 1), (A[:, 1] == 4) & (A[:, 2] == 1)
jump = lambda M: np.linalg.norm(np.nanmean(M[s3, 3:6], 0) - np.nanmean(M[s4, 3:6], 0))
print(f"    camera-switching jump |bias(3 cam) - bias(4 cam)|: off {jump(A):.4f} mm -> on {jump(B):.4f} mm")

print("\n(1b) local 3-cam vs 4-cam difference: error of a matched 3-cam point minus the mean error of its 30 nearest matched 4-cam points")
m4 = (A[:, 1] == 4) & (A[:, 2] == 1)
tree4 = cKDTree(A[m4, 6:9])
e4a, e4b = A[m4, 3:6], B[m4, 3:6]
for nc in (3, 2):
    s_ = (A[:, 1] == nc) & (A[:, 2] == 1) & (B[:, 2] == 1)
    if s_.sum() < 20:
        continue
    d, j = tree4.query(A[s_, 6:9], k=30)
    la, lb = A[s_, 3:6] - e4a[j].mean(1), B[s_, 3:6] - e4b[j].mean(1)
    print(f"    {nc} cams: n {s_.sum():>6}  |local diff| rms off {np.sqrt(np.mean(np.sum(la**2, 1))):.4f} -> on {np.sqrt(np.mean(np.sum(lb**2, 1))):.4f} mm   mean vector off {np.array2string(la.mean(0), precision=4)} on {np.array2string(lb.mean(0), precision=4)}")

print("    per missing camera (3-cam points): |mean local diff vector| off -> on (mm), n")
for c_ in range(4):
    ids_all = np.vstack([np.asarray(st_off.read_correspondences(f)[1])[:, :4] for f in range(1, N + 1)])
    miss = (ids_all[:, c_] < 0) & ((ids_all >= 0).sum(1) == 3)
    s_ = miss & (A[:, 2] == 1) & (B[:, 2] == 1)
    if s_.sum() < 20:
        continue
    d, j = tree4.query(A[s_, 6:9], k=30)
    la, lb = A[s_, 3:6] - e4a[j].mean(1), B[s_, 3:6] - e4b[j].mean(1)
    print(f"      cam {c_ + 1} missing: {np.linalg.norm(la.mean(0)):.4f} -> {np.linalg.norm(lb.mean(0)):.4f}   (n {s_.sum()}, rms {np.sqrt(np.mean(np.sum(la**2, 1))):.4f} -> {np.sqrt(np.mean(np.sum(lb**2, 1))):.4f})")

print("\n(2) size of the correction |x_on - x_off| (mm)")
mv = np.linalg.norm(B[:, 6:9] - A[:, 6:9], axis=1)
for nc in (4, 3, 2):
    s_ = A[:, 1] == nc
    if s_.sum():
        print(f"    {nc} cams: n {s_.sum():>7}  median {np.median(mv[s_]):.4f}  p99 {np.percentile(mv[s_], 99):.4f}  max {mv[s_].max():.4f}")

print("\n(2b) extrapolation: partial-camera points by distance to the nearest 4-camera point (the region the field was fitted on)")
part = (A[:, 1] < 4) & np.isfinite(A[:, 6])
dn, _ = tree4.query(A[part, 6:9])
ok_ = A[part, 2] == 1
mvp = mv[part]
ea, eb = np.linalg.norm(A[part, 3:6], axis=1), np.linalg.norm(B[part, 3:6], axis=1)
okb_ = B[part, 2] == 1
print(f"    {'dist (mm)':>10} {'n':>7} {'|corr| median':>14} {'p99':>8} {'max':>8} {'match off':>9} {'match on':>9} {'lost':>6} {'gained':>7} {'rms err (matched in both)':>26}")
for lo, hi in ((0, .25), (.25, .5), (.5, 1), (1, 2), (2, 1e9)):
    s_ = (dn >= lo) & (dn < hi)
    if s_.sum() < 10:
        continue
    t_ = s_ & ok_ & okb_
    print(f"    {lo:>4g}-{min(hi, 99):<5g} {s_.sum():>7} {np.median(mvp[s_]):>14.4f} {np.percentile(mvp[s_], 99):>8.4f} {mvp[s_].max():>8.4f} {(s_ & ok_).sum():>9} {(s_ & okb_).sum():>9} {(s_ & ok_ & ~okb_).sum():>6} {(s_ & ~ok_ & okb_).sum():>7} {np.sqrt(np.mean(ea[t_]**2)):>12.4f} -> {np.sqrt(np.mean(eb[t_]**2)):.4f}")

print("\n(3) tracking (truth scores; common set = keys of the first run)")
ref_keys = None
from openptv2.merge_post import arrays_from_store
from openptv2.plugins import run_tracking_plugin
cols = ("vel_err_rel", "vel_err_common", "pos_err_mm", "yield", "ghost_point_frac", "long_completeness", "long_fragments", "raw_link_precision")
print(f"    {'tracker':10} {'align':5} " + " ".join(f"{c[:14]:>14}" for c in cols) + "   t_track")
for trk in trackers:
    for lab, src in (("off", off_dir), ("on", on_dir)):
        d = clone(src, f"{lab}_{trk}")
        yml = sp.prepare(d, N, {}) if not (d / "parameters_Run1.yaml").exists() else d / "parameters_Run1.yaml"
        # the loader would align again when align_4cam is on: switch it off for the explicit comparison
        import yaml as _y
        y = _y.safe_load(open(yml)); y.setdefault("track", {})["align_4cam"] = False
        _y.safe_dump(y, open(yml, "w"))
        exp2 = sb._experiment(d, yml, 1, N)
        cwd = os.getcwd(); os.chdir(d)
        try:
            t1 = time.perf_counter(); run_tracking_plugin(trk, exp2); tt = time.perf_counter() - t1
        finally:
            os.chdir(cwd)
        st = RunStore(str(d / "res" / "run.zarr"), mode="r")
        a = arrays_from_store(st, "ptv_is", 1, N, 4)
        ev = sb.evaluate_pred(cd, {"trajid": a["tid"], "frame": a["frame"], "pos": np.asarray(a["pos"], float)}, flow, ref_keys=ref_keys)
        if ref_keys is None and "_keys" in ev:
            ref_keys = ev["_keys"]
        ev.pop("_keys", None)
        print(f"    {trk:10} {lab:5} " + " ".join(f"{ev.get(c, float('nan')):>14.4f}" if isinstance(ev.get(c), (int, float)) else f"{str(ev.get(c)):>14}" for c in cols) + f"   {tt:.1f}s", flush=True)
