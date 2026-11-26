# ruff: noqa
"""Re-calibrate from the tracers or correct the partial-camera points? Scored against truth on rendered syn1.

    POSTPTV_DIR=<clean postptv> uv run python bench/calibration_lab/selfcal_vs_align.py <work> <n_frames> [trackers]

<work> is an align4cam_study work dir (``base`` with the store matched under the processing calibration, ``truth_case``).
Variants, each tracked with every tracker and scored on the common set:
  off      the store as matched
  align    the 4-camera residual field subtracted from the partial-camera points (align_store_correspondences_4cam)
  selfcal  joint self-calibration on the clean 4-camera points (ext + principal point + k1 k2 p1 p2), new .ori files,
           frames matched again with them
Also printed for the calibrations: leave-one-camera-out error and 3D position error (raw and after the best affine map,
which is the part the tracers cannot define: the gauge).
"""
import importlib, os, shutil, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "limits"))
import syn1_pixel_fit as sp
sb = sp.sb
oc = importlib.import_module("openptv2.correspondences")
from openptv2.loco import loco_residuals
from openptv2.merge_post import arrays_from_store
from openptv2.merge_refine import align_store_correspondences_4cam
from openptv2.plugins import run_tracking_plugin
from openptv2.selfcal import joint_self_calibrate, triangulate_pixels
from openptv2.storage import RunStore

work, N = Path(sys.argv[1]), int(sys.argv[2])
trackers = [t for t in (sys.argv[3] if len(sys.argv) > 3 else "two_phase,fast_3d").split(",") if t]
variants = (sys.argv[4] if len(sys.argv) > 4 else "off,align,selfcal").split(",")
base = work / "base"
cd = work / "truth_case"
flow = sb.Flow()
exp = sb._experiment(base, base / "parameters_Run1.yaml", 1, N)
cals, cpar, vpar = list(exp.cals), exp.cpar, exp.vpar
tr = np.load(sb.SRC / "truth" / "truth.npz")


def clone(src, name):
    dst = work / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    return dst


def clean_4cam(st, frames, step=1):
    """(obs_px (n, 4, 2), pos (n, 3)): 4-camera points of ``frames`` (blob centres from the stored targets)."""
    xy_l, pos_l, fr_l = [], [], []
    for f in frames[::step]:
        pos, ids = st.read_correspondences(f)
        ids = np.asarray(ids)[:, :4]
        ok = (ids >= 0).all(1)
        if not ok.any():
            continue
        tg = [np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(4)]
        xy_l.append(np.stack([tg[c][ids[ok, c], 1:3] for c in range(4)], axis=1))
        pos_l.append(np.asarray(pos, float)[ok]); fr_l.append(np.full(ok.sum(), f))
    return np.concatenate(xy_l), np.concatenate(pos_l), np.concatenate(fr_l)


st0 = RunStore(str(base / "res" / "run.zarr"), mode="r")
frames = list(range(1, N + 1))
xy, pos, fr = clean_4cam(st0, frames)
# truth pairing for the scores (nearest true particle, clean 4-camera points only)
tpos = np.full_like(pos, np.nan)
for f in np.unique(fr):
    m = tr["frame"] == f
    d, j = cKDTree(tr["pos_mm"][m]).query(pos[fr == f])
    tp = tr["pos_mm"][m][j]
    tp[d > 0.4] = np.nan
    tpos[fr == f] = tp
good = np.isfinite(tpos[:, 0])
print(f"{len(xy)} clean 4-camera points in {N} frames, {good.sum()} matched to truth within 0.4 mm")


def affine_err(p, t):
    A = np.c_[p, np.ones(len(p))]
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    sv = np.linalg.svd(coef[:3], compute_uv=False)
    return np.linalg.norm(A @ coef - t, axis=1), float(np.prod(sv) ** (1 / 3))


def report(name, cs):
    odd = (fr % 2 == 0) & good                      # held-out frames
    pp = triangulate_pixels(xy[odd], cs, cpar, vpar)
    e = np.linalg.norm(pp - tpos[odd], axis=1)
    ae, sc = affine_err(pp, tpos[odd])
    lo = np.nanmean(np.abs(loco_residuals(xy[odd][:6000], cpar, cs, vpar)))
    print(f"    {name:10} position error vs truth: raw median {np.median(e):.4f} mm | after affine map {np.median(ae):.4f} mm "
          f"(scale of the map {sc:.5f}) | LOCO {lo:.3f} px", flush=True)


print("\ncalibrations (held-out frames, clean 4-camera points)")
report("processing", cals)
dirs = {"off": clone(base, "v_off")}
if "align" in variants:
    d = clone(base, "v_align")
    n_up = align_store_correspondences_4cam(RunStore(str(d / "res" / "run.zarr"), mode="a"), frames, cals, cpar, vpar)
    dirs["align"] = d
    print(f"    align: {n_up} partial points corrected")
if "selfcal" in variants:
    fit = (fr % 2 == 1)
    idx = np.flatnonzero(fit)
    idx = idx[np.random.default_rng(1).permutation(len(idx))[:12000]]
    t0 = time.perf_counter()
    res = joint_self_calibrate(xy[idx], pos[idx], cals, cpar, max_nfev=200, free_radial=("k1", "k2"), ext_sigma_scale=3.0, edge_px=1.0)
    print(f"    selfcal: {res.n_obs} observations, rms {res.rms_before_px:.3f} -> {res.rms_after_px:.3f} px, {res.nfev} evaluations, {time.perf_counter() - t0:.0f}s")
    report("selfcal", res.cals)
    d = clone(base, "v_selfcal")
    for c, cal in enumerate(res.cals):
        ori = d / "cal" / f"cam_{c + 1}.tif.ori"
        cal.write(str(ori), str(ori).replace(".ori", ".addpar"))
    exp_d = sb._experiment(d, d / "parameters_Run1.yaml", 1, N)
    t0 = time.perf_counter()
    oc.match_correspondences_batch_parallel(frames=range(1, N + 1), cpar=exp_d.cpar, cals=exp_d.cals, vpar=exp_d.vpar,
                                            zarr_store_path=str(d / "res" / "run.zarr"), n_workers=1, write_to_store=True)
    print(f"    re-matched {N} frames in {time.perf_counter() - t0:.0f}s")
    dirs["selfcal"] = d

print("\ntracking (truth scores; common set = keys of the first run)")
ref_keys = None
cols = ("vel_err_rel", "vel_err_common", "pos_err_mm", "yield", "ghost_point_frac", "long_completeness", "long_fragments", "raw_link_precision")
print(f"    {'tracker':10} {'variant':8} " + " ".join(f"{c[:14]:>14}" for c in cols))
import yaml as _y
for trk in trackers:
    for lab in variants:
        d = clone(dirs[lab], f"{lab}_{trk}")
        yml = d / "parameters_Run1.yaml"
        y = _y.safe_load(open(yml)); y.setdefault("track", {})["align_4cam"] = False
        _y.safe_dump(y, open(yml, "w"))
        e2 = sb._experiment(d, yml, 1, N)
        cwd = os.getcwd(); os.chdir(d)
        try:
            run_tracking_plugin(trk, e2)
        finally:
            os.chdir(cwd)
        a = arrays_from_store(RunStore(str(d / "res" / "run.zarr"), mode="r"), "ptv_is", 1, N, 4)
        ev = sb.evaluate_pred(cd, {"trajid": a["tid"], "frame": a["frame"], "pos": np.asarray(a["pos"], float)}, flow, ref_keys=ref_keys)
        if ref_keys is None and "_keys" in ev:
            ref_keys = ev["_keys"]
        ev.pop("_keys", None)
        print(f"    {trk:10} {lab:8} " + " ".join(f"{ev.get(c, float('nan')):>14.4f}" if isinstance(ev.get(c), (int, float)) else f"{str(ev.get(c)):>14}" for c in cols), flush=True)
