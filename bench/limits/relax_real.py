# ruff: noqa
"""Relaxed second pass and the light filter on a real recording (scratch copy; the source is only read). Truth-free scores.

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/relax_real.py <source test dir> <work dir> <n_frames> <variants>

variants: name=ratio:min_own[+corr][+zfT]   (ratio none = no second pass), as in relax_exclusivity.py
"""
import importlib, os, shutil, sys, time
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_check")); sys.path.insert(0, str(Path(__file__).parent))
import synth_bench as sb
import shared_pass_real as spr
from openptv2.loco import loco_residuals, project_pixels
from openptv2.recorrespond import load_parameters
from openptv2.gui.ptv import read_frame_images
from openptv2.merge_post import arrays_from_store
from openptv2.storage import RunStore
oc = importlib.import_module("openptv2.correspondences")
exec(open(Path(__file__).parent / "relax_exclusivity.py").read().split("def light_z")[1].split("work, n_frames")[0].join(["def light_z", ""]) if False else "")

def light_z(img, p, r_in=1.5, r_win=7):
    x, y = p; x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 - r_win < 0 or y0 - r_win < 0 or x0 + r_win >= img.shape[1] or y0 + r_win >= img.shape[0]: return np.nan
    win = img[y0 - r_win:y0 + r_win + 1, x0 - r_win:x0 + r_win + 1].astype(float)
    yy, xx = np.mgrid[-r_win:r_win + 1, -r_win:r_win + 1]
    d = np.hypot(xx + x0 + .5 - x, yy + y0 + .5 - y); ring = win[d > 4.0]
    bg = np.median(ring); sd = 1.4826 * np.median(np.abs(ring - bg)) + 1.0
    return (win[d <= r_in].max() - bg) / sd

src, work, N = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
specs = []
for tok in sys.argv[4].split(","):
    name, _, v = tok.partition("="); flags = v.split("+")[1:]; v = v.split("+")[0]
    specs.append((name, None if v == "none" else tuple(float(x) for x in v.split(":")), flags))
orig = oc.shared_blob_pass
work.mkdir(parents=True, exist_ok=True)
for name, v, flags in specs:
    dst = work / name
    over = {"shared_correction": True} if "corr" in flags else {}
    yaml_path = spr.prepare(src, dst, N, over)
    if not (dst / "img").exists(): os.symlink(src / "img", dst / "img")
    exp = sb._experiment(dst, yaml_path, 1, N)
    ratio = None
    if v is not None:
        ratio, mo = v
        oc.shared_blob_pass = lambda *a, _m=int(mo), **k: orig(*a, **{**k, "min_own": _m})
    oc.match_correspondences_batch_parallel(frames=range(1, N + 1), cpar=exp.cpar, cals=exp.cals, vpar=exp.vpar, zarr_store_path=str(dst / "res" / "run.zarr"), n_workers=1, write_to_store=True, shared_pass=ratio)
    zf = [float(x[2:]) for x in flags if x.startswith("zf")]
    if zf:
        stw = RunStore(str(dst / "res" / "run.zarr"), mode="a"); pm = exp.pm; bname = pm.get_parameter("sequence")["base_name"]
        cwd0 = os.getcwd(); os.chdir(dst); kept = tot = 0
        try:
            for f in range(1, N + 1):
                pos, ids = stw.read_correspondences(f); pos = np.asarray(pos); ids = np.asarray(ids)[:, :4]
                imgs = read_frame_images(pm, bname, 4, f); keep = np.ones(len(pos), bool)
                for i in np.flatnonzero((ids >= 0).sum(1) == 3):
                    c = int(np.flatnonzero(ids[i] < 0)[0])
                    z = light_z(imgs[c], project_pixels(pos[i][None], exp.cals[c], exp.cpar)[0])
                    keep[i] = (not np.isfinite(z)) or z >= zf[0]
                tot += len(pos); kept += int(keep.sum()); stw.write_correspondences(f, pos[keep], ids[keep])
        finally:
            os.chdir(cwd0)
    exp = sb._experiment(dst, yaml_path, 1, N)
    from openptv2.plugins import run_tracking_plugin
    cwd = os.getcwd(); os.chdir(dst)
    try:
        run_tracking_plugin("two_phase", exp)
    finally:
        os.chdir(cwd)
    st = RunStore(str(dst / "res" / "run.zarr"), mode="r")
    a = arrays_from_store(st, "ptv_is", 1, N, 4)
    frame, tid, pos, blob = a["frame"], a["tid"], a["pos"], a["blob"]
    ln = np.bincount(np.unique(tid, return_inverse=True)[1])[np.unique(tid, return_inverse=True)[1]]
    o = np.lexsort((frame, tid)); t_s, f_s, p_s = tid[o], frame[o], pos[o]
    ok = (t_s[:-2] == t_s[2:]) & (f_s[2:] - f_s[:-2] == 2)
    rough = np.linalg.norm(p_s[:-2][ok] - 2 * p_s[1:-1][ok] + p_s[2:][ok], axis=1)
    cpar, vpar, cals, _ = load_parameters(dst / "parameters_Run1.yaml")
    res = []
    for f in range(1, N + 1, 3):
        r = np.flatnonzero((frame == f) & (blob >= 0).all(1))
        if len(r) == 0: continue
        px = np.stack([a["centre"][r, c] for c in range(4)], axis=1)
        v = loco_residuals(px, cpar, cals, vpar); res.append(v[np.isfinite(v)])
    res = np.concatenate(res)
    nc = (blob >= 0).sum(1)
    print(f"{name:10s} points/frame {len(frame) / N:7.1f}  3-cam {100 * (nc == 3).mean():4.1f}%  singletons {100 * (ln == 1).mean():4.1f}%  in tracks>=20 {100 * (ln >= 20).mean():4.1f}%  >=30 {100 * (ln >= 30).mean():4.1f}%  mean len {ln.mean():5.1f}  roughness median {np.median(rough):.4f} p90 {np.percentile(rough, 90):.4f}  LOCO median {np.median(res):.3f} p90 {np.percentile(res, 90):.3f}", flush=True)
