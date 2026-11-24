# ruff: noqa
"""Relax blob exclusivity in the matcher (second pass settings) on syn1, scored against truth.

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/relax_exclusivity.py <work> <n_frames> <variants>

variants: comma list of ``name=ratio:min_own`` (ratio None = no second pass); e.g. base=none,pass=1.7:2,p11=1.7:1,all1=0:1
ratio: a blob is offered again when brighter than ratio x the second-dimmest view of its point (0 = every blob);
min_own: a new point needs this many views on blobs no earlier point used (default pass: 2).
Matches in one process, tracks with two_phase, prints truth scores and the loss attribution of loss_in_episodes.py.
"""
import json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import syn1_pixel_fit as sp
sb = sp.sb
import importlib
oc = importlib.import_module("openptv2.correspondences")

def light_z(img, p, r_in=1.5, r_win=7):
    """peak of the raw image within r_in px of p over the local background, in units of the local noise (MAD)"""
    x, y = p
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 - r_win < 0 or y0 - r_win < 0 or x0 + r_win >= img.shape[1] or y0 + r_win >= img.shape[0]:
        return np.nan
    win = img[y0 - r_win:y0 + r_win + 1, x0 - r_win:x0 + r_win + 1].astype(float)
    yy, xx = np.mgrid[-r_win:r_win + 1, -r_win:r_win + 1]
    d = np.hypot(xx + x0 + .5 - x, yy + y0 + .5 - y)
    ring = win[d > 4.0]
    bg = np.median(ring); sd = 1.4826 * np.median(np.abs(ring - bg)) + 1.0
    return (win[d <= r_in].max() - bg) / sd

work, n_frames = Path(sys.argv[1]), int(sys.argv[2])
specs = []
for tok in sys.argv[3].split(","):
    name, _, v = tok.partition("=")
    flags = v.split("+")[1:]
    v = v.split("+")[0]
    specs.append((name, None if v == "none" else tuple(float(x) for x in v.split(":")), flags))
work.mkdir(parents=True, exist_ok=True)
cd = sp.case_dir(work, n_frames)
flow = sb.Flow()
orig = oc.shared_blob_pass
ref_keys = None
out = {}
for name, v, flags in specs:
    dst = work / name
    over = {}
    if 'corr' in flags: over['shared_correction'] = True
    if 'px' in flags: over['holistic_pixel_fit'] = True
    yaml_path = sp.prepare(dst, n_frames, over)
    exp = sb._experiment(dst, yaml_path, 1, n_frames)
    if v is not None:
        ratio, min_own = v
        oc.shared_blob_pass = lambda *a, _m=int(min_own), **k: orig(*a, **{**k, "min_own": _m})
    else:
        ratio = None
    t0 = time.perf_counter()
    oc.match_correspondences_batch_parallel(frames=range(1, n_frames + 1), cpar=exp.cpar, cals=exp.cals, vpar=exp.vpar,
        zarr_store_path=str(dst / "res" / "run.zarr"), n_workers=1, write_to_store=True, shared_pass=ratio if ratio is None else float(ratio) if ratio > 0 else 1e-9)
    t_match = time.perf_counter() - t0
    zf = [float(x[2:]) for x in flags if x.startswith("zf")]
    if zf:
        from openptv2.loco import project_pixels
        from openptv2.storage import RunStore as _RS
        from openptv2.gui.ptv import read_frame_images
        stw = _RS(str(dst / "res" / "run.zarr"), mode="a")
        pm_ = exp.pm; bname = pm_.get_parameter("sequence")["base_name"]
        cwd0 = os.getcwd(); os.chdir(dst)
        kept = tot = 0
        try:
            for f in range(1, n_frames + 1):
                pos, ids = stw.read_correspondences(f); pos = np.asarray(pos); ids = np.asarray(ids)[:, :4]
                imgs = read_frame_images(pm_, bname, 4, f)
                keep = np.ones(len(pos), bool)
                for i in np.flatnonzero((ids >= 0).sum(1) == 3):
                    c = int(np.flatnonzero(ids[i] < 0)[0])
                    pxy = project_pixels(pos[i][None], exp.cals[c], exp.cpar)[0]
                    z = light_z(imgs[c], pxy)
                    keep[i] = bool(np.isfinite(z) and z >= zf[0]) or not np.isfinite(z)
                tot += len(pos); kept += int(keep.sum())
                stw.write_correspondences(f, pos[keep], ids[keep])
        finally:
            os.chdir(cwd0)
        print(f"  {name}: light filter z>={zf[0]:g} kept {kept} of {tot} points", flush=True)
    exp = sb._experiment(dst, yaml_path, 1, n_frames)
    cwd = os.getcwd(); os.chdir(dst)
    try:
        from openptv2.plugins import run_tracking_plugin
        t1 = time.perf_counter(); run_tracking_plugin("two_phase", exp); t_track = time.perf_counter() - t1
    finally:
        os.chdir(cwd)
    from openptv2.merge_post import arrays_from_store
    from openptv2.storage import RunStore
    st = RunStore(str(dst / "res" / "run.zarr"), mode="r")
    a = arrays_from_store(st, "ptv_is", 1, n_frames, 4)
    pred = {"trajid": a["tid"], "frame": a["frame"], "pos": np.asarray(a["pos"], float)}
    ev = sb.evaluate_pred(cd, pred, flow, ref_keys=ref_keys)
    if ref_keys is None and "_keys" in ev: ref_keys = ev["_keys"]
    ev.pop("_keys", None)
    out[name] = {k: ev.get(k) for k in ("vel_err_rel", "vel_err_common", "vel_err_extra", "pos_err_mm", "yield", "ghost_point_frac", "long_completeness", "long_fragments", "raw_link_precision", "out_points")}
    out[name].update(t_match=round(t_match, 1), t_track=round(t_track, 1))
    print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out[name].items()}, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "loss_in_episodes.py"), str(dst), str(n_frames), "6"], capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if any(s in l for s in ("tracker state", "R0", "R1", "R2", "R3", "R4", "id switches with both", "LOST", "held with 3")) ), flush=True)
(work / "result.json").write_text(json.dumps(out, indent=2, default=str))
