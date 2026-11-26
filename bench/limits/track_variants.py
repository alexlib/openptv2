# ruff: noqa
"""Re-track matched stores of syn1 under different track settings; truth scores and where the id switches are.

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/track_variants.py <work> <n_frames> <base src> <relaxed src> <variants json>

Sources are matched (not corrected) stores; shared_correction is switched on in every run. variants: {"name": {track overrides}}.
"""
import json, os, shutil, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import syn1_pixel_fit as sp
sb = sp.sb
from openptv2.merge_post import arrays_from_store
from openptv2.plugins import run_tracking_plugin
from openptv2.storage import RunStore

work, N = Path(sys.argv[1]), int(sys.argv[2])
srcs = {"base": Path(sys.argv[3]), "relaxed": Path(sys.argv[4])}
variants = json.loads(sys.argv[5])
work.mkdir(parents=True, exist_ok=True)
cd = sp.case_dir(work, N); flow = sb.Flow()
sb0 = RunStore(str(srcs["base"] / "res" / "run.zarr"), mode="r")
n_base = {f: len(sb0.read_correspondences(f)[0]) for f in range(1, N + 1)}
tr = np.load(sb.SRC / "truth" / "truth.npz"); kk = tr["frame"] <= N
by = {int(f): (tr["pos_mm"][kk][tr["frame"][kk] == f], tr["particle_id"][kk][tr["frame"][kk] == f], cKDTree(tr["pos_mm"][kk][tr["frame"][kk] == f])) for f in np.unique(tr["frame"][kk])}
ref_keys = None
print(f"{'run':26s} {'yield':>6s} {'ghost%':>6s} {'linkP':>6s} {'velErr':>7s} {'vErrXtra':>8s} {'posErr':>6s}  switches: new-new  new-old  old-old   (rate of pairs, %)")
for sname, src in srcs.items():
    for vname, over in variants.items():
        dst = work / f"{sname}_{vname}"
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        yp = dst / "parameters_Run1.yaml"
        raw = yaml.safe_load(yp.read_text()); raw["track"].update({"shared_correction": True, **over}); yp.write_text(yaml.safe_dump(raw, sort_keys=False))
        exp = sb._experiment(dst, yp, 1, N)
        cwd = os.getcwd(); os.chdir(dst)
        try: run_tracking_plugin("two_phase", exp)
        finally: os.chdir(cwd)
        st = RunStore(str(dst / "res" / "run.zarr"), mode="r")
        a = arrays_from_store(st, "ptv_is", 1, N, 4)
        pred = {"trajid": a["tid"], "frame": a["frame"], "pos": np.asarray(a["pos"], float)}
        ev = sb.evaluate_pred(cd, pred, flow, ref_keys=ref_keys)
        if ref_keys is None and "_keys" in ev: ref_keys = ev["_keys"]
        ev.pop("_keys", None)
        g0, _, off = sb._match(by, a["frame"], a["pos"], 0.6); bias = np.median(off[g0 >= 0], axis=0)
        gid, _, _ = sb._match(by, a["frame"], a["pos"] - bias, 0.4)
        new = a["row"] >= np.array([n_base[int(f)] for f in a["frame"]])
        pt = {(int(gid[i]), int(a["frame"][i])): i for i in np.flatnonzero(gid >= 0)}
        pairs, sw = Counter(), Counter()
        for (p, f), i in pt.items():
            j = pt.get((p, f + 1))
            if j is None: continue
            c = ("new" if new[i] else "old") + "-" + ("new" if new[j] else "old")
            pairs[c] += 1; sw[c] += a["tid"][i] != a["tid"][j]
        rate = lambda c: 100 * sw[c] / max(pairs[c], 1)
        sw_new_old = sw["new-old"] + sw["old-new"]; pr_new_old = pairs["new-old"] + pairs["old-new"]
        f = lambda x, d=4: "   -  " if x is None else f"{x:.{d}f}"
        print(f"{sname + '_' + vname:26s} {f(ev.get('yield'))} {100 * ev.get('ghost_point_frac', 0):6.2f} {f(ev.get('raw_link_precision'))} {f(ev.get('vel_err_rel'))} {f(ev.get('vel_err_extra'), 3):>8s} {f(ev.get('pos_err_mm'))}  {sw['new-new']:4d} ({rate('new-new'):4.1f})  {sw_new_old:4d} ({100 * sw_new_old / max(pr_new_old, 1):4.1f})  {sw['old-old']:4d} ({rate('old-old'):4.1f})", flush=True)
        shutil.rmtree(dst / "res" / "run.zarr", ignore_errors=True)
