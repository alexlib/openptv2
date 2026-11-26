# ruff: noqa
"""ptv.light_check / ptv.shared_pass / track settings through the real pipeline (py_sequence_loop + two_phase), scored.

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/light_check_bench.py syn|real <source test dir> <work> <n_frames> '<variants json>'

variants: {"name": {"ptv": {...}, "track": {...}}}; shared_correction is on in every run. Scratch copies only (the source is read).
syn: truth scores (syn1: vel_err_rel / common / extra, yield, ghost share, link precision, id switches);
real: truth-free scores (points per frame, singletons, share of points in tracks of >= 50 / >= 200 frames, roughness, LOCO).
"""
import json, os, shutil, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
import synth_bench as sb
import syn1_pixel_fit as sp
from openptv2.gui import ptv
from openptv2.loco import loco_residuals
from openptv2.merge_post import arrays_from_store
from openptv2.plugins import run_tracking_plugin
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore

def main():
    global kind, src, work, N, variants, cd, flow, ref_keys, by
    kind, src, work, N = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    variants = json.loads(sys.argv[5])
    work.mkdir(parents=True, exist_ok=True)
    if kind == "syn":
        cd = sp.case_dir(work, N); flow = sb.Flow(); ref_keys = None
        tr = np.load(sb.SRC / "truth" / "truth.npz"); kk = tr["frame"] <= N
        by = {int(f): (tr["pos_mm"][kk][tr["frame"][kk] == f], tr["particle_id"][kk][tr["frame"][kk] == f], cKDTree(tr["pos_mm"][kk][tr["frame"][kk] == f])) for f in np.unique(tr["frame"][kk])}

    def prepare(dst, over_ptv, over_track):
        if dst.exists(): shutil.rmtree(dst)
        (dst / "res").mkdir(parents=True)
        shutil.copytree(src / "cal", dst / "cal")
        shutil.copytree(src / "res" / "run.zarr", dst / "res" / "run.zarr")
        os.symlink(src / "img", dst / "img")
        if (src / "man_ori.dat").exists(): shutil.copy(src / "man_ori.dat", dst / "man_ori.dat")
        raw = yaml.safe_load((src / "parameters_Run1.yaml").read_text())
        raw["sequence"]["first"], raw["sequence"]["last"] = 1, N
        raw["pft_version"]["Existing_Target"] = 1
        raw.setdefault("ptv", {}).update(over_ptv)
        raw["track"].update({"shared_correction": True, **over_track})
        (dst / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        return dst / "parameters_Run1.yaml"

    hdr = False
    for name, v in variants.items():
        dst = work / name
        yp = prepare(dst, v.get("ptv", {}), v.get("track", {}))
        exp = sb._experiment(dst, yp, 1, N)
        cwd = os.getcwd(); os.chdir(dst)
        try:
            t0 = time.perf_counter(); ptv.py_sequence_loop(exp); t_match = time.perf_counter() - t0
            t1 = time.perf_counter(); run_tracking_plugin("two_phase", exp); t_track = time.perf_counter() - t1
        finally:
            os.chdir(cwd)
        st = RunStore(str(dst / "res" / "run.zarr"), mode="r")
        a = arrays_from_store(st, "ptv_is", 1, N, 4)
        frame, tid, pos, blob = a["frame"], a["tid"], np.asarray(a["pos"], float), a["blob"]
        nc = (blob >= 0).sum(1)
        if kind == "syn":
            ev = sb.evaluate_pred(cd, {"trajid": tid, "frame": frame, "pos": pos}, flow, ref_keys=ref_keys)
            if ref_keys is None and "_keys" in ev: ref_keys = ev["_keys"]
            ev.pop("_keys", None)
            g0, _, off = sb._match(by, frame, pos, 0.6); bias = np.median(off[g0 >= 0], axis=0)
            gid, _, _ = sb._match(by, frame, pos - bias, 0.4)
            pt = {(int(gid[i]), int(frame[i])): i for i in np.flatnonzero(gid >= 0)}
            sw = sum(1 for (p, f), i in pt.items() if (p, f + 1) in pt and tid[i] != tid[pt[p, f + 1]])
            g = lambda k, d=4: "  -   " if ev.get(k) is None else f"{ev[k]:.{d}f}"
            if not hdr:
                print(f"{'variant':22s} {'yield':>6s} {'ghost%':>6s} {'linkP':>6s} {'velErr':>6s} {'vErrCom':>7s} {'vErrXtr':>7s} {'posErr':>6s} {'idSw':>5s} {'3cam%':>5s} {'tMatch':>6s} {'tTrack':>6s}", flush=True); hdr = True
            print(f"{name:22s} {g('yield')} {100 * ev['ghost_point_frac']:6.2f} {g('raw_link_precision')} {g('vel_err_rel')} {g('vel_err_common'):>7s} {g('vel_err_extra'):>7s} {g('pos_err_mm')} {sw:5d} {100 * (nc == 3).mean():5.1f} {t_match:6.1f} {t_track:6.1f}", flush=True)
        else:
            u, inv = np.unique(tid, return_inverse=True); ln = np.bincount(inv)[inv]
            o = np.lexsort((frame, tid)); t_s, f_s, p_s = tid[o], frame[o], pos[o]
            ok = (t_s[:-2] == t_s[2:]) & (f_s[2:] - f_s[:-2] == 2)
            rough = np.linalg.norm(p_s[:-2][ok] - 2 * p_s[1:-1][ok] + p_s[2:][ok], axis=1)
            cpar, vpar, cals, _ = load_parameters(dst / "parameters_Run1.yaml")
            res = []
            for f in range(1, N + 1, max(N // 40, 1)):
                r = np.flatnonzero((frame == f) & (blob >= 0).all(1))
                if len(r):
                    v_ = loco_residuals(np.stack([a["centre"][r, c] for c in range(4)], axis=1), cpar, cals, vpar); res.append(v_[np.isfinite(v_)])
            res = np.concatenate(res)
            if not hdr:
                print(f"{'variant':22s} {'pts/fr':>7s} {'3cam%':>5s} {'single%':>7s} {'>=50%':>6s} {'>=200%':>6s} {'ptsIn50/fr':>10s} {'meanLen':>7s} {'rough50':>8s} {'rough90':>8s} {'LOCOmed':>7s} {'LOCO90':>6s} {'tMatch':>6s} {'tTrack':>6s}", flush=True); hdr = True
            print(f"{name:22s} {len(frame) / N:7.1f} {100 * (nc == 3).mean():5.1f} {100 * (ln == 1).mean():7.1f} {100 * (ln >= 50).mean():6.1f} {100 * (ln >= 200).mean():6.1f} {(ln >= 50).sum() / N:10.1f} {ln.mean():7.1f} {np.median(rough):8.4f} {np.percentile(rough, 90):8.4f} {np.median(res):7.3f} {np.percentile(res, 90):6.3f} {t_match:6.1f} {t_track:6.1f}", flush=True)
        shutil.rmtree(dst, ignore_errors=True)


if __name__ == '__main__':
    main()
