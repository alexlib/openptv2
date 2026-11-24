# ruff: noqa
"""Wide epipolar band on real detections (wp2) and on the rendered case with truth (syn1): new buffer vs the old fixed one.

    uv run python bench/limits/wide_band_real.py wp2|syn1 <frame> <eps0 list, comma> [--old] [--timeout SECONDS]

--old emulates the previous behaviour: the candidate buffer is capped at 4 * 20240 rows, so a full stage drops the
candidates of the targets visited last (now with the cap warning). Reports per eps0: quads, 3-camera points, candidates the
4-camera stage found, buffer rows, time, warnings, the share of camera-1 targets matched per x-quartile, and, on syn1,
how many quads are one true particle (truth from the render).
"""
import copy, importlib, sys, time, warnings
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb
from openptv2.storage import RunStore

cm = importlib.import_module("openptv2.algorithms.correspondences")
oc = importlib.import_module("openptv2.correspondences")
H = Path.home() / "Downloads/HiDImaging"
ROOTS = {"wp2": H / "CompleteTest-e2e-local/wp2/test", "syn1": H / "Synthetic-CompleteTest/syn1/test"}

def main():
    name, frame = sys.argv[1], int(sys.argv[2])
    eps_list = [float(x) for x in sys.argv[3].split(",")]
    old = "--old" in sys.argv
    root = ROOTS[name]
    exp = sb._experiment(root, root / "parameters_Run1.yaml", frame, frame)
    st = RunStore(str(root / "res" / "run.zarr"), mode="r")
    if old:
        cm.CANDIDATE_ROWS_PER_CAM = cm.NMAX
        cm.MAX_CANDIDATE_ROWS = 4 * cm.NMAX
    info = []
    orig = cm._run_matching_stage
    def spy(stage, ab, aa, p, c, size, nc, nm, **kw):
        out = orig(stage, ab, aa, p, c, size, nc, nm, **kw); info.append((nm, out[0], out[3])); return out
    cm._run_matching_stage = spy
    if name == "syn1":
        tr = np.load(sb.SRC / "truth" / "truth.npz"); m = tr["frame"] == frame
    print(f"[{name} frame {frame}{' OLD buffer' if old else ''}] default eps0 {exp.vpar.eps0}")
    print(f"{'eps0':>6s} {'time':>7s} {'quads':>6s} {'3cam':>6s} {'quad cands':>10s} {'buffer':>9s} {'matched per x-quartile of cam 1 (%)':>38s}  warnings" + ("   quads=one true particle" if name == "syn1" else ""))
    for eps0 in eps_list:
        dets = [st.read_targets(c, frame) for c in range(4)]
        vpar = copy.copy(exp.vpar); vpar.eps0 = eps0
        info.clear()
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pos, ids = oc.match_frame_correspondences(dets, exp.cpar, exp.cals, vpar)
        dt = time.perf_counter() - t
        ids = np.asarray(ids)[:, :4]
        nc = (ids >= 0).sum(1)
        q = [x for x in info if x[0].startswith("four")][0]
        # camera-1 targets by x quartile: share that ended up in a quad or triplet
        x0 = np.array([d.x for d in dets[0]]); order = np.argsort(x0)
        quart = np.empty(len(x0), int); quart[order] = (np.arange(len(x0)) * 4 // len(x0))
        matched0 = np.zeros(len(x0), bool); matched0[ids[ids[:, 0] >= 0, 0]] = True
        qshare = [100 * matched0[quart == k].mean() for k in range(4)]
        wtxt = ",".join(sorted({str(x.message)[:34] for x in w})) if w else "-"
        extra = ""
        if name == "syn1":
            px = tr["pixel_xy_per_cam"][m]; good = 0; quads = ids[nc == 4]
            owner = []
            for c in range(4):
                xy = np.array([[d.x, d.y] for d in dets[c]]); tree = cKDTree(xy)
                ok = np.isfinite(px[:, 2 * c]); d, j = tree.query(px[ok, 2 * c: 2 * c + 2], distance_upper_bound=3.0)
                own = np.full(len(xy), -1); own[j[np.isfinite(d)]] = np.flatnonzero(ok)[np.isfinite(d)]; owner.append(own)
            for r in quads:
                o = {owner[c][r[c]] for c in range(4)}
                good += (len(o) == 1 and -1 not in o)
            extra = f"   {good}/{len(quads)} ({100 * good / max(len(quads), 1):.0f} %)"
        print(f"{eps0:6.3f} {dt:7.2f} {(nc == 4).sum():6d} {(nc == 3).sum():6d} {q[1]:10d} {q[2]:9d} {'  '.join(f'{v:5.1f}' for v in qshare):>38s}  {wtxt}{extra}", flush=True)

main()
