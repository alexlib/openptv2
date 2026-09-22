"""Diagnose the wp1 ppar regression: which fixed-good links does ppar break?

Saves fixed-forward outputs to res_fixedfwd/, runs ppar-forward, then
profiles ppar-misses-that-fixed-hits (step/acc/crowd/frame) to name the
mechanism instead of guessing it.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from verify_same_trajectories import DS, REF  # noqa: E402

WORK = (Path(__file__).resolve().parent.parent / "scratch" / "_wp1_ab"
        ).resolve()
FIRST, LAST = 100001, 100010


def read_pn(d: Path, f: int):
    lines = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


def read_xyz(f: int):
    lines = (REF / f"rt_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()[1:4]] for l in lines[1: n + 1]])


def main():
    import os

    from phase_scheduler import run_adaptive_pp

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, 5):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets")
    old = os.getcwd()
    os.chdir(WORK)
    try:
        pm = ParameterManager()
        pm.from_yaml(WORK / "parameters_Run_dacc19.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0)
        tr.full_forward()
    finally:
        os.chdir(old)
    snap = WORK / "res_fixedfwd"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(WORK / "res", snap,
                    ignore=shutil.ignore_patterns("rt_is.*"))
    print("fixed-forward saved to res_fixedfwd", flush=True)

    run_adaptive_pp(WORK, 0, 0, hi=1.5, mode="vel", first=FIRST,
                      last=LAST, yaml="parameters_Run_dacc19.yaml")

    # per-step link counts
    print(f"{'frame':>8}{'ref':>7}{'fixed':>7}{'ppar':>7}", flush=True)
    for f in range(100002, 100011):
        r = sum(1 for p in read_pn(REF, f) if p >= 0)
        a = sum(1 for p in read_pn(snap, f) if p >= 0)
        b = sum(1 for p in read_pn(WORK / "res", f) if p >= 0)
        print(f"{f:>8}{r:>7}{a:>7}{b:>7}", flush=True)

    # regression features: ref links fixed-hits but ppar-misses
    from scipy.spatial import cKDTree
    XYZ = {f: read_xyz(f) for f in range(FIRST, LAST + 1)}
    trees = {f: cKDTree(XYZ[f]) for f in XYZ}
    reg, rec = [], []
    for f in range(100003, 100011):
        r = read_pn(REF, f)
        a = read_pn(snap, f)
        b = read_pn(WORK / "res", f)
        rp = read_pn(REF, f - 1)
        for i in range(min(len(r), len(a), len(b))):
            if r[i] < 0:
                continue
            fa = a[i] == r[i]
            fb = b[i] == r[i]
            if fa and not fb:
                p = r[i]
                step = float(np.linalg.norm(XYZ[f][i] - XYZ[f - 1][p]))
                crowd = len(trees[f].query_ball_point(XYZ[f][i], r=1.9)) - 1
                reg.append((step, crowd, f))
            elif fa and fb:
                p = r[i]
                step = float(np.linalg.norm(XYZ[f][i] - XYZ[f - 1][p]))
                rec.append(step)
    reg = np.array(reg, dtype=object)
    print(f"\nregressions (fixed-hits, ppar-misses): {len(reg)}", flush=True)
    if len(reg):
        st = np.array([x[0] for x in reg], dtype=float)
        cr = np.array([x[1] for x in reg], dtype=float)
        print(f"step mean={st.mean():.3f} p90={np.percentile(st, 90):.3f} "
              f"max={st.max():.3f} crowd_mean={cr.mean():.2f}", flush=True)
        import collections
        print("per-frame:", dict(sorted(collections.Counter(
            x[2] for x in reg).items())), flush=True)
        print(f"control (both hit) step mean={np.mean(rec):.3f}", flush=True)


if __name__ == "__main__":
    main()
