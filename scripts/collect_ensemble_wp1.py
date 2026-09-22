"""Ensemble adjudication on wp1: four diverse engines vote per link.

Engines (all on identical pristine inputs):
  parity  : trackcorr loser_retry=0 (existing scratch/_wp1_ab/res)
  default : trackcorr loser_retry=1 fwd+bwd (fresh clone)
  4be     : four-frame best estimate fwd (fresh clone)
  2p      : two-phase+vel v_max=2.0 (arrays -> ptv files)

Vote per (frame,row): unanimous / majority / union / intersection sets,
each scored vs the 3dptv reference. Answers: is there a ghost-free
high-precision subset, and does voting beat the best single engine?

Run from repo root (detached, ~10 min):
    uv run python -u scripts/collect_ensemble_wp1.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
SRC = REPO / "scratch" / "_wp1_ab"
FIRST, LAST = 100001, 100010
SCORE_FIRST, SCORE_LAST = 100003, 100010
NCAMS = 4


def setup_clone(tag: str) -> Path:
    work = REPO / "scratch" / f"_ens_{tag}"
    if work.exists():
        shutil.rmtree(work)
    (work / "img_3dptv").mkdir(parents=True)
    (work / "res").mkdir(parents=True)
    shutil.copytree(SRC / "cal", work / "cal")
    shutil.copytree(SRC / "parameters", work / "parameters")
    shutil.copy(SRC / "parameters_Run_dacc19.yaml",
                work / "parameters_Run1.yaml")
    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", work / "res" / f"rt_is.{f}")
        for cam in range(1, NCAMS + 1):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                work / "img_3dptv" / f"Cam{cam}.{f}_targets")
    return work


def run_corr(work: Path, lr: int):
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=lr, cold_start_neighbour=0)
        tr.full_forward()
        tr.full_backward()
    finally:
        os.chdir(old)


def run_4be(work: Path):
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / "parameters_Run1.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        Tracker(cpar, vpar, track_par, spar, cals,
                default_naming).full_forward_4be()
    finally:
        os.chdir(old)


def run_2p(work: Path) -> Path:
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker, TwoPhaseTrackerConfig)

    cpar = ControlPar.from_file(str(work / "parameters" / "ptv.par"))
    cals = [Calibration.from_file(
        str(work / "cal" / f"cam_{c}.tif.ori"),
        str(work / "cal" / f"cam_{c}.tif.addpar"))
        for c in range(1, NCAMS + 1)]
    frames = list(range(FIRST, LAST + 1))

    def proj(P):
        P = np.asarray(P, dtype=np.float64)
        out = np.full((len(P), NCAMS * 2), np.nan)
        for i, (x, y, z) in enumerate(P):
            for ci in range(NCAMS):
                m = img_coord_batch(np.array([[x, y, z]]), cals[ci],
                                    cpar.mm)[0]
                out[i, 2 * ci] = m[0] / cpar.pix_x + cpar.imx / 2
                out[i, 2 * ci + 1] = cpar.imy / 2 - m[1] / cpar.pix_y
        return np.nan_to_num(out)

    positions, leaves = [], []
    for f in frames:
        lines = (work / "res" / f"rt_is.{f}").read_text().splitlines()
        n = int(lines[0])
        P = np.array([[float(v) for v in l.split()[1:4]]
                      for l in lines[1: n + 1]])
        positions.append(P)
        leaves.append(proj(P))
    cfg = TwoPhaseTrackerConfig(v_max=2.0, leaf_weight=1.0, use_velocity=True,
                                cost_mode="projected", max_gap=1)
    links = TwoPhaseTracker(cfg).track_frames(positions, leaves,
                                              project_fn=proj)
    nrows = [len(P) for P in positions]
    tprev = [np.full(n, -1) for n in nrows]
    tnxt = [np.full(n, -1) for n in nrows]
    for t0, r0, t1, r1 in links:
        if 0 <= t0 < len(frames) and 0 <= t1 < len(frames):
            if r0 < nrows[t0] and r1 < nrows[t1]:
                tnxt[t0][r0] = r1
                tprev[t1][r1] = r0
    out = work / "res_2p"
    out.mkdir(exist_ok=True)
    for k, f in enumerate(frames):
        with open(out / f"ptv_is.{f}", "w") as fh:
            fh.write(f"{nrows[k]}\n")
            for i in range(nrows[k]):
                x, y, z = positions[k][i]
                fh.write(f"{int(tprev[k][i]):4d} {int(tnxt[k][i]):4d} "
                         f"{x:10.3f} {y:10.3f} {z:10.3f}\n")
    return out


def read_prev(res: Path, f: int):
    lines = (res / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


def main():
    print("collecting engines...", flush=True)
    w_def = setup_clone("lr1")
    run_corr(w_def, 1)
    w_4be = setup_clone("4be")
    run_4be(w_4be)
    w_2p = setup_clone("2p")
    res_2p = run_2p(w_2p)

    engs = {
        "parity": SRC / "res",
        "default": w_def / "res",
        "4be": w_4be / "res",
        "2p": res_2p,
    }
    P = {name: {f: read_prev(res, f) for f in range(FIRST, LAST + 1)}
         for name, res in engs.items()}
    R = {f: read_pn_ref(f) for f in range(FIRST, LAST + 1)}

    names = list(engs)
    single = {}
    for name in names:
        tot = mt = 0
        for f in range(SCORE_FIRST, SCORE_LAST + 1):
            r, t = R[f], P[name][f]
            for i in range(min(len(r), len(t))):
                if r[i] >= 0:
                    tot += 1
                    mt += t[i] == r[i]
        single[name] = mt / tot
        print(f"{name:8} recall={mt / tot:.2%} ({mt}/{tot})", flush=True)

    # votes
    una_t = una_m = maj_t = maj_m = uni_m = inter_m = 0
    contested = agree_ref = 0
    win_counts = {n: 0 for n in names}
    for f in range(SCORE_FIRST, SCORE_LAST + 1):
        r = R[f]
        T = {n: P[n][f] for n in names}
        n = min(len(r), *(len(T[x]) for x in names))
        for i in range(n):
            if r[i] < 0:
                continue
            votes = [T[x][i] for x in names]
            vals = [v for v in votes if v is not None]
            top = max(set(vals), key=vals.count)
            cnt = vals.count(top)
            if cnt == len(names):
                una_t += 1
                una_m += top == r[i]
            if cnt >= 3:
                maj_t += 1
                maj_m += top == r[i]
            if any(v == r[i] for v in vals):
                uni_m += 1
            if all(v == r[i] for v in vals):
                inter_m += 1
            if not all(v == vals[0] for v in vals):
                contested += 1
                agree_ref += top == r[i]
                for x in names:
                    if T[x][i] == r[i]:
                        win_counts[x] += 1
    tot = sum(1 for f in range(SCORE_FIRST, SCORE_LAST + 1)
              for p in R[f] if p >= 0)
    print(f"\nref links scored: {tot}", flush=True)
    print(f"unanimous(4/4): n={una_t} precision={una_m / una_t:.2%}",
          flush=True)
    print(f"majority(>=3) : n={maj_t} precision={maj_m / maj_t:.2%} "
          f"recall={maj_m / tot:.2%}", flush=True)
    print(f"union (any engine right): recall={uni_m / tot:.2%}", flush=True)
    print(f"intersection (all right): recall={inter_m / tot:.2%}", flush=True)
    print(f"contested links: {contested} ({contested / tot:.1%}); "
          f"majority right: {agree_ref / contested:.1%}", flush=True)
    print(f"wins on contested: {win_counts}", flush=True)


def read_pn_ref(f: int):
    lines = (REF / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


if __name__ == "__main__":
    main()
