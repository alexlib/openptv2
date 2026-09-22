"""Cascade driver: trackcorr base + additive two-phase merge, scored.

Modes:
  synth : S-scenes with known truth (trackcorr fwd+bwd, then merge).
  wp1   : scratch work dir vs 3dptv reference (merged ptv to res_cascade/).

Run from repo root:
    uv run python -u scripts/run_cascade.py --mode synth [--reps N]
    uv run python -u scripts/run_cascade.py --mode wp1
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from cascade_track import merge_links, write_ptv_is  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
NCAMS = 4


# ------------------------------------------------------------------ trackcorr

def run_trackcorr(work: Path, params_file: str, backward: bool = True):
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    old = os.getcwd()
    os.chdir(work)
    try:
        pm = ParameterManager()
        pm.from_yaml(work / params_file)
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        for cam_id, short in enumerate(pm.get_target_filenames()):
            spar.set_img_base_name(cam_id, str(Path(short).resolve()) + ".")
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0)
        tr.full_forward()
        if backward:
            tr.full_backward()
    finally:
        os.chdir(old)


def read_pn(res: Path, frame: int):
    lines = (res / f"ptv_is.{frame}").read_text().splitlines()
    n = int(lines[0])
    prev, nxt = [], []
    for line in lines[1: n + 1]:
        p = line.split()
        prev.append(int(p[0]))
        nxt.append(int(p[1]))
    return prev, nxt


# ------------------------------------------------------------------ two-phase

def run_two_phase_arrays(positions, leaves, project_fn, v_max: float,
                         max_gap: int):
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )

    cfg = TwoPhaseTrackerConfig(v_max=v_max, leaf_weight=1.0,
                                use_velocity=True, cost_mode="projected",
                                max_gap=max_gap)
    return TwoPhaseTracker(cfg).track_frames(positions, leaves,
                                             project_fn=project_fn)


# ------------------------------------------------------------------ synth mode

def mode_synth(reps: int):
    from bench_two_phase import V_MAX, make_leaves, project_points, score_links
    from synth_crossing import (
        FIRST,
        NF,
        load_optics,
        make_truth,
        setup_work,
        write_scene,
    )

    cpar, cals = load_optics()
    scenarios = [
        ("S1 head-on", {"v": 0.5}),
        ("S8 kick", {"v": 0.5, "maneuver": "kick"}),
        ("S11 kick+noise", {"v": 0.5, "maneuver": "kick", "noise": 0.07}),
        ("S4 head-on+noise", {"v": 0.5, "noise": 0.07}),
        ("S13 gap-occl", {"v": 0.5, "dropout": True}),
    ]
    print(f"{'scenario':16}{'base':>9}{'merged':>9}{'+exact':>8}",
          flush=True)
    for sname, kw in scenarios:
        noisy = kw.get("noise", 0.0) > 0
        nreps = reps if noisy else 3
        base_r, merg_r, adds = [], [], []
        for rep in range(nreps):
            work = REPO / "scratch" / "_synth_casc"
            setup_work(work)
            truth = make_truth(seed=1000 + rep, **kw)
            rows_pf = write_scene(work, truth, cpar, cals)
            run_trackcorr(work, "parameters_Run1.yaml", backward=True)
            # base score from corr ptv
            corr_prev, corr_next, nrows = [], [], []
            for f in range(NF):
                p, x = read_pn(work / "res", FIRST + f)
                corr_prev.append(p)
                corr_next.append(x)
                nrows.append(len(p))
            from bench_two_phase import _pid_frames
            exp = set()
            for pf in _pid_frames(rows_pf, NF, FIRST):
                for a, b in zip(pf, pf[1:]):
                    exp.add((a[0], a[1], b[0], b[1]))
            got_base = {(t0, r0, t0 + 1, r1)
                        for t0 in range(NF - 1) for r0, r1 in
                        enumerate(corr_next[t0]) if r1 >= 0}
            base_r.append(len(exp & got_base) / len(exp))
            # two-phase on the same positions
            positions, leaves = [], []
            for f in range(NF):
                lines = (work / "res" / f"rt_is.{FIRST + f}"
                         ).read_text().splitlines()
                n = int(lines[0])
                P = np.array([[float(v) for v in l.split()[1:4]]
                              for l in lines[1: n + 1]])
                positions.append(P)
                leaves.append(project_points(P, cpar, cals))
            tp = run_two_phase_arrays(
                positions, leaves,
                lambda P: project_points(np.asarray(P), cpar, cals),
                V_MAX, 2)
            mp, mn, nadd, added = merge_links(corr_prev, corr_next, tp,
                                                nrows)
            got_m = {(t0, r0, t0 + 1, r1)
                     for t0 in range(NF - 1) for r0, r1 in enumerate(mn[t0])
                     if r1 >= 0} | set(added)
            merg_r.append(len(exp & got_m) / len(exp))
            adds.append(nadd)
        print(f"{sname:16}{np.mean(base_r):>8.1%}{np.mean(merg_r):>8.1%}  "
              f"+{np.mean(adds):.1f} links", flush=True)


# ------------------------------------------------------------------ wp1 mode

def mode_wp1():
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar

    work = (REPO / "scratch" / "_wp1_ab").resolve()
    first, last = 100001, 100010
    frames = list(range(first, last + 1))
    out = work / "res_cascade"

    for f in frames:
        shutil.copyfile(REF / f"rt_is.{f}", work / "res" / f"rt_is.{f}")
        for cam in range(1, NCAMS + 1):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                work / "img_3dptv" / f"Cam{cam}.{f}_targets")

    run_trackcorr(work, "parameters_Run_dacc19.yaml", backward=True)

    cpar = ControlPar.from_file(str(work / "parameters" / "ptv.par"))
    cals = [Calibration.from_file(
        str(work / "cal" / f"cam_{c}.tif.ori"),
        str(work / "cal" / f"cam_{c}.tif.addpar"))
        for c in range(1, NCAMS + 1)]

    positions, leaves = [], []
    for f in frames:
        lines = (work / "res" / f"rt_is.{f}").read_text().splitlines()
        n = int(lines[0])
        P = np.array([[float(v) for v in l.split()[1:4]]
                      for l in lines[1: n + 1]])
        positions.append(P)
        xy = np.full((n, NCAMS * 2), np.nan)
        for i, (x, y, z) in enumerate(P):
            for ci in range(NCAMS):
                m = img_coord_batch(np.array([[x, y, z]]), cals[ci],
                                    cpar.mm)[0]
                xy[i, 2 * ci] = m[0] / cpar.pix_x + cpar.imx / 2
                xy[i, 2 * ci + 1] = cpar.imy / 2 - m[1] / cpar.pix_y
        leaves.append(np.nan_to_num(xy))

    def project_fn(pred):
        pred = np.asarray(pred, dtype=np.float64)
        out = np.full((len(pred), NCAMS * 2), np.nan)
        for i, (x, y, z) in enumerate(pred):
            for ci in range(NCAMS):
                m = img_coord_batch(np.array([[x, y, z]]), cals[ci],
                                    cpar.mm)[0]
                out[i, 2 * ci] = m[0] / cpar.pix_x + cpar.imx / 2
                out[i, 2 * ci + 1] = cpar.imy / 2 - m[1] / cpar.pix_y
        return np.nan_to_num(out)

    tp = run_two_phase_arrays(positions, leaves, project_fn, v_max=2.0,
                              max_gap=1)
    corr_prev, corr_next, nrows = [], [], []
    for f in frames:
        p, x = read_pn(work / "res", f)
        corr_prev.append(p)
        corr_next.append(x)
        nrows.append(len(p))
    mp, mn, nadd, added = merge_links(corr_prev, corr_next, tp, nrows)
    write_ptv_is(out, frames, mp, mn, positions)

    ref_total = match_base = match_merg = 0
    # exact/extra split by span using merge-time records
    ref_prev_all = {}
    for k, f in enumerate(frames):
        lines = (REF / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        ref_prev_all[k] = [int(l.split()[0]) for l in lines[1: n + 1]]
    span_stats = {"consec": [0, 0], "gap": [0, 0]}  # [exact, extra]
    for t0, r0, t1, r1 in added:
        f1 = frames[t1]
        if not (100003 <= f1 <= 100010):
            continue
        rp = ref_prev_all[t1]
        key = "gap" if t1 - t0 > 1 else "consec"
        if r1 < len(rp) and rp[r1] >= 0 and rp[r1] == r0:
            span_stats[key][0] += 1
        else:
            span_stats[key][1] += 1
    print(f"added split: consecutive exact={span_stats['consec'][0]} "
          f"extra={span_stats['consec'][1]}; gap exact={span_stats['gap'][0]}"
          f" extra={span_stats['gap'][1]}", flush=True)

    ref_total = match_base = match_merg = 0
    for k, f in enumerate(frames):
        if not (100003 <= f <= 100010):
            continue
        lines = (REF / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        rp = [int(l.split()[0]) for l in lines[1: n + 1]]
        ref_total += sum(1 for p in rp if p >= 0)
        for i in range(min(n, nrows[k])):
            if rp[i] >= 0:
                if corr_prev[k][i] == rp[i]:
                    match_base += 1
                if mp[k][i] == rp[i]:
                    match_merg += 1
    print(f"wp1 cascade: base={match_base}/{ref_total} "
          f"({match_base / ref_total:.2%}) merged={match_merg}/{ref_total} "
          f"({match_merg / ref_total:.2%}) +{nadd} links -> {out}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "wp1"], required=True)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()
    if args.mode == "synth":
        mode_synth(args.reps)
    else:
        mode_wp1()


if __name__ == "__main__":
    main()
