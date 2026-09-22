"""Can flowtracks repair make two-phase beat trackcorr on wp1?

Builds Trajectory objects from (a) trackcorr+postprocess linkage and
(b) two-phase links, runs flowtracks.repair_trajectories, rescores
link recall / claims / chains / smoothness vs the 3dptv reference.

Run from repo root:
    uv run python -u scripts/repair_compare_wp1.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, r"C:\Users\alex\projects\postptv")

import numpy as np  # noqa: E402
from verify_same_trajectories import DS, REF  # noqa: E402

WORK = (Path(__file__).resolve().parent.parent / "scratch" / "_wp1_ab").resolve()
FIRST, LAST = 100001, 100010
SFIRST, SLAST = 100003, 100010
NCAMS = 4


def restore():
    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, NCAMS + 1):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets")


def to_traj_list(chains, positions, frames, tid0=0):
    from flowtracks.trajectory import Trajectory
    out = []
    for k, c in enumerate(chains):
        fr = [frames[t] for (t, r) in c]
        ps = np.array([positions[t][r] for (t, r) in c])
        v = np.zeros_like(ps)
        if len(ps) > 2:
            v[:-1] = ps[1:] - ps[:-1]
            v[-1] = v[-2]
        elif len(ps) == 2:
            v[:] = ps[1] - ps[0]
        out.append(Trajectory(ps, v, np.array(fr, dtype=int), tid0 + k))
    return out


def chains_from_linkage(resdir: Path):
    P = {}
    for f in range(FIRST, LAST + 1):
        lines = (resdir / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        P[f] = ([int(l.split()[0]) for l in lines[1: n + 1]],
                [int(l.split()[1]) for l in lines[1: n + 1]])
    nxt = {}
    for f in range(FIRST, LAST + 1):
        for i, v in enumerate(P[f][1]):
            if v >= 0:
                # find target frame (gap-aware via reciprocation)
                for g in range(f + 1, LAST + 1):
                    if v < len(P[g][0]) and P[g][0][v] == i:
                        nxt[(f - FIRST, i)] = (g - FIRST, v)
                        break
    tgt = set(nxt.values())
    chains = []
    for s in sorted(set(nxt) - tgt):
        c, k = [s], s
        while k in nxt:
            k = nxt[k]
            c.append(k)
        chains.append(c)
    # singletons: unlinked particles
    seen = set(n for c in chains for n in c)
    for f in range(FIRST, LAST + 1):
        n = len(P[f][0])
        for i in range(n):
            if (f - FIRST, i) not in seen:
                chains.append([(f - FIRST, i)])
    return chains


def score_trajs(trajs, positions, frames, ref_prev, ref_total, tag):
    exact = claims = consec = 0
    mp = {}
    for t in trajs:
        fr = list(t.time())
        ps = np.asarray(t.pos())
        for a in range(len(fr) - 1):
            f0, f1 = int(fr[a]), int(fr[a + 1])
            if not (SFIRST <= f1 <= SLAST):
                continue
            if f1 - f0 != 1:
                continue
            t0i, t1 = frames.index(f0), frames.index(f1)
            # row lookup by position identity
            d0 = np.linalg.norm(positions[t0i] - ps[a], axis=1)
            d1 = np.linalg.norm(positions[t1] - ps[a + 1], axis=1)
            r0, r1 = int(np.argmin(d0)), int(np.argmin(d1))
            if d0[r0] > 1e-6 or d1[r1] > 1e-6:
                continue
            claims += 1
            consec += 1
            mp[(t1, r1)] = (t0i, r0)
            rp = ref_prev[t1]
            if r1 < len(rp) and rp[r1] >= 0 and rp[r1] == r0:
                exact += 1
    accs = []
    for (t1, r1), (t0i, r0) in mp.items():
        if (t0i, r0) in mp and t0i - mp[(t0i, r0)][0] == 1:
            t_1, r_1 = mp[(t0i, r0)]
            accs.append(float(np.linalg.norm(
                (positions[t1][r1] - positions[t0i][r0]) -
                (positions[t0i][r0] - positions[t_1][r_1]))))
    accs = np.array(accs)
    lens = np.array([len(t.time()) for t in trajs])
    print(f"{tag}: recall={exact / ref_total:.2%} claims~{claims} "
          f"trajs={len(trajs)} lenmean={lens.mean():.2f} "
          f"accmean={accs.mean():.3f} rough={(accs > 1.9).sum()}",
          flush=True)


def main():
    import os

    from flowtracks.repair import repair_trajectories

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    restore()
    os.chdir(WORK)
    try:
        pm = ParameterManager()
        pm.from_yaml(WORK / "parameters_Run_dacc19.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0)
        tr.full_forward()
        tr.full_backward()
        tr.postprocess(cold_start=True, reciprocity=True,
                       gap_relinking=True, max_gap=2)
    finally:
        os.chdir(Path.cwd())

    frames = list(range(FIRST, LAST + 1))
    positions = []
    for f in frames:
        lines = (WORK / "res" / f"rt_is.{f}").read_text().splitlines()
        n = int(lines[0])
        positions.append(np.array([[float(v) for v in l.split()[1:4]]
                                   for l in lines[1: n + 1]]))
    ref_prev = {}
    for k, f in enumerate(frames):
        lines = (REF / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        ref_prev[k] = [int(l.split()[0]) for l in lines[1: n + 1]]
    ref_total = sum(sum(1 for p in ref_prev[k] if p >= 0)
                    for k, f in enumerate(frames) if SFIRST <= f <= SLAST)

    tc_chains = chains_from_linkage(WORK / "res")
    tc_trajs = to_traj_list(tc_chains, positions, frames)
    score_trajs(tc_trajs, positions, frames, ref_prev, ref_total,
                "trackcorr+post raw")
    t0 = time.perf_counter()
    tc_rep, tc_report = repair_trajectories(tc_trajs, long_track=5)
    print(f"repair wall: {time.perf_counter() - t0:.1f}s "
          f"({len(tc_trajs)} -> {len(tc_rep)} trajs) report={tc_report}",
          flush=True)
    score_trajs(tc_rep, positions, frames, ref_prev, ref_total,
                "trackcorr+post repaired")

    # two-phase
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )
    cpar = ControlPar.from_file(str(WORK / "parameters" / "ptv.par"))
    cals = [Calibration.from_file(
        str(WORK / "cal" / f"cam_{c}.tif.ori"),
        str(WORK / "cal" / f"cam_{c}.tif.addpar")) for c in range(1, NCAMS + 1)]
    leaves = []
    for P in positions:
        xy = np.full((len(P), NCAMS * 2), np.nan)
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
                m = img_coord_batch(np.array([[x, y, z]]), cals[ci], cpar.mm)[0]
                out[i, 2 * ci] = m[0] / cpar.pix_x + cpar.imx / 2
                out[i, 2 * ci + 1] = cpar.imy / 2 - m[1] / cpar.pix_y
        return np.nan_to_num(out)

    cfg = TwoPhaseTrackerConfig(v_max=2.0, leaf_weight=1.0, use_velocity=True,
                                cost_mode="projected", max_gap=1)
    links = TwoPhaseTracker(cfg).track_frames(positions, leaves,
                                              project_fn=project_fn)
    nxt = {}
    for (t0i, r0, t1, r1) in links:
        if t1 - t0i == 1:
            nxt[(t0i, r0)] = (t1, r1)
    tgt = set(nxt.values())
    tp_chains = []
    for s in sorted(set(nxt) - tgt):
        c, k = [s], s
        while k in nxt:
            k = nxt[k]
            c.append(k)
        tp_chains.append(c)
    tp_trajs = to_traj_list(tp_chains, positions, frames, tid0=10_000_000)
    score_trajs(tp_trajs, positions, frames, ref_prev, ref_total,
                "two-phase raw")
    t0 = time.perf_counter()
    tp_rep, tp_report = repair_trajectories(tp_trajs, long_track=5)
    print(f"repair wall: {time.perf_counter() - t0:.1f}s "
          f"({len(tp_trajs)} -> {len(tp_rep)} trajs) report={tp_report}",
          flush=True)
    score_trajs(tp_rep, positions, frames, ref_prev, ref_total,
                "two-phase repaired")


if __name__ == "__main__":
    main()
