"""Benchmark trackcorr vs two-phase on wp1: speed + accuracy + trajectory quality.

Phases (each timed):
  A. trackcorr fwd+bwd (dacc=1.9 parity config) -> links/chains/smoothness
  B. two-phase+vel (v_max=2.0, gap=1) -> links/chains/smoothness
  C. trackcorr + postprocess (reciprocity + cold start + gap relink) -> delta
Smoothness = |acc| distribution over consecutive link triples vs reference.

Run from repo root:
    uv run python -u scripts/bench_wp1_trackers.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from verify_same_trajectories import (  # noqa: E402
    DS,
    REF,
    compare_links,
    compare_tracks,
)

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


def read_prev(d: Path, f: int):
    lines = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return [int(l.split()[0]) for l in lines[1: n + 1]]


def read_xyz(d: Path, f: int):
    lines = (d / f"rt_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()[1:4]] for l in lines[1: n + 1]])


def acc_stats(link_dir: Path, tag: str):
    """|acc| over consecutive triples following test linkage."""
    accs, n_links = [], 0
    XYZ = {f: read_xyz(WORK / "res", f) for f in range(FIRST, LAST + 1)}
    for f in range(SFIRST, SLAST + 1):
        pv = read_prev(link_dir, f)
        pp = read_prev(link_dir, f - 1)
        X0, X1, X2 = XYZ[f - 2], XYZ[f - 1], XYZ[f]
        for i in range(min(len(pv), len(X2))):
            p = pv[i]
            if p < 0 or p >= len(X1):
                continue
            q = pp[p] if p < len(pp) else -1
            if q < 0 or q >= len(X0):
                continue
            n_links += 1
            accs.append(float(np.linalg.norm((X2[i] - X1[p]) - (X1[p] - X0[q]))))
    accs = np.array(accs)
    print(f"[{tag}] triples={len(accs)} acc mean={accs.mean():.3f} "
          f"p90={np.percentile(accs, 90):.3f} "
          f"max={accs.max():.3f} rough(>1.9)={(accs > 1.9).sum()}",
          flush=True)


def chain_stats(links, positions, frames, tag: str):
    nxt = {}
    for (t0, r0, t1, r1) in links:
        nxt[(t0, r0)] = (t1, r1)
    tgt = set(nxt.values())
    chains, visited = [], set()
    for s in sorted(set(nxt) - tgt):
        c, k = [s], s
        while k in nxt and nxt[k] not in visited:
            visited.add(k)
            k = nxt[k]
            c.append(k)
        chains.append(c)
    lens = np.array([len(c) for c in chains])
    print(f"[{tag}] chains={len(chains)} len mean={lens.mean():.2f} "
          f"median={np.median(lens):.0f} max={lens.max()} "
          f"fullspan10={(lens >= 10).sum()}", flush=True)


def main():
    import os
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c
    from openptv2.tracker import Tracker, default_naming

    # ---- A. trackcorr ----
    restore()
    os.chdir(WORK)
    try:
        t0 = time.perf_counter()
        pm = ParameterManager()
        pm.from_yaml(WORK / "parameters_Run_dacc19.yaml")
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
        tr = Tracker(cpar, vpar, track_par, spar, cals, default_naming,
                     loser_retry=0, cold_start_neighbour=0)
        tr.full_forward()
        tr.full_backward()
        t_tc = time.perf_counter() - t0
    finally:
        os.chdir(Path.cwd())
    print(f"[trackcorr fwd+bwd] wall={t_tc:.1f}s", flush=True)
    compare_links(WORK, SFIRST, SLAST, "trackcorr")
    compare_tracks(WORK)
    acc_stats(WORK / "res", "trackcorr-smooth")
    acc_stats(REF, "ref-smooth")

    # ---- C. postprocess on trackcorr output ----
    os.chdir(WORK)
    try:
        t0 = time.perf_counter()
        stats = tr.postprocess(cold_start=True, reciprocity=True,
                               gap_relinking=True, max_gap=2)
        t_pp = time.perf_counter() - t0
    finally:
        os.chdir(Path.cwd())
    print(f"[postprocess] wall={t_pp:.1f}s stats={stats}", flush=True)
    compare_links(WORK, SFIRST, SLAST, "trackcorr+post")
    compare_tracks(WORK)
    acc_stats(WORK / "res", "post-smooth")

    # ---- B. two-phase ----
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )
    restore()
    cpar = ControlPar.from_file(str(WORK / "parameters" / "ptv.par"))
    cals = [Calibration.from_file(
        str(WORK / "cal" / f"cam_{c}.tif.ori"),
        str(WORK / "cal" / f"cam_{c}.tif.addpar")) for c in range(1, NCAMS + 1)]
    frames = list(range(FIRST, LAST + 1))
    t0 = time.perf_counter()
    positions, leaves = [], []
    for f in frames:
        P = read_xyz(WORK / "res", f)
        positions.append(P)
        xy = np.full((len(P), NCAMS * 2), np.nan)
        for i, (x, y, z) in enumerate(P):
            for ci in range(NCAMS):
                m = img_coord_batch(np.array([[x, y, z]]), cals[ci], cpar.mm)[0]
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
    t_tp = time.perf_counter() - t0
    print(f"[two-phase+vel] wall={t_tp:.1f}s links={len(links)}", flush=True)
    ref_prev = {}
    for k, f in enumerate(frames):
        ref_prev[k] = read_prev(REF, f)
    ref_total = sum(sum(1 for p in ref_prev[k] if p >= 0)
                    for k, f in enumerate(frames) if SFIRST <= f <= SLAST)
    exact = sum(
        1 for (t0i, r0, t1, r1) in links
        if SFIRST <= frames[t1] <= SLAST and t1 - t0i == 1
        and r1 < len(ref_prev[t1]) and ref_prev[t1][r1] == r0)
    print(f"[two-phase+vel] ref={ref_total} exact={exact} "
          f"recall={exact / ref_total:.2%}", flush=True)
    chain_stats(links, positions, frames, "two-phase-chains")
    # smoothness over two-phase consecutive links
    mp = {(t1, r1): (t0i, r0) for (t0i, r0, t1, r1) in links if t1 - t0i == 1}
    accs = []
    for (t1, r1), (t0i, r0) in mp.items():
        if (t0i, r0) in mp and t0i - mp[(t0i, r0)][0] == 1:
            t_1, r_1 = mp[(t0i, r0)]
            a = positions[t1][r1] - positions[t0i][r0]
            b = positions[t0i][r0] - positions[t_1][r_1]
            accs.append(float(np.linalg.norm(a - b)))
    accs = np.array(accs)
    print(f"[two-phase-smooth] triples={len(accs)} acc mean={accs.mean():.3f} "
          f"p90={np.percentile(accs, 90):.3f} max={accs.max():.3f} "
          f"rough(>1.9)={(accs > 1.9).sum()}", flush=True)


if __name__ == "__main__":
    main()
