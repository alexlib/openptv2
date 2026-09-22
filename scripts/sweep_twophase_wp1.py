"""Sweep two-phase configs on wp1 to beat trackcorr (98.16% / +31 claims).

Leaves built once, configs swept (each track run is milliseconds).
Scores: link recall, claim count, chain census, smoothness.

Run from repo root:
    uv run python -u scripts/sweep_twophase_wp1.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from verify_same_trajectories import DS, REF  # noqa: E402

WORK = (Path(__file__).resolve().parent.parent / "scratch" / "_wp1_ab").resolve()
FIRST, LAST = 100001, 100010
SFIRST, SLAST = 100003, 100010
NCAMS = 4


def main():
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )

    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
    cpar = ControlPar.from_file(str(WORK / "parameters" / "ptv.par"))
    cals = [Calibration.from_file(
        str(WORK / "cal" / f"cam_{c}.tif.ori"),
        str(WORK / "cal" / f"cam_{c}.tif.addpar")) for c in range(1, NCAMS + 1)]
    frames = list(range(FIRST, LAST + 1))
    positions = []
    for f in frames:
        lines = (WORK / "res" / f"rt_is.{f}").read_text().splitlines()
        n = int(lines[0])
        positions.append(np.array([[float(v) for v in l.split()[1:4]]
                                   for l in lines[1: n + 1]]))
    t0 = time.perf_counter()
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
    print(f"leaves build: {time.perf_counter() - t0:.1f}s", flush=True)

    def project_fn(pred):
        pred = np.asarray(pred, dtype=np.float64)
        out = np.full((len(pred), NCAMS * 2), np.nan)
        for i, (x, y, z) in enumerate(pred):
            for ci in range(NCAMS):
                m = img_coord_batch(np.array([[x, y, z]]), cals[ci], cpar.mm)[0]
                out[i, 2 * ci] = m[0] / cpar.pix_x + cpar.imx / 2
                out[i, 2 * ci + 1] = cpar.imy / 2 - m[1] / cpar.pix_y
        return np.nan_to_num(out)

    ref_prev = {}
    for k, f in enumerate(frames):
        lines = (REF / f"ptv_is.{f}").read_text().splitlines()
        n = int(lines[0])
        ref_prev[k] = [int(l.split()[0]) for l in lines[1: n + 1]]
    ref_total = sum(sum(1 for p in ref_prev[k] if p >= 0)
                    for k, f in enumerate(frames) if SFIRST <= f <= SLAST)

    def score(links, tag):
        exact = consec = gaps = 0
        mp = {}
        for (t0i, r0, t1, r1) in links:
            if not (SFIRST <= frames[t1] <= SLAST):
                continue
            if t1 - t0i > 1:
                gaps += 1
                continue
            consec += 1
            mp[(t1, r1)] = (t0i, r0)
            rp = ref_prev[t1]
            if r1 < len(rp) and rp[r1] >= 0 and rp[r1] == r0:
                exact += 1
        # chains + smoothness
        nxt = {(a, b): (c, d) for (a, b, c, d) in links if c - a == 1}
        tgt = set(nxt.values())
        chains = []
        for s in sorted(set(nxt) - tgt):
            c, k = [s], s
            while k in nxt:
                k = nxt[k]
                c.append(k)
            chains.append(c)
        accs = []
        for (t1, r1), (t0i, r0) in mp.items():
            if (t0i, r0) in mp and t0i - mp[(t0i, r0)][0] == 1:
                t_1, r_1 = mp[(t0i, r0)]
                accs.append(float(np.linalg.norm(
                    (positions[t1][r1] - positions[t0i][r0]) -
                    (positions[t0i][r0] - positions[t_1][r_1]))))
        accs = np.array(accs)
        print(f"{tag}: recall={exact / ref_total:.2%} claims={len(links)} "
              f"consec={consec} gaps={gaps} chains={len(chains)} "
              f"accmean={accs.mean():.3f} rough={(accs > 1.9).sum()}",
              flush=True)

    cfgs = [
        ("base v2.0 proj", dict(v_max=2.0)),
        ("confirm0.5", dict(v_max=2.0, confirm_tol=0.5)),
        ("confirm1.0", dict(v_max=2.0, confirm_tol=1.0)),
        ("confirm1.5", dict(v_max=2.0, confirm_tol=1.5)),
        ("confirm2.0", dict(v_max=2.0, confirm_tol=2.0)),
        ("confirm3.0", dict(v_max=2.0, confirm_tol=3.0)),
        ("confirm1.0+ends", dict(v_max=2.0, confirm_tol=1.0,
                                confirm_ends=True)),
        ("confirm1.5+ends", dict(v_max=2.0, confirm_tol=1.5,
                                confirm_ends=True)),
        ("confirm2.0+ends", dict(v_max=2.0, confirm_tol=2.0,
                                confirm_ends=True)),
        ("v2.2 confirm1.5+ends", dict(v_max=2.2, confirm_tol=1.5,
                                     confirm_ends=True)),
        ("v2.5 confirm2.0+ends", dict(v_max=2.5, confirm_tol=2.0,
                                     confirm_ends=True)),
    ]
    for tag, kw in cfgs:
        t0 = time.perf_counter()
        base = dict(leaf_weight=1.0, use_velocity=True,
                    cost_mode="projected", max_gap=1)
        base.update(kw)
        cfg = TwoPhaseTrackerConfig(**base)
        links = TwoPhaseTracker(cfg).track_frames(positions, leaves,
                                                  project_fn=project_fn)
        dt = time.perf_counter() - t0
        score(links, f"{tag} ({dt:.2f}s)")


if __name__ == "__main__":
    main()
