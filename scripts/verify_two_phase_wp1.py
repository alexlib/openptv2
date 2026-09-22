"""wp1 re-verification for the productized two-phase tracker.

Loads pristine rt_is.* (restored first), builds 2D leaves by projection
with the work calibrations, runs TwoPhaseTracker (+velocity, re-projected
costs, gaps) as a pure function (no file writes), and scores per-link exact
reproduction + chain census vs the 3dptv reference.

Run from repo root:
    uv run python -u scripts/verify_two_phase_wp1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
WORK = (REPO / "scratch" / "_wp1_ab").resolve()
FIRST, LAST = 100001, 100010
SCORE_FIRST, SCORE_LAST = 100003, 100010

NCAMS = 4


def restore_inputs() -> None:
    for f in range(FIRST, LAST + 1):
        shutil.copyfile(REF / f"rt_is.{f}", WORK / "res" / f"rt_is.{f}")
        for cam in range(1, NCAMS + 1):
            shutil.copyfile(
                DS / "img_3dptv" / f"Cam{cam}.{f}_targets",
                WORK / "img_3dptv" / f"Cam{cam}.{f}_targets",
            )


def main():
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord_batch
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )

    restore_inputs()

    cpar = ControlPar.from_file(str(WORK / "parameters" / "ptv.par"))
    cals = [Calibration.from_file(
        str(WORK / "cal" / f"cam_{c}.tif.ori"),
        str(WORK / "cal" / f"cam_{c}.tif.addpar"))
        for c in range(1, NCAMS + 1)]

    frames = list(range(FIRST, LAST + 1))
    positions, leaves = [], []
    for f in frames:
        lines = (WORK / "res" / f"rt_is.{f}").read_text().splitlines()
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

    for v_max in (2.0, 2.5, 3.3):
        for max_gap in (1, 2):
            cfg = TwoPhaseTrackerConfig(
                v_max=v_max, leaf_weight=1.0, use_velocity=True,
                cost_mode="projected", max_gap=max_gap)
            links = TwoPhaseTracker(cfg).track_frames(
                positions, leaves, project_fn=project_fn)
            # Link-based stats with correct frames (gap links span >1).
            # ref linkage: consecutive frames only.
            ref_prev = {}
            for k, f in enumerate(frames):
                lines = (REF / f"ptv_is.{f}").read_text().splitlines()
                n = int(lines[0])
                ref_prev[k] = [int(l.split()[0]) for l in lines[1: n + 1]]
            ref_total = sum(
                sum(1 for p in ref_prev[k] if p >= 0)
                for k, f in enumerate(frames)
                if SCORE_FIRST <= f <= SCORE_LAST)
            exact = consec_only = gap_links = 0
            only_steps = []
            for t0, r0, t1, r1 in links:
                f1 = frames[t1]
                if not (SCORE_FIRST <= f1 <= SCORE_LAST):
                    continue
                if t1 - t0 > 1:
                    gap_links += 1
                    continue
                consec_only += 1
                rp = ref_prev[t1]
                if r1 < len(rp) and rp[r1] >= 0 and rp[r1] == r0:
                    exact += 1
                else:
                    only_steps.append(float(np.linalg.norm(
                        positions[t1][r1] - positions[t0][r0])))
            only = np.array(only_steps)
            extra = (f" only-step p50={np.median(only):.3f} "
                     f"p90={np.percentile(only, 90):.3f} "
                     f"max={only.max():.3f}" if len(only) else "")
            print(f"two-phase+vel v_max={v_max} gap={max_gap}: "
                  f"ref={ref_total} exact={exact} "
                  f"recall={exact / ref_total:.2%} links={len(links)} "
                  f"consec={consec_only} gaplinks={gap_links}{extra}",
                  flush=True)


if __name__ == "__main__":
    main()
