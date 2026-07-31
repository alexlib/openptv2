#!/usr/bin/env python
"""Full (unconstrained) calibration, but staged: addpar (cc + distortion) is
only switched on after position/angles/xh/yh have already converged.

Stage 1 (reuses recalibrate_constrained.recalibrate_camera): addpar forced to
zero, only exterior (x0,y0,z0,omega,phi,kappa -- always free in orient()) and
xh,yh adjust.

Stage 2: starting from that converged pose, re-enable cc and try increasingly
rich distortion flag-sets (same progression as the original one-shot
autocalibration), keeping whichever gives the lowest reprojection RMS. Because
distortion only gets to explain residual error *after* the rigid pose is
already right, it can't compensate for a bad pose by warping into an
asymmetric solution the way a from-scratch full-flags fit could.

Camera image/.ori/.addpar paths and the calblock path are resolved from the
dataset YAML's cal_ori: block via openptv2.autocalibration.cam_files() /
resolve_calblock() -- the same files the GUI reads and writes, so there is
no separate camN.tif naming convention to keep in sync by hand.

Also useful as a recovery path when `calib.py run` fails with
"external_calibration did not converge": that happens when the man_ori seed
points are degenerate (e.g. near-collinear, or the same 4 IDs reused
unchanged across every camera). If the dataset already has a real prior
calibration on disk (not a placeholder), this script sidesteps the seed
entirely by starting from the existing .ori as the initial guess instead of
calling external_calibration.

Run with: uv run python skills/openptv-calibrate/scripts/recalibrate_full.py <dataset>
"""
from __future__ import annotations

import copy
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from recalibrate_constrained import recalibrate_camera as fit_constrained

from openptv2.algorithms.orientation import full_calibration
from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.autocalibration import (
    _load_dataset_params,
    _matched_pairs,
    cam_files,
    resolve_calblock,
    rms_px,
    save_overlay,
)

CANDIDATE_FLAGS = [
    ["cc", "xh", "yh"],
    ["cc", "xh", "yh", "k1", "k2"],
    ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2"],
]


def calibrate_camera_full(cam, base, cpar, fix, nfix, eps):
    # Stage 1: converge position/angles/xh/yh with addpar forced to zero.
    cal, n_matched, rms_stage1, _, _, _ = fit_constrained(cam, base, cpar, fix, nfix, eps)

    # Stage 2: only now allow cc + distortion to adjust, seeded from the
    # stage-1 pose (not from the raw manual-orientation seed).
    img, _, _ = cam_files(base, cam)
    pix = read_targets(str(img), 0)
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)

    best = None
    for flags in CANDIDATE_FLAGS:
        trial = copy.deepcopy(cal)
        try:
            full_calibration(trial, fix, sorted_pix, cpar, flags)
        except (ValueError, RuntimeError):
            continue
        ref, det, rep = _matched_pairs(trial, cpar, fix, sorted_pix)
        r = rms_px(det, rep)
        if best is None or r < best[0]:
            best = (r, trial, flags, ref, det, rep)

    if best is None:
        raise RuntimeError(f"cam{cam + 1}: no distortion flag-set converged in stage 2")
    rms, cal_best, flags, ref, det, rep = best
    return cal_best, n_matched, rms_stage1, rms, flags, ref, det, rep


def main():
    if len(sys.argv) < 2:
        print("Usage: recalibrate_full.py <dataset>", file=sys.stderr)
        return 1

    base = Path(sys.argv[1]).resolve()
    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams, eps = dp.cpar, dp.num_cams, dp.eps

    outdir = base / "cal" / "full_calib_staged"
    outdir.mkdir(exist_ok=True)

    @dataclass
    class _Res:
        cam: int
        rms: float
        matched: int
        nfix: int
        flags: list
        det: np.ndarray
        rep: np.ndarray

    print(f"{'cam':<6}{'matched':<10}{'stage1 RMS':<13}{'stage2 RMS':<13}{'flags'}")
    for cam in range(num_cams):
        _, ori, addpar = cam_files(base, cam)
        shutil.copy2(ori, Path(str(ori) + ".pre_full_staged"))
        shutil.copy2(addpar, Path(str(addpar) + ".pre_full_staged"))

        cal, n_matched, rms1, rms2, flags, ref, det, rep = calibrate_camera_full(
            cam, base, cpar, fix, nfix, eps
        )
        cal.write(str(ori), str(addpar))

        res = _Res(cam=cam, rms=rms2, matched=n_matched, nfix=nfix, flags=flags, det=det, rep=rep)
        save_overlay(res, base, outdir)
        print(f"cam{cam + 1:<5}{n_matched}/{nfix:<6}{rms1:<13.4f}{rms2:<13.4f}{'+'.join(flags)}")

    print(f"\nOverlays written to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
