# ruff: noqa: E501
#!/usr/bin/env python
"""Re-fit calibration with distortion removed and focal length fixed.

Starting from the current calibration (already a converged fit, e.g. from
`calib.py run`), this:
  1. Zeroes each camera's .addpar (k1=k2=k3=p1=p2=0, scx=1, she=0) -- "remove
     addpar".
  2. Re-runs sortgrid + bundle adjustment allowing only exterior orientation
     (x0,y0,z0,omega,phi,kappa -- always free in orient()) plus xh,yh. The
     focal length (cc) and all distortion/affine terms stay fixed.

Camera image/.ori/.addpar paths and the calblock path are resolved from the
dataset YAML's cal_ori: block via openptv2.autocalibration.cam_files() /
resolve_calblock() -- the same files the GUI reads and writes, so there is
no separate camN.tif naming convention to keep in sync by hand.

Rationale: a multi-camera rig (e.g. an image-splitter setup) should look
physically sensible about the calibration body's center; a full-distortion
fit with sparse matches can overfit non-physical distortion that masks a bad
pose. Removing those degrees of freedom gives a comparable, physically
constrained pose per camera, useful for diagnosing whether an odd-looking
setup is a real pose problem or just an overfit.

Run with: uv run python skills/openptv-calibrate/scripts/recalibrate_constrained.py <dataset>
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import AddedPar, Calibration
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
    target_base,
)

FLAGS = ["xh", "yh"]  # exterior always free in orient(); cc/distortion/affine fixed
REFINE_ITERS = 3


def recalibrate_camera(cam, base, cpar, fix, nfix, eps):
    _, ori, addpar = cam_files(base, cam)

    cal = Calibration.from_file(str(ori), str(addpar))
    cal.added_par = AddedPar()  # remove addpar: zero distortion, identity affine

    pix = read_targets(str(target_base(base, cam)), 0)
    if not pix:
        raise RuntimeError(f"cam{cam + 1}: no detected targets found")

    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    n_matched = sum(1 for t in sorted_pix if t.pnr >= 0)
    for _ in range(REFINE_ITERS):
        full_calibration(cal, fix, sorted_pix, cpar, FLAGS)
        sp = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
        n = sum(1 for t in sp if t.pnr >= 0)
        sorted_pix = sp
        if n <= n_matched:
            n_matched = n
            break
        n_matched = n

    full_calibration(cal, fix, sorted_pix, cpar, FLAGS)
    ref, det, rep = _matched_pairs(cal, cpar, fix, sorted_pix)
    rms = rms_px(det, rep)
    return cal, n_matched, rms, ref, det, rep


def main():
    from dataclasses import dataclass

    if len(sys.argv) < 2:
        print("Usage: recalibrate_constrained.py <dataset>", file=sys.stderr)
        return 1

    base = Path(sys.argv[1]).resolve()
    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams, eps = dp.cpar, dp.num_cams, dp.eps

    outdir = base / "cal" / "constrained_calib"
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

    print(f"{'cam':<6}{'matched':<10}{'RMS px':<10}")
    for cam in range(num_cams):
        _, ori, addpar = cam_files(base, cam)
        shutil.copy2(ori, Path(str(ori) + ".pre_constrained"))
        shutil.copy2(addpar, Path(str(addpar) + ".pre_constrained"))

        cal, n_matched, rms, ref, det, rep = recalibrate_camera(
            cam, base, cpar, fix, nfix, eps
        )
        cal.write(str(ori), str(addpar))

        res = _Res(
            cam=cam,
            rms=rms,
            matched=n_matched,
            nfix=nfix,
            flags=FLAGS,
            det=det,
            rep=rep,
        )
        save_overlay(res, base, outdir)
        print(f"cam{cam + 1:<5}{n_matched}/{nfix:<6}{rms:<10.4f}")

    print(f"\nOverlays written to {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
