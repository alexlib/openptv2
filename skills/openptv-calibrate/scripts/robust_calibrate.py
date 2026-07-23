#!/usr/bin/env python
"""Robust (RANSAC/IRLS-style) outlier rejection to reach a sub-pixel calibration.

A *polish* step, run AFTER an existing calibration (`calib.py run`,
`recalibrate_*`, or a GUI fit). It does NOT reseed or change the workflow --
it starts from each camera's current .ori/.addpar, matches the calblock
(sortgrid), then removes the worst-reprojecting correspondences one at a time,
refitting the full camera model each time, until the surviving ("inlier") set
fits to a sub-pixel target RMS -- or a keep-floor stops it.

Why this exists: a plain least-squares bundle adjustment is dragged by the few
points the pinhole+Brown model genuinely cannot reproject (a mismatched
sortgrid ID, a dot distorted by a splitter prism at the frame edge, a partly
occluded blob). Squared error punishes the fit for those as hard as for the
good points, so the whole calibration is pulled off. Dropping them yields a
clean fit over the points the model CAN represent.

This is the deterministic cousin of RANSAC: instead of random minimal samples,
it greedily trims the current worst point (the initial pose is already good, so
no random restarts are needed). The "consensus set" is whatever survives.

IMPORTANT TRADE-OFF (why it is not just free accuracy): rejecting a point
because the model can't fit it can mean hiding real model error at the frame
edges, not removing a bad measurement. The keep-floor (--min-keep) guards
against trimming so hard the pose goes unconstrained, and the printed coverage
(bbox of surviving points as a fraction of the image) tells you whether a
camera collapsed to a central band -- if it did, that camera is only trustworthy
where its inliers are. Always eyeball the reproject_on_combined overlay after:
the FULL calblock should still land on the dots (global pose preserved), even
for a camera whose edges were trimmed.

Two rejection rules:
  --target R    (default 1.0) drop worst until inlier RMS <= R px.
  --mad K       instead drop every point with residual > median + K*MAD
                (self-scaling; rejects nothing when the residual is a broad
                coherent warp rather than a few spikes -- itself a useful
                diagnostic that the camera's error is optical, not outliers).

Usage:
    uv run python skills/openptv-calibrate/scripts/robust_calibrate.py <dataset> --dry-run
    uv run python skills/openptv-calibrate/scripts/robust_calibrate.py <dataset>
    uv run python skills/openptv-calibrate/scripts/robust_calibrate.py <dataset> --target 0.8 --min-keep 0.5
    uv run python skills/openptv-calibrate/scripts/robust_calibrate.py <dataset> --mad 2.5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import full_calibration
from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import Target, read_targets
from openptv2.autocalibration import (
    _cpar_from_ptv,
    _find_yaml,
    _matched_pairs,
    cam_files,
    resolve_calblock,
    rms_px,
    target_base,
)

FLAGS = ["cc", "xh", "yh", "k1", "k2", "k3", "p1", "p2"]


def _residuals(cal, cpar, fix, sp):
    """Per-inlier reprojection residual (px) plus the matched det/rep arrays."""
    _, det, rep = _matched_pairs(cal, cpar, fix, sp)
    r = np.hypot(rep[:, 0] - det[:, 0], rep[:, 1] - det[:, 1])
    return r, det, rep


def _fit(cal_path_ori, cal_path_addpar, fix, sp, cpar):
    cal = Calibration.from_file(str(cal_path_ori), str(cal_path_addpar))
    full_calibration(cal, fix, sp, cpar, FLAGS)
    return cal


def _mask_sp(sp, keep):
    """Rebuild the sorted-pix list with only kept indices matched."""
    return [t if (t.pnr >= 0 and keep.get(i, False)) else Target(pnr=-999)
            for i, t in enumerate(sp)]


def robust_camera(cam, base, cpar, fix, nfix, eps, target, min_keep, mad_k):
    _, ori, addpar = cam_files(base, cam)
    pix = read_targets(str(target_base(base, cam)), 0)
    if not pix:
        return None, f"cam{cam + 1}: no detected targets, skipping"

    cal0 = Calibration.from_file(str(ori), str(addpar))
    sp = sortgrid(cal0, cpar, nfix, fix, len(pix), eps, pix)
    matched_idx = [i for i, t in enumerate(sp) if t.pnr >= 0]
    if len(matched_idx) < 6:
        return None, f"cam{cam + 1}: only {len(matched_idx)} matched (<6), skipping"

    # baseline fit over all matched points
    cal_all = _fit(ori, addpar, fix, sp, cpar)
    r0, _, _ = _residuals(cal_all, cpar, fix, sp)
    n_all, rms_all = len(r0), float(np.sqrt((r0 ** 2).mean()))

    keep = {i: True for i in matched_idx}
    n0 = len(matched_idx)

    if mad_k is not None:
        # one-shot self-scaling reject: everything past median + K*MAD
        med = np.median(r0)
        mad = np.median(np.abs(r0 - med)) * 1.4826
        cutoff = med + mad_k * mad
        for i, rr in zip(matched_idx, r0):
            if rr > cutoff:
                keep[i] = False
        sp_in = _mask_sp(sp, keep)
        cal = _fit(ori, addpar, fix, sp_in, cpar)
        r, det, _ = _residuals(cal, cpar, fix, sp_in)
    else:
        # greedy target-RMS reject: drop the worst point, refit, repeat
        cal, det = cal_all, None
        for _ in range(n0):
            sp_in = _mask_sp(sp, keep)
            cal = _fit(ori, addpar, fix, sp_in, cpar)
            r, det, _ = _residuals(cal, cpar, fix, sp_in)
            idx_in = [i for i, t in enumerate(sp_in) if t.pnr >= 0]
            if np.sqrt((r ** 2).mean()) <= target or len(idx_in) <= min_keep * n0:
                break
            worst = idx_in[int(np.argmax(r))]
            keep[worst] = False

    n_in = len(r)
    rms_in = float(np.sqrt((r ** 2).mean()))
    d = np.hypot(det[:, 0] - cpar.imx / 2, det[:, 1] - cpar.imy / 2)
    core = float(np.sqrt((r[d < 120] ** 2).mean())) if (d < 120).any() else float("nan")
    cov_x = (det[:, 0].max() - det[:, 0].min()) / cpar.imx
    cov_y = (det[:, 1].max() - det[:, 1].min()) / cpar.imy
    msg = (f"cam{cam + 1}: all {n_all}/{rms_all:.3f}px -> inliers {n_in}/{rms_in:.3f}px  "
           f"(dropped {n_all - n_in})  core={core:.3f}  cover={cov_x:.2f}x{cov_y:.2f}")
    return (cal, ori, addpar), msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--target", type=float, default=1.0,
                    help="sub-pixel inlier RMS goal in px (greedy mode, default 1.0)")
    ap.add_argument("--min-keep", type=float, default=0.60,
                    help="never keep fewer than this fraction of matched points (default 0.60)")
    ap.add_argument("--mad", type=float, default=None,
                    help="use one-shot median+K*MAD rejection instead of greedy target-RMS")
    ap.add_argument("--dry-run", action="store_true", help="report without writing .ori/.addpar")
    args = ap.parse_args()

    base = Path(args.dataset).resolve()
    import os
    os.chdir(base)

    yaml_path = _find_yaml(base)
    if yaml_path is None:
        print(f"ERROR: no parameters_*.yaml found in {base}", file=sys.stderr)
        return 1
    import yaml
    y = yaml.safe_load(yaml_path.read_text())
    num_cams = int(y.get("num_cams") or y["ptv"].get("num_cams"))
    cpar = _cpar_from_ptv(y["ptv"], num_cams)
    eps = int(y.get("sortgrid", {}).get("radius", 5))

    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))

    print(f"nfix={nfix}  radius(eps)={eps}  "
          f"mode={'MAD K=%s' % args.mad if args.mad is not None else 'target %.2fpx' % args.target}"
          f"  min-keep={args.min_keep:.0%}")
    writes = []
    for cam in range(num_cams):
        res, msg = robust_camera(cam, base, cpar, fix, nfix, eps,
                                 args.target, args.min_keep, args.mad)
        print(msg)
        if res is not None:
            writes.append(res)

    if args.dry_run:
        print("\n(dry run -- nothing written; drop --dry-run to write .ori/.addpar)")
    else:
        for cal, ori, addpar in writes:
            shutil.copy2(ori, str(ori) + ".robustbck")
            shutil.copy2(addpar, str(addpar) + ".robustbck")
            cal.write(str(ori), str(addpar))
        print(f"\nWrote {len(writes)} refined calibrations (originals backed up as *.robustbck).")
        print("Verify: re-run reproject_on_combined.py -- the FULL calblock should still "
              "land on the dots even for a trimmed camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
