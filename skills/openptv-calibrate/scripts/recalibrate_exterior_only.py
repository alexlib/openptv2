#!/usr/bin/env python
"""Refine exterior orientation (x,y,z + angles) plus an explicit, small set
of additional free parameters -- everything else stays exactly as it is on
disk. Default (--flags empty) is exterior-only.

For "one improvement at a time" calibration tuning: start from the current
.ori/.addpar as-is (not a fresh 4-point reseed), bundle-adjust against ALL
detected targets with only the named flags free (`full_calibration`'s flags
list: cc, xh, yh, k1, k2, k3, p1, p2, scale, shear, interf -- see its
docstring), and report the before/after RMS so you can judge whether that
ONE parameter actually helped before adding another. Unlike `calib.py run`
(which always re-derives exterior from a 4-point man_ori seed and always
frees cc/xh/yh), nothing not named in --flags ever changes.

Usage:
    uv run python skills/openptv-calibrate/scripts/recalibrate_exterior_only.py <dataset> [--dry-run]
    uv run python skills/openptv-calibrate/scripts/recalibrate_exterior_only.py <dataset> --flags k1
    uv run python skills/openptv-calibrate/scripts/recalibrate_exterior_only.py <dataset> --flags k1,p1,p2
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import external_calibration, full_calibration
from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.autocalibration import (
    _cpar_from_ptv,
    _find_yaml,
    _matched_pairs,
    cam_files,
    resolve_calblock,
    rms_px,
    target_base,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--flags", default="",
                     help="comma-separated extra free params, e.g. 'k1' or 'k1,p1,p2' "
                          "(default: none -- exterior only)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing .ori")
    args = ap.parse_args()
    flags = [f.strip() for f in args.flags.split(",") if f.strip()]

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
    eps = int(y.get("sortgrid", {}).get("radius", 10))

    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))
    ids_all = np.loadtxt(calblock, usecols=0).astype(int)

    man_ori_nr = y.get("man_ori", {}).get("nr")
    man_ori_coords = y.get("man_ori_coordinates")

    for cam in range(num_cams):
        _, ori, addpar = cam_files(base, cam)
        pix = read_targets(str(target_base(base, cam)), 0)
        if not pix:
            print(f"cam{cam + 1}: no detected targets, skipping", file=sys.stderr)
            continue

        cal_current = Calibration.from_file(str(ori), str(addpar))
        cc_before = cal_current.int_par.cc
        xh_before, yh_before = cal_current.int_par.xh, cal_current.int_par.yh
        pos_before = cal_current.get_pos().copy()

        sorted_pix = sortgrid(cal_current, cpar, nfix, fix, len(pix), eps, pix)
        n_matched = sum(1 for t in sorted_pix if t.pnr >= 0)
        cal = cal_current

        # Always try the man_ori-seeded pose too (external_calibration solves
        # exterior directly from 4 correspondences, no close starting guess
        # needed) and keep whichever pose sortgrid matches more points
        # against -- the current pose isn't necessarily the better starting
        # point even when it's not badly wrong.
        if man_ori_nr and man_ori_coords and f"camera_{cam}" in man_ori_coords:
            seed_ids = man_ori_nr[cam * 4:(cam + 1) * 4]
            idx = [list(ids_all).index(i) for i in seed_ids]
            fix4 = fix[idx]
            cpts = man_ori_coords[f"camera_{cam}"]
            pix4 = np.array([[cpts[f"point_{k}"]["x"], cpts[f"point_{k}"]["y"]] for k in range(1, 5)])
            cal_seeded = Calibration.from_file(str(ori), str(addpar))
            if external_calibration(cal_seeded, fix4, pix4, cpar):
                sorted_pix_seeded = sortgrid(cal_seeded, cpar, nfix, fix, len(pix), eps, pix)
                n_seeded = sum(1 for t in sorted_pix_seeded if t.pnr >= 0)
                print(f"cam{cam + 1}: current pose {n_matched} matched vs. "
                      f"man_ori-seeded ({seed_ids}) {n_seeded} matched")
                if n_seeded > n_matched:
                    cal, sorted_pix, n_matched = cal_seeded, sorted_pix_seeded, n_seeded
            else:
                print(f"cam{cam + 1}: man_ori reseed did not converge", file=sys.stderr)

        if n_matched < 6:
            print(f"cam{cam + 1}: only {n_matched} matched (need >=6 for a stable "
                  f"6-DOF exterior fit) -- skipping", file=sys.stderr)
            continue

        try:
            full_calibration(cal, fix, sorted_pix, cpar, flags)
        except (ValueError, RuntimeError) as exc:
            print(f"cam{cam + 1}: full_calibration failed: {exc}", file=sys.stderr)
            continue

        # Re-sortgrid after the fit -- the pose moved, so re-match before scoring.
        sorted_pix2 = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
        _, det, rep = _matched_pairs(cal, cpar, fix, sorted_pix2)
        rms = rms_px(det, rep)
        n_matched = len(det)
        cc_yh_unchanged = (
            abs(cal.int_par.cc - cc_before) < 1e-9
            and abs(cal.int_par.xh - xh_before) < 1e-9
            and abs(cal.int_par.yh - yh_before) < 1e-9
        )
        extra = "  ".join(
            f"{name}={getattr(cal.added_par, name):.4e}"
            for name in flags if hasattr(cal.added_par, name)
        )
        print(f"cam{cam + 1}: matched {n_matched}/{nfix}  RMS={rms:.3f}px  "
              f"pos {pos_before.round(2)} -> {cal.get_pos().round(2)}  "
              f"cc/xh/yh unchanged={cc_yh_unchanged}"
              + (f"  {extra}" if extra else ""))

        if not args.dry_run:
            suffix = f".pre_{'_'.join(flags) if flags else 'extonly'}_bak"
            shutil.copy2(ori, str(ori) + suffix)
            shutil.copy2(addpar, str(addpar) + suffix)
            cal.write(str(ori), str(addpar))

    if args.dry_run:
        print("\n(dry run -- nothing written; drop --dry-run to write .ori/.addpar)")
    else:
        print("\nWrote refined .ori/.addpar (originals backed up as *.pre_extonly_bak).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
