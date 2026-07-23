#!/usr/bin/env python
"""Greedy, one-parameter-at-a-time distortion tuning per camera.

Builds up the free-parameter set incrementally and remembers what already
helped, rather than throwing every distortion term in at once (which makes
it impossible to tell which parameter did what, or whether one is fighting
another). Per camera:

  1. Start from the current .ori/.addpar and whatever's already free
     (--base-flags, default 'k1' since that's usually already been applied).
  2. Try adding each remaining CANDIDATE parameter one at a time (exterior
     is always free; cc is NEVER a candidate -- keeps focal distance fixed
     throughout, per the user's explicit choice).
  3. Accept the single candidate that improves RMS the most, if the
     improvement clears --min-improve (relative). Repeat with the enlarged
     accepted set until no remaining candidate clears the bar.
  4. Final polish: one more full_calibration with exactly the accepted set,
     against the full matched set -- the "very limited adjustment near
     optimal settings" pass.

Reports a per-camera table: baseline RMS, RMS after each accepted addition,
and the final accepted flag set + RMS + matched count.

Usage:
    uv run python skills/openptv-calibrate/scripts/tune_distortion_stepwise.py <dataset> [--dry-run]
    uv run python skills/openptv-calibrate/scripts/tune_distortion_stepwise.py <dataset> --base-flags k1 --candidates k2,k3,p1,p2,xh,yh
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import full_calibration
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

FORBIDDEN = {"cc"}  # focal distance stays fixed, always


def _drop_excluded(sorted_pix, exclude_ids):
    """Mark specific calblock IDs (1-indexed, matching row i -> id i+1) as
    unmatched, in place, so full_calibration/_matched_pairs ignore them --
    same convention sortgrid itself uses for a point it couldn't match."""
    if not exclude_ids:
        return
    for i, t in enumerate(sorted_pix):
        if (i + 1) in exclude_ids and t.pnr >= 0:
            t.pnr = -999


def _fit_and_score(base_cal, fix, nfix, pix, cpar, eps, flags, exclude_ids=None):
    """Fit a copy of base_cal with the given flags; return (cal, rms, n_matched) or None."""
    cal = copy.deepcopy(base_cal)
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    _drop_excluded(sorted_pix, exclude_ids)
    if sum(1 for t in sorted_pix if t.pnr >= 0) < 6:
        return None
    try:
        full_calibration(cal, fix, sorted_pix, cpar, flags)
    except (ValueError, RuntimeError):
        return None
    sorted_pix2 = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    _drop_excluded(sorted_pix2, exclude_ids)
    _, det, rep = _matched_pairs(cal, cpar, fix, sorted_pix2)
    if len(det) < 6:
        return None
    return cal, rms_px(det, rep), len(det)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--base-flags", default="k1", help="already-accepted flags (default: k1)")
    ap.add_argument("--candidates", default="k2,k3,p1,p2,xh,yh",
                     help="comma-separated candidates to test, in priority order")
    ap.add_argument("--min-improve", type=float, default=0.03,
                     help="minimum relative RMS improvement to accept a candidate (default 0.03 = 3%%)")
    ap.add_argument("--exclude-ids", default="",
                     help="comma-separated cam:id pairs to drop from the fit, 1-indexed camera, "
                          "e.g. '2:53,4:94,4:97,4:87,4:48,4:96,3:104'")
    ap.add_argument("--dry-run", action="store_true", help="report without writing .ori/.addpar")
    args = ap.parse_args()

    base_flags = [f.strip() for f in args.base_flags.split(",") if f.strip()]
    candidates = [f.strip() for f in args.candidates.split(",") if f.strip()]
    for f in base_flags + candidates:
        if f in FORBIDDEN:
            print(f"ERROR: '{f}' is forbidden -- focal distance (cc) stays fixed", file=sys.stderr)
            return 1

    exclude_by_cam: dict[int, set[int]] = {}
    for pair in args.exclude_ids.split(","):
        pair = pair.strip()
        if not pair:
            continue
        cam_str, id_str = pair.split(":")
        exclude_by_cam.setdefault(int(cam_str) - 1, set()).add(int(id_str))

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

    for cam in range(num_cams):
        _, ori, addpar = cam_files(base, cam)
        pix = read_targets(str(target_base(base, cam)), 0)
        if not pix:
            print(f"cam{cam + 1}: no detected targets, skipping", file=sys.stderr)
            continue

        cal0 = Calibration.from_file(str(ori), str(addpar))
        cc0 = cal0.int_par.cc
        exclude_ids = exclude_by_cam.get(cam)
        if exclude_ids:
            print(f"cam{cam + 1}: excluding IDs {sorted(exclude_ids)} from the fit")

        result = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, list(base_flags), exclude_ids)
        if result is None:
            print(f"cam{cam + 1}: baseline ({base_flags}) failed to fit, skipping", file=sys.stderr)
            continue
        best_cal, best_rms, best_n = result
        print(f"\ncam{cam + 1}: baseline flags={base_flags}  RMS={best_rms:.4f}px  n={best_n}")

        accepted = list(base_flags)
        remaining = [c for c in candidates if c not in accepted]
        history = [(tuple(accepted), best_rms, best_n)]

        while remaining:
            trial_results = {}
            for cand in remaining:
                trial_flags = accepted + [cand]
                r = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, trial_flags, exclude_ids)
                if r is not None:
                    trial_results[cand] = r

            if not trial_results:
                break

            # pick the candidate giving the lowest RMS
            best_cand = min(trial_results, key=lambda c: trial_results[c][1])
            cand_cal, cand_rms, cand_n = trial_results[best_cand]
            improve = (best_rms - cand_rms) / best_rms if best_rms > 0 else 0

            print(f"  + {best_cand}: RMS {best_rms:.4f} -> {cand_rms:.4f}px "
                  f"({improve * 100:+.1f}%)  n={cand_n}"
                  + ("  [accepted]" if improve >= args.min_improve else "  [below threshold, stopping]"))

            if improve < args.min_improve:
                break

            accepted.append(best_cand)
            remaining.remove(best_cand)
            best_cal, best_rms, best_n = cand_cal, cand_rms, cand_n
            history.append((tuple(accepted), best_rms, best_n))

        # Final polish: one more fit with exactly the accepted set (already
        # what best_cal is, since it was fit with `accepted` -- but redo once
        # more from cal0 for a clean, single-source-of-truth final result).
        final = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, accepted, exclude_ids)
        if final is not None:
            best_cal, best_rms, best_n = final

        cc_ok = abs(best_cal.int_par.cc - cc0) < 1e-9
        subpixel = "YES" if best_rms < 1.0 else "no"
        print(f"cam{cam + 1}: FINAL flags={accepted}  RMS={best_rms:.4f}px  n={best_n}/{nfix}  "
              f"subpixel={subpixel}  cc unchanged={cc_ok}")

        if not args.dry_run:
            suffix = f".pre_{'_'.join(accepted) if accepted else 'stepwise'}_bak"
            shutil.copy2(ori, str(ori) + suffix)
            shutil.copy2(addpar, str(addpar) + suffix)
            best_cal.write(str(ori), str(addpar))

    if args.dry_run:
        print("\n(dry run -- nothing written; drop --dry-run to write .ori/.addpar)")
    else:
        print("\nWrote tuned .ori/.addpar (originals backed up).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
