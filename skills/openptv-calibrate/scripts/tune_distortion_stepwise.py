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


def _fit_and_score(base_cal, fix, nfix, pix, cpar, eps, flags, exclude_ids=None,
                    auto_reject_mad=None, max_rounds=8, verbose_prefix=None):
    """Fit a copy of base_cal with the given flags; return (cal, rms, n_matched,
    auto_excluded_ids) or None.

    auto_reject_mad: if set, after each fit compute per-point residuals,
    flag any point beyond `median + auto_reject_mad * MAD` (MAD = median
    absolute deviation, scaled by 1.4826 to approximate a standard
    deviation -- robust to the very outliers being rejected, unlike mean/std,
    which the outliers themselves would inflate), add those IDs to the
    exclusion set, and refit. Repeats until no new point is flagged (or
    max_rounds) -- iterative sigma-clipping, not RANSAC: appropriate here
    because we already have a good starting fit and a small outlier
    fraction, not a majority-corrupt dataset needing random-subset bootstrap.
    """
    exclude_ids = set(exclude_ids or [])
    auto_excluded: set[int] = set()

    for round_i in range(max_rounds if auto_reject_mad else 1):
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
        ids_matched = [i + 1 for i, t in enumerate(sorted_pix2) if t.pnr >= 0]
        _, det, rep = _matched_pairs(cal, cpar, fix, sorted_pix2)
        if len(det) < 6:
            return None

        if not auto_reject_mad:
            return cal, rms_px(det, rep), len(det), auto_excluded

        err = np.sqrt(np.sum((det - rep) ** 2, axis=1))
        med = np.median(err)
        mad = np.median(np.abs(err - med)) * 1.4826
        threshold = med + auto_reject_mad * max(mad, 1e-6)
        new_bad = {pid for pid, e in zip(ids_matched, err) if e > threshold}
        if not new_bad:
            return cal, rms_px(det, rep), len(det), auto_excluded

        if verbose_prefix:
            print(f"{verbose_prefix}  round {round_i + 1}: median={med:.2f}px "
                  f"MAD={mad:.2f}px threshold={threshold:.2f}px -> "
                  f"rejecting {sorted(new_bad)}")
        exclude_ids |= new_bad
        auto_excluded |= new_bad

    # Ran out of rounds -- return the last fit anyway.
    return cal, rms_px(det, rep), len(det), auto_excluded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--base-flags", default="k1", help="already-accepted flags (default: k1)")
    ap.add_argument("--candidates", default="k2,k3,p1,p2,xh,yh",
                     help="comma-separated candidates to test, in priority order")
    ap.add_argument("--min-improve", type=float, default=0.03,
                     help="minimum relative RMS improvement to accept a candidate (default 0.03 = 3%%). "
                          "Ignored in --sequential mode, which accepts on ANY improvement.")
    ap.add_argument("--sequential", action="store_true",
                     help="try --candidates in the EXACT order given (not greedy best-first), "
                          "one at a time on top of whatever was already accepted. Keep a candidate "
                          "if it doesn't make RMS worse; revert (skip, don't stop) otherwise, then "
                          "move on to the next candidate in the list regardless.")
    ap.add_argument("--exclude-ids", default="",
                     help="comma-separated cam:id pairs to drop from the fit, 1-indexed camera, "
                          "e.g. '2:53,4:94,4:97,4:87,4:48,4:96,3:104'")
    ap.add_argument("--auto-reject-mad", type=float, default=None,
                     help="iterative sigma-clipping: after each fit, reject points beyond "
                          "median + N*MAD residual and refit, until stable (e.g. 4.0). "
                          "Combines with --exclude-ids (manual exclusions always apply too).")
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

        result = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, list(base_flags), exclude_ids,
                                 args.auto_reject_mad, verbose_prefix=f"cam{cam + 1}:")
        if result is None:
            print(f"cam{cam + 1}: baseline ({base_flags}) failed to fit, skipping", file=sys.stderr)
            continue
        best_cal, best_rms, best_n, all_auto_excluded = result
        print(f"\ncam{cam + 1}: baseline flags={base_flags}  RMS={best_rms:.4f}px  n={best_n}")

        accepted = list(base_flags)
        history = [(tuple(accepted), best_rms, best_n)]

        if args.sequential:
            # Fixed order, one at a time: test candidate, keep it if RMS
            # doesn't get worse, revert (skip) otherwise -- but always move
            # on to the next candidate in the list, never stop early.
            for cand in candidates:
                if cand in accepted:
                    continue
                trial_flags = accepted + [cand]
                r = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, trial_flags, exclude_ids,
                                    args.auto_reject_mad)
                if r is None:
                    print(f"  {cand}: fit failed  [reverted]")
                    continue
                cand_cal, cand_rms, cand_n, cand_auto_excluded = r
                improve = (best_rms - cand_rms) / best_rms if best_rms > 0 else 0
                worse = cand_rms > best_rms
                print(f"  {cand}: RMS {best_rms:.4f} -> {cand_rms:.4f}px "
                      f"({improve * 100:+.1f}%)  n={cand_n}"
                      + ("  [reverted, worse]" if worse else "  [accepted]"))
                if worse:
                    continue
                accepted.append(cand)
                best_cal, best_rms, best_n = cand_cal, cand_rms, cand_n
                all_auto_excluded |= cand_auto_excluded
                history.append((tuple(accepted), best_rms, best_n))
        else:
            remaining = [c for c in candidates if c not in accepted]
            while remaining:
                trial_results = {}
                for cand in remaining:
                    trial_flags = accepted + [cand]
                    r = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, trial_flags, exclude_ids,
                                        args.auto_reject_mad)
                    if r is not None:
                        trial_results[cand] = r

                if not trial_results:
                    break

                # pick the candidate giving the lowest RMS
                best_cand = min(trial_results, key=lambda c: trial_results[c][1])
                cand_cal, cand_rms, cand_n, cand_auto_excluded = trial_results[best_cand]
                improve = (best_rms - cand_rms) / best_rms if best_rms > 0 else 0

                print(f"  + {best_cand}: RMS {best_rms:.4f} -> {cand_rms:.4f}px "
                      f"({improve * 100:+.1f}%)  n={cand_n}"
                      + ("  [accepted]" if improve >= args.min_improve else "  [below threshold, stopping]"))

                if improve < args.min_improve:
                    break

                accepted.append(best_cand)
                remaining.remove(best_cand)
                best_cal, best_rms, best_n = cand_cal, cand_rms, cand_n
                all_auto_excluded |= cand_auto_excluded
                history.append((tuple(accepted), best_rms, best_n))

        # Final polish: one more fit with exactly the accepted set (already
        # what best_cal is, since it was fit with `accepted` -- but redo once
        # more from cal0 for a clean, single-source-of-truth final result).
        final = _fit_and_score(cal0, fix, nfix, pix, cpar, eps, accepted, exclude_ids,
                                args.auto_reject_mad)
        if final is not None:
            best_cal, best_rms, best_n, final_auto_excluded = final
            all_auto_excluded |= final_auto_excluded

        cc_ok = abs(best_cal.int_par.cc - cc0) < 1e-9
        subpixel = "YES" if best_rms < 1.0 else "no"
        auto_note = f"  auto-rejected={sorted(all_auto_excluded)}" if all_auto_excluded else ""
        print(f"cam{cam + 1}: FINAL flags={accepted}  RMS={best_rms:.4f}px  n={best_n}/{nfix}  "
              f"subpixel={subpixel}  cc unchanged={cc_ok}{auto_note}")

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
