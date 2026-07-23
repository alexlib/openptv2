#!/usr/bin/env python
"""Estimate a good eps0 (epipolar-band half-width) directly from the
calibration itself -- no real experiment frame needed.

`tune_eps0.py` sweeps eps0 against real particle detections from one frame,
which is the ground-truth answer but requires a detected sequence frame to
already exist. This script instead reuses the calibration-plate detections
every calibration already produces (cal/calib_matches/camN_matches.txt, from
dump_matches.py): for every calibration-body point ID detected in *both*
cameras of a pair, it computes camera A's actual epipolar line in camera B
(the same `epi_mm()` geometry find_candidate() uses) and measures how far
camera B's *actually detected* point sits from that line. That is exactly
the residual real correspondence search will see, driven by nothing but the
calibration's own internal consistency across camera pairs -- so it's usable
immediately after calibration, before any sequence has been processed.

Run with:
  uv run python skills/openptv-calibrate/scripts/estimate_eps0_from_calibration.py <dataset>

Requires: cal/calib_matches/camN_matches.txt already exist -- run
dump_matches.py first.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.epi import epi_mm
from openptv2.algorithms.parameters import ControlPar, VolumePar
from openptv2.algorithms.tracking_frame_buf import Target, TargetArray
from openptv2.autocalibration import cam_files, _find_yaml
from openptv2.correspondences import MatchedCoords


def load_matches(base: Path, num_cams: int):
    """Return {cam_index: {id: (det_x, det_y)}} from calib_matches/*.txt."""
    out = {}
    for cam in range(num_cams):
        path = base / "cal" / "calib_matches" / f"cam{cam + 1}_matches.txt"
        d = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            d[int(parts[0])] = (float(parts[1]), float(parts[2]))
        out[cam] = d
    return out


def flat_coords(points_by_id: dict, cpar, cal) -> dict:
    """Run a {id: (x,y)} dict through the same pixel->metric->flat correction
    real correspondence matching uses, keeping the id association."""
    ids = list(points_by_id.keys())
    targets = TargetArray([
        Target(pnr=i, x=points_by_id[pid][0], y=points_by_id[pid][1], n=1, nx=1, ny=1, sumg=1, tnr=-1)
        for i, pid in enumerate(ids)
    ])
    mc = MatchedCoords(targets, cpar, cal, reset_numbers=False)
    return {ids[c.pnr]: (c.x, c.y) for c in mc._corrected}


def main():
    if len(sys.argv) < 2:
        print("Usage: estimate_eps0_from_calibration.py <dataset>", file=sys.stderr)
        return 1

    base = Path(sys.argv[1]).resolve()
    yaml_path = _find_yaml(base)
    if yaml_path is None:
        print(f"ERROR: no parameters_*.yaml found in {base}", file=sys.stderr)
        return 1
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    num_cams = cpar.num_cams

    cals = []
    for i in range(num_cams):
        _, ori, addpar = cam_files(base, i)
        c = Calibration()
        c.from_file(str(ori), str(addpar))
        cals.append(c)

    matches = load_matches(base, num_cams)
    flat = {cam: flat_coords(matches[cam], cpar, cals[cam]) for cam in range(num_cams)}

    residuals_mm = []
    print(f"{'pair':<8}{'n common':<10}{'median(px)':<12}{'p90(px)':<10}{'p99(px)':<10}")
    for i, j in combinations(range(num_cams), 2):
        common_ids = sorted(set(matches[i]) & set(matches[j]))
        if not common_ids:
            continue
        pair_res = []
        for pid in common_ids:
            xl, yl = flat[i][pid]
            xa, ya, xb, yb = epi_mm(xl, yl, cals[i], cals[j], cpar.mm, vpar)
            xj, yj = flat[j][pid]
            if xa == xb:
                d = abs(xj - xa)
            else:
                m = (yb - ya) / (xb - xa)
                b = ya - m * xa
                d = abs((yj - m * xj - b) / np.sqrt(m * m + 1))
            pair_res.append(d)
            residuals_mm.append(d)
        pair_res_px = np.array(pair_res) / cpar.pix_x
        print(f"{'cam' + str(i + 1) + '-' + str(j + 1):<8}{len(common_ids):<10}"
              f"{np.median(pair_res_px):<12.3f}{np.percentile(pair_res_px, 90):<10.3f}"
              f"{np.percentile(pair_res_px, 99):<10.3f}")

    residuals_mm = np.array(residuals_mm)
    residuals_px = residuals_mm / cpar.pix_x

    print(f"\nAll pairs combined ({len(residuals_mm)} point-pair observations):")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(residuals_mm, p):.4f} mm "
              f"({np.percentile(residuals_mm, p) / cpar.pix_x:.2f} px)")

    suggested = np.percentile(residuals_mm, 95) * 1.5
    print(f"\nSuggested eps0 (~1.5x the p95 epipolar residual, from calibration-plate "
          f"points alone): {suggested:.4f} mm ({suggested / cpar.pix_x:.2f} px)")
    print("This is a starting point derived purely from the calibration's own "
          "cross-camera consistency, not from real particle density/noise -- "
          "confirm it with tune_eps0.py against a real sequence frame once one "
          "exists; particle images can behave differently (occlusion, overlap, "
          "detection noise) than the calibration plate's clean, well-separated dots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
