#!/usr/bin/env python
"""Turnkey headless calibration CLI for an OpenPTV dataset.

    uv run python scripts/autocalibrate.py test_data/test_cavity
    uv run python scripts/autocalibrate.py test_data/test_cavity \
        --dry-run --no-overlays

Runs external -> sortgrid -> refine -> full_calibration for every camera,
prints a per-camera reprojection-RMS report, saves overlay PNGs, and (unless
--dry-run) writes new .ori/.addpar with *.autobck backups. All real work lives
in openptv2.autocalibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from openptv2.autocalibration import calibrate_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path, help="dataset dir (cal/, parameters/)")
    ap.add_argument(
        "--dry-run", action="store_true", help="do not overwrite .ori/.addpar"
    )
    ap.add_argument("--no-overlays", action="store_true", help="skip overlay PNGs")
    args = ap.parse_args()

    results = calibrate_dataset(
        args.dataset,
        write=not args.dry_run,
        overlays=not args.no_overlays,
    )

    print("=== calibration report ===")
    for r in results:
        print(
            f"cam{r.cam + 1}: matched {r.matched}/{r.nfix}  RMS={r.rms:6.3f}px  "
            f"flags={'+'.join(r.flags)}"
        )
    print(f"\nmean reprojection RMS: {np.mean([r.rms for r in results]):.3f}px")
    print(
        "dry-run: nothing written"
        if args.dry_run
        else "written in place (backups: *.autobck)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
