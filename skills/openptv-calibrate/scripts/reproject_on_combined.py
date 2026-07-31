#!/usr/bin/env python
"""Reproject calblock 3D points onto the non-split (combined) splitter image.

Each camera's calibration only knows pixel coordinates within its own
half-size quadrant. To place a reprojection on the full combined frame,
project per-camera as usual, then shift by that camera's quadrant offset
(derived from ptv.splitter_order in parameters_Run1.yaml).

Useful for a single combined sanity-check image showing all 4 cameras'
calibration quality at once, and for confirming the splitter_order
convention (cam-to-quadrant assignment) is correct for a given rig -- see
`verify_splitter_order` below, which checks camN.tif against both possible
orderings and reports which one actually matches the raw pixel data (do not
trust the convention by assumption; it's cheap to check empirically for
every new rig).

Run with:
  uv run python skills/openptv-calibrate/scripts/reproject_on_combined.py <dataset> <path-to-combined.tif>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.sortgrid import read_calblock
from openptv2.autocalibration import (
    _load_dataset_params,
    _reproject_px,
    cam_files,
    resolve_calblock,
)

# raw quadrant slice offsets (row_offset, col_offset) for a half x half split
QUADRANT_OFFSET = {
    "top-left": (0, 0),
    "top-right": (0, 512),
    "bottom-left": (512, 0),
    "bottom-right": (512, 512),
}
QUADRANT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]


def cam_quadrants(base: Path, num_cams: int) -> list[str]:
    yaml_path = base / "parameters_Run1.yaml"
    cfg = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    splitter_order = (cfg.get("ptv", {}) or {}).get("splitter_order") or [0, 1, 3, 2]
    return [QUADRANT_NAMES[splitter_order[i]] for i in range(num_cams)]


def verify_splitter_order(base: Path, combined_path: Path, num_cams: int) -> None:
    """Empirically check the current splitter_order (and the alternate simple
    raster order) against real pixel data, rather than trusting either by
    assumption. Prints mean abs pixel diff per camera for each candidate;
    the correct order should show near-zero diff, a wrong one a large diff.
    """
    import imageio.v3 as iio

    from openptv2.gui.ptv import image_split

    combined_img = iio.imread(combined_path)
    candidates = {
        "current (from splitter_order)": None,
        "alternate raster [0,1,2,3] (TL,TR,BL,BR)": [0, 1, 2, 3],
    }
    yaml_path = base / "parameters_Run1.yaml"
    cfg = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    candidates["current (from splitter_order)"] = (
        (cfg.get("ptv", {}) or {}).get("splitter_order") or [0, 1, 3, 2]
    )

    for label, order in candidates.items():
        quads = image_split(combined_img, order=order)
        print(f"--- {label}: order={order} ---")
        for i in range(num_cams):
            known_path, _, _ = cam_files(base, i)
            if not known_path.exists():
                continue
            known = iio.imread(known_path)
            q = quads[i]
            if known.shape != q.shape:
                print(f"  cam{i + 1}: shape mismatch {known.shape} vs {q.shape}")
                continue
            diff = np.abs(known.astype(int) - q.astype(int))
            print(f"  cam{i + 1}: mean abs diff={diff.mean():.2f}")


def main():
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sys.argv) < 3:
        print("Usage: reproject_on_combined.py <dataset> <path-to-combined.tif> [--verify-order]",
              file=sys.stderr)
        return 1

    base = Path(sys.argv[1]).resolve()
    combined_path = Path(sys.argv[2])

    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams = dp.cpar, dp.num_cams

    if "--verify-order" in sys.argv:
        verify_splitter_order(base, combined_path, num_cams)

    quadrants = cam_quadrants(base, num_cams)
    combined_img = iio.imread(combined_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(combined_img, cmap="gray")

    colors = ["red", "cyan", "yellow", "magenta"]
    for cam in range(num_cams):
        _, ori, addpar = cam_files(base, cam)
        cal = Calibration.from_file(str(ori), str(addpar))
        row_off, col_off = QUADRANT_OFFSET[quadrants[cam]]

        pts = np.array([_reproject_px(cal, cpar.mm, p, cpar) for p in fix])
        ids = np.arange(1, len(fix) + 1)  # fix rows are in sequential-ID order
        # keep only points that land inside this camera's own quadrant frame
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < 512) & (pts[:, 1] >= 0) & (pts[:, 1] < 512)
        pts, ids = pts[inside], ids[inside]

        x, y = pts[:, 0] + col_off, pts[:, 1] + row_off
        ax.scatter(x, y, s=10, c=colors[cam % len(colors)],
                   label=f"cam{cam + 1} ({quadrants[cam]})")
        for pid, px, py in zip(ids, x, y):
            ax.annotate(str(pid), (px, py), fontsize=5, color=colors[cam % len(colors)],
                       textcoords="offset points", xytext=(2, 2))

    # quadrant boundary lines
    ax.axhline(512, color="white", lw=0.8, alpha=0.6)
    ax.axvline(512, color="white", lw=0.8, alpha=0.6)

    ax.set_title(f"Calblock reprojected onto combined frame: {combined_path.name}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()

    dest = base / "reproject_on_combined.png"
    fig.savefig(dest, dpi=120)
    print(f"Saved {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
