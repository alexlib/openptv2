# ruff: noqa: E501
#!/usr/bin/env python
"""Visualize the reprojection-error VECTOR FIELD per camera (not just RMS).

Reads cal/calib_matches/camN_matches.txt (written by dump_matches.py -- run
that first) and draws each residual (detected -> reprojected) as an
arrow over the camera's own image, magnified by --scale for visibility
(raw residuals are a few px on a 512px frame -- too small to read direction
from at 1:1). Color = magnitude.

The SPATIAL PATTERN is the diagnostic, not any single number:
  - arrows pointing radially in/out from the image center, magnitude growing
    with distance from center -> radial distortion (k1/k2/k3) is dominant.
  - a "quadrupole" pattern (arrows flip direction across two perpendicular
    axes, not through the center) -> decentering distortion (p1/p2).
  - arrows all roughly the same direction/length everywhere -> a uniform
    shift, i.e. principal point (xh/yh) or a residual pose offset, not a
    lens aberration at all.
  - arrows concentrated at a few specific points, near-zero everywhere else
    -> those are bad detections (occlusion/glare/edge), not a model problem
    -- see --exclude-ids in recalibrate_exterior_only.py / tune_distortion_stepwise.py.

Usage:
    uv run python skills/openptv-calibrate/scripts/plot_residual_field.py <dataset> [--scale 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def main() -> int:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import imageio.v3 as iio
    import yaml

    from openptv2.autocalibration import _find_yaml, cam_files

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument(
        "--scale",
        type=float,
        default=15.0,
        help="arrow magnification factor (default 15x, purely visual)",
    )
    ap.add_argument(
        "--output-dir", default=None, help="default: <dataset>/cal/calib_matches"
    )
    args = ap.parse_args()

    base = Path(args.dataset).resolve()
    matches_dir = base / "cal" / "calib_matches"
    if not matches_dir.exists():
        print(
            f"ERROR: {matches_dir} not found -- run dump_matches.py first",
            file=sys.stderr,
        )
        return 1

    yaml_path = _find_yaml(base)
    y = yaml.safe_load(yaml_path.read_text())
    ptv_params = y["ptv"]
    num_cams = int(y.get("num_cams") or ptv_params.get("num_cams"))

    split_views = None
    if ptv_params.get("splitter"):
        from openptv2.gui.ptv import image_split

        img0_path, _, _ = cam_files(base, 0)
        raw = iio.imread(img0_path)
        if raw.ndim > 2:
            from skimage.color import rgb2gray
            from skimage.util import img_as_ubyte

            raw = img_as_ubyte(rgb2gray(raw[:, :, :3]))
        split_views = image_split(
            raw, order=ptv_params.get("splitter_order") or [0, 1, 3, 2]
        )

    outdir = Path(args.output_dir) if args.output_dir else matches_dir

    for cam in range(num_cams):
        mfile = matches_dir / f"cam{cam + 1}_matches.txt"
        if not mfile.exists():
            print(f"cam{cam + 1}: {mfile} missing, skipping", file=sys.stderr)
            continue
        data = np.loadtxt(mfile, ndmin=2)
        ids, det, rep = data[:, 0].astype(int), data[:, 1:3], data[:, 3:5]
        err = np.sqrt(np.sum((det - rep) ** 2, axis=1))

        if split_views is not None:
            img = split_views[cam]
        else:
            img_path, _, _ = cam_files(base, cam)
            img = iio.imread(img_path)

        from openptv2.calibration_diagnostics import save_residual_field_figure

        dest = outdir / f"cam{cam + 1}_residual_field.png"
        save_residual_field_figure(
            det,
            rep,
            err,
            img,
            dest,
            scale=args.scale,
            title=f"cam{cam + 1}  residual vector field  (n={len(ids)}, "
            f"RMS={np.sqrt(np.mean(err**2)):.2f}px)",
        )
        print(f"cam{cam + 1}: wrote {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
