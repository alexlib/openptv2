#!/usr/bin/env python3
"""Generate synthetic TIFF images from OpenPTV target (*_targets) files.

For each `camN.NNNNN_targets` file in a data folder we render a grayscale
image where every tracer particle is drawn as a 2-D Gaussian PSF whose
*geometric* center sits at the OpenPTV sub-pixel centroid of the target,
offset by 0.5 px so that the pipeline's ``targ_rec`` centroiding (which adds
+0.5) recovers exactly the stored ``(x, y)``.

This closes the ground-truth -> image -> detection loop: run detection on the
generated images with the dataset's ``targ_rec`` parameters and compare the
recovered centers to the targets we started from.

Usage (from repo root):
    uv run python scripts/generate_synthetic_images_from_targets.py \
        --data test_data/burgers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def read_targets(path: Path):
    """Return list of (pnr, x, y, n, nx, ny, sumg, tnr)."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    out = []
    for line in lines[1 : n + 1]:
        p = line.split()
        if len(p) < 8:
            continue
        out.append(
            (
                int(p[0]),
                float(p[1]),
                float(p[2]),
                int(p[3]),
                int(p[4]),
                int(p[5]),
                int(p[6]),
                int(p[7]),
            )
        )
    return out


def render_image(targets, *, width=1024, height=1024, sigma=1.0, amplitude=255.0):
    """Render a grayscale uint8 image with one Gaussian per target.

    The Gaussian is placed at the target's *stored* sub-pixel center offset by
    -0.5 px in both axes.  ``targ_rec`` returns ``weighted_mean + 0.5`` for the
    centroid, so this exact offset recovers the original ``(x, y)``.
    """
    img = np.zeros((height, width), dtype=np.uint8)
    for _pnr, x, y, *_rest in targets:
        gx = x - 0.5
        gy = y - 0.5
        r = int(round(3.0 * sigma)) + 1
        x0, x1 = int(np.floor(gx - r)), int(np.ceil(gx + r))
        y0, y1 = int(np.floor(gy - r)), int(np.ceil(gy + r))
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        d2 = (xx - gx) ** 2 + (yy - gy) ** 2
        vals = amplitude * np.exp(-d2 / (2.0 * sigma * sigma))
        patch = img[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
        np.maximum(patch, vals, out=patch)
        img[y0 : y1 + 1, x0 : x1 + 1] = np.clip(patch, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("test_data/burgers"),
        help="Folder containing *_targets files (images written alongside).",
    )
    ap.add_argument(
        "--targets_dir",
        type=Path,
        default=None,
        help="Folder to scan for *_targets files. Defaults to --data itself.",
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian PSF sigma (px)")
    ap.add_argument(
        "--amplitude", type=float, default=255.0, help="Peak grey value of tracers"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing image files"
    )
    args = ap.parse_args()

    data = args.data
    scan = args.targets_dir if args.targets_dir is not None else data
    target_files = sorted(scan.glob("*_targets"))
    if not target_files:
        raise SystemExit(f"No *_targets files found in {scan}")

    n_imgs = 0
    for tf in target_files:
        targets = read_targets(tf)
        # image name: cam1.10002  (strip the "_targets" suffix)
        img_name = tf.name[: -len("_targets")]
        img_path = tf.with_name(img_name)
        if img_path.exists() and not args.overwrite:
            print(f"skip {img_path.name} (exists; use --overwrite)")
            continue
        img = render_image(
            targets,
            width=args.width,
            height=args.height,
            sigma=args.sigma,
            amplitude=args.amplitude,
        )
        imageio.imwrite(img_path, img)
        n_imgs += 1
        print(f"wrote {img_path.name} ({len(targets)} targets)")

    print(f"\nDone: {n_imgs} synthetic images written to {data}")


if __name__ == "__main__":
    main()
