"""Generate demo assets for the 4-view splitter tutorial.

Produces:
  images/demo_4view.tif        -- a synthetic 1024x1024 single-sensor image whose
                                  four 512x512 quadrants show the SAME particle
                                  cloud from four slightly different viewpoints.
  images/quadrant_mapping.png  -- an annotated diagram of the quadrant -> camera
                                  mapping and the image_split(order=[0,1,3,2]).

Run:  uv run python docs/tutorials/four_view_splitter/make_demo_assets.py
"""

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).parent
IMG = HERE / "images"
IMG.mkdir(parents=True, exist_ok=True)

QUAD = 512  # quadrant size (pixels)
RNG = np.random.default_rng(42)
N_PARTICLES = 60


def _render_view(points_xy, jitter):
    """Render one 512x512 8-bit view of a shared particle cloud.

    points_xy : (N, 2) base positions in quadrant pixel coordinates.
    jitter    : per-view (dx, dy) parallax shift so the four views differ,
                mimicking four cameras looking at the same volume.
    """
    view = np.zeros((QUAD, QUAD), dtype=np.float64)
    yy, xx = np.mgrid[0:QUAD, 0:QUAD]
    for (px, py) in points_xy:
        cx = px + jitter[0]
        cy = py + jitter[1]
        if not (8 <= cx < QUAD - 8 and 8 <= cy < QUAD - 8):
            continue
        peak = RNG.uniform(160, 255)
        sigma = RNG.uniform(1.4, 2.2)
        view += peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    return np.clip(view, 0, 255).astype(np.uint8)


def make_demo_image():
    # Shared 3D-ish cloud projected to a base 2D layout, then parallax-shifted
    base = RNG.uniform(60, QUAD - 60, size=(N_PARTICLES, 2))
    # Four viewpoints: small distinct parallax per virtual camera
    jitters = [(-6, -4), (7, -5), (-5, 8), (6, 7)]
    views = [_render_view(base, j) for j in jitters]

    full = np.zeros((2 * QUAD, 2 * QUAD), dtype=np.uint8)
    # Physical quadrant layout on the sensor:
    #   index 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    full[:QUAD, :QUAD] = views[0]
    full[:QUAD, QUAD:] = views[1]
    full[QUAD:, :QUAD] = views[2]
    full[QUAD:, QUAD:] = views[3]

    out = IMG / "demo_4view.tif"
    iio.imwrite(out, full)
    print(f"wrote {out}  shape={full.shape} dtype={full.dtype}")
    return full


def make_mapping_diagram(full):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6))

    # Left: the raw single-sensor image with the split grid overlaid
    axL.imshow(full, cmap="gray", origin="upper")
    axL.axhline(QUAD, color="#ff5555", lw=1.5)
    axL.axvline(QUAD, color="#ff5555", lw=1.5)
    labels = {(0, 0): "quadrant 0\n(top-left)", (0, 1): "quadrant 1\n(top-right)",
              (1, 0): "quadrant 2\n(bottom-left)", (1, 1): "quadrant 3\n(bottom-right)"}
    for (r, c), txt in labels.items():
        axL.text(c * QUAD + QUAD / 2, r * QUAD + QUAD / 2, txt,
                 color="#ffd166", ha="center", va="center", fontsize=11, weight="bold")
    axL.set_title("Raw sensor image (1024x1024)\nsplit into four 512x512 quadrants")
    axL.set_xticks([0, QUAD, 2 * QUAD])
    axL.set_yticks([0, QUAD, 2 * QUAD])

    # Right: quadrant -> camera mapping under order=[0, 1, 3, 2]
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.axis("off")
    axR.set_title("image_split(order=[0, 1, 3, 2])\nquadrant index -> camera index")
    order = [0, 1, 3, 2]
    quad_names = ["0 top-left", "1 top-right", "2 bottom-left", "3 bottom-right"]
    for cam_idx, quad_idx in enumerate(order):
        y = 8.5 - cam_idx * 2.2
        axR.add_patch(Rectangle((0.5, y - 0.6), 3.2, 1.4, fc="#2a3b4d", ec="#8ecae6"))
        axR.text(2.1, y + 0.1, f"quadrant {quad_names[quad_idx]}",
                 color="#8ecae6", ha="center", va="center", fontsize=10)
        axR.add_patch(Rectangle((6.3, y - 0.6), 3.2, 1.4, fc="#3d2a4d", ec="#c39bd3"))
        axR.text(7.9, y + 0.1, f"camera {cam_idx}\ncam_{cam_idx + 1}.tif.ori",
                 color="#c39bd3", ha="center", va="center", fontsize=10)
        axR.add_patch(FancyArrowPatch((3.8, y + 0.1), (6.2, y + 0.1),
                                      arrowstyle="-|>", mutation_scale=16, color="#adb5bd"))
    axR.text(5, 0.6, "order is dataset-specific: it maps each physical quadrant\n"
                     "to the camera whose .ori/.addpar calibration describes it",
             color="#6c757d", ha="center", va="center", fontsize=8, style="italic")

    fig.tight_layout()
    out = IMG / "quadrant_mapping.png"
    fig.savefig(out, dpi=110, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    full = make_demo_image()
    make_mapping_diagram(full)
