"""Click-to-mark bubble centroids on the Illmenau dataset, in a native Tk window,
plus an automatic detector overlay to check against.

Displays the image after openptv2's standard background removal (box-blur
subtract) -- the only view that makes the faint bubble glare-points visible at
all. Left-click marks a point (drawn immediately as a red '+', so there's real
visual feedback); scroll wheel zooms in/out around the cursor; the toolbar's
pan/zoom tools still work (clicks are ignored while one of them is active, so
you can zoom with the toolbar, then click precisely). 'u' undoes the last
mark, 'c' clears all marks, 'a' toggles the automatic-detection overlay
(green circles). Every mark is saved to disk immediately, so points survive
even if the window is closed without a clean exit.

Automatic detection (green circles): a bubble's glare-points are only a few
grey levels above their OWN local neighborhood, but that neighborhood's
brightness varies a lot across the image -- so a single global threshold
either misses faint/far bubbles or drowns in false positives from bright
regions elsewhere. Fix: score every pixel against its own local background
(a per-pixel z-score, via a sliding-window local mean/std) instead of one
global cutoff. Every one of 30 hand-labeled bubbles -- bright/close and
faint/far alike -- scored at least 8 local-sigma above its neighborhood, so
z >= 7 (a small safety margin below that) is the working threshold. A plain
Gaussian blur was tried next, to merge each bubble's 2-3 separate glare
fragments into one blob before centroiding (as suggested) -- but it dilutes
the very local-contrast signal the z-score relies on, and recall dropped
fast as the blur washed out real peaks. Binary dilation on the *thresholded*
mask does the same fragment-merging job without that cost: it only touches
pixels already confirmed as signal. The merged mask is then handed to
openptv2's real compiled detector (``targ_rec_fast``, the same BFS
flood-fill + intensity-weighted centroid used by target_recognition) for
the final connected-component centroid -- so the actual detection step
reuses the tested Cython code, not a one-off reimplementation.

Usage:
    uv run python scripts/label_illmenau_bubbles.py [image_path] [marks_path]

Defaults to the wp4/Messung_5 frame used to develop this tool and an
adjacent bubble_marks.json.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.track_kernels_batch import targ_rec_fast
from openptv2.image_processing import preprocess_image

DEFAULT_IMG_PATH = r"C:\Users\alex\Downloads\Illmenau\Messung_5\00002400.tiff"

# Local-normalization + merge-and-detect parameters (see module docstring).
Z_WINDOW = 25  # local mean/std window, in pixels
Z_THRESHOLD = 7.0  # local-sigma cutoff; validated min over 30 real bubbles was 8.3
MERGE_DILATE_ITERS = 4  # bridges a bubble's separated glare fragments
MIN_TARGET_PIXELS = 1
MAX_TARGET_PIXELS = 5000


def detect_bubbles(hp):
    """Auto-detect bubble centroids in a background-removed image.

    Returns an (N, 2) array of (x, y) centroids from openptv2's own
    ``targ_rec_fast``. See the module docstring for the method.
    """
    local_mean = ndimage.uniform_filter(hp, size=Z_WINDOW)
    local_sqmean = ndimage.uniform_filter(hp**2, size=Z_WINDOW)
    local_std = np.sqrt(np.maximum(local_sqmean - local_mean**2, 1e-6))
    z = (hp - local_mean) / local_std

    mask = z >= Z_THRESHOLD
    merged = ndimage.binary_dilation(mask, iterations=MERGE_DILATE_ITERS)

    # z, scaled to a byte range, as the intensity targ_rec_fast weights its
    # centroid by. Pixels the dilation pulled in (whose own z was <= 0, i.e.
    # not signal) are floored to 1 so they don't break BFS connectivity
    # inside an otherwise-merged blob (they're right at the threshold img
    # uses to decide "is this pixel part of the target at all").
    z_scaled = np.clip(z, 0, 25) / 25.0 * 255.0
    masked = np.where(merged, np.maximum(z_scaled, 1.0), 0.0)
    img_u8 = np.ascontiguousarray(np.clip(masked, 0, 255).astype(np.uint8))
    img0 = img_u8.copy()

    imy, imx = img_u8.shape
    n, xs, ys, *_ = targ_rec_fast(
        img_u8,
        img0,
        gvthres=0,
        discont=255,
        nnmin=MIN_TARGET_PIXELS,
        nnmax=MAX_TARGET_PIXELS,
        nxmin=1,
        nxmax=10000,
        nymin=1,
        nymax=10000,
        sumg_min=0,
        xmin=1,
        ymin=1,
        xmax=imx - 1,
        ymax=imy - 1,
        max_targets=200000,
    )
    return np.stack([xs[:n], ys[:n]], axis=1) if n else np.zeros((0, 2))


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG_PATH
    marks_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("bubble_marks.json")

    raw = iio.imread(img_path)
    imy, imx = raw.shape

    # raw is 12-bit data left-shifted 4 bits into a 16-bit container; the top
    # byte is the real 8-bit intensity.
    img8 = (raw >> 8).astype(np.uint8)

    cpar = ControlPar(num_cams=1, imx=imx, imy=imy)
    hp = np.asarray(
        preprocess_image(img8, filter_hp=0, cpar=cpar, lowpass_dim=1)
    ).astype(np.float64)

    lo, hi = np.percentile(hp, 0.0), np.percentile(hp, 99.9)
    hi = max(hi, lo + 1e-6)
    display_img = np.clip((hp - lo) / (hi - lo), 0, 1)

    marks = []
    if marks_path.exists():
        try:
            marks = json.loads(marks_path.read_text())
        except Exception:
            marks = []

    print("Running automatic detection...")
    auto_detections = detect_bubbles(hp)
    print(f"Auto-detected {len(auto_detections)} candidate(s).")

    fig, ax = plt.subplots(figsize=(12, 9.6))
    ax.imshow(display_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlim(0, imx)
    ax.set_ylim(imy, 0)

    auto_visible = [True]
    (auto_artist,) = ax.plot(
        auto_detections[:, 0],
        auto_detections[:, 1],
        marker="o",
        markerfacecolor="none",
        markeredgecolor="lime",
        markersize=10,
        linestyle="none",
        label="auto-detected",
    )

    mark_artists = []

    def redraw_title():
        ax.set_title(
            f"{len(marks)} manually marked (red +)  |  "
            f"{len(auto_detections)} auto-detected (green o, 'a' to toggle)  --  "
            "left-click to mark  |  scroll to zoom  |  'u' undo  |  'c' clear"
        )
        fig.canvas.draw_idle()

    def draw_mark(x, y):
        (artist,) = ax.plot(x, y, marker="+", color="red", markersize=14, mew=2)
        mark_artists.append(artist)

    def save_marks():
        marks_path.write_text(json.dumps(marks, indent=2))

    for m in marks:
        draw_mark(m["x"], m["y"])

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if fig.canvas.toolbar.mode != "":
            return  # a pan/zoom tool is active -- navigation, not a mark
        x, y = float(event.xdata), float(event.ydata)
        marks.append({"x": x, "y": y})
        draw_mark(x, y)
        save_marks()
        redraw_title()

    def on_key(event):
        if event.key == "u" and marks:
            marks.pop()
            mark_artists.pop().remove()
            save_marks()
            redraw_title()
        elif event.key == "c":
            marks.clear()
            for artist in mark_artists:
                artist.remove()
            mark_artists.clear()
            save_marks()
            redraw_title()
        elif event.key == "a":
            auto_visible[0] = not auto_visible[0]
            auto_artist.set_visible(auto_visible[0])
            fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes != ax:
            return
        base_scale = 1.3
        cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        scale = 1 / base_scale if event.button == "up" else base_scale
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim(xdata - new_w * relx, xdata + new_w * (1 - relx))
        ax.set_ylim(ydata - new_h * rely, ydata + new_h * (1 - rely))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    redraw_title()
    plt.tight_layout()
    plt.show()

    # Final save on window close, in case the last action wasn't a mark.
    save_marks()
    print(f"Saved {len(marks)} mark(s) to {marks_path}")


if __name__ == "__main__":
    main()
