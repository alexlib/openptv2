"""Click-to-mark bubble centroids on the Illmenau dataset, in a native Tk window.

Displays the image after openptv2's standard background removal (box-blur
subtract) plus a contrast stretch -- the only view that makes the faint bubble
glare-points visible at all. Left-click marks a point (drawn immediately as a
red '+', so there's real visual feedback); scroll wheel zooms in/out around the
cursor; the toolbar's pan/zoom tools still work (clicks are ignored while one
of them is active, so you can zoom with the toolbar, then click precisely).
'u' undoes the last mark, 'c' clears all marks. Every mark is saved to disk
immediately, so points survive even if the window is closed without a clean
exit.

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

from openptv2.algorithms.parameters import ControlPar
from openptv2.image_processing import preprocess_image

DEFAULT_IMG_PATH = r"C:\Users\alex\Downloads\Illmenau\Messung_5\00002400.tiff"


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

    fig, ax = plt.subplots(figsize=(12, 9.6))
    ax.imshow(display_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlim(0, imx)
    ax.set_ylim(imy, 0)

    mark_artists = []

    def redraw_title():
        ax.set_title(
            f"{len(marks)} bubble(s) marked  --  "
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
