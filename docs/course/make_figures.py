# ruff: noqa
"""Figures of the course, made from the bundled data (test_data/test_cavity_small). Run from the repository root:

    uv run python docs/course/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from openptv2.recorrespond import load_parameters

root = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small"
out = Path(__file__).resolve().parent / "figures"
cpar, vpar, cals, _ = load_parameters(root / "parameters_Run1.yaml")

# four camera images of one frame
fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
for i, a in enumerate(ax):
    im = np.asarray(Image.open(root / "img" / f"cam{i + 1}.10000"))
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(f"camera {i + 1}, frame 10000", fontsize=9)
    a.set_xticks([0, 128, 255]); a.set_yticks([0, 128, 255])
fig.tight_layout(); fig.savefig(out / "four_images.png", dpi=110); plt.close(fig)

# where the cameras are (top view x-z) and what they look at
fig, a = plt.subplots(figsize=(9, 3.6))
for i, c in enumerate(cals):
    p = np.asarray(c.get_pos(), float)
    m = np.asarray(c.get_rotation_matrix(), float)
    look = m[:, 2] if abs(m[2, 2]) > 0.5 else m[2, :]
    a.plot(p[2], p[0], "ks")
    a.annotate(f"camera {i + 1}", (p[2], p[0]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    a.arrow(p[2], p[0], -np.sign(p[2]) * 260, 0, head_width=12, color="#888", length_includes_head=True)
a.axvspan(-125, 125, color="#d7ecf7", alpha=0.7)
a.axvspan(-131, -125, color="#7fb4e0"); a.axvspan(125, 131, color="#7fb4e0")
a.add_patch(plt.Rectangle((-20, -10), 45, 20, fill=False, ec="#b23a48", lw=2))
a.text(2, 18, "measurement\nvolume", ha="center", color="#b23a48", fontsize=9)
a.set_xlabel("z (mm)"); a.set_ylabel("x (mm)"); a.set_aspect("equal"); a.set_xlim(-650, 650); a.set_ylim(-170, 170)
a.set_title("top view: four cameras, two on each side of a water cell", fontsize=10)
fig.tight_layout(); fig.savefig(out / "camera_layout.png", dpi=110); plt.close(fig)
print("figures written to", out)
