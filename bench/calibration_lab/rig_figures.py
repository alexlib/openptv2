# ruff: noqa
"""Figures that show the truth rig (openptv2.synthetic_rig) working: ray paths and the effect of a wrong assumption.

    uv run python bench/calibration_lab/rig_figures.py [out dir]      # default docs/figures
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openptv2.loco import project_pixels
from openptv2.recorrespond import load_parameters
from openptv2.synthetic_rig import TruthRig

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "docs" / "figures"
out.mkdir(parents=True, exist_ok=True)
SYN1 = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest/syn1/test/parameters_Run1.yaml"
cpar, vpar, cals, _ = load_parameters(SYN1)
rig = TruthRig.from_cals(cals, cpar)
c = 0
cam = np.asarray(cals[c].get_pos(), float)
g = np.asarray(cals[c].get_glass_vec(), float)
s_front = np.linalg.norm(g)          # water-side surface at z = 50, air-side at 50 + d (the glass vector points along z here)
d = rig.glass_thickness

# --- Figure 1: the rays in a plane through the camera axis, and a zoom on the glass ------------------------------------
pts = np.array([[x, cam[1], z] for x in (-20.0, 20.0, 60.0) for z in (-15.0, 15.0)])
paths = rig.ray_path(c, pts)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
for a, zoom in zip(ax, (False, True)):
    a.axhspan(s_front, s_front + d, color="#9ec5e8", alpha=0.8, lw=0)
    a.axhspan(-30, s_front, color="#d7ecf7", alpha=0.7, lw=0)
    a.axhspan(s_front + d, 560, color="#f6f3e8", alpha=0.5, lw=0)
    for p, q in zip(paths, pts):
        x = p[:, 0]; z = p[:, 2]
        a.plot(x, z, "-", color="#b23a48", lw=1.1)
        a.plot(q[0], q[2], "o", color="#b23a48", ms=4)
        a.plot([cam[0], q[0]], [cam[2], q[2]], ":", color="#555", lw=0.9)       # the straight line without refraction
    a.set_xlabel("x (mm)")
    if zoom:
        a.set_xlim(-60, 100); a.set_ylim(-25, s_front + d + 25)
        a.set_title("zoom on the glass: the ray bends at both surfaces\n(red: traced, dotted: straight line to the same particle)")
        a.text(95, s_front + d / 2, f"glass n={rig.n_glass}, d={d} mm", ha="right", va="center", fontsize=9)
        a.text(95, 20, f"water n={rig.n_water}", ha="right", fontsize=9)
        a.text(95, s_front + d + 12, f"air n={rig.n_air}", ha="right", fontsize=9)
    else:
        a.set_xlim(cam[0] - 40, 100); a.set_ylim(-30, 560)
        a.plot(cam[0], cam[2], "ks", ms=7); a.text(cam[0] + 3, cam[2] - 12, "camera 1", fontsize=9)
        a.set_ylabel("z (mm)"); a.set_title("camera 1 to six particles (side view)")
fig.tight_layout(); fig.savefig(out / "synthetic_rig_rays.png", dpi=130); plt.close(fig)

# --- Figure 2: the tracer against the library model, and what a wrong assumption does to the image ----------------------
rng = np.random.default_rng(0)
vol = rng.uniform([-12, -8, -18], [60, 66, 19], (6000, 3))
ideal = rig.project(vol)[:, c]
ref = project_pixels(vol, cals[c], cpar)
ok = np.isfinite(ideal[:, 0])
agree = np.linalg.norm(ideal[ok] - ref[ok], axis=1)
variants = {
    "fluid index +0.04": rig.with_(n_water=rig.n_water + 0.04),
    "glass 4 mm thicker": rig.with_(glass_thickness=rig.glass_thickness + 4.0),
    "water-side surface tilted 1.2 deg": rig.with_(tilt_front=[(np.radians(1.2), 0.0)] * 4),
    "air-side surface tilted 1.2 deg (wedge)": rig.with_(tilt_back=[(0.0, np.radians(1.2))] * 4),
}
fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
axs[0].hist(np.log10(np.maximum(agree, 1e-9)), bins=40, color="#3b6ea5")
axs[0].set_title(f"tracer vs library model\nmax {agree.max():.4f} px, median {np.median(agree):.5f} px", fontsize=10)
axs[0].set_xlabel("log10 |difference| (px)"); axs[0].set_ylabel("points")
for a, (name, r2) in zip(axs[1:], variants.items()):
    dd = r2.project(vol)[:, c] - ideal
    mag = np.linalg.norm(dd, axis=1)
    m = np.isfinite(mag)
    sc = a.scatter(ideal[m, 0], ideal[m, 1], c=mag[m], s=3, cmap="viridis")
    a.set_aspect("equal"); a.invert_yaxis()
    a.set_title(f"{name}\nmax {np.nanmax(mag):.3f} px", fontsize=10)
    plt.colorbar(sc, ax=a, fraction=0.046, label="image shift (px)")
fig.tight_layout(); fig.savefig(out / "synthetic_rig_effects.png", dpi=130); plt.close(fig)
print("written", out / "synthetic_rig_rays.png", out / "synthetic_rig_effects.png")
print(f"agreement with the library model: max {agree.max():.5f} px over {ok.sum()} points")
for name, r2 in variants.items():
    mag = np.linalg.norm(r2.project(vol)[:, c] - ideal, axis=1)
    print(f"{name:42s} image shift median {np.nanmedian(mag):.4f} px, max {np.nanmax(mag):.3f} px")
