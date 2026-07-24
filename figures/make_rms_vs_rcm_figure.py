"""Generate the RMS-vs-RCM explanatory figure (docs/figures/rms_vs_rcm.png).

Two panels, same in-image residual, different stereo angle -> shows how a small
reprojection error (RMS) maps to a large 3D ray-miss (RCM) at shallow parallax.

    uv run python docs/figures/make_rms_vs_rcm_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rays(ax, half_angle_deg, title):
    """Draw two camera rays toward a target point with a small pointing error,
    and shade the resulting along-ray miss region. Geometry is schematic."""
    a = np.radians(half_angle_deg)
    target = np.array([0.0, 0.0])
    # Two camera centres, symmetric about +y, separated by the stereo angle.
    L = 6.0
    cam1 = target + L * np.array([-np.sin(a), np.cos(a)])
    cam2 = target + L * np.array([np.sin(a), np.cos(a)])

    # A small angular pointing error (same for both) -- the calibration residual.
    err = np.radians(2.0)

    def ray_dir(cam, sign):
        base = (target - cam) / np.linalg.norm(target - cam)
        # rotate the ray by +/- err about the camera centre (the calib residual)
        c, s = np.cos(sign * err), np.sin(sign * err)
        return np.array([c * base[0] - s * base[1], s * base[0] + c * base[1]])

    # Rotate the two rays in OPPOSITE senses so the intersection drifts in DEPTH
    # (toward/away), the failure mode shallow parallax amplifies.
    d1, d2 = ray_dir(cam1, +1), ray_dir(cam2, -1)

    # Intersection of the two perturbed rays: cam1 + t1 d1 == cam2 + t2 d2.
    A = np.column_stack([d1, -d2])
    t = np.linalg.solve(A, cam2 - cam1)
    tri = cam1 + t[0] * d1                     # triangulated point
    miss = np.linalg.norm(tri - target)        # 3D positioning error

    for cam, d, col in [(cam1, d1, "#1f77b4"), (cam2, d2, "#d62728")]:
        end = cam + (L + 3.0) * d
        ax.plot([cam[0], end[0]], [cam[1], end[1]], color=col, lw=1.6)
        ax.plot(*cam, "s", color=col, ms=8)
    # true (unperturbed) sight-lines, dashed
    for cam in (cam1, cam2):
        ax.plot([cam[0], target[0]], [cam[1], target[1]], color="gray",
                ls="--", lw=0.8, alpha=0.7)

    ax.plot(*target, "k*", ms=14, label="true 3D point")
    ax.plot(*tri, "o", color="orange", ms=9, label="triangulated point")
    ax.annotate("", xy=tri, xytext=target,
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2))
    ax.text(tri[0] + 0.2, 0.5 * (tri[1] + target[1]), f"RCM ≈ {miss:.2f}",
            color="darkorange", ha="left", fontsize=10, fontweight="bold")

    ax.set_title(f"{title}\nstereo half-angle {half_angle_deg}°  ·  "
                 f"same 2° image residual", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-1.5, 8)
    ax.axis("off")
    return miss


fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
m_wide = rays(axes[0], 35, "Wide parallax (good)")
m_shallow = rays(axes[1], 10, "Shallow parallax (this rig)")
axes[0].legend(loc="lower center", fontsize=8, framealpha=0.9)

fig.suptitle(
    "Same reprojection error (RMS), very different 3D miss (RCM)\n"
    f"the shallow rig's rays miss ~{m_shallow / m_wide:.1f}× farther in depth "
    "for the identical in-image error",
    fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "docs/figures/rms_vs_rcm.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}  (wide miss {m_wide:.2f}, shallow miss {m_shallow:.2f})")
