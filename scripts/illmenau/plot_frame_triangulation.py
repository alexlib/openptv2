"""One 3D PNG per calibration frame: what the delivered .ori/.addpar actually
reconstruct for that plate position.

This script **fits nothing**.  It reads whatever cal/camN.tif.ori currently say
and shows how well that one model reconstructs every plate position, so it is
the honest check on both refit_plate_pinhole.py and bundle_plate_poses.py.  The
`cc` and the model description in the summary figure are read out of the .ori
rather than assumed, so the figure cannot go stale against the files.

Per frame, two panels:
  left   3D view in the global frame -- triangulated dots coloured by their
         distance from the best-fit plane, the fitted plane drawn as a wire
         quad, the four camera positions and their sight lines to the plate
         centre, and the world origin.
  right  the same dots seen face-on (rotated into the fitted plane) with their
         point ids, plus the ideal rigid 6x7 grid Kabsch-fitted onto them.  A
         mislabelled dot shows up here as one id sitting far off its grid node,
         which the 3D view alone cannot make obvious.

Written to $ILLMENAU_DIR/triangulation/frame_XXXXXXXX.png, plus summary.png and
summary.csv of the per-frame numbers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openptv2.plate_multiplane import plate_frame_metrics, triangulate_plate

out = CFG.DIR
dst = out / "triangulation"
dst.mkdir(exist_ok=True)
PITCH, REF = CFG.PITCH, CFG.REF
# Above this, a frame's triangulated pattern is not the plate: the labeller
# assigned dots wrongly.  Below it, what is left is the calibration model.
GRID_DEV_MISLABELLED_MM = 30.0

cpar = CFG.control_par()
cals = CFG.load_calibrations()
cam_C = np.array([[c.ext_par.x0, c.ext_par.y0, c.ext_par.z0] for c in cals])
CC_MM = round(float(cals[0].int_par.cc), 4)  # reported, never assumed

views = CFG.load_views()


def triangulate(fr):
    per = {ci: views[(ci, fr)] for ci in range(CFG.NCAM) if (ci, fr) in views}
    tri = triangulate_plate(per, cals, cpar, min_dots=8)
    return None if tri is None else (tri.pos, tri.ids, tri.rcm)


def mpl(P):
    """world (X,Y,Z) -> matplotlib (x,y,z) so the screen-vertical axis is +Y."""
    P = np.atleast_2d(np.asarray(P, float))
    return P[:, 0], P[:, 2], P[:, 1]


rows = []
for fr in sorted({f for _, f in views}):
    r = triangulate(fr)
    if r is None:
        print(f"{fr}: too few dots, skipped")
        continue
    pos, ids, rcm = r
    m = plate_frame_metrics(pos, ids, CFG.GRID)
    ctr = m.centre
    e1, e2, nrm = m.axes
    resid = m.residual
    rms = m.planarity_rms
    px_ = float(np.median(m.pitch_x)) if len(m.pitch_x) else float("nan")
    py_ = float(np.median(m.pitch_y)) if len(m.pitch_y) else float("nan")

    # ideal rigid plate fitted onto the triangulated dots -> per-dot label error
    fit = m.grid_fit
    lab_err = m.grid_dev
    rows.append(
        (
            fr,
            len(ids),
            rms,
            float(np.abs(resid).max()),
            px_,
            py_,
            float(np.median(rcm)),
            float(np.max(lab_err)),
            float(np.linalg.norm(ctr)),
        )
    )

    fig = plt.figure(figsize=(15.5, 6.6))
    ax = fig.add_subplot(121, projection="3d")
    lim = max(1.0, float(np.abs(resid).max()))
    sc = ax.scatter(
        *mpl(pos),
        c=resid,
        cmap="coolwarm",
        s=34,
        vmin=-lim,
        vmax=lim,
        depthshade=False,
        edgecolors="k",
        linewidths=0.3,
    )
    fig.colorbar(
        sc, ax=ax, shrink=0.6, pad=0.10, label="distance from best-fit plane [mm]"
    )
    a, b = (pos - ctr) @ e1, (pos - ctr) @ e2
    quad = np.array(
        [
            ctr + u * e1 + v * e2
            for u, v in [
                (a.min(), b.min()),
                (a.max(), b.min()),
                (a.max(), b.max()),
                (a.min(), b.max()),
                (a.min(), b.min()),
            ]
        ]
    )
    ax.plot(*mpl(quad), color="tab:green", lw=1.2, alpha=0.8)
    ax.scatter(*mpl(cam_C), c="crimson", marker="^", s=60, depthshade=False)
    for ci, cc_pos in enumerate(cam_C):
        ax.plot(*mpl(np.array([cc_pos, ctr])), color="crimson", lw=0.5, alpha=0.45)
        ax.text(
            cc_pos[0],
            cc_pos[2],
            cc_pos[1],
            f" cam{CFG.cam_number(ci)}",
            color="crimson",
            fontsize=8,
        )
    ax.scatter(
        *mpl(np.zeros(3)), c="gold", marker="*", s=170, edgecolors="k", depthshade=False
    )
    for v, col in ((np.eye(3)[0], "r"), (np.eye(3)[1], "g"), (np.eye(3)[2], "b")):
        q = v * 600.0
        ax.quiver(0, 0, 0, q[0], q[2], q[1], color=col, lw=2, arrow_length_ratio=0.15)
    ax.set(xlabel="X [mm]", ylabel="Z [mm]  object->camera", zlabel="Y [mm]  up")
    ax.view_init(elev=22, azim=52)
    ax.set_title(
        f"frame {fr}   {len(ids)} dots\nplanarity RMS {rms:.3f} mm   "
        f"plate centre |r| = {np.linalg.norm(ctr):.0f} mm",
        fontsize=10,
    )

    a2 = fig.add_subplot(122)
    u, v = (pos - ctr) @ e1, (pos - ctr) @ e2
    fu, fv = (fit - ctr) @ e1, (fit - ctr) @ e2
    a2.plot(
        fu, fv, "s", mfc="none", mec="tab:green", ms=11, label="ideal rigid 6x7 grid"
    )
    for k, pid in enumerate(ids):
        a2.plot([fu[k], u[k]], [fv[k], v[k]], "-", color="tab:orange", lw=1.1)
        a2.annotate(
            str(pid),
            (u[k], v[k]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )
    s2 = a2.scatter(u, v, c=lab_err, cmap="viridis", s=30, zorder=3)
    fig.colorbar(s2, ax=a2, shrink=0.85, label="deviation from the rigid grid [mm]")
    a2.set(xlabel="in-plane u [mm]", ylabel="in-plane v [mm]", aspect="equal")
    a2.set_title(
        f"face-on:  pitch X {px_:.2f} / Y {py_:.2f} mm (nominal {PITCH:.0f})\n"
        f"ray-convergence miss {np.median(rcm):.2f} mm median   |   grid deviation "
        f"{np.median(lab_err):.2f} med / {lab_err.max():.1f} max mm",
        fontsize=10,
    )
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(dst / f"frame_{fr}.png", dpi=110)
    plt.close(fig)
    print(
        f"{fr}  n={len(ids):3d}  planarity {rms:8.3f}  pitch {px_:7.2f}/{py_:7.2f}  "
        f"RCM {np.median(rcm):7.2f}  max grid dev {lab_err.max():7.1f}  -> frame_{fr}.png"
    )

r = np.array([(x[2], x[4], x[6], x[7], x[8]) for x in rows])
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, col, lab in zip(
    axs,
    [r[:, 0], r[:, 2], r[:, 3]],
    [
        "planarity RMS [mm]",
        "median ray-convergence miss [mm]",
        "max deviation from the rigid grid [mm]",
    ],
):
    ax.semilogy(r[:, 4], col, "o")
    ax.set(xlabel="plate centre distance from the world origin [mm]", ylabel=lab)
    ax.grid(alpha=0.3, which="both")
n_bad = int((r[:, 3] > GRID_DEV_MISLABELLED_MM).sum())
axs[2].axhline(GRID_DEV_MISLABELLED_MM, color="crimson", ls="--", lw=1)
axs[2].text(
    0.02,
    0.93,
    f"{GRID_DEV_MISLABELLED_MM:.0f} mm: above this a frame is mislabelled "
    f"({n_bad} of {len(rows)})"
    if n_bad
    else f"{GRID_DEV_MISLABELLED_MM:.0f} mm mislabelling threshold — no frame exceeds it,\n"
    "so everything above the smallest points is model error, not labelling",
    color="crimson",
    fontsize=8,
    va="top",
    transform=axs[2].transAxes,
)
fig.suptitle(
    f"Delivered cc = {CC_MM} mm pinhole model, joint bundle over all plate poses "
    f"(gauge = frame {REF}), applied to all {len(rows)} plate positions"
)
fig.tight_layout()
fig.savefig(dst / "summary.png", dpi=120)
plt.close(fig)

with (dst / "summary.csv").open("w") as f:
    f.write(
        "frame,n,planarity_rms_mm,planarity_max_mm,pitch_x_mm,pitch_y_mm,"
        "rcm_median_mm,grid_dev_max_mm,centre_dist_mm\n"
    )
    for x in rows:
        f.write(
            x[0] + "," + str(x[1]) + "," + ",".join(f"{q:.4f}" for q in x[2:]) + "\n"
        )
print(f"\n{len(rows)} frames -> {dst}  (frame_*.png, summary.png, summary.csv)")
