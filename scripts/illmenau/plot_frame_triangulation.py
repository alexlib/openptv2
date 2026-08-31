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
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import COORD_UNUSED
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
from openptv2.orientation import multi_cam_point_positions

ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))
out = Path(ILLMENAU_DIR)
dst = out / "triangulation"
dst.mkdir(exist_ok=True)
PITCH, NX, NY, REF = 120.0, 6, 7, "00000000"
# Above this, a frame's triangulated pattern is not the plate: the labeller
# assigned dots wrongly.  Below it, what is left is the calibration model.
GRID_DEV_MISLABELLED_MM = 30.0

cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""] * 4,
                  cal_img_base_name=[""] * 4)
cals = []
for ci in range(4):
    c = Calibration()
    c.from_file(str(out / f"cal/cam{ci+1}.tif.ori"), str(out / f"cal/cam{ci+1}.tif.addpar"))
    cals.append(c)
cam_C = np.array([[c.ext_par.x0, c.ext_par.y0, c.ext_par.z0] for c in cals])
CC_MM = round(float(cals[0].int_par.cc), 4)   # reported, never assumed

d = np.load(out / "cal" / "labelled_all_frames.npz")
views = {}
for k in d.files:
    if k.endswith("_ids"):
        c, fr, _ = k.split("_")
        views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])


def triangulate(fr):
    per = {ci: dict(zip(views[(ci, fr)][0].tolist(), views[(ci, fr)][1].tolist()))
           for ci in range(4) if (ci, fr) in views}
    ids = [i for i in sorted({i for m in per.values() for i in m})
           if sum(i in m for m in per.values()) >= 2]
    if len(ids) < 8:
        return None
    t = np.full((len(ids), 4, 2), COORD_UNUSED)
    for k, pid in enumerate(ids):
        for ci, m in per.items():
            if pid in m:
                mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
                a = cals[ci].added_par
                t[k, ci] = dist_to_flat(mx, my, cals[ci].int_par.xh, cals[ci].int_par.yh,
                                        a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she)
    pos, rcm = multi_cam_point_positions(t, cpar, cals)
    rcm = np.asarray(rcm, float)
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1) & np.isfinite(rcm)
    if ok.sum() < 8:
        return None
    return pos[ok], np.array(ids)[ok], rcm[ok]


def ideal_grid(ids):
    """Nominal plate coordinates of those ids, datum dot (ix,iy)=(2,3) at 0."""
    ix, iy = (ids - 1) % NX, (ids - 1) // NX
    return np.stack([(ix - 2) * PITCH, (iy - 3) * PITCH, np.zeros(len(ids))], 1).astype(float)


def kabsch(A, B):
    """Rigidly map A onto B (both (n,3)) -- no scaling, so pitch error stays visible."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return (A - ca) @ R + cb


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
    ctr = pos.mean(0)
    _, _, Vt = np.linalg.svd(pos - ctr)
    e1, e2, nrm = Vt[0], Vt[1], Vt[2]
    resid = (pos - ctr) @ nrm
    rms = float(np.sqrt(np.mean(resid ** 2)))

    idx = {p: k for k, p in enumerate(ids)}
    dx = [np.linalg.norm(pos[idx[p]] - pos[idx[p + 1]]) for p in ids
          if p + 1 in idx and ((p - 1) % NX) < NX - 1]
    dy = [np.linalg.norm(pos[idx[p]] - pos[idx[p + NX]]) for p in ids if p + NX in idx]
    px_ = float(np.median(dx)) if dx else float("nan")
    py_ = float(np.median(dy)) if dy else float("nan")

    # ideal rigid plate fitted onto the triangulated dots -> per-dot label error
    fit = kabsch(ideal_grid(ids), pos)
    lab_err = np.linalg.norm(pos - fit, axis=1)
    rows.append((fr, len(ids), rms, float(np.abs(resid).max()), px_, py_,
                 float(np.median(rcm)), float(np.max(lab_err)),
                 float(np.linalg.norm(ctr))))

    fig = plt.figure(figsize=(15.5, 6.6))
    ax = fig.add_subplot(121, projection="3d")
    lim = max(1.0, float(np.abs(resid).max()))
    sc = ax.scatter(*mpl(pos), c=resid, cmap="coolwarm", s=34, vmin=-lim, vmax=lim,
                    depthshade=False, edgecolors="k", linewidths=.3)
    fig.colorbar(sc, ax=ax, shrink=.6, pad=.10, label="distance from best-fit plane [mm]")
    a, b = (pos - ctr) @ e1, (pos - ctr) @ e2
    quad = np.array([ctr + u * e1 + v * e2 for u, v in
                     [(a.min(), b.min()), (a.max(), b.min()), (a.max(), b.max()),
                      (a.min(), b.max()), (a.min(), b.min())]])
    ax.plot(*mpl(quad), color="tab:green", lw=1.2, alpha=.8)
    ax.scatter(*mpl(cam_C), c="crimson", marker="^", s=60, depthshade=False)
    for ci, C in enumerate(cam_C):
        ax.plot(*mpl(np.array([C, ctr])), color="crimson", lw=.5, alpha=.45)
        ax.text(C[0], C[2], C[1], f" cam{ci+1}", color="crimson", fontsize=8)
    ax.scatter(*mpl(np.zeros(3)), c="gold", marker="*", s=170, edgecolors="k",
               depthshade=False)
    for v, col in ((np.eye(3)[0], "r"), (np.eye(3)[1], "g"), (np.eye(3)[2], "b")):
        q = v * 600.0
        ax.quiver(0, 0, 0, q[0], q[2], q[1], color=col, lw=2, arrow_length_ratio=.15)
    ax.set(xlabel="X [mm]", ylabel="Z [mm]  object->camera", zlabel="Y [mm]  up")
    ax.view_init(elev=22, azim=52)
    ax.set_title(f"frame {fr}   {len(ids)} dots\nplanarity RMS {rms:.3f} mm   "
                 f"plate centre |r| = {np.linalg.norm(ctr):.0f} mm", fontsize=10)

    a2 = fig.add_subplot(122)
    u, v = (pos - ctr) @ e1, (pos - ctr) @ e2
    fu, fv = (fit - ctr) @ e1, (fit - ctr) @ e2
    a2.plot(fu, fv, "s", mfc="none", mec="tab:green", ms=11, label="ideal rigid 6x7 grid")
    for k, pid in enumerate(ids):
        a2.plot([fu[k], u[k]], [fv[k], v[k]], "-", color="tab:orange", lw=1.1)
        a2.annotate(str(pid), (u[k], v[k]), textcoords="offset points", xytext=(5, 4),
                    fontsize=7)
    s2 = a2.scatter(u, v, c=lab_err, cmap="viridis", s=30, zorder=3)
    fig.colorbar(s2, ax=a2, shrink=.85, label="deviation from the rigid grid [mm]")
    a2.set(xlabel="in-plane u [mm]", ylabel="in-plane v [mm]", aspect="equal")
    a2.set_title(f"face-on:  pitch X {px_:.2f} / Y {py_:.2f} mm (nominal {PITCH:.0f})\n"
                 f"ray-convergence miss {np.median(rcm):.2f} mm median   |   grid deviation "
                 f"{np.median(lab_err):.2f} med / {lab_err.max():.1f} max mm", fontsize=10)
    a2.grid(alpha=.3)
    a2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(dst / f"frame_{fr}.png", dpi=110)
    plt.close(fig)
    print(f"{fr}  n={len(ids):3d}  planarity {rms:8.3f}  pitch {px_:7.2f}/{py_:7.2f}  "
          f"RCM {np.median(rcm):7.2f}  max grid dev {lab_err.max():7.1f}  -> frame_{fr}.png")

r = np.array([(x[2], x[4], x[6], x[7], x[8]) for x in rows])
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, col, lab in zip(axs, [r[:, 0], r[:, 2], r[:, 3]],
                        ["planarity RMS [mm]", "median ray-convergence miss [mm]",
                         "max deviation from the rigid grid [mm]"]):
    ax.semilogy(r[:, 4], col, "o")
    ax.set(xlabel="plate centre distance from the world origin [mm]", ylabel=lab)
    ax.grid(alpha=.3, which="both")
n_bad = int((r[:, 3] > GRID_DEV_MISLABELLED_MM).sum())
axs[2].axhline(GRID_DEV_MISLABELLED_MM, color="crimson", ls="--", lw=1)
axs[2].text(0.02, .93,
            f"{GRID_DEV_MISLABELLED_MM:.0f} mm: above this a frame is mislabelled "
            f"({n_bad} of {len(rows)})" if n_bad else
            f"{GRID_DEV_MISLABELLED_MM:.0f} mm mislabelling threshold — no frame exceeds it,\n"
            "so everything above the smallest points is model error, not labelling",
            color="crimson", fontsize=8, va="top", transform=axs[2].transAxes)
fig.suptitle(f"Delivered cc = {CC_MM} mm pinhole model, joint bundle over all plate poses "
             f"(gauge = frame {REF}), applied to all {len(rows)} plate positions")
fig.tight_layout()
fig.savefig(dst / "summary.png", dpi=120)
plt.close(fig)

with (dst / "summary.csv").open("w") as f:
    f.write("frame,n,planarity_rms_mm,planarity_max_mm,pitch_x_mm,pitch_y_mm,"
            "rcm_median_mm,grid_dev_max_mm,centre_dist_mm\n")
    for x in rows:
        f.write(x[0] + "," + str(x[1]) + ","
                + ",".join(f"{q:.4f}" for q in x[2:]) + "\n")
print(f"\n{len(rows)} frames -> {dst}  (frame_*.png, summary.png, summary.csv)")
