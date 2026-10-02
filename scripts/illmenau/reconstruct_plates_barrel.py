"""Where were the calibration plates?  Rebuilt from the delivered .ori files.

For each rig (cams 1-4 and cams 5-8) every labelled plate view is triangulated
with the CURRENT .ori/.addpar, so what is drawn is what the calibration actually
reconstructs, not the poses it was fitted from.  Both rigs are then drawn in one
barrel frame (origin on the axis at mid-height, +Y up) the way the other
software's Calibration_Geometry.png does, and three questions are answered:

  1. Are the cameras where they should be?  Per rig, the 4 camera centres are
     fitted to the wall circle (R = 3575 mm).  The fit residual and the heights
     versus the nominal 700 / 2900 mm say whether the rig scale/geometry is sane.
  2. Do the plates sit inside the barrel and fill it?  Plate dots outside the
     wall circle or above the ceiling would expose a bad rig.
  3. Do the plates fill each camera's field of view?  Detected dots are binned
     on the sensor, and the camera-to-plate distance range is reported.

Each rig is its own world (see _config.py).  Its origin is the datum dot of the
first plate, placed on the barrel axis (plate.yaml), so a rig is moved into the
barrel by a pure y translation (datum height).  No rotation about the vertical is
assumed -- both worlds have +Y up, the cameras of rig 1-4 on +Z, of rig 5-8 on -Z.
Cross-check: with that origin the cameras of BOTH rigs land at the same radius
from the axis (~3350 mm), which a wrong origin would not give.  The alternative
-- cameras on the 3575 mm outer wall -- needs two equal and opposite ~250 mm
origin shifts and is printed for comparison only.

    ILLMENAU_RAW=/Users/alex/Downloads/Ilmenau \\
        uv run python scripts/illmenau/reconstruct_plates_barrel.py

Writes <RAW>/plate_reconstruction/{barrel_top,barrel_3d,fov_coverage}.png and
plates.csv (one row per rig/frame).
"""

import csv
import os
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import COORD_UNUSED
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
from openptv2.orientation import multi_cam_point_positions

RAW = Path(os.environ.get("ILLMENAU_RAW", "/Users/alex/Downloads/Ilmenau"))
OUT = RAW / "plate_reconstruction"
RIGS = {
    "1-4": (RAW / "openptv_illmenau_4cam", [1, 2, 3, 4], "tab:orange"),
    "5-8": (RAW / "openptv_illmenau_5678", [5, 6, 7, 8], "tab:blue"),
}
IMX, IMY, PIX = 2560, 2048, 0.005
PITCH, NX, NY, DATUM = 120.0, 6, 7, (2, 3)
R_WALL, H_BARREL = 3575.0, 3580.0
Y_DATUM_BARREL = -1175.0  # plate.yaml: datum dot, barrel frame (mid-height origin)
NOMINAL_CAM_H = (700.0, 2900.0)  # above the floor, plate.yaml
MIN_POINTS = 8


def cpar_for(n):
    return ControlPar(
        num_cams=n,
        imx=IMX,
        imy=IMY,
        pix_x=PIX,
        pix_y=PIX,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""] * n,
        cal_img_base_name=[""] * n,
    )


def load_rig(folder, cams):
    cal = folder / "cal"
    cals = []
    for n in cams:
        c = Calibration()
        c.from_file(str(cal / f"cam{n}.tif.ori"), str(cal / f"cam{n}.tif.addpar"))
        cals.append(c)
    d = np.load(cal / "labelled_all_frames.npz")
    views = {}
    for k in d.files:
        if k.endswith("_ids"):
            c, fr, _ = k.split("_")
            views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])
    # Only the views the bundle accepted: a partly visible plate with wrong ids
    # fits its own PnP, is excluded from the bundle, and would otherwise be
    # triangulated here into a tilted, non-planar plate.
    poses = cal / "bundle_plate_poses.npz"
    if poses.exists() and "used_views" in np.load(poses).files:
        used = set(np.load(poses)["used_views"].tolist())
        dropped = [k for k in views if f"{k[0]}_{k[1]}" not in used]
        for k in dropped:
            del views[k]
        if dropped:
            print(
                f"  skipping {len(dropped)} views the bundle rejected: {sorted(dropped)}"
            )
    return cals, views


def triangulate(fr, cals, views, cpar):
    per = {
        ci: dict(zip(views[(ci, fr)][0].tolist(), views[(ci, fr)][1].tolist()))
        for ci in range(len(cals))
        if (ci, fr) in views
    }
    ids = [
        i
        for i in sorted({i for m in per.values() for i in m})
        if sum(i in m for m in per.values()) >= 2
    ]
    if len(ids) < MIN_POINTS:
        return None
    t = np.full((len(ids), len(cals), 2), COORD_UNUSED)
    for k, pid in enumerate(ids):
        for ci, m in per.items():
            if pid not in m:
                continue
            mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
            a = cals[ci].added_par
            t[k, ci] = dist_to_flat(
                mx,
                my,
                cals[ci].int_par.xh,
                cals[ci].int_par.yh,
                a.k1,
                a.k2,
                a.k3,
                a.p1,
                a.p2,
                a.scx,
                a.she,
            )
    pos, rcm = multi_cam_point_positions(t, cpar, cals)
    pos, rcm = np.asarray(pos, float), np.asarray(rcm, float)
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1) & np.isfinite(rcm)
    if ok.sum() < MIN_POINTS:
        return None
    return pos[ok], np.array(ids)[ok]


def plane_fit(P):
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]
    return c, n if n[2] >= 0 else -n


def wall_circle_center(cam_xz):
    f = lambda p: np.hypot(cam_xz[:, 0] - p[0], cam_xz[:, 1] - p[1]) - R_WALL  # noqa: E731
    s = least_squares(f, cam_xz.mean(0) * 0.0)
    return s.x, f(s.x)


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for name, (folder, cams, _) in RIGS.items():
        cals, views = load_rig(folder, cams)
        cpar = cpar_for(len(cals))
        cam_xyz = np.array([[c.ext_par.x0, c.ext_par.y0, c.ext_par.z0] for c in cals])
        centre, resid = wall_circle_center(cam_xyz[:, [0, 2]])
        shift = np.array([0.0, -Y_DATUM_BARREL, 0.0])  # world - shift
        cam_r = np.hypot(cam_xyz[:, 0], cam_xyz[:, 2])
        print(
            f"\n(origin on axis) camera radii from the axis: {cam_r.round(0)} mm, "
            f"mean {cam_r.mean():.0f}"
        )
        frames = sorted({fr for (_, fr) in views})
        plates = []
        for fr in frames:
            r = triangulate(fr, cals, views, cpar)
            if r is None:
                continue
            P, ids = r
            c, n = plane_fit(P)
            plates.append(
                dict(
                    frame=fr,
                    P=P,
                    ids=ids,
                    centre=c,
                    normal=n,
                    rms=float(np.sqrt((((P - c) @ n) ** 2).mean())),
                )
            )
        res[name] = dict(
            cals=cals,
            views=views,
            cam_xyz=cam_xyz,
            centre=centre,
            resid=resid,
            plates=plates,
        )
        print(f"\n=== rig {name}: {len(plates)} of {len(frames)} frames triangulated")
        print("camera centres (rig world, mm):")
        for n, p in zip(cams, cam_xyz):
            print(f"  cam{n}: {p.round(0)}")
        print(
            f"alternative, cameras on the {R_WALL:.0f} wall: axis would be at world "
            f"x,z = {centre.round(0)} (residuals {resid.round(0)} mm)"
        )
        print(
            f"camera heights: lowest pair {np.sort(cam_xyz[:, 1])[:2].round(0)}, "
            f"highest pair {np.sort(cam_xyz[:, 1])[2:].round(0)}  "
            f"spacing {np.sort(cam_xyz[:, 1])[2:].mean() - np.sort(cam_xyz[:, 1])[:2].mean():.0f}"
            f" mm (nominal {NOMINAL_CAM_H[1] - NOMINAL_CAM_H[0]:.0f})"
        )
        res[name]["shift"] = shift

    # ---- barrel-frame bookkeeping --------------------------------------
    def to_barrel(name, X):
        s = res[name]["shift"]
        X = np.atleast_2d(X)
        return np.stack([X[:, 0] - s[0], X[:, 1] + Y_DATUM_BARREL, X[:, 2] - s[2]], 1)

    rows = []
    for name in RIGS:
        allp = []
        for p in res[name]["plates"]:
            B = to_barrel(name, p["P"])
            cb = B.mean(0)
            rad = np.hypot(B[:, 0], B[:, 2])
            allp.append(B)
            rows.append(
                [
                    name,
                    p["frame"],
                    len(B),
                    *cb.round(1),
                    round(p["rms"], 2),
                    round(float(rad.max()), 0),
                    round(float(B[:, 1].min()), 0),
                    round(float(B[:, 1].max()), 0),
                ]
            )
        A = np.vstack(allp)
        rad = np.hypot(A[:, 0], A[:, 2])
        out = (rad > R_WALL).sum()
        print(
            f"\nrig {name} plate dots in barrel frame: {len(A)}\n"
            f"  radius max {rad.max():.0f} mm (wall {R_WALL:.0f}); dots outside "
            f"wall {out}\n"
            f"  height {A[:, 1].min():.0f}..{A[:, 1].max():.0f} mm "
            f"(floor {-H_BARREL / 2:.0f}, ceiling {H_BARREL / 2:.0f})\n"
            f"  x {A[:, 0].min():.0f}..{A[:, 0].max():.0f}, "
            f"z {A[:, 2].min():.0f}..{A[:, 2].max():.0f}"
        )
        res[name]["barrel_pts"] = A
    with open(OUT / "plates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rig",
                "frame",
                "n_dots",
                "cx",
                "cy",
                "cz",
                "plane_rms_mm",
                "max_radius",
                "ymin",
                "ymax",
            ]
        )
        w.writerows(rows)

    # ---- barrel_top.png / barrel_3d.png ---------------------------------
    th = np.linspace(0, 2 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.plot(R_WALL * np.cos(th), R_WALL * np.sin(th), "k", lw=1.5)
    r_cam = float(
        np.mean(
            [np.hypot(*to_barrel(n, res[n]["cam_xyz"]).T[[0, 2]]).mean() for n in RIGS]
        )
    )
    ax.plot(r_cam * np.cos(th), r_cam * np.sin(th), "k", lw=0.8, ls="--")
    for name, (_, cams, col) in RIGS.items():
        A = res[name]["barrel_pts"]
        ax.scatter(
            A[:, 0], A[:, 2], s=1.5, c=col, alpha=0.5, label=f"plates, rig {name}"
        )
        C = to_barrel(name, res[name]["cam_xyz"])
        ax.scatter(C[:, 0], C[:, 2], marker="^", s=80, c=col, edgecolors="k", zorder=5)
        for n, c in zip(cams, C):
            ax.annotate(
                f"c{n}", (c[0], c[2]), xytext=(5, 5), textcoords="offset points"
            )
    ax.plot(0, 0, "k+", ms=14)
    ax.set(
        xlabel="X [mm]",
        ylabel="Z [mm]",
        aspect="equal",
        title="Reconstructed calibration plates, top view (barrel frame)",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "barrel_top.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    for y in (-H_BARREL / 2, H_BARREL / 2):
        ax.plot(
            R_WALL * np.cos(th), R_WALL * np.sin(th), np.full_like(th, y), "k", lw=1.2
        )
    for name, (_, cams, col) in RIGS.items():
        A = res[name]["barrel_pts"]
        ax.scatter(
            A[:, 0], A[:, 2], A[:, 1], s=1.5, c=col, alpha=0.5, label=f"rig {name}"
        )
        C = to_barrel(name, res[name]["cam_xyz"])
        ax.scatter(C[:, 0], C[:, 2], C[:, 1], marker="^", s=70, c=col, edgecolors="k")
    ax.set(
        xlabel="X [mm]",
        ylabel="Z [mm]",
        zlabel="Y (up) [mm]",
        xlim=(-R_WALL, R_WALL),
        ylim=(-R_WALL, R_WALL),
        zlim=(-H_BARREL / 2, H_BARREL / 2),
    )
    ax.set_box_aspect((2 * R_WALL, 2 * R_WALL, H_BARREL))
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(OUT / "barrel_3d.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- per-camera field-of-view coverage ------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    NBX, NBY = 8, 8
    print("\nper-camera sensor coverage (8x8 cells hit by >=1 detected dot):")
    for name, (_, cams, col) in RIGS.items():
        for ci, n in enumerate(cams):
            ax = axes.flat[n - 1]
            px = np.vstack(
                [v[1] for (c, _), v in res[name]["views"].items() if c == ci]
            )
            hist, _, _ = np.histogram2d(
                px[:, 0], px[:, 1], bins=[NBX, NBY], range=[[0, IMX], [0, IMY]]
            )
            ax.scatter(px[:, 0], px[:, 1], s=1.5, c=col, alpha=0.5)
            ax.add_patch(plt.Rectangle((0, 0), IMX, IMY, fill=False, ec="k"))
            ax.set(
                xlim=(-50, IMX + 50),
                ylim=(IMY + 50, -50),
                aspect="equal",
                title=f"cam{n}  {int((hist > 0).sum())}/{NBX * NBY} cells",
            )
            # distance camera -> plate dots
            d = np.linalg.norm(
                res[name]["barrel_pts"] - to_barrel(name, res[name]["cam_xyz"][ci]),
                axis=1,
            )
            print(
                f"  cam{n}: {int((hist > 0).sum()):2d}/{NBX * NBY} cells, "
                f"{len(px)} dot detections, plate distance "
                f"{d.min():.0f}..{d.max():.0f} mm"
            )
    fig.suptitle("Detected calibration dots on each sensor (all frames)")
    fig.savefig(OUT / "fov_coverage.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
