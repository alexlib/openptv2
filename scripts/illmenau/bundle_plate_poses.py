"""Illmenau driver for the joint plate bundle.

The solver, the gates and the vertical prior all live in
``openptv2.plate_bundle``; this file only supplies the dataset's numbers, builds
the initial poses with ``cv2.solvePnP``, and writes the ``.ori``.

Held fixed on purpose: ``cc`` (the value verified by hand in the GUI on frame
00000000), zero distortion, the principal point at the sensor centre, and the
reference frame's plate pose as the gauge -- so the world stays pinned to the
coded L-corner dot and ``cal/calibration_block.txt``, ``plate.yaml:datum`` and
any manual check of that frame stay valid.

Three gates reject views BEFORE the bundle, each catching what the previous one
cannot see.  Robust loss and residual trimming alone are not enough: a bad view
still drags the early iterations.

  1. per-camera PnP -- uses no cross-camera information, so its residual is a
     pure labelling test for one view
  2. plate vertical -- the plate is held vertical, so a pose tens of degrees off
     means the labelling is wrong however well it fits its own points
  3. per-dot cross-camera agreement -- per DOT, not per plate centre: a scramble
     can leave the centroid roughly in place while the pattern around it is wrong

Usage:  bundle_plate_poses.py [cc_mm] [--write]
Without --write nothing is changed.  With it, the old files are kept as
cal/camN.tif.ori.prebundle.
"""
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from openptv2.calibration_import import calibration_from_opencv
from openptv2.plate_bundle import (
    PlateObservations,
    agreeing_views,
    bundle_plate_poses,
    project,
    rodrigues,
    rotvec,
    tilt_off_vertical_deg,
)

ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))
out = Path(ILLMENAU_DIR)

PITCH, NX, PIX, IMX, IMY, REF = 120.0, 6, 0.005, 2560, 2048, "00000000"
DATUM_IX, DATUM_IY = 2, 3
NCAM, MIN_DOTS = 4, 12
VIEW_GATE_PX = float(os.environ.get("BUNDLE_VIEW_GATE_PX", 1.0))
AGREE_MM = float(os.environ.get("BUNDLE_AGREE_MM", 100.0))
TILT_GATE_DEG = float(os.environ.get("BUNDLE_TILT_GATE_DEG", 5.0))
VERT_SIGMA_DEG = float(os.environ.get("BUNDLE_VERT_SIGMA_DEG", 1.0))
VERT_PX = float(os.environ.get("BUNDLE_VERT_PX", 10.0))
TRIM_MAD = float(os.environ.get("BUNDLE_TRIM_MAD", 3.0))
TRIM_ROUNDS = int(os.environ.get("BUNDLE_TRIM_ROUNDS", 6))

CC = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 8.5858
WRITE = "--write" in sys.argv
K = np.array([[CC / PIX, 0, IMX / 2], [0, CC / PIX, IMY / 2], [0, 0, 1.0]])
D0 = np.zeros(5)

d = np.load(out / "cal" / "labelled_all_frames.npz")
views = {}
for k in d.files:
    if k.endswith("_ids"):
        c, fr, _ = k.split("_")
        views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])
frames_all = sorted({f for _, f in views})


def obj_of(ids):
    ix, iy = (np.asarray(ids) - 1) % NX, (np.asarray(ids) - 1) // NX
    return np.stack([(ix - DATUM_IX) * PITCH, (iy - DATUM_IY) * PITCH,
                     np.zeros(len(ix))], 1).astype(float)


GRID = obj_of(np.arange(1, 43))


def pnp(ids, px):
    o = obj_of(ids)
    if len(o) < 6:
        return None
    ok, rv, tv = cv2.solvePnP(o, px.astype(float), K, D0)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(o, px.astype(float), K, D0, rv, tv)
    rep, _ = cv2.projectPoints(o, rv, tv, K, D0)
    return rv.ravel(), tv.ravel(), float(
        np.sqrt(np.mean(np.sum((rep.reshape(-1, 2) - px) ** 2, 1))))


# ------------------------------------------------- gate 1: per-camera labelling
good = {}
for fr in frames_all:
    for ci in range(NCAM):
        if (ci, fr) not in views or len(views[(ci, fr)][0]) < MIN_DOTS:
            continue
        p = pnp(*views[(ci, fr)])
        if p is not None and p[2] < VIEW_GATE_PX:
            good[(ci, fr)] = p
if any((ci, REF) not in good for ci in range(NCAM)):
    raise SystemExit(f"reference frame {REF} is not clean in all {NCAM} cameras")
n_pnp = len(good)

ref_R = {ci: rodrigues(good[(ci, REF)][0]) for ci in range(NCAM)}
ref_t = {ci: good[(ci, REF)][1] for ci in range(NCAM)}


def dots_in_world(ci, fr):
    rv, tv, _ = good[(ci, fr)]
    return (GRID @ rodrigues(rv).T + tv - ref_t[ci]) @ ref_R[ci]


# ----------------------------------------------------- gate 2: plate is vertical
tilt_rejects = []
for key in list(good):
    ci, fr = key
    if fr == REF:
        continue
    t = tilt_off_vertical_deg(ref_R[ci].T @ rodrigues(good[key][0]))
    if t > TILT_GATE_DEG:
        tilt_rejects.append((fr, ci, t))
        del good[key]

# --------------------------------------------- gate 3: per-dot cross-camera check
kept, dropped = {}, []
for fr in frames_all:
    vs = [ci for ci in range(NCAM) if (ci, fr) in good]
    if len(vs) < 2:
        continue
    per = {ci: dots_in_world(ci, fr) for ci in vs}
    best = agreeing_views(per, AGREE_MM)
    worst = max(np.linalg.norm(per[a] - per[b], axis=1).max()
                for a in vs for b in vs if a < b)
    if not best:
        dropped.append((fr, vs, worst))
        continue
    if len(best) < len(vs):
        dropped.append((fr, [c for c in vs if c not in best], worst))
    kept[fr] = best
good = {(ci, fr): good[(ci, fr)] for fr, cs in kept.items() for ci in cs}
frames = sorted(kept)
free = [fr for fr in frames if fr != REF]

print(f"cc fixed at {CC} mm, zero distortion, gauge = plate pose of frame {REF}")
print(f"gate 1  per-camera PnP < {VIEW_GATE_PX} px:            {n_pnp}/{NCAM*len(frames_all)} views")
print(f"gate 2  plate vertical within {TILT_GATE_DEG:.0f} deg:         "
      f"{len(tilt_rejects)} views rejected"
      + ("  " + ", ".join(f"{fr[-2:]}/cam{ci+1} {t:.0f}deg"
                          for fr, ci, t in sorted(tilt_rejects, key=lambda r: -r[2])[:8])
         if tilt_rejects else ""))
print(f"gate 3  per-dot agreement < {AGREE_MM:.0f} mm:         {len(good)} views, "
      f"{len(frames)} frames")
for fr, vs, spread in sorted(dropped, key=lambda r: -r[2])[:10]:
    print(f"          {fr}  cams {[c+1 for c in vs]}  worst per-dot {spread:7.0f} mm")
print(f"unknowns: {NCAM} camera poses + {len(free)} plate poses = {6*(NCAM+len(free))}")

# ------------------------------------------------------------------ observations
fidx = {fr: k for k, fr in enumerate(free)}
cam_i, frm_i, objp, pixp = [], [], [], []
for fr in frames:
    for ci in range(NCAM):
        if (ci, fr) not in good:
            continue
        ids, px = views[(ci, fr)]
        o = obj_of(ids)
        cam_i.append(np.full(len(o), ci))
        frm_i.append(np.full(len(o), fidx.get(fr, -1)))
        objp.append(o)
        pixp.append(px.astype(float))
obs = PlateObservations(np.concatenate(cam_i), np.concatenate(frm_i),
                        np.concatenate(objp), np.concatenate(pixp))
print(f"{len(obs.cam)} observations, {2*len(obs.cam)} residuals\n")

cam_rvec0 = np.array([good[(ci, REF)][0] for ci in range(NCAM)])
cam_tvec0 = np.array([good[(ci, REF)][1] for ci in range(NCAM)])
prv0, ptv0 = [], []
for fr in free:
    ci = next(c for c in range(NCAM) if (c, fr) in good)
    rv, tv, _ = good[(ci, fr)]
    prv0.append(rotvec(ref_R[ci].T @ rodrigues(rv)))
    ptv0.append(ref_R[ci].T @ (tv - ref_t[ci]))
prv0, ptv0 = np.array(prv0), np.array(ptv0)

x0 = np.concatenate([cam_rvec0.ravel(), cam_tvec0.ravel(),
                     np.column_stack([prv0, ptv0]).ravel()])
e0 = np.linalg.norm(project(x0, obs, K, NCAM, len(free)) - obs.pix, axis=1)
print(f"{'before (anchored)':22s} reproj RMS {np.sqrt(np.mean(e0**2)):6.3f} px   "
      f"median {np.median(e0):6.3f}   max {e0.max():8.3f}")

res = bundle_plate_poses(obs, cam_rvec0, cam_tvec0, prv0, ptv0, K,
                         vertical_px=VERT_PX, vertical_sigma_deg=VERT_SIGMA_DEG,
                         trim_rounds=TRIM_ROUNDS, trim_mad=TRIM_MAD)
for i, (n, rms, thr) in enumerate(res.trim_history):
    print(f"  round {i}: {n:5d} dots in, reproj RMS {rms:7.3f} px  ->  trim at {thr:5.2f} px")
e = res.residual_px[res.keep]
print(f"\nconverged on {int(res.keep.sum())}/{len(obs.cam)} dots across {len(frames)} frames")
print(f"{'after (bundled)':22s} reproj RMS {np.sqrt(np.mean(e**2)):6.3f} px   "
      f"median {np.median(e):6.3f}   p99 {np.percentile(e, 99):7.3f}   max {e.max():8.3f}")

tilts = np.array([tilt_off_vertical_deg(rodrigues(r)) for r in res.plate_rvec])
print(f"plate tilt off vertical: median {np.median(tilts):.2f} deg, max {tilts.max():.2f} deg"
      f"  (prior {VERT_PX} px per {VERT_SIGMA_DEG} deg; BUNDLE_VERT_PX=0 disables)")

print("\ncam        anchored (X,Y,Z)              bundled (X,Y,Z)              moved by")
for ci in range(NCAM):
    a = -rodrigues(cam_rvec0[ci]).T @ cam_tvec0[ci]
    b = res.camera_centre(ci)
    print(f" {ci+1}   ({a[0]:8.1f},{a[1]:7.1f},{a[2]:8.1f})   "
          f"({b[0]:8.1f},{b[1]:7.1f},{b[2]:8.1f})   {np.linalg.norm(b-a):7.1f} mm")
print("\npairwise camera distances (frame-invariant)   anchored / bundled [mm]")
for a, b in ((0, 2), (1, 3), (0, 1), (2, 3)):
    def cen0(ci):
        return -rodrigues(cam_rvec0[ci]).T @ cam_tvec0[ci]
    print(f"  cam{a+1}-cam{b+1}: {np.linalg.norm(cen0(a)-cen0(b)):7.1f} / "
          f"{np.linalg.norm(res.camera_centre(a)-res.camera_centre(b)):7.1f}")
print(f"\nreference plate pose held at identity -- the world origin is still the "
      f"L-corner dot of frame {REF}")

if WRITE:
    for ci in range(NCAM):
        for ext in ("ori", "addpar"):
            src = out / "cal" / f"cam{ci+1}.tif.{ext}"
            if src.exists() and not src.with_suffix(f".{ext}.prebundle").exists():
                shutil.copy2(src, src.with_suffix(f".{ext}.prebundle"))
        cal, _ = calibration_from_opencv(K, D0, res.cam_rvec[ci], res.cam_tvec[ci],
                                         imx=IMX, imy=IMY, pix_x=PIX,
                                         pixel_origin="corner")
        cal.to_file(str(out / "cal" / f"cam{ci+1}.tif.ori"),
                    str(out / "cal" / f"cam{ci+1}.tif.addpar"))
    np.savez(out / "cal" / "bundle_plate_poses.npz", frames=np.array(free),
             plate_rvec=res.plate_rvec, plate_tvec=res.plate_tvec, cc=CC)
    print("\nwrote .ori + zeroed .addpar (first run kept the old ones as *.prebundle)")
    print("now re-run check_plate_triangulation.py, check_epipolar.py and "
          "plot_frame_triangulation.py")
else:
    print("\ndry run -- pass --write to overwrite cal/camN.tif.ori")
