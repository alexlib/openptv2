"""Joint bundle adjustment over many plate positions, with cc held fixed.

The delivered .ori solve all four camera poses on the reference frame alone, so
the model is exact on that plane by construction and its error grows linearly
away from it -- measured at ~0.58 % of the plate's distance from the anchor
plane (docs/illmenau-4cam-calibration.md section 8).  This spreads that error
over the whole volume instead of concentrating it far from the anchor.

Unknowns:
    4 camera poses            (rvec, tvec)  world -> camera
    one plate pose per frame  (rvec, tvec)  plate -> world

Held fixed on purpose:
    cc                  the value verified by hand in the GUI on frame 00000000
    distortion          zero, for the reasons in section 4 Trap 1
    principal point     sensor centre
    the reference frame's plate pose = identity

That last one is the gauge, and it is the whole point: the world stays pinned to
the coded L-corner dot of frame 00000000, so cal/calibration_block.txt, the
datum in plate.yaml and any manual GUI check of that frame all stay valid.  With
the gauge fixed there is no free similarity, so scale cannot drift even though
cc is not being fitted.

Frames enter the bundle per VIEW, not per frame: a camera's labelling can be
broken in one image while the other three are fine, so each (camera, frame) view
must pass a per-camera PnP reprojection gate on its own, and a frame is used
when at least two of its views survive.  A bundle fed mislabelled views diverges
-- that is what sank the earlier bundle_shared_cc.py attempt.

Usage:  bundle_plate_poses.py [cc_mm] [--write]
Without --write it reports and changes nothing.  With --write it backs the old
files up to cal/camN.tif.ori.prebundle before overwriting.
"""
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from openptv2.calibration_import import calibration_from_opencv

ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))
out = Path(ILLMENAU_DIR)

PITCH, NX, PIX, IMX, IMY, REF = 120.0, 6, 0.005, 2560, 2048, "00000000"
DATUM_IX, DATUM_IY = 2, 3
VIEW_GATE_PX = 1.0          # per-camera PnP reprojection above this = mislabelled view
TRIM_ROUNDS = int(os.environ.get("BUNDLE_TRIM_ROUNDS", 6))
TRIM_MAD = float(os.environ.get("BUNDLE_TRIM_MAD", 3.0))     # keep dots under MAD x median
TRIM_FLOOR_PX = float(os.environ.get("BUNDLE_TRIM_FLOOR", 1.0))
MIN_DOTS = 12
AGREE_MM = float(os.environ.get("BUNDLE_AGREE_MM", 100.0))
# The plate is held vertical: its own +Y is world +Y and only the yaw about +Y is
# free.  Measured spread over this dataset is <= 1.23 deg, so 5 deg is a loose
# outlier gate and 1.0 deg is the soft prior's sigma.  VERT_PX is what one sigma
# of tilt costs in equivalent pixels -- 0 disables the prior entirely.
TILT_GATE_DEG = float(os.environ.get("BUNDLE_TILT_GATE_DEG", 5.0))
VERT_SIGMA_DEG = float(os.environ.get("BUNDLE_VERT_SIGMA_DEG", 1.0))
VERT_PX = float(os.environ.get("BUNDLE_VERT_PX", 10.0))

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


def obj_of(ids):
    ix, iy = (np.asarray(ids) - 1) % NX, (np.asarray(ids) - 1) // NX
    return np.stack([(ix - DATUM_IX) * PITCH, (iy - DATUM_IY) * PITCH,
                     np.zeros(len(ix))], 1).astype(float)


def pnp(ids, px):
    """Per-camera pose of the rigid plate -- uses no cross-camera information,
    so its residual is a pure labelling test for that one view."""
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


# ---------------------------------------------------------------- select views
frames_all = sorted({f for _, f in views})
good_views = {}
for fr in frames_all:
    for ci in range(4):
        if (ci, fr) not in views or len(views[(ci, fr)][0]) < MIN_DOTS:
            continue
        p = pnp(*views[(ci, fr)])
        if p is not None and p[2] < VIEW_GATE_PX:
            good_views[(ci, fr)] = p
ref_ok = [ci for ci in range(4) if (ci, REF) in good_views]
if len(ref_ok) < 4:
    raise SystemExit(f"reference frame {REF} is not clean in all four cameras: {ref_ok}")
n_pnp = len(good_views)

# ---------------------------------------------- stage 2: cross-camera agreement
# A per-camera PnP residual cannot see a labelling that is wrong but internally
# consistent -- shift the whole grid by one column and it still fits its own
# points perfectly.  Only another camera can catch that.  Using the reference
# frame's camera poses, every surviving view implies where the plate must be in
# the world; correct labellings agree, a mislabelled view lands somewhere else
# entirely.  Keep the largest mutually-agreeing subset of views per frame.
#
# This runs BEFORE the bundle on purpose.  Letting a mislabelled view in and
# relying on robust loss + residual trimming to remove it afterwards still lets
# it drag the initial iterations, which is what left frames 37/38/39/42 in the
# first run of this script.
ref_R = {ci: cv2.Rodrigues(good_views[(ci, REF)][0])[0] for ci in range(4)}
ref_t = {ci: good_views[(ci, REF)][1] for ci in range(4)}


GRID = obj_of(np.arange(1, 43))       # the whole 6x7 plate, in plate coordinates


def plate_dots_in_world(ci, fr):
    """Where view (ci, fr) says every dot of the plate is, in world coordinates.

    Compared PER DOT rather than per plate centre: a scrambled labelling can
    leave the centroid almost where it belongs while the pattern around it is
    wrong, so a centre-only test passes frames 39 and 42 that are visibly broken
    in triangulation/frame_*.png.
    """
    rv, tv, _ = good_views[(ci, fr)]
    R = cv2.Rodrigues(rv)[0]
    return (GRID @ R.T + tv - ref_t[ci]) @ ref_R[ci]


def plate_R_in_world(ci, fr):
    """Plate -> world rotation as view (ci, fr) sees it."""
    return ref_R[ci].T @ cv2.Rodrigues(good_views[(ci, fr)][0])[0]


def tilt_off_vertical_deg(R):
    """How far this plate pose departs from a pure rotation about world +Y.

    The plate is held vertical, so its own +Y axis should be world +Y and its
    normal should be horizontal; the only free rotation is the yaw about +Y.
    Measured over the well-labelled frames of this dataset: normal off
    horizontal <= 0.83 deg (median 0.23), plate-up off world +Y <= 1.23 deg
    (median 0.70), yaw spanning -24 to +30 deg.  So the two off-yaw degrees of
    freedom are real but small, which is exactly what makes them useful both as
    an outlier test and as a soft prior.
    """
    return np.degrees(np.arcsin(np.clip(
        max(abs(R[1, 0]), abs(R[1, 2])), 0.0, 1.0)))


# ------------------------------------------- stage 3: the plate is held vertical
verticality_rejects = []
for key in list(good_views):
    ci, fr = key
    if fr == REF:
        continue
    tilt = tilt_off_vertical_deg(plate_R_in_world(ci, fr))
    if tilt > TILT_GATE_DEG:
        verticality_rejects.append((fr, ci, tilt))
        del good_views[key]

kept, dropped = {}, []
for fr in frames_all:
    vs = [ci for ci in range(4) if (ci, fr) in good_views]
    if len(vs) < 2:
        continue
    C = {ci: plate_dots_in_world(ci, fr) for ci in vs}
    # largest subset whose implied plate centres all agree within AGREE_MM
    best = []
    for size in range(len(vs), 1, -1):
        for sub in __import__("itertools").combinations(vs, size):
            if all(np.linalg.norm(C[a] - C[b], axis=1).max() < AGREE_MM
                   for a in sub for b in sub if a < b):
                best = list(sub)
                break
        if best:
            break
    if not best:
        dropped.append((fr, vs, max(np.linalg.norm(C[a] - C[b], axis=1).max()
                                    for a in vs for b in vs if a < b)))
        continue
    if len(best) < len(vs):
        dropped.append((fr, [c for c in vs if c not in best],
                        max(np.linalg.norm(C[a] - C[b], axis=1).max()
                            for a in vs for b in vs if a < b)))
    kept[fr] = best
good_views = {(ci, fr): good_views[(ci, fr)] for fr, cs in kept.items() for ci in cs}
if len(kept.get(REF, [])) < 4:
    raise SystemExit(f"reference frame {REF} lost views in the agreement gate")

frames = sorted(kept)
free = [fr for fr in frames if fr != REF]
print(f"cc fixed at {CC} mm, zero distortion, gauge = plate pose of frame {REF}")
print(f"stage 1  per-camera PnP < {VIEW_GATE_PX} px:        {n_pnp}/{4*len(frames_all)} views")
print(f"stage 2  plate vertical within {TILT_GATE_DEG:.0f} deg + cross-camera agreement "
      f"< {AGREE_MM:.0f} mm:  {len(good_views)} views, {len(frames)} frames")
if verticality_rejects:
    print(f"         {len(verticality_rejects)} views rejected as non-vertical "
          f"(worst first): " + ", ".join(
              f"{fr[-2:]}/cam{ci+1} {t:.0f}deg"
              for fr, ci, t in sorted(verticality_rejects, key=lambda r: -r[2])[:10]))
if dropped:
    print("         rejected before the bundle (frame, views, worst per-dot disagreement):")
    for fr, vs, spread in sorted(dropped, key=lambda r: -r[2])[:16]:
        print(f"           {fr}  cams {[c+1 for c in vs]}  {spread:9.0f} mm")
print(f"unknowns: 4 camera poses + {len(free)} plate poses = {24 + 6*len(free)}")

# ---------------------------------------------------------------- observations
fidx = {fr: k for k, fr in enumerate(free)}
cam_i, frm_i, objp, pixp = [], [], [], []
for fr in frames:
    for ci in range(4):
        if (ci, fr) not in good_views:
            continue
        ids, px = views[(ci, fr)]
        o = obj_of(ids)
        cam_i.append(np.full(len(o), ci))
        frm_i.append(np.full(len(o), fidx.get(fr, -1)))
        objp.append(o)
        pixp.append(px.astype(float))
cam_i = np.concatenate(cam_i)
frm_i = np.concatenate(frm_i)
objp = np.concatenate(objp)
pixp = np.concatenate(pixp)
print(f"{len(objp)} observations, {2*len(objp)} residuals\n")

# ---------------------------------------------------------------- initial guess
x0 = np.zeros(24 + 6 * len(free))
for ci in range(4):
    rv, tv, _ = good_views[(ci, REF)]
    x0[3 * ci:3 * ci + 3] = rv
    x0[12 + 3 * ci:12 + 3 * ci + 3] = tv
for fr in free:
    # plate -> world from whichever camera saw it: R_if = R_i R_f, t_if = R_i t_f + t_i
    ci = next(c for c in range(4) if (c, fr) in good_views)
    R_i = cv2.Rodrigues(x0[3 * ci:3 * ci + 3])[0]
    t_i = x0[12 + 3 * ci:12 + 3 * ci + 3]
    rv, tv, _ = good_views[(ci, fr)]
    R_f = R_i.T @ cv2.Rodrigues(rv)[0]
    k = fidx[fr]
    x0[24 + 6 * k:24 + 6 * k + 3] = cv2.Rodrigues(R_f)[0].ravel()
    x0[24 + 6 * k + 3:24 + 6 * k + 6] = R_i.T @ (tv - t_i)


def project(p):
    Rc = np.array([cv2.Rodrigues(p[3 * i:3 * i + 3])[0] for i in range(4)])
    tc = p[12:24].reshape(4, 3)
    Rf = np.concatenate([np.eye(3)[None],
                         np.array([cv2.Rodrigues(p[24 + 6 * k:24 + 6 * k + 3])[0]
                                   for k in range(len(free))])]) if free else np.eye(3)[None]
    tf = np.concatenate([np.zeros((1, 3)), p[24:].reshape(-1, 6)[:, 3:]]) if free \
        else np.zeros((1, 3))
    fi = frm_i + 1                                    # -1 (reference) -> row 0 = identity
    Xw = np.einsum("nij,nj->ni", Rf[fi], objp) + tf[fi]
    Xc = np.einsum("nij,nj->ni", Rc[cam_i], Xw) + tc[cam_i]
    z = Xc[:, 2]
    return np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2],
                     K[1, 1] * Xc[:, 1] / z + K[1, 2]], 1)


def vertical_penalty(p):
    """Soft prior: each plate pose should be a pure yaw about world +Y.

    R_f[1,0] is the world-Y component of the plate's own +X axis and R_f[1,2] the
    world-Y component of its normal; both vanish for a pure yaw, and together
    they are exactly the two extra rotational degrees of freedom.  A soft
    penalty rather than a hard constraint because the plate is hand-held and the
    measured departure is ~0.2-1.2 deg, not zero -- forcing it to zero would
    bias the far corners of the plate by ~15 mm.
    """
    if not free or VERT_PX <= 0.0:
        return np.zeros(0)
    w = VERT_PX / np.sin(np.radians(VERT_SIGMA_DEG))
    e = np.empty(2 * len(free))
    for k in range(len(free)):
        R = cv2.Rodrigues(p[24 + 6 * k:24 + 6 * k + 3])[0]
        e[2 * k] = w * R[1, 0]
        e[2 * k + 1] = w * R[1, 2]
    return e


def resid(p):
    return np.concatenate([(project(p) - pixp).ravel(), vertical_penalty(p)])


def report(tag, p):
    e = np.linalg.norm((project(p) - pixp), axis=1)
    print(f"{tag:22s} reproj RMS {np.sqrt(np.mean(e**2)):6.3f} px   "
          f"median {np.median(e):6.3f}   p99 {np.percentile(e, 99):7.3f}   max {e.max():8.3f}")


report("before (anchored)", x0)

# The per-view PnP gate cannot catch a labelling that is wrong but self-consistent
# (a whole-grid shift fits its own points perfectly), so trim on the bundle's own
# residuals: fit, drop the dots that do not agree with everyone else, refit.  The
# reference frame is never trimmed -- it is the gauge.
keep = np.ones(len(objp), bool)
x = x0
for it in range(TRIM_ROUNDS):
    sel = keep
    def resid_sel(p, sel=sel):
        return np.concatenate([(project(p) - pixp)[sel].ravel(), vertical_penalty(p)])
    res = least_squares(resid_sel, x, method="trf", loss="soft_l1", f_scale=1.0,
                        xtol=1e-12, ftol=1e-12, verbose=0, max_nfev=300)
    x = res.x
    e = np.linalg.norm(project(x) - pixp, axis=1)
    thr = max(TRIM_FLOOR_PX, TRIM_MAD * np.median(e[keep]))
    new_keep = (e < thr) | (frm_i < 0)
    print(f"  round {it}: {keep.sum():5d} dots in, reproj RMS {np.sqrt(np.mean(e[keep]**2)):7.3f} px"
          f"  ->  trim at {thr:5.2f} px, {new_keep.sum():5d} remain")
    if new_keep.sum() == keep.sum():
        keep = new_keep
        break
    keep = new_keep

res_x = x
kept_frames = {free[k] for k in np.unique(frm_i[keep & (frm_i >= 0)])}
print(f"\nconverged on {keep.sum()}/{len(objp)} dots "
      f"across {len(kept_frames)+1} frames")
e = np.linalg.norm(project(res_x) - pixp, axis=1)[keep]
print(f"{'after (bundled)':22s} reproj RMS {np.sqrt(np.mean(e**2)):6.3f} px   "
      f"median {np.median(e):6.3f}   p99 {np.percentile(e, 99):7.3f}   max {e.max():8.3f}")


class _R:
    x = res_x


res = _R()

tilts = np.array([tilt_off_vertical_deg(cv2.Rodrigues(res_x[24 + 6*k:24 + 6*k + 3])[0])
                  for k in range(len(free))])
print(f"\nplate tilt off vertical after the bundle: median {np.median(tilts):.2f} deg, "
      f"max {tilts.max():.2f} deg  (prior: {VERT_PX} px per {VERT_SIGMA_DEG} deg, "
      f"0 disables)")

print("\ncamera positions C = -R^T t   [mm]")
print("cam        anchored (X,Y,Z)              bundled (X,Y,Z)              moved by")
for ci in range(4):
    def centre(p):
        R = cv2.Rodrigues(p[3 * ci:3 * ci + 3])[0]
        return -R.T @ p[12 + 3 * ci:12 + 3 * ci + 3]
    a, b = centre(x0), centre(res.x)
    print(f" {ci+1}   ({a[0]:8.1f},{a[1]:7.1f},{a[2]:8.1f})   "
          f"({b[0]:8.1f},{b[1]:7.1f},{b[2]:8.1f})   {np.linalg.norm(b-a):7.1f} mm")

print("\npairwise camera distances (frame-invariant)   anchored / bundled [mm]")
for a, b in ((0, 2), (1, 3), (0, 1), (2, 3)):
    def cen(p, ci):
        return -cv2.Rodrigues(p[3 * ci:3 * ci + 3])[0].T @ p[12 + 3 * ci:12 + 3 * ci + 3]
    print(f"  cam{a+1}-cam{b+1}: {np.linalg.norm(cen(x0,a)-cen(x0,b)):7.1f} / "
          f"{np.linalg.norm(cen(res.x,a)-cen(res.x,b)):7.1f}")

# how far the bundle moved the world: the reference plate pose is fixed, so this
# is a pure check that the datum did not drift
print(f"\nreference plate pose held at identity -- the world origin is still the "
      f"L-corner dot of frame {REF}")

if WRITE:
    for ci in range(4):
        for ext in ("ori", "addpar"):
            src = out / "cal" / f"cam{ci+1}.tif.{ext}"
            if src.exists():
                shutil.copy2(src, src.with_suffix(f".{ext}.prebundle"))
        cal, _ = calibration_from_opencv(K, D0, res.x[3*ci:3*ci+3], res.x[12+3*ci:12+3*ci+3],
                                         imx=IMX, imy=IMY, pix_x=PIX, pixel_origin="corner")
        cal.to_file(str(out / "cal" / f"cam{ci+1}.tif.ori"),
                    str(out / "cal" / f"cam{ci+1}.tif.addpar"))
    np.save(out / "cal" / "bundle_plate_poses.npy",
            {"frames": free, "poses": res.x[24:].reshape(-1, 6), "cc": CC}, allow_pickle=True)
    print("\nwrote .ori + zeroed .addpar (old files kept as *.prebundle)")
    print("now re-run check_plate_triangulation.py, check_epipolar.py and "
          "plot_frame_triangulation.py")
else:
    print("\ndry run -- pass --write to overwrite cal/camN.tif.ori")
