"""Illmenau driver for the joint plate bundle.

The solver and the vertical prior live in ``openptv2.plate_bundle``, the gates
and the initial poses in ``openptv2.plate_multiplane.prepare_bundle``; this file
only supplies the dataset's numbers, reports, and writes the ``.ori``.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402

from openptv2.plate_bundle import (  # noqa: E402
    bundle_plate_poses,
    project,
    rodrigues,
    tilt_off_vertical_deg,
)
from openptv2.plate_multiplane import (  # noqa: E402
    pinhole_calibration,
    pinhole_K,
    prepare_bundle,
)

out = CFG.DIR
PIX, IMX, IMY, REF = CFG.PIX, CFG.IMX, CFG.IMY, CFG.REF
NCAM, MIN_DOTS = CFG.NCAM, 12
VIEW_GATE_PX = float(os.environ.get("BUNDLE_VIEW_GATE_PX", 1.0))
AGREE_MM = float(os.environ.get("BUNDLE_AGREE_MM", 100.0))
TILT_GATE_DEG = float(os.environ.get("BUNDLE_TILT_GATE_DEG", 5.0))
VERT_SIGMA_DEG = float(os.environ.get("BUNDLE_VERT_SIGMA_DEG", 1.0))
VERT_PX = float(os.environ.get("BUNDLE_VERT_PX", 10.0))
TRIM_MAD = float(os.environ.get("BUNDLE_TRIM_MAD", 3.0))
TRIM_ROUNDS = int(os.environ.get("BUNDLE_TRIM_ROUNDS", 6))

CC = (
    float(sys.argv[1])
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else 8.5858
)
WRITE = "--write" in sys.argv
K = pinhole_K(CC, PIX, IMX, IMY)

views = CFG.load_views()
frames_all = sorted({f for _, f in views})
try:
    setup = prepare_bundle(
        views,
        CFG.GRID,
        K,
        NCAM,
        REF,
        min_dots=MIN_DOTS,
        view_gate_px=VIEW_GATE_PX,
        tilt_gate_deg=TILT_GATE_DEG,
        agree_mm=AGREE_MM,
    )
except ValueError as e:
    raise SystemExit(str(e)) from None
good, frames, free = setup.poses, setup.frames, setup.free
tilt_rejects, dropped, obs = setup.tilt_rejects, setup.dropped, setup.obs

print(f"cc fixed at {CC} mm, zero distortion, gauge = plate pose of frame {REF}")
print(
    f"gate 1  per-camera PnP < {VIEW_GATE_PX} px:            {setup.n_pnp}/{NCAM * len(frames_all)} views"
)
print(
    f"gate 2  plate vertical within {TILT_GATE_DEG:.0f} deg:         "
    f"{len(tilt_rejects)} views rejected"
    + (
        "  "
        + ", ".join(
            f"{fr[-2:]}/cam{CFG.cam_number(ci)} {t:.0f}deg"
            for fr, ci, t in sorted(tilt_rejects, key=lambda r: -r[2])[:8]
        )
        if tilt_rejects
        else ""
    )
)
print(
    f"gate 3  per-dot agreement < {AGREE_MM:.0f} mm:         {len(good)} views, "
    f"{len(frames)} frames"
)
for fr, vs, spread in sorted(dropped, key=lambda r: -r[2])[:10]:
    print(f"          {fr}  cams {[c + 1 for c in vs]}  worst per-dot {spread:7.0f} mm")
print(
    f"unknowns: {NCAM} camera poses + {len(free)} plate poses = {6 * (NCAM + len(free))}"
)

print(f"{len(obs.cam)} observations, {2 * len(obs.cam)} residuals\n")

cam_rvec0, cam_tvec0 = setup.cam_rvec0, setup.cam_tvec0
prv0, ptv0 = setup.plate_rvec0, setup.plate_tvec0

x0 = np.concatenate(
    [cam_rvec0.ravel(), cam_tvec0.ravel(), np.column_stack([prv0, ptv0]).ravel()]
)
e0 = np.linalg.norm(project(x0, obs, K, NCAM, len(free)) - obs.pix, axis=1)
print(
    f"{'before (anchored)':22s} reproj RMS {np.sqrt(np.mean(e0**2)):6.3f} px   "
    f"median {np.median(e0):6.3f}   max {e0.max():8.3f}"
)

res = bundle_plate_poses(
    obs,
    cam_rvec0,
    cam_tvec0,
    prv0,
    ptv0,
    K,
    vertical_px=VERT_PX,
    vertical_sigma_deg=VERT_SIGMA_DEG,
    trim_rounds=TRIM_ROUNDS,
    trim_mad=TRIM_MAD,
)
for i, (n, rms, thr) in enumerate(res.trim_history):
    print(
        f"  round {i}: {n:5d} dots in, reproj RMS {rms:7.3f} px  ->  trim at {thr:5.2f} px"
    )
e = res.residual_px[res.keep]
print(
    f"\nconverged on {int(res.keep.sum())}/{len(obs.cam)} dots across {len(frames)} frames"
)
print(
    f"{'after (bundled)':22s} reproj RMS {np.sqrt(np.mean(e**2)):6.3f} px   "
    f"median {np.median(e):6.3f}   p99 {np.percentile(e, 99):7.3f}   max {e.max():8.3f}"
)

tilts = np.array([tilt_off_vertical_deg(rodrigues(r)) for r in res.plate_rvec])
print(
    f"plate tilt off vertical: median {np.median(tilts):.2f} deg, max {tilts.max():.2f} deg"
    f"  (prior {VERT_PX} px per {VERT_SIGMA_DEG} deg; BUNDLE_VERT_PX=0 disables)"
)

print(
    "\ncam        anchored (X,Y,Z)              bundled (X,Y,Z)              moved by"
)
for ci in range(NCAM):
    a = -rodrigues(cam_rvec0[ci]).T @ cam_tvec0[ci]
    b = res.camera_centre(ci)
    print(
        f" {CFG.cam_number(ci)}   ({a[0]:8.1f},{a[1]:7.1f},{a[2]:8.1f})   "
        f"({b[0]:8.1f},{b[1]:7.1f},{b[2]:8.1f})   {np.linalg.norm(b - a):7.1f} mm"
    )
print("\npairwise camera distances (frame-invariant)   anchored / bundled [mm]")
for a, b in [(x, y) for x in range(NCAM) for y in range(x + 1, NCAM)]:

    def cen0(ci):
        return -rodrigues(cam_rvec0[ci]).T @ cam_tvec0[ci]

    print(
        f"  cam{CFG.cam_number(a)}-cam{CFG.cam_number(b)}: {np.linalg.norm(cen0(a) - cen0(b)):7.1f} / "
        f"{np.linalg.norm(res.camera_centre(a) - res.camera_centre(b)):7.1f}"
    )
print(
    f"\nreference plate pose held at identity -- the world origin is still the "
    f"L-corner dot of frame {REF}"
)

if WRITE:
    for ci in range(NCAM):
        for ext in ("ori", "addpar"):
            src = Path(CFG.cam_ori(ci)[0 if ext == "ori" else 1])
            if src.exists() and not src.with_suffix(f".{ext}.prebundle").exists():
                shutil.copy2(src, src.with_suffix(f".{ext}.prebundle"))
        cal = pinhole_calibration(
            K, res.cam_rvec[ci], res.cam_tvec[ci], imx=IMX, imy=IMY, pix_mm=PIX
        )
        cal.to_file(*CFG.cam_ori(ci))
    np.savez(
        out / "cal" / "bundle_plate_poses.npz",
        frames=np.array(free),
        plate_rvec=res.plate_rvec,
        plate_tvec=res.plate_tvec,
        cc=CC,
        # views that passed every gate; consumers that triangulate the raw
        # labels must skip the rest (partly visible plates with wrong ids)
        used_views=np.array([f"{ci}_{fr}" for ci, fr in sorted(good)], dtype=str),
    )
    print("\nwrote .ori + zeroed .addpar (first run kept the old ones as *.prebundle)")
    print(
        "now re-run check_plate_triangulation.py, check_epipolar.py and "
        "plot_frame_triangulation.py"
    )
else:
    print("\ndry run -- pass --write to overwrite cal/camN.tif.ori")
