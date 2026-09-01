"""Refit every camera pose as a PURE PINHOLE on the reference frame.

Zero distortion, principal point at the sensor centre, one shared `cc` (from
fit_plate_cc.py).  `.addpar` is written as zeros -- see
docs/illmenau-4cam-calibration.md section 4, Trap 1 for why distortion fitted
from a single plane is worse than none.

The labelled dots come from the CACHE that detect_plate_frames.py wrote, not
from a fresh detection here.  That matters more than it looks: this script used
to re-detect and re-label the reference frame on its own, through
`label_plate()` with no `corner_index` and no up-hint -- the unsafe anchoring
path that pins the grid to the smallest index the view happened to see.  On the
near-wall cameras the reference view shows the whole lattice, so it agreed with
the cache by luck.  On the far wall it does not, and the poses came out with
3-4.5 px reprojection and camera positions metres from the wall they are bolted
to.  One labelling, used everywhere, is the fix.

    refit_plate_pinhole.py <cc_mm>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from openptv2.algorithms.imgcoord import img_coord  # noqa: E402
from openptv2.algorithms.trafo import metric_to_pixel  # noqa: E402
from openptv2.calibration_import import calibration_from_opencv  # noqa: E402

PIX, IMX, IMY = CFG.PIX, CFG.IMX, CFG.IMY
CC = float(sys.argv[1]) if len(sys.argv) > 1 else 9.44
cpar = CFG.control_par()
K = np.array([[CC / PIX, 0, IMX / 2], [0, CC / PIX, IMY / 2], [0, 0, 1.0]])

views = CFG.load_views()
print(f"pure pinhole refit on frame {CFG.REF}, cc = {CC} mm shared, zero distortion")
print(f"labels from cal/{CFG.NPZ} (the same ones every other step uses)\n")

for ci in range(CFG.NCAM):
    cam = CFG.cam_number(ci)
    if (ci, CFG.REF) not in views:
        raise SystemExit(
            f"cam{cam} has no labelled reference frame {CFG.REF} in the "
            f"cache -- re-run detect_plate_frames.py"
        )
    ids, ip = views[(ci, CFG.REF)]
    ip = ip.astype(np.float64)
    obj = CFG.obj_of(ids)
    ok, rvec, tvec = cv2.solvePnP(obj, ip, K, np.zeros(5))
    if not ok:
        raise SystemExit(f"cam{cam}: solvePnP failed on {len(obj)} points")
    rvec, tvec = cv2.solvePnPRefineLM(obj, ip, K, np.zeros(5), rvec, tvec)
    cal, _ = calibration_from_opencv(
        K, np.zeros(5), rvec, tvec, imx=IMX, imy=IMY, pix_x=PIX, pixel_origin="corner"
    )
    rep = np.array([metric_to_pixel(*img_coord(p, cal, cpar.mm), cpar) for p in obj])
    err = np.linalg.norm(rep - ip, axis=1)
    e = cal.ext_par
    print(
        f"cam{cam}: C=({e.x0:8.1f},{e.y0:7.1f},{e.z0:8.1f})  cc={cal.int_par.cc:.3f}  "
        f"{len(obj):2d} dots  reproj RMS {np.sqrt(np.mean(err**2)):6.3f} px "
        f"(max {err.max():.2f})"
    )
    cal.to_file(*CFG.cam_ori(ci))
print("\nwrote .ori + zeroed .addpar")
print(
    "A reprojection RMS much above ~1 px here means the reference frame is "
    "mislabelled for that camera, not that the pose is hard to fit."
)
