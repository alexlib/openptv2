"""Triangulate the plate dots and check three things:
1. do they lie on a plane,  2. is the pitch 120 mm,  3. are the ABSOLUTE
positions right (compare to the known block coords -- no alignment applied)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402

from openptv2.plate_multiplane import (  # noqa: E402
    plate_frame_metrics,
    triangulate_plate,
)

PITCH, REF = CFG.PITCH, CFG.REF
cpar = CFG.control_par()
cals = CFG.load_calibrations()
views = CFG.load_views()

tri = triangulate_plate(
    {ci: views[(ci, REF)] for ci in range(CFG.NCAM) if (ci, REF) in views},
    cals,
    cpar,
)
pos, ids, ncam = tri.pos, tri.ids, tri.ncam
m = plate_frame_metrics(pos, ids, CFG.GRID)
nom = CFG.obj_of(ids)
c, n, resid = m.centre, m.normal, m.residual
dx, dy = m.pitch_x, m.pitch_y
err = m.nominal_error
print(
    f"frame {REF}: {len(ids)} dots triangulated ({int((ncam == 4).sum())} from 4 cameras)\n"
)
print(
    "1) PLANE   normal ({:.4f},{:.4f},{:.4f})  offset {:.3f} mm from origin".format(
        *n, abs(np.dot(c, n))
    )
)
print(
    f"           planarity residual  RMS {np.sqrt(np.mean(resid**2)):.3f} mm   max {np.abs(resid).max():.3f} mm"
)
print(
    f"\n2) PITCH   along X: median {np.median(dx):7.3f} mm  std {np.std(dx):5.3f}  "
    f"({100 * (np.median(dx) / PITCH - 1):+.3f} % of {PITCH})"
)
print(
    f"           along Y: median {np.median(dy):7.3f} mm  std {np.std(dy):5.3f}  "
    f"({100 * (np.median(dy) / PITCH - 1):+.3f} %)"
)
print("\n3) ABSOLUTE position vs the known block coords (no alignment applied)")
print(
    f"           |error|  median {np.median(np.linalg.norm(err, axis=1)):.3f}  "
    f"max {np.max(np.linalg.norm(err, axis=1)):.3f} mm"
)
print(
    f"           bias  X {err[:, 0].mean():+7.3f}  Y {err[:, 1].mean():+7.3f}  Z {err[:, 2].mean():+7.3f} mm"
)
print(
    f"           std   X {err[:, 0].std():7.3f}  Y {err[:, 1].std():7.3f}  Z {err[:, 2].std():7.3f} mm"
)
print("\n   id   nominal (X,Y,Z)        triangulated (X,Y,Z)         error (mm)   ncam")
for k in list(range(0, len(ids), max(1, len(ids) // 12))):
    print(
        f"  {ids[k]:3d}  ({nom[k, 0]:7.1f},{nom[k, 1]:7.1f},{nom[k, 2]:6.1f})  "
        f"({pos[k, 0]:8.2f},{pos[k, 1]:8.2f},{pos[k, 2]:7.2f})  "
        f"({err[k, 0]:+6.2f},{err[k, 1]:+6.2f},{err[k, 2]:+6.2f})   {ncam[k]}"
    )
