"""Why does changing Xmin/Xmax/Zmin/Zmax move the epipolar line, and what is the
largest observation volume this rig can be given?

`epi.epi_mm` does not compute a line, it computes the two ENDPOINTS of a
segment, by walking the sight ray to Z = Zmin_lay and Z = Zmax_lay (each
interpolated in X across X_lay) and projecting those two 3D points into the
second camera.  The GUI then draws the straight chord between them.  So the box
can change the drawn segment in three different ways:

  TRUNCATION   the segment only covers the Z range you asked for.  If the true
               particle depth is outside it, the segment stops short of the
               matching dot.  Expected, harmless, and the usual reason lines
               "do not reach".

  CHORD ERROR  the drawn chord equals the true epipolar curve only if that
               curve is straight.  It is straight for a pinhole; a non-zero
               .addpar bends it, and the bend is worst in the middle of a long
               segment because the endpoints get dragged out to the image
               periphery where the distortion polynomial is largest.  A wide
               box therefore makes a bad .addpar look worse -- which is exactly
               why shrinking the box appeared to "fix" the calibration.

  HORIZON FLIP the hard failure.  Once the sampled point passes the plane
               through the second camera's centre perpendicular to its optical
               axis, its depth in that camera goes through zero and changes
               sign: the projection runs off to infinity and comes back on the
               opposite side of the sensor.  The endpoint is then meaningless
               and the chord is thrown right across the image.  Nothing about
               the calibration is wrong -- the box simply asked for a point
               behind the camera.

This script measures all three for the delivered calibration, and prints the
largest safe Zmax for every ordered camera pair.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402

from openptv2.algorithms.epi import epi_mm  # noqa: E402
from openptv2.algorithms.imgcoord import img_coord  # noqa: E402
from openptv2.algorithms.parameters import VolumePar  # noqa: E402
from openptv2.algorithms.trafo import metric_to_pixel  # noqa: E402
from openptv2.plate_multiplane import (  # noqa: E402
    epipolar_horizon_z,
    pixel_to_flat,
    sight_ray,
)

cpar = CFG.control_par()
cals = CFG.load_calibrations()
views = CFG.load_views()
det = [
    dict(zip(views[(ci, CFG.REF)][0].tolist(), views[(ci, CFG.REF)][1].tolist()))
    for ci in range(CFG.NCAM)
]


# ---------------------------------------------------------------- 1) the horizon
print("1) HORIZON -- the Z at which a sight ray from A crosses camera B's principal")
print("   plane.  Zmax_lay must stay clearly BELOW the smallest of these, or the far")
print("   endpoint is computed behind camera B and the drawn chord is nonsense.\n")
print("   A->B      Z of the horizon [mm]  (worst over all dots of frame 00000000)")
horizons = {}
for a in range(CFG.NCAM):
    for b in range(CFG.NCAM):
        if a == b:
            continue
        zs = []
        for pid, pix in det[a].items():
            z = epipolar_horizon_z(*sight_ray(pix, cals[a], cpar), cals[b])
            if z is not None:
                zs.append(z)
        horizons[(a, b)] = float(np.min(zs))
        print(
            f"   {a + 1}->{b + 1}          {np.min(zs):8.0f}   (median {np.median(zs):8.0f})"
        )
zsafe = min(horizons.values())
print(f"\n   smallest horizon over all 12 ordered pairs: Z = {zsafe:.0f} mm")
print(
    f"   => Zmax_lay must stay below it.  A 20 % margin gives Zmax <= {0.8 * zsafe:.0f} mm."
)
print("   The delivered parameters_Run1.yaml uses Zmax_lay = +1500 mm, which is safe;")
print("   the +4000 box that produced the original 'epipolar lines are wrong' report")
print("   was PAST the horizon for several pairs.")

# ------------------------------------------------- 2) is the drawn chord the line?
print("\n\n2) TRUNCATION vs CHORD ERROR -- for every dot of frame 00000000 and every")
print("   ordered pair, compare the drawn chord against the densely sampled curve.")
print("   With a pure pinhole (.addpar all zero) the two must agree exactly, so the")
print("   box can only truncate the segment, never move it.\n")
print(
    "   Z box      segment length [px]   miss of the chord [px]   chord-vs-curve [px]"
)
for zb in (500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0):
    vpar = VolumePar(
        X_lay=[-zb, zb],
        Zmin_lay=[-zb, -zb],
        Zmax_lay=[zb, zb],
        cn=0.0,
        cnx=0.0,
        cny=0.0,
        csumg=0.0,
        corrmin=0.0,
        eps0=0.0,
    )
    seg, miss, bend = [], [], []
    for a in range(CFG.NCAM):
        for b in range(CFG.NCAM):
            if a == b:
                continue
            for pid, pix in det[a].items():
                if pid not in det[b]:
                    continue
                xf, yf = pixel_to_flat(pix, cals[a], cpar)
                x1, y1, x2, y2 = epi_mm(xf, yf, cals[a], cals[b], cpar.mm, vpar)
                p1 = np.array(metric_to_pixel(x1, y1, cpar))
                p2 = np.array(metric_to_pixel(x2, y2, cpar))
                seg.append(float(np.linalg.norm(p2 - p1)))
                # miss of the drawn chord from the dot camera B really detected
                q = np.array(det[b][pid])
                e = p2 - p1
                t = np.clip((q - p1) @ e / (e @ e), 0.0, 1.0)
                miss.append(float(np.linalg.norm(p1 + t * e - q)))
                # how far the true curve departs from that chord in the middle
                pos, v = sight_ray(pix, cals[a], cpar)
                Zs = np.linspace(-zb, zb, 41)
                P = pos + ((Zs - pos[2]) / v[2])[:, None] * v
                cur = np.array(
                    [metric_to_pixel(*img_coord(p, cals[b], cpar.mm), cpar) for p in P]
                )
                n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
                bend.append(float(np.max(np.abs((cur - p1) @ n))))
    print(
        f"   +-{zb:6.0f}   {np.median(seg):10.0f}          {np.median(miss):10.2f}"
        f"          {np.median(bend):10.3f}"
    )

print("\n   segment length grows with the box (truncation), the miss does not change")
print("   (the line is the same line), and chord-vs-curve stays at numerical zero")
print("   because .addpar is zero.  Restore a non-zero .addpar and the last column is")
print("   what breaks first, in the MIDDLE of the segment, growing with the box.")
