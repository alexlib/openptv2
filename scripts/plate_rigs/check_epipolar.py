"""Epipolar check done properly: sample the ray densely, project each sample,
keep only samples that land inside the sensor, and measure the closest approach
to the dot camera B actually detected.

Do NOT approximate the curve by the chord between two far endpoints -- that
produced a spurious 289 px reading during this work.  Also assert the projected
ray is monotone (straight) inside the sensor: a straight 3D ray that doubles
back means the distortion model is unphysical, which no miss distance shows.

The dots come from the detection CACHE, like every other step.  This script used
to re-detect and re-label the reference frame itself through `label_plate()`
with no `corner_index`, i.e. the unsafe anchoring path that pins the grid to the
smallest index a view happens to see.  When that disagreed with the cache the
check compared dot 7 of one camera against dot 8 of another and reported
100-400 px misses for a calibration that was fine -- a false alarm that looks
exactly like the real failure this script exists to catch.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np  # noqa: E402

from openptv2.plate_multiplane import epipolar_curve_misses  # noqa: E402

cpar = CFG.control_par()
cals = CFG.load_calibrations()
views = CFG.load_views()
det = []
for ci in range(CFG.NCAM):
    if (ci, CFG.REF) not in views:
        raise SystemExit(
            f"cam{CFG.cam_number(ci)} has no labelled reference frame "
            f"{CFG.REF} in the cache -- re-run detect_plate_frames.py"
        )
    ids, ip = views[(ci, CFG.REF)]
    det.append(dict(zip(ids.tolist(), ip.tolist())))

Zs = np.linspace(-1500, 1500, 601)
print(
    "A->B   n   closest approach of the epipolar CURVE to the dot [px]     "
    "curve monotone inside sensor?"
)
print("           median      p90       max")
for a in range(CFG.NCAM):
    for b in range(CFG.NCAM):
        if a == b:
            continue
        ds, mono = epipolar_curve_misses(det[a], det[b], cals[a], cals[b], cpar, Zs)
        if ds:
            print(
                f"{CFG.cam_number(a)}->{CFG.cam_number(b)}  {len(ds):3d}  {np.median(ds):8.2f} {np.percentile(ds, 90):8.2f} "
                f"{np.max(ds):8.2f}          {mono}/{len(ds)}"
            )
