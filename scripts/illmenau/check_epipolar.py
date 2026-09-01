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
import numpy as np

from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.ray_tracing import ray_tracing
from openptv2.algorithms.trafo import metric_to_pixel, pixel_to_metric

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
        ds, mono = [], 0
        ca, cb = cals[a], cals[b]
        for pid, pa in det[a].items():
            if pid not in det[b]:
                continue
            mx, my = pixel_to_metric(pa[0], pa[1], cpar)
            pos, v = ray_tracing(
                mx,
                my,
                ca.ext_par.dm,
                ca.ext_par.x0,
                ca.ext_par.y0,
                ca.ext_par.z0,
                ca.int_par.cc,
                ca.glass_par.vec_x,
                ca.glass_par.vec_y,
                ca.glass_par.vec_z,
                1.0,
                1.0,
                1.0,
                0.0,
            )
            pos, v = np.asarray(pos), np.asarray(v)
            P = pos + ((Zs - pos[2]) / v[2])[:, None] * v
            q = np.array([metric_to_pixel(*img_coord(p, cb, cpar.mm), cpar) for p in P])
            inside = (
                (q[:, 0] > -200)
                & (q[:, 0] < 2760)
                & (q[:, 1] > -200)
                & (q[:, 1] < 2248)
            )
            if inside.sum() < 3:
                continue
            qi = q[inside]
            ds.append(float(np.min(np.linalg.norm(qi - np.array(det[b][pid]), axis=1))))
            step = np.diff(qi, axis=0)
            mono += int(np.all(step @ step[0] > 0))  # curve does not double back
        if ds:
            print(
                f"{CFG.cam_number(a)}->{CFG.cam_number(b)}  {len(ds):3d}  {np.median(ds):8.2f} {np.percentile(ds, 90):8.2f} "
                f"{np.max(ds):8.2f}          {mono}/{len(ds)}"
            )
