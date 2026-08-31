"""Epipolar check done properly: sample the ray densely, project each sample,
keep only samples that land inside the sensor, and measure the closest approach
to the dot camera B actually detected."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import numpy as np
from PIL import Image

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.ray_tracing import ray_tracing
from openptv2.algorithms.trafo import metric_to_pixel, pixel_to_metric
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_plate

base = CFG.RAW
out = CFG.DIR
PITCH, NX, NY = 120.0, 6, 7
cpar = CFG.control_par()
tpar = plate_tpar_from_yaml(out / "parameters_Run1.yaml")
cals, det = [], []
for ci in range(CFG.NCAM):
    c = Calibration()
    c.from_file(*CFG.cam_ori(ci))
    cals.append(c)
    f = sorted(CFG.image_dir(ci).glob(f"{CFG.REF}*.tif*"))[0]
    res = detect_plate_targets(np.array(Image.open(f)), tpar, cpar, cam=ci, coded_thr=30.0)
    ip, rp, _ = label_plate(res.centroids, res.coded_mask, pitch_x=PITCH, pitch_y=PITCH, nx=NX, ny=NY, y_sign=1)
    ids = np.round(rp[:,1]/PITCH).astype(int)*NX + np.round(rp[:,0]/PITCH).astype(int) + 1
    det.append(dict(zip(ids.tolist(), ip.tolist())))

Zs = np.linspace(-1500, 1500, 601)
print("A->B   n   closest approach of the epipolar CURVE to the dot [px]     "
      "curve monotone inside sensor?")
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
            pos, v = ray_tracing(mx, my, ca.ext_par.dm, ca.ext_par.x0, ca.ext_par.y0,
                                 ca.ext_par.z0, ca.int_par.cc, ca.glass_par.vec_x,
                                 ca.glass_par.vec_y, ca.glass_par.vec_z, 1., 1., 1., 0.)
            pos, v = np.asarray(pos), np.asarray(v)
            P = pos + ((Zs - pos[2]) / v[2])[:, None] * v
            q = np.array([metric_to_pixel(*img_coord(p, cb, cpar.mm), cpar) for p in P])
            inside = (q[:,0] > -200) & (q[:,0] < 2760) & (q[:,1] > -200) & (q[:,1] < 2248)
            if inside.sum() < 3:
                continue
            qi = q[inside]
            ds.append(float(np.min(np.linalg.norm(qi - np.array(det[b][pid]), axis=1))))
            step = np.diff(qi, axis=0)
            mono += int(np.all(step @ step[0] > 0))       # curve does not double back
        if ds:
            print(f"{a+1}->{b+1}  {len(ds):3d}  {np.median(ds):8.2f} {np.percentile(ds,90):8.2f} "
                  f"{np.max(ds):8.2f}          {mono}/{len(ds)}")
