"""Epipolar check done properly: sample the ray densely, project each sample,
keep only samples that land inside the sensor, and measure the closest approach
to the dot camera B actually detected."""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.ray_tracing import ray_tracing
from openptv2.algorithms.trafo import metric_to_pixel, pixel_to_metric
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_plate

# Dataset location; override with ILLMENAU_RAW / ILLMENAU_DIR.
ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))

base = Path(ILLMENAU_RAW)
out = Path(ILLMENAU_DIR)
PITCH, NX, NY = 120.0, 6, 7
cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""]*4, cal_img_base_name=[""]*4)
tpar = plate_tpar_from_yaml(out / "parameters_Run1.yaml")
cals, det = [], []
for ci in range(4):
    c = Calibration()
    c.from_file(str(out / f"cal/cam{ci+1}.tif.ori"), str(out / f"cal/cam{ci+1}.tif.addpar"))
    cals.append(c)
    f = sorted((base / f"Kalibrierung_{ci+1}").glob("00000000*.tif*"))[0]
    res = detect_plate_targets(np.array(Image.open(f)), tpar, cpar, cam=ci, coded_thr=30.0)
    ip, rp, _ = label_plate(res.centroids, res.coded_mask, pitch_x=PITCH, pitch_y=PITCH, nx=NX, ny=NY, y_sign=1)
    ids = np.round(rp[:,1]/PITCH).astype(int)*NX + np.round(rp[:,0]/PITCH).astype(int) + 1
    det.append(dict(zip(ids.tolist(), ip.tolist())))

Zs = np.linspace(-1500, 1500, 601)
print("A->B   n   closest approach of the epipolar CURVE to the dot [px]     "
      "curve monotone inside sensor?")
print("           median      p90       max")
for a in range(4):
    for b in range(4):
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
