"""Triangulate the plate dots and check three things:
   1. do they lie on a plane,  2. is the pitch 120 mm,  3. are the ABSOLUTE
   positions right (compare to the known block coords -- no alignment applied)."""
import os
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import COORD_UNUSED
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
from openptv2.orientation import multi_cam_point_positions

# Dataset location; override with ILLMENAU_RAW / ILLMENAU_DIR.
ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))

out = Path(ILLMENAU_DIR)
PITCH, NX, NY, REF = 120.0, 6, 7, "00000000"
cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""]*4, cal_img_base_name=[""]*4)
cals = []
for ci in range(4):
    c = Calibration()
    c.from_file(str(out/f"cal/cam{ci+1}.tif.ori"), str(out/f"cal/cam{ci+1}.tif.addpar"))
    cals.append(c)
d = np.load(out / "cal" / "labelled_all_frames.npz")
views = {}
for k in d.files:
    if k.endswith("_ids"):
        c, fr, _ = k.split("_")
        views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])

def nominal(pid):
    ix, iy = (pid - 1) % NX, (pid - 1) // NX
    return np.array([(ix - 2) * PITCH, (iy - 3) * PITCH, 0.0])

def triangulate(fr):
    per = {ci: dict(zip(views[(ci, fr)][0].tolist(), views[(ci, fr)][1].tolist()))
           for ci in range(4) if (ci, fr) in views}
    ids = [i for i in sorted({i for m in per.values() for i in m})
           if sum(i in m for m in per.values()) >= 2]
    if len(ids) < 6:
        return None, None, None
    t = np.full((len(ids), 4, 2), COORD_UNUSED)
    ncam = []
    for k, pid in enumerate(ids):
        n = 0
        for ci, m in per.items():
            if pid in m:
                mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
                a = cals[ci].added_par
                t[k, ci] = dist_to_flat(mx, my, cals[ci].int_par.xh, cals[ci].int_par.yh,
                                        a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she)
                n += 1
        ncam.append(n)
    pos, rcm = multi_cam_point_positions(t, cpar, cals)
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1)
    return pos[ok], [p for p, k in zip(ids, ok) if k], np.array(ncam)[ok]

pos, ids, ncam = triangulate(REF)
nom = np.array([nominal(p) for p in ids])
c = pos.mean(0)
n = np.linalg.svd(pos - c)[2][2]
resid = (pos - c) @ n
print(f"frame {REF}: {len(ids)} dots triangulated ({int((ncam==4).sum())} from 4 cameras)\n")
print("1) PLANE   normal ({:.4f},{:.4f},{:.4f})  offset {:.3f} mm from origin".format(*n, abs(np.dot(c, n))))
print(f"           planarity residual  RMS {np.sqrt(np.mean(resid**2)):.3f} mm   max {np.abs(resid).max():.3f} mm")
idx = {p: k for k, p in enumerate(ids)}
dx = [np.linalg.norm(pos[idx[p]]-pos[idx[p+1]]) for p in ids if p+1 in idx and ((p-1) % NX) < NX-1]
dy = [np.linalg.norm(pos[idx[p]]-pos[idx[p+NX]]) for p in ids if p+NX in idx]
print(f"\n2) PITCH   along X: median {np.median(dx):7.3f} mm  std {np.std(dx):5.3f}  "
      f"({100*(np.median(dx)/PITCH-1):+.3f} % of {PITCH})")
print(f"           along Y: median {np.median(dy):7.3f} mm  std {np.std(dy):5.3f}  "
      f"({100*(np.median(dy)/PITCH-1):+.3f} %)")
err = pos - nom
print("\n3) ABSOLUTE position vs the known block coords (no alignment applied)")
print(f"           |error|  median {np.median(np.linalg.norm(err,axis=1)):.3f}  "
      f"max {np.max(np.linalg.norm(err,axis=1)):.3f} mm")
print(f"           bias  X {err[:,0].mean():+7.3f}  Y {err[:,1].mean():+7.3f}  Z {err[:,2].mean():+7.3f} mm")
print(f"           std   X {err[:,0].std():7.3f}  Y {err[:,1].std():7.3f}  Z {err[:,2].std():7.3f} mm")
print("\n   id   nominal (X,Y,Z)        triangulated (X,Y,Z)         error (mm)   ncam")
for k in list(range(0, len(ids), max(1, len(ids)//12))):
    print(f"  {ids[k]:3d}  ({nom[k,0]:7.1f},{nom[k,1]:7.1f},{nom[k,2]:6.1f})  "
          f"({pos[k,0]:8.2f},{pos[k,1]:8.2f},{pos[k,2]:7.2f})  "
          f"({err[k,0]:+6.2f},{err[k,1]:+6.2f},{err[k,2]:+6.2f})   {ncam[k]}")
