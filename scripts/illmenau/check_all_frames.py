"""Triangulate EVERY calibration frame with the final .ori and report, per frame,
whether the plate comes back as a plane at the right pitch.

The reference frame is the one the world is anchored to, so it is expected to be
near-perfect; the value of this check is the other frames, which the calibration
never saw as a rigid target.
"""
import os
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.orientation import COORD_UNUSED
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.trafo import dist_to_flat, pixel_to_metric
from openptv2.orientation import multi_cam_point_positions

ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))
out = Path(ILLMENAU_DIR)
PITCH, NX, REF = 120.0, 6, "00000000"

cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""] * 4, cal_img_base_name=[""] * 4)
cals = []
for ci in range(4):
    c = Calibration()
    c.from_file(str(out / f"cal/cam{ci+1}.tif.ori"), str(out / f"cal/cam{ci+1}.tif.addpar"))
    cals.append(c)

d = np.load(out / "cal" / "labelled_all_frames.npz")
views = {}
for k in d.files:
    if k.endswith("_ids"):
        c, fr, _ = k.split("_")
        views[(int(c[1:]), fr)] = (d[k], d[f"{c}_{fr}_px"])


def triangulate(fr):
    per = {ci: dict(zip(views[(ci, fr)][0].tolist(), views[(ci, fr)][1].tolist()))
           for ci in range(4) if (ci, fr) in views}
    ids = [i for i in sorted({i for m in per.values() for i in m})
           if sum(i in m for m in per.values()) >= 2]
    if len(ids) < 12:
        return None, None, 0
    t = np.full((len(ids), 4, 2), COORD_UNUSED)
    for k, pid in enumerate(ids):
        for ci, m in per.items():
            if pid in m:
                mx, my = pixel_to_metric(m[pid][0], m[pid][1], cpar)
                a = cals[ci].added_par
                t[k, ci] = dist_to_flat(mx, my, cals[ci].int_par.xh, cals[ci].int_par.yh,
                                        a.k1, a.k2, a.k3, a.p1, a.p2, a.scx, a.she)
    pos, _ = multi_cam_point_positions(t, cpar, cals)
    ok = np.isfinite(pos).all(1) & (np.abs(pos) < 1e5).all(1)
    return pos[ok], [p for p, k in zip(ids, ok) if k], len(per)


print("frame       n  cams   plane RMS / max [mm]   pitch X    pitch Y   plate distance"
      "\n                                                 [mm]       [mm]     from ref [mm]")
rows = []
for fr in sorted({f for _, f in views}):
    pos, ids, ncam = triangulate(fr)
    if pos is None or ncam < 2:
        continue
    c = pos.mean(0)
    try:
        n = np.linalg.svd(pos - c)[2][2]
    except np.linalg.LinAlgError:
        continue
    resid = (pos - c) @ n
    idx = {p: k for k, p in enumerate(ids)}
    dx = [np.linalg.norm(pos[idx[p]] - pos[idx[p + 1]]) for p in ids
          if p + 1 in idx and ((p - 1) % NX) < NX - 1]
    dy = [np.linalg.norm(pos[idx[p]] - pos[idx[p + NX]]) for p in ids if p + NX in idx]
    if not dx or not dy:
        continue
    rms = float(np.sqrt(np.mean(resid ** 2)))
    row = (fr, len(ids), ncam, rms, float(np.abs(resid).max()),
           float(np.median(dx)), float(np.median(dy)), float(np.linalg.norm(c)))
    rows.append(row)
    print(f"{fr}  {len(ids):3d}   {ncam}    {rms:7.3f} / {row[4]:7.3f}     "
          f"{row[5]:8.3f}   {row[6]:8.3f}   {row[7]:9.1f}")

r = np.array([(x[3], x[5], x[6]) for x in rows])
print(f"\n{len(rows)} frames.  planarity RMS: median {np.median(r[:,0]):.3f} mm, "
      f"p90 {np.percentile(r[:,0],90):.3f} mm")
print(f"pitch X median {np.median(r[:,1]):.3f} mm ({100*(np.median(r[:,1])/PITCH-1):+.2f} %), "
      f"pitch Y median {np.median(r[:,2]):.3f} mm ({100*(np.median(r[:,2])/PITCH-1):+.2f} %)")
good = r[r[:, 0] < 2.0]
print(f"frames with planarity RMS < 2 mm: {len(good)}/{len(rows)}")
