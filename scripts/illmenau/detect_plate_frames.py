"""Detect + L-code label every calibration frame of cams 1-4 once, cache to npz.

Cache layout: for each cam, a dict frame -> (ids[n], pixels[n,2]) with
id = iy*6 + ix + 1 on the datum-shifted grid (id 21 = origin).
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_plate

# Dataset location; override with ILLMENAU_RAW / ILLMENAU_DIR.
ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))

base = Path(ILLMENAU_RAW)
out = Path(ILLMENAU_DIR)
PITCH, NX, NY = 120.0, 6, 7
tpar = plate_tpar_from_yaml(out / "parameters_Run1.yaml")
cpar = ControlPar(num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""]*4, cal_img_base_name=[""]*4)

store = {}
for ci in range(4):
    for f in sorted((base / f"Kalibrierung_{ci+1}").glob("*.tif*")):
        fr = f.name.split("_")[0]
        try:
            res = detect_plate_targets(np.array(Image.open(f)), tpar, cpar, cam=ci, coded_thr=30.0)
            ip, rp, _ = label_plate(res.centroids, res.coded_mask, pitch_x=PITCH, pitch_y=PITCH,
                                    nx=NX, ny=NY, y_sign=1)
        except Exception as e:
            print(f"cam{ci+1} {fr}: {type(e).__name__}", flush=True)
            continue
        ix = np.round(rp[:, 0] / PITCH).astype(int)
        iy = np.round(rp[:, 1] / PITCH).astype(int)
        store[f"c{ci}_{fr}_ids"] = iy * NX + ix + 1
        store[f"c{ci}_{fr}_px"] = ip
        print(f"cam{ci+1} {fr}: {len(ip)} dots", flush=True)
np.savez_compressed(out / "cal" / "labelled_all_frames.npz", **store)
print("wrote", out / "cal" / "labelled_all_frames.npz", len(store)//2, "views")
