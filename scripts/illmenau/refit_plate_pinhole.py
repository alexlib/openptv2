"""Refit all four poses as a PURE PINHOLE on frame 00000000: zero distortion,
principal point at the sensor centre, cc shared.  .addpar is written as zeros."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.trafo import metric_to_pixel
from openptv2.calibration_import import calibration_from_opencv
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_plate

# Dataset location; override with ILLMENAU_RAW / ILLMENAU_DIR.
ILLMENAU_RAW = os.environ.get("ILLMENAU_RAW", r"C:\Users\alex\Downloads\Illmenau")
ILLMENAU_DIR = os.environ.get("ILLMENAU_DIR",
                              os.path.join(ILLMENAU_RAW, "openptv_illmenau_4cam"))

base = Path(ILLMENAU_RAW)
out = Path(ILLMENAU_DIR)
PITCH, NX, NY, PIX, IMX, IMY = 120.0, 6, 7, 0.005, 2560, 2048
CC = float(sys.argv[1]) if len(sys.argv) > 1 else 9.44
cpar = ControlPar(num_cams=4, imx=IMX, imy=IMY, pix_x=PIX, pix_y=PIX,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""]*4, cal_img_base_name=[""]*4)
tpar = plate_tpar_from_yaml(out / "parameters_Run1.yaml")
K = np.array([[CC/PIX, 0, IMX/2], [0, CC/PIX, IMY/2], [0, 0, 1.0]])
print(f"pure pinhole refit, cc = {CC} mm shared, zero distortion\n")
for ci in range(4):
    f = sorted((base / f"Kalibrierung_{ci+1}").glob("00000000*.tif*"))[0]
    res = detect_plate_targets(np.array(Image.open(f)), tpar, cpar, cam=ci, coded_thr=30.0)
    ip, rp, _ = label_plate(res.centroids, res.coded_mask, pitch_x=PITCH, pitch_y=PITCH,
                            nx=NX, ny=NY, y_sign=1)
    obj = rp.copy()
    obj[:, 0] -= 2*PITCH
    obj[:, 1] -= 3*PITCH
    ok, rvec, tvec = cv2.solvePnP(obj, ip.astype(np.float64), K, np.zeros(5))
    rvec, tvec = cv2.solvePnPRefineLM(obj, ip.astype(np.float64), K, np.zeros(5), rvec, tvec)
    cal, _ = calibration_from_opencv(K, np.zeros(5), rvec, tvec,
                                     imx=IMX, imy=IMY, pix_x=PIX, pixel_origin="corner")
    rep = np.array([metric_to_pixel(*img_coord(p, cal, cpar.mm), cpar) for p in obj])
    rms = float(np.sqrt(np.mean(np.sum((rep - ip)**2, 1))))
    e = cal.ext_par
    print(f"cam{ci+1}: C=({e.x0:8.1f},{e.y0:7.1f},{e.z0:8.1f})  cc={cal.int_par.cc:.3f}  "
          f"reproj RMS {rms:6.3f} px (max {np.max(np.linalg.norm(rep-ip,axis=1)):.2f})")
    cal.to_file(str(out / "cal" / f"cam{ci+1}.tif.ori"), str(out / "cal" / f"cam{ci+1}.tif.addpar"))
print("\nwrote .ori + zeroed .addpar")
