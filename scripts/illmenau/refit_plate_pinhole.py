"""Refit all four poses as a PURE PINHOLE on frame 00000000: zero distortion,
principal point at the sensor centre, cc shared.  .addpar is written as zeros."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config as CFG  # noqa: E402
import cv2
import numpy as np
from PIL import Image

from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.trafo import metric_to_pixel
from openptv2.calibration_import import calibration_from_opencv
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_plate

base = CFG.RAW
out = CFG.DIR
PITCH, NX, NY, PIX, IMX, IMY = 120.0, 6, 7, 0.005, 2560, 2048
CC = float(sys.argv[1]) if len(sys.argv) > 1 else 9.44
cpar = CFG.control_par()
tpar = plate_tpar_from_yaml(out / "parameters_Run1.yaml")
K = np.array([[CC/PIX, 0, IMX/2], [0, CC/PIX, IMY/2], [0, 0, 1.0]])
print(f"pure pinhole refit, cc = {CC} mm shared, zero distortion\n")
for ci in range(CFG.NCAM):
    f = sorted(CFG.image_dir(ci).glob(f"{CFG.REF}*.tif*"))[0]
    res = detect_plate_targets(np.array(Image.open(f)), tpar, cpar, cam=ci, coded_thr=30.0)
    ip, rp, _ = label_plate(res.centroids, res.coded_mask, pitch_x=PITCH, pitch_y=PITCH,
                            nx=NX, ny=NY, y_sign=1)
    obj = rp.copy()
    obj[:, 0] -= CFG.DATUM_IX * PITCH
    obj[:, 1] -= CFG.DATUM_IY * PITCH
    ok, rvec, tvec = cv2.solvePnP(obj, ip.astype(np.float64), K, np.zeros(5))
    rvec, tvec = cv2.solvePnPRefineLM(obj, ip.astype(np.float64), K, np.zeros(5), rvec, tvec)
    cal, _ = calibration_from_opencv(K, np.zeros(5), rvec, tvec,
                                     imx=IMX, imy=IMY, pix_x=PIX, pixel_origin="corner")
    rep = np.array([metric_to_pixel(*img_coord(p, cal, cpar.mm), cpar) for p in obj])
    rms = float(np.sqrt(np.mean(np.sum((rep - ip)**2, 1))))
    e = cal.ext_par
    print(f"cam{CFG.cam_number(ci)}: C=({e.x0:8.1f},{e.y0:7.1f},{e.z0:8.1f})  "
          f"cc={cal.int_par.cc:.3f}  "
          f"reproj RMS {rms:6.3f} px (max {np.max(np.linalg.norm(rep-ip,axis=1)):.2f})")
    cal.to_file(*CFG.cam_ori(ci))
print("\nwrote .ori + zeroed .addpar")
