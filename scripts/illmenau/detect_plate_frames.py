"""Detect + L-code label every calibration frame of a camera group, cache to npz.

    detect_plate_frames.py [--cams 1,2,3,4] [--npz cal/labelled_all_frames.npz]

Cache layout: for each cam INDEX WITHIN THE GROUP (0-based), a dict
frame -> (ids[n], pixels[n,2]) with id = iy*6 + ix + 1 on the datum-shifted grid
(id 21 = origin).  Group-relative indexing is what lets cams 5-8 reuse every
downstream script unchanged.

Two pieces of knowledge are handed to the labeller that it cannot get from one
image on its own, and both matter more than the detector settings:

* ``corner_index=(2, 3)`` -- where the coded L corner sits on the physical
  plate.  Without it the labeller anchors the grid on the smallest index it
  happened to detect, so any view missing the leftmost column or the bottom row
  labels every dot one step off.  That is wrong but perfectly self-consistent,
  so no single-view check can catch it.
* ``up_hint`` -- the image direction of world +Y, from the current .ori.  The
  plate is held vertical, so its own +Y is world +Y; under strong perspective
  the coded L's 1:2 right angle is distorted enough that the wrong dot can win
  on geometry alone, rotating or reflecting the whole grid.  The hint settles
  it.  Needs cal/camN.tif.ori to exist -- set ILLMENAU_NO_HINT=1 for the very
  first run of a new rig, then re-run detection once a calibration exists.

Cameras on the far wall see the BACK face of the plate.  The L code still names
the corner and its +X / +Y partners by leg length, and those are properties of
the physical plate, not of the side you view it from -- so ``(ix, iy)`` comes out
right and no mirroring is applied here.  What differs for the far group is the
dot plane's offset through the plate and the camera being on the other side;
both are handled at fit time by resection_from_known_plates.py, which determines
them from the data rather than assuming.
"""
import os
import sys
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

CAMS = [int(c) for c in
        os.environ.get("ILLMENAU_CAMS", "1,2,3,4").split(",") if c.strip()]
NPZ = os.environ.get("ILLMENAU_NPZ", "labelled_all_frames.npz")
for _i, _a in enumerate(sys.argv):
    if _a == "--cams":
        CAMS = [int(c) for c in sys.argv[_i + 1].split(",")]
    elif _a == "--npz":
        NPZ = sys.argv[_i + 1]

base = Path(ILLMENAU_RAW)
out = Path(ILLMENAU_DIR)
PITCH, NX, NY = 120.0, 6, 7
tpar = plate_tpar_from_yaml(out / "parameters_Run1.yaml")
cpar = ControlPar(num_cams=len(CAMS), imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                  mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0), chfield=0, tiff_flag=1,
                  hp_flag=1, allCam_flag=0, img_base_name=[""]*len(CAMS), cal_img_base_name=[""]*len(CAMS))

DATUM_IX, DATUM_IY = 2, 3
CODED_THR = (30, 25, 20, 15, 10)
USE_HINT = os.environ.get("ILLMENAU_NO_HINT", "") == ""
cals = {}
if USE_HINT:
    from openptv2.algorithms.calibration import Calibration
    from openptv2.plate_labeler import image_up_direction
    for ci, cam in enumerate(CAMS):
        ori = out / f"cal/cam{cam}.tif.ori"
        if not ori.exists():
            print(f"cam{cam}: no .ori yet, labelling without the up hint")
            continue
        c = Calibration()
        c.from_file(str(ori), str(out / f"cal/cam{cam}.tif.addpar"))
        cals[ci] = c

print(f"cameras {CAMS} -> cal/{NPZ}")
store = {}
for ci, cam in enumerate(CAMS):
    for f in sorted((base / f"Kalibrierung_{cam}").glob("*.tif*")):
        fr = f.name.split("_")[0]
        try:
            img = np.array(Image.open(f))
            # The plate carries exactly three coded dots, so search for the
            # threshold that finds three rather than fixing one.  A single fixed
            # coded_thr=30 missed them entirely on two views, and label_plate
            # then had nothing to anchor on.
            res = None
            for thr in CODED_THR:
                r = detect_plate_targets(img, tpar, cpar, cam=ci, coded_thr=float(thr))
                if int(r.coded_mask.sum()) == 3:
                    res = r
                    break
            if res is None:
                print(f"cam{cam} {fr}: no coded triple at any threshold, skipped", flush=True)
                continue
            hint = None
            if ci in cals and len(res.centroids):
                hint = image_up_direction(cals[ci], cpar, np.mean(res.centroids, axis=0))
            ip, rp, _ = label_plate(res.centroids, res.coded_mask, pitch_x=PITCH, pitch_y=PITCH,
                                    nx=NX, ny=NY, y_sign=1,
                                    corner_index=(DATUM_IX, DATUM_IY), up_hint=hint)
        except Exception as e:
            print(f"cam{cam} {fr}: {type(e).__name__}", flush=True)
            continue
        ix = np.round(rp[:, 0] / PITCH).astype(int)
        iy = np.round(rp[:, 1] / PITCH).astype(int)
        store[f"c{ci}_{fr}_ids"] = iy * NX + ix + 1
        store[f"c{ci}_{fr}_px"] = ip
        print(f"cam{cam} {fr}: {len(ip)} dots", flush=True)
np.savez_compressed(out / "cal" / NPZ, **store)
print("wrote", out / "cal" / NPZ, len(store)//2, "views")
