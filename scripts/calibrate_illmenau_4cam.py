#!/usr/bin/env python
"""Headless 4-camera calibration pipeline (Illmenau and multi-view 6x7 dot plates).

Uses:
- openptv2.detect_plate (ROI + negative target recognition + coded dot classification)
- openptv2.plate_labeler (L-code corner detection + affine / homography grid registration)
- OpenCV / DLT camera calibration with centered reference frame
- OpenPTV2 .ori and .addpar conversion

Usage:
  uv run python scripts/calibrate_illmenau_4cam.py
  uv run python scripts/calibrate_illmenau_4cam.py --base "C:/Users/alex/Downloads/Illmenau" --out "C:/Users/alex/Downloads/Illmenau/openptv_illmenau_4cam"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import openptv2.algorithms.trafo as tr
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.calibration_import import calibration_from_opencv
from openptv2.detect_plate import detect_plate_targets, plate_tpar_from_yaml
from openptv2.plate_labeler import label_plate

# Labeller grid index of the coded L-corner dot = the world origin (see plate.yaml datum)
DATUM_IX, DATUM_IY = 2, 3


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-camera plate calibration")
    ap.add_argument("--base", type=str, default=r"C:\Users\alex\Downloads\Illmenau", help="Kalibrierung_1..4 parent folder")
    ap.add_argument("--out", type=str, default=r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam", help="OpenPTV2 dataset out folder")
    ap.add_argument("--frame", type=str, default="00000000",
                    help="Reference frame prefix -- its plate pose defines the world")
    ap.add_argument("--all-frames", action="store_true",
                    help="Fit intrinsics over every frame of each folder (multi-plane, "
                         "focal free) instead of the single reference frame with cc pinned")
    ap.add_argument("--min-dots", type=int, default=20,
                    help="Drop a frame from the multi-plane fit below this many labelled dots")
    ap.add_argument("--pitch", type=float, default=120.0, help="Grid dot pitch in mm")
    ap.add_argument("--focal-mm", type=float, default=9.44, help="Nominal lens focal length in mm")
    ap.add_argument("--update-rig", action="store_true", help="Write rig.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    base = Path(args.base)
    out = Path(args.out)
    pitch_val = float(args.pitch)
    focal_mm = float(args.focal_mm)

    folders = [base / f"Kalibrierung_{i}" for i in (1, 2, 3, 4)]
    for f in folders:
        if not f.exists():
            raise FileNotFoundError(f"Missing camera calibration folder: {f}")

    yaml_path = out / "parameters_Run1.yaml"
    tpar = plate_tpar_from_yaml(yaml_path)
    cpar = ControlPar(
        num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
        img_base_name=[""] * 4, cal_img_base_name=[""] * 4,
    )

    print(f"=== Multi-Camera Plate Calibration (Frame {args.frame}) ===")
    img_pts_all = []
    ref_pts_all = []

    for ci, fld in enumerate(folders):
        tifs = list(fld.glob(f"{args.frame}*.tiff")) + list(fld.glob(f"{args.frame}*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No image matching '{args.frame}*' in {fld}")
        img_path = tifs[0]
        raw = np.array(Image.open(img_path))
        res = detect_plate_targets(raw, tpar, cpar, cam=ci, coded_thr=30.0)
        img_pts, ref_pts, idx = label_plate(
            res.centroids, res.coded_mask,
            pitch_x=pitch_val, pitch_y=pitch_val,
            nx=6, ny=7, y_sign=1,
        )
        # World origin (0,0,0) is the coded L-corner dot of frame 00000000:
        # 3rd column from the left, 4th row from top and from bottom, i.e.
        # labeller grid (ix, iy) = (2, 3) -- verified from the three coded dots
        # at (2,3), (2,4), (4,3).  In the barrel frame (origin on the axis at
        # mid-height) this datum sits at (0, -3580/2 + 615, 0) = (0, -1175, 0).
        ref_pts[:, 0] -= DATUM_IX * pitch_val
        ref_pts[:, 1] -= DATUM_IY * pitch_val
        img_pts_all.append(img_pts)
        ref_pts_all.append(ref_pts)
        print(f"Cam {ci+1}: Detected {len(res.targets)} dots | Labeled {len(img_pts)}/42 dots")

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    cals = []
    out_cal = out / "cal"
    out_cal.mkdir(parents=True, exist_ok=True)

    f_px = focal_mm / 0.005
    K_init = np.array([[f_px, 0, 1280.0], [0, f_px, 1024.0], [0, 0, 1.0]], dtype=np.float32)

    for cam in range(4):
        if has_cv2:
            obj = [ref_pts_all[cam].astype(np.float32)]
            img = [img_pts_all[cam].astype(np.float32)]
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                obj, img, (2560, 2048), K_init.copy(), None,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH
            )
            cal, _ = calibration_from_opencv(
                K, dist, rvecs[0], tvecs[0],
                imx=2560, imy=2048, pix_x=0.005, pixel_origin="corner",
            )
        else:
            from openptv2.calibration_seed import seed_from_dlt
            cal = seed_from_dlt(ref_pts_all[cam], img_pts_all[cam], cpar)

        cals.append(cal)
        ori_path = out_cal / f"cam{cam+1}.tif.ori"
        addpar_path = out_cal / f"cam{cam+1}.tif.addpar"
        cal.to_file(str(ori_path), str(addpar_path))

    print("\n=== Calibrated Camera Orientations ===")
    for ci, cal in enumerate(cals):
        errors = []
        for k in range(len(ref_pts_all[ci])):
            P = ref_pts_all[ci][k]
            x_mm, y_mm = img_coord(P, cal, cpar.mm)
            px, py = tr.metric_to_pixel(x_mm, y_mm, cpar)
            det_px, det_py = img_pts_all[ci][k]
            err = np.sqrt((px - det_px) ** 2 + (py - det_py) ** 2)
            errors.append(err)
        print(f"Cam {ci+1}: Pos=({cal.ext_par.x0:7.1f}, {cal.ext_par.y0:7.1f}, {cal.ext_par.z0:7.1f}) mm | cc={cal.int_par.cc:.2f} mm | Reproj RMS={np.mean(errors):.3f} px (max {np.max(errors):.3f} px)")

    # Write rig.yaml
    rig_path = out / "rig.yaml"
    rig = {
        "volume_centre": [0.0, 0.0, 0.0],
        "cameras": [
            {
                "id": ci + 1,
                "position": [float(cal.ext_par.x0), float(cal.ext_par.y0), float(cal.ext_par.z0)],
                "focal_mm": float(cal.int_par.cc),
                "orientation_angles_rad": [float(cal.ext_par.omega), float(cal.ext_par.phi), float(cal.ext_par.kappa)],
            }
            for ci, cal in enumerate(cals)
        ]
    }
    rig_path.write_text(yaml.safe_dump(rig, sort_keys=False))
    print(f"\nWrote calibration files to {out_cal}")
    print(f"Updated {rig_path}")


if __name__ == "__main__":
    main()
