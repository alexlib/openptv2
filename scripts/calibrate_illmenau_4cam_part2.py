#!/usr/bin/env python
"""Part 2 — Illmenau 1-4: 3D-2D pairs -> openPTV calibration (.ori/.addpar).

Reads Part 1 outputs (per-camera 3D-2D pairs) and calibrates each camera.
Prefers OpenCV (cv2.calibrateCamera) when available, else DLT fallback.

Usage:
  uv run --with opencv-python python scripts/calibrate_illmenau_4cam_part2.py
  uv run --with opencv-python python scripts/calibrate_illmenau_4cam_part2.py --out "C:/Users/alex/Downloads/Illmenau/openptv_illmenau_4cam"

Reads:
  <out>/cal/pairs_cam1.npz ... cam4.npz  (from Part 1)
  <out>/cal/collections.npz (fallback)

Writes:
  <out>/cal/cam1.tif.ori ... cam4.tif.ori (+ .addpar)
  <out>/rig.yaml (if missing, look-at seed)
  <out>/cal/calib_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def load_pairs(out_cal: Path, ci: int):
    # Try pairs_camN.npz first (from Part 1), else collections.npz
    p = out_cal / f"pairs_cam{ci+1}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        frames = d["frames"].tolist() if "frames" in d else []
        refs = []; imgs = []
        # refs are stored as ref_0, ref_1, ...
        idx = 0
        while f"ref_{idx}" in d:
            refs.append(d[f"ref_{idx}"]); imgs.append(d[f"img_{idx}"]); idx+=1
        return list(zip(refs, imgs, frames))
    # fallback collections.npz
    c = out_cal / "collections.npz"
    if c.exists():
        data = np.load(c, allow_pickle=True)
        refs = data.get(f"cam{ci}_refs", [])
        imgs = data.get(f"cam{ci}_imgs", [])
        # sync_frames not needed
        return list(zip(refs, imgs, [""]*len(refs)))
    return []


def main():
    ap = argparse.ArgumentParser(description="Part 2: 3D-2D pairs -> .ori")
    ap.add_argument("--out", type=str, default=r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam")
    ap.add_argument("--update-rig", action="store_true", help="overwrite rig.yaml with calibrated extrinsics")
    args = ap.parse_args()
    out = Path(args.out); out_cal = out / "cal"
    if not out_cal.exists():
        raise FileNotFoundError(f"{out_cal} missing — run Part 1 first")
    try:
        import cv2  # type: ignore
        has_cv2 = True
        print(f"OpenCV {cv2.__version__} available")
    except ImportError:
        has_cv2 = False
        print("OpenCV not available -> DLT fallback")

    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.calibration_import import calibration_from_opencv
    from openptv2.calibration_seed import seed_from_dlt

    cpar_dummy = ControlPar(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                            mm=MmNp(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
                            img_base_name=[""], cal_img_base_name=[""])

    report = []
    for ci in range(4):
        pairs = load_pairs(out_cal, ci)
        if len(pairs) < 6:
            msg = f"cam{ci+1}: not enough pairs ({len(pairs)}) skip"
            print(msg); report.append(msg); continue
        refs_list = [np.asarray(r, dtype=np.float32) for r, _, _ in pairs]
        imgs_list = [np.asarray(p, dtype=np.float32) for _, p, _ in pairs]
        frames = [f for _, _, f in pairs]
        print(f"\ncam{ci+1}: {len(pairs)} frames, {sum(len(r) for r in refs_list)} points e.g. {frames[:3]}")

        if has_cv2:
            objp = refs_list; imgp = imgs_list
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, (2560, 2048), None, None,
                                                             flags=cv2.CALIB_FIX_K3 if len(objp) < 10 else 0)
            msg = f"cam{ci+1} cv2 RMS {ret:.3f} K {K[0,0]:.1f},{K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f} dist {dist.ravel()[:4].tolist()}"
            print(msg); report.append(msg)
            target_frame = "00000000"
            idx0 = next((i for i, f in enumerate(frames) if f == target_frame), 0)
            rvec0 = rvecs[idx0].ravel(); tvec0 = tvecs[idx0].ravel()
            cal, pix_y = calibration_from_opencv(K, dist.ravel(), rvec0, tvec0, imx=2560, imy=2048, pix_x=0.005)
            import scipy.spatial.transform
            Rs = [scipy.spatial.transform.Rotation.from_rotvec(r.ravel()).as_matrix() for r in rvecs]
            Cs = [ -R.T @ t.ravel() for R, t in zip(Rs, tvecs) ]
            Cmean = np.mean(Cs, axis=0)
            msg2 = f"  cam{ci+1} C mean {Cmean.tolist()} first {Cs[idx0].tolist()} cc {cal.int_par.cc:.2f}"
            print(msg2); report.append(msg2)
        else:
            all_ref = []; all_img = []
            for pi, (r, im, _) in enumerate(pairs):
                rr = r.astype(float).copy(); rr[:, 2] = pi * 200.0
                all_ref.append(rr); all_img.append(im)
            all_ref = np.vstack(all_ref); all_img = np.vstack(all_img)
            cal = seed_from_dlt(all_ref, all_img, cpar_dummy)
            msg = f"cam{ci+1} DLT cc {cal.int_par.cc:.2f} C {cal.ext_par.x0:.0f},{cal.ext_par.y0:.0f},{cal.ext_par.z0:.0f}"
            print(msg); report.append(msg)

        ori_path = out_cal / f"cam{ci+1}.tif.ori"
        addpar_path = out_cal / f"cam{ci+1}.tif.addpar"
        ori_path.parent.mkdir(parents=True, exist_ok=True)
        cal.to_file(str(ori_path), str(addpar_path))
        print(f"  wrote {ori_path}"); report.append(f"wrote {ori_path}")

    # ensure rig.yaml exists
    rig_path = out / "rig.yaml"
    if not rig_path.exists():
        rig = {
            "volume_centre": [0, 615, 0],
            "cameras": [
                {"position": [ 2528, 700, 2528], "target": [0, 615, 0], "focal_mm": 35, "up": [0,1,0]},
                {"position": [ 2528,2900, 2528], "target": [0, 615, 0], "focal_mm": 35, "up": [0,1,0]},
                {"position": [-2528, 700, 2528], "target": [0, 615, 0], "focal_mm": 35, "up": [0,1,0]},
                {"position": [-2528,2900, 2528], "target": [0, 615, 0], "focal_mm": 35, "up": [0,1,0]},
            ]
        }
        rig_path.write_text(yaml.safe_dump(rig, sort_keys=False))
        print(f"wrote {rig_path}")

    (out_cal / "calib_report.txt").write_text("\n".join(report))
    print(f"\nPart 2 done -> {out_cal}/cam*.ori  verify: uv run python scripts/verify_plate.py --cals {out_cal} --points-dir {out_cal}")

if __name__ == "__main__":
    main()
