#!/usr/bin/env python
"""Headless Illmenau 1-4 calibration — ROI + NEG + 42 lattice filter -> XYZ->xy collections -> .ori.

Reuses the marimo-paired logic from notebooks/illmenau_4cam_pipeline.py without
requiring a running marimo kernel:

* find_plate_roi: very-blurred bright rectangle (gaussian sigma=25 -> Otsu -> largest label -> pad 0.07)
* NEG: 255-work8 before preprocess_image (dark-on-white plate -> bright via negative)
* positional outlier: KDTree k=5 neighbor-cost keep 42 lowest + plate-intensity gate outer>100

Produces flat collections per cam and calibrates each cam via OpenCV
(``cv2.calibrateCamera`` when available, else DLT fallback), converts to
openPTV ``.ori/.addpar`` via ``calibration_import.calibration_from_opencv``.

Usage:
  uv run python scripts/calibrate_illmenau_4cam.py
  uv run python scripts/calibrate_illmenau_4cam.py --base "C:/Users/alex/Downloads/Illmenau" --out "C:/Users/alex/Downloads/Illmenau/openptv_illmenau_4cam" --pitch 120 --gv 20 --sumg 5000

Writes:
  <out>/cal/cam1.tif.ori ... cam4.tif.ori (+ .addpar)
  <out>/cal/collections.npz  (flat XYZ->xy per cam for debugging)
  updates <out>/rig.yaml with calibrated positions if --update-rig
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# detectors (verbatim from notebook sfhG, but without marimo graph duplicates)
# ---------------------------------------------------------------------------

def find_plate_roi(work8: np.ndarray, sigma: float = 25, pad: float = 0.07):
    imy, imx = work8.shape
    blurred = gaussian_filter(work8.astype(float), sigma=sigma)
    hist, _ = np.histogram(blurred, bins=256, range=(0, 255))
    total = blurred.size
    sum_tot = (hist * np.arange(256)).sum()
    sumB = 0
    wB = 0
    max_var = 0
    thresh = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_tot - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > max_var:
            max_var = var
            thresh = t
    bw = (blurred > thresh).astype(np.uint8) * 255
    labeled, n = nd_label(bw)
    if n == 0:
        return 1, imx - 1, 1, imy - 1, thresh, bw
    areas = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        area = len(xs)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        areas.append((area, (x0, y0, x1 - x0 + 1, y1 - y0 + 1)))
    areas.sort(reverse=True)
    _, (x, y, w, h) = areas[0]
    x0 = int(max(1, x - w * pad))
    y0 = int(max(1, y - h * pad))
    x1 = int(min(imx - 1, x + w + w * pad))
    y1 = int(min(imy - 1, y + h + h * pad))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return 1, imx - 1, 1, imy - 1, thresh, bw
    return x0, x1, y0, y1, thresh, bw


def _outer_mean(work8: np.ndarray, x: float, y: float, r_outer: int = 25, r_inner: int = 6) -> float:
    xi = int(round(x))
    yi = int(round(y))
    imy, imx = work8.shape
    x0 = max(0, xi - r_outer); x1 = min(imx, xi + r_outer + 1)
    y0 = max(0, yi - r_outer); y1 = min(imy, yi + r_outer + 1)
    win = work8[y0:y1, x0:x1]
    if win.size == 0:
        return 0
    cx = xi - x0; cy = yi - y0
    cx0 = max(0, cx - r_inner); cx1 = min(win.shape[1], cx + r_inner + 1)
    cy0 = max(0, cy - r_inner); cy1 = min(win.shape[0], cy + r_inner + 1)
    outer_sum = int(win.sum()) - int(win[cy0:cy1, cx0:cx1].sum())
    outer_area = win.size - (cy1 - cy0) * (cx1 - cx0)
    return outer_sum / outer_area if outer_area > 0 else 0


def _estimate_grid_axes(cent: np.ndarray):
    cov = np.cov(cent.T)
    vals, vecs = np.linalg.eigh(cov)
    ex = vecs[:, np.argmax(vals)]
    ey = np.array([-ex[1], ex[0]])
    ex /= np.linalg.norm(ex); ey /= np.linalg.norm(ey)
    tree = cKDTree(cent)
    dists, _ = tree.query(cent, k=2)
    pitch = float(np.median(dists[:, 1]))
    return ex, ey, pitch


def reject_outside_grid(cent: np.ndarray, work8=None, target: int = 42, outer_thresh: float = 100):
    cent = np.asarray(cent, float)
    if len(cent) == 0:
        return cent
    if work8 is not None:
        outer = np.array([_outer_mean(work8, x, y) for x, y in cent])
        keep_int = outer > outer_thresh
        if keep_int.sum() >= target and keep_int.sum() < len(cent):
            cent = cent[keep_int]
            if len(cent) == target:
                return cent
    if len(cent) <= target:
        return cent
    ex, ey, pitch = _estimate_grid_axes(cent)
    median = np.median(cent, axis=0)
    vx = ey * pitch; vy = ex * pitch
    expected = []
    for iy in [-3, -2, -1, 0, 1, 2, 3]:
        for ix in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
            expected.append(median + ix * vx + iy * vy)
    expected = np.array(expected)
    tree_exp = cKDTree(expected)
    dists, _ = tree_exp.query(cent, k=1)
    order = np.argsort(dists)
    keep_idx = order[:target]
    return cent[keep_idx]


def reject_by_neighbor_cost(cent: np.ndarray, target: int = 42, k: int = 5):
    cent = np.asarray(cent, float)
    if len(cent) <= target:
        return cent
    tree = cKDTree(cent)
    dists, _ = tree.query(cent, k=k)
    cost = np.sum(dists[:, 1:], axis=1)
    keep = np.argsort(cost)[:target]
    return cent[keep]


def reject_outside_grid_v2(cent: np.ndarray, work8=None, target: int = 42, outer_thresh: float = 100):
    filt = reject_by_neighbor_cost(cent, target=target, k=5)
    if work8 is not None and len(filt) == target:
        tree = cKDTree(cent)
        dists, _ = tree.query(cent, k=5)
        cost = np.sum(dists[:, 1:], axis=1)
        order = np.argsort(cost)
        outer_filt = np.array([_outer_mean(work8, x, y) for x, y in filt])
        low_mask = outer_filt < outer_thresh
        if low_mask.any():
            for idx in order[target:]:
                if _outer_mean(work8, cent[idx, 0], cent[idx, 1]) > outer_thresh:
                    worst = int(np.argmin(outer_filt))
                    filt[worst] = cent[idx]
                    outer_filt[worst] = _outer_mean(work8, cent[idx, 0], cent[idx, 1])
                    if (outer_filt < outer_thresh).sum() == 0:
                        break
    return filt


def detect_plate_points(image_path: Path, pitch_val: float, gv_val: int, sumg_val: int):
    from openptv2.algorithms.parameters import ControlPar, MmNp, TargetPar
    from openptv2.image_processing import preprocess_image
    from openptv2.segmentation import target_recognition

    raw = np.array(Image.open(image_path))
    if raw.ndim == 3:
        raw = np.mean(raw, axis=2).astype(raw.dtype)
    lo, hi = float(np.percentile(raw, 1)), float(np.percentile(raw, 99.5))
    work8 = np.clip((raw.astype(float) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    xmin, xmax, ymin, ymax, _, _ = find_plate_roi(work8, sigma=25, pad=0.07)
    work8_neg = (255 - work8).astype(np.uint8)
    cpar = ControlPar(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                      mm=MmNp(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
                      img_base_name=[""], cal_img_base_name=[""])
    hp = preprocess_image(work8_neg, 1, cpar, 25)
    tpar = TargetPar(gvthres=[gv_val] * 4, discont=80, nnmin=10, nnmax=5000, nxmin=8, nxmax=80, nymin=8, nymax=80, sumg_min=sumg_val, cr_sz=3)
    tg = target_recognition(hp, tpar, 0, cpar, subrange_x=(xmin, xmax), subrange_y=(ymin, ymax))
    tg = [t for t in tg if not (t.n == 1 and t.x == 1 and t.y == 1)]
    cent = np.array([[t.x, t.y] for t in tg], float) if tg else np.zeros((0, 2))
    n_raw = len(cent)
    if n_raw > 42:
        cent = reject_outside_grid_v2(cent, work8=work8, target=42, outer_thresh=100)
    # cv2 alt on ROI (optional, for parity with notebook)
    cv2_corners = None
    try:
        import cv2  # type: ignore
        roi = work8[ymin:ymax, xmin:xmax]
        found, corners = cv2.findCirclesGrid(roi, (6, 7), flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        if found:
            corners = corners.reshape(-1, 2)
            corners[:, 0] += xmin
            corners[:, 1] += ymin
            cv2_corners = corners
    except Exception:
        cv2_corners = None
    return {
        "raw_path": image_path,
        "work8": work8,
        "roi": (xmin, xmax, ymin, ymax),
        "cent_raw": np.array([[t.x, t.y] for t in tg], float) if tg else np.zeros((0, 2)),
        "cent_filt": cent,
        "n_raw": n_raw,
        "n_filt": len(cent),
        "cv2_corners": cv2_corners,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Illmenau 1-4 headless calibration")
    ap.add_argument("--base", type=str, default=r"C:\Users\alex\Downloads\Illmenau", help="Kalibrierung_1..4 parent")
    ap.add_argument("--out", type=str, default=r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam", help="openptv2 dataset out")
    ap.add_argument("--pitch", type=float, default=120.0, help="grid pitch mm")
    ap.add_argument("--gv", type=int, default=20, help="gvthres")
    ap.add_argument("--sumg", type=int, default=5000, help="sumg_min")
    ap.add_argument("--update-rig", action="store_true", help="overwrite rig.yaml with calibrated extrinsics")
    return ap.parse_args()


def main():
    args = parse_args()
    base = Path(args.base)
    out = Path(args.out)
    pitch_val = float(args.pitch); gv_val = int(args.gv); sumg_val = int(args.sumg)

    folders = [base / f"Kalibrierung_{i}" for i in (1, 2, 3, 4)]
    for f in folders:
        if not f.exists():
            raise FileNotFoundError(f"missing {f}")

    # sync frames by pre-underscore
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for ci, fld in enumerate(folders):
        tifs = list(fld.glob("*.tiff")) + list(fld.glob("*.tif"))
        for p in sorted([x for x in tifs if x.name[:8].isdigit()]):
            frame = p.name.split("_")[0]
            groups[frame][ci] = p
    sync_frames = [k for k, v in groups.items() if len(v) == 4]
    # sort numerically
    sync_frames = sorted(sync_frames)
    print(f"sync_frames={len(sync_frames)} e.g. {sync_frames[:5]}")

    from openptv2.plate_labeler import label_plate

    # XYZ->xy collections per cam: list of (ref, img, frame)
    flat_collections: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {ci: [] for ci in range(4)}
    all_stats = []

    for frame in sync_frames:
        row = [frame]
        for ci, fld in enumerate(folders):
            path = groups[frame][ci]
            det = detect_plate_points(path, pitch_val, gv_val, sumg_val)
            cent = det["cent_filt"]
            try:
                img_pts, ref_pts, _idx = label_plate(cent, None, pitch_x=pitch_val, pitch_y=pitch_val, nx=6, ny=7, y_sign=1)
            except Exception as e:
                print(f"  {frame} cam{ci+1} label failed: {e}")
                img_pts, ref_pts = cent, np.zeros((0, 3))
            # keep well-labeled frames for calibration (allow slightly loose)
            if len(img_pts) >= 20:
                flat_collections[ci].append((ref_pts, img_pts, frame))
            row.append(f"{det['n_raw']}->{len(cent)}->{len(img_pts)}")
            print(f"  {frame} cam{ci+1}: raw {det['n_raw']} filt {len(cent)} labeled {len(img_pts)} roi {det['roi'][:2]} cv2={'y' if det['cv2_corners'] is not None else 'n'}")
        all_stats.append(row)

    # summary
    print("\nframe | cam1 | cam2 | cam3 | cam4")
    for r in all_stats:
        print(" | ".join(r))

    flat_counts = {ci: (len(v), sum(len(x[0]) for x in v)) for ci, v in flat_collections.items()}
    print(f"\nkept per cam (frames, points): {flat_counts}")

    # save collections for debugging
    out_cal = out / "cal"
    out_cal.mkdir(parents=True, exist_ok=True)
    np.savez(out_cal / "collections.npz",
             **{f"cam{ci}_refs": np.array([x[0] for x in v], dtype=object) if v else np.array([]) for ci, v in flat_collections.items()},
             **{f"cam{ci}_imgs": np.array([x[1] for x in v], dtype=object) if v else np.array([]) for ci, v in flat_collections.items()},
             flat_counts=np.array(flat_counts))

    # calibrate each cam
    try:
        import cv2  # type: ignore
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("OpenCV not available -> DLT fallback per cam")

    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.calibration_import import calibration_from_opencv
    from openptv2.calibration_seed import seed_from_dlt

    cpar_dummy = ControlPar(num_cams=1, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
                            mm=MmNp(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
                            img_base_name=[""], cal_img_base_name=[""])

    for ci in range(4):
        views = flat_collections[ci]
        if len(views) < 6:
            print(f"cam{ci+1}: not enough frames ({len(views)}) skip")
            continue
        refs_list = [r for r, _, _ in views]  # each (n,3) with Z=0
        imgs_list = [i for _, i, _ in views]  # each (n,2)

        # need common n? calibrateCamera handles varying n per plane
        if has_cv2:
            # OpenCV expects objectPoints as list of (n,3) float32, imagePoints as (n,2)
            objp = [np.asarray(r, dtype=np.float32) for r in refs_list]
            imgp = [np.asarray(p, dtype=np.float32) for p in imgs_list]
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, (2560, 2048), None, None,
                                                             flags=cv2.CALIB_FIX_K3 if len(objp) < 10 else 0)
            print(f"cam{ci+1} cv2 RMS {ret:.3f} K {K[0,0]:.1f},{K[1,1]:.1f} cx {K[0,2]:.1f} cy {K[1,2]:.1f} dist {dist.ravel()[:4]}")
            # use first plane's pose (frame 00000000 if present) as global rig pose, else average
            # find frame 00000000
            target_frame = "00000000"
            idx0 = next((i for i, (_, _, fr) in enumerate(views) if fr == target_frame), 0)
            rvec0 = rvecs[idx0].ravel(); tvec0 = tvecs[idx0].ravel()
            cal, pix_y = calibration_from_opencv(K, dist.ravel(), rvec0, tvec0, imx=2560, imy=2048, pix_x=0.005)
            # average C for reporting
            Rs = [__import__("scipy").spatial.transform.Rotation.from_rotvec(r.ravel()).as_matrix() for r in rvecs]
            Cs = [ -R.T @ t.ravel() for R, t in zip(Rs, tvecs) ]
            Cmean = np.mean(Cs, axis=0)
            print(f"  cam{ci+1} C mean {Cmean} first {Cs[idx0]}")
        else:
            # DLT fallback: stack all points into one big non-coplanar set by giving each plane a fake Z offset
            # Use per-plane Z = plane_index * 200 mm to make non-coplanar for DLT, then average
            all_ref = []
            all_img = []
            for pi, (r, im, _) in enumerate(views):
                # fake Z per plane to avoid coplanarity degenerate
                rr = r.copy().astype(float)
                rr[:, 2] = pi * 200.0
                all_ref.append(rr); all_img.append(im)
            all_ref = np.vstack(all_ref); all_img = np.vstack(all_img)
            cal = seed_from_dlt(all_ref, all_img, cpar_dummy)
            print(f"cam{ci+1} DLT cc {cal.int_par.cc:.2f} C {cal.ext_par.x0:.0f},{cal.ext_par.y0:.0f},{cal.ext_par.z0:.0f}")

        # write .ori / .addpar
        ori_path = out_cal / f"cam{ci+1}.tif.ori"
        addpar_path = out_cal / f"cam{ci+1}.tif.addpar"
        ori_path.parent.mkdir(parents=True, exist_ok=True)
        cal.to_file(str(ori_path), str(addpar_path))
        print(f"  wrote {ori_path}")

    # also ensure rig.yaml and parameters_Run1.yaml exist for GUI
    # copy/merge rig if needed
    rig_path = out / "rig.yaml"
    if not rig_path.exists():
        # create from look-at seed (same as notebook)
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

    print("done — verify with: uv run python scripts/verify_plate.py --cals openptv_illmenau_4cam/cal --points-dir openptv_illmenau_4cam/cal")


if __name__ == "__main__":
    main()
