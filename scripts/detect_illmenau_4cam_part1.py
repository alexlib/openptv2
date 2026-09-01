#!/usr/bin/env python
"""Part 1 — Illmenau 1-4: Kalibrierung images -> per-camera 3D-2D pairs.

Implements the marimo-paired pipeline headless:
  ROI (very-blurred bright rect, sigma=25 -> Otsu -> pad 0.07)
  NEG (255-work8 for dark-on-white plate)
  target_recognition inside ROI only
  positional outlier: KDTree k=5 neighbor-cost keep 42 + plate-intensity gate

Labels with plate_labeler (6x7 pitch 120 y_sign=1) -> XYZ (world) -> xy (image)
Saves per-camera collections for Part 2 calibration.

Usage:
  uv run --with opencv-python python scripts/detect_illmenau_4cam_part1.py
  uv run --with opencv-python python scripts/detect_illmenau_4cam_part1.py --base "C:/Users/alex/Downloads/Illmenau" --out "C:/Users/alex/Downloads/Illmenau/openptv_illmenau_4cam" --pitch 120 --gv 20 --sumg 5000

Outputs:
  <out>/cal/detections_cam1.npz ... cam4.npz  (flat XYZ->xy, per-frame)
  <out>/cal/collections.npz                (all 4 cams together)
  <out>/cal/detections_summary.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree


def find_plate_roi(work8: np.ndarray, sigma: float = 25, pad: float = 0.07):
    imy, imx = work8.shape
    blurred = gaussian_filter(work8.astype(float), sigma=sigma)
    hist, _ = np.histogram(blurred, bins=256, range=(0, 255))
    total = blurred.size
    sum_tot = (hist * np.arange(256)).sum()
    sumB = wB = max_var = thresh = 0
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


def _outer_mean(work8, x, y, r_outer=25, r_inner=6):
    xi = int(round(x))
    yi = int(round(y))
    imy, imx = work8.shape
    x0 = max(0, xi - r_outer)
    x1 = min(imx, xi + r_outer + 1)
    y0 = max(0, yi - r_outer)
    y1 = min(imy, yi + r_outer + 1)
    win = work8[y0:y1, x0:x1]
    if win.size == 0:
        return 0
    cx = xi - x0
    cy = yi - y0
    cx0 = max(0, cx - r_inner)
    cx1 = min(win.shape[1], cx + r_inner + 1)
    cy0 = max(0, cy - r_inner)
    cy1 = min(win.shape[0], cy + r_inner + 1)
    outer_sum = int(win.sum()) - int(win[cy0:cy1, cx0:cx1].sum())
    outer_area = win.size - (cy1 - cy0) * (cx1 - cx0)
    return outer_sum / outer_area if outer_area > 0 else 0


def reject_by_neighbor_cost(cent, target=42, k=5):
    cent = np.asarray(cent, float)
    if len(cent) <= target:
        return cent
    tree = cKDTree(cent)
    dists, _ = tree.query(cent, k=k)
    cost = np.sum(dists[:, 1:], axis=1)
    keep = np.argsort(cost)[:target]
    return cent[keep]


def reject_outside_grid_v2(cent, work8=None, target=42, outer_thresh=100):
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
    cpar = ControlPar(
        num_cams=1,
        imx=2560,
        imy=2048,
        pix_x=0.005,
        pix_y=0.005,
        mm=MmNp(n1=1, n2=[1], d=[0], n3=1),
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""],
        cal_img_base_name=[""],
    )
    hp = preprocess_image(work8_neg, 1, cpar, 25)
    tpar = TargetPar(
        gvthres=[gv_val] * 4,
        discont=80,
        nnmin=10,
        nnmax=5000,
        nxmin=8,
        nxmax=80,
        nymin=8,
        nymax=80,
        sumg_min=sumg_val,
        cr_sz=3,
    )
    tg = target_recognition(
        hp, tpar, 0, cpar, subrange_x=(xmin, xmax), subrange_y=(ymin, ymax)
    )
    tg = [t for t in tg if not (t.n == 1 and t.x == 1 and t.y == 1)]
    cent = np.array([[t.x, t.y] for t in tg], float) if tg else np.zeros((0, 2))
    n_raw = len(cent)
    if n_raw > 42:
        cent = reject_outside_grid_v2(cent, work8=work8, target=42, outer_thresh=100)
    cv2_corners = None
    try:
        import cv2

        roi = work8[ymin:ymax, xmin:xmax]
        found, corners = cv2.findCirclesGrid(
            roi, (6, 7), flags=cv2.CALIB_CB_SYMMETRIC_GRID
        )
        if found:
            corners = corners.reshape(-1, 2)
            corners[:, 0] += xmin
            corners[:, 1] += ymin
            cv2_corners = corners
    except Exception:
        pass
    return {
        "raw_path": image_path,
        "work8": work8,
        "roi": (xmin, xmax, ymin, ymax),
        "cent_raw": np.array([[t.x, t.y] for t in tg], float)
        if tg
        else np.zeros((0, 2)),
        "cent_filt": cent,
        "n_raw": n_raw,
        "n_filt": len(cent),
        "cv2_corners": cv2_corners,
    }


def main():
    ap = argparse.ArgumentParser(description="Part 1: Kalibrierung -> 3D-2D pairs")
    ap.add_argument("--base", type=str, default=r"C:\Users\alex\Downloads\Illmenau")
    ap.add_argument(
        "--out",
        type=str,
        default=r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam",
    )
    ap.add_argument("--pitch", type=float, default=120.0)
    ap.add_argument("--gv", type=int, default=20)
    ap.add_argument("--sumg", type=int, default=5000)
    args = ap.parse_args()
    base = Path(args.base)
    out = Path(args.out)
    pitch_val = float(args.pitch)
    gv_val = int(args.gv)
    sumg_val = int(args.sumg)
    folders = [base / f"Kalibrierung_{i}" for i in (1, 2, 3, 4)]
    for f in folders:
        if not f.exists():
            raise FileNotFoundError(f)
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for ci, fld in enumerate(folders):
        tifs = list(fld.glob("*.tiff")) + list(fld.glob("*.tif"))
        for p in sorted([x for x in tifs if x.name[:8].isdigit()]):
            frame = p.name.split("_")[0]
            groups[frame][ci] = p
    sync_frames = sorted([k for k, v in groups.items() if len(v) == 4])
    print(f"sync_frames={len(sync_frames)} e.g. {sync_frames[:5]}")
    from openptv2.plate_labeler import label_plate

    flat_collections: dict[int, list[tuple[np.ndarray, np.ndarray, str]]] = {
        ci: [] for ci in range(4)
    }
    per_cam_frames: dict[int, list[dict]] = {ci: [] for ci in range(4)}
    for frame in sync_frames:
        for ci in range(4):
            path = groups[frame][ci]
            det = detect_plate_points(path, pitch_val, gv_val, sumg_val)
            cent = det["cent_filt"]
            try:
                img_pts, ref_pts, _idx = label_plate(
                    cent,
                    None,
                    pitch_x=pitch_val,
                    pitch_y=pitch_val,
                    nx=6,
                    ny=7,
                    y_sign=1,
                )
            except Exception as e:
                print(f"  {frame} cam{ci + 1} label fail {e}")
                img_pts, ref_pts = cent, np.zeros((0, 3))
            # store per-frame
            per_cam_frames[ci].append(
                {
                    "frame": frame,
                    "path": str(path),
                    "roi": det["roi"],
                    "n_raw": det["n_raw"],
                    "n_filt": len(cent),
                    "n_labeled": len(img_pts),
                    "img_pts": img_pts,
                    "ref_pts": ref_pts,
                    "cv2_corners": det["cv2_corners"],
                }
            )
            if len(img_pts) >= 20:
                flat_collections[ci].append((ref_pts, img_pts, frame))
            print(
                f"  {frame} cam{ci + 1}: {det['n_raw']}->{len(cent)}->{len(img_pts)} roi {det['roi'][:2]}"
            )
    # save
    out_cal = out / "cal"
    out_cal.mkdir(parents=True, exist_ok=True)
    for ci in range(4):
        # per-camera npz with 3D-2D pairs, flat for cal
        np.savez(
            out_cal / f"detections_cam{ci + 1}.npz",
            frames=np.array([x["frame"] for x in per_cam_frames[ci]]),
            n_raw=np.array([x["n_raw"] for x in per_cam_frames[ci]]),
            n_filt=np.array(
                [len(reject_by_neighbor_cost(np.array([[0, 0]])))]
            ),  # placeholder
        )
        # save explicit 3D-2D pairs as separate npz for Part 2
        # flat_collections already has ref/img
        np.savez(
            out_cal / f"pairs_cam{ci + 1}.npz",
            **{f"ref_{i}": r for i, (r, _, _) in enumerate(flat_collections[ci])},
            **{f"img_{i}": p for i, (_, p, _) in enumerate(flat_collections[ci])},
            frames=np.array([f for _, _, f in flat_collections[ci]]),
        )
        # also save as simple txt per frame for inspection (optional)
    # combined
    np.savez(
        out_cal / "collections.npz",
        **{
            f"cam{ci}_refs": np.array([x[0] for x in v], dtype=object)
            if v
            else np.array([])
            for ci, v in flat_collections.items()
        },
        **{
            f"cam{ci}_imgs": np.array([x[1] for x in v], dtype=object)
            if v
            else np.array([])
            for ci, v in flat_collections.items()
        },
        sync_frames=np.array(sync_frames),
    )
    # summary txt
    with open(out_cal / "detections_summary.txt", "w") as fh:
        fh.write(
            f"pitch {pitch_val} gv {gv_val} sumg {sumg_val} sync_frames {len(sync_frames)}\n"
        )
        for ci in range(4):
            fh.write(
                f"cam{ci + 1}: {len(flat_collections[ci])} frames kept, {sum(len(x[0]) for x in flat_collections[ci])} points total\n"
            )
            for fr, img, ref in [
                (f, img, ref) for ref, img, f in flat_collections[ci][:3]
            ]:
                fh.write(f"  e.g. {fr}: {len(ref)} pts\n")
    print(f"\nPart 1 done -> {out_cal}")
    for ci in range(4):
        print(
            f" cam{ci + 1}: {len(flat_collections[ci])} frames, {sum(len(x[0]) for x in flat_collections[ci])} points -> pairs_cam{ci + 1}.npz + detections_cam{ci + 1}.npz"
        )
    print(
        "Next: Part 2 -> uv run --with opencv-python python scripts/calibrate_illmenau_4cam_part2.py"
    )


if __name__ == "__main__":
    main()
