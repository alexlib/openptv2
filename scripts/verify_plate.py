#!/usr/bin/env python
"""Verify plate reconstruction — per-plane planarity + pitch.

Headless check mirroring ``manual_openptv_orientation_from_opencv_pipeline``'s
2D reprojection and 3D horizontal/vertical error prints, plus the hub Step 7
``cross_camera_rcm`` report.  Run after ``import_calibration.py``.

    uv run python scripts/verify_plate.py --cals out/ --points points_dir --imx 2560 --imy 2048 --pix 0.005

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.calibration_import import read_xyXYZ


def _load_cals(cals_dir: Path, num_cams: int, pix: float):
    cals = []
    for cam in range(num_cams):
        # Try out/camN.ori then cal/camN.tif.ori
        cand = [
            cals_dir / f"cam{cam + 1}.ori",
            cals_dir / f"cam{cam + 1}.tif.ori",
            cals_dir / f"c{cam}_cal.ori",
        ]
        ori = next((p for p in cand if p.exists()), None)
        if ori is None:
            raise FileNotFoundError(f"no .ori for cam {cam + 1} in {cals_dir}")
        add = (
            ori.with_suffix(".addpar")
            if not str(ori).endswith(".ori")
            else Path(str(ori).replace(".ori", ".addpar"))
        )
        # Fallback: same dir camN.addpar
        if not add.exists():
            alt = cals_dir / f"cam{cam + 1}.addpar"
            if alt.exists():
                add = alt
        cals.append(Calibration.from_file(str(ori), str(add) if add.exists() else None))
    return cals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cals", type=Path, required=True, help="dir with camN.ori/.addpar"
    )
    ap.add_argument(
        "--points-dir",
        type=Path,
        default=None,
        help="dir with c*_xyXYZ.txt for triangulation check",
    )
    ap.add_argument("--num-cams", type=int, default=4)
    ap.add_argument("--imx", type=int, default=2560)
    ap.add_argument("--imy", type=int, default=2048)
    ap.add_argument("--pix", type=float, default=0.005)
    ap.add_argument(
        "--pitch", type=float, default=40.0, help="expected grid pitch in mm"
    )
    args = ap.parse_args()

    cals = _load_cals(args.cals, args.num_cams, args.pix)
    cpar = ControlPar(
        num_cams=args.num_cams,
        imx=args.imx,
        imy=args.imy,
        pix_x=args.pix,
        pix_y=args.pix,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""] * args.num_cams,
        cal_img_base_name=[""] * args.num_cams,
    )

    # Per-cam reprojection on points files if given
    if args.points_dir is not None:
        from openptv2.autocalibration import _reproject_px, rms_px

        for cam in range(args.num_cams):
            # Find points file
            candidates = (
                list(args.points_dir.glob(f"c{cam}_xyXYZ.txt"))
                + list(args.points_dir.glob(f"c{cam + 1}_xyXYZ.txt"))
                + list(args.points_dir.glob("*xyXYZ*.txt"))
            )
            if not candidates:
                continue
            # need per-cam file, not all
            pfile = None
            for n in [f"c{cam}_xyXYZ.txt", f"c{cam + 1}_xyXYZ.txt"]:
                q = args.points_dir / n
                if q.exists():
                    pfile = q
                    break
            if pfile is None:
                pfile = candidates[cam] if cam < len(candidates) else candidates[0]
            img_pts, ref_pts = read_xyXYZ(pfile)
            rep = np.array(
                [_reproject_px(cals[cam], cpar.mm, r, cpar) for r in ref_pts]
            )
            rms = rms_px(img_pts, rep)
            print(
                f"cam{cam + 1}: {len(ref_pts)} pts  RMS={rms:.4f}px  (file {pfile.name})"
            )

            # Per-point planarity/pitch if ref points are on planes (Z varies)
            # Group by rounded Z to find planes
            zs = np.round(ref_pts[:, 2], 1)
            uniq = np.unique(zs)
            for z in uniq[:3]:
                mask = zs == z
                plane = ref_pts[mask]
                if len(plane) < 4:
                    continue
                # Fit plane via SVD
                ctr = plane.mean(axis=0)
                _, s, Vt = np.linalg.svd(plane - ctr)
                n = Vt[-1]
                dists = np.abs((plane - ctr) @ n)
                print(
                    f"  Z={z:g}  n={len(plane)}  planarity RMS={float(np.sqrt(np.mean(dists**2))):.4f}mm max={float(dists.max()):.4f}mm"
                )
                # Pitch check: nearest neighbour distances
                from scipy.spatial import cKDTree

                tree = cKDTree(plane[:, :2])
                d, _ = tree.query(plane[:, :2], k=2)
                nn = d[:, 1]
                print(
                    f"    pitch NN median={float(np.median(nn)):.2f} (expected {args.pitch}) std={float(nn.std()):.2f}"
                )

    # Pairwise camera separations (frame-invariant)
    poses = np.array([c.get_pos() for c in cals])
    print("pairwise |C_a - C_b| [mm]:")
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            print(
                f"  cam{i + 1}-cam{j + 1}: {float(np.linalg.norm(poses[i] - poses[j])):.1f}"
            )

    # Cross-camera RCM if we can build a joint set from points files
    if args.points_dir is not None:
        from openptv2.autocalibration import CamResult, cross_camera_rcm

        # Build CamResults synthetically for RCM (need det/rep/ref)
        results = []
        for cam in range(args.num_cams):
            pfile = None
            for n in [f"c{cam}_xyXYZ.txt", f"c{cam + 1}_xyXYZ.txt"]:
                q = args.points_dir / n
                if q.exists():
                    pfile = q
                    break
            if pfile is None or not pfile.exists():
                continue
            img_pts, ref_pts = read_xyXYZ(pfile)
            from openptv2.autocalibration import _reproject_px, rms_px

            rep = np.array(
                [_reproject_px(cals[cam], cpar.mm, r, cpar) for r in ref_pts]
            )
            rms = rms_px(img_pts, rep)
            results.append(
                CamResult(
                    cam=cam,
                    matched=len(ref_pts),
                    nfix=len(ref_pts),
                    rms=rms,
                    flags=[],
                    cal=cals[cam],
                    ref=ref_pts,
                    det=img_pts,
                    rep=rep,
                )
            )
        if len(results) >= 2:
            rcm = cross_camera_rcm(results, cpar)
            if rcm:
                print(
                    f"cross_camera RCM: median={rcm['median']:.4f}mm p90={rcm['p90']:.4f} max={rcm['max']:.4f} n={rcm['n_points']}"
                )
            else:
                print("cross_camera RCM: n/a")

    return 0


if __name__ == "__main__":
    sys.exit(main())
