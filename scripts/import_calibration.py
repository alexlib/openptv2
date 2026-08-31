#!/usr/bin/env python
"""Import a foreign calibration (OpenCV / points-file) into openPTV .ori/.addpar.

Two doors → one bundle adjust, per docs/plans/2026-08-30-calibration-hub-multi-source.md
and ...-illmenau-dots-plate-pipeline.md:

    uv run python scripts/import_calibration.py \
        --model-dir <dir with calib_c*.txt> \
        --points-dir <dir with c*_xyXYZ.txt> \
        --num-cams 4 --imx 2560 --imy 2048 --pix 0.005 \
        --out cal_refined [--no-refine] [--pixel-origin corner|centre] [--json report.json]

    uv run python scripts/import_calibration.py seed \
        --rig rig.yaml --dataset <dir> [--out <dir>] [--overwrite] [--dry-run]

No cv2 dependency — numpy/scipy only.  The OpenCV converter is the verified
``S``-on-right block (hub 108); the points reader covers proPTV/MyPTV/
Multiview-Calibration/DaVis 5-col files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# allow `uv run python scripts/...` without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.autocalibration import calibrate_from_source, cross_camera_rcm
from openptv2.calibration_import import (
    calibration_from_opencv,
    read_opencv_flat15,
    read_xyXYZ,
)
from openptv2.calibration_registry import CalibrationPointSet


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    # default import command (no subcommand name for backwards compat — hub Step 5
    # shows `import_calibration.py --model-dir ...` with no subcommand)
    # We implement it as optional subcommand "import" plus top-level flags.
    ap_import = sub.add_parser("import", help="OpenCV model (+ optional points) → .ori")
    ap_seed = sub.add_parser("seed", help="rig.yaml look-at → .ori")

    def _add_import_args(p):
        p.add_argument("--model-dir", type=Path, help="dir with calib_c*.txt (15-float OpenCV)")
        p.add_argument("--points-dir", type=Path, help="dir with c*_xyXYZ.txt (5-col)")
        p.add_argument("--num-cams", type=int, default=4)
        p.add_argument("--imx", type=int, required=True)
        p.add_argument("--imy", type=int, required=True)
        p.add_argument("--pix", type=float, required=True, help="pixel pitch in mm (pix_x); pix_y derived as cc/fy")
        p.add_argument("--pix-y", type=float, default=None, help="override pix_y when fx!=fy is not isotropic")
        p.add_argument("--out", type=Path, required=True, help="output directory for camN.ori/.addpar")
        p.add_argument("--no-refine", action="store_true", help="pure algebraic import, no bundle adjust")
        p.add_argument("--pixel-origin", choices=["corner", "centre"], default="corner")
        p.add_argument("--json", type=Path, default=None, dest="json_path", help="write report JSON here")
        p.add_argument("--eps", type=int, default=15, help="sortgrid eps for refinement (presorted bypass still needs it for report)")
        p.add_argument("--glass-vec", nargs=3, type=float, default=[0, 0, 1], help="glass normal vector")

    _add_import_args(ap)
    _add_import_args(ap_import)

    ap_seed.add_argument("--rig", type=Path, required=True, help="rig.yaml path")
    ap_seed.add_argument("--dataset", type=Path, required=True, help="dataset root (for cam_files resolution)")
    ap_seed.add_argument("--out", type=Path, default=None, help="override output dir (default: dataset via cam_files)")
    ap_seed.add_argument("--overwrite", action="store_true")
    ap_seed.add_argument("--dry-run", action="store_true", help="print derived eps / sortgrid counts without writing")

    args = ap.parse_args()
    # Normalize: if cmd is None but import flags given, treat as import
    if args.cmd is None and (getattr(args, "model_dir", None) is not None or getattr(args, "points_dir", None) is not None):
        args.cmd = "import"
    if args.cmd is None:
        ap.print_help()
        sys.exit(2)
    return args


def _find_model_files(model_dir: Path, num_cams: int) -> list[Path]:
    cands = []
    for cam in range(num_cams):
        # Try calib_c{N}.txt, calib_c{N}.txt with 1-indexed, and generic
        for name in [f"calib_c{cam}.txt", f"calib_c{cam+1}.txt", f"cam{cam}.txt", f"cam{cam+1}.txt"]:
            p = model_dir / name
            if p.exists():
                cands.append(p)
                break
        else:
            # glob fallback
            matches = sorted(model_dir.glob(f"*c{cam}*calib*.txt")) + sorted(model_dir.glob("*calib*.txt"))
            if matches:
                cands.append(matches[cam] if cam < len(matches) else matches[0])
            else:
                raise FileNotFoundError(f"no model file for cam {cam} in {model_dir}")
    return cands


def _find_points_files(points_dir: Path, num_cams: int) -> list[Path]:
    out = []
    for cam in range(num_cams):
        for name in [f"c{cam}_xyXYZ.txt", f"c{cam+1}_xyXYZ.txt", f"cam{cam}_xyXYZ.txt"]:
            p = points_dir / name
            if p.exists():
                out.append(p)
                break
        else:
            # glob any xyXYZ
            matches = sorted(points_dir.glob("*xyXYZ*.txt"))
            if len(matches) >= num_cams:
                # Heuristic: sort and take cam index
                out.append(matches[cam])
            else:
                raise FileNotFoundError(f"no points file for cam {cam} in {points_dir} (tried c{{cam}}_xyXYZ.txt)")
    return out


def _run_import(args) -> int:
    model_dir: Path | None = getattr(args, "model_dir", None)
    points_dir: Path | None = getattr(args, "points_dir", None)
    num_cams: int = int(getattr(args, "num_cams", 4) or 4)
    imx = int(args.imx)
    imy = int(args.imy)
    pix = float(args.pix)
    pix_y = getattr(args, "pix_y", None)
    out: Path = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pixel_origin = getattr(args, "pixel_origin", "corner")
    no_refine = bool(getattr(args, "no_refine", False))
    glass_vec = tuple(getattr(args, "glass_vec", [0, 0, 1]))
    json_path: Path | None = getattr(args, "json_path", None)
    eps = int(getattr(args, "eps", 15))

    if model_dir is None:
        print("need --model-dir for import", file=sys.stderr)
        return 2
    model_files = _find_model_files(model_dir, num_cams)
    points_files = None
    if points_dir is not None and not no_refine:
        points_files = _find_points_files(points_dir, num_cams)

    # Build ControlPar for refinement (air, no glass — hub optics policy)
    cpar = ControlPar(
        num_cams=num_cams, imx=imx, imy=imy, pix_x=pix,
        pix_y=pix_y if pix_y is not None else pix,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
        img_base_name=[""] * num_cams, cal_img_base_name=[""] * num_cams,
    )

    seeds: list[Calibration] = []
    for cam, mf in enumerate(model_files):
        rec = read_opencv_flat15(mf)
        K = np.array([[rec["fx"], 0, rec["cx"]], [0, rec["fy"], rec["cy"]], [0, 0, 1.0]])
        cal, pix_y_used = calibration_from_opencv(
            K, rec["dist"], rec["rvec"], rec["tvec"],
            imx=imx, imy=imy, pix_x=pix, pix_y=pix_y,
            glass_vec=glass_vec, pixel_origin=pixel_origin,
        )
        seeds.append(cal)
        if pix_y is None:
            cpar.pix_y = float(pix_y_used)
        print(f"cam{cam+1}: seed C=({cal.ext_par.x0:.1f},{cal.ext_par.y0:.1f},{cal.ext_par.z0:.1f}) cc={cal.int_par.cc:.3f} pix_y={pix_y_used:.6f}")

    if no_refine or points_files is None:
        for cam, cal in enumerate(seeds):
            ori = out / f"cam{cam+1}.ori"
            add = out / f"cam{cam+1}.addpar"
            if ori.exists():
                shutil.copy2(ori, Path(str(ori) + ".autobck"))
            cal.to_file(str(ori), str(add))
            print(f"wrote {ori} + {add}")
        return 0

    # Refine per cam on all points (presorted)
    results = []
    for cam in range(num_cams):
        img_pts, ref_pts = read_xyXYZ(points_files[cam])
        print(f"cam{cam+1}: {len(ref_pts)} points from {points_files[cam].name}")
        ps = CalibrationPointSet(ref_pts=ref_pts, img_pts=img_pts, seed=seeds[cam])
        res = calibrate_from_source("points_file", cam, cpar, ps, eps=eps, presorted=True)
        results.append(res)
        ori = out / f"cam{cam+1}.ori"
        add = out / f"cam{cam+1}.addpar"
        if ori.exists():
            shutil.copy2(ori, Path(str(ori) + ".autobck"))
        res.cal.to_file(str(ori), str(add))
        print(f"cam{cam+1}: refined matched {res.matched}/{res.nfix} RMS={res.rms:.4f}px flags={'+'.join(res.flags)} -> {ori}")

    # Report — cross_camera_rcm + pairwise separations + per-cam RMS
    rcm = cross_camera_rcm(results, cpar)
    if rcm:
        print(f"cross_camera RCM: median={rcm['median']:.4f}mm p90={rcm['p90']:.4f} n={rcm['n_points']} common={rcm['n_common']}")
    else:
        print("cross_camera RCM: n/a (<2 cams or <3 common points)")
    # pairwise separations (frame-invariant)
    poses = np.array([r.cal.get_pos() for r in results if r.cal is not None])
    if len(poses) >= 2:
        print("pairwise |C_a - C_b| [mm]:")
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                print(f"  cam{i+1}-cam{j+1}: {float(np.linalg.norm(poses[i]-poses[j])):.1f}")
    # JSON report
    if json_path is not None:
        rep = {
            "per_cam": [
                {"cam": r.cam, "matched": int(r.matched), "nfix": int(r.nfix), "rms": float(r.rms), "flags": r.flags}
                for r in results
            ],
            "rcm": rcm,
            "pairwise_sep": {
                f"{i+1}-{j+1}": float(np.linalg.norm(poses[i]-poses[j]))
                for i in range(len(poses)) for j in range(i+1, len(poses))
            },
            "out": str(out),
        }
        json_path.write_text(json.dumps(rep, indent=2))
        print(f"report JSON: {json_path}")
    return 0


def _run_seed(args) -> int:
    from openptv2.calibration_seed import seed_rig, write_rig_ori

    rig = Path(args.rig)
    dataset = Path(args.dataset)
    if getattr(args, "out", None):
        # write_rig_ori resolves its paths inside dataset_dir via cam_files, so
        # there is nowhere to put an override.  This used to be parsed and then
        # dropped on the floor, quietly writing to the dataset instead.
        print("error: seed --out is not implemented; .ori are written into the "
              "dataset via cam_files", file=sys.stderr)
        return 2
    overwrite = bool(getattr(args, "overwrite", False))
    dry_run = bool(getattr(args, "dry_run", False))

    if dry_run:
        cals = seed_rig(rig)
        for cam, cal in cals.items():
            print(f"cam{cam+1}: C=({cal.ext_par.x0:.1f},{cal.ext_par.y0:.1f},{cal.ext_par.z0:.1f}) "
                  f"angles=({cal.ext_par.omega:.4f},{cal.ext_par.phi:.4f},{cal.ext_par.kappa:.4f}) "
                  f"cc={cal.int_par.cc:.3f}")
        print("dry-run: nothing written")
        return 0

    written = write_rig_ori(rig, dataset, overwrite=overwrite)
    for p in written:
        print(f"wrote {p}")
    return 0


def main() -> int:
    args = _parse_args()
    if args.cmd == "seed":
        return _run_seed(args)
    # import (including top-level without subcommand)
    return _run_import(args)


if __name__ == "__main__":
    sys.exit(main())
