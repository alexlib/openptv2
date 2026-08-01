# ruff: noqa: E501,F401
"""openptv-particle-calib — iterative particle-based calibration refinement."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


def _find_yaml(dataset: Path) -> Path | None:
    for pat in ("parameters_*.yaml", "*.yaml"):
        hits = sorted(dataset.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_core(base: Path):
    """Return (cpar, num_cams, cals, seq_bases, mm) for the dataset."""
    import yaml

    from openptv2.algorithms.calibration import Calibration
    from openptv2.autocalibration import _find_yaml as ac_find_yaml
    from openptv2.autocalibration import _load_dataset_params

    yp = ac_find_yaml(base)
    if yp is None:
        raise RuntimeError("no parameters YAML found")
    y = yaml.safe_load(yp.read_text())
    calblock = base / "cal" / "target_on_a_side.txt"
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams = dp.cpar, dp.num_cams

    seq_raw = y["sequence"]["base_name"]
    seq_bases = [str(base / s.replace("%d", "")) for s in seq_raw]

    cals = []
    for cam in range(num_cams):
        ori = base / "cal" / f"cam{cam + 1}.tif.ori"
        addpar = base / "cal" / f"cam{cam + 1}.tif.addpar"
        cals.append(Calibration.from_file(str(ori), str(addpar)))

    return cpar, num_cams, cals, seq_bases


def _snapshot_refine_once(base: Path, tol_px: float, frames_filter: list[int] | None):
    """Run one snapshot-refine pass. Returns per-cam (n_pts, rms_before, rms_after, flags)."""
    import numpy as np
    import yaml

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.orientation import full_calibration
    from openptv2.algorithms.tracking_frame_buf import read_targets
    from openptv2.autocalibration import (
        _find_yaml as ac_find_yaml,
    )
    from openptv2.autocalibration import (
        _load_dataset_params,
        _reproject_px,
        rms_px,
    )

    yp = ac_find_yaml(base)
    y = yaml.safe_load(yp.read_text())
    calblock = base / "cal" / "target_on_a_side.txt"
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams = dp.cpar, dp.num_cams

    seq_raw = y["sequence"]["base_name"]
    seq_bases = [str(base / s.replace("%d", "")) for s in seq_raw]

    cals = []
    for cam in range(num_cams):
        ori = base / "cal" / f"cam{cam + 1}.tif.ori"
        addpar = base / "cal" / f"cam{cam + 1}.tif.addpar"
        cals.append(Calibration.from_file(str(ori), str(addpar)))

    ptv_is_files = sorted(
        (base / "res").glob("ptv_is.*"),
        key=lambda p: int(p.suffix.lstrip(".")),
    )
    if frames_filter:
        ptv_is_files = [
            p for p in ptv_is_files if int(p.suffix.lstrip(".")) in set(frames_filter)
        ]

    per_cam_ref: list[list] = [[] for _ in range(num_cams)]
    per_cam_det: list[list] = [[] for _ in range(num_cams)]

    for pf in ptv_is_files:
        frame = int(pf.suffix.lstrip("."))
        lines = pf.read_text().splitlines()
        n = int(lines[0])
        pts3d = []
        for line in lines[1 : n + 1]:
            parts = line.split()
            if len(parts) >= 5:
                pts3d.append([float(parts[2]), float(parts[3]), float(parts[4])])
        if not pts3d:
            continue
        pts3d_arr = np.array(pts3d)

        for cam in range(num_cams):
            targets = read_targets(seq_bases[cam], frame)
            if not targets:
                continue
            proj = np.array(
                [_reproject_px(cals[cam], cpar.mm, p, cpar) for p in pts3d_arr]
            )
            tgt_xy = np.array([[t.x, t.y] for t in targets])
            for i, pp in enumerate(proj):
                d = np.linalg.norm(tgt_xy - pp, axis=1)
                j = int(np.argmin(d))
                if d[j] <= tol_px:
                    per_cam_ref[cam].append(pts3d_arr[i])
                    per_cam_det[cam].append(tgt_xy[j])

    SNAP_FLAGS = [
        [],
        ["cc", "xh", "yh"],
        ["cc", "xh", "yh", "k1"],
        ["cc", "xh", "yh", "k1", "k2"],
    ]

    results = []
    new_cals = list(cals)

    for cam in range(num_cams):
        ref = np.array(per_cam_ref[cam]) if per_cam_ref[cam] else np.empty((0, 3))
        det = np.array(per_cam_det[cam]) if per_cam_det[cam] else np.empty((0, 2))

        if len(ref) < 6:
            results.append((len(ref), float("nan"), float("nan"), None))
            continue

        rep_before = np.array([_reproject_px(cals[cam], cpar.mm, r, cpar) for r in ref])
        rms_before = rms_px(det, rep_before)

        best_cal, best_rms, best_flags = None, rms_before, None
        for flags in SNAP_FLAGS:
            trial = copy.deepcopy(cals[cam])
            try:
                full_calibration(trial, ref, det, cpar, flags)
                rep = np.array([_reproject_px(trial, cpar.mm, r, cpar) for r in ref])
                r = rms_px(det, rep)
                if np.isnan(r) or np.any(np.isnan(rep)):
                    continue
                if r < best_rms * 0.99:
                    best_rms, best_cal, best_flags = r, trial, flags
            except Exception:
                continue

        results.append(
            (len(ref), rms_before, best_rms if best_cal else rms_before, best_flags)
        )
        if best_cal is not None:
            new_cals[cam] = best_cal

    return results, new_cals


def cmd_run(args) -> int:
    import numpy as np

    base = Path(args.dataset).resolve()
    if not base.is_dir():
        print(f"ERROR: dataset dir not found: {base}", file=sys.stderr)
        return 1

    if not (base / "res").is_dir() or not list((base / "res").glob("ptv_is.*")):
        print(
            "ERROR: no res/ptv_is.* tracking results found — run tracking first",
            file=sys.stderr,
        )
        return 1

    frames_filter = [int(f) for f in args.frames.split(",")] if args.frames else None
    max_iters = args.max_iters
    tol_rms = args.tol_rms
    tol_px = args.tol_px

    try:
        from openptv2.algorithms.calibration import Calibration
    except ImportError as e:
        print(
            f"ERROR: {e}\nRun from the openptv2 checkout with `uv run`.",
            file=sys.stderr,
        )
        return 1

    print(f"Particle calibration: {base}")
    print(f"  max_iters={max_iters}  tol_rms={tol_rms}px  tol_px={tol_px}px")
    if args.dry_run:
        print("  Dry-run — no files will be written")
    print()

    # header
    try:
        _, num_cams, _, _ = _load_core(base)
    except Exception as e:
        print(f"ERROR loading dataset: {e}", file=sys.stderr)
        return 1

    cam_cols = "  ".join(f"cam{c + 1:>6}" for c in range(num_cams))
    print(f"{'iter':>4}  {cam_cols}  note")
    print("-" * (4 + 2 + num_cams * 9 + 8))

    prev_mean_rms: float | None = None

    for it in range(1, max_iters + 1):
        try:
            results, new_cals = _snapshot_refine_once(base, tol_px, frames_filter)
        except Exception as e:
            print(f"ERROR in iteration {it}: {e}", file=sys.stderr)
            return 1

        after_rms = [r[2] for r in results]
        valid = [r for r in after_rms if not (r != r)]  # filter NaN
        mean_rms = float(np.mean(valid)) if valid else float("nan")

        cam_vals = "  ".join(
            f"{r:>8.3f}" if not (r != r) else "     nan" for r in after_rms
        )
        improved = ""
        if prev_mean_rms is not None and not (mean_rms != mean_rms):
            delta = prev_mean_rms - mean_rms
            improved = f"Δ={delta:+.4f}px"
            if delta < tol_rms:
                print(f"{it:>4}  {cam_vals}  {improved}")
                print(f"\nConverged (Δ < {tol_rms}px). Done after {it} iteration(s).")
                break

        print(f"{it:>4}  {cam_vals}  {improved}")

        if not args.dry_run:
            for cam, cal in enumerate(new_cals):
                ori = base / "cal" / f"cam{cam + 1}.tif.ori"
                addpar = base / "cal" / f"cam{cam + 1}.tif.addpar"
                ori.rename(ori.with_suffix(f".pcbak{it}"))
                addpar.rename(addpar.with_suffix(f".addpar.pcbak{it}"))
                cal.write(str(ori), str(addpar))

        prev_mean_rms = mean_rms
    else:
        print(f"\nReached max_iters={max_iters} without convergence.")

    return 0


def cmd_status(args) -> int:
    """Show current cal-plate RMS vs latest snapshot RMS."""

    base = Path(args.dataset).resolve()
    try:
        results, _ = _snapshot_refine_once(base, tol_px=args.tol_px, frames_filter=None)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"{'cam':<6}  {'n_pts':>6}  {'before_rms':>10}  {'after_rms':>10}")
    for cam, (n, before, after, flags) in enumerate(results):
        b_str = f"{before:.3f}px" if not (before != before) else "   nan"
        a_str = f"{after:.3f}px" if not (after != after) else "   nan"
        print(f"cam{cam + 1:<3}  {n:>6}  {b_str:>10}  {a_str:>10}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="openptv iterative particle-based calibration"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="iterate snapshot-refine until convergence")
    p.add_argument("dataset")
    p.add_argument(
        "--max-iters", type=int, default=5, help="max refinement iterations (default 5)"
    )
    p.add_argument(
        "--tol-rms",
        type=float,
        default=0.05,
        help="stop when mean RMS improvement < this value in px (default 0.05)",
    )
    p.add_argument(
        "--tol-px",
        type=float,
        default=5.0,
        help="match tolerance for particle↔target pairing in px (default 5)",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="comma-separated frame numbers to use (default: all)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="compute but do not write .ori/.addpar"
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "status", help="show current vs potential snapshot RMS, no writing"
    )
    p.add_argument("dataset")
    p.add_argument("--tol-px", type=float, default=5.0)
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
