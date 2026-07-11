"""openptv-dumbbell — dumbbell calibration CLI for OpenPTV datasets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_yaml(dataset: Path) -> Path | None:
    for pat in ("parameters_*.yaml", "*.yaml"):
        hits = sorted(dataset.glob(pat))
        if hits:
            return hits[0]
    return None


def cmd_check(args) -> int:
    """Validate dumbbell section in the YAML."""
    import yaml
    yp = Path(args.yaml_or_dataset)
    if yp.is_dir():
        yp2 = _find_yaml(yp)
        if yp2 is None:
            print("ERROR: no YAML found", file=sys.stderr)
            return 1
        yp = yp2

    data = yaml.safe_load(yp.read_text())
    db = data.get("dumbbell")
    if db is None:
        print("ERROR: 'dumbbell' section missing from YAML")
        return 1

    required = {
        "dumbbell_scale": "known dumbbell length (mm)",
        "dumbbell_penalty_weight": "weight of length vs ray error",
        "dumbbell_eps": "max length deviation to keep a frame (0 = keep all)",
        "dumbbell_step": "frame stride through sequence",
    }
    ok = True
    for key, desc in required.items():
        val = db.get(key)
        if val is None:
            print(f"MISSING: dumbbell.{key}  ({desc})")
            ok = False
        else:
            print(f"OK: dumbbell.{key} = {val}  ({desc})")

    scale = db.get("dumbbell_scale", 0)
    if float(scale) <= 0:
        print("ERROR: dumbbell_scale must be > 0")
        ok = False
    return 0 if ok else 1


def cmd_run(args) -> int:
    """Run dumbbell calibration end-to-end."""
    yp = Path(args.yaml_or_dataset)
    if yp.is_dir():
        yp2 = _find_yaml(yp)
        if yp2 is None:
            print("ERROR: no YAML found", file=sys.stderr)
            return 1
        yp = yp2

    fixed_cams: list[int] = []
    if args.fixed_cams:
        fixed_cams = [int(c) for c in args.fixed_cams.split(",")]

    try:
        from openptv2.gui.standalone_dumbbell_calibration import run_dumbbell_calibration
    except ImportError as e:
        print(f"ERROR: {e}\nRun from the openptv2 checkout with `uv run`.", file=sys.stderr)
        return 1

    step = args.step  # None means use YAML default
    write = not args.dry_run

    print(f"Dumbbell calibration: {yp}")
    if fixed_cams:
        print(f"  Fixed cameras (0-based): {fixed_cams}")
    if args.dry_run:
        print("  Dry-run — calibration will NOT be written")

    try:
        result = run_dumbbell_calibration(
            yp,
            step=step,
            fixed_cams=fixed_cams,
            maxiter=args.maxiter,
            write=write,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"\nResult:")
    print(f"  Frames used:       {result.n_used} / {result.n_total}")
    print(f"  RMS before (px):   {result.rms_before:.4f}")
    print(f"  RMS after  (px):   {result.rms_after:.4f}")
    print(f"  Improvement:       {result.rms_before - result.rms_after:+.4f} px")
    if not write:
        print("  (dry-run: no files written)")
    else:
        print("  Written: cal/camN.tif.ori + .addpar (originals backed up as *.dbbak)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="openptv dumbbell calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="validate dumbbell YAML section")
    p.add_argument("yaml_or_dataset", help="parameters_*.yaml file or dataset directory")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="run dumbbell calibration")
    p.add_argument("yaml_or_dataset", help="parameters_*.yaml file or dataset directory")
    p.add_argument("--step", type=int, default=None,
                   help="frame stride (default: from YAML dumbbell.dumbbell_step)")
    p.add_argument("--fixed-cams", default="",
                   help="comma-separated 0-based camera indices to keep fixed")
    p.add_argument("--maxiter", type=int, default=1000,
                   help="max optimizer iterations (default 1000)")
    p.add_argument("--dry-run", action="store_true",
                   help="compute but do not write .ori/.addpar")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
