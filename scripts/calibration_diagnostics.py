#!/usr/bin/env python
"""Headless calibration sanity report -- print, no GUI/marimo required.

    uv run python scripts/calibration_diagnostics.py \\
        --models "current=cal/parameters_Run1.yaml"
    uv run python scripts/calibration_diagnostics.py \\
        --models "before=cal_backup,after=cal" --calblock cal/atrium_calblock_new.txt
    uv run python scripts/calibration_diagnostics.py \\
        --models "current=cal" --plot out.png

Checks a passing reprojection RMS doesn't catch: sight-line angle from each
camera to the calibration-body centroid (a camera aimed away from the target
is a red flag even with good RMS at its matched points), and cross-camera
centroid-distance spread (an asymmetric rig, or a bad manual-orientation seed
on one camera, shows up here even when the bundle adjustment still converged).
See openptv2.calibration_diagnostics for the shared logic, also used by the
interactive marimo viewer at src/openptv2/gui/visualize_calibration_nb.py.

The numeric report needs only numpy (already a core dependency); --plot
additionally needs matplotlib (`uv run --extra viz ...`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openptv2.calibration_diagnostics import (
    compute_diagnostics,
    load_model,
    parse_models_arg,
    resolve_centroid,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models",
        required=True,
        help="label=path pairs, comma-separated. path is a parameters_*.yaml "
        "or a directory of cam*.tif.ori/.addpar",
    )
    ap.add_argument(
        "--calblock",
        default=None,
        help="calibration-body .txt (optional if a model YAML has fixp_name)",
    )
    ap.add_argument(
        "--angle-threshold",
        type=float,
        default=15.0,
        help="flag a camera whose optical axis is off the centroid by more "
        "than this many degrees (default 15)",
    )
    ap.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="save a static 3D comparison PNG here (requires matplotlib)",
    )
    args = ap.parse_args()

    models = {}
    calblock_path = (
        Path(args.calblock).expanduser().resolve() if args.calblock else None
    )
    for label, path in parse_models_arg(args.models):
        cams, calblock_guess = load_model(path)
        models[label] = cams
        if calblock_path is None and calblock_guess is not None:
            calblock_path = calblock_guess

    if not models:
        print("no models loaded -- check --models", file=sys.stderr)
        return 1

    body, centroid = resolve_centroid(models, calblock_path)
    if body is None:
        print(
            "no calibration body found -- sight-line checks use the "
            "camera-cluster centroid instead\n"
        )

    diagnostics = compute_diagnostics(
        models, centroid, angle_flag_deg=args.angle_threshold
    )

    any_flag = False
    for label, d in diagnostics.items():
        print(f"=== {label} ===")
        for c in d.cameras:
            rms = f"{c.rms:.2f}px" if c.rms is not None else "n/a"
            matched = str(c.matched) if c.matched is not None else "n/a"
            flag = "  <-- axis not pointing at calblock" if c.flag else ""
            any_flag = any_flag or c.flag
            print(
                f"{c.name}: dist={c.dist:7.1f}mm  axis_off={c.angle:5.1f}deg  "
                f"RMS={rms:>8}  matched={matched}{flag}"
            )
        spread_flag = "  <-- large spread relative to rig size" if d.flag else ""
        any_flag = any_flag or d.flag
        print(f"centroid-distance spread: {d.spread:.1f}mm{spread_flag}\n")

    if args.plot:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        if body is not None and len(body):
            ax.scatter(
                body[:, 0],
                body[:, 1],
                body[:, 2],
                s=6,
                c="gray",
                alpha=0.4,
                label="calibration body",
            )
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (label, cams) in enumerate(models.items()):
            color = colors[i % len(colors)]
            for name, pos, rot, _ori_path in cams:
                ax.scatter(*pos, s=70, c=color, marker="^")
                ax.text(*pos, f"  {label}:{name}", fontsize=7, color=color)
        ax.scatter(*centroid, s=40, c="black", marker="x", label="centroid")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")
        ax.legend(loc="upper left")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"saved plot: {args.plot}")

    return 1 if any_flag else 0


if __name__ == "__main__":
    sys.exit(main())
