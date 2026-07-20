#!/usr/bin/env python
"""Dump per-camera matched calibration-body point IDs + detected/reprojected
pixel coords for the *current* calibration, and render an ID-labeled overlay
PNG per camera (the overlays `calib.py run`/`recalibrate_*` produce only show
green/red dots with no way to tell which calibration-body point is which --
useful when diagnosing why specific points fail to detect or match).

Camera image/.ori/.addpar paths and the calblock path are resolved from the
dataset YAML's cal_ori: block via openptv2.autocalibration.cam_files() /
resolve_calblock() -- the same files the GUI uses, not a separate camN.tif
naming convention.

Writes, per camera, into cal/calib_matches/:
  camN_matches.txt       "id det_x det_y rep_x rep_y"
  camN_overlay_ids.png   image + detected(green)/reprojected(red) + point ID labels

Run with: uv run python skills/openptv-calibrate/scripts/dump_matches.py <dataset>
"""
from __future__ import annotations

import sys
from pathlib import Path

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.sortgrid import read_calblock, sortgrid
from openptv2.algorithms.tracking_frame_buf import read_targets
from openptv2.autocalibration import _load_dataset_params, _reproject_px, cam_files, resolve_calblock


def render_overlay(base, cam, img_path, ids, det, rep, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v3 as iio
    import numpy as np

    img = iio.imread(img_path)
    rms = float(np.sqrt(np.mean(np.sum((det - rep) ** 2, axis=1)))) if len(det) else float("nan")

    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.imshow(img, cmap="gray")
    ax.scatter(det[:, 0], det[:, 1], s=40, facecolors="none", edgecolors="lime",
               linewidths=1.2, label="detected")
    ax.scatter(rep[:, 0], rep[:, 1], s=8, c="red", label="reprojected")
    for pid, (x, y) in zip(ids, det):
        ax.annotate(str(pid), (x, y), fontsize=6, color="yellow",
                    textcoords="offset points", xytext=(3, 3))
    ax.set_title(f"cam{cam + 1}  RMS={rms:.3f}px  n={len(ids)}  (yellow = calibration-body point ID)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def main():
    import numpy as np

    if len(sys.argv) < 2:
        print("Usage: dump_matches.py <dataset>", file=sys.stderr)
        return 1

    base = Path(sys.argv[1]).resolve()
    calblock = resolve_calblock(base)
    fix, nfix = read_calblock(str(calblock))
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams, eps = dp.cpar, dp.num_cams, dp.eps

    outdir = base / "cal" / "calib_matches"
    outdir.mkdir(exist_ok=True)

    for cam in range(num_cams):
        img, ori, addpar = cam_files(base, cam)
        cal = Calibration.from_file(str(ori), str(addpar))

        pix = read_targets(str(img), 0)
        sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)

        ids, lines, det, rep = [], [], [], []
        for i, t in enumerate(sorted_pix):
            if t.pnr < 0:
                continue
            point_id = i + 1  # fix rows are in sequential-ID order
            rep_x, rep_y = _reproject_px(cal, cpar.mm, fix[i], cpar)
            lines.append(f"{point_id} {t.x:.4f} {t.y:.4f} {rep_x:.4f} {rep_y:.4f}")
            ids.append(point_id)
            det.append((t.x, t.y))
            rep.append((rep_x, rep_y))

        dest = outdir / f"cam{cam + 1}_matches.txt"
        dest.write_text("\n".join(lines) + "\n")

        overlay_dest = outdir / f"cam{cam + 1}_overlay_ids.png"
        render_overlay(base, cam, img, ids, np.asarray(det), np.asarray(rep), overlay_dest)

        print(f"cam{cam + 1}: {len(lines)} matches -> {dest}  overlay -> {overlay_dest}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
