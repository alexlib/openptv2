"""D1 step 4: faint candidates (lower-threshold blobs not already detected) against rendered truth, syn1.

    uv run python scripts/d1_faint.py [--frames 1 5] [--thresholds 3 5 7]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SYN = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"
RUN = SYN / "syn1" / "test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--thresholds", type=int, nargs="+", default=[3, 5, 7])
    a = ap.parse_args()
    os.chdir(RUN)
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import (
        DEFAULT_HIGHPASS_FILTER_SIZE,
        py_start_proc_c,
        read_frame_images,
        simple_highpass,
    )
    from openptv2.segmentation import _vectorized_targ_rec

    pm = ParameterManager()
    pm.from_yaml(RUN / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    ncam = len(cals)
    base = pm.get_parameter("sequence")["base_name"]
    ptv = pm.get_parameter("ptv") or {}
    tr = np.load(SYN / "truth/truth.npz")
    imx, imy = cpar.get_image_size()
    thr0 = int(tpar.get_grey_thresholds()[0])

    def detect(hp, thr):
        ts = _vectorized_targ_rec(hp, thr, int(tpar.get_max_discontinuity()), 2, *tpar.get_pixel_count_bounds()[1:],
                                  *tpar.get_xsize_bounds(), *tpar.get_ysize_bounds(), 0, 1, imx - 1, 1, imy - 1)
        return np.array([[t.pos()[0], t.pos()[1], t.count_pixels()[0]] for t in ts if t.count_pixels()[0] > 0])

    tot_missing = 0
    stats = {t: [0, 0, 0] for t in a.thresholds}      # recovered, candidates, false
    for f in a.frames:
        raw = read_frame_images(pm, base, ncam, f)
        m = tr["frame"] == f
        for c in range(ncam):
            hp = simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
            tp = tr["pixel_xy_per_cam"][m][:, 2 * c: 2 * c + 2]
            tp = tp[~np.isnan(tp[:, 0])]
            base_det = detect(hp, thr0)
            d0, _ = cKDTree(base_det[:, :2]).query(tp)
            missing = tp[d0 > 3.0]
            tot_missing += len(missing)
            for t in a.thresholds:
                det = detect(hp, t)
                if len(det) == 0:
                    continue
                dd, _ = cKDTree(base_det[:, :2]).query(det[:, :2])
                cand = det[dd > 3.0]                      # not already a standard blob
                stats[t][1] += len(cand)
                if len(cand):
                    tree = cKDTree(cand[:, :2])
                    dm, _ = tree.query(missing, distance_upper_bound=1.0)
                    stats[t][0] += int(np.isfinite(dm).sum())
                    dt_, _ = cKDTree(tp).query(cand[:, :2])
                    stats[t][2] += int((dt_ > 2.0).sum())
    print(f"standard threshold {thr0}; truth spots with no standard blob: {tot_missing}")
    for t in a.thresholds:
        rec, cand, false = stats[t]
        print(f"  faint threshold {t}: candidates {cand}, recovered missing spots {rec} ({rec/tot_missing:.2f}), "
              f"false candidates {false} ({false/max(rec,1):.2f} per recovered)")


if __name__ == "__main__":
    main()
