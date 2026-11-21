"""D1 step 1: how well do single-blob cues separate one tracer from two or more? (rendered truth, syn1)

    uv run python scripts/d1_features.py [--frames 1 5 9 13]
Writes the per-blob table to the scratch npz given by --out.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SYN = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"
RUN = SYN / "syn1" / "test"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > random negative); ties count half."""
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    for v in np.unique(allv):                      # average ranks for ties
        m = allv == v
        r[m] = r[m].mean()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 5, 9, 13])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.chdir(RUN)
    from openptv2.blob_hypotheses import FEATURES, blob_features
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import (
        DEFAULT_HIGHPASS_FILTER_SIZE,
        py_start_proc_c,
        read_frame_images,
        simple_highpass,
    )
    from openptv2.segmentation import _vectorized_targ_rec, target_recognition

    pm = ParameterManager()
    pm.from_yaml(RUN / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    ncam = len(cals)
    base = pm.get_parameter("sequence")["base_name"]
    ptv = pm.get_parameter("ptv") or {}
    thres = float(tpar.get_grey_thresholds()[0])
    tr = np.load(SYN / "truth/truth.npz")
    rows = []
    for f in a.frames:
        raw = read_frame_images(pm, base, ncam, f)
        m = tr["frame"] == f
        for c in range(ncam):
            hp = simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
            pix: list = []
            ts = _vectorized_targ_rec(
                hp, int(tpar.get_grey_thresholds()[c]), int(tpar.get_max_discontinuity()),
                *tpar.get_pixel_count_bounds(), *tpar.get_xsize_bounds(), *tpar.get_ysize_bounds(),
                int(tpar.get_min_sum_grey()), 1, cpar.get_image_size()[0] - 1, 1, cpar.get_image_size()[1] - 1,
                pixels_out=pix)
            if f == a.frames[0] and c == 0:      # same detections as the production detector?
                prod = target_recognition(hp, tpar, c, cpar)
                print(f"reference {len(ts)} blobs, production {len(prod)} blobs")
            det = np.array([[t.pos()[0], t.pos()[1], t.count_pixels()[0], t.count_pixels()[1], t.count_pixels()[2],
                             t.sum_grey_value()] for t in ts])
            tp = tr["pixel_xy_per_cam"][m][:, 2 * c: 2 * c + 2]
            tp = tp[~np.isnan(tp[:, 0])]
            d, j = cKDTree(det[:, :2]).query(tp, distance_upper_bound=3.0)
            cnt = np.bincount(j[np.isfinite(d)], minlength=len(det))
            for i, (x, y, n, nx, ny, sg) in enumerate(det):
                ft = blob_features(pix[i], thres)
                if ft is not None:
                    rows.append((f, c, i, cnt[i]) + ft)
    arr = np.array(rows)
    if a.out:
        np.savez(a.out, arr=arr, cols=("frame", "cam", "blob", "truth_count") + FEATURES)
    tc = arr[:, 3]
    print(f"blobs {len(arr)}: truth count 0: {(tc==0).sum()}, 1: {(tc==1).sum()}, 2: {(tc==2).sum()}, >=3: {(tc>=3).sum()}")
    one, two = tc == 1, tc >= 2
    print(f"thres {thres}; AUC of each cue for 'two or more' against 'one' (0.5 = no information)")
    for k, name in enumerate(FEATURES):
        v = arr[:, 4 + k]
        print(f"  {name:9s} AUC {auc(v[two], v[one]):.3f}   median one {np.median(v[one]):8.3f}   median two+ {np.median(v[two]):8.3f}")
    sumg = arr[:, 4] * arr[:, 5]
    print(f"  sumg      AUC {auc(sumg[two], sumg[one]):.3f}")


if __name__ == "__main__":
    main()
