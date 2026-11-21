"""D1 step 2: two-tracer split of merged blobs on rendered truth (syn1): centre error against the true centres.

    uv run python scripts/d1_split.py [--frames 1 5] [--max-blobs 400]
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
    ap.add_argument("--max-blobs", type=int, default=600)
    a = ap.parse_args()
    os.chdir(RUN)
    from openptv2.blob_hypotheses import fit_one, fit_two
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
    blobs = []          # (hp, pixels, truth spots, centre)
    for f in a.frames:
        raw = read_frame_images(pm, base, ncam, f)
        m = tr["frame"] == f
        for c in range(ncam):
            hp = simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
            pix: list = []
            ts = _vectorized_targ_rec(hp, int(tpar.get_grey_thresholds()[c]), int(tpar.get_max_discontinuity()),
                                      *tpar.get_pixel_count_bounds(), *tpar.get_xsize_bounds(), *tpar.get_ysize_bounds(),
                                      int(tpar.get_min_sum_grey()), 1, imx - 1, 1, imy - 1, pixels_out=pix)
            det = np.array([[t.pos()[0], t.pos()[1]] for t in ts])
            tp = tr["pixel_xy_per_cam"][m][:, 2 * c: 2 * c + 2]
            tp = tp[~np.isnan(tp[:, 0])]
            d, j = cKDTree(det).query(tp, distance_upper_bound=3.0)
            own = np.where(np.isfinite(d), j, -1)
            for i in range(len(det)):
                blobs.append((hp, pix[i], tp[own == i], det[i]))
    # convention offset between target centres and truth, from clean blobs
    off = np.median([b[3] - b[2][0] for b in blobs if len(b[2]) == 1], axis=0)
    print("target centre minus truth, median over clean blobs (px):", np.round(off, 3))
    # single-tracer variance for the Gaussian model: fit on clean blobs
    clean = [b for b in blobs if len(b[2]) == 1][:300]
    best = None
    for s2 in (0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        e = np.median([np.linalg.norm(np.array(fit_one(b[0], b[1], s2)[:2]) + 0.5 - off - b[2][0]) for b in clean])
        c = np.median([fit_one(b[0], b[1], s2)[3] / max(b[1][:, 2].sum(), 1) for b in clean])
        print(f"  psf variance {s2:.2f}: clean centre error median {e:.3f} px, residual cost {c:.3f}")
        if best is None or c < best[1]:
            best = (s2, c)
    s2 = best[0]
    print("using psf variance", s2)
    rng = np.random.default_rng(0)
    two = [b for b in blobs if len(b[2]) == 2]
    rng.shuffle(two)
    one_err, cen_err, fit_err, sep_truth = [], [], [], []
    for hp, px, tp, ctr in two[: a.max_blobs]:
        r = fit_two(hp, px, s2)
        p1, p2 = np.array(r[0:2]) + 0.5 - off, np.array(r[3:5]) + 0.5 - off
        # best assignment of fitted centres to the two truth spots
        e_a = np.linalg.norm(p1 - tp[0]) + np.linalg.norm(p2 - tp[1])
        e_b = np.linalg.norm(p1 - tp[1]) + np.linalg.norm(p2 - tp[0])
        fit_err.extend([np.linalg.norm(p1 - tp[0]), np.linalg.norm(p2 - tp[1])] if e_a <= e_b else
                       [np.linalg.norm(p1 - tp[1]), np.linalg.norm(p2 - tp[0])])
        cen_err.extend([np.linalg.norm(ctr - off - tp[0]), np.linalg.norm(ctr - off - tp[1])])
        sep_truth.append(np.linalg.norm(tp[0] - tp[1]))
    fit_err, cen_err, sep_truth = map(np.array, (fit_err, cen_err, sep_truth))
    print(f"\ntwo-tracer blobs scored: {len(sep_truth)}; true separation median {np.median(sep_truth):.2f} px")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 9)):
        m = np.repeat((sep_truth >= lo) & (sep_truth < hi), 2)
        if m.sum():
            print(f"  separation {lo}-{hi} px ({m.sum()//2} blobs): blob centre error median {np.median(cen_err[m]):.2f}, "
                  f"fitted centre error median {np.median(fit_err[m]):.2f}, share within 0.5 px {np.mean(fit_err[m] < 0.5):.2f}")
    print(f"all: blob centre {np.median(cen_err):.2f} px, fitted {np.median(fit_err):.2f} px, within 0.5 px {np.mean(fit_err < 0.5):.2f}")


if __name__ == "__main__":
    main()
