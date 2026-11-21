"""D1 step 3: decide which blobs hold more than one tracer (rendered truth, syn1).

Per blob: shape cues + the gain of a two-Gaussian fit over a one-Gaussian fit. Logistic regression trained on some
frames, scored on others (AUC, calibration, and the gate: what share of true two-tracer blobs is caught at what
false-alarm rate on single-tracer blobs).

    uv run python scripts/d1_classify.py --train 1 5 --test 9 13 --out SCRATCH.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SYN = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"
RUN = SYN / "syn1" / "test"
S2 = 0.4          # single-tracer image variance found in d1_split.py


def collect(frames, tr, pm, cpar, tpar, ptv, ncam):
    from openptv2.blob_hypotheses import blob_features, fit_one, fit_two
    from openptv2.gui.ptv import (
        DEFAULT_HIGHPASS_FILTER_SIZE,
        read_frame_images,
        simple_highpass,
    )
    from openptv2.segmentation import _vectorized_targ_rec

    base = pm.get_parameter("sequence")["base_name"]
    imx, imy = cpar.get_image_size()
    rows = []
    for f in frames:
        raw = read_frame_images(pm, base, ncam, f)
        m = tr["frame"] == f
        for c in range(ncam):
            hp = simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
            pix: list = []
            thres = float(tpar.get_grey_thresholds()[c])
            ts = _vectorized_targ_rec(hp, int(thres), int(tpar.get_max_discontinuity()), *tpar.get_pixel_count_bounds(),
                                      *tpar.get_xsize_bounds(), *tpar.get_ysize_bounds(), int(tpar.get_min_sum_grey()),
                                      1, imx - 1, 1, imy - 1, pixels_out=pix)
            det = np.array([[t.pos()[0], t.pos()[1]] for t in ts])
            tp = tr["pixel_xy_per_cam"][m][:, 2 * c: 2 * c + 2]
            tp = tp[~np.isnan(tp[:, 0])]
            d, j = cKDTree(det).query(tp, distance_upper_bound=3.0)
            cnt = np.bincount(j[np.isfinite(d)], minlength=len(det))
            for i in range(len(det)):
                ft = blob_features(pix[i], thres)
                if ft is None or cnt[i] == 0 or len(pix[i]) < 4:
                    continue
                o = fit_one(hp, pix[i], S2)
                t2 = fit_two(hp, pix[i], S2)
                sep = float(np.hypot(t2[0] - t2[3], t2[1] - t2[4]))
                a1, a2 = t2[2], t2[5]
                rows.append((f, c, cnt[i]) + ft + (o[3], t2[6], np.log((o[3] + 1) / (t2[6] + 1)), sep,
                                                   min(a1, a2) / max(a1, a2, 1e-9)))
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=None, help="report from a saved npz instead of re-detecting")
    ap.add_argument("--train", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--test", type=int, nargs="+", default=[9, 13])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.load:
        z = np.load(a.load, allow_pickle=True)
        report({"train": z["train"], "test": z["test"]}, list(z["names"]))
        return
    os.chdir(RUN)
    from openptv2.blob_hypotheses import FEATURES
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c

    pm = ParameterManager()
    pm.from_yaml(RUN / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    tr = np.load(SYN / "truth/truth.npz")
    ptv = pm.get_parameter("ptv") or {}
    names = ("frame", "cam", "truth_count") + FEATURES + ("cost1", "cost2", "log_gain", "fit_sep", "amp_ratio")
    A = {k: collect(fr, tr, pm, cpar, tpar, ptv, len(cals)) for k, fr in (("train", a.train), ("test", a.test))}
    if a.out:
        np.savez(a.out, train=A["train"], test=A["test"], names=names)
    print("blobs: train", len(A["train"]), "test", len(A["test"]))
    report(A, names)


def report(A, names):
    from d1_features import auc

    from openptv2.blob_hypotheses import Logistic

    use = ["n", "mean_int", "lam1", "lam2", "ecc", "n_max", "log_gain", "fit_sep", "amp_ratio"]
    idx = [names.index(u) for u in use]
    tc_i = names.index("truth_count")
    def XY(M):
        return M[:, idx], (M[:, tc_i] >= 2).astype(int)
    Xtr, ytr = XY(A["train"]); Xte, yte = XY(A["test"])
    print(f"test blobs: {len(yte)}, two-or-more share {yte.mean():.3f}")
    for k in ("lam1", "n", "log_gain", "fit_sep"):
        v = Xte[:, use.index(k)]
        print(f"  single cue {k:9s} test AUC {auc(v[yte == 1], v[yte == 0]):.3f}")
    clf = Logistic.fit(Xtr, ytr)
    p = clf.predict(Xte)
    print(f"logistic (all cues) test AUC {auc(p[yte == 1], p[yte == 0]):.3f}")
    print("coefficients:", {u: round(float(c), 2) for u, c in zip(use, clf.coef)})
    print("calibration (predicted p bin -> actual share two-or-more, n):")
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  {lo:.1f}-{min(hi,1):.1f}: predicted mean {p[m].mean():.2f}, actual {yte[m].mean():.2f}, n {m.sum()}")
    print("operating points (catch share of true two+ blobs vs false alarms among single blobs):")
    for thr in (0.2, 0.3, 0.5, 0.7):
        print(f"  p>{thr}: caught {np.mean(p[yte == 1] > thr):.2f}, false alarms {np.mean(p[yte == 0] > thr):.3f}")


if __name__ == "__main__":
    main()
