#!/usr/bin/env python
"""Bench prototype detectors vs. targ_rec on test_cavity cam1.10000.

Usage: uv run python scripts/bench_detection_prototypes.py [--frames N]

Reports per method: wall time, count, median nearest-neighbour distance to
baseline targets (px), recall @1.5px. "Exactly accurate" = recall 1.0 AND
sub-pixel median distance; the script prints the verdict instead of assuming it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _load_params() -> dict:
    from openptv2.algorithms.parameters import TargetPar

    tpar = TargetPar.from_file("test_data/test_cavity/parameters/targ_rec.par")
    return {
        "gvthres": int(tpar.get_grey_thresholds()[0]),
        "discont": int(tpar.get_max_discontinuity()),
        "nnmin": int(tpar.get_pixel_count_bounds()[0]),
        "nnmax": int(tpar.get_pixel_count_bounds()[1]),
        "nxmin": int(tpar.get_xsize_bounds()[0]),
        "nxmax": int(tpar.get_xsize_bounds()[1]),
        "nymin": int(tpar.get_ysize_bounds()[0]),
        "nymax": int(tpar.get_ysize_bounds()[1]),
        "sumg_min": int(tpar.get_min_sum_grey()),
    }


def _load_images(n: int) -> list[np.ndarray]:
    from skimage.io import imread

    return [
        np.ascontiguousarray(imread(f"test_data/test_cavity/img/cam1.{10000 + k}"), dtype=np.uint8)
        for k in range(n)
    ]


def _match_stats(base: np.ndarray, cand: np.ndarray, tol: float = 1.5) -> tuple[float, float]:
    """Median NN distance cand->base and recall of base targets within tol."""
    if len(base) == 0 or len(cand) == 0:
        return float("nan"), 0.0
    from scipy.spatial import cKDTree

    tree = cKDTree(base[:, 1:3])
    dist, _ = tree.query(cand[:, 1:3], k=1)
    tree2 = cKDTree(cand[:, 1:3])
    dist2, _ = tree2.query(base[:, 1:3], k=1)
    return float(np.median(dist)), float(np.mean(dist2 <= tol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()

    from openptv2.algorithms import detection_prototypes as dp

    params = _load_params()
    print(f"targ_rec params: {params}")
    images = _load_images(args.frames)
    print(f"images: {len(images)} x {images[0].shape} {images[0].dtype}")

    methods = {
        "baseline targ_rec": lambda im: dp.baseline_targets(im, params),
        "dask-image label+measure": lambda im: dp.detect_dask_label(im, params),
        "trackpy.locate": lambda im: dp.detect_trackpy(im, params),
        "skimage peak+regionprops": lambda im: dp.detect_skimage(im, params),
        "proPTV port": lambda im: dp.detect_proptv(im),
    }
    results: dict[str, list[np.ndarray]] = {}
    timings: dict[str, float] = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        outs = [fn(im) for im in images]
        dt = time.perf_counter() - t0
        results[name] = outs
        timings[name] = dt
        print(f"{name}: {sum(len(o) for o in outs)} targets total in {dt:.2f}s")

    print("\nvs baseline (per-frame mean, tol=1.5px):")
    base_all = results["baseline targ_rec"]
    for name in list(methods)[1:]:
        meds, recs, counts = [], [], []
        for b, c in zip(base_all, results[name]):
            med, rec = _match_stats(b, c)
            meds.append(med)
            recs.append(rec)
            counts.append(len(c) - len(b))
        print(
            f"  {name}: dT={timings[name]/timings['baseline targ_rec']:.2f}x baseline, "
            f"count_delta={np.mean(counts):+.1f}/frame, "
            f"medianNN={np.nanmean(meds):.3f}px, recall@1.5px={np.mean(recs):.3f}"
        )
    print("\nVerdict: 'faster and exactly accurate' requires dT<1.0 AND recall==1.0 above.")


if __name__ == "__main__":
    main()
