"""Single-tracer particle image size per camera, from isolated blobs (stage D0 of the detection-hypotheses plan).

For each camera: detect blobs as the pipeline does (high-pass + ``target_recognition``), keep the isolated ones
(no other blob within ``--iso`` px), compute the grey-weighted second moments over the blob's pixel window, and
report their median and spread. Run on a scratch or read-only run directory (reads images only).

    uv run python scripts/psf_estimate.py RUN_DIR [--frames 1 5 9] [--iso 8]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def blob_moments(img: np.ndarray, x: float, y: float, nx: int, ny: int, thres: float):
    """Grey-weighted central second moments (sxx, sxy, syy) over the blob's bounding window."""
    h, w = img.shape
    r = max(nx, ny) // 2 + 2
    x0, x1 = max(int(round(x)) - r, 0), min(int(round(x)) + r + 1, w)
    y0, y1 = max(int(round(y)) - r, 0), min(int(round(y)) + r + 1, h)
    win = img[y0:y1, x0:x1].astype(float) - thres
    win[win < 0] = 0
    tot = win.sum()
    if tot <= 0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cx, cy = (win * xx).sum() / tot, (win * yy).sum() / tot
    return ((win * (xx - cx) ** 2).sum() / tot, (win * (xx - cx) * (yy - cy)).sum() / tot,
            (win * (yy - cy) ** 2).sum() / tot, tot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--frames", type=int, nargs="+", default=[1, 5, 9])
    ap.add_argument("--iso", type=float, default=8.0)
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    os.chdir(run)
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import (
        DEFAULT_HIGHPASS_FILTER_SIZE,
        py_start_proc_c,
        read_frame_images,
        simple_highpass,
    )
    from openptv2.segmentation import target_recognition

    pm = ParameterManager()
    pm.from_yaml(run / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    ncam = len(cals)
    base = pm.get_parameter("sequence")["base_name"]
    ptv = pm.get_parameter("ptv") or {}
    thres = float((pm.get_parameter("targ_rec") or {}).get("gvthres", [0])[0] if isinstance(
        (pm.get_parameter("targ_rec") or {}).get("gvthres", 0), (list, tuple)) else (pm.get_parameter("targ_rec") or {}).get("gvthres", 0))
    out = {c: [] for c in range(ncam)}
    nblob = {c: 0 for c in range(ncam)}
    for f in a.frames:
        raw = read_frame_images(pm, base, ncam, f)
        for c in range(ncam):
            hp = simple_highpass(raw[c], cpar, ptv.get("highpass_size", DEFAULT_HIGHPASS_FILTER_SIZE))
            ts = target_recognition(hp, tpar, c, cpar)
            det = np.array([[t.pos()[0], t.pos()[1], t.count_pixels()[1], t.count_pixels()[2],
                             t.count_pixels()[0], t.sum_grey_value()] for t in ts])
            if len(det) < 3:
                continue
            nblob[c] += len(det)
            d, _ = cKDTree(det[:, :2]).query(det[:, :2], k=2)
            for (x, y, nx, ny, n, sg), dn in zip(det, d[:, 1]):
                if dn < a.iso or n < 3:
                    continue
                m = blob_moments(hp, x, y, int(nx), int(ny), thres)
                if m is not None:
                    out[c].append((m[0], m[2], m[1], n, sg / n, sg))
    print(f"run {run}  gvthres {thres}  frames {a.frames}")
    print("cam  blobs  isolated   median sxx  syy   sxy    n_pix   mean_int   sd(sxx)/sxx  sd(n)/n")
    for c in range(ncam):
        v = np.array(out[c])
        if len(v) == 0:
            continue
        med = np.median(v, axis=0)
        print(f"{c+1:3d} {nblob[c]:6d} {len(v):9d}   {med[0]:7.2f} {med[1]:6.2f} {med[2]:6.2f} {med[3]:8.1f} {med[4]:9.1f}"
              f"   {v[:,0].std()/med[0]:9.2f}   {v[:,3].std()/med[3]:7.2f}")


if __name__ == "__main__":
    main()
