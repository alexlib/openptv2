"""Prototype alternative particle detectors vs. ``targ_rec``.

Branch: ``feat/detection-prototypes-dask-trackpy-skimage-proptv``.
Status: PROTOTYPE ONLY — not wired into any pipeline path.

All prototypes take a 2D uint8 image and return an ``(N, 8)`` float64 array
with openptv columns ``[pnr, x, y, n, nx, ny, sumg, tnr]`` so they compare
directly against :func:`openptv2.algorithms.segmentation.targ_rec`
(via :func:`baseline_targets`).

Prototypes:
- :func:`detect_dask_label` — dask-image ``label`` + ``ndmeasure``
  (center_of_mass / sum / area) + openptv size/sum filters.
- :func:`detect_trackpy` — ``trackpy.locate`` mapped onto openptv columns.
- :func:`detect_skimage` — ``skimage.feature.peak_local_max`` +
  ``skimage.measure.regionprops`` mapped onto openptv columns.
- :func:`detect_proptv` — port of Robin Barta's proPTV ``ImageProcessing``
  peak-search core (single-frame, no file I/O): dynamic-range norm,
  Gaussian blur, iterative ``peak_local_max`` + Gaussian subtraction.

Caveat (see bench script): none of these reproduce ``targ_rec``'s
scanline-discontinuity growth and peak-reunification logic, so "exactly
accurate" is not expected — the bench quantifies the gap.
"""

from __future__ import annotations

import numpy as np


def baseline_targets(img: np.ndarray, params: dict) -> np.ndarray:
    """Run the in-tree ``targ_rec`` and return ``(N, 8)`` rows."""
    from openptv2.algorithms.segmentation import targ_rec

    if np.asarray(img).dtype != np.uint8:
        raise TypeError("baseline needs uint8 (targ_rec wraps on cast)")
    targets = targ_rec(
        np.ascontiguousarray(img, dtype=np.uint8),
        int(params.get("gvthres", 9)),
        int(params.get("discont", 100)),
        int(params.get("nnmin", 4)),
        int(params.get("nnmax", 500)),
        int(params.get("nxmin", 2)),
        int(params.get("nxmax", 100)),
        int(params.get("nymin", 2)),
        int(params.get("nymax", 100)),
        int(params.get("sumg_min", 150)),
    )
    if not targets:
        return np.zeros((0, 8), dtype=np.float64)
    rows = np.array(
        [[t.pnr(), t.x(), t.y(), t.n, t.nx, t.ny, t.sumg, t.tnr()] for t in targets],
        dtype=np.float64,
    )
    return rows


def _to_rows(xs, ys, ns, nxs, nys, sumgs) -> np.ndarray:
    n = len(xs)
    rows = np.zeros((n, 8), dtype=np.float64)
    for k in range(n):
        rows[k] = [k, float(xs[k]), float(ys[k]), int(ns[k]), int(nxs[k]), int(nys[k]), int(sumgs[k]), -1]
    return rows


def detect_dask_label(img: np.ndarray, params: dict) -> np.ndarray:
    """dask-image ``label`` + ``ndmeasure`` prototype.

    Graph: ``from_array(chunks=whole)`` -> ``> gvthres`` -> ``label`` ->
    ``center_of_mass`` / ``sum`` / ``area`` -> compute once -> numpy filter
    on ``nn/nx/ny/sumg``. One ``compute()`` total; per-frame chunking means
    no halo exchange is needed when this is later mapped over ``(N,H,W)``.
    """
    import dask.array as da
    import dask_image.ndmeasure

    img = np.ascontiguousarray(img, dtype=np.uint8)
    gv = int(params.get("gvthres", 9))
    d = da.from_array(img, chunks=img.shape)
    mask = d > gv
    label_image, _num = dask_image.ndmeasure.label(mask)
    nlab = int(np.asarray(_num.compute()).ravel()[0]) if hasattr(_num, "compute") else int(np.asarray(_num).ravel()[0])
    if nlab == 0:
        return np.zeros((0, 8), dtype=np.float64)
    idx = list(range(1, nlab + 1))
    # ndmeasure reductions return lazy arrays; compute together below.
    com = dask_image.ndmeasure.center_of_mass(np.asarray(img), label_image, index=idx)
    sums = dask_image.ndmeasure.sum(np.asarray(img), label_image, index=idx)
    areas = dask_image.ndmeasure.area(mask, label_image, index=idx)
    com_v, sums_v, areas_v, lab_v = tuple(
        np.asarray(x) for x in (com.compute(), sums.compute(), areas.compute(), label_image.compute())
    )
    # com_v rows: (y, x) per label.
    ys, xs = com_v[:, 0], com_v[:, 1]
    sumgs, ns = np.asarray(sums_v).ravel(), np.asarray(areas_v).ravel().astype(int)
    # bbox extents per label for nx/ny (numpy, single pass over small label map).
    nx = np.zeros_like(ns)
    ny = np.zeros_like(ns)
    for k in range(len(ns)):
        r, c = np.nonzero(lab_v == (k + 1))
        if r.size:
            nx[k] = int(c.max() - c.min() + 1)
            ny[k] = int(r.max() - r.min() + 1)
    keep = (
        (ns >= int(params.get("nnmin", 1)))
        & (ns <= int(params.get("nnmax", 10**9)))
        & (nx >= int(params.get("nxmin", 1)))
        & (nx <= int(params.get("nxmax", 10**9)))
        & (ny >= int(params.get("nymin", 1)))
        & (ny <= int(params.get("nymax", 10**9)))
        & (sumgs >= int(params.get("sumg_min", 0)))
    )
    return _to_rows(xs[keep], ys[keep], ns[keep], nx[keep], ny[keep], sumgs[keep])


def detect_trackpy(img: np.ndarray, params: dict) -> np.ndarray:
    """``trackpy.locate`` prototype mapped onto openptv columns."""
    import trackpy as tp

    img = np.ascontiguousarray(img)
    diameter = int(params.get("diameter", 7))
    if diameter % 2 == 0:
        diameter += 1
    df = tp.locate(
        img,
        diameter=diameter,
        minmass=int(params.get("sumg_min", 150)),
        separation=int(params.get("separation", 3)),
        threshold=int(params.get("gvthres", 9)),
        preprocess=False,
    )
    if df is None or len(df) == 0:
        return np.zeros((0, 8), dtype=np.float64)
    xs = df["x"].to_numpy(float)
    ys = df["y"].to_numpy(float)
    sumgs = df["mass"].to_numpy(float) if "mass" in df else np.zeros_like(xs)
    size = df["size"].to_numpy(float) if "size" in df else np.ones_like(xs)
    ns = np.maximum(1, np.rint(np.pi * size**2)).astype(int)
    nxy = np.maximum(1, np.rint(2 * size).astype(int))
    return _to_rows(xs, ys, ns, nxy, nxy, sumgs)


def detect_skimage(img: np.ndarray, params: dict) -> np.ndarray:
    """``peak_local_max`` + ``regionprops`` prototype."""
    from skimage.feature import peak_local_max
    from skimage.measure import label, regionprops

    img = np.ascontiguousarray(img, dtype=np.uint8)
    gv = int(params.get("gvthres", 9))
    lab = label(img > gv, connectivity=2)
    props = regionprops(lab, intensity_image=img)
    # peaks only used to break ties toward local maxima (kept for parity
    # with the proPTV/trackpy peak-first family); the region list drives output.
    _peaks = peak_local_max(
        img,
        min_distance=int(params.get("min_distance", 2)),
        threshold_abs=gv,
        exclude_border=False,
    )
    xs, ys, ns, nxs, nys, sumgs = [], [], [], [], [], []
    for p in props:
        n = p.area
        minr, minc, maxr, maxc = p.bbox
        nx_, ny_ = maxc - minc, maxr - minr
        s = float(getattr(p, "intensity_sum", p.intensity_mean * n))
        if not (
            int(params.get("nnmin", 1)) <= n <= int(params.get("nnmax", 10**9))
            and int(params.get("nxmin", 1)) <= nx_ <= int(params.get("nxmax", 10**9))
            and int(params.get("nymin", 1)) <= ny_ <= int(params.get("nymax", 10**9))
            and s >= int(params.get("sumg_min", 0))
        ):
            continue
        ys.append(p.centroid[0])
        xs.append(p.centroid[1])
        ns.append(n)
        nxs.append(nx_)
        nys.append(ny_)
        sumgs.append(s)
    return _to_rows(np.array(xs), np.array(ys), np.array(ns), np.array(nxs), np.array(nys), np.array(sumgs))


def detect_proptv(
    img: np.ndarray,
    params: dict | None = None,
    min_img: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Port of Robin Barta's proPTV ``ImageProcessing`` core (single frame).

    Steps kept: optional temporal-min subtraction, masking, dynamic-range
    norm ``250*(img-mean)/std`` + Gaussian + ``20*sqrt``, iterative
    ``peak_local_max`` (``runs_search``) with 3-px parabolic subpixel
    refinement and Gaussian-peak subtraction, unique-ify. File I/O, debug
    plots, and proc-image writing from the original are dropped; pass
    ``min_img``/``mask`` arrays directly instead of path templates.
    Returns ``(N, 8)`` with ``n/nx/ny = particleSize`` box and ``sumg`` sampled
    from the processed image (informational only).
    """
    import cv2
    from skimage.feature import peak_local_max

    p = dict(
        window=3,
        weight_min=1.2,
        threshold=100,
        blur=False,
        Gauskernel=[3, 3],
        Gauskernelstd=1.0,
        runs_search=3,
        std=0.7,
        Imean=4000,
        maxParticle=25000,
        particleSize=2,
    )
    if params:
        p.update(params)

    img_origin = np.ascontiguousarray(img)
    work = img_origin.astype(np.float64).copy()
    if min_img is not None:
        work = work - float(p["weight_min"]) * np.ascontiguousarray(min_img, dtype=np.float64)
    if mask is not None:
        work[np.ascontiguousarray(mask) == 0] = 1
    work[work < float(p["threshold"])] = 1
    std = float(np.std(work))
    work = 250.0 * (work - np.mean(work)) / (std if std > 0 else 1.0)
    work = cv2.GaussianBlur(work, tuple(int(v) for v in p["Gauskernel"]), float(p["Gauskernelstd"]))
    work[work < 1] = 0
    work = 20.0 * np.sqrt(np.maximum(work, 0))
    if bool(p["blur"]):
        work = cv2.GaussianBlur(work, tuple(int(v) for v in p["Gauskernel"]), float(p["Gauskernelstd"]))
    work = np.rint(work)

    img_peak = work.astype(float)
    h, w = img_peak.shape[:2]
    y_coords, x_coords = np.mgrid[:h, :w]
    ps = int(p["particleSize"])
    final: list[tuple[float, float]] = []
    for _ in range(int(p["runs_search"])):
        peaks = peak_local_max(
            img_peak,
            min_distance=int(ps),
            num_peaks=int(p["maxParticle"]) - len(final),
            exclude_border=False,
        )
        if len(peaks) == 0:
            break
        cx_list, cy_list = [], []
        sub = np.zeros_like(img_peak)
        for y, x in zip(peaks[:, 0], peaks[:, 1]):
            xs3 = np.array([x, x + 1, x - 1], dtype=int)
            ys3 = np.array([y, y + 1, y - 1], dtype=int)
            xs3 = np.clip(xs3, 0, w - 1)
            ys3 = np.clip(ys3, 0, h - 1)
            vx, vy = img_peak[y, xs3], img_peak[ys3, x]
            sx, sy = float(vx.sum()), float(vy.sum())
            mx = float(np.sum(xs3 * vx) / sx) if sx > 0 else float(x)
            my = float(np.sum(ys3 * vy) / sy) if sy > 0 else float(y)
            cx_list.append(mx)
            cy_list.append(my)
            x0, x1 = int(round(mx)) - ps, int(round(mx)) + ps + 1
            y0, y1 = int(round(my)) - ps, int(round(my)) + ps + 1
            x0c, x1c, y0c, y1c = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
            if x1c > x0c and y1c > y0c:
                g = float(p["Imean"]) * np.exp(
                    -0.5
                    * (
                        ((x_coords[y0c:y1c, x0c:x1c] - mx) / float(p["std"])) ** 2
                        + ((y_coords[y0c:y1c, x0c:x1c] - my) / float(p["std"])) ** 2
                    )
                )
                sub[y0c:y1c, x0c:x1c] += g
        img_peak = img_peak - sub
        img_peak[img_peak < float(p["threshold"])] = 0
        final += list(zip(cx_list, cy_list))
        if len(final) >= int(p["maxParticle"]):
            break
    if not final:
        return np.zeros((0, 8), dtype=np.float64)
    arr = np.unique(np.asarray(final, dtype=float), axis=0)
    xs, ys = arr[:, 0], arr[:, 1]
    box = 2 * ps + 1
    sumgs = []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < h and 0 <= xi < w:
            sumgs.append(float(work[yi, xi]))
        else:
            sumgs.append(0.0)
    return _to_rows(xs, ys, np.full(len(xs), box * box), np.full(len(xs), box), np.full(len(xs), box), np.array(sumgs))
