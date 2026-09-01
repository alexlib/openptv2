"""Tunable plate-dot detection — wrapper around :mod:`openptv2.segmentation`.

Covers small-dot Illmenau failure mode: ``cv2.findCirclesGrid`` is kept only as
an opt-in fallback, the default path is ``target_recognition`` (already proven
for tracer ``sumg``/``nn`` bands) with a separate ``detect_plate`` YAML block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from openptv2.algorithms.parameters import ControlPar, TargetPar
from openptv2.algorithms.tracking_frame_buf import Target


@dataclass
class PlateDetectionResult:
    targets: list[Target]
    coded_mask: np.ndarray  # bool array, True = white-in-black

    @property
    def centroids(self) -> np.ndarray:
        return np.array([[t.x, t.y] for t in self.targets], dtype=float)

    @property
    def types(self) -> np.ndarray:
        # 0=black, 1=coded
        return self.coded_mask.astype(int)


def plate_tpar_from_yaml(yaml_path: str | Path, key: str = "detect_plate") -> TargetPar:
    """Load a ``TargetPar`` from ``key`` in a dataset YAML or a bare dict.

    Falls back to a sensible default for Illmenau 16-bit plates when the key
    is absent.
    """
    p = Path(yaml_path)
    if p.is_file():
        raw = yaml.safe_load(p.read_text()) or {}
        cfg = raw.get(key) or raw.get("targ_rec") or {}
    else:
        cfg = {}

    def _int(k, alt_k, default):
        v = cfg.get(k, cfg.get(alt_k, default))
        try:
            return int(v)
        except Exception:
            return default

    gv = cfg.get("gvthres")
    if gv is None:
        gv = [cfg.get(f"gvth_{i}", cfg.get(f"gvthres_{i}", 20)) for i in range(1, 5)]
    elif isinstance(gv, (int, float)):
        gv = [int(gv)] * 4

    return TargetPar(
        discont=_int("discont", "tol_dis", 80),
        nnmin=_int("nnmin", "min_npix", 10),
        nnmax=_int("nnmax", "max_npix", 5000),
        nxmin=_int("nxmin", "min_npix_x", 8),
        nxmax=_int("nxmax", "max_npix_x", 80),
        nymin=_int("nymin", "min_npix_y", 8),
        nymax=_int("nymax", "max_npix_y", 80),
        sumg_min=_int("sumg_min", "sum_grey", 5000),
        cr_sz=_int("cr_sz", "size_cross", 3),
        gvthres=list(map(int, gv[:4])),
    )


def _classify_coded(
    image: np.ndarray, targets: list[Target], thr: float = 40.0
) -> np.ndarray:
    """White-in-black (bright centre + dark ring) classifier.

    For each target, samples a centre mean ``I_c`` and an annulus
    ``r≈5px`` mean ``I_r`` on the *raw* (pre-hp) image.
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2).astype(image.dtype)
    mask = np.zeros(len(targets), dtype=bool)
    if not targets:
        return mask

    h, w = image.shape
    contrasts = []
    for t in targets:
        x, y = int(round(t.x)), int(round(t.y))
        if not (6 <= x < w - 6 and 6 <= y < h - 6):
            contrasts.append(-1.0)
            continue
        centre = image[y - 2 : y + 3, x - 2 : x + 3].astype(float)
        Ic = float(centre.mean())
        # annulus at radius 5-6
        coords = [
            (y - 6, x),
            (y + 6, x),
            (y, x - 6),
            (y, x + 6),
            (y - 5, x - 5),
            (y - 5, x + 5),
            (y + 5, x - 5),
            (y + 5, x + 5),
            (y - 4, x - 4),
            (y - 4, x + 4),
            (y + 4, x - 4),
            (y + 4, x + 4),
        ]
        ring_vals = [
            float(image[ry, rx]) for ry, rx in coords if 0 <= ry < h and 0 <= rx < w
        ]
        Ir = float(np.mean(ring_vals)) if ring_vals else Ic
        contrasts.append(Ic - Ir)

    contrasts_arr = np.array(contrasts, dtype=float)
    # If 3 distinct peaks above threshold exist, pick them
    if (contrasts_arr > thr).sum() == 3:
        mask[contrasts_arr > thr] = True
    elif len(contrasts_arr) >= 3:
        top3 = np.argsort(contrasts_arr)[-3:]
        # Check that top3 contrast is distinctly above background
        if contrasts_arr[top3[0]] > (thr * 0.5):
            mask[top3] = True
    return mask


def find_plate_roi(
    work8: np.ndarray,
    sigma: float = 25.0,
    pad: float = 0.07,
) -> tuple[int, int, int, int]:
    """Find bounding ROI of the calibration plate within the full image."""
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import label as nd_label

    imy, imx = work8.shape
    blurred = gaussian_filter(work8.astype(float), sigma=sigma)
    hist, _ = np.histogram(blurred, bins=256, range=(0, 255))
    total = blurred.size
    sum_tot = float((hist * np.arange(256)).sum())
    sumB = 0.0
    wB = 0.0
    max_var = 0.0
    thresh = 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_tot - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > max_var:
            max_var = var
            thresh = t

    bw = (blurred > thresh).astype(np.uint8) * 255
    labeled, n = nd_label(bw)
    if n == 0:
        return 1, imx - 1, 1, imy - 1

    areas = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        areas.append((len(xs), (x0, y0, x1 - x0 + 1, y1 - y0 + 1)))

    if not areas:
        return 1, imx - 1, 1, imy - 1

    areas.sort(reverse=True)
    _, (x, y, w, h) = areas[0]
    x0 = int(max(1, x - w * pad))
    y0 = int(max(1, y - h * pad))
    x1 = int(min(imx - 1, x + w + w * pad))
    y1 = int(min(imy - 1, y + h + h * pad))
    return x0, x1, y0, y1


def detect_plate_targets(
    image: np.ndarray,
    tpar: TargetPar,
    cpar: ControlPar,
    cam: int = 0,
    *,
    coded_thr: float = 40.0,
    raw_for_coded: np.ndarray | None = None,
    use_roi: bool = True,
    scaling: dict | None = None,
) -> PlateDetectionResult:
    """Detect plate dots and classify coded ones.

    ``image`` should already be the raw 8/16-bit frame (we do hp inside).
    ``tpar`` is the plate-specific ``TargetPar``.  Returns centroids + coded
    mask.  ``raw_for_coded`` can override the image used for the ring test
    (e.g. keep raw before hp).
    """
    from openptv2.image_processing import preprocess_image
    from openptv2.segmentation import target_recognition

    # Normalize 16-bit to 8-bit for legacy pipeline (target_recognition expects uint8-ish)
    work = image
    if work.ndim == 3:
        work = np.mean(work, axis=2).astype(work.dtype)

    # Grey scaling decides what every threshold below MEANS.  The default stays
    # the per-image percentile stretch this function has always used, so nothing
    # changes unless a caller asks; pass `scaling` (from
    # openptv2.image_scaling.from_parameters) to make it explicit and fixed.
    # See docs/plans/2026-08-31-16bit-image-handling.md.
    from openptv2.image_scaling import to_uint8

    _rule = scaling or {"mode": "stretch"}
    work8 = to_uint8(
        work, _rule.get("mode", "stretch"), lo=_rule.get("lo"), hi=_rule.get("hi")
    )

    # ROI detection
    subrange_x = None
    subrange_y = None
    if use_roi:
        xmin, xmax, ymin, ymax = find_plate_roi(work8)
        subrange_x = (xmin, xmax)
        subrange_y = (ymin, ymax)
        roi_mean = float(np.mean(work8[ymin:ymax, xmin:xmax]))
    else:
        roi_mean = float(np.mean(work8))

    # Dark dots on bright background: invert for target_recognition
    is_negative = getattr(cpar, "negative", False) or (roi_mean > 80.0)
    work_for_rec = (255 - work8) if is_negative else work8

    # hp_flag governs high-pass; plate images benefit from it
    hp = preprocess_image(work_for_rec, cpar.hp_flag or 1, cpar, 25)

    # target_recognition returns list[Target] with pnr, x, y, sumg, etc.
    raw_targets = target_recognition(
        hp,
        tpar,
        cam,
        cpar,
        subrange_x=subrange_x,
        subrange_y=subrange_y,
    )

    # Filter the single dummy sentinel (pnr=1,x=y=1,n=1) returned when
    # target_recognition finds nothing.  The previous filter dropped any
    # real target with pnr==1 when ≥2 targets were found.
    def _is_dummy(t) -> bool:
        return (
            getattr(t, "pnr", -999) == 1
            and getattr(t, "n", 0) == 1
            and float(getattr(t, "x", 0)) == 1.0
            and float(getattr(t, "y", 0)) == 1.0
        )

    if len(raw_targets) == 1 and _is_dummy(raw_targets[0]):
        targets = []
    else:
        targets = [t for t in raw_targets if not _is_dummy(t)]

    coded = _classify_coded(
        raw_for_coded if raw_for_coded is not None else work8, targets, thr=coded_thr
    )
    return PlateDetectionResult(targets=targets, coded_mask=coded)
