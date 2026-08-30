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

    def _int(k, default):
        v = cfg.get(k, default)
        try:
            return int(v)
        except Exception:
            return default

    gv = cfg.get("gvthres", [30, 30, 30, 30])
    if isinstance(gv, (int, float)):
        gv = [int(gv)] * 4
    return TargetPar(
        discont=_int("discont", cfg.get("tol_dis", 80)),
        nnmin=_int("nnmin", cfg.get("min_npix", 3)),
        nnmax=_int("nnmax", cfg.get("max_npix", 500)),
        nxmin=_int("nxmin", cfg.get("min_npix_x", 2)),
        nxmax=_int("nxmax", cfg.get("max_npix_x", 40)),
        nymin=_int("nymin", cfg.get("min_npix_y", 2)),
        nymax=_int("nymax", cfg.get("max_npix_y", 40)),
        sumg_min=_int("sumg_min", cfg.get("sum_grey", 200)),
        cr_sz=_int("cr_sz", cfg.get("size_cross", 3)),
        gvthres=list(map(int, gv[:4])),
    )


def _classify_coded(image: np.ndarray, targets: list[Target], thr: float = 40.0) -> np.ndarray:
    """White-in-black (bright centre + dark ring) classifier.

    For each target, samples a 5×5 centre mean ``I_c`` and an annulus
    ``r±2px`` mean ``I_r`` on the *raw* (pre-hp) image.  Coded ⇔
    ``I_c - I_r > thr`` and ``I_c`` in the upper quartile. Threshold is a
    YAML param ``coded_thr``.
    """
    if image.ndim == 3:
        # Use first channel or mean — plate TIFFs are mono; keep it simple
        image = np.mean(image, axis=2).astype(image.dtype)
    mask = np.zeros(len(targets), dtype=bool)
    h, w = image.shape
    # Robust scale for coded decision — upper quartile
    vals = []
    for t in targets:
        x, y = int(round(t.x)), int(round(t.y))
        if 2 <= x < w - 2 and 2 <= y < h - 2:
            patch = image[max(0, y - 2): y + 3, max(0, x - 2): x + 3]
            vals.append(float(patch.mean()))
    hi = float(np.percentile(vals, 75)) if vals else float(image.max()) * 0.7

    for i, t in enumerate(targets):
        x, y = int(round(t.x)), int(round(t.y))
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            continue
        centre = image[y - 2:y + 3, x - 2:x + 3].astype(float)
        Ic = float(centre.mean())
        # annulus: 8 neighbours at r≈3
        coords = [(y-3, x), (y+3, x), (y, x-3), (y, x+3),
                  (y-2, x-2), (y-2, x+2), (y+2, x-2), (y+2, x+2)]
        ring_vals = []
        for ry, rx in coords:
            if 0 <= ry < h and 0 <= rx < w:
                ring_vals.append(float(image[ry, rx]))
        Ir = float(np.mean(ring_vals)) if ring_vals else Ic
        if Ic > hi and (Ic - Ir) > thr:
            mask[i] = True
    return mask


def detect_plate_targets(
    image: np.ndarray,
    tpar: TargetPar,
    cpar: ControlPar,
    cam: int = 0,
    *,
    coded_thr: float = 40.0,
    raw_for_coded: np.ndarray | None = None,
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
    if work.dtype == np.uint16:
        # Scale via percentile to keep dots in range, similar to GUI autoscale
        lo, hi = float(np.percentile(work, 1)), float(np.percentile(work, 99.5))
        if hi <= lo:
            hi = float(work.max()) or 1.0
        work8 = np.clip((work.astype(float) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    else:
        work8 = work.astype(np.uint8) if work.dtype != np.uint8 else work

    # hp_flag governs high-pass; plate images benefit from it
    hp = preprocess_image(work8, cpar.hp_flag or 1, cpar, 25)
    # target_recognition returns list[Target] with pnr, x, y, sumg, etc.
    raw_targets = target_recognition(hp, tpar, cam, cpar)
    # Filter sentinel
    targets = [t for t in raw_targets if getattr(t, "pnr", -999) != 1 or len(raw_targets) == 1]
    # Fallback: if sentinel slipped through (single dummy), drop it
    if len(targets) == 1 and getattr(targets[0], "n", 0) == 1 and targets[0].x == 1.0 and targets[0].y == 1.0:
        targets = []

    coded = _classify_coded(raw_for_coded if raw_for_coded is not None else work, targets, thr=coded_thr)
    return PlateDetectionResult(targets=targets, coded_mask=coded)
