"""Calibration source descriptors and registry.

openptv2 calibrates a camera from a `CalibrationPointSet` -- a plain set of
(lab XYZ, image xy) correspondences per camera, with an optional seed pose --
via the source-agnostic bundle adjustment in `autocalibration.py`
(`external_calibration`/`full_calibration`, wrapped by
`autocalibration._refine_and_select`). Any calibration *source* (a 3D
calibration object with known point positions, a checkerboard, a multiplane
target, or another package's calibration points) only needs to produce that
shape; it never needs to touch `.ori`/`.addpar` writing itself.

`CALIBRATION_SOURCE_REGISTRY` documents each available source the way
`tracking_registry.TRACKER_REGISTRY` documents each tracker: self-describing
metadata for a CLI table or GUI selector, not a dynamic plugin dispatcher
(there is only one source today; a name -> module dispatch table can be
added once there is more than one implementation to dispatch between).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationPointSet:
    """One camera's calibration correspondences, ready for bundle adjustment.

    `ref_pts` and `img_pts` are exactly what `external_calibration`/
    `full_calibration` (`algorithms/orientation.py`) already accept: plain
    arrays, no assumption about where the correspondences came from.
    """

    ref_pts: np.ndarray  # (n, 3) known lab positions
    img_pts: np.ndarray  # (n, 2) matched pixel observations
    seed: object | None = None  # optional Calibration to start refinement from


@dataclass(frozen=True)
class CalibrationSourceInfo:
    """Self-describing metadata for a calibration source."""

    name: str
    display_name: str
    short_description: str  # one-liner for tables

    algorithm_summary: str
    requires: str  # plain-English description of the physical inputs needed
    produces_seed: bool  # can this source bootstrap a pose without a manual click?

    best_for: str = ""
    avoid_when: str = ""
    citation: str = ""


CALIBRATION_SOURCE_REGISTRY: dict[str, CalibrationSourceInfo] = {}

CALIBRATION_OBJECT_INFO = CalibrationSourceInfo(
    name="calibration_object",
    display_name="3D calibration object (calblock)",
    short_description="Surveyed 3D target + 4-point manual seed (today's default)",
    algorithm_summary=(
        "A rigid body with known point positions (calblock.txt) is photographed once "
        "per camera. A 4-point manual click (or an existing .ori) seeds "
        "external_calibration, sortgrid matches the rest of the body's points to "
        "detected targets, and full_calibration bundle-adjusts exterior + interior + "
        "distortion, trying progressively richer distortion flag-sets and keeping the "
        "lowest reprojection RMS."
    ),
    requires=(
        "A calibration body with surveyed 3D point positions, one image per camera, "
        "and either an existing .ori/.addpar or 4 manually clicked seed points."
    ),
    produces_seed=False,
    best_for="Any setup with a physical calibration target available -- the reference path.",
    avoid_when="No physical calibration body exists, or points can't be surveyed.",
)
CALIBRATION_SOURCE_REGISTRY[CALIBRATION_OBJECT_INFO.name] = CALIBRATION_OBJECT_INFO

OPENCV_MODEL_INFO = CalibrationSourceInfo(
    name="opencv_model",
    display_name="OpenCV / COLMAP / Metashape pinhole+Brown",
    short_description="Existing pinhole+Brown (K,dist,rvec,tvec) → openPTV seed",
    algorithm_summary=(
        "Algebraic conversion of an existing OpenCV/COLMAP/Metashape "
        "calibration (K, dist, rvec, tvec) via calibration_from_opencv "
        "(verified to 1e-12 px when xh=yh=0; otherwise a seed for one "
        "full_calibration pass to absorb the distortion-centre offset). "
        "First three pinhole models are the same up to the p-swap, half-pixel "
        "origin, R vs Rᵀ and mm scaling — one converter with thin front-ends."
    ),
    requires="Per-camera OpenCV intrinsics (K, dist) + extrinsics (rvec, tvec), image size and pixel pitch.",
    produces_seed=True,
    best_for="Ilmenau barrel / any dataset arriving from OpenCV, COLMAP or Metashape with an existing calibration.",
    avoid_when="Foreign model used rational/thin-prism/tilt terms (k4.., s1.., taux…) — use door C (resample) instead.",
)
CALIBRATION_SOURCE_REGISTRY[OPENCV_MODEL_INFO.name] = OPENCV_MODEL_INFO

POINTS_FILE_INFO = CalibrationSourceInfo(
    name="points_file",
    display_name="Universal points file (x y X Y Z)",
    short_description="5-col x y X Y Z per camera (already matched) → bundle adjust",
    algorithm_summary=(
        "One reader for proPTV markers_cN.txt, MyPTV camN_cal_points, "
        "Multiview-Calibration cN_xyXYZ.txt and DaVis plate exports — all the "
        "same five columns, differing only in separator/header.  Needs a seed "
        "pose from elsewhere; then presorted full_calibration with no sortgrid."
    ),
    requires="Per-camera 5-col file plus a seed pose (e.g. from opencv_model or rig_lookat).",
    produces_seed=False,
    best_for="proPTV / MyPTV / DaVis / Multiview-Calibration exports.",
    avoid_when="No point correspondences available — use a model or image door instead.",
)
CALIBRATION_SOURCE_REGISTRY[POINTS_FILE_INFO.name] = POINTS_FILE_INFO

RIG_LOOKAT_INFO = CalibrationSourceInfo(
    name="rig_lookat",
    display_name="Rig look-at + lens focal length (rig.yaml)",
    short_description="Tape-measure rig: position + target + focal_mm → .ori seed",
    algorithm_summary=(
        "Human-friendly rig description: each camera's position, what it looks "
        "at, and the lens focal length from the barrel.  Builds dm via "
        "dm_from_lookat and cc=focal_mm; position error ~ standoff × relative "
        "focal error (M2)."
    ),
    requires="rig.yaml with focal_mm per camera (no default — M5).",
    produces_seed=True,
    best_for="First-time setup with a tape measure and lens marking.",
    avoid_when="You need camera positions better than standoff × lens-marking error — use DLT instead.",
)
CALIBRATION_SOURCE_REGISTRY[RIG_LOOKAT_INFO.name] = RIG_LOOKAT_INFO

DLT_RESECTION_INFO = CalibrationSourceInfo(
    name="dlt_resection",
    display_name="DLT resection (≥6 known points)",
    short_description="DLT solves pose + cc from ≥6 non-coplanar points",
    algorithm_summary=(
        "Classic Abdel-Aziz & Karara DLT on known 3D↔2D correspondences.  The "
        "only tier that solves for cc without a prior."
    ),
    requires="≥6 non-coplanar identified correspondences per camera.",
    produces_seed=True,
    best_for="Any dataset with a points file and no prior calibration.",
    avoid_when="Points are coplanar or <6 — DLT degenerate.",
)
CALIBRATION_SOURCE_REGISTRY[DLT_RESECTION_INFO.name] = DLT_RESECTION_INFO


def get_source_info(name: str) -> CalibrationSourceInfo:
    """Look up a calibration source's metadata by name."""
    try:
        return CALIBRATION_SOURCE_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(CALIBRATION_SOURCE_REGISTRY))
        raise KeyError(f"Unknown calibration source {name!r}. Available: {available}") from None


def list_sources() -> list[str]:
    """Names of all registered calibration sources."""
    return sorted(CALIBRATION_SOURCE_REGISTRY)


__all__ = [
    "CALIBRATION_SOURCE_REGISTRY",
    "CalibrationPointSet",
    "CalibrationSourceInfo",
    "get_source_info",
    "list_sources",
]
