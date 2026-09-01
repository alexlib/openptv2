"""Warn the user before tracking runs when calibration/reconstruction noise
is too large relative to the true flow for individual trajectories to be
trustworthy -- rather than let them discover it from a plot of short,
jump-riddled trajectories after the fact.

This is a real, measured failure mode, not a hypothetical: on test_cavity
(docs/plans/two-subrig-calibration.md), the true flow is ~0.2-0.3mm/frame
while z-reconstruction noise is ~0.3-0.6mm (from ~7x worse depth than
in-plane pixel sensitivity, a property of the camera rig geometry). No
tracker parameter or algorithm choice fixes that; global Hungarian
assignment (nearest_hungarian_3d) was tried on this dataset and did not help,
because the limitation is in the input data's information content, not in how
the tracker decides between candidates. The only way to raise the ceiling is more/better-conditioned
calibration data or different rig geometry (see the plan doc); the only
thing software can do about an *existing* dataset is say so up front.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


def measure_motion_scale(pos_a, pos_b) -> Optional[tuple[float, float]]:
    """(displacement, spacing) in the same units as pos_a/pos_b, or None if
    there isn't enough data to estimate them.

    displacement: 10th percentile (not the median) of each frame-a point's
    distance to its nearest frame-b neighbor. Real correspondence data
    includes spurious "ghost" matches (2-camera epipolar accidents with no
    real particle behind them -- measured up to 64% of 2-camera
    correspondences on test_cavity) that have no genuine match next frame
    and so contribute large, noisy nearest-neighbor distances; genuine
    matches cluster tightly near the true (small) displacement. The median
    is dominated by ghost noise; the 10th percentile isolates the tight
    genuine-match cluster while still separating cleanly from a genuinely
    fast flow with no ghost contamination at all.

    spacing: median nearest-neighbor distance within frame a (how close
    together candidates typically sit -- the scale a wrong match jumps by).
    """
    pos_a = np.asarray(pos_a, dtype=float)
    pos_b = np.asarray(pos_b, dtype=float)
    if len(pos_a) < 5 or len(pos_b) < 5:
        return None

    tree_a = cKDTree(pos_a)
    spacing_d, _ = tree_a.query(pos_a, k=2)
    spacing = float(np.median(spacing_d[:, 1]))
    if spacing <= 0:
        return None

    tree_b = cKDTree(pos_b)
    nearest_d, _ = tree_b.query(pos_a, k=1)
    displacement = float(np.percentile(nearest_d, 10))
    return displacement, spacing


def z_noise_floor_mm(
    cals, cpar, detection_noise_px: float = 0.5, probe_mm: float = 10.0
) -> float:
    """Implied z-position noise (mm) from a nominal 2D detection precision,
    via each camera's own depth (z) sensitivity: how many image pixels a
    small along-z displacement produces, at the volume centroid. Camera
    geometry sets this -- a rig with cameras clustered in near-parallel
    viewing directions has poor z sensitivity (few px per mm of z motion),
    so the same detection-pixel noise implies a much larger 3D z error than
    it would for a well-converged rig. Returns the WORST camera's implied
    z-noise (the weakest link, since triangulation needs every camera's
    2D position and the worst one's noise propagates through).
    """
    from openptv2.imgcoord import image_coordinates
    from openptv2.transforms import convert_arr_metric_to_pixel

    worst = 0.0
    probe = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, probe_mm]])
    for cal in cals:
        px = convert_arr_metric_to_pixel(
            image_coordinates(probe, cal, cpar.get_multimedia_params()), cpar
        )
        px_per_mm_z = float(np.linalg.norm(px[1] - px[0])) / probe_mm
        if px_per_mm_z <= 0:
            continue
        implied_z_noise = detection_noise_px / px_per_mm_z
        worst = max(worst, implied_z_noise)
    return worst


@dataclass
class ConditioningReport:
    displacement_mm: float
    spacing_mm: float
    z_noise_mm: float
    ratio: float  # z_noise / displacement -- how much of the true signal the noise could swallow
    verdict: str  # "well-conditioned" | "marginal" | "poorly-conditioned"
    message: str


def assess_tracking_conditioning(
    pos_a, pos_b, cals, cpar, detection_noise_px: float = 0.5
) -> Optional[ConditioningReport]:
    """Compare the estimated true frame-to-frame motion against the
    calibration's implied z-reconstruction noise floor, and classify
    whether individual trajectories from this dataset are likely to be
    trustworthy, marginal, or dominated by reconstruction noise.

    Returns None if there isn't enough data (too few points in either
    frame) to estimate motion scale at all.
    """
    scale = measure_motion_scale(pos_a, pos_b)
    if scale is None:
        return None
    displacement, spacing = scale
    z_noise = z_noise_floor_mm(cals, cpar, detection_noise_px=detection_noise_px)
    ratio = z_noise / displacement if displacement > 0 else float("inf")

    if ratio < 0.3:
        verdict = "well-conditioned"
        message = (
            f"Tracking conditioning: well-conditioned (z-noise/motion = {ratio:.2f}). "
            f"Estimated motion {displacement:.3f}mm/frame comfortably exceeds the "
            f"z-reconstruction noise floor ({z_noise:.3f}mm) -- individual trajectories "
            f"should be trustworthy."
        )
    elif ratio < 1.0:
        verdict = "marginal"
        message = (
            f"Tracking conditioning: MARGINAL (z-noise/motion = {ratio:.2f}). "
            f"Estimated motion {displacement:.3f}mm/frame is comparable to the "
            f"z-reconstruction noise floor ({z_noise:.3f}mm, implied by this "
            f"calibration's depth sensitivity and a {detection_noise_px}px detection "
            f"precision). Expect some short/fragmented trajectories even with good "
            f"tracker parameters; ensemble or phase-averaged fields will be more "
            f"reliable than individual long Lagrangian tracks."
        )
    else:
        verdict = "poorly-conditioned"
        message = (
            f"Tracking conditioning: POORLY-CONDITIONED (z-noise/motion = {ratio:.2f}). "
            f"The z-reconstruction noise floor ({z_noise:.3f}mm) exceeds the estimated "
            f"true motion ({displacement:.3f}mm/frame) -- this is a property of the "
            f"input data (this calibration's depth sensitivity vs how slow this flow "
            f"is), not the tracker's parameters or algorithm. Neither tighter gates, a "
            f"Kalman filter, nor a global (Hungarian) assignment can recover trajectory "
            f"length that the reconstruction noise itself has erased; expect mostly "
            f"2-3 point fragments regardless of tracker choice. Consider: ensemble/"
            f"phase-averaged Eulerian fields instead of individual trajectories, a "
            f"higher frame rate (larger motion per frame), or recalibrating with "
            f"depth-spanning tracer particles rather than a shallow calibration plate."
        )

    return ConditioningReport(
        displacement_mm=displacement,
        spacing_mm=spacing,
        z_noise_mm=z_noise,
        ratio=ratio,
        verdict=verdict,
        message=message,
    )
