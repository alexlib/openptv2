"""Theoretical performance models for stereo matching and tracking.

Two analytic results from Raghunathan, Vradman & Shnapp (2026),
"Time-marching stereo-matching for 3D-PTV flow measurements" (main.pdf):

1. Ghost-particle scaling (their Appendix A, Eq. 7): for ``n_c`` cameras,
   ``n_p`` particles in a domain of linear size ``L`` and a ray-crossing
   tolerance ``d_min``, the expected number of ghost particles per frame is

       n_ghost ~ n_p**n_c * (d_min / L)**(2*n_c - 3)

   (extension of Graff & Gharib's thin-volume model to thick 3D-PTV
   volumes). This gives a principled criterion for choosing the
   triangulation tolerance ``d_min`` given a ghost budget, and defines
   sparse / intermediate / dense stereo-matching regimes.

2. Trackability (their Eq. 2): temporal (prediction-based) matching works
   when the inter-frame displacement in image space stays well below the
   inter-particle separation:

       Tr = dx * n_p**(1/2) / l     (dx: displacement, l: image linear
                                     dimension, both in pixels)

   The same condition in physical space reads: the position-uncertainty
   radius around a predicted point must stay below the inter-particle
   distance for the nearest candidate to be the right one. It bounds the
   search radius of any prediction-seeded linking step (e.g. the dacc
   seeded-step box in the trackers, or track_assisted's radius_px).

All functions are pure and unit-tested in tests/unit/test_theory.py.
This module is deliberately NOT in algorithms/ so it needs no Cython build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "expected_ghosts",
    "ghost_regime",
    "recommend_tolerance",
    "trackability",
    "prediction_search_radius",
    "TheoryReport",
]


def expected_ghosts(
    n_p: int,
    n_c: int,
    d_min: float,
    L: float,
) -> float:
    """Expected ghost particles per frame, Eq. 7 of the paper.

    Parameters
    ----------
    n_p : int
        Number of (correctly segmented) particles per frame.
    n_c : int
        Number of cameras whose rays must cross for an accepted particle.
    d_min : float
        Ray-crossing (triangulation) tolerance, same units as ``L``.
    L : float
        Linear dimension of the measured domain, same units as ``d_min``
        (e.g. cube root of the calibrated volume).

    Returns
    -------
    float
        Expected number of ghost particles. Valid for n_p >> 1 and
        d_min << L; returns inf for n_c < 2 (no constraint at all).
    """
    if n_c < 2:
        return float("inf")
    if n_p <= 0 or d_min <= 0 or L <= 0:
        return 0.0
    ratio = d_min / L
    if ratio >= 1.0:
        return float("inf")
    return float((n_p**n_c) * ratio ** (2 * n_c - 3))


def ghost_regime(
    n_p: int,
    n_c: int,
    d_min: float,
    L: float,
) -> Literal["sparse", "intermediate", "dense"]:
    """Classify the stereo-matching density regime from the scaling law.

    The paper defines three regimes by how ghosts scale: below the onset
    density ghosts are negligible ("sparse"); past it they grow
    super-linearly ("intermediate"); when n_ghost ~ n_p the matching is
    ghost-dominated ("dense"). We use expected_ghosts directly: < 1 ghost
    per frame is sparse, >= n_p is dense, in between is intermediate.
    """
    g = expected_ghosts(n_p, n_c, d_min, L)
    if g < 1.0:
        return "sparse"
    if g >= n_p:
        return "dense"
    return "intermediate"


def recommend_tolerance(
    n_p: int,
    n_c: int,
    L: float,
    ghost_budget: float = 1.0,
) -> float:
    """Largest ray-crossing tolerance keeping expected ghosts <= budget.

    Inverts Eq. 7: solve n_p**n_c * (d/L)**(2n_c-3) = ghost_budget for d.

    Returns the tolerance in the same units as ``L``. If even d -> 0
    cannot meet the budget (over-dense for this camera count), returns 0.0.
    """
    if n_c < 2 or n_p <= 0 or L <= 0 or ghost_budget <= 0:
        return 0.0
    # (d/L)^(2nc-3) = budget / n_p^nc
    base = ghost_budget / (n_p**n_c)
    if base > 1.0:
        # budget exceeds even the particle count; any tolerance up to L works
        return L
    return float(L * base ** (1.0 / (2 * n_c - 3)))


def trackability(
    displacement_px: float,
    n_p: float,
    image_size_px: float,
) -> float:
    """Trackability parameter Tr = dx * sqrt(n_p) / l (Eq. 2 of the paper).

    ``displacement_px``: typical inter-frame particle translation in image
    space (pixels). ``n_p``: particles per frame. ``image_size_px``: image
    linear dimension (pixels). Temporal matching is effective when Tr is
    sufficiently small (paper: ~95% temporal matches at Tr ~ 0.25 with the
    full predictor; the naive nearest-neighbour variant already degrades
    around Tr ~ 0.1).
    """
    if image_size_px <= 0 or n_p <= 0:
        return float("inf")
    return float(displacement_px * (n_p**0.5) / image_size_px)


def prediction_search_radius(
    prediction_error: float,
    mean_interparticle_distance: float,
    safety: float = 0.5,
) -> float:
    """Cap on the search radius around a predicted particle position.

    The paper's working condition is that the error of the predicted
    position must be smaller than inter-particle separations; otherwise
    the nearest candidate to the prediction is not necessarily the right
    particle and temporal matching starts producing ghosts. We therefore
    cap any prediction-seeded search radius at ``safety`` x the mean
    inter-particle distance (default 0.5), regardless of how large the
    kinematic displacement statistics say it could be.

    Returns min(prediction_error * safety_margin_within_error, cap) --
    practically: the radius should cover the prediction error, but never
    exceed the safety fraction of the mean inter-particle distance.
    """
    cap = safety * mean_interparticle_distance
    if prediction_error <= 0:
        return cap
    return min(prediction_error, cap)


@dataclass
class TheoryReport:
    """Combined theory assessment for one dataset + camera setup."""

    n_p: int
    n_c: int
    d_min: float
    L: float
    expected_ghosts: float
    regime: str
    recommended_tolerance: float
    trackability: float
    notes: list[str]

    def summary(self) -> str:
        lines = [
            f"particles/frame={self.n_p}, cameras={self.n_c}",
            f"tolerance d_min={self.d_min:.4g} (domain {self.L:.4g})",
            f"expected ghosts/frame ~ {self.expected_ghosts:.2f} "
            f"[regime: {self.regime}]",
            f"recommended d_max for <=1 ghost: {self.recommended_tolerance:.4g}",
            f"trackability Tr={self.trackability:.3f}",
        ]
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)


def theory_report(
    n_p: int,
    n_c: int,
    d_min: float,
    L: float,
    displacement_px: float | None = None,
    image_size_px: float | None = None,
) -> TheoryReport:
    """Build a TheoryReport, optionally including the trackability check."""
    g = expected_ghosts(n_p, n_c, d_min, L)
    regime = ghost_regime(n_p, n_c, d_min, L)
    d_rec = recommend_tolerance(n_p, n_c, L, ghost_budget=1.0)
    notes: list[str] = []
    if regime == "dense":
        notes.append(
            "ghost-dominated: shrink d_min, add cameras, or lower seeding "
            "(n_ghost ~ n_p**n_c * (d/L)**(2nc-3) >= n_p)"
        )
    elif regime == "intermediate":
        notes.append(
            "ghosts grow super-linearly with density here; tightening the "
            "tolerance buys the most headroom"
        )
    if n_c > 2 and d_min > d_rec > 0:
        notes.append(
            f"current d_min exceeds the <=1 ghost bound ({d_rec:.4g}); "
            "shrinking it trades a few missed marginal matches for fewer ghosts"
        )
    tr = float("nan")
    if displacement_px is not None and image_size_px is not None:
        tr = trackability(displacement_px, n_p, image_size_px)
        if tr > 0.25:
            notes.append(
                f"Tr={tr:.3f} > 0.25: temporal prediction will match a "
                "shrinking fraction of particles; raise the frame rate or "
                "lower seeding"
            )
        else:
            notes.append(f"Tr={tr:.3f} <= 0.25: temporal prediction viable")
    return TheoryReport(
        n_p=n_p,
        n_c=n_c,
        d_min=d_min,
        L=L,
        expected_ghosts=g,
        regime=regime,
        recommended_tolerance=d_rec,
        trackability=tr,
        notes=notes,
    )
