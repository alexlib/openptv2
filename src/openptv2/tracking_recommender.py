"""Auto-recommendation of tracker & parameters from dataset characteristics.

Analyses a set of 3D particle positions (read from ``rt_is.#`` files or
provided in-memory) and produces:

* A recommended tracker name (and rationale)
* Suggested velocity bounds and acceleration limits
* A confidence score for the recommendation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from openptv2.tracking_registry import (
    TrackerInfo,
    get_tracker_info,
    list_trackers,
)


@dataclass
class DatasetStats:
    """Statistics computed from a particle sequence."""

    num_frames: int = 0
    num_particles_per_frame: list[int] = field(default_factory=list)
    mean_particles_per_frame: float = 0.0
    max_particles_per_frame: int = 0
    std_particles_per_frame: float = 0.0

    # Velocity statistics
    max_displacement: float = 0.0
    mean_displacement: float = 0.0
    std_displacement: float = 0.0
    percentiles_displacement: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Acceleration statistics
    max_acceleration: float = 0.0
    mean_acceleration: float = 0.0
    percentile95_acceleration: float = 0.0

    # Spatial statistics
    domain_size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_interparticle_distance: float = 0.0
    density_category: str = "unknown"

    # Derived flags
    has_gaps: bool = False
    gap_fraction: float = 0.0
    particle_consistency: str = "stable"  # stable | dropping | growing | erratic


@dataclass
class Recommendation:
    """Output of the recommender."""

    tracker_name: str
    tracker_info: TrackerInfo | None = None
    confidence: float = 0.0  # 0–1
    rationale: list[str] = field(default_factory=list)

    suggested_params: dict[str, Any] = field(default_factory=dict)
    alternate_choices: list[str] = field(default_factory=list)


def compute_dataset_stats(
    frame_particles: list[np.ndarray],
) -> DatasetStats:
    """Compute statistics from a list of frame particle arrays.

    Parameters
    ----------
    frame_particles : list[np.ndarray]
        ``[(N_i, 3), ...]`` arrays of 3D positions for each frame.

    Returns
    -------
    DatasetStats
    """
    stats = DatasetStats()
    stats.num_frames = len(frame_particles)
    counts = [len(p) for p in frame_particles if len(p) > 0]
    stats.num_particles_per_frame = counts
    stats.mean_particles_per_frame = float(np.mean(counts)) if counts else 0.0
    stats.max_particles_per_frame = int(np.max(counts)) if counts else 0
    stats.std_particles_per_frame = float(np.std(counts)) if counts else 0.0

    nonempty = [p for p in frame_particles if len(p) > 0]
    if len(nonempty) < 2:
        return stats

    # Domain extent
    all_pts = np.concatenate(nonempty, axis=0)
    if len(all_pts) == 0:
        return stats
    xmin, ymin, zmin = all_pts.min(axis=0)
    xmax, ymax, zmax = all_pts.max(axis=0)
    stats.domain_size = (xmax - xmin, ymax - ymin, zmax - zmin)

    # Inter-particle distance (mean of nearest-neighbour distances, sampled)
    if len(all_pts) < 5000:
        sample = all_pts
    else:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_pts), 5000, replace=False)
        sample = all_pts[idx]
    from scipy.spatial import KDTree

    tree = KDTree(sample)
    dists, _ = tree.query(sample, k=2)  # k=2 because first is self
    stats.mean_interparticle_distance = float(np.mean(dists[:, 1]))

    # Frame-to-frame displacements & accelerations
    all_displacements = []
    all_accelerations = []
    prev_frame = nonempty[0]

    for i in range(1, len(nonempty)):
        curr = nonempty[i]
        # For each particle in prev, find nearest in curr
        tree = KDTree(curr)
        for p in prev_frame:
            d, _ = tree.query(p)
            all_displacements.append(d)

        # Acceleration: second difference of positions
        if i >= 2:
            prev2 = nonempty[i - 2]
            tree_prev = KDTree(prev2)
            for c in curr:
                d_prev, idx_prev = tree_prev.query(c)
                if idx_prev < len(prev_frame):
                    a = abs(d_prev - all_displacements[-1])  # rough
                    all_accelerations.append(a)

        prev_frame = curr

    arr_d = np.array(all_displacements)
    if len(arr_d) > 0:
        stats.max_displacement = float(arr_d.max())
        stats.mean_displacement = float(arr_d.mean())
        stats.std_displacement = float(arr_d.std())
        stats.percentiles_displacement = (
            float(np.percentile(arr_d, 25)),
            float(np.percentile(arr_d, 50)),
            float(np.percentile(arr_d, 95)),
        )

    arr_a = np.array(all_accelerations)
    if len(arr_a) > 0:
        stats.max_acceleration = float(arr_a.max())
        stats.mean_acceleration = float(arr_a.mean())
        stats.percentile95_acceleration = float(np.percentile(arr_a, 95))

    # Density category
    if stats.mean_interparticle_distance > 0 and stats.max_displacement > 0:
        ratio = stats.max_displacement / stats.mean_interparticle_distance
        if ratio < 0.3:
            stats.density_category = "low"
        elif ratio < 0.6:
            stats.density_category = "low_to_moderate"
        elif ratio < 0.9:
            stats.density_category = "moderate"
        else:
            stats.density_category = "high"

    # Gap detection
    if len(counts) > 0:
        min_count = min(counts)
        max_count = max(counts)
        if max_count > 0:
            stats.gap_fraction = 1.0 - (min_count / max_count)
        stats.has_gaps = stats.gap_fraction > 0.15

        # Particle count consistency
        if stats.std_particles_per_frame <= stats.mean_particles_per_frame * 0.1:
            stats.particle_consistency = "stable"
        elif stats.num_frames > 2:
            half = stats.num_frames // 2
            first_half = np.mean(counts[:half])
            second_half = np.mean(counts[half:])
            if second_half > first_half * 1.2:
                stats.particle_consistency = "growing"
            elif second_half < first_half * 0.8:
                stats.particle_consistency = "dropping"
            else:
                stats.particle_consistency = "erratic"

    return stats


def recommend_tracker(
    stats: DatasetStats,
    user_preferences: dict[str, Any] | None = None,
) -> Recommendation:
    """Recommend a tracker and parameters based on dataset statistics.

    Parameters
    ----------
    stats : DatasetStats
        Computed from ``compute_dataset_stats``.
    user_preferences : dict, optional
        Optional constraints::

            {"priority": "speed" | "accuracy" | "default",
             "max_tracker_speed": "fastest" | "fast" | "moderate" | "slow",
             "require_2d": bool,
             "require_backward": bool}

    Returns
    -------
    Recommendation
    """
    prefs = user_preferences or {}
    priority = prefs.get("priority", "default")

    rationale: list[str] = []
    candidates = list(list_trackers())

    # Filter
    if prefs.get("require_backward"):
        candidates = [c for c in candidates if c.supports_backward]
    if prefs.get("require_2d"):
        candidates = [c for c in candidates if c.supports_2d]
    if prefs.get("max_tracker_speed"):
        speed_rank = {"fastest": 0, "fast": 1, "moderate": 2, "slow": 3}
        max_val = speed_rank.get(prefs["max_tracker_speed"], 3)
        candidates = [
            c for c in candidates
            if speed_rank.get(c.speed_ranking, 3) <= max_val
        ]

    # Score each candidate
    scored: list[tuple[float, TrackerInfo]] = []
    for info in candidates:
        score = _score_tracker(info, stats, priority)
        scored.append((score, info))

    scored.sort(key=lambda x: (-x[0], x[1].speed_ranking))
    if not scored:
        return Recommendation(
            tracker_name="priority_segment_3d",
            confidence=0.0,
            rationale=["No suitable tracker found, falling back to default."],
        )

    best_score, best_info = scored[0]
    total = sum(s for s, _ in scored) or 1
    confidence = best_score / total if total > 0 else 0.0

    # Build rationale
    best_score_details = _explain_score(best_info, stats)
    rationale.extend(best_score_details)
    if stats.density_category != "unknown":
        rationale.append(
            f"Particle density: {stats.density_category} "
            f"({stats.mean_particles_per_frame:.0f} particles/frame avg, "
            f"inter-particle distance ~{stats.mean_interparticle_distance:.2f} mm)"
        )

    # Parameter suggestions
    suggested = _suggest_params(best_info, stats)

    remaining = [c for s, c in scored[1:3] if s > 0]
    alternates = [c.name for c in remaining[:2]]

    return Recommendation(
        tracker_name=best_info.name,
        tracker_info=best_info,
        confidence=confidence,
        rationale=rationale,
        suggested_params=suggested,
        alternate_choices=alternates,
    )


def _score_tracker(info: TrackerInfo, stats: DatasetStats, priority: str) -> float:
    """Score a tracker on how well it matches the dataset."""
    score = 0.0

    # Density match
    density_order = ["low", "low_to_moderate", "moderate", "high"]
    info_density = {"low": 0, "low_to_moderate": 1, "moderate": 2, "high": 3}
    data_density = density_order.index(stats.density_category) if stats.density_category in density_order else 1
    info_val = info_density.get(info.density_ranking, 2)
    density_diff = abs(info_val - data_density)
    score += max(0, 3.0 - density_diff * 1.5)

    # Gaps
    if stats.has_gaps:
        if info.supports_gap_relinking or info.supports_backward:
            score += 3.0
        else:
            score -= 1.0

    # Particle consistency
    if stats.particle_consistency == "dropping" or stats.particle_consistency == "erratic":
        if info.supports_new_particles:
            score += 2.0
    if stats.particle_consistency == "growing":
        if info.supports_new_particles:
            score += 2.0

    # Priority
    if priority == "speed":
        speed_scores = {"fastest": 5, "fast": 4, "moderate": 2, "slow": 0}
        score += speed_scores.get(info.speed_ranking, 2)
    elif priority == "accuracy":
        accuracy_scores = {"highest": 5, "high": 4, "standard": 2, "draft": 0}
        score += accuracy_scores.get(info.accuracy_ranking, 2)

    return score


def _explain_score(info: TrackerInfo, stats: DatasetStats) -> list[str]:
    lines: list[str] = []
    lines.append(f"Recommended tracker: {info.display_name}")
    lines.append(f"  Why: {info.short_description}")

    if stats.has_gaps and (info.supports_gap_relinking or info.supports_backward):
        lines.append(f"  Good for gaps: Dataset has {stats.gap_fraction:.0%} particle dropout.")
    if stats.particle_consistency != "stable" and info.supports_new_particles:
        lines.append(f"  Handles variable particle count ({stats.particle_consistency}).")

    return lines


def _suggest_params(info: TrackerInfo, stats: DatasetStats) -> dict[str, Any]:
    """Suggest kinematic search bounds from this dataset's own displacement/
    acceleration statistics.

    Uses the 95th percentile, not the raw max, as the basis: max_displacement
    / max_acceleration are nearest-neighbour estimates (no true correspondence
    is known ahead of tracking), so at any real particle density a handful of
    frames will have their nearest neighbour be the WRONG particle -- a
    single such mismatch inflates the max to an outlier untethered from the
    dataset's actual kinematics, which upstream feeds an unbounded search
    cone straight into the tracker (observed: large enough to crash the C
    tracker outright). The 95th percentile is still a generous bound, just
    not dictated by the single worst mismatch in the dataset.

    Margin calibrated empirically (scripts/tune_tracker_params.py, a grid
    sweep over dv/dacc/angle on test_data/synthetic_turbulent), not derived
    analytically: the previous 1.2x displacement margin with a SEPARATE
    p95_acceleration*1.2 dacc measurably starved priority_segment_3d,
    kalman_hungarian_3d, and nearest_hungarian_3d of recall (mean trajectory
    length as low as 1 frame on some trackers) -- the p95_acceleration
    estimate in particular compounds nearest-neighbour-mismatch noise across
    THREE frames (a second difference), so it runs even more inflated than
    p95_displacement. Across all three trackers swept, capping dacc at the
    SAME value as the (now wider) displacement bound outperformed the
    separate, larger acceleration-derived value.

    That earlier sweep only ever compared dacc values *at or above*
    half_window, so it established which of those was best, not that
    half_window itself was. A later sweep going *below* it (see the inline
    table at the ``dacc`` assignment) shows dacc == half_window is never the
    best choice and is the worst for every tracker at high density; dacc is
    now 0.6 * half_window.
    """
    params: dict[str, Any] = {}

    p95_displacement = stats.percentiles_displacement[2] or stats.max_displacement

    if p95_displacement > 0:
        margin = 3.0  # empirically calibrated, see docstring
        half_window = p95_displacement * margin
        params["dvxmin"] = -half_window
        params["dvxmax"] = half_window
        params["dvymin"] = -half_window
        params["dvymax"] = half_window
        params["dvzmin"] = -half_window
        params["dvzmax"] = half_window
        # Not derived from p95_acceleration -- see docstring -- but no longer
        # equal to half_window either. dacc is the SEEDED-STEP search box
        # (track_kernels_track3d._track3d_full_loop levels 1-2), a position
        # tolerance around a velocity prediction, so it should be tighter
        # than the raw displacement window. At dacc == half_window the port
        # is bit-identical to the C original, which throws away the one thing
        # the port does better (docs/plans/2026-08-16-tracking-next-steps.md
        # §3.3).
        #
        # 0.6 is measured, at dvxmax = 6 on both ground-truth synthetic sets,
        # precision / yield, postprocess off:
        #
        #   dacc/dvxmax                0.4            0.6            1.0
        #   --- 220 particles/frame ---
        #   priority_segment_3d   .978 / .850    .975 / .891    .967 / .894
        #   nearest_hungarian_3d  .980 / .778    .987 / .882    .984 / .890
        #   kalman_hungarian_3d   .968 / .679    .978 / .825    .974 / .874
        #   --- 970 particles/frame ---
        #   priority_segment_3d   .950 / .873    .938 / .872    .916 / .867
        #   nearest_hungarian_3d  .960 / .860    .946 / .853    .848 / .765
        #   kalman_hungarian_3d   .938 / .798    .936 / .839    .875 / .783
        #
        # 1.0 is never the best column and is the worst for every tracker at
        # the higher density. 0.6 takes large gains where results are poor
        # (dense: up to +9.8 points of precision and +8.8 of yield) against
        # small losses where they are already good (sparse: kalman gives up
        # 4.9 points of yield, the worst case). The optimum is genuinely
        # density-dependent -- 0.8 wins at 220/frame, 0.4 at 970/frame -- so
        # a future refinement could scale it by
        # stats.mean_interparticle_distance rather than use one constant.
        params["dacc"] = half_window * 0.6

    # Cap only against a genuinely extreme cone (several times the mean
    # particle spacing), not "wider than a single mean spacing": the
    # previous 0.8x cap was clamping DOWN into exactly the range the sweep
    # (see this function's docstring) showed starved these trackers of
    # recall -- assignment is cost/Hungarian-based, not raw nearest-
    # neighbour, so ambiguity near typical spacing is exactly what the
    # optimal assignment step is for, not something to avoid by shrinking
    # the search cone first.
    if stats.mean_interparticle_distance > 0:
        extreme_cap = stats.mean_interparticle_distance * 2.5
        if "dvxmax" in params and params["dvxmax"] > extreme_cap:
            params["dvxmax"] = extreme_cap
            params["dvxmin"] = -extreme_cap
            params["dvymax"] = extreme_cap
            params["dvymin"] = -extreme_cap
            params["dvzmax"] = extreme_cap
            params["dvzmin"] = -extreme_cap
            params["dacc"] = min(params["dacc"], extreme_cap)

    # Tracker-specific params
    if info.name == "nearest_hungarian_3d":
        if p95_displacement > 0:
            params["v_max"] = params["dvxmax"]
            params["a_max"] = params["dacc"]
        # This tracker's angle filter is a hard binary reject (not a soft
        # cost term like kalman_hungarian_3d's), so an overly tight bound
        # throws away good matches rather than just de-weighting them --
        # swept best at effectively unrestricted (200 gon = 180 deg).
        params["angle"] = 200.0
    elif info.name == "predictive_gmm_3d":
        if p95_displacement > 0:
            params["maxvel"] = params["dvxmax"]

    return params


def recommend_from_files(
    rt_is_dir: str | Path,
    first: int,
    last: int,
    user_preferences: dict[str, Any] | None = None,
) -> Recommendation:
    """Analyse ``rt_is.#`` files and recommend a tracker.

    Parameters
    ----------
    rt_is_dir : str or Path
        Directory containing ``rt_is.#`` files (e.g. ``"res"``).
    first, last : int
        Frame range.
    user_preferences : dict, optional
        Passed to ``recommend_tracker``.

    Returns
    -------
    Recommendation
    """
    from openptv2.algorithms.tracking_frame_buf import Frame

    rt_is_dir = Path(rt_is_dir)
    corres_base = str(rt_is_dir / "rt_is")
    frame_particles: list[np.ndarray] = []
    for fn in range(first, last + 1):
        if not (rt_is_dir / f"rt_is.{fn}").exists():
            frame_particles.append(np.empty((0, 3)))
            continue
        frame = Frame(num_cams=4, max_targets=10000)
        frame.read(corres_base, "", target_file_base="", frame_num=fn)
        frame_particles.append(frame.positions())

    stats = compute_dataset_stats(frame_particles)
    return recommend_tracker(stats, user_preferences)


def print_recommendation(rec: Recommendation) -> str:
    """Format a Recommendation for terminal display."""
    lines = [
        "=" * 70,
        "  Tracker Recommendation",
        "=" * 70,
        "",
    ]
    for r in rec.rationale:
        lines.append(f"  {r}")

    lines.append("")
    lines.append(f"  Confidence: {rec.confidence:.0%}")

    if rec.alternate_choices:
        lines.append(f"  Alternate choices: {', '.join(rec.alternate_choices)}")

    if rec.suggested_params:
        lines.append("")
        lines.append("  Suggested parameters:")
        for k, v in rec.suggested_params.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.2f}")
            else:
                lines.append(f"    {k}: {v}")

    lines.append("")
    lines.append("  For full tracker details: openptv list-trackers --show <name>")
    lines.append("=" * 70)
    return "\n".join(lines)


__all__ = [
    "DatasetStats",
    "Recommendation",
    "compute_dataset_stats",
    "recommend_tracker",
    "recommend_from_files",
    "print_recommendation",
]
