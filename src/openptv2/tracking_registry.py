"""The three trackers, described once.

``TRACKER_REGISTRY`` has exactly three entries, ``two_phase`` (default), ``trackcorr`` and
``fast_3d``; it is the single source of truth for ``openptv list-trackers``, the recommender and
the GUI help text. Old tracker names (``priority_segment_3d``, ``full_multipass`` ...) are not
entries: ``get_tracker_info`` translates them (see ``openptv2.tracking_setup``) and the user guide is
``docs/trackers.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ParameterGuide:
    """Guidance for a single tracking parameter."""

    name: str
    type: Literal["float", "int", "bool", "choice"]
    default: str
    description: str
    how_to_choose: str = ""
    typical_range: str = ""
    unit: str = ""


@dataclass(frozen=True)
class TrackerInfo:
    """Self-describing metadata for a tracker plugin.

    Every field is plain text suitable for printing in a terminal or
    displaying in a GUI tooltip.
    """

    # ── Identity ──────────────────────────────────────────────────────
    name: str
    display_name: str
    short_description: str  # one-liner for tables

    # ── Algorithm ─────────────────────────────────────────────────────
    algorithm_summary: str  # paragraph describing how it works
    algorithm_detail: str = ""  # longer technical description
    citation: str = ""  # DOI or BibTeX key if published

    # ── Capabilities ──────────────────────────────────────────────────
    supports_backward: bool = False
    supports_new_particles: bool = True
    supports_2d: bool = False  # uses 2D image targets (not just 3D)
    supports_postprocessing: bool = False
    supports_gap_relinking: bool = False
    supports_multimedia: bool = False
    supports_splitter: bool = False
    supports_cost_weights: bool = False  # configurable cost terms

    # ── Performance characteristics ───────────────────────────────────
    speed_ranking: Literal["fastest", "fast", "moderate", "slow"] = "moderate"
    density_ranking: Literal["low", "low_to_moderate", "moderate", "high"] = "moderate"
    accuracy_ranking: Literal["draft", "standard", "high", "highest"] = "standard"

    # ── Parameter guidance ────────────────────────────────────────────
    parameters: tuple[ParameterGuide, ...] = field(default_factory=tuple)
    default_preset: str = ""

    # ── Use-case recommendations ──────────────────────────────────────
    best_for: str = ""
    avoid_when: str = ""
    typical_datasets: str = ""

    # ── Integration ───────────────────────────────────────────────────
    plugin_module: str = ""  # resolved import path
    preset_name: str | None = None  # TrackingPreset enum value if any


# ── Registry ──────────────────────────────────────────────────────────

TRACKER_REGISTRY: dict[str, TrackerInfo] = {}

_BOUNDS = (
    ParameterGuide(
        name="dvxmin / dvxmax, dvymin / dvymax, dvzmin / dvzmax",
        type="float",
        default="-10 / 10",
        description="Velocity search box per axis. Same name, unit and meaning for every tracker.",
        how_to_choose="Just above the largest displacement between frames (95th percentile of a probe run).",
        typical_range="1 – 100",
        unit="mm/frame",
    ),
    ParameterGuide(
        name="dacc",
        type="float",
        default="5.0",
        description="Largest acceleration (change of velocity between frames).",
        how_to_choose="From a probe run's acceleration statistics; too tight breaks fast turns, too loose admits wrong links.",
        typical_range="0.5 – 50",
        unit="mm/frame²",
    ),
    ParameterGuide(
        name="angle",
        type="float",
        default="120",
        description="Largest change of direction between frames.",
        how_to_choose="Leave at the default unless the flow turns sharply.",
        typical_range="20 – 200",
        unit="gon",
    ),
)

_DIRECTION = ParameterGuide(
    name="direction",
    type="choice",
    default="both",
    description="forward = frames in order. both = also run the same tracker on the reversed frame order and keep the links both agree on or that fill gaps.",
    how_to_choose="Keep both for quality; forward when speed matters more (about half the time).",
    typical_range="forward | both",
)

TWO_PHASE_INFO = TrackerInfo(
    name="two_phase",
    display_name="Two-phase (default)",
    short_description="3D candidates, image-based cost, optimal assignment, two-hop confirmation",
    algorithm_summary=(
        "Phase 1: a 3D KD-tree finds every next-frame point within the velocity radius of each track's "
        "predicted position. Phase 2: inside each connected group of candidates, the distance in the camera "
        "images between the prediction and each candidate (the 2D 'leaf' positions) becomes a cost, and the "
        "Hungarian algorithm picks the best assignment. A link is kept only if the next link continues "
        "smoothly (two-hop confirmation). Tracks carry their velocity and coast through short gaps; "
        "ghost-like and brightness-jump candidates are refused."
    ),
    algorithm_detail=(
        "Pure Python/NumPy/SciPy (plugins/two_phase_tracking.py). Needs the run store with per-camera "
        "targets. direction both runs the backward pass on reversed frames and merges (Forward-First). "
        "Add-ons: shared_correction re-estimates the points that touch a shared blob from their tracks and links "
        "again; holistic_rounds > 0 adds births of hidden partners."
    ),
    supports_backward=True,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=True,
    supports_cost_weights=True,
    speed_ranking="fastest",
    density_ranking="moderate",
    accuracy_ranking="highest",
    parameters=(
        _DIRECTION,
        *_BOUNDS[:1],
        ParameterGuide(
            name="shared_correction",
            type="bool",
            default="false",
            description="Re-estimate the points that touch a blob shared with another particle from their tracks, then link again.",
            how_to_choose="Off by default (about 4 times the run time). Helps when particles overlap in the images; "
                          "pairs well with ptv.shared_pass.",
            typical_range="false / true",
        ),
        ParameterGuide(
            name="holistic_rounds",
            type="int",
            default="0",
            description="0 = off. N > 0 adds the merged-blob loop: N rounds of refine, birth and relink.",
            how_to_choose="Off by default (slower). 2 gets most of the gain; a third round changed little.",
            typical_range="0 – 3",
        ),
        ParameterGuide(
            name="leaf_weight",
            type="float",
            default="1.0",
            description="Weight of the 2D image distance in the cost. 0 falls back to pure 3D matching.",
            how_to_choose="Raise when 3D positions are noisy but detections are clean; 0 to disable.",
            typical_range="0 – 2",
        ),
        ParameterGuide(
            name="max_gap",
            type="int",
            default="2",
            description="Frames a track may go unmatched before it ends.",
            how_to_choose="Match the expected occlusion length.",
            typical_range="1 – 5",
            unit="frames",
        ),
    ),
    best_for="Quality: the most accurate tracker on the benchmark, and the one the merged-blob loop works with.",
    avoid_when="You need the legacy algorithm's exact behaviour (use trackcorr) or the lowest run time (use fast_3d).",
    typical_datasets="Any run with a run store (the standard pipeline).",
    plugin_module="openptv2.plugins.two_phase_tracking",
)

TRACKCORR_INFO = TrackerInfo(
    name="trackcorr",
    display_name="Trackcorr (legacy algorithm)",
    short_description="The legacy trackcorr: predict, project into each camera, search the 2D targets",
    algorithm_summary=(
        "For each particle it predicts the next 3D position from its history, projects the prediction into "
        "every camera, searches the detected 2D targets inside a window there, and confirms candidates by "
        "epipolar matching. Faithful port of the C trackcorr (compiled Cython kernel); it reproduces the "
        "legacy results at the same search bounds. Nothing else is added."
    ),
    algorithm_detail=(
        "Tracker.full_forward() / full_backward() over trackcorr_c_loop. direction both = forward, "
        "backward, reciprocity check and gap bridging (postprocess)."
    ),
    supports_backward=True,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=True,
    supports_gap_relinking=True,
    speed_ranking="slow",
    density_ranking="low_to_moderate",
    accuracy_ranking="standard",
    parameters=(
        _DIRECTION,
        *_BOUNDS,
        ParameterGuide(
            name="postprocess",
            type="bool",
            default="true",
            description="After the passes: reciprocity check and gap bridging.",
            how_to_choose="Keep true unless you want the raw links.",
        ),
    ),
    best_for="Comparison with the legacy software and old results; compatibility.",
    avoid_when="You want the best quality (two_phase) or the lowest run time (fast_3d).",
    typical_datasets="Any; sparse to moderate seeding.",
    plugin_module="openptv2.plugins.default_tracking",
)

FAST_3D_INFO = TrackerInfo(
    name="fast_3d",
    display_name="Fast 3D (3D points only)",
    short_description="Links the triangulated 3D points only; never searches the images",
    algorithm_summary=(
        "Three cheapest-first passes over the 3D points of consecutive frames: first points that already have "
        "a previous link (cost = change of velocity), then points that borrow their neighbours' velocity, then "
        "isolated points. Each pass takes the cheapest links first and never uses an end twice. Optional "
        "epi_weight adds a soft penalty from how well each point's camera rays met (the old '3d_hybrid')."
    ),
    algorithm_detail=(
        "Compiled kernel track3d_loop_fast (algorithms/track_kernels_track3d.py), Tracker.full_forward_3d(). "
        "Greedy per pass (not optimal), no image information beyond epi_weight, no link confirmation."
    ),
    supports_backward=True,
    supports_new_particles=False,
    supports_2d=False,
    supports_postprocessing=True,
    supports_gap_relinking=True,
    speed_ranking="fast",
    density_ranking="moderate",
    accuracy_ranking="high",
    parameters=(
        _DIRECTION,
        *_BOUNDS,
        ParameterGuide(
            name="epi_weight",
            type="float",
            default="0",
            description="0 = pure 3D. Above 0, adds weight × (1 − epipolar score) to each link cost.",
            how_to_choose="1.0 is what the old '3d_hybrid' meant. Measured gain on the benchmark: negligible "
                          "(velocity error 0.1665 -> 0.1661), so the default is 0.",
            typical_range="0 – 2",
        ),
        ParameterGuide(
            name="accel_gate",
            type="bool",
            default="false",
            description="Also refuse links whose acceleration exceeds dacc or whose direction change exceeds angle.",
            how_to_choose="With a loose dacc (6) it gave velocity error 0.1595 against 0.1665; a tight dacc (0.4) lost links "
                          "(link recall 0.887). Needs dacc from a probe run.",
        ),
        ParameterGuide(
            name="postprocess",
            type="bool",
            default="true",
            description="After the passes: reciprocity check and gap bridging.",
            how_to_choose="Keep true unless you want the raw links.",
        ),
    ),
    best_for="Runs where only triangulated points are available, and as a 3D-only baseline; "
             "about as fast as two_phase forward (4.9 s against 4.2 s on the benchmark, docs/trackers.md).",
    avoid_when="You want the best quality or the shortest run time on the standard pipeline (two_phase).",
    typical_datasets="Any with triangulated points.",
    plugin_module="openptv2.plugins.default_tracking",
)


def _build_registry() -> None:
    for info in (TWO_PHASE_INFO, TRACKCORR_INFO, FAST_3D_INFO):
        TRACKER_REGISTRY[info.name] = info


_build_registry()


def get_tracker_info(name: str) -> TrackerInfo | None:
    """The info for a tracker name. Old names (``priority_segment_3d``, ``full_multipass``,
    ``holistic`` ...) are translated to the tracker they became."""
    from openptv2.tracking_setup import canonical_tracker_name

    key = canonical_tracker_name(name)
    return TRACKER_REGISTRY.get(key) if key else None


def list_trackers() -> list[TrackerInfo]:
    return list(TRACKER_REGISTRY.values())


def print_tracker_table() -> str:
    """Return a formatted terminal table of all trackers."""
    lines = [
        f"{'Name':<28} {'Speed':<9} {'Accuracy':<9} {'Density':<16} {'Backward':<9} {'New Pts':<9} {'2D':<5}",
        "-" * 90,
    ]
    for info in list_trackers():
        lines.append(
            f"{info.display_name:<28} {info.speed_ranking:<9} {info.accuracy_ranking:<9} "
            f"{info.density_ranking:<16} {'Yes' if info.supports_backward else 'No':<9} "
            f"{'Yes' if info.supports_new_particles else 'No':<9} "
            f"{'Yes' if info.supports_2d else 'No':<5}"
        )
    return "\n".join(lines)


def print_tracker_detail(name: str) -> str:
    """Return a detailed description of one tracker."""
    info = get_tracker_info(name)
    if info is None:
        return f"Unknown tracker: {name!r}"

    lines = [
        f"{'=' * 70}",
        f"  {info.display_name}",
        f"{'=' * 70}",
        "",
        f"  {info.short_description}",
        "",
        "  Algorithm:",
        f"    {info.algorithm_summary}",
    ]
    if info.algorithm_detail:
        lines.append("")
        lines.append("  Technical detail:")
        lines.append(f"    {info.algorithm_detail}")
    if info.citation:
        lines.append("")
        lines.append(f"  Citation: {info.citation}")

    lines.append("")
    lines.append("  Capabilities:")
    lines.append(
        f"    {'Backward pass':.<30} {'Yes' if info.supports_backward else 'No'}"
    )
    lines.append(
        f"    {'New particles mid-seq':.<30} {'Yes' if info.supports_new_particles else 'No'}"
    )
    lines.append(
        f"    {'2D target tracking':.<30} {'Yes' if info.supports_2d else 'No'}"
    )
    lines.append(
        f"    {'Post-processing':.<30} {'Yes' if info.supports_postprocessing else 'No'}"
    )
    lines.append(
        f"    {'Gap relinking':.<30} {'Yes' if info.supports_gap_relinking else 'No'}"
    )
    lines.append(
        f"    {'Cost weights':.<30} {'Yes' if info.supports_cost_weights else 'No'}"
    )

    lines.append("")
    lines.append("  Performance:")
    lines.append(f"    {'Speed':.<30} {info.speed_ranking}")
    lines.append(f"    {'Accuracy':.<30} {info.accuracy_ranking}")
    lines.append(f"    {'Density handling':.<30} {info.density_ranking}")

    if info.parameters:
        lines.append("")
        lines.append("  Parameters:")
        for p in info.parameters:
            lines.append(f"    {p.name}")
            lines.append(
                f"      Default: {p.default}  |  Range: {p.typical_range}  |  Unit: {p.unit}"
            )
            lines.append(f"      {p.description}")
            if p.how_to_choose:
                lines.append(f"      Tuning: {p.how_to_choose}")

    if info.best_for:
        lines.append("")
        lines.append(f"  Best for: {info.best_for}")
    if info.avoid_when:
        lines.append(f"  Avoid when: {info.avoid_when}")
    if info.typical_datasets:
        lines.append(f"  Typical datasets: {info.typical_datasets}")

    return "\n".join(lines)


__all__ = [
    "TRACKER_REGISTRY",
    "TrackerInfo",
    "ParameterGuide",
    "get_tracker_info",
    "list_trackers",
    "print_tracker_table",
    "print_tracker_detail",
]
