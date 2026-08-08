"""Tracker capability descriptors and registry.

Every tracker plugin is described by a ``TrackerInfo`` dataclass that
documents its algorithm, input requirements, performance characteristics,
recommended use cases, and parameter guidance.  The ``TRACKER_REGISTRY``
dict maps plugin name → ``TrackerInfo`` and is the single source of truth
for the CLI list-trackers command, the recommender, and the GUI plugin
selector.
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
    short_description: str                     # one-liner for tables

    # ── Algorithm ─────────────────────────────────────────────────────
    algorithm_summary: str                     # paragraph describing how it works
    algorithm_detail: str = ""                 # longer technical description
    citation: str = ""                         # DOI or BibTeX key if published

    # ── Capabilities ──────────────────────────────────────────────────
    supports_backward: bool = False
    supports_new_particles: bool = True
    supports_2d: bool = False                  # uses 2D image targets (not just 3D)
    supports_postprocessing: bool = False
    supports_gap_relinking: bool = False
    supports_multimedia: bool = False
    supports_splitter: bool = False
    supports_cost_weights: bool = False        # configurable cost terms

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
    plugin_module: str = ""                    # resolved import path
    preset_name: str | None = None             # TrackingPreset enum value if any


# ── Registry ──────────────────────────────────────────────────────────

TRACKER_REGISTRY: dict[str, TrackerInfo] = {}

HYBRID_INFO = TrackerInfo(
    name="hybrid_3d_corr",
    display_name="Hybrid 3D + 2D Correlation (Recommended Default)",
    short_description="Two-pass adaptive tracker: 3D kinematic kernel then 2D re-triangulation for new particles",
    algorithm_summary=(
        "Pass 1 runs the compiled Cython 3D Euclidean kinematic kernel (track3d_loop) across all frames, "
        "linking ~95 % of particles by velocity/acceleration consistency alone.  "
        "Pass 2 re-triangulates remaining unlinked 2D camera targets into fresh 3D particles and seeds "
        "new tracks, so particles that first appear mid-sequence are captured without a full multi-camera "
        "epipolar pass.  Runs entirely at compiled C-speed."
    ),
    algorithm_detail=(
        "Uses the Cython track_hybrid_kernel_loop which calls track3d_loop for the 3D pass, then the "
        "trackcorr_c_finish step.  No Python-level loop overhead per frame.  "
        "The resulting linkages are written to ptv_is.# files in the same format as all other trackers."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=False,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="fast",
    density_ranking="moderate",
    accuracy_ranking="high",
    parameters=(
        ParameterGuide(
            name="dvxmin / dvxmax",
            type="float",
            default="±15.5",
            description="Velocity search window in X (mm/frame).",
            how_to_choose=(
                "Compute max inter-frame displacement from a probe run and add 10 % margin. "
                "Must be larger than true particle velocities but smaller than the typical "
                "inter-particle distance to avoid ambiguity."
            ),
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dvymin / dvymax",
            type="float",
            default="±15.5",
            description="Velocity search window in Y (mm/frame).",
            how_to_choose="Same as dvx, set per axis based on measured displacements.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dvzmin / dvzmax",
            type="float",
            default="±15.5",
            description="Velocity search window in Z (mm/frame).",
            how_to_choose="Same as dvx, set per axis based on measured displacements.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.5",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose=(
                "Compute max observed acceleration from a probe run and add 10 % margin. "
                "Too tight → broken tracks; too loose → false links in dense regions."
            ),
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation between successive velocity vectors (gon).",
            how_to_choose=(
                "Set to 120 gon for turbulent flow or 20–40 gon for laminar flow.  400 gon = 360°."
            ),
            typical_range="20 – 200",
            unit="gon",
        ),
    ),
    default_preset="hybrid_3d_corr",
    best_for=(
        "General-purpose PTV with moderate particle density and standard 3–6 camera setups. "
        "The best starting point for most datasets."
    ),
    avoid_when="Low signal-to-noise ratio where 2D re-triangulation introduces many ghost particles.",
    typical_datasets="Water-channel experiments, wind-tunnel flows, standard laboratory PTV.",
)

FULL_MULTIPASS_INFO = TrackerInfo(
    name="full_multipass",
    display_name="Full Multi-Pass (Highest Accuracy)",
    short_description="Three-pass pipeline: forward → backward → reciprocity post-processing",
    algorithm_summary=(
        "Pass 1 runs the full multi-camera 2D+3D epipolar tracking forward.  "
        "Pass 2 scans the sequence backward to recover particles that were missed because "
        "their velocity history was insufficient on the first pass.  "
        "Pass 3 enforces link reciprocity (every forward link must have a matching backward link), "
        "seeds cold starts for the first frame transition using the established velocity field, "
        "and bridges gaps up to a configurable number of frames."
    ),
    algorithm_detail=(
        "Uses the Cython trackcorr_c_loop (multi-camera projection search), trackback_c for the "
        "reverse pass, and the Python tracking_postprocess module for disk-level pruning and gap "
        "relinking.  The postprocessor reads ptv_is.# directly, so it is tracker-agnostic and can "
        "be applied after any tracker that writes compatible files."
    ),
    supports_backward=True,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=True,
    supports_gap_relinking=True,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="moderate",
    density_ranking="low_to_moderate",
    accuracy_ranking="highest",
    parameters=(
        ParameterGuide(
            name="dvxmin / dvxmax",
            type="float",
            default="±10.0",
            description="Velocity search window in X (mm/frame).",
            how_to_choose="Same as hybrid_3d_corr — compute from probe data.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.0",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Same as hybrid_3d_corr — compute from probe data.",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation (gon).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="20 – 200",
            unit="gon",
        ),
        ParameterGuide(
            name="postprocess",
            type="bool",
            default="true",
            description="Run reciprocity, cold-start, and gap-relinking after backward pass.",
            how_to_choose="Keep true unless you want raw forward+backward links only.",
            typical_range="",
            unit="",
        ),
    ),
    default_preset="full_multipass",
    best_for="Datasets where trajectory accuracy is paramount and processing time is secondary. Best recovery of long trajectories.",
    avoid_when="Very high particle density where the multi-camera projection becomes the bottleneck, or real-time applications.",
    typical_datasets="Scientific studies requiring publication-quality trajectories, benchmark validation.",
)

FAST_INFO = TrackerInfo(
    name="fast_3d",
    display_name="Fast 3D-Only (Segment Mode)",
    short_description="Single-pass forward tracking using 3D coordinates only — no 2D targets needed",
    algorithm_summary=(
        "Links particles by 3D Euclidean proximity and acceleration consistency (second derivative). "
        "Does not project to 2D camera images, does not add new particles mid-sequence, and does not "
        "support backward tracking.  The simplest and fastest tracker in the suite."
    ),
    algorithm_detail=(
        "Calls track3d_loop per frame (compiled Cython).  The track_mode=1 flag disables the 2D "
        "epipolar search and new-particle seeding.  Use for low-density, high-SNR data where "
        "all particles are visible in every frame."
    ),
    supports_backward=False,
    supports_new_particles=False,
    supports_2d=False,
    supports_postprocessing=False,
    supports_gap_relinking=False,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="fastest",
    density_ranking="low",
    accuracy_ranking="draft",
    parameters=(
        ParameterGuide(
            name="dvxmin / dvxmax",
            type="float",
            default="±15.5",
            description="Velocity search window in X (mm/frame).",
            how_to_choose="Set just above max observed displacement.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.5",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Set just above max observed acceleration.",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
    ),
    default_preset="fast_3d",
    best_for="Quick sanity checks, algorithm debugging, previewing a dataset before committing to a full pipeline.",
    avoid_when="Particles appear/disappear mid-sequence, high density, or when accurate trajectory completeness is required.",
    typical_datasets="Synthetic-data verification, low-noise calibration-check runs.",
)

STANDARD_FORWARD_INFO = TrackerInfo(
    name="standard_forward",
    display_name="Standard Forward (2D+3D)",
    short_description="Single forward pass using full multi-camera 2D+3D epipolar tracking with new-particle seeding",
    algorithm_summary=(
        "Runs the full multi-camera 2D+3D epipolar tracking forward across all frames. "
        "Projects a 3D search volume onto each camera image, finds candidate 2D targets, "
        "sorts by cross-camera frequency, and links using an angle + acceleration metric. "
        "New particles can seed tracks mid-sequence."
    ),
    algorithm_detail=(
        "Calls Tracker.full_forward() which loops over trackcorr_c_loop per frame. "
        "The 4-frame buffer maintains velocity and acceleration history for smooth prediction. "
        "No backward pass or post-processing is performed."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=False,
    supports_multimedia=True,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="fast",
    density_ranking="moderate",
    accuracy_ranking="standard",
    parameters=(
        ParameterGuide(
            name="dvxmin / dvxmax",
            type="float",
            default="±10.0",
            description="Velocity search window in X (mm/frame).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.0",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation (gon).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="20 – 200",
            unit="gon",
        ),
    ),
    default_preset="standard_forward",
    best_for="Quick runs with new-particle capture when backward tracking is unnecessary.",
    avoid_when="Accuracy-critical applications — use full_multipass instead.",
    typical_datasets="Preview runs, quick parameter scans.",
)

TWO_DIRECTIONAL_INFO = TrackerInfo(
    name="two_directional",
    display_name="Two-Directional (Forward + Backward)",
    short_description="Forward pass followed by backward pass without reciprocity post-processing",
    algorithm_summary=(
        "Runs the full forward multi-camera 2D+3D epipolar tracking, then a backward pass "
        "to recover tracks that were missed on the first pass.  Unlike full_multipass, no "
        "reciprocity pruning, cold-start seeding, or gap relinking is performed."
    ),
    algorithm_detail=(
        "Calls Tracker.full_forward() then Tracker.full_backward().  The backward pass uses "
        "trackback_c which reads the forward linkage files and re-links from the end of the sequence."
    ),
    supports_backward=True,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=False,
    supports_multimedia=True,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="moderate",
    density_ranking="moderate",
    accuracy_ranking="standard",
    parameters=(
        ParameterGuide(
            name="dvxmin / dvxmax",
            type="float",
            default="±10.0",
            description="Velocity search window in X (mm/frame).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.0",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation (gon).",
            how_to_choose="Same as hybrid_3d_corr.",
            typical_range="20 – 200",
            unit="gon",
        ),
    ),
    default_preset="two_directional",
    best_for="When backward tracking helps but you want to skip post-processing for speed.",
    avoid_when="Raw forward+backward links contain many non-reciprocal connections.",
    typical_datasets="Parameter-tuning runs to compare forward-only vs forward+backward.",
)

MYPTV_3D_INFO = TrackerInfo(
    name="myptv_3d_tracking",
    display_name="MyPTV 3D Kinematic (Python)",
    short_description="Python implementation of MyPTV's 3D kinematic prediction + Hungarian assignment",
    algorithm_summary=(
        "Predicts particle positions using polynomial velocity extrapolation from track history, "
        "then solves the frame-to-frame assignment with a radius-limited Hungarian algorithm. "
        "Supports configurable multi-term cost weights (distance, velocity, acceleration, intensity). "
        "Runs in pure Python, not Cython, so it is slower but fully customisable."
    ),
    algorithm_detail=(
        "Uses the openptv2.tracking_cost module for multi-term cost matrices and "
        "openptv2.plugins._assignment.match_within_radius for the sparse Hungarian solver. "
        "Tracks can persist through gaps (configurable max_gap frames) by extrapolating velocity. "
        "Reads input from rt_is.# files and writes ptv_is.# files in the canonical format."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=False,
    supports_postprocessing=False,
    supports_gap_relinking=True,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=True,
    speed_ranking="slow",
    density_ranking="moderate",
    accuracy_ranking="standard",
    parameters=(
        ParameterGuide(
            name="v_max",
            type="float",
            default="10.0",
            description="Maximum search radius for unseeded (new) tracks (mm).",
            how_to_choose="Set to max expected inter-frame displacement + margin.",
            typical_range="1 – 100",
            unit="mm",
        ),
        ParameterGuide(
            name="a_max",
            type="float",
            default="50.0",
            description="Maximum search radius for seeded tracks with velocity history (mm).",
            how_to_choose="Set to a_max = v_max + expected acceleration × dt.",
            typical_range="1 – 100",
            unit="mm",
        ),
        ParameterGuide(
            name="max_gap",
            type="int",
            default="2",
            description="Number of frames a track can be invisible before it is terminated.",
            how_to_choose="Match the expected occlusion length in your experiment.",
            typical_range="1 – 5",
            unit="frames",
        ),
        ParameterGuide(
            name="cost_weights",
            type="choice",
            default="distance only",
            description="Multi-term cost weights {w_distance, w_velocity, w_acceleration, w_intensity}.",
            how_to_choose="Distance-only for sparse data; add velocity/acceleration for dense or high-speed flows.",
            typical_range="(1,0,0,0) to (0.4,0.3,0.2,0.1)",
            unit="",
        ),
    ),
    default_preset="",
    best_for="When you need to customise the cost function (e.g., add intensity similarity) or compare against MyPTV reference results.",
    avoid_when="Large datasets where Python-speed tracking is too slow — use the Cython trackers instead.",
    typical_datasets="Benchmark comparisons, developing new cost terms, datasets with particle intensity data.",
)

MYPTV_2D_INFO = TrackerInfo(
    name="myptv_2d_tracking",
    display_name="MyPTV 2D Image-Space (Python)",
    short_description="Tracks particles independently per camera in 2D image space, then fuses across cameras",
    algorithm_summary=(
        "For each camera independently, links 2D blob detections across frames using velocity "
        "prediction + Hungarian assignment.  Then projects 2D tracks to 3D by multi-camera "
        "triangulation and resolves conflicts by consensus voting."
    ),
    algorithm_detail=(
        "Pure Python implementation.  Unlike all other openptv2 trackers which work in 3D space "
        "first, this tracker operates in 2D image coordinates per camera and fuses later."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=True,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=True,
    speed_ranking="slow",
    density_ranking="low_to_moderate",
    accuracy_ranking="standard",
    parameters=(),
    default_preset="",
    best_for="Datasets where 3D particles are unreliable (e.g., poor calibration) but 2D detections are clean.",
    avoid_when="Good calibration is available — 3D-space tracking is simpler and faster.",
    typical_datasets="Legacy MyPTV datasets, debugging 2D detection quality.",
)

SPLITTER_INFO = TrackerInfo(
    name="splitter_tracking",
    display_name="Splitter Quad-View Tracker",
    short_description="Forward tracking pipeline optimised for four-view image-splitter cameras",
    algorithm_summary=(
        "Runs the standard forward tracker with settings tuned for splitter-camera setups "
        "where a single sensor captures four quadrant views of the measurement volume."
    ),
    algorithm_detail=(
        "Same internal algorithm as standard_forward but with default parameters and data paths "
        "configured for splitter-frame sequences (4 cameras multiplexed into one image)."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=False,
    supports_multimedia=False,
    supports_splitter=True,
    supports_cost_weights=False,
    speed_ranking="fast",
    density_ranking="moderate",
    accuracy_ranking="standard",
    parameters=(),
    default_preset="",
    best_for="Datasets acquired with four-view image splitters on a single camera sensor.",
    avoid_when="Standard multi-camera setups — use hybrid_3d_corr or full_multipass instead.",
    typical_datasets="Splitter-based tomo-PTV experiments.",
)

PROPTV_INFO = TrackerInfo(
    name="proptv_tracking",
    display_name="proPTV Probabilistic (GMM)",
    short_description="Probabilistic PTV using Gaussian Mixture Model for smooth track approximation",
    algorithm_summary=(
        "Fits Gaussian basis functions to each track's time-position history and differentiates "
        "them analytically to obtain smooth velocity and acceleration estimates.  "
        "Predicts the next 3D position from that smooth GMM extrapolation and links it to a "
        "candidate by minimising a distance + velocity + acceleration continuity cost.  "
        "Runs entirely in 3D from the already-triangulated particles."
    ),
    algorithm_detail=(
        "Tracks in 3D directly from rt_is.# particles (no 2D re-triangulation).  Each track's "
        "history is smoothed with the proPTV GMM basis approximation to get a robust current "
        "velocity/acceleration; the next position is predicted and matched with a radius-limited "
        "Hungarian assignment using openptv2's multi-term cost function.  This adapts proPTV's "
        "probabilistic-smoothing concept to openptv2's own machinery rather than porting it."
    ),
    citation="Barta et al., Meas. Sci. Technol. (2024) — https://doi.org/10.1088/1361-6501/ad6e04",
    supports_backward=True,
    supports_new_particles=True,
    supports_2d=False,
    supports_postprocessing=False,
    supports_gap_relinking=True,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="moderate",
    density_ranking="low_to_moderate",
    accuracy_ranking="highest",
    parameters=(
        ParameterGuide(
            name="maxvel",
            type="float",
            default="20.0",
            description="Maximum absolute velocity for a track (mm/frame).",
            how_to_choose="Set based on max expected particle speed from a probe run.",
            typical_range="5 – 200",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="30",
            description="Maximum angle between successive velocity vectors (degrees).",
            how_to_choose="Lower for laminar flow, higher for turbulent.",
            typical_range="10 – 60",
            unit="deg",
        ),
        ParameterGuide(
            name="epsR",
            type="float",
            default="3.0",
            description="Search radius in image pixels for candidate 2D points.",
            how_to_choose="Set based on expected pixel displacement between frames.",
            typical_range="1.0 – 10.0",
            unit="px",
        ),
        ParameterGuide(
            name="t_init",
            type="int",
            default="4",
            description="Number of frames used for track initialisation before main tracking begins.",
            how_to_choose="More frames = better initial velocity estimates but later start of tracking.",
            typical_range="3 – 6",
            unit="frames",
        ),
        ParameterGuide(
            name="backtracking",
            type="bool",
            default="false",
            description="Enable backward tracking pass to recover broken tracks.",
            how_to_choose="Enable for higher trajectory completeness; adds ~2× runtime.",
            typical_range="",
            unit="",
        ),
        ParameterGuide(
            name="gaptracking",
            type="bool",
            default="false",
            description="Allow tracks to skip one missing frame and continue.",
            how_to_choose="Enable if particles occasionally disappear for one frame.",
            typical_range="",
            unit="",
        ),
    ),
    default_preset="",
    best_for="Datasets requiring smooth velocity/acceleration fields, irregular time steps, or where track smoothness is critical.",
    avoid_when="Very high particle density (>1000 particles/frame) — the GMM fitting overhead scales with track count.",
    typical_datasets="Biological flows, Lagrangian turbulence analysis, experiments with uneven frame spacing.",
)


def _build_registry() -> None:
    entries: list[TrackerInfo] = [
        HYBRID_INFO,
        FULL_MULTIPASS_INFO,
        FAST_INFO,
        STANDARD_FORWARD_INFO,
        TWO_DIRECTIONAL_INFO,
        MYPTV_3D_INFO,
        MYPTV_2D_INFO,
        SPLITTER_INFO,
        PROPTV_INFO,
    ]
    for info in entries:
        TRACKER_REGISTRY[info.name] = info


_build_registry()


def get_tracker_info(name: str) -> TrackerInfo | None:
    return TRACKER_REGISTRY.get(name)


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
        f"",
        f"  {info.short_description}",
        f"",
        f"  Algorithm:",
        f"    {info.algorithm_summary}",
    ]
    if info.algorithm_detail:
        lines.append(f"")
        lines.append(f"  Technical detail:")
        lines.append(f"    {info.algorithm_detail}")
    if info.citation:
        lines.append(f"")
        lines.append(f"  Citation: {info.citation}")

    lines.append(f"")
    lines.append(f"  Capabilities:")
    lines.append(f"    {'Backward pass':.<30} {'Yes' if info.supports_backward else 'No'}")
    lines.append(f"    {'New particles mid-seq':.<30} {'Yes' if info.supports_new_particles else 'No'}")
    lines.append(f"    {'2D target tracking':.<30} {'Yes' if info.supports_2d else 'No'}")
    lines.append(f"    {'Post-processing':.<30} {'Yes' if info.supports_postprocessing else 'No'}")
    lines.append(f"    {'Gap relinking':.<30} {'Yes' if info.supports_gap_relinking else 'No'}")
    lines.append(f"    {'Cost weights':.<30} {'Yes' if info.supports_cost_weights else 'No'}")

    lines.append(f"")
    lines.append(f"  Performance:")
    lines.append(f"    {'Speed':.<30} {info.speed_ranking}")
    lines.append(f"    {'Accuracy':.<30} {info.accuracy_ranking}")
    lines.append(f"    {'Density handling':.<30} {info.density_ranking}")

    if info.parameters:
        lines.append(f"")
        lines.append(f"  Parameters:")
        for p in info.parameters:
            lines.append(f"    {p.name}")
            lines.append(f"      Default: {p.default}  |  Range: {p.typical_range}  |  Unit: {p.unit}")
            lines.append(f"      {p.description}")
            if p.how_to_choose:
                lines.append(f"      Tuning: {p.how_to_choose}")

    if info.best_for:
        lines.append(f"")
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
