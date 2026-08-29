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
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data) — compute from probe data.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.0",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data) — compute from probe data.",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation (gon).",
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
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

PRIORITY_SEGMENT_3D_INFO = TrackerInfo(
    name="priority_segment_3d",
    display_name="3D Segment-Priority (Cython Engine - Default)",
    short_description="Single-pass 3D tracking using 4-level acceleration-priority segment linking",
    algorithm_summary=(
        "Links 3D particles using a 4-level cascade: Level 1 claims high-confidence particles "
        "meeting an acceleration threshold (dacc) globally across all tracks in ascending cost order. "
        "Level 2 uses local neighbor velocity averaging. Level 3 handles static/unseeded displacement. "
        "Executed in single-pass compiled Cython for maximum throughput."
    ),
    algorithm_detail=(
        "Calls track3d_loop_fast in track_kernels_track3d.py (compiled Cython). "
        "Global cost-ordered claiming within each level prevents index-order claim bias. "
        "This is the authoritative default 3D tracking engine in OpenPTV2."
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
    density_ranking="low_to_moderate",
    accuracy_ranking="high",
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
            description=(
                "Seeded-step search box (mm): the radius searched around the "
                "velocity-extrapolated prediction for a particle that is "
                "already being tracked. Named for an acceleration bound, but "
                "it is used as a position tolerance."
            ),
            how_to_choose=(
                "About 0.6 x dvxmax -- NOT 'just above max observed "
                "acceleration', and not equal to dvxmax. This is the knob "
                "that controls tracking of particles that already have a "
                "velocity; dvxmax only widens the cold search for particles "
                "that do not. Setting dacc = dvxmax makes the tracker "
                "bit-identical to the C original and measures worst of every "
                "value tried at high seeding density. Lower it further "
                "(~0.4 x) when densely seeded, raise it (~0.8 x) when sparse."
            ),
            typical_range="0.5 – 50",
            unit="mm (position tolerance, despite the name)",
        ),
    ),
    default_preset="priority_segment_3d",
    best_for="Primary 3D particle tracking on sparse to moderate density flows.",
    avoid_when="Particles appear/disappear frequently mid-sequence or extreme optical occlusions require multi-frame gap bridging.",
    typical_datasets="3D PTV benchmark sequences, turbulent flow sequences.",
)

FOUR_BE_INFO = TrackerInfo(
    name="4be",
    display_name="4BE (Four-frame Best Estimate, Cython Engine)",
    short_description="Pure 3D linking scored by how well a candidate predicts a real particle two frames ahead",
    algorithm_summary=(
        "Ouellette, Xu & Bodenschatz (2006)'s four-frame best-estimate cost: for each "
        "frame n -> n+1 candidate found within the velocity search box, extrapolate a "
        "further constant-velocity estimate into frame n+2 and score the candidate by "
        "how close a REAL particle sits to that n+2 estimate (support distance). Cost is "
        "that support distance plus how well the candidate itself matched the frame n+1 "
        "prediction -- summed, not support-distance-alone, since a candidate can no "
        "longer win purely because a coincidental real particle happens to sit near its "
        "own bad n+2 extrapolation while grossly failing the n+1 match (see "
        "docs/holistic-3d-ptv-systems-research-program.md's 2026-08-18 case study)."
    ),
    algorithm_detail=(
        "Calls track4be_loop_fast in track_kernels_track3d.py (compiled Cython). Give-up "
        "on conflict (the paper's rule, greedy_conflicts=0) by default; an unsupported "
        "candidate (nothing real near its n+2 estimate) falls back to the 3MA "
        "acceleration residual rather than being rejected outright (strict_support=0), "
        "which recovers yield lost to genuine 1-frame detection gaps. strict_support and "
        "greedy_conflicts are module-level constants in track4be.py, not exposed via "
        "track.par -- editing the source is currently the only way to change them."
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
    density_ranking="low_to_moderate",
    accuracy_ranking="standard",
    parameters=(
        ParameterGuide(
            name="dvxmin / dvxmax (and dvy../dvz..)",
            type="float",
            default="±15.5",
            description="Velocity search window, per axis (mm/frame) -- same track.par "
                         "fields priority_segment_3d/trackcorr use.",
            how_to_choose="Set just above max observed displacement, same guidance as "
                          "priority_segment_3d -- there is no 4BE-specific tuning here.",
            typical_range="1 - 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.5",
            description="Unused directly by 4BE's own cost (which is a distance sum, not "
                         "an acceleration bound) -- retained for API parity with "
                         "priority_segment_3d/trackcorr; the candidate search box itself "
                         "is dvxmax/dvymax/dvzmax.",
            how_to_choose="Leave at the same value used for priority_segment_3d on this "
                          "dataset; it does not change 4BE's own linking decisions.",
            typical_range="0.5 - 50",
            unit="mm",
        ),
    ),
    default_preset="4be",
    best_for="Sparse-to-moderate density flows where a real particle usually exists two "
              "frames ahead to disambiguate close candidates.",
    avoid_when="High density or high-noise data (see the case study above) -- more "
               "candidates means more chances for a coincidental n+2 match to compete "
               "with the correct one, even with the summed cost.",
    typical_datasets="3D PTV benchmark sequences, turbulent flow sequences.",
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
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.0",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation (gon).",
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
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
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="dacc",
            type="float",
            default="5.0",
            description="Maximum acceleration (mm/frame²).",
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
            typical_range="0.5 – 50",
            unit="mm/frame²",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="120",
            description="Maximum angular deviation (gon).",
            how_to_choose="Same parameter guidance as priority_segment_3d (from probe data).",
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
    name="nearest_hungarian_3d",
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
            how_to_choose=(
                "Not a separate field -- derived from the same "
                "track.dvxmax/dvymax/dvzmax as every other tracker via "
                "tracking_presets.unified_velocity_bound() (largest of the "
                "three; this tracker searches an isotropic radius, not "
                "trackcorr's per-axis box). Set to max expected inter-frame "
                "displacement + margin."
            ),
            typical_range="1 – 100",
            unit="mm",
        ),
        ParameterGuide(
            name="a_max",
            type="float",
            default="50.0",
            description="Maximum search radius for seeded tracks with velocity history (mm).",
            how_to_choose="Same track.dacc field trackcorr/priority_segment_3d use. Set to a_max = v_max + expected acceleration × dt.",
            typical_range="1 – 100",
            unit="mm",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="45",
            description=(
                "Cone-of-continuity filter for seeded tracks: rejects a candidate "
                "whose implied velocity direction breaks continuity by more than "
                "this angle from the track's established direction."
            ),
            how_to_choose=(
                "track.angle, same field trackcorr/priority_segment_3d use, but "
                "in GON there (400 gon = 360 deg) -- converted to degrees for "
                "this tracker's own angle comparison via "
                "tracking_presets.unified_angle_deg(). Lower for smooth/laminar "
                "flow, higher for turbulent."
            ),
            typical_range="20 – 90",
            unit="deg (source field is gon)",
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
    avoid_when="Standard multi-camera setups — use priority_segment_3d or full_multipass instead.",
    typical_datasets="Splitter-based tomo-PTV experiments.",
)

PROPTV_INFO = TrackerInfo(
    name="predictive_gmm_3d",
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
            how_to_choose=(
                "Defaults from the same track.dvxmax/dvymax/dvzmax as every "
                "other tracker (tracking_presets.unified_velocity_bound()); "
                "set proptv.maxvel explicitly only to override that default "
                "for this tracker specifically. Set based on max expected "
                "particle speed from a probe run."
            ),
            typical_range="5 – 200",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="angle",
            type="float",
            default="30",
            description="Maximum angle between successive velocity vectors (degrees).",
            how_to_choose=(
                "Defaults from track.angle converted from GON to degrees "
                "(tracking_presets.unified_angle_deg() -- trackcorr/"
                "priority_segment_3d's angle field is in gon, 400 gon = 360 "
                "deg; this tracker's own angle comparison is in degrees). Set "
                "proptv.angle explicitly to override. Lower for laminar flow, "
                "higher for turbulent."
            ),
            typical_range="10 – 60",
            unit="deg (source field is gon)",
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


HYBRID_DELTAT_3D_INFO = TrackerInfo(
    name="hybrid_deltat_3d",
    display_name="Hybrid Multi-Delta-t (Python)",
    short_description=(
        "Coarse-to-fine tracking for slow flows: link every N-th frame first, "
        "then refine through the intermediate frames"
    ),
    algorithm_summary=(
        "For datasets where per-frame displacement is smaller than the 3D "
        "reconstruction noise (poorly-conditioned), links particle clouds "
        "across a larger time step N first -- displacement grows N-fold while "
        "the noise floor stays put -- then re-walks the intermediate frames, "
        "attaching detections that lie within refine_gate_mm of a cubic-Hermite "
        "prediction built from the coarse segment's endpoints."
    ),
    algorithm_detail=(
        "Coarse pass uses the predictive Hungarian tracker (MyPTV3DTracker) on "
        "the strided clouds with stride-scaled search radii; the refine pass "
        "chains consecutive-frame detections along each segment. Chains break "
        "where no intermediate detection fits the prediction; postptv "
        "stitching can re-glue those small gaps downstream."
    ),
    citation=(
        "Concept: multi-step / variable-Delta-t Lagrangian PTV for slow flows "
        "(cf. Cierpka et al., volumetric PTV via multi-frame tracking)."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=False,
    supports_postprocessing=False,
    supports_gap_relinking=True,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=False,
    speed_ranking="moderate",
    density_ranking="low_to_moderate",
    accuracy_ranking="high",
    parameters=(
        ParameterGuide(
            name="stride",
            type="int",
            default="5",
            description="Coarse-pass step N: link frames i and i+N before refining.",
            how_to_choose=(
                "Pick N so displacement over N frames exceeds the z-noise floor "
                "reported by the tracking-conditioning diagnostic (ratio < ~1): "
                "N >= z_noise_mm / per_frame_displacement_mm. Cap N at roughly "
                "half the median inter-particle spacing divided by the "
                "per-frame displacement so the coarse search stays unambiguous."
            ),
            typical_range="2 – 20",
            unit="frames",
        ),
        ParameterGuide(
            name="refine_gate_mm",
            type="float",
            default="0.8",
            description=(
                "Maximum distance between a Hermite-predicted intermediate "
                "position and a detection for attachment."
            ),
            how_to_choose=(
                "About 1-2x the reconstruction noise floor from the conditioning "
                "report. Too small: chains fragment at every noisy frame. Too "
                "large: wrong particles get attached in dense regions."
            ),
            typical_range="0.1 – 5",
            unit="mm",
        ),
    ),
    default_preset="",
    best_for=(
        "High-frame-rate recordings of slow flow (z-noise/motion ratio > 1) "
        "where standard trackers produce only 2-6 frame fragments."
    ),
    avoid_when=(
        "Well-conditioned data -- use priority_segment_3d or the Cython default; "
        "very high particle density where coarse-step ambiguity is high."
    ),
    typical_datasets="High-speed camera PTV of slow physiological / micro flows.",
)


TWO_PHASE_INFO = TrackerInfo(
    name="two_phase",
    display_name="Two-Phase 3D+2D Leaf Ranking",
    short_description="3D KD-tree candidate search refined by 2D per-camera leaf-distance Hungarian assignment",
    algorithm_summary=(
        "Phase 1: a 3D KD-tree finds every frame-(t+1) particle within a velocity-derived "
        "search radius of each frame-t particle. Phase 2: within each connected component of "
        "candidates, per-camera 2D pixel distances ('leaf' positions) become a cost matrix and "
        "Hungarian assignment picks the globally best match. Exploits the tree-forest storage "
        "layout: 3D positions are the structural 'trunk', 2D leaf positions the disambiguating "
        "'signature' when 3D candidates are ambiguous or noisy."
    ),
    algorithm_detail=(
        "Pure Python/NumPy/SciPy (cKDTree + scipy.sparse.csgraph.connected_components + "
        "linear_sum_assignment). Implemented in plugins/two_phase_tracking.py "
        "(TwoPhaseTracker/TwoPhaseTrackerConfig). Falls back to pure 3D matching if no leaf "
        "features are supplied (leaf_weight=0)."
    ),
    supports_backward=False,
    supports_new_particles=True,
    supports_2d=True,
    supports_postprocessing=False,
    supports_gap_relinking=True,
    supports_multimedia=False,
    supports_splitter=False,
    supports_cost_weights=True,
    speed_ranking="fast",
    density_ranking="moderate",
    accuracy_ranking="standard",
    parameters=(
        ParameterGuide(
            name="v_max",
            type="float",
            default="5.0",
            description="Maximum velocity (mm/frame); 3D KD-tree search radius for phase 1.",
            how_to_choose="Set just above max observed displacement, same guidance as other trackers' velocity bound.",
            typical_range="1 – 100",
            unit="mm/frame",
        ),
        ParameterGuide(
            name="leaf_weight",
            type="float",
            default="1.0",
            description="Weight of 2D per-camera leaf distances in the phase-2 cost matrix. 0 falls back to pure 3D matching.",
            how_to_choose="Raise when 3D positions are noisy/ambiguous (e.g. weak calibration) but 2D detections are clean; 0 to disable.",
            typical_range="0 – 2",
            unit="",
        ),
        ParameterGuide(
            name="max_gap",
            type="int",
            default="2",
            description="Number of frames a track can go unmatched before being terminated.",
            how_to_choose="Match expected occlusion length in the experiment.",
            typical_range="1 – 5",
            unit="frames",
        ),
        ParameterGuide(
            name="dt",
            type="float",
            default="1.0",
            description="Time step between frames, used for velocity computation.",
            how_to_choose="Leave at 1.0 unless frames are unevenly spaced.",
            typical_range="",
            unit="",
        ),
    ),
    default_preset="",
    best_for="Datasets where 3D triangulation is ambiguous or noisy (weak calibration, dense scenes) but per-camera 2D detections are clean enough to disambiguate.",
    avoid_when="Well-conditioned 3D data where the extra 2D leaf-ranking pass adds cost without improving on 3D-only trackers, or when a backward pass / post-processing is required.",
    typical_datasets="Tree-forest-storage experiments with per-camera leaf/2D data retained alongside 3D positions.",
)


def _build_registry() -> None:
    entries: list[TrackerInfo] = [
        PRIORITY_SEGMENT_3D_INFO,
        FOUR_BE_INFO,
        FULL_MULTIPASS_INFO,
        STANDARD_FORWARD_INFO,
        TWO_DIRECTIONAL_INFO,
        MYPTV_3D_INFO,
        MYPTV_2D_INFO,
        SPLITTER_INFO,
        PROPTV_INFO,
        HYBRID_DELTAT_3D_INFO,
        TWO_PHASE_INFO,
    ]
    for info in entries:
        TRACKER_REGISTRY[info.name] = info

    # Register legacy aliases
    TRACKER_REGISTRY["fast_3d"] = PRIORITY_SEGMENT_3D_INFO
    TRACKER_REGISTRY["fast"] = PRIORITY_SEGMENT_3D_INFO
    TRACKER_REGISTRY["myptv_3d_tracking"] = MYPTV_3D_INFO
    TRACKER_REGISTRY["proptv_tracking"] = PROPTV_INFO
    TRACKER_REGISTRY["trackcorr"] = FULL_MULTIPASS_INFO
    TRACKER_REGISTRY["multi_deltat_3d"] = HYBRID_DELTAT_3D_INFO


_build_registry()


def get_tracker_info(name: str) -> TrackerInfo | None:
    # Handle alias lookups
    alias_map = {
        "fast_3d": "priority_segment_3d",
        "fast": "priority_segment_3d",
        "myptv_3d_tracking": "nearest_hungarian_3d",
        "proptv_tracking": "predictive_gmm_3d",
        "multi_deltat_3d": "hybrid_deltat_3d",
    }
    key = alias_map.get(name, name)
    return TRACKER_REGISTRY.get(key)


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
    lines.append(f"    {'Backward pass':.<30} {'Yes' if info.supports_backward else 'No'}")
    lines.append(f"    {'New particles mid-seq':.<30} {'Yes' if info.supports_new_particles else 'No'}")
    lines.append(f"    {'2D target tracking':.<30} {'Yes' if info.supports_2d else 'No'}")
    lines.append(f"    {'Post-processing':.<30} {'Yes' if info.supports_postprocessing else 'No'}")
    lines.append(f"    {'Gap relinking':.<30} {'Yes' if info.supports_gap_relinking else 'No'}")
    lines.append(f"    {'Cost weights':.<30} {'Yes' if info.supports_cost_weights else 'No'}")

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
            lines.append(f"      Default: {p.default}  |  Range: {p.typical_range}  |  Unit: {p.unit}")
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

