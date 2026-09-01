"""Tracking preset definitions, configuration mapping, and preset resolution."""

from enum import Enum
from typing import Any, Dict


class TrackingPreset(str, Enum):
    PRIORITY_SEGMENT_3D = (
        "priority_segment_3d"  # 3D Segment-Priority (Cython Engine - Default)
    )
    FAST_3D = "fast_3d"  # Alias for priority_segment_3d
    FAST = "fast"  # Alias for priority_segment_3d
    FULL_MULTIPASS = "full_multipass"  # Forward + Backward + Reciprocity Postprocessing
    TWO_DIRECTIONAL = "two_directional"  # Forward + Backward
    STANDARD_FORWARD = "standard_forward"  # Standard Forward Pass (2D+3D)
    CUSTOM_PLUGIN = "custom_plugin"  # Custom / Plugin Algorithm


PRESET_CHOICES = [
    (
        "priority_segment_3d",
        "OpenPTV Fast 3D (Default - Cython Engine)",
    ),
    ("fast_3d", "OpenPTV Fast 3D (Alias)"),
    ("fast", "OpenPTV Fast 3D (Alias)"),
    (
        "cython_epipolar_tracking",
        "OpenPTV Epipolar (Multi-Camera Cython)",
    ),
    (
        "full_multipass",
        "High Accuracy Multi-Pass (Forward + Backward + Reciprocity)",
    ),
    ("two_directional", "Two-Directional (Forward + Backward)"),
    ("standard_forward", "Standard Forward (2D + 3D)"),
    ("custom_plugin", "Custom / Plugin Algorithm"),
]

PRESET_MAP = dict(PRESET_CHOICES)
REVERSE_PRESET_MAP = {v: k for k, v in PRESET_CHOICES}

# ``postprocess`` on the three priority_segment_3d aliases below is ON. It
# used to be off, correctly: gap bridging was self-defeating (its cross-frame
# links were severed by the reciprocity pass that runs right after it) and it
# was handed ``dvxmax``, a velocity gate, as an acceleration-scale tolerance.
# Both are fixed (docs/plans/2026-08-16-tracking-next-steps.md §3.1, §3.2), and
# the pass is now a measured win on both ground-truth synthetic sets:
#
#   220 p/frame   yield 0.894 -> 0.936, mean track length  7.13 -> 10.61,
#                 precision 0.9667 -> 0.9596, wall 6.2s -> 7.5s
#   970 p/frame   yield 0.867 -> 0.889, mean track length  8.18 -> 11.04,
#                 precision 0.9157 -> 0.9059, wall 7.0s -> 9.9s
#
# The ~20-40% wall cost buys ~50% longer trajectories, which is what actually
# determines the quality of the Lagrangian velocity/acceleration statistics
# this project exists to produce; the <1 point of precision is a good trade
# for that. Set ``track.postprocess: false`` to opt out.
#
# NOT flipped for the 4BE preset: bridging *hurts* 4BE (§3.4) -- its fragments
# end at conflicts it deliberately declined, and the bridger re-creates
# exactly those links.
PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "priority_segment_3d": {
        "track_mode": 1,
        "flagNewParticles": False,
        "postprocess": True,
        "selected_tracking": "priority_segment_3d",
    },
    "fast": {
        "track_mode": 1,
        "flagNewParticles": False,
        "postprocess": True,
        "selected_tracking": "priority_segment_3d",
    },
    "fast_3d": {
        "track_mode": 1,
        "flagNewParticles": False,
        "postprocess": True,
        "selected_tracking": "priority_segment_3d",
    },
    "full_multipass": {
        "track_mode": 0,
        "flagNewParticles": True,
        "postprocess": True,
        "selected_tracking": "full_multipass",
    },
    "two_directional": {
        "track_mode": 0,
        "flagNewParticles": True,
        "postprocess": False,
        "selected_tracking": "two_directional",
    },
    "standard_forward": {
        "track_mode": 0,
        "flagNewParticles": True,
        "postprocess": False,
        "selected_tracking": "standard_forward",
    },
}


def infer_preset(
    track_params: Dict[str, Any], plugins_params: Dict[str, Any] | None = None
) -> str:
    """Infer preset / plugin pipeline name from track and plugins parameters."""
    selected_tracking = (plugins_params or {}).get("selected_tracking", "default")
    # Resolve aliases
    if selected_tracking in ("fast", "fast_3d", "priority_segment_3d"):
        return "priority_segment_3d"
    if selected_tracking in (
        "full_multipass",
        "two_directional",
        "standard_forward",
    ):
        return str(selected_tracking)
    if selected_tracking != "default":
        return TrackingPreset.CUSTOM_PLUGIN.value

    if "preset" in track_params and track_params["preset"]:
        p = str(track_params["preset"])
        if p in PRESET_MAP:
            return p

    track_mode = int(track_params.get("track_mode", 0))
    if track_mode == 1:
        return TrackingPreset.PRIORITY_SEGMENT_3D.value

    postprocess = bool(track_params.get("postprocess", False))
    if postprocess:
        return TrackingPreset.FULL_MULTIPASS.value

    return TrackingPreset.PRIORITY_SEGMENT_3D.value


def apply_preset(
    preset_name: str,
    track_params: Dict[str, Any],
    plugins_params: Dict[str, Any] | None = None,
    custom_plugin_name: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply tracking pipeline choice to track and plugins dicts while strictly preserving velocity search bounds."""
    track_params = dict(track_params)
    plugins_params = dict(plugins_params) if plugins_params else {}

    if preset_name in PRESET_CONFIGS:
        cfg = PRESET_CONFIGS[preset_name]
        track_params["track_mode"] = cfg["track_mode"]
        track_params["flagNewParticles"] = cfg["flagNewParticles"]
        track_params["postprocess"] = cfg["postprocess"]
        plugins_params["selected_tracking"] = preset_name
    elif preset_name == "splitter_tracking":
        plugins_params["selected_tracking"] = "splitter_tracking"
    elif preset_name == "custom_plugin":
        if custom_plugin_name:
            plugins_params["selected_tracking"] = custom_plugin_name
        elif "selected_tracking" not in plugins_params:
            plugins_params["selected_tracking"] = "default"
    else:
        plugins_params["selected_tracking"] = preset_name

    track_params["preset"] = plugins_params.get("selected_tracking", "default")

    return track_params, plugins_params


# ---------------------------------------------------------------------------
# Unified tracker picker: (tracker, direction, postprocess) as three
# orthogonal choices, replacing the old preset system's conflation of
# engine + direction + postprocess into named presets (full_multipass /
# standard_forward / two_directional all being the same "trackcorr" engine
# at different direction/postprocess settings). Old preset/plugin names
# above still work as inputs (infer_tracker/infer_direction understand
# them), so previously-saved YAMLs keep loading and running unchanged.
# ---------------------------------------------------------------------------

TRACKER_CHOICES = [
    ("priority_segment_3d", "OpenPTV Fast 3D (Default - Cython)"),
    ("4be", "OpenPTV 4BE (Four-Frame Best Estimate)"),
    ("trackcorr", "OpenPTV Epipolar (Multi-Camera 2D+3D)"),
    ("nearest_hungarian_3d", "MyPTV 3D (Nearest-Neighbor Hungarian)"),
    ("myptv_2d_tracking", "MyPTV 2D (Image-Space Assignment)"),
    ("predictive_gmm_3d", "proPTV (Predictive GMM - Optional)"),
    ("two_phase", "Two-Phase 3D+2D Leaf Ranking"),
]

DIRECTION_CHOICES = [
    ("forward", "Forward only"),
    ("forward_backward", "Forward + Backward"),
]

# Trackers whose engine supports a backward / bidirectional pass and accumulation.
TRACKER_SUPPORTS_BACKWARD: Dict[str, bool] = {
    "priority_segment_3d": True,
    "4be": True,
    "trackcorr": True,
    "nearest_hungarian_3d": True,
    "myptv_2d_tracking": True,
    "predictive_gmm_3d": True,
    "two_phase": False,
}

# Trackers whose do_tracking() supports post-processing (reciprocity + cold start + gap relinking).
TRACKER_SUPPORTS_POSTPROCESS = {
    "priority_segment_3d",
    "4be",
    "trackcorr",
    "nearest_hungarian_3d",
    "myptv_2d_tracking",
    "predictive_gmm_3d",
}

_LEGACY_TRACKER_ALIASES = {
    "default": "priority_segment_3d",
    "fast": "priority_segment_3d",
    "fast_3d": "priority_segment_3d",
    "cython_3d": "priority_segment_3d",
    "cython_3d_tracking": "priority_segment_3d",
    "cython_epipolar_tracking": "trackcorr",
    "cython_epipolar": "trackcorr",
    "openptv_epipolar": "trackcorr",
    "full_multipass": "trackcorr",
    "standard_forward": "trackcorr",
    "two_directional": "trackcorr",
    "splitter_tracking": "trackcorr",
    "myptv_3d_tracking": "nearest_hungarian_3d",
    "proptv_tracking": "predictive_gmm_3d",
    "proptv": "predictive_gmm_3d",
}

# Legacy preset names that ran forward+backward (used only when a saved
# track section has no explicit "direction" key yet).
_DIRECTION_BACKWARD_PRESETS = {"full_multipass", "two_directional"}


def infer_tracker(plugins_params: Dict[str, Any] | None) -> str:
    """Map any current or legacy ``selected_tracking`` value onto one of
    the canonical tracker keys. An unrecognised name (a custom
    experiment-local plugin) passes through unchanged."""
    selected = (plugins_params or {}).get("selected_tracking", "default")
    return _LEGACY_TRACKER_ALIASES.get(selected, selected)


def infer_direction(
    track_params: Dict[str, Any], plugins_params: Dict[str, Any] | None = None
) -> str:
    """Forward-only vs forward+backward, from ``track.run_backward``, ``track.backward``,
    or ``track.direction``, falling back to legacy preset names for old saved YAMLs."""
    if "run_backward" in track_params:
        return "forward_backward" if track_params["run_backward"] else "forward"
    if "backward" in track_params:
        return "forward_backward" if track_params["backward"] else "forward"
    if "direction" in track_params:
        return track_params["direction"]
    selected = (plugins_params or {}).get("selected_tracking", "default")
    if selected in _DIRECTION_BACKWARD_PRESETS:
        return "forward_backward"
    return "forward"


def apply_tracker(
    tracker: str,
    direction: str,
    postprocess: bool,
    track_params: Dict[str, Any],
    plugins_params: Dict[str, Any] | None = None,
    proptv_params: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Apply the (tracker, direction, postprocess) choice to the track /
    plugins / proptv parameter dicts, preserving kinematic search bounds."""
    track_params = dict(track_params)
    plugins_params = dict(plugins_params) if plugins_params else {}
    proptv_params = dict(proptv_params) if proptv_params else {}

    run_backward = direction == "forward_backward"

    if tracker == "priority_segment_3d":
        track_params["track_mode"] = 1
        track_params["flagNewParticles"] = False
        track_params["direction"] = direction
    elif tracker == "4be":
        track_params["track_mode"] = 1
        track_params["flagNewParticles"] = False
        track_params["direction"] = direction
    elif tracker == "trackcorr":
        track_params["track_mode"] = 0
        track_params["flagNewParticles"] = True
        track_params["direction"] = direction
    elif tracker == "predictive_gmm_3d":
        track_params["track_mode"] = 1
        track_params["flagNewParticles"] = True
        track_params["direction"] = direction
        proptv_params["backtracking"] = run_backward
    else:  # nearest_hungarian_3d, myptv_2d_tracking, etc.
        track_params["track_mode"] = 1
        track_params["flagNewParticles"] = True
        track_params["direction"] = direction

    track_params["run_backward"] = run_backward
    track_params["postprocess"] = bool(postprocess)
    plugins_params["selected_tracking"] = tracker
    track_params["preset"] = tracker

    return track_params, plugins_params, proptv_params


# ---------------------------------------------------------------------------
# Unified parameter TYPES across all 5 trackers: dvxmin/dvxmax/dvymin/dvymax/
# dvzmin/dvzmax/dacc/dangle, from the shared "track" YAML section, mm/frame
# (dv*), mm/frame^2 (dacc), gon (angle) -- trackcorr/priority_segment_3d's
# native units (dt is always 1 frame throughout openptv2, so mm/frame and
# mm/frame^2 are numerically what the isotropic trackers below also expect
# for their own v_max/a_max). nearest_hungarian_3d searches an isotropic
# sphere/radius rather than trackcorr's per-axis box, so it needs a single
# scalar bound; predictive_gmm_3d's own angle concept is in degrees, not gon.
# These two helpers are the ONE place that derives those isotropic/degree
# values from the shared per-axis/gon inputs, so every tracker reads the same
# parameter types with the same meaning.
# ---------------------------------------------------------------------------

GON_TO_DEG = 0.9  # 400 gon = 360 deg (gon is trackcorr's legacy convention)


def unified_velocity_bound(track_params: Dict[str, Any]) -> float:
    """Isotropic velocity-bound scalar derived from the full per-axis
    dvxmax/dvymax/dvzmax search box, for a tracker that searches a sphere
    rather than trackcorr's per-axis box. Falls back to dvxmax alone when
    the other axes aren't set (older saved parameter files)."""
    dvxmax = float(track_params.get("dvxmax", 10.0))
    dvymax = float(track_params.get("dvymax", dvxmax))
    dvzmax = float(track_params.get("dvzmax", dvxmax))
    return max(abs(dvxmax), abs(dvymax), abs(dvzmax))


def unified_angle_deg(track_params: Dict[str, Any], default_deg: float = 30.0) -> float:
    """track.angle is stored in gon (trackcorr/priority_segment_3d's legacy
    photogrammetry convention) -- convert to degrees for a tracker whose own
    algorithm compares against degrees directly."""
    if "angle" not in track_params:
        return default_deg
    return float(track_params["angle"]) * GON_TO_DEG
