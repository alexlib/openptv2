"""Tracking preset definitions, configuration mapping, and preset resolution."""

from enum import Enum
from typing import Any, Dict


class TrackingPreset(str, Enum):
    HYBRID_3D_CORR = "hybrid_3d_corr"  # Adaptive Hybrid Tracker (3D Kinematic + 2D Re-triangulation) [Recommended Default]
    FAST = "fast"  # Fast 3D Kinematic Tracker (Sub-millisecond)
    FAST_3D = "fast_3d"  # Fast 3D Kinematic Tracker (Alias)
    FULL_MULTIPASS = "full_multipass"  # Forward + Backward + Reciprocity Postprocessing
    TWO_DIRECTIONAL = "two_directional"  # Forward + Backward
    STANDARD_FORWARD = "standard_forward"  # Standard Forward Pass (2D+3D)
    CUSTOM_PLUGIN = "custom_plugin"  # Custom / Plugin Algorithm


PRESET_CHOICES = [
    (
        "hybrid_3d_corr",
        "Adaptive Hybrid Tracker (3D Kinematic + 2D Re-triangulation) [Recommended Default]",
    ),
    ("fast", "Fast 3D Kinematic Tracker (Sub-millisecond)"),
    ("fast_3d", "Fast 3D Kinematic Tracker (Alias)"),
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

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "hybrid_3d_corr": {
        "track_mode": 1,
        "flagNewParticles": True,
        "postprocess": False,
        "selected_tracking": "hybrid_3d_corr",
    },
    "fast": {
        "track_mode": 1,
        "flagNewParticles": False,
        "postprocess": False,
        "selected_tracking": "fast",
    },
    "fast_3d": {
        "track_mode": 1,
        "flagNewParticles": False,
        "postprocess": False,
        "selected_tracking": "fast_3d",
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
    if selected_tracking in (
        "hybrid_3d_corr",
        "fast",
        "fast_3d",
        "full_multipass",
        "two_directional",
        "standard_forward",
    ):
        return selected_tracking
    if selected_tracking != "default":
        return TrackingPreset.CUSTOM_PLUGIN.value

    if "preset" in track_params and track_params["preset"]:
        p = str(track_params["preset"])
        if p in PRESET_MAP:
            return p

    track_mode = int(track_params.get("track_mode", 0))
    if track_mode == 1:
        return TrackingPreset.HYBRID_3D_CORR.value

    postprocess = bool(track_params.get("postprocess", False))
    if postprocess:
        return TrackingPreset.FULL_MULTIPASS.value

    flag_new = bool(track_params.get("flagNewParticles", True))
    if flag_new:
        return TrackingPreset.HYBRID_3D_CORR.value

    return TrackingPreset.HYBRID_3D_CORR.value


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
