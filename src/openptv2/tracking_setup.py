"""The tracker setup: three trackers, two switches, one add-on.

Read ``docs/trackers.md`` first. In short::

    track:
      tracker: two_phase      # trackcorr | fast_3d | two_phase      (default two_phase)
      direction: both         # forward | both                       (default both)
      postprocess: true       # reciprocity + gap bridging           (default true)
      holistic_rounds: 0      # merged-blob loop, 0 = off            (default 0)
      shared_correction: false  # corrects points in shared blobs    (default false)

This module is the ONE place that turns a saved ``track`` / ``plugins`` section
(new keys, or any old name or key) into those choices. The GUI, the batch
runner and the plugins all call :func:`resolve_tracking_setup`; nothing else
interprets tracker names or direction keys.

Old names keep working for one release: they are translated here and a note says
what they became (``TrackingSetup.notes``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Mapping

TRACKERS: tuple[str, ...] = ("two_phase", "trackcorr", "fast_3d")
DEFAULT_TRACKER = "two_phase"
DEFAULT_DIRECTION = "both"
DIRECTIONS: tuple[str, ...] = ("forward", "both")

TRACKER_LABELS: dict[str, str] = {
    "two_phase": "Two-phase (default: 3D candidates, image cost, confirmation)",
    "trackcorr": "Trackcorr (legacy algorithm: 2D search in the images)",
    "fast_3d": "Fast 3D (3D points only, fastest)",
}
DIRECTION_LABELS: dict[str, str] = {
    "forward": "Forward only",
    "both": "Forward + backward",
}

# Trackers that can run the merged-blob loop. Tried with fast_3d and rejected: it made the velocity error worse
# (0.1665 -> 0.1986) while the loop cost 100 s; trackcorr works on 2D targets (docs/plans/2026-11-21-tracker-setup.md).
HOLISTIC_TRACKERS: frozenset[str] = frozenset({"two_phase"})
# Trackers whose ``postprocess`` switch does something (two_phase carries gaps and
# confirms links inside its loop).
POSTPROCESS_TRACKERS: frozenset[str] = frozenset({"trackcorr", "fast_3d"})

# old name -> (tracker, direction the old name implied or None, epi_weight or None)
_FAST = ("fast_3d", "forward", None)
_LEGACY_NAMES: dict[str, tuple[str, str | None, float | None]] = {
    "priority_segment_3d": _FAST,
    "fast": _FAST,
    "cython_3d": _FAST,
    "cython_3d_tracking": _FAST,
    "openptv_fast_3d": _FAST,
    "splitter_tracking": _FAST,
    "cython_epipolar_tracking": ("trackcorr", "both", None),
    "cython_epipolar": ("trackcorr", "forward", None),
    "openptv_epipolar": ("trackcorr", "forward", None),
    "full_multipass": ("trackcorr", "both", None),
    "two_directional": ("trackcorr", "both", None),
    "standard_forward": ("trackcorr", "forward", None),
    # retired code, shown as the nearest kept tracker
    "4be": ("fast_3d", "forward", None),
    "four_be": ("fast_3d", "forward", None),
    "hybrid_deltat_3d": ("fast_3d", "forward", None),
    "multi_deltat_3d": ("fast_3d", "forward", None),
    "unified": ("two_phase", "forward", None),
    "unified_tracking": ("two_phase", "forward", None),
    "holistic": ("two_phase", "forward", None),
}
RETIRED_NAMES: frozenset[str] = frozenset({
    "4be", "four_be", "hybrid_deltat_3d", "multi_deltat_3d", "splitter_tracking",
    "unified", "unified_tracking", "holistic",
})


@dataclass(frozen=True)
class TrackingSetup:
    """The resolved tracker choice. ``tracker`` is one of :data:`TRACKERS`, or the
    name of a user plugin (``custom`` is then True and the other fields are unused)."""

    tracker: str = DEFAULT_TRACKER
    direction: str = DEFAULT_DIRECTION
    postprocess: bool = True
    holistic_rounds: int = 0
    shared_correction: bool = False   # two_phase only: re-estimate the points that touch a shared blob, link again
    epi_weight: float = 0.0           # fast_3d only: soft epipolar penalty, 0 = pure 3D
    accel_gate: bool = False          # fast_3d only: also refuse links beyond dacc / angle
    custom: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def both_directions(self) -> bool:
        return self.direction == "both"


def normalise_direction(value: Any) -> str | None:
    """'forward' | 'both' from any spelling (old: 'forward_backward', booleans); None if not usable."""
    if isinstance(value, bool):
        return "both" if value else "forward"
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("both", "forward_backward", "forward+backward", "bidirectional"):
        return "both"
    if text in ("forward", "forward_only"):
        return "forward"
    return None


def _old_direction_key(track: Mapping[str, Any]) -> str | None:
    """Direction from the new key, then the old spellings."""
    for key in ("direction", "run_backward", "backward", "bidirectional"):
        if key in track:
            found = normalise_direction(track[key])
            if found is not None:
                return found
    return None


def canonical_tracker_name(name: str | None) -> str | None:
    """The kept tracker for a name (new or old), or None for a custom plugin name or 'default'."""
    if not name or name == "default":
        return None
    if name in TRACKERS:
        return name
    if name in _LEGACY_NAMES:
        return _LEGACY_NAMES[name][0]
    return None


def resolve_tracking_setup(
    track_cfg: Mapping[str, Any] | None,
    plugins_cfg: Mapping[str, Any] | None = None,
    *,
    name: str | None = None,
) -> TrackingSetup:
    """Resolve the tracker, direction, postprocess, holistic_rounds from a saved config.

    ``name`` (optional) is an explicit tracker/plugin name from a command line or a
    caller; it wins over the YAML, as the batch flag always did. ``"default"`` or None
    means "whatever the YAML says", and an empty YAML means the documented defaults.
    """
    track = dict(track_cfg or {})
    plugins = dict(plugins_cfg or {})
    notes: list[str] = []

    # which name: explicit argument, then the new `track.tracker`, then the old plugin selection / preset
    selected: str | None = None
    for candidate in (name, track.get("tracker"), plugins.get("selected_tracking"), track.get("preset")):
        if candidate and str(candidate) != "default":
            if candidate is track.get("preset") and str(candidate) not in tuple(_LEGACY_NAMES) + TRACKERS:
                continue
            selected = str(candidate)
            break

    legacy_direction: str | None = None
    accel_gate = False
    epi_weight = 0.0
    legacy_rounds = 0

    if selected is None:
        tracker = DEFAULT_TRACKER
    elif selected in TRACKERS:
        tracker = selected
    elif selected in _LEGACY_NAMES:
        tracker, legacy_direction, _ = _LEGACY_NAMES[selected]
        shown = f"tracker {tracker}"
        if selected == "holistic":
            legacy_rounds = 2
            shown += ", holistic_rounds 2"
        notes.append(
            f"tracking name {selected!r} is retired; it now means {shown}"
            if selected in RETIRED_NAMES
            else f"tracking name {selected!r} is an old name for {shown}"
        )
    else:
        return TrackingSetup(tracker=selected, custom=True)

    # the old hybrid engine: `trackcorr` + engine 3d_hybrid was fast_3d with the epipolar penalty
    engine = track.get("trackcorr_engine")
    if tracker == "trackcorr" and engine == "3d_hybrid":
        tracker, epi_weight = "fast_3d", float(track.get("epi_weight", 1.0))
        legacy_direction = legacy_direction or "both"
        notes.append("trackcorr_engine '3d_hybrid' is now tracker fast_3d with epi_weight "
                     f"{epi_weight:g}")
    elif tracker == "trackcorr" and engine in (None, "epipolar") and selected in (
        "trackcorr", "full_multipass", "two_directional", "standard_forward", "cython_epipolar_tracking",
    ) and "trackcorr_engine" not in track:
        notes.append(
            "'trackcorr' now always means the legacy algorithm (2D search in the images). "
            "It used to run the faster 3D 'hybrid' unless trackcorr_engine: epipolar was set; "
            "to keep that behaviour use tracker: fast_3d with epi_weight: 1."
        )
    if tracker == "fast_3d":
        sub = track.get("fast_3d") if isinstance(track.get("fast_3d"), Mapping) else {}
        epi_weight = float(track.get("epi_weight", sub.get("epi_weight", epi_weight)))
        accel_gate = bool(track.get("accel_gate", sub.get("accel_gate", False)))

    # direction: the new key, then the old keys, then what an old name implied, then the default
    direction = _old_direction_key(track) or legacy_direction or DEFAULT_DIRECTION

    postprocess = bool(track.get("postprocess", True))

    rounds_raw = track.get("holistic_rounds", legacy_rounds)
    try:
        rounds = max(0, int(rounds_raw or 0))
    except (TypeError, ValueError):
        rounds = 0
    if rounds and tracker not in HOLISTIC_TRACKERS:
        notes.append(
            f"holistic_rounds {rounds} is ignored: the merged-blob loop runs with two_phase only "
            f"(tracker is {tracker})"
        )
        rounds = 0

    shared_correction = bool(track.get("shared_correction", False))
    if shared_correction and tracker not in HOLISTIC_TRACKERS:
        notes.append(f"shared_correction is ignored: it runs with two_phase only (tracker is {tracker})")
        shared_correction = False

    return TrackingSetup(
        tracker=tracker,
        direction=direction,
        postprocess=postprocess,
        holistic_rounds=rounds,
        shared_correction=shared_correction,
        epi_weight=epi_weight,
        accel_gate=accel_gate,
        notes=tuple(notes),
    )


def tracking_setup_for(exp: Any) -> TrackingSetup:
    """The setup of a running experiment: the one the plugin loader attached, else
    resolved from the experiment's parameters."""
    attached = getattr(exp, "tracking_setup", None)
    if isinstance(attached, TrackingSetup):
        return attached
    pm = getattr(exp, "pm", None)
    if pm is None and hasattr(exp, "exp1"):
        pm = getattr(exp.exp1, "pm", None)
    params = getattr(pm, "parameters", None) or {}
    return resolve_tracking_setup(params.get("track", {}), params.get("plugins", {}))


def apply_tracking_setup(
    setup: TrackingSetup,
    track_params: Mapping[str, Any],
    plugins_params: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write a setup into the ``track`` / ``plugins`` dicts in the new key spelling and drop
    the old direction keys, so a saved YAML has one way of saying each thing."""
    track = dict(track_params)
    plugins = dict(plugins_params or {})
    for old in ("run_backward", "backward", "bidirectional", "preset", "trackcorr_engine"):
        track.pop(old, None)
    track["tracker"] = setup.tracker
    track["direction"] = setup.direction
    track["postprocess"] = bool(setup.postprocess)
    track["holistic_rounds"] = int(setup.holistic_rounds)
    track["shared_correction"] = bool(setup.shared_correction)
    if setup.tracker == "fast_3d":
        track["epi_weight"] = float(setup.epi_weight)
        track["accel_gate"] = bool(setup.accel_gate)
    # keep the plugin selection in step with the tracker (older readers look at it)
    plugins["selected_tracking"] = setup.tracker
    # the engine flags the kernels still read
    if setup.tracker == "trackcorr":
        track["track_mode"], track["flagNewParticles"] = 0, True
    elif setup.tracker == "fast_3d":
        track["track_mode"], track["flagNewParticles"] = 1, False
    else:
        track["track_mode"], track["flagNewParticles"] = 1, True
    return track, plugins


def add_tracker_arguments(parser: Any) -> None:
    """The batch command-line flags for the setup (same words as the YAML keys)."""
    parser.add_argument(
        "--tracker", "--tracking-plugin", dest="tracking_plugin", default=None,
        help="two_phase | trackcorr | fast_3d (or a custom plugin name). Default: track.tracker in the "
             "YAML, which defaults to two_phase. Old names are translated.",
    )
    parser.add_argument(
        "--direction", choices=DIRECTIONS, default=None,
        help="forward | both. Default: track.direction in the YAML, which defaults to both.",
    )
    parser.add_argument(
        "--holistic-rounds", type=int, default=None, metavar="N",
        help="Merged-blob loop rounds (two_phase only). Default: track.holistic_rounds, which defaults to 0.",
    )
    parser.add_argument(
        "--shared-correction", dest="shared_correction", action=argparse.BooleanOptionalAction,
        default=None,
        help="Re-estimate points that touch a shared blob from their tracks and link again (two_phase only). "
             "Default: track.shared_correction, which defaults to false.",
    )


def tracker_overrides_from_argv(argv: list[str] | None = None) -> dict[str, Any]:
    """``{"direction": ..., "holistic_rounds": ...}`` for the flags that were given (others absent)."""
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    add_tracker_arguments(parser)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    out: dict[str, Any] = {}
    if known.direction is not None:
        out["direction"] = known.direction
    if known.holistic_rounds is not None:
        out["holistic_rounds"] = int(known.holistic_rounds)
    if known.shared_correction is not None:
        out["shared_correction"] = bool(known.shared_correction)
    return out
