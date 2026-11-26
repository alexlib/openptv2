"""Resolve and run sequence/tracking plugins.

Single resolution path shared by the GUI (``openptv2.gui.ptv``) and the batch
pipeline (``openptv2.batch.pyptv_batch_plugins``). Resolution order:

1. Built-in plugins shipped in this package (``BUILTIN_SEQUENCE_PLUGINS`` /
   ``BUILTIN_TRACKING_PLUGINS``).
2. Third-party plugins registered via the ``openptv2.plugins`` entry-point
   group.
3. An experiment-local ``plugins/`` directory (defaults to
   ``<cwd>/plugins``), for one-off per-dataset scripts that don't warrant
   shipping in the package or a separate distribution.

Legacy ``ext_sequence_*`` / ``ext_tracker_*`` names (from when plugins lived
in per-experiment ``plugins/`` folders) still resolve, via ``LEGACY_ALIASES``.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

_log = logging.getLogger(__name__)

_DEPRECATED_TRACKING_PLUGINS = frozenset({
    "fast",
    "fast_3d",
    "priority_segment_3d",
    "openptv_fast_3d",
    "cython_3d",
    "cython_3d_tracking",
    "standard_forward",
    "two_directional",
    "four_be",
    "4be",
    "full_multipass",
    "trackcorr",
    "cython_epipolar",
    "cython_epipolar_tracking",
    "openptv_epipolar",
})

# Still resolvable by name (saved YAMLs, scripts), but not offered in pickers: use the kept trackers
# ``two_phase`` (default), ``trackcorr`` and ``fast_3d``; ``holistic_rounds`` > 0 adds the merged-blob loop.
RETIRED_TRACKING_PLUGINS = frozenset({
    "4be", "four_be", "hybrid_deltat_3d", "multi_deltat_3d", "splitter_tracking",
    "unified", "unified_tracking", "holistic",
})

BUILTIN_SEQUENCE_PLUGINS = {
    # "default" is the core algorithm wrapped in the plugin contract, so
    # every caller (GUI, batch) can always go through the same code path
    # instead of special-casing the string "default". Built-ins are tried
    # first in resolve_plugin_module, so this can never be shadowed by an
    # experiment-local plugins/default_sequence.py — deliberately.
    "default": "openptv2.plugins.default_sequence",
    "splitter_sequence": "openptv2.plugins.splitter_sequence",
    "contour_sequence": "openptv2.plugins.contour_sequence",
    "rembg_sequence": "openptv2.plugins.rembg_sequence",
    "rembg_contour_sequence": "openptv2.plugins.rembg_contour_sequence",
    "bubble_sequence": "openptv2.plugins.bubble_sequence",
    "bubble": "openptv2.plugins.bubble_sequence",
    "bubble_detection": "openptv2.plugins.bubble_sequence",
}

BUILTIN_TRACKING_PLUGINS = {
    "default": "openptv2.plugins.cython_3d_tracking",
    "priority_segment_3d": "openptv2.plugins.cython_3d_tracking",
    "openptv_fast_3d": "openptv2.plugins.cython_3d_tracking",
    "cython_3d_tracking": "openptv2.plugins.cython_3d_tracking",
    "cython_3d": "openptv2.plugins.cython_3d_tracking",
    "openptv_epipolar": "openptv2.plugins.cython_epipolar_tracking",
    "cython_epipolar_tracking": "openptv2.plugins.cython_epipolar_tracking",
    "cython_epipolar": "openptv2.plugins.cython_epipolar_tracking",
    "fast": "openptv2.plugins.cython_3d_tracking",
    "fast_3d": "openptv2.plugins.cython_3d_tracking",
    "4be": "openptv2.plugins.four_be_tracking",
    "four_be": "openptv2.plugins.four_be_tracking",
    "hybrid_deltat_3d": "openptv2.plugins.hybrid_deltat_3d",
    "multi_deltat_3d": "openptv2.plugins.hybrid_deltat_3d",
    "trackcorr": "openptv2.plugins.cython_epipolar_tracking",
    "full_multipass": "openptv2.plugins.cython_epipolar_tracking",
    "standard_forward": "openptv2.plugins.cython_epipolar_tracking",
    "two_directional": "openptv2.plugins.cython_epipolar_tracking",
    "splitter_tracking": "openptv2.plugins.cython_3d_tracking",
    "two_phase": "openptv2.plugins.two_phase_tracking",
    "holistic": "openptv2.plugins.holistic_tracking",
    "unified": "openptv2.plugins.unified_tracking",
    "unified_tracking": "openptv2.plugins.unified_tracking",
}

LEGACY_ALIASES = {
    "ext_sequence_splitter": "splitter_sequence",
    "ext_tracker_splitter": "splitter_tracking",
    "ext_sequence_contour": "contour_sequence",
    "ext_sequence_rembg": "rembg_sequence",
    "ext_sequence_rembg_contour": "rembg_contour_sequence",
    "ext_sequence_bubble": "bubble_sequence",
    "fast": "priority_segment_3d",
    "fast_3d": "priority_segment_3d",
    "cython_3d": "cython_3d_tracking",
    "cython_epipolar": "cython_epipolar_tracking",
}

ENTRY_POINT_GROUP = "openptv2.plugins"


class PluginError(RuntimeError):
    """Raised when a plugin cannot be resolved or fails to run."""


def _canonical_name(name: str) -> str:
    return LEGACY_ALIASES.get(name, name)


def _load_builtin(name: str, registry: dict[str, str]) -> ModuleType | None:
    module_path = registry.get(name)
    if module_path is None:
        return None
    return importlib.import_module(module_path)


def _load_entry_point(name: str) -> ModuleType | None:
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP, name=name):
        return cast(ModuleType | None, ep.load())
    return None


def _load_local(name: str, plugins_dir: Path) -> ModuleType | None:
    file_path = plugins_dir / f"{name}.py"
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_plugin_module(
    name: str, registry: dict[str, str], plugins_dir: Path | None = None
) -> ModuleType:
    """Resolve a plugin name to its module, built-ins first."""
    canonical = _canonical_name(name)
    if registry is BUILTIN_TRACKING_PLUGINS and name in RETIRED_TRACKING_PLUGINS:
        _log.warning(
            "Tracking plugin %r is retired; use 'two_phase' (optionally with track.holistic_rounds), "
            "'fast_3d' or 'trackcorr'.",
            name,
        )
    elif registry is BUILTIN_TRACKING_PLUGINS and (
        name in _DEPRECATED_TRACKING_PLUGINS
        or canonical in _DEPRECATED_TRACKING_PLUGINS
    ):
        _log.info(
            "Tracking plugin %r is deprecated; consider migrating to 'two_phase'.",
            name,
        )

    module = _load_builtin(canonical, registry)
    if module is not None:
        return module

    module = _load_entry_point(canonical)
    if module is not None:
        return module

    if plugins_dir is not None and plugins_dir.exists():
        module = _load_local(name, plugins_dir) or _load_local(canonical, plugins_dir)
        if module is not None:
            return module

    raise PluginError(
        f"Plugin {name!r} not found as a built-in, an 'openptv2.plugins' "
        f"entry point, or in {plugins_dir}"
    )


def run_sequence_plugin(name: str, exp: Any, plugins_dir: Path | None = None) -> None:
    """Instantiate and run a sequence plugin's ``Sequence.do_sequence()``."""
    if plugins_dir is None:
        plugins_dir = Path.cwd() / "plugins"

    module = resolve_plugin_module(name, BUILTIN_SEQUENCE_PLUGINS, plugins_dir)
    if not hasattr(module, "Sequence"):
        raise PluginError(f"Sequence plugin {name!r} has no Sequence class")

    from openptv2.gui import ptv as ptv_module

    plugin = module.Sequence(ptv=ptv_module, exp=exp)
    plugin.do_sequence()


# The three kept trackers and the module that runs each (docs/trackers.md).
TRACKER_MODULES = {
    "two_phase": "openptv2.plugins.two_phase_tracking",
    "trackcorr": "openptv2.plugins.default_tracking",
    "fast_3d": "openptv2.plugins.default_tracking",
}


def _experiment_params(exp: Any) -> dict:
    pm = getattr(exp, "pm", None)
    if pm is None and hasattr(exp, "exp1"):
        pm = getattr(exp.exp1, "pm", None)
    params = getattr(pm, "parameters", None)
    return params if isinstance(params, dict) else {}


def _maybe_align_store_4cam(exp: Any, setup: Any) -> None:
    """Align partial-camera (3-cam) points in ``exp``'s RunStore onto the 4-camera gauge
    before any tracker runs (eliminates 3-cam/4-cam calibration jumps for all trackers)."""
    if not getattr(setup, "align_4cam", True):
        return
    cals = getattr(exp, "cals", None)
    cpar = getattr(exp, "cpar", None)
    vpar = getattr(exp, "vpar", None)
    if cals is None or cpar is None or vpar is None or len(cals) < 4:
        return
    store = getattr(exp, "_store", None)
    if store is None:
        zarr_path = Path("res") / "run.zarr"
        if not zarr_path.exists():
            return
        try:
            from openptv2.storage import RunStore

            store = RunStore(zarr_path, mode="a")
        except Exception:
            return
    if not hasattr(store, "root") or not hasattr(store, "read_correspondences"):
        return
    spar = getattr(exp, "spar", None)
    if spar is not None and hasattr(spar, "get_first") and hasattr(spar, "get_last"):
        frames = list(range(int(spar.get_first()), int(spar.get_last()) + 1))
    elif hasattr(store, "frames"):
        frames = list(store.frames())
    else:
        return
    if not frames:
        return
    try:
        from openptv2.merge_refine import align_store_correspondences_4cam

        n_aligned = align_store_correspondences_4cam(store, frames, list(cals), cpar, vpar)
        if n_aligned:
            _log.info("align_4cam: aligned %d partial-camera points onto the 4-camera gauge", n_aligned)
    except Exception as exc:
        _log.debug("align_4cam skipped: %s", exc)


def run_tracking_plugin(name: str, exp: Any, plugins_dir: Path | None = None) -> None:
    """Run the tracker for this experiment.

    ``name`` is "default" (the YAML decides: ``track.tracker``, default ``two_phase``), one of
    ``two_phase`` / ``trackcorr`` / ``fast_3d``, an old tracker name (translated, with a
    warning), or the name of a custom plugin. See ``openptv2.tracking_setup``.
    """
    from openptv2.tracking_setup import resolve_tracking_setup

    if plugins_dir is None:
        plugins_dir = Path.cwd() / "plugins"

    params = _experiment_params(exp)
    setup = resolve_tracking_setup(params.get("track", {}), params.get("plugins", {}), name=name)

    if not setup.custom:
        for note in setup.notes:
            _log.warning("Tracking setup: %s", note)
        try:
            exp.tracking_setup = setup
        except AttributeError:
            pass
        module = importlib.import_module(TRACKER_MODULES[setup.tracker])
    else:
        module = resolve_plugin_module(name, BUILTIN_TRACKING_PLUGINS, plugins_dir)
    if not hasattr(module, "Tracking"):
        raise PluginError(f"Tracking plugin {name!r} has no Tracking class")

    _maybe_align_store_4cam(exp, setup)

    from openptv2.gui import ptv as ptv_module

    plugin = module.Tracking(ptv=ptv_module, exp=exp)
    plugin.do_tracking()


def discover_available_plugins(plugins_dir: Path | str | None = None) -> dict:
    """Return the ``plugins:`` YAML section shape: built-ins plus whatever
    extra scripts sit in an experiment-local ``plugins/`` directory.
    """
    available_sequence = set(BUILTIN_SEQUENCE_PLUGINS)
    # "default" = the YAML's `track.tracker`; then the three kept trackers; then custom plugins.
    available_tracking = {"default", *TRACKER_MODULES}

    if plugins_dir is not None:
        plugins_dir = Path(plugins_dir)
        if plugins_dir.exists() and plugins_dir.is_dir():
            for entry in plugins_dir.iterdir():
                if entry.is_file() and entry.suffix == ".py":
                    name = entry.stem
                    if "sequence" in name:
                        available_sequence.add(name)
                    if "track" in name:
                        available_tracking.add(name)

    return {
        "available_tracking": sorted(available_tracking),
        "available_sequence": sorted(available_sequence),
        "selected_tracking": "default",
        "selected_sequence": "default",
    }
