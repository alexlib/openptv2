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
import sys
from pathlib import Path
from types import ModuleType

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
    "openptv2_3d_smooth": "openptv2.plugins.fast_3d_smooth_tracking",
    "sg_hungarian_3d": "openptv2.plugins.fast_3d_smooth_tracking",
    "fast_3d_smooth": "openptv2.plugins.fast_3d_smooth_tracking",
    "nearest_hungarian_3d": "openptv2.plugins.nearest_hungarian_3d",
    "kalman_hungarian_3d": "openptv2.plugins.kalman_hungarian_3d",
    "myptv_3d_tracking": "openptv2.plugins.myptv_3d_tracking",
    "myptv_2d_tracking": "openptv2.plugins.myptv_2d_tracking",
    "predictive_gmm_3d": "openptv2.plugins.predictive_gmm_3d",
    "proptv_tracking": "openptv2.plugins.proptv_tracking",
    "proptv": "openptv2.plugins.proptv_tracking",
    "trackcorr": "openptv2.plugins.cython_epipolar_tracking",
    "full_multipass": "openptv2.plugins.cython_epipolar_tracking",
    "standard_forward": "openptv2.plugins.cython_epipolar_tracking",
    "two_directional": "openptv2.plugins.cython_epipolar_tracking",
    "splitter_tracking": "openptv2.plugins.cython_3d_tracking",
}

LEGACY_ALIASES = {
    "ext_sequence_splitter": "splitter_sequence",
    "ext_tracker_splitter": "splitter_tracking",
    "ext_sequence_contour": "contour_sequence",
    "ext_sequence_rembg": "rembg_sequence",
    "ext_sequence_rembg_contour": "rembg_contour_sequence",
    "fast": "priority_segment_3d",
    "fast_3d": "priority_segment_3d",
    "cython_3d": "cython_3d_tracking",
    "cython_epipolar": "cython_epipolar_tracking",
    "quality_3d": "kalman_hungarian_3d",
    "quality_3d_tracking": "kalman_hungarian_3d",
    "myptv_3d_tracking": "nearest_hungarian_3d",
    "proptv_tracking": "predictive_gmm_3d",
    "proptv": "predictive_gmm_3d",
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
        return ep.load()
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


def run_sequence_plugin(name: str, exp, plugins_dir: Path | None = None) -> None:
    """Instantiate and run a sequence plugin's ``Sequence.do_sequence()``."""
    if plugins_dir is None:
        plugins_dir = Path.cwd() / "plugins"

    module = resolve_plugin_module(name, BUILTIN_SEQUENCE_PLUGINS, plugins_dir)
    if not hasattr(module, "Sequence"):
        raise PluginError(f"Sequence plugin {name!r} has no Sequence class")

    from openptv2.gui import ptv as ptv_module

    plugin = module.Sequence(ptv=ptv_module, exp=exp)
    plugin.do_sequence()


def run_tracking_plugin(name: str, exp, plugins_dir: Path | None = None) -> None:
    """Instantiate and run a tracking plugin's ``Tracking.do_tracking()``."""
    if plugins_dir is None:
        plugins_dir = Path.cwd() / "plugins"

    module = resolve_plugin_module(name, BUILTIN_TRACKING_PLUGINS, plugins_dir)
    if not hasattr(module, "Tracking"):
        raise PluginError(f"Tracking plugin {name!r} has no Tracking class")

    from openptv2.gui import ptv as ptv_module

    plugin = module.Tracking(ptv=ptv_module, exp=exp)
    plugin.do_tracking()


def discover_available_plugins(plugins_dir: Path | str | None = None) -> dict:
    """Return the ``plugins:`` YAML section shape: built-ins plus whatever
    extra scripts sit in an experiment-local ``plugins/`` directory.
    """
    available_sequence = set(BUILTIN_SEQUENCE_PLUGINS)
    available_tracking = set(BUILTIN_TRACKING_PLUGINS)

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
