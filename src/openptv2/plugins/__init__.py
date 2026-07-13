"""Sequence/tracking plugin system for openptv2.

Built-in plugins ship as modules in this package. Third-party plugins
register via the ``openptv2.plugins`` entry-point group. An experiment-local
``plugins/`` directory (next to the parameters file) remains a supported
override for one-off, per-dataset scripts — resolved last, after built-ins
and entry points.

Plugins are intentionally pure Python, not Cython — they are the
extensibility surface (I/O and glue around the compiled ``algorithms/``
kernels), and compiling them would work against that.
"""

from .loader import (
    BUILTIN_SEQUENCE_PLUGINS,
    BUILTIN_TRACKING_PLUGINS,
    LEGACY_ALIASES,
    PluginError,
    discover_available_plugins,
    resolve_plugin_module,
    run_sequence_plugin,
    run_tracking_plugin,
)

__all__ = [
    "BUILTIN_SEQUENCE_PLUGINS",
    "BUILTIN_TRACKING_PLUGINS",
    "LEGACY_ALIASES",
    "PluginError",
    "discover_available_plugins",
    "resolve_plugin_module",
    "run_sequence_plugin",
    "run_tracking_plugin",
]
