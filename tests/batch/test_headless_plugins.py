"""Guard: the whole batch/plugin execution path must stay importable without
any GUI toolkit. This is the property that lets a GUI-tuned YAML run in a
slim cloud container (headless install profile, see docs/cloud-batch.md).
"""

import subprocess
import sys

GUI_PACKAGES = ("PySide6", "chaco", "enable", "traitsui", "pyface", "matplotlib")

_PROBE = f"""
import sys
from openptv2.plugins import (
    resolve_plugin_module,
    BUILTIN_SEQUENCE_PLUGINS,
    BUILTIN_TRACKING_PLUGINS,
)
for name in BUILTIN_SEQUENCE_PLUGINS:
    resolve_plugin_module(name, BUILTIN_SEQUENCE_PLUGINS)
for name in BUILTIN_TRACKING_PLUGINS:
    resolve_plugin_module(name, BUILTIN_TRACKING_PLUGINS)
import openptv2.gui.ptv
import openptv2.batch.pyptv_batch
import openptv2.batch.pyptv_batch_parallel
bad = sorted({{m.split('.')[0] for m in sys.modules
              if m.split('.')[0] in {GUI_PACKAGES!r}}})
assert not bad, f"GUI packages imported on the headless plugin path: {{bad}}"
print("headless OK")
"""


def test_plugin_and_batch_path_is_headless():
    """Resolving every built-in plugin and importing both batch runners must
    not pull in a GUI toolkit (run in a subprocess for a clean sys.modules).
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "headless OK" in result.stdout
