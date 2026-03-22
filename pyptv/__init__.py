"""Compatibility wrapper exposing :mod:`gui.pyptv` as :mod:`pyptv`.

The test suite and older code paths still import modules such as
``pyptv.ptv`` and ``pyptv.parameter_manager``.  This shim keeps those imports
working while the actual implementation lives under ``gui.pyptv``.
"""

from __future__ import annotations

import pkgutil
import sys
from importlib import import_module

_REAL_PACKAGE = import_module("gui.pyptv")

# Re-export the public package attributes from gui.pyptv.
for _name, _value in _REAL_PACKAGE.__dict__.items():
    if not _name.startswith("_") or _name in {"__version__"}:
        globals()[_name] = _value


def _alias_submodule(module_name: str) -> None:
    module = import_module(f"gui.pyptv.{module_name}")
    sys.modules[f"{__name__}.{module_name}"] = module
    globals()[module_name] = module


for _module_info in pkgutil.iter_modules(_REAL_PACKAGE.__path__):
    try:
        _alias_submodule(_module_info.name)
    except Exception:
        # Some optional modules may pull in GUI-only dependencies. Leave them
        # to fail lazily if a test actually imports them.
        continue

__all__ = [name for name in globals() if not name.startswith("_")]
