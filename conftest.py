"""
Root conftest.py for openptv2 tests.

This module provides session-scoped fixtures for test data setup and cleanup.
"""




import pytest
import sys
import importlib

# Register optv package and its submodules as aliases in sys.modules for legacy compatibility
try:
    import openptv2
    sys.modules["optv"] = openptv2
    for sub in ["correspondences", "tracker", "orientation", "calibration", "parameters", "imgcoord"]:
        try:
            mod = importlib.import_module(f"openptv2.{sub}")
            sys.modules[f"optv.{sub}"] = mod
        except ImportError:
            pass
except ImportError:
    pass

# Register pyptv package and its submodules as aliases in sys.modules
try:
    import openptv2.gui as _gui
    import openptv2.gui.pyptv as _pyptv_base
    sys.modules["pyptv"] = _pyptv_base
    
    for sub, target in _gui.submodule_mapping.items():
        try:
            # Map the alias directly using the registered lazy shim
            shim = sys.modules[f"openptv2.gui.pyptv.{sub}"]
            sys.modules[f"pyptv.{sub}"] = shim
        except KeyError:
            try:
                mod = importlib.import_module(target)
                sys.modules[f"pyptv.{sub}"] = mod
                sys.modules[f"openptv2.gui.pyptv.{sub}"] = mod
            except ImportError:
                pass
except ImportError:
    pass


# Add any fixtures needed for the current pure-NumPy/algorithms tests below.


