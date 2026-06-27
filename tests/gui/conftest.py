import pytest
import sys
import importlib

# Register pyptv package and its submodules as aliases to openptv2.gui in sys.modules
try:
    import openptv2
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

from pathlib import Path
import shutil

from openptv2.test_support import find_test_data_root


TEST_DATA_ROOT = find_test_data_root(Path(__file__))


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture to set up test data directory"""
    test_dir = TEST_DATA_ROOT / "test_cavity"
    if not test_dir.exists():
        pytest.skip(f"Test data directory {test_dir} not found")
    return test_dir


@pytest.fixture(scope="session")
def test_cavity_integration_dir(test_data_dir):
    """Backward-compatible alias to the canonical cavity test dataset."""
    return test_data_dir


# Note: Test data setup/cleanup is handled by root conftest.py's
# setup_and_cleanup_test_environment fixture (session-scoped autouse).
# This avoids duplicate session-scoped autouse fixtures that could
# cause order-dependent behavior.


def pytest_runtest_setup(item):
    if "qt" in item.keywords:
        try:
            import PySide6  # or PySide6, depending on your package
        except ImportError:
            pytest.skip("Skipping Qt-dependent test: Qt not available")
