import pytest
import sys
import importlib
from pathlib import Path

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
    import openptv2
    import openptv2.gui as _gui
    import openptv2.gui.pyptv as _pyptv_base
    sys.modules["pyptv"] = _pyptv_base
    
    for sub, target in _gui.submodule_mapping.items():
        try:
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


@pytest.fixture(scope="session")
def small_dir():
    """Path to the test_cavity_small dataset."""
    d = TEST_DATA_ROOT / "test_cavity_small"
    if not d.exists():
        pytest.skip(f"test_cavity_small not found at {d}")
    return d


@pytest.fixture(scope="session")
def small_yaml(small_dir):
    y = small_dir / "parameters_Run1.yaml"
    if not y.exists():
        pytest.skip(f"parameters_Run1.yaml not found in {small_dir}")
    return y
