import pytest
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
def test_cavity_integration_dir():
    """Fixture to set up test_cavity_integration directory"""
    test_dir = TEST_DATA_ROOT / "test_cavity_integration"
    if not test_dir.exists():
        pytest.skip(f"Test data directory {test_dir} not found")
    return test_dir


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
