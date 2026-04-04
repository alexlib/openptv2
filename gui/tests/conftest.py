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


@pytest.fixture(scope="session", autouse=True)
def clean_test_environment(test_data_dir):
    """Clean up test environment before and after tests"""
    import sys

    temp_patterns = ["tmp*.yaml", "tmp*.txt", "*.yaml.bak", "*_summary.csv"]

    def cleanup_temp_files(directory=None):
        target = directory or test_data_dir
        for pattern in temp_patterns:
            for f in target.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass

    # Clean up any existing test results
    results_dir = test_data_dir / "res"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Copy res_orig to res (not just create empty dir)
    res_orig = test_data_dir / "res_orig"
    if res_orig.exists():
        shutil.copytree(res_orig, results_dir)
        print(f"[FIXTURE] Copied {res_orig} to {results_dir}", file=sys.stderr)
    else:
        results_dir.mkdir(exist_ok=True)
        print(f"[FIXTURE] Created empty {results_dir}", file=sys.stderr)

    # Clean up temporary files before test
    cleanup_temp_files()

    # Also clean test_cavity_integration
    integration_dir = TEST_DATA_ROOT / "test_cavity_integration"
    if integration_dir.exists():
        integration_res = integration_dir / "res"
        if integration_res.exists():
            shutil.rmtree(integration_res)
        cleanup_temp_files(integration_dir)

    yield

    # Cleanup after tests
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Clean up temporary files after test
    cleanup_temp_files()

    # Also clean test_cavity_integration after tests
    if integration_dir.exists():
        integration_res = integration_dir / "res"
        if integration_res.exists():
            shutil.rmtree(integration_res)
        cleanup_temp_files(integration_dir)


def pytest_runtest_setup(item):
    if "qt" in item.keywords:
        try:
            import PySide6  # or PySide6, depending on your package
        except ImportError:
            pytest.skip("Skipping Qt-dependent test: Qt not available")
