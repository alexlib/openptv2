import pytest
from pathlib import Path
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture to set up test data directory"""
    # Get the absolute path to the test_cavity directory
    test_dir = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not test_dir.exists():
        pytest.skip(f"Test data directory {test_dir} not found")
    return test_dir


@pytest.fixture(scope="session", autouse=True)
def clean_test_environment(test_data_dir):
    """Clean up test environment before and after tests"""
    temp_patterns = ["tmp*.yaml", "tmp*.txt", "*.yaml.bak", "*_summary.csv"]

    def cleanup_temp_files():
        for pattern in temp_patterns:
            for f in test_data_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass

    # Clean up any existing test results
    results_dir = test_data_dir / "res"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Clean up temporary files before test
    cleanup_temp_files()

    # Create fresh directories
    results_dir.mkdir(exist_ok=True)

    yield

    # Cleanup after tests
    if results_dir.exists():
        shutil.rmtree(results_dir)

    # Clean up temporary files after test
    cleanup_temp_files()


def pytest_runtest_setup(item):
    if "qt" in item.keywords:
        try:
            import PySide6  # or PySide6, depending on your package
        except ImportError:
            pytest.skip("Skipping Qt-dependent test: Qt not available")
