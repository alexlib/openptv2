"""
Root conftest.py for openptv2 tests.

This module provides session-scoped fixtures for test data setup and cleanup.
"""

import pytest
from pathlib import Path
import shutil
from openptv2.test_support import find_test_data_root

TEST_DATA_DIR = find_test_data_root(Path(__file__))


@pytest.fixture(scope="session")
def test_data_root():
    """Shared repo-local test_data root for tests that need it directly."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session", autouse=True)
def setup_and_cleanup_test_environment():
    """
    Set up test environment before all tests and clean up after.

    This fixture:
    1. Copies res_orig to res for test_cavity
    2. Cleans up temporary files after all tests
    """
    # Set up test_cavity/res from test_cavity/res_orig
    test_cavity = TEST_DATA_DIR / "test_cavity"
    if test_cavity.exists():
        res_dir = test_cavity / "res"
        res_orig = test_cavity / "res_orig"

        # Remove existing res directory
        if res_dir.exists():
            shutil.rmtree(res_dir)

        # Copy res_orig to res
        if res_orig.exists():
            shutil.copytree(res_orig, res_dir)

    # Also set up test_cavity_integration
    integration_dir = TEST_DATA_DIR / "test_cavity_integration"
    if integration_dir.exists():
        integration_res = integration_dir / "res"
        if integration_res.exists():
            shutil.rmtree(integration_res)

    yield

    # Cleanup after all tests
    temp_patterns = ["tmp*.yaml", "tmp*.txt", "*.yaml.bak", "*_summary.csv"]

    def cleanup_temp_files(directory):
        if directory.exists():
            for pattern in temp_patterns:
                for f in directory.glob(pattern):
                    try:
                        f.unlink()
                    except OSError:
                        pass

    # Clean up test_cavity
    if test_cavity.exists():
        res_dir = test_cavity / "res"
        if res_dir.exists():
            shutil.rmtree(res_dir)
        cleanup_temp_files(test_cavity)

    # Clean up test_cavity_integration
    if integration_dir.exists():
        integration_res = integration_dir / "res"
        if integration_res.exists():
            shutil.rmtree(integration_res)
        cleanup_temp_files(integration_dir)

    # Clean up track directory
    track_dir = TEST_DATA_DIR / "track"
    if track_dir.exists():
        track_res = track_dir / "res"
        if track_res.exists():
            shutil.rmtree(track_res)
        cleanup_temp_files(track_dir)
