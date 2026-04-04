from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from openptv2.test_support import find_test_data_root


TEST_DATA_ROOT = find_test_data_root(Path(__file__))
REPO_ROOT = TEST_DATA_ROOT.parent


@pytest.fixture(scope="session", autouse=True)
def _bindings_cwd():
    """Run bindings tests from the repo root so relative test_data paths resolve."""
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


@pytest.fixture(scope="session")
def test_data_root() -> Path:
    """Shared repo-local test_data root for bindings tests."""
    return TEST_DATA_ROOT


# Add the optv package directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
