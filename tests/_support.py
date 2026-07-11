from __future__ import annotations

from pathlib import Path


def find_test_data_root(start: Path | None = None) -> Path:
    """Locate the repository-local test_data directory by walking upward."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        test_data_dir = candidate / "test_data"
        if test_data_dir.is_dir():
            return test_data_dir
    raise FileNotFoundError("Could not locate test_data directory")
