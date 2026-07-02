#!/usr/bin/env python
"""Prepare CI test environment: copy test_data and tests into the CIBW venv.

Used by CIBW_TEST_COMMAND in .github/workflows/cibuildwheel.yml so that path
handling does not depend on shell string escaping (which broke on Windows
where cibuildwheel substitutes `{project}` as a raw backslash path like
``D:\\a\\openptv2\\openptv2`` -- embedding this directly in a Python string
literal via -c causes `\\a` to be interpreted as the bell escape character).

Usage:
    python scripts/ci_test_setup.py <project_dir>
"""

import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python ci_test_setup.py <project_dir>", file=sys.stderr)
        return 1

    project = Path(sys.argv[1])

    shutil.copytree(
        project / "test_data", "test_data", dirs_exist_ok=True, symlinks=True
    )
    shutil.copytree(
        project / "tests",
        "tests",
        dirs_exist_ok=True,
        symlinks=True,
        ignore=shutil.ignore_patterns("testing_fodder"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
