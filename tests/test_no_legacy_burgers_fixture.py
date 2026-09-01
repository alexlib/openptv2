"""Phase 4 enforcement (see docs/plans/2026-09-02-refactor-burgers-synthetic-tests.md):
new tests must not depend on the deprecated fixed 5-frame img fixture — use
tests/helpers/synthetic_scene.py::make_cavity_scene instead.
"""

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent

# Legacy tests explicitly kept against the fixed 5-frame fixture, plus
# test_parameters_coverage.py which parses burgers/parameters/*.par legacy
# fixture files (unrelated to the deprecated img/res_orig 5-frame smoke).
ALLOWED = {
    TESTS_DIR / "batch" / "test_burgers_synthetic.py",
    TESTS_DIR / "unit" / "test_track.py",
    TESTS_DIR / "unit" / "test_track3d.py",
    TESTS_DIR / "unit" / "test_parameters_coverage.py",
}

_PATTERN = re.compile(r"""["']test_data/burgers|BURGERS_DIR\s*=""")

pytestmark = pytest.mark.ci


def test_no_new_test_depends_on_legacy_burgers_fixture():
    offenders = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path in ALLOWED or "__pycache__" in path.parts:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.strip().startswith("#"):
                continue
            if _PATTERN.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "new tests must not depend on the deprecated test_data/burgers "
        "5-frame fixture — use tests/helpers/synthetic_scene.py::make_cavity_scene "
        "instead:\n" + "\n".join(offenders)
    )
