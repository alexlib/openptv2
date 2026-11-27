"""The course pages (docs/course) are tested: every ``python -c`` command in them is run on the bundled example, and its output
must equal the output block printed on the page. A change in the code that alters what a page shows fails here, not in front of a
reader. Pages with a command and a ``text`` block right after "What you should see" are covered; add yours by following that shape.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "docs" / "course").glob("0*.md"))

pytestmark = pytest.mark.unit


def _commands_and_expected(text: str) -> list[tuple[str, str]]:
    """(python -c code, expected output) pairs: a ``python -c "..."`` block followed (after 'What you should see') by a text block."""
    out = []
    for m in re.finditer(r'```bash\n\s*python -c "\n(.*?)\n\s*"\n\s*```', text, re.S):
        code = "\n".join(line[4:] if line.startswith("    ") else line for line in m.group(1).split("\n"))
        tail = text[m.end():]
        e = re.search(r"## What you should see\s+(.*?)\n## ", tail, re.S)
        block = re.search(r"```text\n(.*?)```", e.group(1), re.S) if e else None
        if block:
            out.append((code, block.group(1)))
    return out


def _norm(s: str) -> str:
    lines = [ln.rstrip() for ln in s.strip().splitlines() if not ln.startswith("Loaded calibration")]
    return "\n".join(" ".join(ln.split()) for ln in lines)


CASES = [(p.name, code, exp) for p in PAGES for code, exp in _commands_and_expected(p.read_text())]


def test_course_has_tested_commands():
    assert len(CASES) >= 2, "the course pages lost their runnable commands"


@pytest.mark.parametrize("page,code,expected", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_page_output_matches(page, code, expected):
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert _norm(r.stdout) == _norm(expected), f"{page}: the output differs from the page"


def test_self_check_passes():
    r = subprocess.run([sys.executable, "-m", "openptv2.cli", "self-check"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    m = re.search(r"Summary: (\d+) passed, (\d+) failed", r.stdout)
    assert m and int(m.group(2)) == 0
