"""The course pages (docs/course/0*.md) are tested. In page order, in one scratch folder that sees ``test_data``:

* ``cp -r test_data/... my_run`` (and other ``cp`` lines) are executed;
* ``openptv2-batch ...`` is executed, it must finish, and the batch output lines the page prints (``Average over``, ``Sealed``) must appear;
* every ``python -c "..."`` block is executed and its output must equal the ``text`` block under "What you should see" on the page.

So a change in the code that alters what a page shows fails here, not in front of a reader. Commands that install, open a window or
need the network (pip, git, openptv2-gui, marimo) are not run.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "docs" / "course").glob("0*.md")) + [ROOT / "docs" / "recipes" / n for n in ("camera-set-changes.md", "dense-seeding.md", "water-cell-two-sides.md", "many-runs-and-servers.md", "calibration-known-target.md")]
BATCH = [sys.executable, "-c", "import sys; from openptv2.batch.pyptv_batch import main_cli; sys.argv[0]='openptv2-batch'; main_cli()"]

pytestmark = pytest.mark.unit


def _dedent(block: str) -> str:
    """Remove the 4 spaces of a block inside a tab, only when every line has them."""
    lines = block.split("\n")
    if all(line.startswith("    ") or not line.strip() for line in lines):
        return "\n".join(line[4:] for line in lines)
    return block


def _steps(text: str) -> list[tuple[str, str, str | None]]:
    """(kind, command, expected) in page order. kind: cp | batch | python."""
    out = []
    for m in re.finditer(r"```bash\n(.*?)```", text, re.S):
        cmd = _dedent(m.group(1)).strip()
        if cmd.startswith("cp -r "):
            out.append(("cp", cmd, None))
        elif cmd.startswith("cp "):
            out.append(("shell", cmd, None))
        elif cmd.startswith("python skills/"):
            tail = text[m.end():]
            near = re.match(r"\s*(?:[^`#]*?)```text\n(.*?)```", tail, re.S)
            if near:        # a command without an output block (pick opens a window) is not run
                out.append(("tool", cmd, near.group(1)))
        elif cmd.startswith("python -m openptv2."):
            tail = text[m.end():]
            e = re.search(r"## What you should see\s+.*?```text\n(.*?)```", tail, re.S)
            # a command whose output block is right after it (before the next heading), else the one under "What you should see"
            near = re.match(r"\s*(?:[^`#]*?)```text\n(.*?)```", tail, re.S)
            out.append(("module", cmd, (near or e).group(1)))
        elif cmd.startswith("openptv2-batch"):
            tail = text[m.end():]
            e = re.search(r"```text\n(.*?)```", tail, re.S)
            out.append(("batch", cmd, e.group(1) if e else ""))
        elif cmd.startswith('python -c "'):
            code = cmd[len('python -c "'):].rsplit('"', 1)[0]
            tail = text[m.end():]
            e = re.search(r"## What you should see\s+.*?```text\n(.*?)```", tail, re.S)
            near = re.match(r"\s*(?:[^`#]*?)```text\n(.*?)```", tail, re.S)
            hit = near or e
            out.append(("python", code, hit.group(1) if hit else None))
    return out


def _norm(s: str) -> str:
    lines = [ln.rstrip() for ln in s.strip().splitlines() if not ln.startswith("Loaded calibration")]
    return "\n".join(" ".join(ln.split()) for ln in lines)


def test_course_is_tested_end_to_end(tmp_path):
    (tmp_path / "test_data").symlink_to(ROOT / "test_data", target_is_directory=True)
    (tmp_path / "skills").symlink_to(ROOT / "skills", target_is_directory=True)
    ran = {"cp": 0, "shell": 0, "batch": 0, "python": 0, "module": 0, "tool": 0}
    for page in PAGES:
        for kind, cmd, expected in _steps(page.read_text()):
            if kind == "cp":
                src = cmd.split()[2]
                dst = cmd.split()[3]
                import shutil

                ignore = shutil.ignore_patterns("tmp*.txt", "res") if src.startswith("test_data") else None
                shutil.copytree(tmp_path / src, tmp_path / dst, ignore=ignore)
            elif kind == "shell":
                r = subprocess.run(["bash", "-ec", cmd], cwd=tmp_path, capture_output=True, text=True, timeout=60)
                assert r.returncode == 0, f"{page.name}: {r.stderr[-1500:]}"
            elif kind == "batch":
                args = cmd.split()[1:]
                r = subprocess.run(BATCH + args, cwd=tmp_path, capture_output=True, text=True, timeout=300)
                assert r.returncode == 0, f"{page.name}: {r.stderr[-1500:]}"
                assert "Batch processing completed successfully" in r.stdout, page.name
                for line in (expected or "").splitlines():
                    if line.startswith(("Average over sequence", "Sealed:", "Batch processing")):
                        assert line.strip() in r.stdout, f"{page.name}: the batch output no longer contains {line!r}"
            elif kind == "tool":
                r = subprocess.run([sys.executable] + cmd.split()[1:], cwd=tmp_path, capture_output=True, text=True, timeout=300)
                assert r.returncode == 0, f"{page.name}: {r.stderr[-1500:]}"
                assert _norm(r.stdout) == _norm(expected), f"{page.name}: the output differs from the page\n{r.stdout}"
            elif kind == "module":
                r = subprocess.run([sys.executable] + cmd.split()[1:], cwd=tmp_path, capture_output=True, text=True, timeout=300)
                assert r.returncode in (0, 1), f"{page.name}: {r.stderr[-1500:]}"      # 1 = a verdict other than "consistent"
                assert _norm(r.stdout) == _norm(expected), f"{page.name}: the output differs from the page\n{r.stdout}"
            else:
                r = subprocess.run([sys.executable, "-c", cmd], cwd=tmp_path, capture_output=True, text=True, timeout=300)
                assert r.returncode == 0, f"{page.name}: {r.stderr[-1500:]}"
                assert expected is not None, f"{page.name}: a python -c block without an output block"
                assert _norm(r.stdout) == _norm(expected), f"{page.name}: the output differs from the page\n{r.stdout}"
            ran[kind] += 1
    assert ran["cp"] >= 5 and ran["shell"] >= 1 and ran["batch"] >= 6 and ran["python"] >= 6 and ran["module"] >= 5 and ran["tool"] >= 2, ran


def test_self_check_passes():
    r = subprocess.run([sys.executable, "-m", "openptv2.cli", "self-check"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    m = re.search(r"Summary: (\d+) passed, (\d+) failed", r.stdout)
    assert m and int(m.group(2)) == 0
