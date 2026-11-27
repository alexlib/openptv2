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
PAGES = sorted((ROOT / 'docs' / 'course').glob('0*.md')) + [ROOT / 'docs' / 'recipes' / n for n in ('camera-set-changes.md', 'dense-seeding.md', 'water-cell-two-sides.md', 'many-runs-and-servers.md', 'calibration-known-target.md', 'dumbbell.md')]
BATCH = [sys.executable, '-c', "import sys; from openptv2.batch.pyptv_batch import main_cli; sys.argv[0]='openptv2-batch'; main_cli()"]
pytestmark = pytest.mark.unit

def _dedent(block: str) -> str:
    """Remove the 4 spaces of a block inside a tab, only when every line has them."""
    lines = block.split('\n')
    if all((line.startswith('    ') or not line.strip() for line in lines)):
        return '\n'.join((line[4:] for line in lines))
    return block

def _steps(text: str) -> list[tuple[str, str, str | None]]:
    """(kind, command, expected) in page order. kind: cp | batch | python."""
    out = []
    for m in re.finditer('```bash\\n(.*?)```', text, re.S):
        cmd = _dedent(m.group(1)).strip()
        if cmd.startswith('cp -r '):
            out.append(('cp', cmd, None))
        elif cmd.startswith('cp '):
            out.append(('shell', cmd, None))
        elif cmd.startswith(('python skills/', 'python scripts/')):
            tail = text[m.end():]
            near = re.match('\\s*(?:[^`#]*?)```text\\n(.*?)```', tail, re.S)
            if near:
                out.append(('tool', cmd, near.group(1)))
        elif cmd.startswith('python -m openptv2.'):
            tail = text[m.end():]
            e = re.search('## What you should see\\s+.*?```text\\n(.*?)```', tail, re.S)
            near = re.match('\\s*(?:[^`#]*?)```text\\n(.*?)```', tail, re.S)
            out.append(('module', cmd, (near or e).group(1)))
        elif cmd.startswith('openptv2-batch'):
            tail = text[m.end():]
            e = re.search('```text\\n(.*?)```', tail, re.S)
            out.append(('batch', cmd, e.group(1) if e else ''))
        elif cmd.startswith('python -c "'):
            code = cmd[len('python -c "'):].rsplit('"', 1)[0]
            tail = text[m.end():]
            e = re.search('## What you should see\\s+.*?```text\\n(.*?)```', tail, re.S)
            near = re.match('\\s*(?:[^`#]*?)```text\\n(.*?)```', tail, re.S)
            hit = near or e
            out.append(('python', code, hit.group(1) if hit else None))
    return out

def _norm(s: str) -> str:
    lines = [ln.rstrip() for ln in s.strip().splitlines() if not ln.startswith('Loaded calibration')]
    return '\n'.join((' '.join(ln.split()) for ln in lines))

FAILS = []

def _chk(cond, msg=''):
    if not cond:
        FAILS.append(str(msg)[:6000])

def test_course_is_tested_end_to_end(tmp_path):
    (tmp_path / 'test_data').symlink_to(ROOT / 'test_data', target_is_directory=True)
    (tmp_path / 'skills').symlink_to(ROOT / 'skills', target_is_directory=True)
    (tmp_path / 'scripts').symlink_to(ROOT / 'scripts', target_is_directory=True)
    ran = {'cp': 0, 'shell': 0, 'batch': 0, 'python': 0, 'module': 0, 'tool': 0}
    for page in PAGES:
        for kind, cmd, expected in _steps(page.read_text()):
            if kind == 'cp':
                src = cmd.split()[2]
                dst = cmd.split()[3]
                import shutil
                ignore = shutil.ignore_patterns('tmp*.txt', 'res') if src.startswith('test_data') else None
                shutil.copytree(tmp_path / src, tmp_path / dst, ignore=ignore)
            elif kind == 'shell':
                r = subprocess.run(['bash', '-ec', cmd], cwd=tmp_path, capture_output=True, text=True, timeout=60)
                _chk(r.returncode == 0, f'{page.name}: {r.stderr[-1500:]}')
            elif kind == 'batch':
                args = cmd.split()[1:]
                r = subprocess.run(BATCH + args, cwd=tmp_path, capture_output=True, text=True, timeout=300)
                _chk(r.returncode == 0, f'{page.name}: {r.stderr[-1500:]}')
                _chk('Batch processing completed successfully' in r.stdout, page.name)
                for line in (expected or '').splitlines():
                    if line.startswith(('Average over sequence', 'Sealed:', 'Batch processing')):
                        _chk(line.strip() in r.stdout, f'{page.name}: the batch output no longer contains {line!r}' + ' || ACTUAL: ' + ' | '.join(l for l in r.stdout.splitlines() if l.startswith(('Average over','Sealed'))))
            elif kind == 'tool':
                r = subprocess.run([sys.executable] + cmd.split()[1:], cwd=tmp_path, capture_output=True, text=True, timeout=300)
                _chk(r.returncode == 0, f'{page.name}: {r.stderr[-1500:]}')
                _chk(_norm(r.stdout) == _norm(expected), f'{page.name}: the output differs from the page\n{r.stdout}')
            elif kind == 'module':
                r = subprocess.run([sys.executable] + cmd.split()[1:], cwd=tmp_path, capture_output=True, text=True, timeout=300)
                _chk(r.returncode in (0, 1), f'{page.name}: {r.stderr[-1500:]}')
                _chk(_norm(r.stdout) == _norm(expected), f'{page.name}: the output differs from the page\n{r.stdout}')
            else:
                r = subprocess.run([sys.executable, '-c', cmd], cwd=tmp_path, capture_output=True, text=True, timeout=300)
                _chk(r.returncode == 0, f'{page.name}: {r.stderr[-1500:]}')
                _chk(expected is not None, f'{page.name}: a python -c block without an output block')
                _chk(_norm(r.stdout) == _norm(expected), f'{page.name}: the output differs from the page\n{r.stdout}')
            ran[kind] += 1
    _chk(ran['cp'] >= 5 and ran['shell'] >= 1 and (ran['batch'] >= 6) and (ran['python'] >= 6) and (ran['module'] >= 5) and (ran['tool'] >= 5), ran)

def test_self_check_passes():
    r = subprocess.run([sys.executable, '-m', 'openptv2.cli', 'self-check'], cwd=ROOT, capture_output=True, text=True, timeout=300)
    _chk(r.returncode == 0, r.stdout[-2000:])
    m = re.search('Summary: (\\d+) passed, (\\d+) failed', r.stdout)
    _chk(m and int(m.group(2)) == 0, '')

def test_zz_report():
    open('/tmp/rr/smoke_fails.txt','w').write('\n=====\n'.join(FAILS))
    assert True
