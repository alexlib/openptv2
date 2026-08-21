# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openptv2 is a Particle Tracking Velocimetry (PTV) library centered on a single Cython 3 Pure Python implementation:
- **Pure Python/NumPy/Cython modules** (`src/openptv2/algorithms/`) — the only algorithm implementation path
- **openptv2 package** (`src/openptv2/`) — public API and compatibility namespace over that runtime
- **GUI** (`src/openptv2/gui/`) — Tkinter/ttkbootstrap-based desktop application

The current focus is optimizing and validating the single-engine pure Python/Cython modules and keeping the GUI/API aligned with that runtime. See `STATUS.md` for translation and migration progress.

## Commands

Always use `uv` — never bare `python` or `pip`.

```bash
# Setup
uv sync --extra dev

# Run all tests (configured testpaths: tests)
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_vec_utils.py -v

# Run a single test function
uv run pytest tests/unit/test_vec_utils.py::test_dot -v

# Run tests by marker
uv run pytest -m unit
uv run pytest -m "not slow"

# Hot-path smoke test (tracking + correspondences)
uv run pytest tests/unit/test_track.py tests/unit/test_track3d.py tests/unit/test_correspondences.py -v --tb=short

# Lint
uv run ruff check .

# Type check
uv run mypy src/openptv2/

# Editable install — three profiles:
uv pip install -e .            # default: headless library + openptv2-batch pipeline
uv pip install -e ".[gui]"     # + desktop GUI (traitsui/chaco/enable/PySide6)
uv pip install -e ".[dev]"     # everything: GUI, tests, lint, notebooks, docs

# Cython build (required after any .py change in algorithms/)
uv run python setup.py build_ext --inplace

# Clean Cython rebuild (use after structural changes — new files, renames)
rm -f src/openptv2/algorithms/*.c src/openptv2/algorithms/*.so
rm -rf build/
uv run python setup.py build_ext --inplace

# Check annotation scores (open generated HTML in browser)
# annotate=True is already set in setup.py — every rebuild generates <module>.html
```

### Pure-Python fallback tests (opt-in)

`tests/unit/test_*_coverage.py` (16 files) run the `algorithms/*.py` sources
**interpreted**, not compiled — each guards itself with `if is_compiled():
pytest.skip(...)`, so under the normal compiled build (the default `uv run
pytest` above) they always skip. They exist for two reasons:

1. **Line coverage.** `coverage.py` can't see into a compiled `.pyd`/`.so`
   (see `docs/coverage-to-100-plan.md`), so these are the only tests that ever
   report real line coverage for `algorithms/`.
2. **The pure-Python backup path.** If the Cython build fails on a user's
   machine (missing compiler, ABI mismatch, etc.), `openptv2` still imports
   and runs on the interpreted `.py` sources — `cython.compiled` is just
   `False` and every module falls back automatically. These 952 tests are the
   only thing that actually exercises that fallback; without them a break in
   it would go unnoticed until a user hit it.

Run them by making the compiled extensions unimportable for one Python
version, so the pure-Python source loads instead:

```bash
# Move the active interpreter's compiled extensions aside (adjust the tag,
# e.g. cp312-win_amd64 / cpython-312-x86_64-linux-gnu, to match `python --version`)
mkdir .pyd_backup && mv src/openptv2/algorithms/*.cp312-win_amd64.pyd .pyd_backup/

uv run pytest tests/unit/test_*_coverage.py -m '' -q

# Restore and rebuild
mv .pyd_backup/*.pyd src/openptv2/algorithms/ && rmdir .pyd_backup
uv run python setup.py build_ext --inplace
```

Do **not** run the full suite this way — hot-path tests (`test_track.py`,
`test_track3d.py`, `test_correspondences.py`, the batch/tracking-quality
tests) run real per-particle loops over real datasets and are 50-100x slower
uncompiled; they can run for tens of minutes without failing. The 16
`*_coverage.py` files use tiny synthetic fixtures specifically so this check
stays fast (<15s) — that's the intended scope for this mode, not a general
"run everything without Cython" substitute.

Last verified 2026-08-21: passes clean (952/952) after fixing three real bugs
this mode caught that the compiled build's `@cython.boundscheck(False)`
hides — see `git log --oneline -- src/openptv2/algorithms/track.py
src/openptv2/algorithms/track_kernels_search.py` around that date. Re-run
this before relying on the fallback after any change to `algorithms/`.

## Architecture

**Runtime model**: the same `src/openptv2/algorithms/*.py` modules run interpreted in development and compiled when built through Cython 3. There is no runtime engine selector.

**algorithms/ design principles**:
- Structure-of-Arrays (SoA) layout for batch data
- Dataclasses for parameters — no getter/setter boilerplate
- No adapter layers or dual storage
- Each module is a standalone translation of its C counterpart

**Test data**: `test_data/` contains calibration files, parameter files, and fixture data used across all test suites. Tests import from `openptv2.algorithms.*` directly (e.g., `from openptv2.algorithms.vec_utils import ...`).

**Batch/cloud running**: `docs/cloud-batch.md` (single run) and `docs/multi-folder-runs.md` (several runs sharing one calibration — folder layout, calibration-sharing checklist, YAML prep, `scripts/run_pipeline_multi.sh`). Calibration sanity checks (sight-line angle, cross-camera symmetry — catches a self-consistent-but-physically-wrong bundle adjustment that reprojection RMS alone misses) live in `openptv2.calibration_diagnostics`, used by both `scripts/calibration_diagnostics.py` (headless) and `src/openptv2/gui/visualize_calibration_nb.py` (interactive marimo viewer, also wired as the `visualize-calibration` Claude Code skill). Cross-camera ray-convergence miss distance (RCM, mm) over calblock points seen by >=2 cameras — a cross-camera geometric consistency check that per-camera reprojection RMS cannot see — is reported by `calib.py run` and `openptv2.autocalibration.cross_camera_rcm`.

## Code Style

- Python 3.11+, line length 88 (ruff configured)
- ruff lint rules: E, W, F, I (no docstring enforcement)
- Match the direct-translation style in `src/openptv2/algorithms/`: function names mirror C originals, SoA patterns, numpy vectorized operations
- Tests use pytest with markers: `unit`, `parity`, `perf`, `integration`, `slow`, `gui`

## Behavioral Guidelines

- State assumptions explicitly before implementing. If uncertain, ask.
- Minimum code that solves the problem — no speculative features or abstractions.
- Touch only what the task requires. Don't "improve" adjacent code.
- Define verifiable success criteria. Reproduce bugs with tests before fixing.
