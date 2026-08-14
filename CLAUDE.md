# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openptv2 is a Particle Tracking Velocimetry (PTV) library centered on a single Cython 3 Pure Python implementation:
- **Pure Python/NumPy/Cython modules** (`src/openptv2/algorithms/`) — the only algorithm implementation path
- **openptv2 package** (`src/openptv2/`) — public API and compatibility namespace over that runtime
- **GUI** (`src/openptv2/gui/`) — TraitsUI/Chaco (PySide6) desktop application, migrating to marimo notebooks

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
