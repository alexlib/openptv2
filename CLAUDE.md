# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

openptv2 is a Particle Tracking Velocimetry (PTV) library with a dual-engine architecture:
- **C core + Cython bindings** (`lib/` + `bindings/`) — production engine (`optv`)
- **Pure Python/NumPy** (`algorithms/`) — debuggable engine, a direct C-to-Python translation
- **openptv2 package** (`openptv2/`) — unified API with engine selection
- **GUI** (`gui/`) — TraitsUI-based desktop application

Active work: translating remaining C modules in `lib/src/` to Python in `algorithms/`. Each `algorithms/*.py` maps 1:1 to a `lib/src/*.c` file. See `STATUS.md` for translation progress.

## Commands

Always use `uv` — never bare `python` or `pip`.

```bash
# Setup
uv sync --extra dev

# Run all tests (configured testpaths: algorithms/tests, bindings/tests, gui/tests)
uv run pytest

# Run a single test file
uv run pytest algorithms/tests/test_vec_utils.py -v

# Run a single test function
uv run pytest algorithms/tests/test_vec_utils.py::test_dot -v

# Run tests by marker
uv run pytest -m unit
uv run pytest -m "not slow"

# Lint
uv run ruff check .

# Type check
uv run mypy openptv2/

# Build C library (needed for optv engine)
cd lib && mkdir -p build && cd build && cmake .. && make

# Python-only install (skips Cython build, ~100x faster)
OPENPTV_PYTHON_ONLY=1 uv pip install -e .
```

## Architecture

**Engine selection** (`openptv2/engine.py`): Thread-local `EngineSelector` picks between `optv` (C/Cython) and `python` (NumPy). Falls back to Python if optv is unavailable.

**algorithms/ design principles** (from `algorithms/__init__.py`):
- Structure-of-Arrays (SoA) layout for batch data
- Dataclasses for parameters — no getter/setter boilerplate
- No adapter layers or dual storage
- Each module is a standalone translation of its C counterpart

**Test data**: `test_data/` contains calibration files, parameter files, and fixture data used across all test suites. Tests import from `algorithms.*` directly (e.g., `from algorithms.vec_utils import ...`).

## Code Style

- Python 3.11+, line length 88 (ruff configured)
- ruff lint rules: E, W, F, I (no docstring enforcement)
- Match the direct-translation style in `algorithms/`: function names mirror C originals, SoA patterns, numpy vectorized operations
- Tests use pytest with markers: `unit`, `parity`, `perf`, `integration`, `requires_optv`, `slow`, `gui`

## Behavioral Guidelines

- State assumptions explicitly before implementing. If uncertain, ask.
- Minimum code that solves the problem — no speculative features or abstractions.
- Touch only what the task requires. Don't "improve" adjacent code.
- Define verifiable success criteria. Reproduce bugs with tests before fixing.
