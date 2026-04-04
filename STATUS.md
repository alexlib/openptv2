# Project Status

**Last updated**: 2026-04-04

---

## Design Plan Alignment

Based on `DESIGN_PLAN.md`, here's where we stand:

### Phase 1: Copy & Integrate — ✅ LARGELY COMPLETE

| Criterion | Status | Notes |
|-----------|--------|-------|
| `uv sync` builds C library + bindings + GUI | ✅ | Working |
| GUI launches and tracks particles | ✅ | Working |
| All existing optv + pyptv tests pass | ✅ | Algorithms (260+), Bindings (70), GUI (245) |
| Wheel installs on Linux, Windows, macOS | ⏳ | CI pipeline not yet configured |

### Phase 2: Python/Numba Fallback — ✅ SUBSTANTIALLY COMPLETE

| Criterion | Status | Notes |
|-----------|--------|-------|
| `--engine=python` flag works | ✅ | Python engine runs via `algorithms/track.py` |
| Engine comparison tests pass | ✅ | Isolated workspaces, identical results verified |
| `--validate-engine` CLI flag | ❌ | Not yet implemented |

### Phase 3: Unification & Polish — ⏳ NOT STARTED

| Criterion | Status | Notes |
|-----------|--------|-------|
| `import openptv2` works | ❌ | `openptv2/` folder exists but not functional |
| Documentation merged and deployed | ❌ | Not started |
| CI builds wheels automatically | ❌ | No GitHub Actions workflow |
| Version 1.0.0 released on PyPI | ❌ | Not started |

---

## What's Been Done

### Engine Parity — Verified ✅

Both engines (Cython/C and Python/Numba) produce **identical results** when given isolated inputs:

- **Particle positions**: Identical counts and coordinates
- **Linkage data**: Identical after normalizing C's `-1` vs Python's `0` empty-frame convention
- **_targets files**: Identical content, no cross-contamination

### Bugs Discovered

1. **C `read_path_frame` bug** (`lib/src/tracking_frame_buf.c:227-336`): `do...while(!feof)` always executes once, setting `num_parts=-1` for empty particle files instead of `0`. Python is correct here.

2. **Cython `read_targets()` wrapper** (`bindings/optv/tracking_framebuf.pyx:227`): When C returns -1 (file not found), wrapper creates TargetArray with -1 targets → `SystemError` on `len()`.

3. **Cython `Frame` constructor**: Passing `linkage_file_base=None` causes segfault (None → NULL conversion issue).

### Test Coverage

| Test Suite | Count | Status |
|------------|-------|--------|
| `test_frame_reading_parity.py` | 7 | ✅ All pass |
| `test_isolated_engine_comparison.py` | 4 | ✅ All pass |
| `test_20_track3d_engine_comparison.py` | 3 | ✅ All pass |
| Algorithm tests (total) | 260+ | ✅ All pass |
| Binding tests | 70 | ✅ All pass |
| GUI tests | 245 | ✅ All pass |

---

## Known Issues (Pre-existing, Not Introduced)

1. **C empty-frame bug**: `read_path_frame` returns -1 for files with 0 particles. Affects statistics reporting but not actual tracking results.

2. **Cython error handling**: `read_targets()` and `Frame` constructor have edge-case failures on missing/None inputs.

3. **Python `read_targets()`**: Raises `FileNotFoundError` instead of returning empty list for missing files.

---

## Plan: What to Do Next

### Priority 1: Fix the C empty-frame bug (1-2 hours)

**File**: `lib/src/tracking_frame_buf.c:227-336`

The `do...while(!feof)` should be a `while` loop, or the error handler should distinguish between "no data" and "read error":

```c
// Current (buggy):
targets = 0;
do {
    // ... fscanf fails ...
    if (read_res != 8) { targets = -1; break; }
} while (!feof(filein));

// Fix: check feof before attempting read, or don't reset targets to -1
// when the first read simply finds EOF (empty file after header).
```

This fix will:
- Make C report `0` particles for empty files (matching Python)
- Fix the `-1` vs `0` discrepancy in linkage file headers
- Fix the average particle count reporting (0.5 → 0.8)

### Priority 2: Implement `openptv2/` unification package (Phase 3)

Per `DESIGN_PLAN.md` Section 4, create the unified entry point:

1. **`openptv2/__init__.py`** — Single import: `from openptv2 import Tracker, Target, ...`
2. **`openptv2/engine.py`** — Engine selector (optv vs python)
3. **`openptv2/calibration.py`** — Unified calibration wrapper
4. **`openptv2/tracker.py`** — Unified tracker wrapper
5. **`openptv2/tracking_framebuf.py`** — Unified frame buffer wrapper
6. **`openptv2/validate.py`** — Validation CLI (`openptv2-validate`)

### Priority 3: Engine selector CLI flags

Add `--engine optv|python` to:
- GUI launch command
- Batch processing command
- Validation command (`openptv2-validate` runs both, compares)

### Priority 4: CI/CD pipeline

GitHub Actions workflow for:
- Building wheels (Linux, Windows, macOS) via cibuildwheel
- Running all tests on PRs
- Uploading to PyPI on tag

### Priority 5: Documentation

- Merge existing READMEs
- API docs from docstrings (Sphinx)
- User guides

---

## Relevant Files

### Tests
- `algorithms/tests/test_frame_reading_parity.py` — Frame I/O layer parity
- `algorithms/tests/test_isolated_engine_comparison.py` — Full engine isolation comparison
- `algorithms/tests/test_20_track3d_engine_comparison.py` — Engine comparison with temp .par files
- `algorithms/tests/conftest.py` — Shared fixtures

### Source
- `algorithms/tracking_frame_buf.py` — Python frame reader
- `algorithms/track.py` — Python tracking pipeline
- `algorithms/tracking_run.py` — TrackingRun class
- `lib/src/tracking_frame_buf.c` — C frame reader (has bug)
- `lib/src/track3d.c` — C 3D tracking loop
- `bindings/optv/tracking_framebuf.pyx` — Cython frame reader
- `bindings/optv/tracker.pyx` — Cython tracker

### Design
- `DESIGN_PLAN.md` — Full design plan
- `STATUS.md` — This file
