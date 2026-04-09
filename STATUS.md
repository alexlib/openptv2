# Project Status - April 2026

**Last updated**: 2026-04-08

---

## Debug Visualization Feature - In Progress

### Goal
Implement "Debugging with display" tracking visualization for OpenPTV that allows users to click on detected particles in PyPTV GUI and visualize search volumes, candidates, and epipolar lines.

### Accomplished (April 8, 2026)

#### 1. Fixed `_tracking_debug_click` (pyptv_gui.py)
- Changed to read parameters from `self.exp1.pm.parameters` directly (dictionary already in memory)
- Uses both `"track"` and `"tracking"` keys for tracking params (YAML compatibility)
- Uses both `"criteria"` and `"volume"` keys for volume params (YAML compatibility)
- Fixed VolumePar to use correct attribute names (`x_lay`, `z_min_lay`, `z_max_lay`)
- Uses TrackParTuple directly with dict values instead of convert function
- Handles both YAML formats (simple keys like "xmin" and nested like "X_lay")

#### 2. Added Python-only Installation Mode
- Modified `setup.py` to check `OPENPTV_PYTHON_ONLY` environment variable
- When set, skips Cython compilation and returns empty extensions list
- Full install: ~2 minutes, Python-only: ~1.4 seconds
- Added documentation in: INSTALL.md, README.md, docs/developer_guide/building.md

#### 3. Documentation Updates
- Added "Option D — Python-only" to INSTALL.md
- Added Python-only section to README.md
- Added Python-only section to docs/developer_guide/building.md

### Key Discoveries
- VolumePar uses `x_lay`, `z_min_lay`, `z_max_lay` lists (not Xmin/Xmax/Ymin/Ymax)
- Tracking parameters use key `"track"` not `"tracking"` in YAML
- Volume parameters use key `"criteria"` not `"volume"` in YAML
- `convert_track_par_to_tuple()` requires a TrackPar object, not dict - must use TrackParTuple directly with dict values
- The `OPENPTV_PYTHON_ONLY=1` environment variable can skip Cython build (~100x faster install)
- Default install always rebuilds Cython even when only Python code changes

### Remaining Issue
- **Visualization doesn't appear on camera views** - The click handler works but no visualization draws on the canvas

---

## Status vs DESIGN_PLAN.md

### Phase 1: Copy & Integrate — ✅ COMPLETE

| Criterion | Status | Notes |
|-----------|--------|-------|
| `uv sync` builds C library + bindings + GUI | ✅ | Working |
| GUI launches and tracks particles | ✅ | Working |
| All existing optv + pyptv tests pass | ✅ | 341+ passing (bindings + algorithms + gui) |
| Wheel builds locally | ✅ | manylinux2014 wheels for cp311, cp312, cp313 |
| Wheel installs on Linux | ✅ | Verified in clean venv |

### Phase 2: Python/Numba Fallback — ✅ SUBSTANTIALLY COMPLETE

| Criterion | Status | Notes |
|-----------|--------|-------|
| `--engine=python` flag works | ✅ | Via `openptv2.set_engine("python")` or per-call |
| Engine comparison tests pass | ✅ | 1e-10 tolerance verified |
| `--validate-engine` CLI flag | ❌ | Not yet implemented |

### Phase 3: Unification & Polish — ⏳ IN PROGRESS

| Criterion | Status | Notes |
|-----------|--------|-------|
| `import openptv2` works | ✅ | Unified package working |
| Documentation merged | ⚠️ | Partial (BUILDING_BINARY_WHEELS.md added) |
| CI builds wheels automatically | ✅ | cibuildwheel.yml configured |
| Version 1.0.0 released on PyPI | ❌ | Not yet |

---

## Recent Changes (Since April 4th)

### Documentation
- Added `BUILDING_BINARY_WHEELS.md` - Complete wheel build guide
- Updated `docs/index.md` with available documentation
- Added PyPI publishing section to `docs/developer_guide/building.md`
- Updated README.md with helper scripts section

### CI/CD Improvements
- Fixed cibuildwheel.yml to use `pypa/cibuildwheel` action
- Added manylinux2014 for portable Linux wheels
- Added aarch64 support for ARM64 Linux
- Removed Python 3.14 (not yet implemented)
- Fixed test_package to select specific wheel

### Bug Fixes
- `test_18_frame_io.py`: Fixed path `test_data/sample.` → `test_data/sample_`
- `algorithms/__init__.py`: Lazy loading now only requires numba for modules that use it
- `calibration.py`: Changed from_file to classmethod (was instance method, broke API)
- `test_burgers.py`: Fixed bytes paths/keys (cross-platform Windows compatibility)
- `gui/tests/conftest.py`: Centralized test data setup in root conftest.py
- `bindings/tests/conftest.py`: Fixed tab indentation → spaces (PEP 8)

### Test Results

| Test Suite | Count | Status |
|------------|-------|--------|
| bindings/tests/ | 70 | ✅ All pass |
| algorithms/tests/ | 260+ | ✅ All pass |
| gui/tests/ | 246 | ✅ All pass |
| **Total** | **576+** | ✅ All pass |

---

## Outstanding Items from DESIGN_PLAN.md

| Item | Priority | Status |
|------|----------|--------|
| `--engine=python` flag | High | ✅ Works via Python API |
| `--validate-engine` CLI | High | ❌ Not implemented |
| Documentation merged | Medium | ⚠️ Partial |
| PyPI release | High | ❌ Not done |
| Engine CLI flags | Medium | ❌ Not implemented |

---

## Build Artifacts

Binary wheels available in `dist/`:
- `openptv2-1.0.0-cp311-cp311-manylinux2014_x86_64.whl`
- `openptv2-1.0.0-cp312-cp312-manylinux2014_x86_64.whl`
- `openptv2-1.0.0-cp313-cp313-manylinux2014_x86_64.whl`

---

## Key Files

| Category | Files |
|----------|-------|
| Build | `pyproject.toml`, `setup.py`, `CMakeLists.txt` |
| CI/CD | `.github/workflows/cibuildwheel.yml` |
| Docs | `README.md`, `BUILDING_BINARY_WHEELS.md`, `docs/` |
| Tests | `bindings/tests/`, `algorithms/tests/`, `gui/tests/` |
| Scripts | `scripts/build_wheel.sh`, `scripts/install_wheel.sh`, `scripts/run_tests.sh` |
| Plans | `DESIGN_PLAN.md`, `STATUS.md`, `SETUP_PLAN.md` |
