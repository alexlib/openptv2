# Project Status - April 2026

**Last updated**: 2026-04-05

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

## Next Steps

### Priority 1: Complete Phase 3

1. **Implement `--validate-engine` CLI flag**
   - Add `openptv2-validate` command
   - Run both engines and compare results

2. **Complete documentation**
   - Merge READMEs into comprehensive guide
   - Add API reference (Sphinx)
   - User tutorials

### Priority 2: PyPI Release

1. **Configure trusted publishing** on PyPI
2. **Tag version** v1.0.0
3. **Push tag** to trigger CI build and upload

### Priority 3: Engine Improvements

1. Add `--engine` flag to GUI
2. Add `--engine` flag to batch processing
3. Performance benchmarks (optional)

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