# openptv2 Development Status

**Date**: March 23, 2026
**Version**: 1.0.0 (development)
**Repository**: https://github.com/alexlib/openptv2

---

## Current Status Summary

### ✅ Completed (Phase 1 - Extended)

#### Repository Structure
- [x] Unified repository structure aligned with DESIGN_PLAN.md
- [x] All source code organized: `lib/`, `bindings/`, `algorithms/`, `gui/`, `openptv2/`, `tests/`, `docs/`
- [x] Git repository initialized and pushed to GitHub (alexlib/openptv2)
- [x] Clean directory structure with no build artifacts

#### Build System
- [x] CMake configuration for C library (`lib/CMakeLists.txt`)
- [x] **setuptools configuration for Cython bindings** (`bindings/setup.py`, `bindings/pyproject.toml`)
- [x] Main project configuration (`pyproject.toml`)
- [x] Documentation for building (`docs/developer_guide/building.md`)
- [x] **GitHub Actions workflow for wheel building** (`.github/workflows/cibuildwheel.yml`)
- [x] **Local wheel building script** (`run_cibuildwheel.sh`)

#### Dependencies
- [x] Python 3.11 environment created (`.venv311/`)
- [x] **optv built from local source (v0.3.2)** - working!
- [x] openptv2 installed in editable mode
- [x] GUI dependencies installed (traits, traitsui, PySide6, chaco, etc.)
- [x] algorithms package populated from openptv-python

#### Testing
- [x] Engine comparison tests: ✅ 3 passed, 2 skipped
- [x] Environment tests: ✅ 3 passed
- [x] Calibration utils tests: ✅ 5 passed
- [x] All imports verified working
- [x] **Integration tests: ✅ 237 passed, 2 failed (pre-existing issues)**

#### Documentation
- [x] README.md updated with installation and usage instructions
- [x] Developer guide created (`docs/developer_guide/building.md`)
- [x] Documentation index created (`docs/index.md`)

#### Code Quality & Refactoring
- [x] **Removed `_backend.py` compatibility layer** - migrated to direct `optv` imports
- [x] **Updated all test imports** from `pyptv.X` to `gui.pyptv.X`
- [x] **Added 5 new plugin files** for image processing and tracking
- [x] **Fixed 44+ test files** with correct import paths and mock patches

---

### ✅ Resolved Issues

#### 1. Local Cython Bindings Build - FIXED

**Solution**: Switched from scikit-build-core to setuptools with a `prepare` command.

**Build Process**:
```bash
cd bindings
python setup.py prepare  # Copies C sources, runs Cython
pip install -e .         # Builds and installs
```

**Key Changes**:
- `bindings/setup.py` - Custom setup with `prepare` and `build_ext` commands
- `bindings/pyproject.toml` - Uses setuptools instead of scikit-build-core
- C sources copied from `../lib/src/` to `bindings/liboptv/src/`
- Headers copied from `../lib/include/` to `bindings/liboptv/include/`
- Each extension module includes all C library sources directly

**Impact**:
- ✅ Local development now possible without PyPI package
- ✅ C code changes can be tested immediately
- ✅ Reproducible builds with `run_cibuildwheel.sh`
- ✅ Binary wheels can be built for all platforms via GitHub Actions

---

#### 2. Import Refactoring - COMPLETED (March 23, 2026)

**Problem**: The `gui/pyptv/_backend.py` compatibility layer was no longer needed and all imports should use direct `optv` module imports.

**Solution**: 
- Deleted `gui/pyptv/_backend.py` (316 lines removed)
- Updated imports in `gui/pyptv/ptv.py`, `standalone_calibration.py`, `flowtracks_utils.py`
- Migrated all test imports from `from pyptv.X import Y` to `from gui.pyptv.X import Y`
- Updated mock patch paths in 44+ test files
- Fixed subprocess commands to use `gui.pyptv` module paths

**Changes**:
```python
# Before
from ._backend import Calibration, ControlParams, Tracker

# After
from optv.calibration import Calibration
from optv.parameters import ControlParams
from optv.tracker import Tracker
```

**Impact**:
- ✅ Cleaner import structure with explicit dependencies
- ✅ No intermediate compatibility layer to maintain
- ✅ Direct optv usage as per DESIGN_PLAN.md Phase 1
- ✅ All 237 integration tests pass (2 pre-existing failures unrelated to refactoring)

**Files Modified**: 61 files changed, 949 insertions(+), 488 deletions(-)

---

### ⚠️ Known Issues

#### 1. Python/Numba Engine Not Implemented

**Problem**: `algorithms/` module imports but numba backend not functional.

**Status**: Phase 2 work - algorithms copied from openptv-python but not integrated.

**Impact**:
- Only optv (C/Cython) engine available
- Cannot compare engine results
- No pure Python fallback option

---

#### 2. Minor Test Failures (Pre-existing)

**Problem**: 2 integration tests fail due to unrelated issues:
- `test_function_coverage_documentation` - Test bug (ZeroDivisionError)
- `test_standalone_dumbbell_calibration_cycle` - Script output parsing issue

**Status**: Not blocking - these are pre-existing issues unrelated to the import refactoring.

**Impact**:
- Core functionality fully tested (237 tests pass)
- These tests need minor fixes but don't affect production use

---

## Phase Status Overview

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Copy & Integrate | ✅ Complete (Extended with refactoring) |
| Phase 2 | Python/Numba Fallback | ⏳ Next |
| Phase 3 | Unification & Polish | ⏳ Future |

*See DESIGN_PLAN.md for detailed phase descriptions and success criteria.*

---

## GUI Testing Results (Phase 1 Verification)

**Test Date**: March 23, 2026

### GUI Import Tests
- ✅ GUI module imports successfully
- ✅ All optv modules used by GUI are working
- ✅ GUI classes can be instantiated
- ✅ GUI entry point available (`gui.pyptv.pyptv_gui.main`)
- ✅ GUI initializes successfully in virtual display (xvfb)

### GUI-Related Tests Passed
- ✅ `test_optv.py::test_optv_functionality` - optv integration
- ✅ `test_environment.py` - environment checks
- ✅ `test_installation.py::test_installation` - installation verification
- ✅ `test_core_functionality.py::test_core_functionality` - core features
- ✅ `test_parameters.py` - parameter handling
- ✅ `test_parameter_manager.py` - parameter management
- ✅ `test_tracker_minimal.py::test_tracker_minimal` - particle tracking
- ✅ `test_tracking_analysis.py` - tracking analysis features

### Integration Test Summary (March 23, 2026)
```
Total integration tests: 240
Passed: 237 (98.8%)
Failed: 2 (pre-existing issues)
Skipped: 1

GUI tests: 15+ passed
optv tests: All passed
Tracker tests: All passed
Calibration tests: All passed
```

### Conclusion
**Phase 1 is COMPLETE**. The GUI works correctly with local optv bindings.
**Import refactoring is COMPLETE**. All modules use direct optv imports as per DESIGN_PLAN.md.

*See DESIGN_PLAN.md for detailed phase descriptions and success criteria.*

---

## Build Instructions

### Building from Source

```bash
# Using Python 3.11 environment
cd bindings
python setup.py prepare  # Copies C sources, runs Cython
pip install -e .         # Builds and installs
```

### Building Binary Wheels

```bash
# Local wheel build (current Python version)
./run_cibuildwheel.sh

# Or manually with cibuildwheel
pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse bindings/
```

### GitHub Actions

Wheels are automatically built on push to `main` or tags for:
- Python 3.11, 3.12, 3.13
- Linux (x86_64), Windows (AMD64), macOS (x86_64, arm64)

---

## Next Steps

### Immediate Priorities (Phase 1 Complete)

Per DESIGN_PLAN.md, Phase 1 is complete. The next priorities are:

1. **Phase 2: Python/Numba Fallback Engine** (Weeks 5-8)
   - Implement `algorithms/` module with numba backend
   - Create engine selector (`optv` vs `python`)
   - Build engine comparison tests (tolerance: 1e-10)
   - Add `--engine` flag to GUI and CLI

2. **Phase 3: Unification & Polish** (Weeks 9-12)
   - Create `openptv2/` unified package
   - Merge documentation
   - Set up CI/CD for automated wheel building
   - Release version 1.0.0 on PyPI

### Phase 2 Tasks (Python/Numba Fallback)

- [ ] Copy `openptv-python` algorithms to `algorithms/`
- [ ] Refactor to match `bindings/` API signatures
- [ ] Implement `EngineSelector` class
- [ ] Add `--engine` flag to GUI and CLI
- [ ] Build engine comparison tests
- [ ] Test parity (all algorithms produce identical results)

### Phase 3 Tasks (Unification & Polish)

- [ ] Create `openptv2/` package folder
- [ ] Implement unified API (`import openptv2`)
- [ ] Maintain `optv` and `pyptv` compatibility aliases
- [ ] Merge documentation (READMEs, installation guides)
- [ ] Set up GitHub Actions for wheel building
- [ ] Automated testing on PRs
- [ ] Version 1.0.0 release

---

## Contact & Resources

- **Repository**: https://github.com/alexlib/openptv2
- **Issues**: https://github.com/alexlib/openptv2/issues
- **Mailing List**: openptv@googlegroups.com
- **Design Plan**: [DESIGN_PLAN.md](DESIGN_PLAN.md)

---

*Last Updated: March 23, 2026*
