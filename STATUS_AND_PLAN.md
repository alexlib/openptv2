# openptv2 Development Status

**Date**: March 22, 2026  
**Version**: 1.0.0 (development)  
**Repository**: https://github.com/alexlib/openptv2

---

## Current Status Summary

### ✅ Completed (Phase 1)

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

#### Documentation
- [x] README.md updated with installation and usage instructions
- [x] Developer guide created (`docs/developer_guide/building.md`)
- [x] Documentation index created (`docs/index.md`)

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

### ⚠️ Known Issues

#### 1. Python/Numba Engine Not Implemented

**Problem**: `algorithms/` module imports but numba backend not functional.

**Status**: Phase 2 work - algorithms copied from openptv-python but not integrated.

**Impact**:
- Only optv (C/Cython) engine available
- Cannot compare engine results
- No pure Python fallback option

---

#### 2. Test Coverage Incomplete

**Problem**: Many integration tests have collection errors or skip.

**Status**: Test fixtures and data need updating.

**Impact**:
- Cannot run full integration test suite
- No automated regression testing

---

#### 3. GUI Not Yet Tested

**Problem**: GUI has not been tested with local bindings.

**Status**: Pending verification.

**Impact**:
- Phase 1 completion not fully verified
- End-to-end workflow unconfirmed

---

## Phase Status Overview

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Copy & Integrate | ✅ Complete (GUI test pending) |
| Phase 2 | Python/Numba Fallback | ⏳ Not started |
| Phase 3 | Unification & Polish | ⏳ Not started |

*See DESIGN_PLAN.md for detailed phase descriptions and success criteria.*

---

## Build Instructions

### Building from Source

```bash
# Using Python 3.11 environment
cd bindings
python setup.py prepare
pip install -e .
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

Per DESIGN_PLAN.md, the next priorities are:

1. **Test GUI** - Verify Phase 1 completion
2. **Phase 2** - Implement Python/Numba fallback engine
3. **Phase 3** - Unification and polish

---

## Contact & Resources

- **Repository**: https://github.com/alexlib/openptv2
- **Issues**: https://github.com/alexlib/openptv2/issues
- **Mailing List**: openptv@googlegroups.com
- **Design Plan**: [DESIGN_PLAN.md](DESIGN_PLAN.md)

---

*Last Updated: March 22, 2026*
