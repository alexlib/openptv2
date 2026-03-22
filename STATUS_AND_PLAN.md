# openptv2 Development Status & Next Steps Plan

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
- [x] **optv built from local source (v0.3.1)** - working!
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

#### 2. Python/Numba Engine Not Implemented

**Problem**: `algorithms/` module imports but numba backend not functional.

**Status**: Phase 2 work - algorithms copied from openptv-python but not integrated.

**Impact**:
- Only optv (C/Cython) engine available
- Cannot compare engine results
- No pure Python fallback option

---

#### 3. Test Coverage Incomplete

**Problem**: Many integration tests have collection errors or skip.

**Status**: Test fixtures and data need updating.

**Impact**:
- Cannot run full integration test suite
- No automated regression testing

---

## Next Steps Plan

### Phase 2: Python/Numba Engine (Priority: HIGH)

**Goal**: Dual-engine architecture with identical results

#### Task 2.1: Integrate algorithms Package
- [ ] Fix algorithms imports (openptv_python dependency)
- [ ] Create `algorithms/numba_impl.py` entry point
- [ ] Add numba to dependencies
- [ ] Test: `from openptv2.algorithms import numba_impl`

#### Task 2.2: Engine Selector
- [ ] Update `openptv2/engine.py` to use algorithms.numba_impl
- [ ] Add `set_engine("python")` support
- [ ] Add `set_engine("optv")` support
- [ ] Test: Both engines can be selected

#### Task 2.3: Engine Comparison Tests
- [ ] Create test fixtures for comparison
- [ ] Implement `validate_all()` function
- [ ] Add `--validate-engine` CLI flag
- [ ] Test: Both engines produce identical results (tolerance 1e-10)

**Estimated Time**: 16-24 hours
**Success Criteria**: `openptv2.set_engine("python")` works, engine comparison tests pass

---

### Phase 3: GUI Integration (Priority: MEDIUM)

**Goal**: Fully functional GUI with local optv bindings

#### Task 3.1: GUI Testing
- [ ] Launch GUI: `openptv2-gui`
- [ ] Test basic tracking workflow
- [ ] Test calibration workflow
- [ ] Document any GUI issues

#### Task 3.2: GUI Improvements (Optional)
- [ ] Add engine selection dropdown
- [ ] Add real-time visualization
- [ ] Add marimo notebook integration
- [ ] Test: GUI works with both engines

**Estimated Time**: 8-16 hours
**Success Criteria**: GUI launches and tracks particles successfully

---

### Phase 4: CI/CD Setup (Priority: MEDIUM)

**Goal**: Automated testing and wheel building

#### Task 4.1: GitHub Actions
- [x] Create `.github/workflows/cibuildwheel.yml` for wheel building
- [ ] Create `.github/workflows/test.yml` for PR testing
- [ ] Configure cibuildwheel for multi-platform wheels
- [ ] Test: Push triggers automated builds

#### Task 4.2: PyPI Release
- [ ] Configure PyPI credentials
- [ ] Create release workflow
- [ ] Test: Tag creates release and uploads wheels

**Estimated Time**: 8-12 hours
**Success Criteria**: Every push runs tests, tags create releases

---

### Phase 5: Documentation & Examples (Priority: LOW)

**Goal**: Complete user documentation

#### Task 5.1: User Guides
- [ ] Create `docs/tutorials/quickstart.md`
- [ ] Create `docs/user_guide/gui.md`
- [ ] Create `docs/user_guide/batch.md`
- [ ] Create `docs/algorithms/` explanations

#### Task 5.2: API Documentation
- [ ] Configure Sphinx in `docs/sphinx/`
- [ ] Generate API reference from docstrings
- [ ] Deploy to GitHub Pages
- [ ] Test: Documentation builds and deploys

#### Task 5.3: Example Notebooks
- [ ] Create Jupyter/marimo notebooks
- [ ] Add example datasets
- [ ] Test: Notebooks execute successfully

**Estimated Time**: 12-20 hours
**Success Criteria**: Complete documentation deployed, examples work

---

## Immediate Next Actions

### This Week
1. **Test GUI** (Phase 3.1) - Verify end-to-end workflow with local bindings
2. **Start Phase 2** - Integrate Python/Numba engine
3. **Update README** - Document new build process

### Next Week
1. **Complete Phase 2** - Dual engine with comparison tests
2. **Set up CI/CD** (Phase 4) - Automated testing on PRs
3. **Document current state** - Update all docs

### This Month
1. **Complete Phases 2, 3, 4** - Core functionality
2. **Prepare v1.0.0 release** - First official release

---

## Technical Debt

| Issue | Impact | Priority |
|-------|--------|----------|
| No Python engine | No fallback option | HIGH |
| Incomplete tests | No regression testing | MEDIUM |
| CI/CD not fully configured | Manual testing required | MEDIUM |
| Sparse docs | Hard for new users | LOW |

---

## Success Metrics

### Phase 1 Complete ✅
- [x] Repository structure matches design plan
- [x] All imports work with local optv
- [x] Basic tests pass
- [x] Local build from source works
- [x] Wheel building configured

### Phase 2 Complete
- [ ] Both engines available
- [ ] Engine comparison tests pass
- [ ] `--engine` flag works in GUI and CLI

### Phase 3 Complete
- [ ] GUI fully functional
- [ ] End-to-end tracking works
- [ ] No critical bugs

### Phase 4 Complete
- [ ] CI runs on every PR
- [ ] Wheels built for Linux, Windows, macOS
- [ ] PyPI releases automated

### Phase 5 Complete
- [ ] Complete documentation
- [ ] Example notebooks
- [ ] v1.0.0 released

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

## Contact & Resources

- **Repository**: https://github.com/alexlib/openptv2
- **Issues**: https://github.com/alexlib/openptv2/issues
- **Mailing List**: openptv@googlegroups.com
- **Design Plan**: DESIGN_PLAN.md

---

*Last Updated: March 22, 2026*
