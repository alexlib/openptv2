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
- [x] scikit-build-core configuration for Cython bindings (`bindings/pyproject.toml`)
- [x] Main project configuration (`pyproject.toml`)
- [x] Documentation for building (`docs/developer_guide/building.md`)

#### Dependencies
- [x] Python 3.11 environment created (`.venv311/`)
- [x] optv package installed from PyPI (v0.3.2) - pre-built C/Cython bindings
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

### ⚠️ Known Issues

#### 1. Local Cython Bindings Build Fails

**Problem**: Building optv from local `bindings/` source fails with C++ compilation errors.

**Error**:
```
fatal error: optv/vec_utils.h: No such file or directory
```

**Root Cause**: 
- Cython-generated C++ code uses `#include "optv/vec_utils.h"` format
- Header files need to be in `lib/include/optv/` subdirectory (done)
- C++ complex type declarations conflict with Cython-generated code

**Current Workaround**: Use pre-built optv package from PyPI (v0.3.2)

**Impact**: 
- Development requires PyPI package, not local source
- Cannot test local C code changes easily
- Build reproducibility depends on external package

---

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

### Phase 1B: Fix Local Build (Priority: HIGH)

**Goal**: Build optv from local source successfully

#### Task 1.1: Fix Header Include Paths
- [ ] Update Cython `.pxd` files to use correct include paths
- [ ] Modify CMakeLists.txt to copy headers to build directory
- [ ] Test: `cd bindings && uv pip install --no-build-isolation .`

#### Task 1.2: Fix C++ Compilation
- [ ] Add `#include <complex>` to Cython-generated code
- [ ] Or switch to C mode: `cython --c` instead of `--cplus`
- [ ] Update `bindings/CMakeLists.txt` Python_add_library flags
- [ ] Test: Build completes without errors

#### Task 1.3: Create Build Script
- [ ] Create `scripts/build_bindings.sh` for reproducible builds
- [ ] Document build process in README
- [ ] Test: Fresh environment can build from source

**Estimated Time**: 4-8 hours  
**Success Criteria**: `uv pip install bindings/` produces working optv package

---

### Phase 2: Python/Numba Engine (Priority: MEDIUM)

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

### Phase 4: CI/CD Setup (Priority: LOW)

**Goal**: Automated testing and wheel building

#### Task 4.1: GitHub Actions
- [ ] Create `.github/workflows/test.yml` for PR testing
- [ ] Create `.github/workflows/build.yml` for wheel building
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
1. **Fix local build** (Phase 1B) - Enable local development
2. **Test GUI** (Phase 3.1) - Verify end-to-end workflow
3. **Document issues** - Create GitHub issues for tracking

### Next Week
1. **Start Phase 2** - Integrate Python/Numba engine
2. **Write engine comparison tests** - Ensure parity
3. **Update README** - Document current state

### This Month
1. **Complete Phases 1B, 2, 3** - Core functionality
2. **Set up CI/CD** (Phase 4) - Automated testing
3. **Prepare v1.0.0 release** - First official release

---

## Technical Debt

| Issue | Impact | Priority |
|-------|--------|----------|
| Local build fails | Cannot test C changes | HIGH |
| No Python engine | No fallback option | MEDIUM |
| Incomplete tests | No regression testing | MEDIUM |
| No CI/CD | Manual testing required | LOW |
| Sparse docs | Hard for new users | LOW |

---

## Success Metrics

### Phase 1 Complete (Current State)
- [x] Repository structure matches design plan
- [x] All imports work with PyPI optv
- [x] Basic tests pass
- [ ] Local build from source works ⚠️

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

## Contact & Resources

- **Repository**: https://github.com/alexlib/openptv2
- **Issues**: https://github.com/alexlib/openptv2/issues
- **Mailing List**: openptv@googlegroups.com
- **Design Plan**: DESIGN_PLAN.md

---

*Last Updated: March 22, 2026*
