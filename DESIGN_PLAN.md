# openptv2 Software Design Plan

**Version**: 1.0  
**Created**: March 21, 2026  
**Status**: Approved for Implementation

---

## 1. Executive Summary

**openptv2** is a unified repository combining the best of three existing repositories:
- **openptv** — Core C library (`liboptv`) + Cython bindings (`py_bind`)
- **pyptv** — TraitsUI-based GUI for particle tracking velocimetry
- **openptv-python** — Python/Numba fallback engine concept

**Goal**: Create a single, maintainable repository with:
- Working GUI (as in pyptv)
- C source code + Cython bindings inside the repo (for context, faster bindings, lower memory, future C algorithms)
- Python/Numba fallback engine (for debugging, visualization, algorithm development)
- Identical results between engines with automated testing
- Easy installation via binary wheels (no build tools for end users)

---

## 2. Repository Structure

```
openptv2/
├── lib/                    # C core library (from openptv/liboptv)
│   ├── src/
│   ├── include/
│   ├── tests/              # C library tests (Check framework)
│   └── CMakeLists.txt
│
├── bindings/               # Cython bindings (from openptv/py_bind)
│   ├── src/
│   ├── tests/
│   └── pyproject.toml      # setuptools build config
│
├── algorithms/             # Python/Numba fallback engine (from openptv-python)
│   ├── tracking.py
│   ├── correspondence.py
│   ├── calibration.py
│   ├── image_processing.py
│   └── tests/
│
├── gui/                    # TraitsUI GUI (from pyptv)
│   ├── pyptv/
│   ├── tests/
│   └── plugins/
│
├── openptv2/               # Unified Python package (Phase 3)
│   ├── __init__.py         # Main entry point, exports all classes
│   ├── engine.py           # Engine selector (optv vs python)
│   ├── calibration.py      # Calibration wrapper
│   ├── correspondence.py   # Correspondence wrapper
│   ├── tracker.py          # Tracker wrapper
│   ├── tracking_framebuf.py # Frame buffer wrapper
│   ├── validate.py         # Validation CLI
│   └── version.py          # Version info
│
├── docs/                   # Combined documentation
│   ├── sphinx/             # API reference
│   ├── tutorials/          # User guides
│   └── algorithms/         # Algorithm explanations
│
├── tests/                  # Integration tests (full pipeline)
│   ├── integration/
│   ├── engine_comparison/  # C vs Python results
│   └── fixtures/
│
├── scripts/                # Build, version bump, CI helpers
│
├── pyproject.toml          # Main project config (uv/pip)
├── README.md
└── LICENSE
```

**Note**: The `openptv2/` folder is the **Phase 3 unification package** that provides:
- Unified API: `import openptv2` as single entry point
- Engine selection: Switch between optv (C/Cython) and python (Numba) engines
- Flexible integration: Glues together Cython bindings, Python algorithms, and GUI
- Backward compatibility: Maintains `optv` and `pyptv` API aliases

---

## 3. Build System Architecture

### 3.1 Build Toolchain

| Component | Tool | Purpose |
|-----------|------|---------|
| C library | CMake | Cross-platform C build |
| Cython bindings | scikit-build-core + Cython | Wheel building |
| Python package | uv/pip | Dependency management |
| Wheel distribution | cibuildwheel (GitHub Actions) | Multi-platform wheels |

### 3.2 Build Flow

```
┌─────────────────┐
│  C source (lib/)│
└────────┬────────┘
         │ CMake compiles
         ▼
┌─────────────────┐
│  .so/.dll/.dylib│
└────────┬────────┘
         │ Cython wraps
         ▼
┌─────────────────┐
│  Python module  │
│  (optv engine)  │
└─────────────────┘
```

### 3.3 Wheel Distribution

- **End users**: `pip install openptv2` or `uv add openptv2` — pre-built wheels
- **Developers**: `uv sync --extra dev` — builds from source
- **CI**: GitHub Actions builds wheels for Linux, Windows, macOS (x86_64 + ARM64 optional)

---

## 4. Engine Selection Architecture

### 4.1 Engine Selector

```python
# src/openptv2/__init__.py
from .engine import EngineSelector, get_engine

# Global default
DEFAULT_ENGINE = "optv"  # or "python"

# Per-call override
result = track_particles(images, engine="python")  # override
result = track_particles(images)  # uses global default
```

### 4.2 Engine Selector Class

```python
class EngineSelector:
    def __init__(self, default_engine: str = "optv"):
        self.default_engine = default_engine
        self._validate_engines()
    
    def select(self, engine: str | None = None):
        """Return engine instance (global or per-call)"""
        engine = engine or self.default_engine
        if engine == "optv":
            return self._get_optv_engine()
        elif engine == "python":
            return self._get_python_engine()
        else:
            raise ValueError(f"Unknown engine: {engine}")
    
    def _get_optv_engine(self):
        try:
            from .bindings import optv_core
            return optv_core
        except ImportError as e:
            self._fallback_reason = str(e)
            return self._get_python_engine()
    
    def _get_python_engine(self):
        from .algorithms import numba_impl
        return numba_impl
```

### 4.3 Engine Comparison Test

```python
# tests/engine_comparison/test_tracking.py
import pytest
import numpy as np
from openptv2 import track_particles

@pytest.mark.parametrize("engine", ["optv", "python"])
def test_tracking_identical_results(engine):
    images = load_test_images()
    result_optv = track_particles(images, engine="optv")
    result_python = track_particles(images, engine="python")
    
    # Floating-point tolerance: 1e-10
    np.testing.assert_allclose(
        result_optv.coordinates,
        result_python.coordinates,
        rtol=1e-10,
        atol=1e-10
    )
```

### 4.4 CLI Flag

```bash
# GUI
openptv2-gui --engine optv
openptv2-gui --engine python

# Batch
openptv2-batch params.yaml --engine optv
openptv2-batch params.yaml --engine python

# Validation (runs both, compares)
openptv2-validate params.yaml --tolerance 1e-10
```

---

## 5. Migration Strategy

### Phase 1: Copy & Integrate (Weeks 1-4)

**Goal**: Working version with existing tests passing

1. **Copy `openptv/liboptv` → `lib/`**
   - Preserve C source, headers, CMakeLists.txt
   - Keep existing C tests (Check framework)

2. **Copy `openptv/py_bind` → `bindings/`**
   - Convert to scikit-build-core build
   - Update Cython `.pyx` files for new paths

3. **Copy `pyptv/pyptv` → `gui/`**
   - Keep TraitsUI code unchanged
   - Update imports to point to new `bindings/` location

4. **Build combined wheel**
   - Test: `pip install .` works
   - Test: GUI launches and runs basic tracking

5. **Run existing tests**
   - C library tests: `cd lib && cmake && ctest`
   - Binding tests: `cd bindings && pytest`
   - GUI tests: `cd gui && pytest`

### Phase 2: Python/Numba Fallback (Weeks 5-8)

**Goal**: `--engine=python` flag works with identical results

1. **Copy `openptv-python` algorithms → `algorithms/`**
   - Refactor to match `bindings/` API signatures

2. **Implement engine selector**
   - Add `--engine` flag to GUI and CLI

3. **Build engine comparison tests**
   - Automated byte-by-byte comparison
   - Floating-point tolerance: 1e-10
   - `--validate-engine` CLI flag

4. **Test parity**
   - All algorithms produce identical results
   - Performance benchmarks (optional)

### Phase 3: Unification & Polish (Weeks 9-12)

**Goal**: Single coherent package, documentation merged

1. **Unify API**
   - `import openptv2` as single entry point
   - `openptv2/` folder provides unified interface to all components
   - Engine selector integrates optv and python engines
   - Maintain `optv` and `pyptv` compatibility aliases

2. **Merge documentation**
   - Combine READMEs, installation guides
   - Preserve existing tutorials

3. **CI/CD setup**
   - GitHub Actions: build wheels for Linux, Windows, macOS
   - Automated testing on PRs
   - Wheel upload to PyPI on tag

4. **Version 1.0.0 release**

---

## 6. Testing Infrastructure

### 6.1 Test Layers

```
┌─────────────────────────────────────┐
│  Integration Tests (full pipeline)  │
│  - Image → Results in working folder│
└─────────────────────────────────────┘
         ▲
┌─────────────────────────────────────┐
│  Engine Comparison Tests            │
│  - optv vs python identical results │
└─────────────────────────────────────┘
         ▲
┌─────────────────────────────────────┐
│  Unit Tests (per module)            │
│  - C library tests (Check)          │
│  - Binding tests (pytest)           │
│  - Algorithm tests (pytest)         │
│  - GUI tests (pytest + UI automation│
└─────────────────────────────────────┘
```

### 6.2 Test Commands

```bash
# All tests
uv run pytest

# C library tests
cd lib && mkdir build && cd build && cmake .. && ctest

# Binding tests
uv run pytest bindings/tests/

# Engine comparison
uv run pytest tests/engine_comparison/ --validate-engine

# GUI tests (headless)
uv run pytest gui/tests/ --headless

# Full integration test
uv run python -m openptv2.tests.integration test_experiment/
```

---

## 7. Documentation Plan

### 7.1 Documentation Structure

```
docs/
├── index.md                  # Landing page
├── installation.md           # Install guide
├── quickstart.md             # 5-minute tutorial
├── api/                      # Auto-generated API docs (Sphinx)
├── user_guide/
│   ├── gui.md               # GUI usage
│   ├── batch.md             # Batch processing
│   └── parameters.md        # Parameter reference
├── algorithms/
│   ├── tracking.md          # Tracking algorithm explanation
│   ├── correspondence.md    # Correspondence matching
│   └── calibration.md       # Camera calibration
├── developer_guide/
│   ├── build.md             # Building from source
│   ├── engine_selector.md   # Adding new engines
│   └── testing.md           # Running tests
└── migration/
    ├── from_optv.md         # Migrating from optv
    └── from_pyptv.md        # Migrating from pyptv
```

### 7.2 Documentation Tools

- **Sphinx**: API reference (auto-generated from docstrings)
- **MkDocs or Sphinx**: User guides (Markdown)
- **Future**: Jupyter/marimo notebooks for interactive tutorials

---

## 8. Platform Support Matrix

| Platform | Python Versions | Wheel Format | CI Builder |
|----------|-----------------|--------------|------------|
| Linux (x86_64) | 3.11, 3.12, 3.13, 3.14 | manylinux2014 | GitHub Actions |
| Windows (x86_64) | 3.11, 3.12, 3.13, 3.14 | win_amd64 | GitHub Actions |
| macOS (x86_64) | 3.11, 3.12, 3.13, 3.14 | macosx_10_15_x86_64 | GitHub Actions |
| macOS (ARM64) | 3.11, 3.12, 3.13, 3.14 | macosx_11_0_arm64 | GitHub Actions (optional) |

---

## 9. Backward Compatibility

### 9.1 API Compatibility

```python
# Old optv usage (still works)
from optv.tracking_framebuf import Target
from optv.tracker import Tracker

# New openptv2 usage
from openptv2 import Target, Tracker

# Both produce identical results
```

### 9.2 File Format Compatibility

- Working folder structure: unchanged
- Parameter YAML files: unchanged
- Experiment files: unchanged

### 9.3 CLI Compatibility

```bash
# Old pyptv command
pyptv --params parameters.yaml

# New openptv2 command (alias)
openptv2-gui --params parameters.yaml

# Both work identically
```

---

## 10. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| C/Cython build breaks on some platform | Use cibuildwheel with tested configurations |
| Engine results diverge | Automated comparison tests in CI |
| GUI breaks after refactor | Keep GUI code unchanged in Phase 1; test early |
| Migration takes too long | Phase-based approach; working version after Phase 1 |
| Users resist change | Maintain backward compatibility; deprecation warnings |

---

## 11. Success Criteria

### Phase 1 Complete (working version)

- [ ] `uv sync` builds C library + bindings + GUI
- [ ] GUI launches and tracks particles
- [ ] All existing optv + pyptv tests pass
- [ ] Wheel installs on Linux, Windows, macOS

### Phase 2 Complete (dual engine)

- [ ] `--engine=python` flag works
- [ ] Engine comparison tests pass (tolerance 1e-10)
- [ ] `--validate-engine` CLI flag implemented

### Phase 3 Complete (unified package)

- [ ] `import openptv2` works
- [ ] Documentation merged and deployed
- [ ] CI builds wheels automatically
- [ ] Version 1.0.0 released on PyPI

---

## 12. Design Decisions Log

| Decision | Date | Rationale |
|----------|------|-----------|
| Repository name: `openptv2` | 2026-03-21 | Clear evolution from existing repos |
| Directory structure: Option B (functional separation) | 2026-03-21 | Clear separation of concerns, easier maintenance |
| Build system: setuptools + Cython | 2026-03-22 | Proven approach from openptv, avoids scikit-build-core complexity |
| Engine switching: Both global and per-function | 2026-03-21 | Maximum flexibility for users and developers |
| Engine results: Automated testing with 1e-10 tolerance | 2026-03-21 | Ensure identical results between engines |
| Python/C implementations: Side-by-side permanent | 2026-03-21 | Python for debugging/development, C for production |
| Visualization: Debug mode + separate viewer + notebooks | 2026-03-21 | Multiple approaches for different use cases |
| GUI framework: TraitsUI (for now) | 2026-03-21 | Working version first, modernize later |
| Performance: Keep as-is initially | 2026-03-21 | Focus on functionality first |
| Algorithms: Keep current, future PyTorch/JAX possible | 2026-03-21 | Solid algorithms, API stability |
| Python versions: 3.11+ with 3.14+ planning | 2026-03-21 | Modern Python, forward compatible |
| Platforms: Linux, Windows, macOS (ARM64 optional) | 2026-03-21 | Major platforms first |
| Backward compatibility: Maintain optv/pyptv APIs | 2026-03-21 | Existing users' workflows must continue |
| Migration: Copy → Refactor → Unify | 2026-03-21 | Safest path to working version |
| Testing: Start with existing tests, expand later | 2026-03-21 | Preserve test coverage, add gradually |
| Documentation: Combine existing docs first | 2026-03-21 | Working docs before enhancements |
| `openptv2/` package folder | 2026-03-22 | Phase 3 unification: flexible integration layer for bindings, algorithms, and GUI |

---

## 13. Next Steps

1. **Review this design plan** — Discuss and refine
2. **Create repository structure** — Initial `openptv2` repo with directory layout
3. **Phase 1 implementation** — Copy code, get tests passing
4. **Iterate** — Phase 2 and 3 based on learnings

---

## Appendix A: Key Requirements Summary

| Requirement | Priority | Status |
|-------------|----------|--------|
| Single unified repository | High | Approved |
| Working GUI (TraitsUI) | High | Phase 1 |
| C source + Cython in repo | High | Phase 1 |
| Python/Numba fallback engine | High | Phase 2 |
| Identical results (automated testing) | High | Phase 2 |
| Binary wheels for end users | High | Phase 1 |
| Engine selector (`--engine` flag) | High | Phase 2 |
| Backward compatibility | High | All phases |
| Real-time visualization | Medium | Phase 2+ |
| Jupyter/marimo notebooks | Low | Future |
| C++ migration exploration | Low | Future |
| PyTorch/JAX algorithms | Low | Future |
| `openptv2/` unification package | High | Phase 3 - folder created, implementation pending |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **optv** | The C library + Cython bindings (from openptv repo) |
| **pyptv** | The TraitsUI GUI application |
| **openptv-python** | The Python/Numba fallback engine concept |
| **openptv2** | The unified repository combining all three |
| **Engine** | Implementation backend: `optv` (C/Cython) or `python` (Numba) |
| **scikit-build-core** | Modern build system for C/Cython Python extensions |
| **cibuildwheel** | Tool for building wheels across platforms in CI |

---

*This document is intended for use with AI coding agents (vibe coding) and human developers. Keep it updated as design decisions evolve.*
