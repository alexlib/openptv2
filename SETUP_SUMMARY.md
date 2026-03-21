# openptv2 Repository Setup Summary

**Date**: March 21, 2026  
**Status**: Phase 1 - Initial Structure Complete

---

## What Has Been Created

### Directory Structure

```
openptv2/
├── lib/                    # C core library (copied from openptv/liboptv)
│   ├── src/               # C source files
│   ├── include/           # C headers
│   ├── tests/             # C library tests (Check framework)
│   └── CMakeLists.txt
│
├── bindings/              # Cython bindings (copied from openptv/py_bind)
│   ├── src/               # Cython .pyx files
│   ├── optv/              # Python package
│   ├── test/              # Binding tests
│   ├── setup.py           # Legacy build (for reference)
│   └── pyproject.toml     # Build config
│
├── algorithms/            # Python/Numba fallback (Phase 2)
│   ├── __init__.py
│   └── numba_impl.py      # Placeholder
│
├── gui/                   # TraitsUI GUI (copied from pyptv)
│   ├── pyptv/             # GUI code
│   ├── tests/             # GUI tests
│   └── plugins/           # Plugin system
│
├── openptv2/              # Main Python package
│   ├── __init__.py        # Package entry point
│   ├── engine.py          # Engine selector
│   ├── version.py         # Version info
│   ├── validate.py        # Engine validation CLI
│   ├── tracking_framebuf.py
│   ├── tracker.py
│   ├── calibration.py
│   ├── correspondence.py
│   ├── bindings/          # C/Cython bindings stub
│   └── gui/               # GUI module
│       └── cli.py         # GUI CLI entry point
│
├── tests/                 # Integration tests (copied from pyptv)
│   ├── integration/       # Full pipeline tests
│   ├── engine_comparison/ # C vs Python tests (Phase 2)
│   └── fixtures/          # Test data
│
├── docs/                  # Documentation (Phase 3)
│   ├── sphinx/
│   ├── tutorials/
│   └── algorithms/
│
├── scripts/               # Build helpers
├── CMakeLists.txt         # Unified CMake build
├── pyproject.toml         # Main project config (scikit-build-core)
├── README.md              # Updated for openptv2
├── LICENSE                # LGPL-3.0
├── .gitignore
└── DESIGN_PLAN.md         # Full design document
```

---

## Configuration Files Created

### pyproject.toml
- Build system: scikit-build-core + Cython
- Python support: 3.11, 3.12, 3.13
- Dependencies: numpy, scipy, pyyaml
- Optional: gui, numba, test, dev, docs
- Entry points: openptv2-gui, openptv2-batch, openptv2-validate
- cibuildwheel configuration for multi-platform wheels

### CMakeLists.txt
- CMake 3.15+ required
- Builds C library from lib/
- Builds Cython extensions from bindings/
- Installs Python package to openptv2/
- Test integration

---

## Code Copied From Source Repositories

| Source | Destination | Status |
|--------|-------------|--------|
| openptv/liboptv/* | lib/ | ✓ Complete |
| openptv/py_bind/* | bindings/ | ✓ Complete |
| pyptv/pyptv/* | gui/pyptv/ | ✓ Complete |
| pyptv/tests/* | tests/ | ✓ Complete |

---

## New Code Created

| Module | Purpose | Status |
|--------|---------|--------|
| openptv2/__init__.py | Package entry, backward compat | ✓ Complete |
| openptv2/engine.py | Engine selector | ✓ Complete |
| openptv2/version.py | Version info | ✓ Complete |
| openptv2/validate.py | Engine validation CLI | ✓ Complete |
| openptv2/tracking_framebuf.py | API stub | ✓ Placeholder |
| openptv2/tracker.py | API stub | ✓ Placeholder |
| openptv2/calibration.py | API stub | ✓ Placeholder |
| openptv2/correspondence.py | API stub | ✓ Placeholder |
| openptv2/bindings/__init__.py | Bindings stub | ✓ Placeholder |
| algorithms/numba_impl.py | Python engine | Phase 2 |

---

## Next Steps (Phase 1)

1. **Update bindings CMakeLists.txt**
   - Configure scikit-build-core properly
   - Ensure Cython files compile

2. **Test build**
   ```bash
   uv sync --extra dev
   # or
   pip install -e .
   ```

3. **Run existing tests**
   ```bash
   # C library tests
   cd lib && mkdir build && cd build && cmake .. && ctest
   
   # Python tests
   pytest bindings/test/
   
   # GUI tests
   pytest gui/tests/
   ```

4. **Fix any build issues**
   - Update include paths
   - Fix Cython compilation errors
   - Ensure tests pass

---

## Phase 2: Python/Numba Fallback

- Implement algorithms/numba_impl.py
- Create engine comparison tests
- Add --engine flag to GUI and CLI
- Implement --validate-engine CLI

---

## Phase 3: Unification & Polish

- Merge documentation from all three repos
- Set up CI/CD (GitHub Actions)
- Configure cibuildwheel for wheel building
- Release version 1.0.0

---

## Commands to Try

```bash
# Check structure
tree -L 2

# Install in development mode
uv sync --extra dev

# Run tests
pytest tests/ -v

# Launch GUI (after build)
openptv2-gui

# Validate engines (after Phase 2)
openptv2-validate
```

---

**Note**: This is an initial structure. The bindings need to be properly integrated
with scikit-build-core, and the Python API stubs need to be connected to the actual
C/Cython implementations.
