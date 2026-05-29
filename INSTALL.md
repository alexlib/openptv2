# OpenPTV2 Installation Guide

**One command to build everything.**

---

## Quick Start

### For Users (Pre-built Wheels — Coming Soon)

```bash
pip install openptv2
```

### For Developers (Build from Source)

**Prerequisites:** Python 3.11+, C compiler (gcc/clang/MSVC), Cython, NumPy.

```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
uv sync --extra dev          # Installs deps + builds C + Cython + all packages
```

That's it. Everything builds in one step.

---

## How the Build Works

OpenPTV2 has **four components** that must be built together:

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
│  lib/src/*.c │────▶│ Cython .pyx files│────▶│  .so modules │────▶│  Python pkg │
│  (C library) │     │  (bindings/)     │     │  (optv.*)    │     │  (import)   │
└──────────────┘     └──────────────────┘     └──────────────┘     └─────────────┘
       │                      │                      │                      │
       ▼                      ▼                      ▼                      ▼
  tracking_frame_buf.c  tracking_framebuf.pyx  optv/tracking_framebuf.cpython  from optv.tracking_framebuf import Target
  calibration.c         calibration.pyx        optv/calibration.cpython          from optv.calibration import Calibration
  ...                   ...                    ...                               ...
```

**Build flow:**

1. **C sources** (`lib/src/*.c`) are copied into `bindings/liboptv/src/`
2. **Cython** compiles each `bindings/optv/*.pyx` → `.c` file
3. **C compiler** links each Cython `.c` + all C library `.c` → `.so` extension module
4. **setuptools** installs the `.so` files + all Python packages (`openptv2/`, `algorithms/`, `gui/`)

All four steps happen automatically with `uv sync` or `pip install -e .`.

---

## Detailed Instructions

### Step 1: System Dependencies

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install -y build-essential python3-dev
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install -y gcc gcc-c++ python3-devel
```

**macOS:**
```bash
xcode-select --install   # Installs clang
brew install cmake       # Optional, for standalone C builds
```

**Windows:**
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Select "Desktop development with C++" workload

### Step 2: Clone

```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
```

### Step 3: Build and Install

**Option A — Using uv (recommended):**
```bash
uv sync --extra dev
```

**Option B — Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**Option C — Manual (for debugging the build):**
```bash
# Step 1: Prepare C sources + run Cython
cd bindings && python setup.py prepare && cd ..

# Step 2: Build extensions in place
python setup.py build_ext --inplace

# Step 3: Install everything else
pip install -e ".[dev]"
```

**Option D — Python-only (no C/Cython build):**

If you only need the pure Python algorithms engine and don't require the Cython `optv` bindings:

```bash
OPENPTV_PYTHON_ONLY=1 uv pip install -e .
```

Or add to your shell profile:
```bash
export OPENPTV_PYTHON_ONLY=1
```

This skips the Cython compilation and installs ~100x faster. Only the `algorithms/` module will be available (not `optv.*`).

### Step 4: Verify

> [!IMPORTANT]
> Always ensure you run commands within the active virtual environment where `openptv2` is installed to avoid running global python/pytest versions, which can lead to `ModuleNotFoundError` or segmentation faults due to mismatched compiled library versions.

Either activate your virtual environment first:
```bash
source .venv/bin/activate
```

Or run all verification commands with `uv run`:

**1. Test Core Modules and Bindings:**
```bash
uv run python -c "from optv.tracking_framebuf import Target; print('optv bindings: OK')"
uv run python -c "from algorithms.tracking_frame_buf import Target; print('algorithms engine: OK')"
uv run python -c "from openptv2 import Tracker; print('openptv2 unified: OK')"
uv run python -c "import gui.pyptv.pyptv_gui; print('pyptv GUI imports: OK')"
```

**2. Run the Test Suite:**
To run the standard C library and Python algorithm tests:
```bash
uv run python -m pytest algorithms/tests/ bindings/tests/ -v
```

To run all tests including the GUI:
```bash
uv run python -m pytest -v
```

---

## GUI Installation

The GUI requires additional dependencies (Qt, TraitsUI, etc.):

```bash
uv sync --all-extras
```

Launch:
```bash
openptv2-gui
# or
pyptv
```

---

## Binary Wheels (for Deployment)

To build distributable wheels for Linux, Windows, and macOS:

### Local Wheel Build

```bash
pip install build
python -m build --wheel
```

This produces `dist/openptv2-1.0.0-cp311-cp311-linux_x86_64.whl` (platform-specific).

### Multi-Platform Wheels (CI)

Use [cibuildwheel](https://cibuildwheel.pypa.io/) in GitHub Actions:

```yaml
# .github/workflows/wheels.yml
name: Build Wheels
on: [push, pull_request]

jobs:
  build_wheels:
    name: Build on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
      - uses: actions/checkout@v4

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.17
        env:
          CIBW_BUILD: "cp311-* cp312-* cp313-*"
          CIBW_SKIP: "*-musllinux_*"
          CIBW_BEFORE_BUILD: "pip install cython numpy"

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: ./wheelhouse/*.whl
```

### Publish to PyPI

```bash
pip install twine
twine upload dist/*
```

---

## Troubleshooting

### "Cython not found"
```bash
pip install cython>=3.0.0
```

### "NumPy not found"
```bash
pip install numpy>=2.0.0
```

### "C compiler error"
- **Linux:** `sudo apt-get install build-essential`
- **macOS:** `xcode-select --install`
- **Windows:** Install MSVC Build Tools

### "optv module not found" after `pip install -e .`
The C extensions may not have built. Force rebuild:
```bash
rm -rf build/ bindings/liboptv/ bindings/optv/*.c
pip install -e ".[dev]"
```

### "ImportError: undefined symbol"
The C library wasn't linked into the extension. Clean and rebuild:
```bash
rm -rf build/ bindings/liboptv/ bindings/optv/*.c bindings/optv/*.so
pip install -e ".[dev]"
```

---

## Architecture Summary

| Component | Location | What it does |
|-----------|----------|-------------|
| C library | `lib/src/` | Core algorithms (calibration, tracking, correspondence) |
| Cython bindings | `bindings/optv/*.pyx` | Python interface to C library |
| Python engine | `algorithms/` | Pure Python/Numba fallback (same algorithms) |
| Unified package | `openptv2/` | Single entry point, engine selector |
| GUI | `gui/` | TraitsUI application |

All components share the same test data in `test_data/` and are tested together.
