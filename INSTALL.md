# OpenPTV2 Installation Guide

**One command to build everything.**

---

## 🚀 Quick Start

### For Developers (Build from Source)

**Prerequisites:** Python 3.11+, C/C++ compiler (`gcc`/`clang`/`MSVC`), Cython, NumPy.

```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
uv sync --extra dev          # Installs dependencies and compiles the Cython extensions in-place
```

That's it. Everything builds automatically in one step.

---

## 🛠️ How the Build Works

OpenPTV2 compiles standard annotated Python algorithms into high-performance native machine code via **Cython 3 (Pure Python Mode)**.

```
┌─────────────────────────────────┐     ┌────────────────────────────────┐     ┌───────────────────────────────┐
│     src/openptv2/algorithms/    │────▶│       Cython Compilation       │────▶│       Shared Libraries        │
│    (Annotated Python Sources)   │     │ (setup.py compiles to C/C++)   │     │      (High-Performance)       │
└─────────────────────────────────┘     └────────────────────────────────┘     └───────────────────────────────┘
```

**Build Flow:**
1. During `uv sync` or `pip install -e .`, `setup.py` triggers `cythonize` on standard Python modules under `src/openptv2/algorithms/`.
2. Cython compiles the annotated python files (`.py`) into intermediate C files (`.c`), using type hints and compiler directives.
3. The host C compiler compiles and links these files into native binary shared libraries (`.so`/`.pyd` files) inside the package.
4. Python imports are routed through these precompiled native libraries, running at native C-level speed.

---

## 📋 Detailed Instructions

### Step 1: System Dependencies

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install -y gcc gcc-c++ python3-devel
```

**macOS:**
```bash
xcode-select --install   # Installs clang
```

**Windows:**
- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Select "Desktop development with C++" workload

---

### Step 2: Build and Install

**Option A — Using uv (Recommended):**
```bash
uv sync --extra dev
```

**Option B — Using pip (Standard editable install):**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

### Step 3: Verify the Installation

> [!IMPORTANT]
> Always run your verification and testing commands with `uv run` or within the active virtual environment to ensure the correct precompiled binaries are targeted.

**1. Test Imports and Compilation Status:**
```bash
# Verify that the core module imports and detects the compiled state
uv run python -c "from openptv2.algorithms.trafo import img_coord; print('OpenPTV2 Algorithms: OK')"
```

**2. Run the Test Suite:**
To run all tests including unit and parity validation:
```bash
uv run pytest -v
```

---

## 🎨 GUI Launch and Setup

The modern desktop interface is implemented in Python utilizing standard **Tkinter** and **ttkbootstrap** for responsive, high-performance styling.

### Launch the GUI:

```bash
uv run pyptv_gui
```

*Or via the explicit module path:*
```bash
uv run python -m openptv2.gui.pyptv.pyptv_gui
```

---

## 📦 Binary Wheels (Deployment)

To build redistributable precompiled wheels for distribution:

```bash
pip install build
python -m build --wheel
```

This produces platform-specific precompiled binaries under `dist/`.

---

## 🔍 Troubleshooting

### "Cython not found" or compilation fails
Ensure that you are running within your virtual environment or prefixing your commands with `uv run`. To manually install pre-requisites:
```bash
pip install "cython>=3.0.0" "numpy>=2.0.0" setuptools
```

### Force Clean Rebuild
If Cython fails to pick up changes or gets into a corrupted build state, clean the build directories:
```bash
# Delete build and compiled cache files
rm -rf build/ dist/ src/openptv2/algorithms/*.c src/openptv2/algorithms/*.so src/openptv2/algorithms/*.pyd src/openptv2.egg-info/
uv sync --extra dev
```

---

## 📂 Architecture Summary

| Component | Path | What it does |
|-----------|------|-------------|
| **Core Algorithms** | `src/openptv2/algorithms/` | High-performance compiled single-source modules |
| **Namespace Exports** | `src/openptv2/` | Backward compatibility layer and public module exports |
| **Tkinter GUI** | `src/openptv2/gui/pyptv/` | ttkbootstrap-powered interactive user interface |
| **Tests** | `tests/` | Integrated Unit, Parity, and GUI automated test suites |
