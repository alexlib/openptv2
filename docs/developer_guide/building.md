# Building OpenPTV2 from Source

This document provides instructions for compiling, building, and running OpenPTV2 from source in a developer environment.

---

## 1. Prerequisites

### Required Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11, 3.12, 3.13 | Runtime Environment |
| C Compiler | gcc / clang / MSVC | Compiling Cython modules into native machine extensions |
| Cython | 3.0+ | Code Compilation / Translation |
| NumPy | 2.0+ | Fast Vector & Array Operations |

### Recommended: uv Package Manager

We strongly recommend using [uv](https://github.com/astral-sh/uv) by Astral for extremely fast, reproducible virtualenv setup and package management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Quick Start Development Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
```

### Step 2: Install and Compile in Editable Mode

By default, the unified `setup.py` build script compiles all 18 performance-critical modules in `src/openptv2/algorithms/` in-place using standard system compilation tools.

```bash
# Using uv (strongly recommended):
uv sync --extra dev --all-extras

# Alternatively, using pip:
uv pip install -e .
```

### Step 3: Verify the Installation

```bash
# Verify version
uv run python -c "import openptv2; print(openptv2.__version__)"

# Verify active compile status
uv run python -c "import openptv2; print(openptv2.get_runtime_info())"
```
It should report: `engine: 'cython'`, `compiled: True`.

---

## 3. Build Architecture

OpenPTV2 compiles a single codebase. All mathematical core algorithms are written in standard Python files in `src/openptv2/algorithms/*.py` utilizing Cython 3 type-annotations and decorators.

```
┌──────────────────────────────────────┐
│  src/openptv2/algorithms/*.py        │  ◄── Standard Python Source Code
└──────────────────┬───────────────────┘
                   │
                   │ cythonize
                   ▼
┌──────────────────────────────────────┐
│  src/openptv2/algorithms/*.c         │  ◄── Generated C Source Code
└──────────────────┬───────────────────┘
                   │
                   │ gcc / clang / MSVC compiling
                   ▼
┌──────────────────────────────────────┐
│  src/openptv2/algorithms/*.so        │  ◄── Native Compiled Machine Extension
└──────────────────────────────────────┘
```

### Benefits of this Layout
1. **Single Source of Truth**: There are no separate C libraries or independent binding definitions to synchronize. Everything resides inside the Python module tree.
2. **Double Nature**: Delete the compiled `.so` / `.pyd` files, and the code runs as regular, interpreted Python code (perfect for step-by-step IDE debuggers).
3. **Editable compilation**: Run `uv pip install -e .` once, and any modification to `.py` source files will automatically trigger recompilation upon import or test execution.

---

## 4. Development Workflow & Commands

### Code Recompilation
If you modify Cython-annotated code in `src/openptv2/algorithms/` and want to compile it:
```bash
# Trigger an automatic re-compiles / install
uv pip install -e .

# Or compile Extensions in-place explicitly
uv run python setup.py build_ext --inplace
```

To enable faster compile times during local rapid-prototyping, you can compile with lower optimization flags (bypassing slow `-O3` vectorizations) using:
```bash
DEV_BUILD=1 uv pip install -e .
```

### Running Tests
Our tests are located under `/tests`:
```bash
# Run all unit tests (excluding slow tracking tests)
uv run pytest -m "not slow" -v

# Run the complete batch tests suite
uv run pytest tests/batch/ -v

# Run the GUI tests headlessly
uv run pytest tests/gui/ -v
```

### Code Formatting, Quality & Types
We use `ruff` for extremely fast linting and formatting, and `mypy` for static type verification.

```bash
# Format the codebase
uv run ruff format .

# Check Lint constraints
uv run ruff check .

# Static type verification
uv run mypy src/openptv2/
```

---

## 5. Building Binary Wheels

OpenPTV2 uses `cibuildwheel` inside a GitHub Actions CI pipeline to compile native binary wheels across multiple platforms (Linux manylinux, macOS Apple Silicon/Intel, and Windows AMD64) and Python versions.

### Local Wheels Build Verification
To test compile wheels locally on your machine:
```bash
# Install cibuildwheel
uv pip install cibuildwheel build

# Build local target platforms wheels
uv run python -m build
```
This produces source distributions and binary wheel `.whl` files inside the `dist/` folder.
