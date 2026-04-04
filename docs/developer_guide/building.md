# Building openptv2 from Source

This document provides instructions for building openptv2 from source for development.

## Prerequisites

### Required Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11, 3.12, 3.13 | Runtime |
| CMake | 3.15+ | C library build |
| C Compiler | gcc/clang/MSVC | C/C++ compilation |
| Cython | 3.0+ | Python bindings |
| NumPy | 2.0+ | Array operations |

### Recommended: uv Package Manager

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quick Start (Development Installation)

### 1. Clone the Repository

```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
```

### 2. Install Dependencies and Build

**Using uv (recommended):**
```bash
uv sync --extra dev
```

**Using pip:**
```bash
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
python -c "import openptv2; print(openptv2.__version__)"
python -c "from openptv2 import Tracker; print('OK')"
```

---

## Detailed Build Steps

### Step 1: Install System Dependencies

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install -y cmake gcc gcc-c++ python3-devel
```

**macOS:**
```bash
xcode-select --install
brew install cmake  # Optional
```

**Windows:**
- Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [CMake](https://cmake.org/download/)

### Step 2: Set Up Virtual Environment

**Using uv:**
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

**Using venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Step 3: Install Python Dependencies

**Using uv:**
```bash
uv sync --extra dev
```

**Using pip:**
```bash
pip install scikit-build-core cython numpy>=2.0.0
pip install -e ".[dev]"
```

### Step 4: Build Process (Automatic)

When you run `uv sync` or `pip install -e .`, the build system automatically:

1. **Configures CMake** for the C library (`lib/`)
2. **Compiles C code** into a static library
3. **Runs Cython** on all `.pyx` files in `bindings/optv/`
4. **Compiles Cython extensions** and links with C library
5. **Installs** the `optv` package to site-packages
6. **Installs** the `openptv2` Python package
7. **Installs** the `algorithms` package (Python/Numba engine)
8. **Installs** the `gui` package

### Step 5: Verify Build

```bash
# Test imports
python -c "import openptv2; print('openptv2:', openptv2.__version__)"
python -c "import optv; print('optv:', optv.__version__)"
python -c "from openptv2.algorithms import numba_impl; print('algorithms: OK')"

# Test engine info
python -c "import openptv2; print(openptv2.get_engine_info())"

# Run tests
pytest tests/ -v --tb=short
```

---

## Build Architecture

### Build Flow

```
┌─────────────────┐
│  C source (lib/)│
└────────┬────────┘
         │ CMake compiles
         ▼
┌─────────────────┐
│  Static library │
└────────┬────────┘
         │ Cython wraps
         ▼
┌─────────────────┐
│  optv package   │
│  (site-packages)│
└─────────────────┘
```

### Build System Components

| Component | Tool | Purpose |
|-----------|------|---------|
| C library | CMake | Cross-platform C build |
| Cython bindings | scikit-build-core + Cython | Wheel building |
| Python package | setuptools | Package installation |
| Dependency management | uv/pip | Package management |

### Project Structure and Build

```
openptv2/
├── lib/                    # C source → compiled by CMake
│   ├── src/
│   ├── include/
│   └── CMakeLists.txt
├── bindings/               # Cython sources → compiled by scikit-build-core
│   ├── optv/*.pyx         # Cython modules
│   ├── optv/*.pxd
│   └── pyproject.toml     # scikit-build-core config
├── openptv2/              # Pure Python → installed as-is
│   └── *.py
├── algorithms/            # Pure Python (Numba) → installed as-is
│   └── *.py
├── gui/                   # Pure Python (TraitsUI) → installed as-is
│   └── pyptv/
└── CMakeLists.txt         # Root build config
```

---

## Manual Build (Advanced)

### Build C Library Only

```bash
cd lib
mkdir build && cd build
cmake ..
make
```

### Build Cython Bindings Only

```bash
cd bindings
pip install scikit-build-core cython numpy
python -m build --wheel
```

### Install Without Editable Mode

```bash
# Build wheel
python -m build

# Install wheel
pip install dist/openptv2-*.whl
```

---

## Development Workflow

### 1. Make Changes

Edit source files in:
- `lib/src/` - C code (requires rebuild)
- `bindings/optv/` - Cython code (requires rebuild)
- `openptv2/` - Python code (no rebuild needed)
- `algorithms/` - Python code (no rebuild needed)
- `gui/` - Python GUI code (no rebuild needed)

### 2. Rebuild (if C/Cython changed)

```bash
# Using uv
uv sync --extra dev

# Using pip
pip install -e ".[dev]"
```

### 3. Run Tests

```bash
# All tests
pytest

# Specific test categories
pytest tests/integration/ -v
pytest tests/engine_comparison/ -v
pytest bindings/tests/ -v
pytest gui/tests/ -v

# C library tests
cd lib && mkdir build && cd build && cmake .. && ctest
```

### 4. Check Code Quality

```bash
# Type checking
mypy openptv2/ algorithms/

# Linting
ruff check openptv2/ algorithms/ gui/

# Format code
ruff format openptv2/ algorithms/ gui/
```

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'optv'"

The C/Cython bindings weren't built. Run:

```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

#### 2. "CMake not found"

Install CMake:

```bash
# Linux
sudo apt-get install cmake  # Debian/Ubuntu
sudo dnf install cmake      # Fedora

# macOS
brew install cmake

# Windows
# Download from https://cmake.org/download/
```

#### 3. "Cython not found"

```bash
pip install cython>=3.0.0
```

#### 4. "ImportError: file too short" or ".so corrupted"

Clean and rebuild:

```bash
# Remove build artifacts
rm -rf build/ .eggs/ *.egg-info

# Remove compiled extensions
find . -name "*.so" -delete

# Rebuild
uv sync --extra dev
```

#### 5. C Compilation Errors

**Linux:** Ensure build tools are installed:
```bash
sudo apt-get install build-essential
```

**macOS:** Install Xcode Command Line Tools:
```bash
xcode-select --install
```

**Windows:** Install MSVC Build Tools from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

#### 6. NumPy Version Mismatch

```bash
pip install numpy>=2.0.0
```

#### 7. "No module named 'algorithms'"

The algorithms package should be installed. Check:

```bash
pip install -e ".[dev]"
```

---

## Building Binary Wheels

For building portable binary wheels for distribution (manylinux, macOS, Windows), see:

- [Building Binary Wheels](../../BUILDING_BINARY_WHEELS.md) - Complete guide

Quick commands:

```bash
# Linux manylinux2014 wheels
CIBW_BUILD="cp311-*" CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014" \
  uvx cibuildwheel --platform linux --output-dir ./dist

# Build all platforms via CI
# See .github/workflows/cibuildwheel.yml
```

```bash
# Install build tools
pip install build

# Build
python -m build

# Output:
# dist/openptv2-1.0.0.tar.gz      (source distribution)
# dist/openptv2-1.0.0-*.whl       (wheel)
```

### Build with scikit-build-core

```bash
pip install scikit-build-core
python -m build --wheel
```

### Platform-Specific Notes

**Linux:**
- Use `auditwheel` to check wheel compatibility:
  ```bash
  auditwheel repair dist/*.whl
  ```

**macOS:**
- Set architecture for Apple Silicon:
  ```bash
  export ARCHFLAGS="-arch arm64"
  ```
- For Intel:
  ```bash
  export ARCHFLAGS="-arch x86_64"
  ```

**Windows:**
- Use `delvewheel` to repair wheels:
  ```bash
  delvewheel repair dist/*.whl
  ```

---

## Performance Tips

### Release Build

```bash
# Set build type
export CMAKE_BUILD_TYPE=Release

# Build
uv sync --extra dev --no-build-isolation
```

### Compiler Optimizations

```bash
# GCC/Clang optimizations
export CFLAGS="-O3 -march=native"
export CXXFLAGS="-O3 -march=native"

# Build
uv sync --extra dev
```

---

## Continuous Integration

The project uses GitHub Actions to:
- Build wheels for Linux, Windows, macOS
- Run tests on pull requests
- Upload wheels to PyPI on release

See [Publishing to PyPI](#publishing-to-pypi) for detailed CI/CD and release instructions.

---

## Publishing to PyPI

This section covers how to release openptv2 to PyPI, including version management and CI/CD setup.

### Version Management

The version is defined in `pyproject.toml`:

```toml
[project]
version = "1.0.0"
```

To update the version:

```bash
# Edit pyproject.toml and change the version
vim pyproject.toml
# Or use sed
sed -i 's/version = "1.0.0"/version = "1.0.1"/' pyproject.toml
```

### CI/CD: Automatic Publishing

The project uses GitHub Actions to automatically build and publish wheels to PyPI when a tag is pushed.

#### Workflow: `.github/workflows/cibuildwheel.yml`

The workflow has these jobs:

1. **build_wheels** - Builds wheels for all platforms (Linux, Windows, macOS) and Python versions (3.11-3.14)
2. **build_sdist** - Builds source distribution
3. **test_package** - Verifies wheel installation and runs tests
4. **upload_pypi** - Uploads to PyPI (only on tags)

#### Trigger Conditions

Publishing happens automatically when:

```yaml
on:
  push:
    tags: ['v*', '[0-9]*']  # Tags like v1.0.0, v2.1.3
```

#### How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Push tag   │────▶│ CI builds   │────▶│  Upload to  │
│  v1.0.0     │     │  wheels     │     │   PyPI      │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Creating a Release

#### Step 1: Update Version

```bash
# Edit version in pyproject.toml
vim pyproject.toml
```

#### Step 2: Commit and Tag

```bash
# Commit the version change
git add pyproject.toml
git commit -m "Bump version to 1.0.0"

# Create an annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push the tag (this triggers the workflow)
git push origin v1.0.0
```

#### Step 3: Verify

1. Check GitHub Actions workflow run
2. Verify wheels built for all platforms
3. Check PyPI for the new release: https://pypi.org/project/openptv2/

### Manual Publishing (Without CI)

If you need to publish manually:

```bash
# Install build tools
pip install build twine

# Build source and wheels
python -m build

# Check the built files
ls dist/

# Test upload to Test PyPI first
twine upload --repository testpypi dist/*
# Test: pip install --index-url https://test.pypi.org/simple/ openptv2

# Upload to production PyPI
twine upload dist/*
```

### PyPI Setup (First Time)

#### 1. Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create an account
3. Create an API token at https://pypi.org/manage/account/token/

#### 2. Configure Trusted Publishing (Recommended)

For GitHub Actions to publish securely:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **Project name**: `openptv2`
   - **Owner**: Your GitHub username/organization
   - **Repository**: `openptv2`
   - **Workflow**: `cibuildwheel.yml`

This enables OIDC authentication without storing API tokens.

#### 3. Set Up GitHub Secrets (Legacy)

If not using trusted publishing:

1. Go to GitHub repo → Settings → Secrets and variables → Actions
2. Add secret:
   - Name: `PYPI_API_TOKEN`
   - Value: Your PyPI API token

### CI/CD Workflow Details

#### Build Matrix

| OS | Python | Architectures |
|----|--------|---------------|
| ubuntu-latest | cp311-cp313 | x86_64, aarch64 (manylinux2014) |
| windows-latest | cp311-cp313 | AMD64 |
| macos-latest | cp311-cp313 | x86_64, arm64 |

#### Wheel Outputs

```
dist/
├── openptv2-1.0.0-cp311-cp311-manylinux2014_x86_64.whl
├── openptv2-1.0.0-cp312-cp312-manylinux2014_x86_64.whl
├── openptv2-1.0.0-cp311-cp311-macosx_x86_64.whl
├── openptv2-1.0.0-cp311-cp311-macosx_arm64.whl
├── openptv2-1.0.0-cp311-cp311-win_amd64.whl
└── ...
```

#### Testing in CI

Each wheel is tested during build:

```bash
CIBW_TEST_COMMAND: "python -c \"import openptv2; print('OK')\""
```

Full test suite runs in `test_package` job (tags only).

### Release Checklist

Before creating a release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Run tests locally: `pytest`
- [ ] Push changes to main branch
- [ ] Create and push tag: `git tag -a vX.Y.Z && git push origin vX.Y.Z`
- [ ] Verify CI builds succeed
- [ ] Verify PyPI shows new release
- [ ] Test installation: `pip install openptv2`

### Troubleshooting

#### "Permission denied" when uploading

- Check API token is correct
- Ensure trusted publishing is configured
- Verify project name matches

#### "File already exists" on PyPI

- Use `skip-existing: true` in the upload action
- Or increment version number

#### Build fails in CI

- Check GitHub Actions logs
- Common issues: missing dependencies, wrong Python version

---

## Getting Help

If you encounter issues:

1. Check this document
2. Search existing [GitHub issues](https://github.com/openptv/openptv2/issues)
3. Ask on the [mailing list](https://groups.google.com/g/openptv)
4. Check the [FAQ](docs/faq.md) (if available)

---

## License

LGPL-3.0 or later. See [LICENSE](LICENSE) for details.
