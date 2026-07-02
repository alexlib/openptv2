# Packaging, Wheels, & Releases

This guide provides instructions on how to install OpenPTV2 from different sources, build optimized binary wheels, and manage package releases on GitHub and PyPI.

---

## 1. Installation Methods

OpenPTV2 can be installed either as a precompiled release package (recommended for users) or directly from the source repository (recommended for developers and contributors).

### Method A: Installing from PyPI
Installing from PyPI provides the easiest setup by fetching precompiled binary wheels specifically built for your operating system and Python version.

```bash
# Install the core tracking engine
pip install openptv2

# Install with Matplotlib GUI support
pip install "openptv2[gui]"

# Alternatively, using the fast 'uv' manager:
uv pip install "openptv2[gui]"
```

### Method B: Installing from Git Source
If you need the latest development features or plan to modify the codebase, you can install directly from GitHub.

#### 1. Standard Git Installation (Direct)
To pull and install the latest commit from the main branch:
```bash
# Standard installation
pip install git+https://github.com/openptv/openptv2.git

# With GUI dependencies
pip install "openptv2[gui] @ git+https://github.com/openptv/openptv2.git"
```

#### 2. Editable Development Installation
To clone and install the repository in editable mode so changes to the code are immediately reflected:
```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2

# Option 1: Using 'uv' (recommended)
uv sync --extra dev

# Option 2: Using standard 'pip'
pip install scikit-build-core cython numpy>=2.0.0
pip install -e ".[dev]"
```

---

## 2. Creating Binary Wheels

Because OpenPTV2 contains a high-performance C library (`liboptv`) linked via Cython, it must be compiled into native machine code (a binary wheel) to achieve optimal performance. We use `cibuildwheel` to automate compilation.

### Local Platform Wheel Build
To build a binary wheel targeted for your local operating system and active Python version (e.g. `cp313` on Linux):

```bash
# Run the packaging test pipeline
uv run python scripts/wheel_test_pipeline.py
```

This script automatically:
1. Calls `cibuildwheel` targeting the current platform and active Python runtime.
2. Compiles the static C libraries and runs the Cython wrappers.
3. Places the resulting `.whl` file inside the `wheelhouse/` directory.

### Multi-Platform Production Build (via CI/CD)
To build production wheels for all platforms (Windows, macOS Intel/Arm, Linux manylinux), we use GitHub Actions. The matrix is defined in `.github/workflows/cibuildwheel.yml`:

- **Linux**: Compiled inside a `manylinux2014` Docker container for broad GLIBC compatibility.
- **macOS**: Built on macOS runners generating universal binaries (`x86_64` and `arm64`).
- **Windows**: Built using MSVC compiler tools targeting `AMD64`.

---

## 3. Releasing on GitHub and PyPI

Releasing a new version is fully automated via our CI/CD pipeline using **Trusted Publishing (OIDC)** on PyPI and automated GitHub Release attachments.

### Step 1: Update the Version
1. Open `pyproject.toml` and locate the `[project]` configuration block.
2. Increment the version string (following [Semantic Versioning](https://semver.org/)):
   ```toml
   [project]
   name = "openptv2"
   version = "1.0.1" # Update this line
   ```
3. Commit the change:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 1.0.1"
   ```

### Step 2: Push a Release Tag
Pushing an annotated Git tag matching the version pattern (e.g., `v*` or `[0-9]*`) will automatically trigger the compilation and release pipelines:

```bash
# Create annotated tag
git tag -a v1.0.1 -m "Release version 1.0.1"

# Push tag to GitHub
git push origin v1.0.1
```

### Step 3: PyPI Automated Publishing (OIDC)
When the release tag is pushed, the `.github/workflows/cibuildwheel.yml` action triggers:
1. **Compilation**: Launches parallel compilation tasks for all matrix platforms using `cibuildwheel`.
2. **Packaging**: Creates the source distribution (`.tar.gz`).
3. **PyPI Upload**: Authenticates with PyPI using **Trusted Publishing (OIDC)** (no password/token storage needed) and uploads all generated wheel assets to PyPI.

### Step 4: GitHub Release Creation
The CI/CD workflow automatically creates a GitHub Release for your tag:
1. Gathers all `.whl` binaries and the `.tar.gz` source package.
2. Creates a GitHub Release drafts with those assets attached as downloadable release artifacts.
