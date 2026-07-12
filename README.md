# openptv2

**Unified OpenPTV**: Particle Tracking Velocimetry with dual-engine support

[![Python](https://img.shields.io/pypi/pyversions/openptv2.svg)](https://pypi.org/project/openptv2/)
[![License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://alexlib.github.io/openptv2/)
[![Coverage](https://img.shields.io/badge/coverage-22%25-red)](scripts/update_coverage_badge.sh)

## Overview

openptv2 combines the best of three repositories into a single, maintainable package:

- **C core library** (`lib/`) - High-performance particle tracking algorithms
- **Cython bindings** (`bindings/`) - Python interface to C library
- **Python/Numba fallback** (`algorithms/`) - Pure Python implementation for debugging
- **TraitsUI GUI** (`gui/`) - Full-featured graphical interface

## Key Features

- **Dual-engine architecture**: Use fast C/Cython (`optv`) or debuggable Python/Numba (`python`)
- **Identical results**: Both engines produce the same output (within floating-point tolerance)
- **Backward compatible**: Works with existing `optv` and `pyptv` code
- **Easy installation**: Pre-built wheels for Linux, Windows, macOS

---

## Installation

There are three install profiles:

| Command | For | Includes |
| --- | --- | --- |
| `pip install openptv2` | scripting + `openptv2-batch` (default) | headless algorithms/API + sequence/tracking pipeline |
| `pip install openptv2[gui]` | desktop users | default **plus** the TraitsUI/Chaco/PySide6 GUI |
| `pip install openptv2[dev]` | contributors | everything: GUI, tests, lint, type-check, notebooks, docs |

### Default (headless / batch)

The bare install *is* the headless batch runtime — the library, the
`openptv2-batch` sequence + tracking pipeline, no GUI:

```bash
uv pip install openptv2
# or
pip install openptv2
```

Don't have `uv`? `curl -LsSf https://astral.sh/uv/install.sh | sh`

### GUI (desktop users)

```bash
uv pip install openptv2[gui]
# or
pip install openptv2[gui]
```

#### Verify Installation

```bash
python -c "import openptv2; print(f'openptv2 version: {openptv2.__version__}')"
python -c "from openptv2 import Tracker; print('Tracker imported successfully')"
```

> **Note:** launching the GUI (`openptv2-gui`) requires the `[gui]` extra. A
> default install that then launches the GUI will fail with
> `ModuleNotFoundError: No module named 'traitsui'` (and chaco/PySide6) — that
> is expected; install `openptv2[gui]`.

### Docker (no Python setup required)

A single image serves both the GUI and batch. It bakes in a trimmed
`test_cavity` demo at `/demo/test_cavity` so the first run works with no data.

```bash
# Build once
docker build -t openptv2 .

# GUI on the host X display (Linux/X11), current folder mounted at /data
./docker/run-gui.sh
#   in the GUI, open /demo/test_cavity to try the baked demo

# Headless batch on your own data
docker run --rm -v "$PWD:/data" openptv2 \
  openptv2-batch /data/<experiment>/parameters_Run1.yaml <first> <last>
```

X11 notes: `run-gui.sh` handles `xhost` and mounts `/tmp/.X11-unix`. On Wayland
run `xhost +local:root` in an XWayland session; on macOS/Windows use an X
server (XQuartz / VcXsrv) and set `DISPLAY` accordingly.

**Headless cloud batch:** `Dockerfile.cloud` is a slim, no-GUI, free-threaded
3.14t image for servers/Cloud Run. See [docs/cloud-batch.md](docs/cloud-batch.md)
for the one-command install, `openptv2-batch` usage, and measured timings.

---

### For Developers (Build from Source)

**Prerequisites:**
- Python 3.11, 3.12, or 3.13
- CMake 3.15+
- C compiler (gcc on Linux, clang on macOS, MSVC on Windows)
- Cython 3.0+
- NumPy 2.0+
- uv (recommended) or pip

#### System Dependencies

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
# Install Xcode Command Line Tools
xcode-select --install

# Install cmake via Homebrew (optional, if not using system cmake)
brew install cmake
```

**Windows:**
- Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Install [CMake](https://cmake.org/download/)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
```

#### Step 2: Install Dependencies and Build

**Using uv (recommended):**
```bash
# Sync all dependencies and build the package
uv sync --extra dev
```

**Using pip:**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install build dependencies
pip install scikit-build-core cython numpy

# Install in development mode
pip install -e ".[dev]"
```

The default editable install builds the Cython 3 pure-Python modules:

```bash
uv pip install -e .
```

#### Step 3: Verify Build

Always run verification inside your virtual environment to ensure the compiled libraries are correctly found:

```bash
# Test imports
uv run python -c "import openptv2; print(f'openptv2 version: {openptv2.__version__}')"
uv run python -c "from openptv2 import Tracker; print('Tracker imported successfully')"

# Run core tests
uv run pytest tests/unit/ -v
```

---

### GUI Dependencies

The `[gui]` extra includes:
- traits, traitsui (Enthought framework)
- enable, chaco (visualization)
- PySide6 (Qt bindings)
- scikit-image, pandas, matplotlib (analysis)

---

### Installing from Binary Wheels

Pre-built manylinux wheels are available for Linux:

```bash
# Using pip
pip install openptv2

# Using uv
uv pip install openptv2
```

The wheels are compatible with glibc 2.17+ (CentOS 7, Ubuntu 14.04, Debian 8, etc.)

---

### Building Binary Wheels from Source

See [BUILDING_BINARY_WHEELS.md](BUILDING_BINARY_WHEELS.md) for detailed instructions on building portable binary wheels using cibuildwheel.

---

### Troubleshooting Installation

#### Common Issues

**1. "CMake not found"**
```bash
# Install CMake
# Linux: sudo apt-get install cmake
# macOS: brew install cmake
# Windows: Download from https://cmake.org/download/
```

**2. "C compiler not found"**
```bash
# Linux: sudo apt-get install build-essential
# macOS: xcode-select --install
# Windows: Install MSVC Build Tools
```

**3. "Cython not found"**
```bash
pip install cython>=3.0.0
```

**4. "NumPy version mismatch"**
```bash
pip install numpy>=2.0.0
```

**5. "optv module not found" (after cloning)**
```bash
# The optv package is built by CMake - you need to build the package
uv sync --extra dev
# or
pip install -e ".[dev]"
```

---

## Usage

### Basic Tracking

```python
import openptv2
from openptv2 import Tracker, detect_targets

# Load images
from skimage import io
images = [io.imread(f"cam1_{i:04d}.tif") for i in range(100)]

# Detect particles
targets = [detect_targets(img) for img in images]

# Track particles
tracker = Tracker()
tracks = tracker.track([t.coordinates for t in targets])

print(f"Found {len(tracks)} tracks")
```

### Runtime

```python
import openptv2

print(openptv2.get_runtime_info())
# {'engine': 'cython3-pure-python', 'compiled': True/False, 'package': 'openptv2'}
```

### GUI

Launch the graphical interface using the unified console scripts. Ensure your virtual environment is activated (`source .venv/bin/activate`) or prefix the commands with `uv run`:

```bash
# Launch the GUI using the standard openptv2-gui or shorter pyptv_gui shortcut
uv run pyptv_gui

# Launch the GUI in the single-engine runtime
uv run pyptv_gui --workdir=./test_data/test_cavity
```

### Batch Processing

Run high-throughput processing sequences using command-line batch utilities (ensure your virtual environment is activated or use `uv run`):

```bash
# Run batch processing with the single runtime
uv run pyptv_batch --workdir=./test_data/test_cavity --first=10000 --last=10005

# Run with legacy positional arguments (for backward compatibility)
uv run pyptv_batch ./test_data/test_cavity/parameters_Run1.yaml 10000 10004

# Parallel batch processing (distributes frame chunks to multiple cores)
uv run python -m openptv2.batch.pyptv_batch_parallel test_data/test_cavity/parameters_Run1.yaml 10000 10004 4 --mode sequence

### Parallel Processing Config
OpenPTV2 supports multi-core parallel processing during image preprocessing and target detection (Approach C). This is highly effective for accelerating execution on multi-core systems.

* **GUI Configuration**: Open the **Main Parameters** dialog, navigate to the **Sequence** tab, check the **Parallel Pre-processing** box, and set the **Number of workers** (e.g., `4` or `0` for automatic core detection).
* **CLI/Environment Configuration**: Set environment variables before running any tracking or batch sequence:
  ```bash
  export OPENPTV_PARALLEL_PREPROCESS=True
  export OPENPTV_NUM_WORKERS=4  # Set worker processes (omitting uses all CPU cores)
  ```


### Command-line Shortcuts and Running Without `uv`

You can run these command-line tools without prefixing them with `uv run` using any of the following approaches:

#### 1. Activate the Virtual Environment (Standard Workflow)
By activating the project's virtual environment, the environment's `bin/` directory is added to your shell's `PATH`. This registers all entry points (like `pyptv_gui`, `pyptv`, `pyptv_batch`) directly in your terminal:
```bash
source .venv/bin/activate

# Now run directly without uv
pyptv_gui -w ./test_data/test_cavity
```

#### 2. Execute via Direct Path
If you do not wish to activate the virtual environment, you can run the built executable directly from the local `.venv` folder:
```bash
./.venv/bin/pyptv_gui -w ./test_data/test_cavity
```

#### 3. Define Shell Aliases (Global Access)
To run these shortcuts cleanly from any directory without manual paths, add alias entries to your shell profile (e.g., `~/.bashrc` or `~/.zshrc`):
```bash
# Add these lines to ~/.bashrc or ~/.zshrc
alias pyptv_gui='/home/user/Documents/GitHub/openptv2/.venv/bin/pyptv_gui'
alias pyptv_batch='/home/user/Documents/GitHub/openptv2/.venv/bin/pyptv_batch'
```
```bash
# Reload shell profile
source ~/.bashrc

# Now launch cleanly from any folder
pyptv_gui -w ./test_data/test_cavity
```

---

### Single Runtime Behavior

OpenPTV2 now ships a single runtime:

1. **One codebase:** `algorithms/*.py` is the only implementation path.
2. **Two execution modes:** the same modules run interpreted during development and compiled when built through Cython 3.
3. **No engine switching:** `OPENPTV_ENGINE`, `set_engine()`, and `optv` dispatching are no longer part of the runtime model.

---

### Short vs. Long Options

Both styles are fully supported across all command-line scripts. Choose based on your context:

* **Short Options (`-e`, `-w`, `-f`, `-l`)**: Best for manual terminal use, quick debugging, and live interactive typing.
* **Long Options (`--engine`, `--workdir`, `--first`, `--last`)**: Best for automation scripts, documentation, and config files to maximize readability.


---

## Documentation

- [Installation Guide](docs/developer_guide/building.md)
- [Quick Start](docs/tutorials/)
- [API Reference](docs/sphinx/)
- [Algorithm Documentation](docs/algorithms/)
- [Developer Guide](docs/developer_guide/)

---

## Repository Structure

```
openptv2/
├── algorithms/        # Python/Numba fallback engine
│   ├── calibration.py
│   ├── correspondences.py
│   ├── image_processing.py
│   ├── orientation.py
│   ├── parameters.py
│   ├── segmentation.py
│   ├── track.py
│   └── ...
├── bindings/          # Cython bindings source
│   ├── optv/          # Cython .pyx, .pxd files
│   ├── tests/         # Binding tests
│   └── pyproject.toml # scikit-build-core config
├── gui/               # TraitsUI GUI application
│   ├── pyptv/         # Main GUI package
│   ├── plugins/       # GUI plugins
│   └── tests/         # GUI tests
├── lib/               # C core library
│   ├── include/       # C headers
│   ├── src/           # C source files
│   ├── tests/         # C library tests
│   └── CMakeLists.txt
├── openptv2/          # Main Python package
│   ├── __init__.py
│   ├── calibration.py
│   ├── correspondence.py
│   ├── engine.py      # Engine selector
│   ├── tracker.py
│   └── ...
├── tests/             # Integration tests
│   ├── engine_comparison/
│   ├── fixtures/
│   └── integration/
├── docs/              # Documentation
│   ├── algorithms/
│   ├── developer_guide/
│   ├── sphinx/
│   └── tutorials/
├── scripts/           # Build helpers
├── CMakeLists.txt     # Root CMake build config
├── pyproject.toml     # Python project config
└── README.md
```

---

## Engine Comparison

| Feature | optv (C/Cython) | python (Numba) |
|---------|-----------------|----------------|
| Speed | Fastest | Fast (JIT compiled) |
| Debugging | Harder | Easy |
| Visualization | Limited | Full |
| Use case | Production | Development |

---

## Testing

```bash
# All tests
pytest

# C library tests
cd lib && mkdir build && cd build && cmake .. && ctest

# Engine comparison
pytest tests/engine_comparison/ --validate-engine

# GUI tests (headless)
pytest gui/tests/ --headless

# Integration tests
pytest tests/integration/ -v
```

---

## Migration from optv/pyptv

openptv2 maintains backward compatibility:

```python
# Old optv code (still works after installation)
from optv.tracking_framebuf import Target
from optv.tracker import Tracker

# New openptv2 code
from openptv2 import Target, Tracker

# Both work identically
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

### Development Workflow

```bash
# Set up development environment
uv sync --extra dev

# Make changes to source code

# Run tests
pytest tests/ -v

# Build documentation (optional)
cd docs && make html
```

---

## License

LGPL-3.0 or later. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

openptv2 combines work from:
- [openptv](https://github.com/openptv/openptv) - C library and bindings
- [pyptv](https://github.com/alexlib/pyptv) - Python GUI
- [openptv-python](https://github.com/openptv/openptv-python) - Python/Numba engine

---

## Contact

- Mailing list: openptv@googlegroups.com
- GitHub: https://github.com/openptv/openptv2
- Issues: https://github.com/openptv/openptv2/issues

---

## Helper Scripts

The project includes scripts for building and testing:

| Script | Purpose |
|--------|---------|
| `scripts/build_wheel.sh` | Build binary wheel from source |
| `scripts/install_wheel.sh` | Install wheel in clean test environment |
| `scripts/run_tests.sh` | Run test suite in test environment |
| `scripts/Dockerfile.slim` | Slim Docker image for testing |

See [BUILDING_BINARY_WHEELS.md](BUILDING_BINARY_WHEELS.md) for detailed usage.
