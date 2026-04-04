# openptv2

**Unified OpenPTV**: Particle Tracking Velocimetry with dual-engine support

[![Python](https://img.shields.io/pypi/pyversions/openptv2.svg)](https://pypi.org/project/openptv2/)
[![License](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](LICENSE)

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

### For End Users (Pre-built Wheels)

**Prerequisites:**
- Python 3.11, 3.12, or 3.13
- pip or uv package manager

#### Option 1: Using pip

```bash
pip install openptv2
```

#### Option 2: Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install openptv2
uv add openptv2
```

#### Verify Installation

```bash
python -c "import openptv2; print(f'openptv2 version: {openptv2.__version__}')"
python -c "from openptv2 import Tracker; print('Tracker imported successfully')"
```

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

#### Step 3: Verify Build

```bash
# Test imports
python -c "import openptv2; print(f'openptv2 version: {openptv2.__version__}')"
python -c "from openptv2 import Tracker; print('Tracker imported successfully')"

# Run tests
pytest tests/ -v --tb=short
```

---

### Installation with GUI Dependencies

The GUI requires additional dependencies:

```bash
# Using uv
uv sync --all-extras

# Using pip
pip install -e ".[gui,dev]"
```

**GUI dependencies include:**
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

### Engine Selection

```python
import openptv2

# Use default (fastest) engine
openptv2.set_engine("optv")

# Use Python engine for debugging
openptv2.set_engine("python")

# Per-call engine selection
from openptv2 import track_particles
result = track_particles(images, engine="python")
```

### GUI

```bash
# Launch the GUI
openptv2-gui

# Or with specific engine
openptv2-gui --engine python
```

### Batch Processing

```bash
# Serial batch
openptv2-batch parameters.yaml 1 100

# Parallel batch
python -m openptv2.gui.pyptv_batch_parallel parameters.yaml 1 100 4
```

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
