# openptv2

**Particle Tracking Velocimetry**, single Cython 3 pure-Python engine

[![Python](https://img.shields.io/pypi/pyversions/openptv2.svg)](https://pypi.org/project/openptv2/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://alexlib.github.io/openptv2/)
[![Coverage](https://img.shields.io/badge/coverage-34%25-red)](scripts/update_coverage_badge.sh)
[![Code health](https://api.repowise.dev/badge/health/alexlib/openptv2.svg)](https://repowise.dev/repo/alexlib/openptv2)

## Overview

openptv2 is a single-engine PTV library: the same
`src/openptv2/algorithms/` modules run interpreted in development and
compiled through Cython 3 when built, with no separate C core, no
Cython bindings to an external library, and no runtime engine
selector.

- **Algorithms** (`src/openptv2/algorithms/`) - Cython 3 pure-Python particle tracking, correspondence, and calibration code
- **Plugins** (`src/openptv2/plugins/`) - pluggable tracker implementations (fast, MyPTV/ProPTV-informed trackers)
- **GUI** (`src/openptv2/gui/`) - TraitsUI/Chaco desktop application
- **Batch pipeline** (`openptv2-batch`) - headless sequence/tracking runner for scripting and cloud use

## What's Inside (and Where the Tracking Concepts Come From)

openptv2 is a self-contained PTV engine: detection, camera calibration,
correspondence/stereo-matching, and the classic forward tracking loop are all
implemented from scratch here in `src/openptv2/algorithms/`.

On top of that native engine, openptv2 ships **plugins** that bring in tracking
concepts from two external open-source projects:

- **[MyPTV](https://github.com/ronshnapp/MyPTV)** — MIT licensed
  (© 2022 Ron Shnapp). openptv2's `myptv_2d_tracking` / `nearest_hungarian_3d`
  plugins adapt MyPTV's algorithm *ideas* (2D per-camera image-space tracking
  with multi-camera consensus, and 3D kinematic prediction + linear-assignment
  matching) onto openptv2's own data structures and assignment machinery.

- **[proPTV](https://github.com/RobinBarta/proPTV)** — MIT licensed
  (© 2023 DLR, Robin Barta). openptv2 **vendors** the small pure-numpy core of
  proPTV (`src/openptv2/plugins/proptv/`): the Gaussian-Mixture-Model / basis
  approximation and Savitzky-Golay smoothing routines used in proPTV's track
  prediction. The `predictive_gmm_3d` plugin wires those routines into openptv2's
  tracker.

> **Important:** openptv2 incorporates and adapts *parts* of these projects as
> plugins — not their full frameworks. The complete tracking capabilities of
> each project — e.g. proPTV's 2D-image triangulation pipeline, Soloff
> calibration, backtracking/repair and Eulerian field estimation, or MyPTV's
> full photogrammetry, trajectory smoothing and anisotropic-particle tools —
> live in and are available only from their own repositories:
>
> - MyPTV: <https://github.com/ronshnapp/MyPTV>
> - proPTV: <https://github.com/RobinBarta/proPTV>
>
> If you need those advanced features, please use those projects directly.
> Both are permissively MIT-licensed (see [License](#license)).

## Key Features

- **Single runtime**: one codebase, no C/Cython vs. Python fallback split to keep in sync
- **Pluggable trackers**: swap tracking algorithms via the plugin architecture
- **Easy installation**: `pip install openptv2` for headless/batch use, `openptv2[gui]` to add the desktop GUI

---

## Installation

There are three install profiles:

| Command | For | Includes |
| --- | --- | --- |
| `pip install openptv2` | scripting + `openptv2-batch` (default) | headless algorithms/API + sequence/tracking pipeline |
| `pip install openptv2[gui]` | desktop users | default **plus** the TraitsUI/Chaco/PySide6 desktop GUI |
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
docker build -t openptv2 -f docker/Dockerfile .

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

**Headless cloud batch:** `docker/Dockerfile.cloud` is a slim, no-GUI, free-threaded
3.14t image for servers/Cloud Run. See [docs/cloud-batch.md](docs/cloud-batch.md)
for the one-command install, `openptv2-batch` usage, and measured timings.

**Zarr + HDF5 Cloud Storage:** OpenPTV2 includes a native, high-performance Zarr storage engine (`res/run.zarr`) replacing thousands of per-frame text files with a cloud-native chunked format. See [docs/zarr-hdf5-storage.md](docs/zarr-hdf5-storage.md) for usage, terminal inspection, and Flowtracks HDF5 export.

---

### For Developers (Build from Source)

**Prerequisites:**
- Python 3.11–3.14 (free-threaded 3.14t works too, see `docker/Dockerfile.cloud`)
- C compiler (gcc on Linux, clang on macOS, MSVC Build Tools on Windows) — needed to build the Cython 3 extensions, no CMake involved
- uv (recommended) or pip

#### System Dependencies

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
xcode-select --install
```

**Windows:**
- Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### Step 1: Clone the Repository

```bash
git clone https://github.com/alexlib/openptv2.git
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
pip install setuptools cython numpy

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
- traitsui, enable, chaco (Enthought framework + visualization)
- PySide6 (Qt bindings)
- matplotlib, pandas, flowtracks (analysis)

(`scikit-image` and `numpy` are core dependencies, installed either way.)

---

### Installing from Binary Wheels

Pre-built wheels are published to PyPI:

```bash
# Using pip
pip install openptv2

# Using uv
uv pip install openptv2
```

---

### Troubleshooting Installation

#### Common Issues

**1. "C compiler not found"**
```bash
# Linux: sudo apt-get install build-essential
# macOS: xcode-select --install
# Windows: Install MSVC Build Tools
```

**2. "Cython not found"**
```bash
pip install cython>=3.0.0
```

**3. "NumPy version mismatch"**
```bash
pip install numpy>=2.0.0
```

**4. Cython extensions not rebuilt after editing `src/openptv2/algorithms/`**
```bash
uv run python setup.py build_ext --inplace
```

---

## Usage

### Basic Tracking

The quickest way to run a full detection → correspondence → tracking
pipeline is the batch CLI (see [Batch Processing](#batch-processing)
below) or the GUI. For scripting against the library directly —
loading calibrations/parameters and driving `openptv2.Tracker` — see
[docs/tutorials/](docs/tutorials/).

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

### OpenPTV³ Auto-Research Dashboard

Live, zero-install demo of the differentiable PTV pipeline
(`docs/plans/differentiable_ptv_nextgen_plan.md`): drag the Stage-1 intensity
threshold and watch the gradient flow through to the Lagrangian physics loss,
live acceleration PDF, and velocity power spectrum.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alexlib/openptv2/blob/main/notebooks/marimo_autoresearch_dashboard.py)

---

## Repository Structure

```
openptv2/
├── src/openptv2/
│   ├── algorithms/    # Cython 3 pure-Python engine: calibration,
│   │                  # correspondences, orientation, tracking, etc.
│   │                  # (the only algorithm implementation path)
│   ├── plugins/       # Pluggable tracker/sequence implementations
│   │                  # (fast, MyPTV/ProPTV-inspired trackers, rembg, ...)
│   ├── batch/         # openptv2-batch / pyptv_batch headless pipeline
│   ├── storage/       # Zarr frame store (res/run.zarr)
│   ├── gui/           # TraitsUI/Chaco desktop application
│   ├── tracker.py, calibration.py, correspondences.py, ...  # public API
│   └── __init__.py
├── tests/             # Test suite (unit, parity, perf, integration, gui)
├── docs/              # Documentation, tutorials, developer guide
├── scripts/           # Build/analysis/benchmark helper scripts
├── docker/            # Dockerfiles for GUI and cloud batch images
├── test_data/         # Calibration files, parameter files, fixtures
├── pyproject.toml     # Python project config (build, deps, entry points)
└── README.md
```

---

## Testing

```bash
# All tests
uv run pytest

# By marker
uv run pytest -m unit
uv run pytest -m "not slow"

# GUI tests (headless)
uv run pytest tests/gui/ -v

# Hot-path smoke test (tracking + correspondences)
uv run pytest tests/unit/test_track.py tests/unit/test_track3d.py tests/unit/test_correspondences.py -v --tb=short
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

MIT. See [LICENSE](LICENSE) for details.

As of 0.3.2, openptv2 is relicensed from LGPL-3.0 to MIT. The C/Cython
core that carried the LGPL license now lives in the separate
[openptv](https://github.com/openptv/openptv) repo; this codebase is a
from-scratch pure-Python/Cython PTV engine plus a pluggable tracker
architecture.

### Third-party tracking code

openptv2 includes and adapts tracking code from two MIT-licensed projects, in
accordance with their licenses:

- **MyPTV** (MIT, © 2022 Ron Shnapp) — algorithm concepts adapted into the
  `myptv_2d_tracking` / `nearest_hungarian_3d` plugins. Their MIT notice is
  incorporated; see <https://github.com/ronshnapp/MyPTV>.
- **proPTV** (MIT, © 2023 DLR / Robin Barta) — the GMM / Savitzky-Golay
  routines in `src/openptv2/plugins/proptv/` are **vendored** from proPTV.
  proPTV's license requires that its copyright/permission notice be included
  in copies and that its underlying publication be cited. Accordingly:

  > MIT License, Copyright (c) 2023 DLR (Project owner: Robin Barta).
  > GMM / Savitzky-Golay routines adapted from
  > **Barta, Robin, et al. "proPTV – A probabilistic particle tracking
  > velocimetry framework." *Journal of Computational Physics* (2024),
  > <https://doi.org/10.1016/j.jcp.2024.113212>.**

  The vendored modules at `src/openptv2/plugins/proptv/` carry the MIT
  copyright/permission notice in their headers.

Using either project's advanced, project-specific features is out of scope
here — refer to and use those projects directly.

---

## Acknowledgments

openptv2 combines work from:
- [openptv](https://github.com/openptv/openptv) - C library and bindings
- [pyptv](https://github.com/alexlib/pyptv) - Python GUI
- [openptv-python](https://github.com/openptv/openptv-python) - Python/Numba engine
- [MyPTV](https://github.com/ronshnapp/MyPTV) - 2D/3D tracking algorithm concepts adapted into the MyPTV plugins (MIT)
- [proPTV](https://github.com/RobinBarta/proPTV) - GMM / Savitzky-Golay track-prediction routines vendored into `src/openptv2/plugins/proptv/` (MIT, © 2023 DLR / Robin Barta)

See [License](#license) for the licensing and citation details.

---

## Contact

- Mailing list: openptv@googlegroups.com
- GitHub: https://github.com/alexlib/openptv2
- Issues: https://github.com/alexlib/openptv2/issues

---

## Helper Scripts

The project includes scripts for building and testing:

| Script | Purpose |
|--------|---------|
| `scripts/build_wheel.sh` | Build binary wheel from source |
| `scripts/install_wheel.sh` | Install wheel in clean test environment |
| `scripts/run_tests.sh` | Run test suite in test environment |
| `docker/Dockerfile.slim` | Slim Docker image for testing |
