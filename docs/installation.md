# Installation & Getting Started

This guide walks you through setting up OpenPTV2 on your local machine, inside virtual machines (VMs), or in headless servers.

---

## Supported Environments & Architecture

OpenPTV2 is compatible with **Python 3.11, 3.12, and 3.13** on the following platforms:
- **Linux** (x86_64, aarch64) — glibc 2.17+
- **macOS** (Intel x86_64, Apple Silicon arm64) — macOS 11+
- **Windows** (AMD64) — Windows 10+

### Dual-Engine Execution Modes
OpenPTV2 has a single Python codebase (`algorithms/`) that can run in two modes:
1. **Precompiled Mode (Default, Optimized)**: Compiled to native machine code via Cython 3 for maximum performance.
2. **Interpreted Mode (Developer/Fallback)**: Runs as pure interpreted Python with JIT compilation via Numba for debugging.

---

## Installation Methods

### Method A: Quick Install (Precompiled Binary Wheels)
Most users should install the precompiled binary wheels. This installs the fully compiled Cython 3 algorithms without requiring a local compiler.

```bash
# Recommended: Install using uv for lightning fast setups
uv pip install openptv2[gui]

# Or using standard pip
pip install "openptv2[gui]"
```

> [!NOTE]
> The `[gui]` extra installs the TraitsUI/Chaco (PySide6) desktop GUI, Matplotlib, and other scientific python libraries required for visualization. If you only need batch tracking on a remote server, omit the `[gui]` extra.

### Method B: For Developers (Building from Source)
If you want to modify OpenPTV2 or build it on an unsupported platform, you must build it from source.

#### System Prerequisites

=== "Linux (Ubuntu/Debian)"
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential python3-dev cmake git
    ```

=== "Linux (Fedora/RHEL)"
    ```bash
    sudo dnf install -y gcc gcc-c++ python3-devel cmake git
    ```

=== "macOS"
    ```bash
    # Install command line tools
    xcode-select --install
    # Install cmake via Homebrew (optional)
    brew install cmake
    ```

=== "Windows"
    1. Download and install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
    2. Select the **"Desktop development with C++"** workload during installation.
    3. Install [CMake](https://cmake.org/download/).

#### 1. Clone the Repository
```bash
git clone https://github.com/openptv/openptv2.git
cd openptv2
```

#### 2. Synchronize and Build
We recommend [uv](https://astral.sh/) for compiling and synchronizing the environment in a single command:

```bash
uv sync --extra dev
```

This automatically compiles the underlying C library, compiles the Cython bindings, and places the modules inside your virtual environment.

If you are using standard pip and venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install build-time requirements
pip install scikit-build-core cython "numpy>=2.0.0"

# Install in development (editable) mode
pip install -e ".[dev]"
```

---

## Installing on Virtual Machines (VMs) & Headless Hosts

Running scientific GUI software on Virtual Machines, Windows Subsystem for Linux (WSL), or headless cloud instances requires additional configuration for display forwarding and graphics rendering.

### 1. Windows Subsystem for Linux (WSL2)
WSL2 supports GUI applications out-of-the-box on Windows 11 and Windows 10 (Build 19044+).

1. Install dependencies inside WSL2:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 python3-tk
   ```
2. Set up `uv` and install `openptv2` inside your WSL virtual environment.
3. Launch the GUI:
   ```bash
   uv run openptv2-gui
   ```

### 2. Cloud VMs & Headless Servers (SSH/X11 Forwarding)
To run OpenPTV2 GUI on a remote server (e.g. AWS, DigitalOcean) and display it on your local desktop:

1. Connect to the VM via SSH with X11 forwarding enabled:
   ```bash
   ssh -X user@your-vm-ip
   ```
2. Install display-rendering dependencies on the remote VM:
   ```bash
   sudo apt-get install -y x11-apps mesa-utils libgl1-mesa-glx python3-tk
   ```
3. Test that X11 forwarding is working:
   ```bash
   xeyes
   ```
   *A pair of eyes should pop up on your local screen.*
4. Launch the OpenPTV2 GUI inside your activated virtual environment on the remote server.

### 3. Running Headless (No GUI)
If you are running automated particle tracking batch scripts on a server without a display server (no X11/Wayland), do **not** install the `[gui]` extras.

Use the pure CLI batch processing command `pyptv_batch`:
```bash
# Execute batch tracking on a headless VM without a display
uv run pyptv_batch --workdir=./test_data/test_cavity --first=10000 --last=10005
```

If you must run GUI-bound automated scripts or tests on headless servers, use `xvfb` (X Virtual Framebuffer):
```bash
# Install xvfb on Debian/Ubuntu VM
sudo apt-get install -y xvfb

# Run with virtual framebuffer
xvfb-run -a uv run python scripts/run_all_tests.py
```

### 4. VM Hypervisors (VirtualBox, VMware)
If you run OpenPTV2 inside a Linux guest OS on VirtualBox or VMware:
- **Enable 3D Acceleration**: In your VM settings, ensure **"Enable 3D Acceleration"** is checked.
- **Install Guest Additions**: Install guest additions inside the guest OS to provide optimal OpenGL graphic drivers (required for Matplotlib rendering).
- **Vbox Graphics Controller**: For VirtualBox, use the `VMSVGA` graphics controller.

---

## Verifying the Installation

After installing, verify that the package is correctly installed and utilizing the high-performance **precompiled Cython binary wheel** rather than the slow Python interpreter:

```bash
# 1. Inspect package runtime environment
uv run python -c "import openptv2; print(openptv2.get_runtime_info())"
```

Expected output:
```json
{"engine": "cython3-pure-python", "compiled": true, "package": "openptv2"}
```

If `compiled` is `true`, your installation is successfully using the optimized precompiled platform binaries.

### Running a Core Module Test
Ensure the unified APIs are available:
```bash
uv run python -c "from openptv2 import Tracker; print('OpenPTV2 Unified Tracker API: OK')"
```

---

## Troubleshooting: Windows Installation Issues

If `openptv2.get_runtime_info()` reports `"compiled": false`, or the GUI raises
`OverflowError`/`RuntimeWarning: overflow encountered...` from
`track_kernels_batch.py` during detection, you're running the pure-Python
interpreted fallback instead of the compiled extensions. This is functionally
correct but much slower, and (on very old NumPy or, before this project's fix,
NumPy 2.x with dense/bright images) can hit integer-overflow edge cases.

### Quick reference: remove, recreate, and verify the environment

When in doubt, start clean. These three steps resolve most Windows install
issues; the numbered sections below explain *why* each one matters.

**1. Remove the virtual environment** (and any locally-built extensions):
```powershell
Remove-Item -Recurse -Force .venv
Remove-Item -Force src\openptv2\algorithms\*.so, src\openptv2\algorithms\*.c -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
```
(Omit the second and third lines if you're not working inside a cloned
openptv2 checkout — a plain `pip install`-only setup has no local `src/` or
`build/` to clean up.)

**2. Create a new one**, pinned to a Python version openptv2 has wheels for
(3.11, 3.12, or 3.13 — see item 2 below for why this matters):
```powershell
uv venv --python 3.13
uv pip install --python .venv\Scripts\python.exe "openptv2[gui]"
```

**3. Test whether you got the compiled extensions or the interpreted fallback**:
```powershell
.venv\Scripts\python.exe -c "import openptv2; print(openptv2.get_runtime_info())"
```
```json
{"engine": "cython3-pure-python", "compiled": true, "package": "openptv2"}
```
`"compiled": true` means you're running the fast, pre-compiled `.pyd`
extensions. `"compiled": false` means something below is still going wrong —
work through the numbered sections in order.

### 1. Don't share a repo directory between WSL and native Windows

If you `git clone`d (or otherwise use) the same folder from **both** WSL and a
native Windows shell — e.g. editing/running it via `/mnt/c/Users/you/openptv2`
in WSL and `C:\Users\you\openptv2` in PowerShell, which are the same
filesystem path — a `.venv` or compiled `.so`/`.pyd` files built from one side
are incompatible with the other and will confuse `uv`/`pip` on the other side
(look for `pyvenv.cfg` inside `.venv` mentioning a Linux path, or a `Scripts/`
folder that's actually missing because a Linux `bin/`-layout venv is sitting
there instead). Fix: pick one platform per checkout, or delete and rebuild the
environment before switching:
```powershell
Remove-Item -Recurse -Force .venv
Remove-Item -Force src\openptv2\algorithms\*.so, src\openptv2\algorithms\*.c -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
```

### 2. Pin the Python version — wheels only cover 3.11–3.13

`uv` defaults to the newest Python it can find, which may be newer than what
openptv2 publishes wheels for. If `uv venv` picks e.g. 3.14, `pip install
openptv2` has no wheel to use and silently falls back to building from source
(needing a local compiler) or to the interpreted `.py` modules. Pin explicitly:
```powershell
uv venv --python 3.13
```

### 3. `uv venv` doesn't include `pip` — and `uv run`/`uv sync` install the *local* project, not the PyPI wheel

A venv created by `uv venv` has no `pip` module (`python -m pip install ...`
fails with `No module named pip`) — use `uv pip install --python <path-to-venv-python>
<package>` instead. Also, running `uv run`/`uv sync` **inside a cloned
openptv2 checkout** installs your local source tree (equivalent to `pip
install -e .`), which is never pre-compiled automatically — Cython
compilation is always the separate, manual `uv run python setup.py
build_ext --inplace` step (requires Visual Studio Build Tools, "Desktop
development with C++" workload). To test the actual **published PyPI wheel**
in isolation, install into a venv *outside* any openptv2 checkout:
```powershell
cd C:\
uv venv --python 3.13 wheel-test-venv
uv pip install --python wheel-test-venv\Scripts\python.exe "openptv2[gui]"
wheel-test-venv\Scripts\python.exe -c "import openptv2; print(openptv2.get_runtime_info())"
```
This should print `{"engine": "cython3-pure-python", "compiled": true, ...}`
immediately, with no build step — the wheel ships pre-compiled `.pyd` files.

### 4. PowerShell blocks `Scripts\activate.ps1`

`<venv>\Scripts\activate` fails with `running scripts is disabled on this
system` under PowerShell's default execution policy. Either call the venv's
`python.exe` by full path instead of activating (as in the commands above),
or allow local scripts for your user once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
