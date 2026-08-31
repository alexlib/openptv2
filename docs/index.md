# openptv2 Documentation

Welcome to the openptv2 documentation.

## Documentation Index

### Getting Started

- [Installation & Platform Setup](installation.md) - Installing on Linux, macOS, Windows, VMs, and WSL
- [First Steps](first_steps.md) - Basic programmatic usage, GUI navigation, and command-line batching
- [End-to-End Tutorial](tutorials/getting_started_tutorial.md) - Step-by-step 3D particle tracking tutorial

### User Documentation

- [Tracking Pipeline & Results Guide](tracking_guide.md) - Pipeline workflow, parameter guide, multi-pass tracking, and ptv_is.# output format
- [Lid-Driven Cavity Flow Tutorial](tutorials/cavity_flow_tutorial.md) - End-to-end 3D-PTV case study: Autocalibration, Tracer Shaking, Warmup, and 3D Trajectories
- [Aortic Pulsatile Flow Tutorial](aorta_tutorial.md) - Cloud-native 3D-PTV on complex aortic flow
- [Auto-Calibration with `openptv warmup`](tutorials/warmup_tutorial.md) - Standalone parameter/engine auto-tuning before tracking
- [Tracer Self-Calibration ("Shaking")](tracer-self-calibration.md) - Refining an existing calibration on real tracked tracer particles, GUI walkthrough
- [Command-Line Batch Processing](tutorials/batch_processing.md) - Running `openptv2-batch` headlessly
- [Sequence & Tracking Plugins](tutorials/plugins.md) - Built-in plugins, selecting one, writing your own
- [Tracking Visualization](tutorials/tracking_visualization.md) - Preview tracking results
- [Tracking Debug Visualization](tutorials/tracking_debug_visualization.md) - Debug tracking parameters interactively
- [Running Tests](tests.md) - Command-line test suite, excluding slow tests, running native C and virtual wheel pipelines
- [Algorithm Documentation](algorithms/tracking.md) - Algorithm explanations
- [Burgers Case Study](algorithms/burgers_gap_relinking_case_study.md) - Detailed analysis of tracking deviation and recovery

### Developer Documentation

- [Building from Source](developer_guide/building.md) - Detailed build instructions
- [Cython & Pure Python Modes](developer_guide/cython_and_pure_python.md) - Switch between high-performance Cython and debuggable Pure Python
- [Documentation Workflow](developer_guide/documentation_workflow.md) - Editing documentation and deploying to GitHub Pages
- [Packaging & Releases](developer_guide/packaging_and_releases.md) - Building binary wheels and releasing them on PyPI and GitHub
- [GUI Testing Guide](HOW_TO_TEST_GUI.md) - How to test the GUI
- [Development Workflow](developer_guide/building.md#4-development-workflow-commands) - How to develop

---

## Quick Links

- [GitHub Repository](https://github.com/openptv/openptv2)
- [Issue Tracker](https://github.com/openptv/openptv2/issues)
- [Mailing List](https://groups.google.com/g/openptv)

---

## Documentation Structure

```
docs/
├── index.md                 # This file
├── HOW_TO_TEST_GUI.md       # GUI testing guide
├── BUILDING_BINARY_WHEELS.md # Wheel building guide
├── developer_guide/
│   └── building.md          # Build instructions
├── algorithms/
│   └── tracking.md          # Tracking algorithm documentation
├── sphinx/                  # API reference (TODO)
└── tutorials/               # User tutorials (TODO)
```

---

## Available Documentation

### Algorithms
- [Tracking Algorithms](algorithms/tracking.md) - Explanation of track.c vs track3d.c

### Developer Guide
- [Building from Source](developer_guide/building.md) - Full build instructions
- [Cython & Pure Python Modes](developer_guide/cython_and_pure_python.md) - Seamlessly switch between C-level speed and pure python debugging
- [Building Binary Wheels](https://github.com/alexlib/openptv2/blob/main/BUILDING_BINARY_WHEELS.md) - manylinux/macOS/Windows wheels

### GUI
- [Testing the GUI](HOW_TO_TEST_GUI.md) - How to test the GUI

### Scripts
- [scripts/build_wheel.sh](https://github.com/alexlib/openptv2/blob/main/scripts/build_wheel.sh) - Build binary wheel
- [scripts/install_wheel.sh](https://github.com/alexlib/openptv2/blob/main/scripts/install_wheel.sh) - Install wheel in clean venv
- [scripts/run_tests.sh](https://github.com/alexlib/openptv2/blob/main/scripts/run_tests.sh) - Run test suite

---

## For Users

1. **Installation**: See [README.md](https://github.com/alexlib/openptv2/blob/main/README.md#installation)
2. **Basic Usage**: See [README.md](https://github.com/alexlib/openptv2/blob/main/README.md#usage)
3. **GUI Usage**: Launch `openptv2-gui` and explore the interface

## For Developers

1. **Build Setup**: See [building.md](developer_guide/building.md)
2. **Cython & Pure Python modes**: See [cython_and_pure_python.md](developer_guide/cython_and_pure_python.md)
3. **Development Workflow**: See [building.md#development-workflow](developer_guide/building.md#4-development-workflow-commands)
4. **Testing**: See [building.md#running-tests](developer_guide/building.md#running-tests)

## For Contributors

1. **Fork the repository**
2. **Set up development environment**: `uv sync --extra dev`
3. **Make changes and run tests**: `pytest`
4. **Submit a pull request**

---

*Last updated: March 2026*
