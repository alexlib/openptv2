# openptv2 Documentation

Welcome to the openptv2 documentation.

## Documentation Index

### Getting Started

- [Installation](../../README.md#installation) - Install openptv2
- [Quick Start](../../README.md#usage) - Basic usage examples
- [Building from Source](building.md) - Development installation

### User Documentation

- [User Guide](../tutorials/) - Tutorials and how-to guides
- [Tracking Visualization](../tutorials/tracking_visualization.md) - Preview tracking results
- [Tracking Debug Visualization](../tutorials/tracking_debug_visualization.md) - Debug tracking parameters interactively
- [API Reference](../sphinx/) - API documentation
- [Algorithm Documentation](../algorithms/) - Algorithm explanations

### Developer Documentation

- [Building from Source](building.md) - Detailed build instructions
- [Building Binary Wheels](../../BUILDING_BINARY_WHEELS.md) - Creating portable wheels
- [Development Workflow](building.md#development-workflow) - How to develop
- [Testing](building.md#testing) - Running tests
- [Code Quality](building.md#check-code-quality) - Linting and formatting

### Architecture

- [Repository Structure](../../README.md#repository-structure) - Project layout
- [Engine Architecture](../../README.md#engine-comparison) - Dual-engine design
- [Build System](building.md#build-architecture) - How the build works
- [Binary Wheels](../../BUILDING_BINARY_WHEELS.md) - manylinux2014 wheels

### Migration

- [From optv](../../README.md#migration-from-optvpyptv) - Migrating from optv
- [From pyptv](../../README.md#migration-from-optvpyptv) - Migrating from pyptv

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
- [Building Binary Wheels](../../BUILDING_BINARY_WHEELS.md) - manylinux/macOS/Windows wheels

### GUI
- [Testing the GUI](HOW_TO_TEST_GUI.md) - How to test the GUI

### Scripts
- [scripts/build_wheel.sh](../../scripts/build_wheel.sh) - Build binary wheel
- [scripts/install_wheel.sh](../../scripts/install_wheel.sh) - Install wheel in clean venv
- [scripts/run_tests.sh](../../scripts/run_tests.sh) - Run test suite

---

## For Users

1. **Installation**: See [README.md](../../README.md#installation)
2. **Basic Usage**: See [README.md](../../README.md#usage)
3. **GUI Usage**: Launch `openptv2-gui` and explore the interface

## For Developers

1. **Build Setup**: See [building.md](building.md)
2. **Development Workflow**: See [building.md#development-workflow](building.md#development-workflow)
3. **Testing**: See [building.md#testing](building.md#testing)

## For Contributors

1. **Fork the repository**
2. **Set up development environment**: `uv sync --extra dev`
3. **Make changes and run tests**: `pytest`
4. **Submit a pull request**

---

*Last updated: March 2026*
