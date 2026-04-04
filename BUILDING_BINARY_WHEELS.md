# Building Binary Wheels

**Last Updated**: 2026-04-04

This document describes how to build binary wheels for openptv2 for distribution across different platforms and Python versions.

---

## Why Binary Wheels?

Binary wheels contain pre-compiled Cython/C extensions, avoiding users needing to compile them locally. This provides:

- **Faster installation** - No compilation needed
- **Cross-platform compatibility** - Wheels work without build tools
- **Consistent builds** - Same binary everywhere

---

## manylinux2014 Standard

We use `manylinux2014` standard for Linux wheels, which:
- Compatible with glibc 2.17+ (CentOS 7, Ubuntu 14.04, Debian 8, etc.)
- Works on virtually all modern Linux distributions
- Audited and approved by Python Packaging Authority

---

## Prerequisites

- Docker installed and running
- Python 3.11+ with `uv` installed
- Internet connection for pulling Docker images

---

## Local Build Commands

### Build for Specific Python Version

```bash
# Python 3.11
CIBW_BUILD="cp311-*" CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014" \
  uvx cibuildwheel --platform linux --output-dir ./dist

# Python 3.12
CIBW_BUILD="cp312-*" CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014" \
  uvx cibuildwheel --platform linux --output-dir ./dist

# Python 3.13
CIBW_BUILD="cp313-*" CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014" \
  uvx cibuildwheel --platform linux --output-dir ./dist
```

### Build All Python Versions

```bash
# Build all (3.11-3.14)
CIBW_MANYLINUX_X86_64_IMAGE="manylinux2014" \
  uvx cibuildwheel --platform linux --output-dir ./dist
```

### Build for Current Platform Only

```bash
uvx cibuildwheel --output-dir ./dist
```

---

## Build Outputs

### Linux Wheels (manylinux2014)

| Python | Filename |
|--------|----------|
| 3.11 | `openptv2-1.0.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` |
| 3.12 | `openptv2-1.0.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` |
| 3.13 | `openptv2-1.0.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` |

### Expected Wheel Size

~6.3 MB per wheel (includes Cython extensions and C library)

---

## Testing Wheels Locally

### Create Test Environment

```bash
# Create venv with matching Python version
uv venv /tmp/test_venv --python 3.11

# Install wheel
uv pip install --python /tmp/test_venv/bin/python /path/to/wheel.whl

# Test import
/tmp/test_venv/bin/python -c "import openptv2; import optv; print('OK')"
```

### Full Test Suite

```bash
# Install test dependencies
uv pip install --python /tmp/test_venv/bin/python pytest pytest-cov numba

# Run tests
/tmp/test_venv/bin/python -m pytest bindings/tests/ algorithms/tests/ -v --tb=short
```

---

## CI/CD (GitHub Actions)

The project uses `.github/workflows/cibuildwheel.yml` for automated builds.

### Trigger Conditions

- Push to `main`/`master`
- Tags starting with `v*` or `[0-9]*`
- Pull requests to `main`/`master`
- Manual workflow dispatch

### Matrix

| OS | Python Versions |
|---|-----------------|
| ubuntu-latest | cp311, cp312, cp313, cp314 |
| windows-latest | cp311, cp312, cp313, cp314 |
| macos-latest | cp311, cp312, cp313, cp314 |

### Output

- **Linux**: manylinux2014 (x86_64, aarch64)
- **macOS**: x86_64, arm64
- **Windows**: AMD64

### Workflow Jobs

1. `build_wheels` - Build wheels for all platform/Python combinations
2. `build_sdist` - Build source distribution (tags only)
3. `test_package` - Verify wheel installation and tests (tags only)
4. `upload_pypi` - Publish to PyPI (tags only)

---

## Publishing to PyPI

Wheels are automatically published to PyPI when:

1. A tag is pushed (e.g., `v1.0.0`)
2. All builds succeed
3. All tests pass

### Manual Upload

```bash
# Install twine
pip install twine

# Upload to Test PyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ openptv2

# Upload to production PyPI
twine upload dist/*
```

---

## Troubleshooting

### Docker Not Running

```
Error response from daemon: No such image
```

**Solution**: Start Docker daemon (`dockerd` or Docker Desktop)

### Build Fails with Missing Headers

**Solution**: Ensure C sources are prepared:
```bash
python setup.py prepare
```

### Wrong Manylinux Tag

If wheel shows wrong platform tag, rebuild with:
```bash
CIBW_REPAIR_WHEEL_COMMAND="auditwheel repair -w {dest_dir} {wheel}"
```

### macOS Cross-Compilation

For universal wheels on macOS:
```bash
CIBW_ARCHS_MACOS="x86_64 arm64"
```

---

## Wheel Naming Convention

```
openptv2-{version}-cp{python}-cp{python}-{platform}.whl
```

Example: `openptv2-1.0.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`

- **cp311**: Python interpreter (CPython 3.11)
- **cp311**: ABI version
- **manylinux2014**: Manylinux standard
- **x86_64**: Architecture
- **manylinux_2_17**: glibc version compatibility

---

## Files

- `.github/workflows/cibuildwheel.yml` - CI workflow
- `pyproject.toml` - Build configuration
- `setup.py` - Build script with Cython extensions
- `scripts/build_wheel.sh` - Local build script
- `scripts/install_wheel.sh` - Local installation script
- `scripts/run_tests.sh` - Local test runner

---

## References

- [cibuildwheel documentation](https://cibuildwheel.pypa.io/)
- [manylinux specification](https://github.com/pypa/manylinux)
- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPA manylinux images](https://github.com/pypa/manylinux)
