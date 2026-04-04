# Build Wheels Plan: Multi-Platform Binary Distribution

**Created**: 2026-04-04  
**Status**: Ready for Implementation

---

## Executive Summary

Create portable binary wheels for openptv2 using cibuildwheel to support:
- **Linux**: manylinux2014 (x86_64, aarch64)
- **macOS**: x86_64, arm64 (universal optional)
- **Windows**: AMD64
- **Python**: 3.11, 3.12, 3.13, 3.14

---

## Problem Statement

### Current Issues

1. **Linux wheels not portable**: Built against host system glibc, fails on other Linux distros with different glibc versions
2. **Python version gaps**: Supports cp311-cp313 only, missing cp314
3. **Limited architecture**: Only x86_64 on Linux, no aarch64 support
4. **pyproject.toml restriction**: `requires-python = ">=3.11,<3.14"` excludes Python 3.14

### Why This Matters

- Users cannot install the wheel on Amazon Linux, AlmaLinux, Debian oldstable, etc.
- Cannot deploy on ARM64 Linux servers (AWS Graviton, Apple Silicon via Linux VMs)
- Python 3.14 will be released soon and needs support
- Missing the large ARM64 macOS user base

---

## Solution Architecture

### Tool Selection: cibuildwheel

**Rationale**:
- Industry standard for building Python wheels
- Native manylinux2014 support for portable Linux wheels
- Cross-compilation support for aarch64
- Integration with GitHub Actions
- Automatic testing after build

### Manylinux2014 Standard

- Based on CentOS 7 glibc (2.17)
- Compatible with virtually all modern Linux distributions
- Pre-built images available from pypa
- Audited and approved by Python Packaging Authority

---

## Implementation Steps

### Step 1: Update pyproject.toml

**File**: `pyproject.toml`

| Change | From | To |
|--------|------|-----|
| requires-python | `>=3.11,<3.14` | `>=3.11,<3.15` |
| Classifiers | Add | `"Programming Language :: Python :: 3.14"` |

```toml
requires-python = ">=3.11,<3.15"
# In classifiers array, add:
"Programming Language :: Python :: 3.14",
```

### Step 2: Update cibuildwheel Workflow

**File**: `.github/workflows/cibuildwheel.yml`

| Configuration | Current | New |
|--------------|---------|-----|
| Python matrix | cp311, cp312, cp313 | cp311, cp312, cp313, cp314 |
| CIBW_SKIP | `"*musllinux*"` | (remove) |
| CIBW_MANYLINUX_X86_64_IMAGE | (not set) | `manylinux2014` |
| CIBW_MANYLINUX_AARCH64_IMAGE | (not set) | `manylinux2014` |
| CIBW_ARCHS_LINUX | x86_64 | x86_64 aarch64 |

### Step 3: Verify Build Configuration

**Prerequisites**:
- `setup.py` or `pyproject.toml` with build system configured
- Cython extensions in `bindings/optv/*.pyx`
- Proper package metadata in pyproject.toml

---

## Expected Outputs

### Wheel Naming Convention

```
openptv2-{version}-cp{python}-cp{python}-{platform}.whl
```

### Generated Wheels

| Platform | Architecture | Python | Example Filename |
|----------|--------------|--------|------------------|
| Linux | x86_64 | 3.11 | openptv2-1.0.0-cp311-cp311-manylinux2014_x86_64.whl |
| Linux | x86_64 | 3.12 | openptv2-1.0.0-cp312-cp312-manylinux2014_x86_64.whl |
| Linux | x86_64 | 3.13 | openptv2-1.0.0-cp313-cp313-manylinux2014_x86_64.whl |
| Linux | x86_64 | 3.14 | openptv2-1.0.0-cp314-cp314-manylinux2014_x86_64.whl |
| Linux | aarch64 | 3.11 | openptv2-1.0.0-cp311-cp311-manylinux2014_aarch64.whl |
| Linux | aarch64 | 3.12 | openptv2-1.0.0-cp312-cp312-manylinux2014_aarch64.whl |
| Linux | aarch64 | 3.13 | openptv2-1.0.0-cp313-cp313-manylinux2014_aarch64.whl |
| Linux | aarch64 | 3.14 | openptv2-1.0.0-cp314-cp314-manylinux2014_aarch64.whl |
| macOS | x86_64 | 3.11-3.14 | openptv2-1.0.0-cp311-cp311-macosx_x86_64.whl |
| macOS | arm64 | 3.11-3.14 | openptv2-1.0.0-cp311-cp311-macosx_arm64.whl |
| Windows | AMD64 | 3.11-3.14 | openptv2-1.0.0-cp311-cp311-win_amd64.whl |

### Total Wheels Per Release: 32

- Linux x86_64: 4 (Python 3.11-3.14)
- Linux aarch64: 4 (Python 3.11-3.14)
- macOS x86_64: 4 (Python 3.11-3.14)
- macOS arm64: 4 (Python 3.11-3.14)
- Windows AMD64: 4 (Python 3.11-3.14)

---

## CI/CD Pipeline Design

### GitHub Actions Workflow

```yaml
# Trigger: Push to main/master, tags v*, PR to main/master
on:
  push:
    branches: [main, master]
    tags: ['v*']
  pull_request:
    branches: [main, master]
```

### Jobs

1. **build_wheels**: Matrix across OS × Python
2. **build_sdist**: Source distribution (tags only)
3. **test_package**: Verify wheel installation
4. **upload_pypi**: Publish to PyPI (tags only)

### Testing Strategy

**CIBW_TEST_COMMAND** (runs after each wheel build):
```bash
python -c "import openptv2; print('openptv2 OK')"
python -c "from optv.tracking_framebuf import Target; print('optv OK')"
python -c "from algorithms.tracking_frame_buf import Target; print('algorithms OK')"
```

**Full test suite** (test_package job):
```bash
pytest bindings/tests/ algorithms/tests/ -v --tb=short
```

---

## Verification Checklist

- [ ] pyproject.toml updated for Python 3.14
- [ ] cibuildwheel.yml configured for manylinux2014
- [ ] aarch64 added to Linux architectures
- [ ] Python 3.14 added to build matrix
- [ ] Test build completes without errors
- [ ] All import tests pass
- [ ] Wheels can be installed on different Linux distro
- [ ] PyPI upload works correctly

---

## Rollback Plan

If issues occur:

1. **Revert commit**: `git revert <commit>`
2. **Temporary fix**: Set `CIBW_SKIP: "*manylinux*"` to use system Python
3. **Disable aarch64**: Set `CIBW_ARCHS_LINUX: x86_64`
4. **Remove 3.14**: Remove from Python matrix

---

## References

- [cibuildwheel documentation](https://cibuildwheel.readthedocs.io/)
- [manylinux2014 specification](https://github.com/pypa/manylinux)
- [Python packaging user guide](https://packaging.python.org/)
- [PyPA manylinux images](https://github.com/pypa/manylinux)
