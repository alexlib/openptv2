# Plan: cibuildwheel Multi-Platform Binary Wheel Distribution

**Created**: 2026-04-04  
**Status**: Planning

---

## Current Issues

1. **Linux wheel not portable**: Built against host system glibc, fails on other Linux distros
2. **Python version support**: Supports cp311-cp313
3. **Platform coverage**: Missing proper manylinux, macOS universal, Windows wheels
4. **pyproject.toml**: `requires-python = ">=3.11,<3.14"` excludes 3.14

---

## Goals

1. Generate portable binary wheels using manylinux2014 standard
2. Support Python 3.11, 3.12, 3.13
3. Support Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (AMD64)
4. Build and test on GitHub Actions via cibuildwheel

---

## Implementation Plan

### Step 1: Update pyproject.toml

- Change `requires-python = ">=3.11,<3.14"` to `">=3.11,<3.15"`
- Add Python 3.14 to classifiers
- Update CIBW_BUILD matrix in workflow

### Step 2: Update cibuildwheel.yml

**Linux (manylinux2014):**
- Use `manylinux2014` image instead of skipping musllinux
- Build for x86_64 and aarch64
- Remove `CIBW_SKIP: "*musllinux*"`

**macOS:**
- Already set up for x86_64 and arm64
- Consider building universal wheels or separate

**Windows:**
- Already set for AMD64

### Step 3: Add Python 3.14 to Build Matrix

- Update `python` matrix to include `cp314`
- Update pyproject.toml classifiers

---

## cibuildwheel Configuration Details

```yaml
CIBW_BUILD: "cp311-* cp312-* cp313-* cp314-*"
CIBW_SKIP: ""  # Remove musllinux skip
CIBW_MANYLINUX_X86_64_IMAGE: "manylinux2014"
CIBW_MANYLINUX_AARCH64_IMAGE: "manylinux2014"
CIBW_ARCHS_LINUX: "x86_64 aarch64"
CIBW_ARCHS_MACOS: "x86_64 arm64"
CIBW_ARCHS_WINDOWS: "AMD64"
```

---

## Expected Outputs

| Platform | Python | Wheel |
|----------|--------|-------|
| Linux x86_64 | 3.11-3.14 | openptv2-*-cp311-*-manylinux2014_x86_64.whl |
| Linux aarch64 | 3.11-3.14 | openptv2-*-cp311-*-manylinux2014_aarch64.whl |
| macOS x86_64 | 3.11-3.14 | openptv2-*-cp311-*_macosx_x86_64.whl |
| macOS arm64 | 3.11-3.14 | openptv2-*-cp311-*_macosx_arm64.whl |
| Windows AMD64 | 3.11-3.14 | openptv2-*-cp311-*_win_amd64.whl |

---

## Files to Modify

1. `.github/workflows/cibuildwheel.yml` - Main CI workflow
2. `pyproject.toml` - Python version requirements

---

## Testing Strategy

- Test import on all platforms
- Run subset of tests in CI
- Full test suite on tag/push to main
