# Setup Plan: Binary Wheel Build & Clean Environment Test

**Created**: 2026-04-04  
**Status**: Ready for execution

---

## Goal

Build a binary wheel from source, install it in a completely clean environment using `uv`, and run the full test suite to verify everything works correctly.

---

## Tooling

- **Package manager**: `uv` exclusively (v0.10.7 at `/home/user/.local/bin/uv`)
- **System Python**: 3.12.3 (lacks `pip`/`venv`)
- **Dev environment**: `.venv/` with Python 3.11.13
- **Existing wheel**: `dist/openptv2-1.0.0-cp311-cp311-linux_x86_64.whl` (~6.3 MB)

---

## Steps

### Step 1: Build the wheel (if needed)

```bash
uv run --with build --with setuptools --with cython --with numpy python -m build --wheel
```

Or use the existing wheel in `dist/`.

### Step 2: Create clean environment with `uv`

```bash
uv venv /tmp/openptv2_test_venv --python 3.11
```

### Step 3: Install wheel in clean environment

```bash
uv pip install /tmp/openptv2_test_venv dist/openptv2-1.0.0-cp311-cp311-linux_x86_64.whl[gui]
```

### Step 4: Install test dependencies

```bash
uv pip install /tmp/openptv2_test_venv pytest pytest-cov numba
```

### Step 5: Run import verification

```bash
uv run --python /tmp/openptv2_test_venv/bin/python -c "import openptv2; print('OK')"
```

### Step 6: Run full test suite

```bash
uv run --python /tmp/openptv2_test_venv/bin/python -m pytest bindings/tests/ algorithms/tests/ -v --tb=short
```

### Step 7: Cleanup

```bash
rm -rf /tmp/openptv2_test_venv
```

---

## Success Criteria

1. Wheel builds successfully from source
2. Clean environment creation succeeds with `uv venv`
3. Wheel installs with GUI extras via `uv pip install`
4. All 9 import verification tests pass
5. Full test suite (`bindings/tests/` + `algorithms/tests/`) passes
6. Both `import openptv2` and `import optv` work after installation

---

## Known Issues to Investigate

1. **`test_tracker.py::test_forward`** and **`test_forward_3d`**: Pre-existing failures in binding tests
2. **Previous test suite run**: Failed after ~323s with raw C tracking loop prints in output — exact error not captured due to pipeline output truncation

---

## Relevant Files

- `scripts/wheel_test_pipeline.py` — Existing pipeline (uses stdlib `venv`/`pip`, will bypass with direct `uv` commands)
- `setup.py` — Custom build script
- `pyproject.toml` — Project metadata and dependencies
- `dist/openptv2-1.0.0-cp311-cp311-linux_x86_64.whl` — Pre-built wheel
