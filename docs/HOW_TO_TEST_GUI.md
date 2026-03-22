# How to Test the GUI

This document explains how to test that the openptv2 GUI works correctly.

## Quick Test

```bash
# Launch the GUI
openptv2-gui

# Or alternatively
python -m gui.pyptv.pyptv_gui
```

## Automated Tests

### Run the test suite

```bash
cd /home/user/Documents/GitHub/openptv2
source .venv311/bin/activate

# Run GUI-related tests
pytest gui/tests/test_optv.py -v
pytest gui/tests/test_environment.py -v
pytest gui/tests/test_installation.py -v
pytest gui/tests/test_core_functionality.py -v
pytest gui/tests/test_tracker_minimal.py -v
```

### Run the GUI functionality test script

```bash
python test_gui_functionality.py
```

This script tests:
- optv module imports
- Target detection (segmentation)
- Calibration object manipulation
- Epipolar geometry module
- Coordinate transforms
- GUI class instantiation
- Tracker initialization

## What Has Been Tested ✅

### Phase 1 GUI Verification (Complete)

| Test | Status | Details |
|------|--------|---------|
| GUI imports | ✅ PASS | All modules import correctly |
| optv integration | ✅ PASS | All optv modules work |
| GUI classes | ✅ PASS | Can be instantiated |
| GUI entry point | ✅ PASS | `openptv2-gui` available |
| Virtual display test | ✅ PASS | Works with xvfb |
| Core functionality | ✅ PASS | 15+ pytest tests pass |
| Tracker tests | ✅ PASS | Particle tracking works |

### Test Results Summary

```
GUI tests: 15+ passed
optv tests: All passed  
Tracker tests: All passed
Coordinate transforms: Working
Calibration: Working
```

## Manual GUI Testing

### 1. Launch the GUI

```bash
openptv2-gui
```

### 2. Load Test Data

The test fixtures are in `gui/tests/`:
- `gui/tests/test_rembg/` - Complete test dataset
- `gui/tests/test_splitter/` - Multi-camera test data
- `gui/tests/test_cavity/` - Cavity flow experiment

### 3. Test Workflows

#### Detection Workflow
1. Load parameters from YAML
2. Load an image
3. Run target detection
4. Verify targets appear

#### Calibration Workflow  
1. Load calibration files (.ori, .addpar)
2. Verify camera parameters display
3. Test epipolar lines

#### Tracking Workflow
1. Load sequence parameters
2. Run tracking
3. Verify particle tracks

## Troubleshooting

### GUI won't launch

Check dependencies:
```bash
python -c "import traits; import traitsui; import chaco; import PySide6; print('OK')"
```

### optv import errors

Verify optv is installed:
```bash
python -c "import optv; print(optv.__version__)"
```

Should show: `0.3.2` (local build)

### Test fixture not found

Test data should be in `gui/tests/`. If missing, the tests will skip gracefully.

## CI/CD Testing

The GitHub Actions workflow (`.github/workflows/cibuildwheel.yml`) automatically tests:
- Build succeeds
- Basic imports work
- Version is correct

For full GUI testing, a display or virtual framebuffer (xvfb) is needed.

## Next Steps

After GUI testing is complete, proceed to **Phase 2: Python/Numba Engine** as described in DESIGN_PLAN.md.
