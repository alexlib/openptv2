# Debug Tracking Visualization Plan

## Overview

Create an interactive marimo notebook for debugging tracking by visualizing correspondences, search volumes, and candidate selection directly from the algorithms module.

## Implementation Complete

The notebook is implemented at: `gui/notebooks/marimo_tracking_debug.py`

## Data Flow

```
YAML (ParameterManager) → calibrations → images → targets → correspondences → visualization
```

## Notebook Structure

| Cell | Purpose |
|------|---------|
| 1 | YAML path selection (configurable) |
| 2 | Load parameters from YAML using ParameterManager |
| 3 | Load 4 calibrations from cal_ori |
| 4 | Load images from sequence |
| 5 | Load pre-detected targets from _targets files |
| 6 | Load ControlParams and VolumeParams |
| 7 | Run correspondences using optv (Cython) |
| 8 | Interactive 4-panel visualization with click handler |

## Key Technologies Used

- **optv (Cython bindings)**:
  - `Calibration` - camera calibration
  - `ControlParams`, `VolumeParams` - parameters
  - `TargetArray`, `MatchedCoords` - target handling
  - `correspondences()` - find matches between cameras
  - `epipolar_curve()` - draw epipolar lines

- **algorithms**:
  - `read_targets()` - read target files

## Click Behavior

1. User clicks on a particle in any camera view
2. System finds nearest correspondence match (within 20px threshold)
3. System draws:
   - Crosshair on matched position (cyan circle)
   - Epipolar lines (cyan) to other cameras
4. System prints statistics to console

## Test Data

Default: `test_data/test_cavity/parameters_Run1.yaml`
- 4 cameras
- Frames 10001-10004
- Pre-detected targets available

## Usage

1. Open the notebook in marimo
2. Run cells 1-7 to load data and compute correspondences
3. Cell 8 shows interactive visualization
4. Click on any colored dot to see epipolar lines
