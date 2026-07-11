# openptv-dumbbell

## Overview
Run dumbbell-based camera calibration — a self-calibrating method where a
two-point rigid body (dumbbell) is moved through the measurement volume and
all cameras are calibrated simultaneously without a fixed calibration target.

Wraps `openptv2.gui.standalone_dumbbell_calibration.run_dumbbell_calibration`
as a discoverable CLI tool.

## Dependencies
- openptv2 checkout with `uv` venv, compiled `algorithms` modules
- Run with `uv run python` from the openptv2 checkout
- YAML must have a `dumbbell` section with `dumbbell_scale` set

## Quick Reference
```
DB=skills/openptv-dumbbell/scripts/dumbbell.py
uv run python $DB check <dataset>           # validate YAML dumbbell section
uv run python $DB run   <dataset>           # calibrate, write .ori/.addpar
uv run python $DB run   <dataset> --dry-run # compute only, no files written
```

## YAML Requirements
The dataset's `parameters_*.yaml` must have a `dumbbell` section:
```yaml
dumbbell:
  dumbbell_scale: 46.0          # known length between the two points (mm)
  dumbbell_penalty_weight: 1.0  # relative weight: ray convergence vs length error
  dumbbell_eps: 0               # max allowed length deviation (0 = keep all frames)
  dumbbell_step: 1              # use every Nth frame (1 = all frames)
```
Run `check` to validate before `run`.

## Workflow

### 1. Check the configuration
```
uv run python $DB check <dataset>
```
All fields must show `OK`. Fix any `MISSING` or `ERROR` lines before continuing.

### 2. Dry run first
```
uv run python $DB run <dataset> --dry-run
```
Reports frames used, RMS before/after. Check that RMS improves.

### 3. Commit the calibration
```
uv run python $DB run <dataset>
```
Writes `cal/camN.tif.ori` and `cal/camN.tif.addpar` for all cameras.
Originals are backed up as `*.dbbak`.

## Options
```
--step N           frame stride (overrides YAML dumbbell_step)
--fixed-cams 0,2   keep cameras 0 and 2 fixed (0-based), only move others
--maxiter 1000     optimizer iteration limit
--dry-run          compute but do not write files
```

## When to Use Dumbbell vs Plate Calibration
- **Plate calibration** (`openptv-calibrate`): needs a physical calibration
  target, best for initial calibration when you have a good target body.
- **Dumbbell calibration** (`openptv-dumbbell`): self-calibrating from motion;
  useful when the plate is unavailable, or to improve an existing calibration
  using in-situ measurements.

## Interpreting Results
```
Frames used:     42 / 100       # 58 frames excluded by dumbbell_eps filter
RMS before (px): 2.3410
RMS after  (px): 1.1852
Improvement:     +1.1558 px
```
- **RMS > 3 px** after calibration: the dumbbell_scale may be wrong, or the
  initial calibration is very far off — run plate calibration first.
- **Few frames used**: relax `dumbbell_eps` (increase or set to 0) or reduce
  `dumbbell_step` to use more frames.
- **No improvement**: try fixing the best-calibrated cameras with `--fixed-cams`
  and only optimizing the worse ones.

## Common Mistakes
- **dumbbell_scale wrong**: must be the actual physical distance between the
  two dumbbell points in millimeters.
- **Running without a prior calibration**: dumbbell refinement needs a
  reasonable starting guess in `cal/camN.tif.ori`. Run plate calibration first
  if `.ori` files are missing or clearly wrong.
- **n2/n3 swap**: validate refractive indices with `openptv-params` before
  running — a swapped n2/n3 will cause the optimizer to converge to a wrong
  minimum silently.
