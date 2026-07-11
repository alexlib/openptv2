# openptv-particle-calib

## Overview
Iterative particle-based calibration refinement. After tracking, the 3D
particle positions in `res/ptv_is.*` are used as calibration targets — each
particle's known 3D position is matched to the nearest detected 2D target in
each camera image and the calibration is refined iteratively until the
reprojection RMS converges.

This is a "Phase 5" step: plate calibration → (optional) dumbbell calibration
→ tracking → particle calibration → re-tracking.

## Dependencies
- openptv2 checkout with `uv` venv, compiled `algorithms` modules
- Tracking must have already been run: `res/ptv_is.*` files must exist
- Run with `uv run python` from the openptv2 checkout

## Quick Reference
```
PC=skills/openptv-particle-calib/scripts/particle_calib.py
uv run python $PC status <dataset>           # show potential RMS improvement
uv run python $PC run    <dataset> --dry-run # iterate, no files written
uv run python $PC run    <dataset>           # iterate and write .ori/.addpar
```

## Workflow

### 1. Check that tracking results exist
```
ls <dataset>/res/ptv_is.*
```
Must have at least one `ptv_is.N` file. If not, run tracking first.

### 2. Check current calibration quality
```
uv run python $PC status <dataset>
```
Shows n_pts matched, RMS before (current cal) and RMS after (potential with
this iteration's improvement). If `before_rms` is already < 0.5 px, particle
calibration is unlikely to help further.

### 3. Dry run the iteration loop
```
uv run python $PC run <dataset> --dry-run
```
Prints a table: `iter | cam1 RMS | cam2 RMS | cam3 RMS | cam4 RMS | Δ note`.
Runs until convergence or `--max-iters`.

### 4. Run for real
```
uv run python $PC run <dataset>
```
Writes updated `.ori`/`.addpar` after each improving iteration.
Originals backed up as `*.pcbakN` (N = iteration number).

## Options
```
--max-iters N      stop after N iterations even if not converged (default 5)
--tol-rms F        convergence threshold: stop when Δ_mean_RMS < F px (default 0.05)
--tol-px F         particle↔target match radius in pixels (default 5.0)
--frames a,b,c     use only these frame numbers (default: all ptv_is.* files)
--dry-run          compute all iterations but do not write files
```

## Interpreting the Output
```
Particle calibration: /path/to/dataset
  max_iters=5  tol_rms=0.05px  tol_px=5.0px

iter     cam1     cam2     cam3     cam4  note
----------------------------------------------
   1    1.234    1.456    1.102    1.389
   2    1.198    1.321    1.089    1.298  Δ=-0.1265px
   3    1.187    1.290    1.083    1.271  Δ=-0.0620px
   4    1.183    1.281    1.081    1.266  Δ=-0.0172px

Converged (Δ < 0.05px). Done after 4 iteration(s).
```
- **RMS not decreasing**: the initial calibration is already optimal for the
  available particle data. More tracking frames or a better plate calibration
  is needed.
- **nan in a camera column**: fewer than 6 particles matched in that camera —
  increase `--tol-px` or check that target files exist for that camera.
- **Oscillating RMS**: tighten `--tol-px` to reduce noisy matches.

## When to Run This
- After initial plate calibration + first tracking pass
- When correspondence quality (quad/triplet counts) is poor
- Iteratively: run → track → run → track until quads plateau

## Common Mistakes
- **No ptv_is.* files**: must run tracking before particle calibration
- **tol_px too large**: if you use 10+ px tolerance, wrong particles get
  matched and calibration degrades — start with 5 px
- **n2/n3 swap**: always validate refractive indices with `openptv-params`
  first — a swapped n2/n3 makes particle matches systematically wrong
- **Skipping --dry-run**: always dry-run first to confirm improvement trend
  before committing new .ori files
