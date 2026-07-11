# Particle-Based Calibration Refinement

This tutorial covers iterative calibration refinement using tracked particle
positions (`openptv-particle-calib`). After an initial plate calibration and
a first tracking pass, the 3D particle trajectories provide thousands of
additional calibration points scattered through the entire measurement volume —
far more than a calibration plate can provide.

## When to Use Particle Calibration

- After plate calibration + first tracking pass
- When correspondence quality (quadruplet/triplet counts) is lower than expected
- Iteratively: calibrate → track → refine → track until convergence

This is a refinement step, not a replacement for initial calibration.

## Prerequisites

1. Plate calibration complete (`cal/camN.tif.ori` and `.addpar` exist)
2. Tracking has been run and results are in `res/ptv_is.*`

## Setup

```bash
cd /path/to/openptv2
PC=skills/openptv-particle-calib/scripts/particle_calib.py
```

## Workflow

### 1. Check that tracking results exist

```bash
ls <dataset>/res/ptv_is.*
```

Must have at least a few files. If none, run tracking first.

### 2. Check potential improvement

```bash
uv run python $PC status <dataset>
```

Example output:
```
cam     n_pts    before_rms     after_rms
cam1       87    2.341px        1.823px
cam2       94    2.156px        1.640px
cam3       91    2.498px        1.901px
cam4       88    2.267px        1.712px
```

If `before_rms` is already < 0.5 px across all cameras, particle calibration
is unlikely to improve things further.

### 3. Dry run the full iteration loop

```bash
uv run python $PC run <dataset> --dry-run
```

Output:
```
Particle calibration: /path/to/dataset
  max_iters=5  tol_rms=0.05px  tol_px=5.0px
  Dry-run — no files will be written

iter     cam1     cam2     cam3     cam4  note
----------------------------------------------
   1    1.823    1.640    1.901    1.712
   2    1.792    1.615    1.873    1.688  Δ=-0.0615px
   3    1.778    1.601    1.862    1.673  Δ=-0.0205px
   4    1.775    1.598    1.860    1.669  Δ=-0.0067px

Converged (Δ < 0.05px). Done after 4 iteration(s).
```

If the RMS decreases each iteration, proceed to the real run.

### 4. Run for real

```bash
uv run python $PC run <dataset>
```

Files are written after each improving iteration. Originals backed up as
`*.pcbakN` where N is the iteration number.

### 5. Re-run tracking

After updating the calibration, re-run tracking to get improved particle
positions, then optionally repeat particle calibration:

```bash
# In the openptv2 GUI or via the tracking API
# Then check if another refinement round helps:
uv run python $PC status <dataset>
```

Stop iterating when `status` shows no more potential improvement.

## Options

```bash
# Match tolerance (default 5px — tighten for cleaner matches)
uv run python $PC run <dataset> --tol-px 3.0

# Use only selected frames (useful for large datasets)
uv run python $PC run <dataset> --frames 10,20,30,40,50

# More iterations before giving up
uv run python $PC run <dataset> --max-iters 10

# Tighter convergence threshold
uv run python $PC run <dataset> --tol-rms 0.01
```

## Troubleshooting

**`nan` in a camera column:**
The camera had fewer than 6 matched particles. Try:
- Increasing `--tol-px` (e.g. `--tol-px 8`)
- Checking that the target files exist for that camera
- Checking that the camera's `.ori` is not wildly wrong

**RMS not decreasing:**
- The calibration is already near-optimal for the available data
- Try plate re-calibration or dumbbell calibration to improve the starting point
- More tracking frames (longer sequence) give more calibration points

**RMS oscillates:**
Tighten `--tol-px` — noisy matches at the tolerance boundary cause instability.

**`ERROR: no res/ptv_is.* tracking results found`:**
Run tracking before particle calibration.

**Refractive index warning:**
Validate parameters first — a swapped n2/n3 causes systematically wrong
reprojections that particle calibration cannot correct.

## Iterative Loop Summary

```
Initial plate calibration
         ↓
    First tracking
         ↓
  Particle calibration ─┐
         ↓              │  repeat until status shows < 0.1px improvement
    Re-tracking  ───────┘
         ↓
   Final analysis
```
