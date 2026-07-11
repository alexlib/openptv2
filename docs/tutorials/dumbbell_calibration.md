# Dumbbell Calibration

This tutorial covers running dumbbell-based camera calibration using the
`openptv-dumbbell` skill. Dumbbell calibration is a self-calibrating method
where a two-point rigid body (dumbbell) is moved through the measurement
volume; all cameras are calibrated simultaneously without needing a fixed
calibration target in the measurement volume.

## When to Use Dumbbell Calibration

- When you cannot place a calibration plate inside the measurement volume
- To refine an existing plate calibration using in-situ measurements
- When the measurement volume is larger than the calibration plate

For initial calibration from scratch, consider plate calibration first
(`docs/tutorials/calibration.md`) and use dumbbell to refine it.

## Setup

```bash
cd /path/to/openptv2
DB=skills/openptv-dumbbell/scripts/dumbbell.py
```

## Dataset Requirements

```
<dataset>/
  parameters_<name>.yaml     ← must have a 'dumbbell' section
  cal/camN.tif.ori            ← existing initial calibration (needed as starting guess)
  cal/camN.tif.addpar
  img/                        ← dumbbell image sequence
```

## Configuring the YAML

Add or check the `dumbbell` section in your `parameters_*.yaml`:

```yaml
dumbbell:
  dumbbell_scale: 46.0          # physical distance between dumbbell points (mm)
  dumbbell_penalty_weight: 1.0  # 1.0 balances ray convergence vs length constraint
  dumbbell_eps: 0               # 0 = use all frames; >0 filters by length deviation
  dumbbell_step: 1              # use every Nth frame (1 = all, 5 = every 5th)
```

`dumbbell_scale` is the most important parameter — measure it carefully.

## Workflow

### 1. Validate parameters

```bash
uv run python $PP validate <dataset>    # PP = skills/openptv-params/scripts/params.py
```

Fix any refractive index issues before calibrating (see `docs/tutorials/parameters.md`).

### 2. Check the dumbbell section

```bash
uv run python $DB check <dataset>
```

All fields must show `OK`. Example:
```
OK: dumbbell.dumbbell_scale = 46.0  (known dumbbell length (mm))
OK: dumbbell.dumbbell_penalty_weight = 1.0  (weight of length vs ray error)
OK: dumbbell.dumbbell_eps = 0  (max length deviation to keep a frame (0 = keep all))
OK: dumbbell.dumbbell_step = 1  (frame stride through sequence)
```

### 3. Dry run

```bash
uv run python $DB run <dataset> --dry-run
```

Check the output:
```
Dumbbell calibration: /path/to/parameters.yaml

Result:
  Frames used:       42 / 100
  RMS before (px):   2.3410
  RMS after  (px):   1.1852
  Improvement:       +1.1558 px
  (dry-run: no files written)
```

A good result: RMS decreases, frames used is a reasonable fraction of total.

### 4. Commit the calibration

```bash
uv run python $DB run <dataset>
```

Writes updated `cal/camN.tif.ori` and `cal/camN.tif.addpar`. The originals are
backed up as `*.dbbak`.

## Advanced Options

```bash
# Use every 5th frame (faster, for long sequences)
uv run python $DB run <dataset> --step 5

# Keep cameras 0 and 2 fixed, only optimize 1 and 3
uv run python $DB run <dataset> --fixed-cams 0,2

# More optimizer iterations for a difficult dataset
uv run python $DB run <dataset> --maxiter 5000
```

## Troubleshooting

**RMS doesn't improve:**
- Check `dumbbell_scale` — it must be the actual physical distance in mm
- The initial calibration (`.ori` files) may be too far off; run plate
  calibration first
- Try fixing the best-calibrated cameras with `--fixed-cams` and only
  optimizing the worse ones

**Very few frames used:**
- Relax `dumbbell_eps` (set to 0 to keep all frames)
- Reduce `dumbbell_step` to use more frames

**RMS gets worse:**
- Check for swapped refractive indices (`openptv-params validate`)
- The optimizer may have diverged — run again with `--maxiter 500` to stop
  earlier, or use `--fixed-cams` to anchor the well-calibrated cameras
