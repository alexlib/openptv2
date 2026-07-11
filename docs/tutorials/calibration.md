# Camera Calibration Tutorial

This guide walks you through calibrating cameras for an OpenPTV dataset using
the `calib.py` command-line tool bundled in this repository.

## Prerequisites

- openptv2 installed in its `uv` environment (`uv sync --extra dev`)
- A dataset in classic OpenPTV layout (see below)
- Calibration images (`cal/camN.tif`) and a detected-targets file (`cal/camN.tif_targets`) per camera
- A 3D calibration body file (`cal/target_on_a_side.txt`)

All commands must be run from the **openptv2 checkout** using `uv run` so the
correct Python environment is active.

```bash
cd /path/to/openptv2
SK=skills/openptv-calibrate/scripts/calib.py
```

## Dataset layout

```
<dataset>/
  parameters/ptv.par           # camera count, image size, pixel pitch, refractive indices
  parameters/sortgrid.par      # matching radius in pixels
  parameters/man_ori.par       # 4 calibration-point IDs per camera  (seed)
  parameters/man_ori.dat       # 4 clicked pixel coords per camera   (seed)
  cal/target_on_a_side.txt     # 3D calibration body: id  x  y  z
  cal/camN.tif                 # calibration image, one per camera
  cal/camN.tif.ori             # initial (or existing) orientation file
  cal/camN.tif.addpar          # distortion/additional parameters
  cal/camN.tif_targets         # detected dot targets (run target detection first)
```

## Step 1 — Inspect the dataset

```bash
uv run python $SK inspect <dataset> --output /tmp/inspect.json
cat /tmp/inspect.json
```

The JSON reports `ready_headless` (can run without a display), `has_seed`,
`has_targets`, and a `problems` list. Fix any problems before continuing.

## Step 2 — Create the orientation seed (if missing)

The seed is four well-separated calibration points per camera: their 3D IDs
and pixel coordinates. Skip this step if `has_seed` is already `true`.

### Interactive (recommended — needs a display)

```bash
uv run python $SK pick <dataset>
```

A window opens for each camera. Click the four highlighted points in the
calibration image in the order shown, then press any key. The seed files
`parameters/man_ori.par` and `parameters/man_ori.dat` are written after the
last camera.

### Headless fallback

Pre-render reference images:
```bash
uv run python $SK render <dataset> --output-dir <dataset>/cal/seed_help
```

Build a `seeds.json` from the rendered images:
```json
{"0": [[id,x,y], [id,x,y], [id,x,y], [id,x,y]],
 "1": [...],
 "2": [...],
 "3": [...]}
```

Write the seed files:
```bash
uv run python $SK seed <dataset> --seed-json /tmp/seeds.json
```

## Step 3 — Dry-run calibration

```bash
uv run python $SK run <dataset> --output /tmp/calib.json --dry-run
```

Inspect the per-camera output: `matched/nfix`, `RMS px`, and `flags`. RMS
below 2 px is good. Overlay images are written to `<dataset>/cal/auto_calib/`
(green = detected targets, red = reprojected model — red dots should sit
inside green).

## Step 4 — Write calibration

```bash
uv run python $SK run <dataset> --output /tmp/calib.json
```

Writes `cal/camN.tif.ori` and `cal/camN.tif.addpar` for every camera.
Previous files are backed up as `*.autobck`.

## Step 5 — Snapshot refinement (optional)

If tracking results exist in `res/ptv_is.*`, use them to refine the
calibration using real particle 3D positions as additional control points:

```bash
uv run python $SK snapshot-refine <dataset> --dry-run   # preview
uv run python $SK snapshot-refine <dataset>             # write
```

Each camera reports:
```
cam1: 47 pts  before=2.930px  after=2.590px  flags=[]
```

The previous `.ori`/`.addpar` files are backed up as `*.snpbck` /
`*.addpar.snpbck`.

After snapshot refinement, re-run tracking to take advantage of the improved
calibration, then optionally run snapshot-refine again for a second-pass
improvement.

## Refractive index note

The `ptv.par` parameter file (or its YAML equivalent) defines three refractive
indices in this order:

```
mmp_n1   # air (always 1.0)
mmp_n2   # glass window (typically 1.46)
mmp_n3   # water (typically 1.33)
```

A common mistake is swapping n2 and n3. Check your parameters file if
reprojection errors are unexpectedly high after calibration.

## Full command reference

```
uv run python $SK inspect        <dataset> --output F
uv run python $SK render         <dataset> --output-dir D
uv run python $SK pick           <dataset> [--ids a,b,c,d]
uv run python $SK seed           <dataset> --seed-json F
uv run python $SK run            <dataset> --output F [--dry-run]
uv run python $SK snapshot-refine <dataset> [--tol-px N] [--frames F1,F2] [--dry-run]
```
