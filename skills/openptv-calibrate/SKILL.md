---
name: openptv-calibrate
description: >-
  Turnkey multi-camera calibration for an OpenPTV / openptv2 dataset, end to
  end, without the GUI. Use when a user wants to calibrate cameras, compute
  .ori/.addpar orientation files, fix a bad calibration, or set up a new PTV
  experiment and does not know the steps. Inspects the dataset, guides creation
  of the manual-orientation seed (interactive mouse click-picker) if missing,
  runs external orientation -> sortgrid -> bundle adjustment, and verifies with
  reprojection-overlay images and RMS. Triggers: "calibrate my cameras",
  "calibrate test_cavity", "make the .ori files", "my PTV calibration is off",
  "set up calibration".
---

# openptv-calibrate

## Overview
Takes a user from "I have calibration images and a target body" to verified
camera calibration (`.ori` + `.addpar` per camera) with no GUI point-picking
knowledge required. The numerical pipeline lives in the `openptv2` package
(`openptv2.autocalibration.calibrate_dataset`); this skill drives it and handles
onboarding: dataset inspection, the manual-orientation seed (via mouse clicks),
and visual verification.

Pipeline per camera:
`external_calibration` (4 seed points) → `sortgrid` (match full 3D body to
detected targets) → refine loop → `full_calibration` (bundle adjustment, best
distortion flag-set by RMS).

## Dependencies
- **openptv2** installed in the active `uv` venv (compiled `algorithms` modules
  + `openptv2.autocalibration`). Run every command with `uv run` from the
  openptv2 checkout.
- Env deps already present: `numpy`, `matplotlib`, `imageio`. The interactive
  `pick` step needs a display (TkAgg/QtAgg backend).
- No new skills required. Reuses repo code — do NOT reimplement the solver.

## Expected dataset layout
Classic OpenPTV (e.g. `test_data/test_cavity`):
```
<dataset>/
  parameters/ptv.par        control params: cams, image size, pixel size, mm
  parameters/sortgrid.par    matching radius in px
  parameters/man_ori.par     4 calibration-point IDs per camera   (seed)
  parameters/man_ori.dat     4 clicked pixel coords per camera     (seed)
  cal/target_on_a_side.txt   3D calibration body: id x y z         (calblock)
  cal/camN.tif               calibration image per camera
  cal/camN.tif.ori/.addpar   existing calibration = initial guess
  cal/camN.tif_targets       detected targets per camera
```

## Quick Start (headless, when the seed already exists)
```
SK=skills/openptv-calibrate/scripts/calib.py
uv run python $SK inspect <dataset> --output /tmp/inspect.json
uv run python $SK run     <dataset> --output /tmp/calib.json
```
Then show the overlay PNGs in `<dataset>/cal/auto_calib/` and confirm the report.

## Workflow (checkpoint at every stage — pause for the user between each)
`SK=skills/openptv-calibrate/scripts/calib.py`

### 1. Inspect — understand the dataset
```
uv run python $SK inspect <dataset> --output /tmp/inspect.json
```
Read the JSON; tell the user in plain language what is present/missing
(`ready_headless`, `problems`).
- `has_targets` false → targets must be detected first; stop (out of scope).
- `ready_headless` true → skip to step 3.
- only the seed missing (`has_seed` false) → step 2.
**Checkpoint:** summarize findings; confirm before continuing.

### 2. Seed — create the manual orientation (only if missing)
The seed = 4 well-spread, unambiguous calibration points per camera: their **3D
IDs** and their **pixel (x,y)** in each cal image. Prefer the interactive
click-picker; fall back to manual entry only with no display.

**2a. Interactive (recommended — guided mouse clicks).**
Optionally pre-render the reference (the picker also shows it live):
```
uv run python $SK render <dataset> --output-dir <dataset>/cal/seed_help
```
Then launch the guided picker (uses 4 corner IDs by default, or pass your own):
```
uv run python $SK pick <dataset>            # or: pick <dataset> --ids 2,3,71,73
```
The picker tells the user *which* point to click, one at a time — no typing of
IDs needed. **Instruct the user through the UI, exactly:**
1. A two-panel window opens for **camera 1**: the cal image (left) and a
   numbered map of the 3D body (right) with axes **X → (left-right), Y ↑
   (bottom-top)**.
2. The title says **"CLICK point ID N (k/4)"** and that same point N is circled
   in red on the map. **Left-click that exact point** in the cal image.
3. It advances to the next of the 4 points automatically; repeat until all 4
   are clicked. Each click is marked with a yellow "+" and its ID.
4. After the 4th click the script draws the **initial-guess overlay** — the
   whole 3D body reprojected in red using the just-computed orientation. Red
   dots should land on the real target dots. **Press any key** to accept and
   move to the next camera. (If the red cloud is clearly wrong, the seed was
   mis-clicked — re-run `pick` for a clean result.)
5. After the last camera it writes `man_ori.par` + `man_ori.dat`.

The same 4 IDs are used for every camera, so the correspondence is unambiguous.

**2b. Fallback (no display / headless).** Use the saved `camN_grid.png` +
`body_ids.png` to read coordinates by eye, build `seeds.json`
(`{"0": [[id,x,y] x4], "1": [...], ...}`), then:
```
uv run python $SK seed <dataset> --seed-json /tmp/seeds.json
```
**Checkpoint:** re-run `inspect`; confirm `has_seed` is true before running.

### 3. Calibrate — dry run first
```
uv run python $SK run <dataset> --output /tmp/calib.json --dry-run
```
Report per-camera `matched/nfix`, `RMS px`, `flags`. Interpret:
- RMS ≲ 2 px is good. A much higher RMS on one camera usually means a bad seed
  click or wrong ID — redo step 2 for that camera.
- `matched` well below `nfix` is normal when the body is partly occluded in that
  view (check the overlay). It matters only if the RMS is also high.
Show overlay PNGs from `<dataset>/cal/auto_calib/` (green = detected,
red = reprojected; red should sit inside green).
**Checkpoint:** confirm results look right before writing.

### 4. Write — commit the calibration
```
uv run python $SK run <dataset> --output /tmp/calib.json
```
Writes `cal/camN.tif.ori` / `.addpar`; originals backed up as `*.autobck`.
Tell the user where files + overlays are, and the final mean RMS.

### 5. Snapshot Refinement — use tracking results to further improve calibration
After step 4, if `res/ptv_is.*` tracking results exist, run this to refine
calibration using real 3D particle positions as additional control points:
```
uv run python $SK snapshot-refine <dataset> --dry-run        # preview
uv run python $SK snapshot-refine <dataset>                   # write
```
Per-camera output: `N pts  before=X.XXXpx  after=Y.YYYpx  flags=[...]`

How it works: projects 3D particle positions (from tracking) onto each camera
image, matches to detected targets within `--tol-px` (default 5 px), then
runs bundle adjustment with the matched pairs. Originals backed up as
`*.snpbck` / `*.addpar.snpbck`.

**Interpret:** improvement of 0.1–0.5 px is typical. The before/after RMS is
against noisy tracking data; run `run --dry-run` after to verify the cal plate
RMS is preserved or improved.

**Note:** tracking results from before calibration fixing may be slightly off;
best practice is to re-run tracking with the new calibration, then
snapshot-refine again for a second-pass improvement.

**Maximizing quadruplets/triplets:** after snapshot-refine, re-run tracking
to get more correspondences seen across cameras, which feeds back into
snapshot-refine for the next iteration.

## Utility Scripts
`scripts/calib.py <subcommand>` — all write to files; stdout is short status.
- `inspect <dataset> --output F` — readiness JSON.
- `render <dataset> --output-dir D` — cal-image grids + labeled 3D body.
- `pick <dataset> [--ids a,b,c,d]` — interactive mouse click-picker for the seed.
- `seed <dataset> --seed-json F` — write `man_ori.par` + `man_ori.dat` from JSON.
- `run <dataset> --output F [--dry-run]` — calibrate, overlays, report JSON.
- `snapshot-refine <dataset> [--tol-px N] [--frames F1,F2] [--dry-run]` — refine from tracking results.

## Common Mistakes
- **Running outside the openptv2 venv** — imports fail. Always `uv run` from the
  checkout.
- **Inconsistent click order across cameras** — `pick` pairs click *k* with ID
  *k*; click the same physical points in the same order for every camera.
- **Confusing the seed files** — `man_ori.par` = point **IDs**;
  `man_ori.dat` = pixel **coords**. `pick`/`seed` write both.
- **Treating low `matched` as failure** — usually occlusion; judge by RMS + the
  overlay, not match count.
- **Skipping the dry run** — always dry-run and eyeball overlays before writing
  over existing `.ori`/`.addpar`.
