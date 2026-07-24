---
name: openptv-calibrate
description: >-
  Turnkey multi-camera calibration for an OpenPTV / openptv2 dataset, end to
  end, without the GUI. Use when a user wants to calibrate cameras, compute
  .ori/.addpar orientation files, fix a bad calibration, or set up a new PTV
  experiment and does not know the steps. Inspects the dataset, guides creation
  of the manual-orientation seed (interactive mouse click-picker) if missing,
  runs external orientation -> sortgrid -> bundle adjustment, and verifies with
  reprojection-overlay images and RMS. Also covers image-splitter (4 views
  tiled into one multiplexed frame) datasets: quadrant-order verification,
  ID-labeled overlays, staged recalibration when the seed is degenerate, and
  an interactive 3D setup viewer. Triggers: "calibrate my cameras", "calibrate
  test_cavity", "make the .ori files", "my PTV calibration is off", "set up
  calibration", "calibrate the splitter", "image splitter calibration".
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

## Truly fresh dataset (no targets, no seed, no prior .ori/.addpar at all)
A dataset that never went through calibration before is missing three things
`inspect` now names explicitly in `problems`, each with its own fix:
```
uv run python $SK inspect <dataset> --output /tmp/inspect.json   # names all 3 gaps
uv run python skills/openptv-calibrate/scripts/detect_targets.py <dataset>  # targets
uv run python $SK init <dataset>                                  # initial guess
# then step 2 (seed) below, then step 3 (run)
```
- `detect_targets.py` runs the same detection the GUI's calibration
  "Detection" button uses (`detect_plate` thresholds from the dataset YAML),
  closing the gap `inspect`'s own `has_targets` check used to dead-end on
  ("targets must be detected first; stop (out of scope)") -- detection isn't
  really *calibration*, it's a prerequisite this skill can now do for you.
- `init` writes a naive default `.ori`/`.addpar` per camera missing one:
  `external_calibration`'s exterior solve (`raw_orient`) needs a starting
  interior-parameter guess (cc/xh/yh) as a fixed input even though it
  recomputes the exterior pose from the 4 seed points -- there is no
  calibration path that needs literally zero prior information. The default
  guesses `cc` from sensor width (`imx * pix_x`, ~90deg FOV) and places the
  camera on the -Z axis at `--distance` (default 500mm) with identity
  rotation. **This mechanically unblocks `run` -- it does not reliably
  produce a *useful* calibration.** On a real two-splitter-rig dataset tested
  here, 3 of 4 cameras "converged" (`raw_orient` returned success) but
  matched 0/135 calblock points once sortgrid ran -- the pose was numerically
  stable but pointed nowhere near the actual target, and the 4th camera
  didn't converge at all. **Always check `run --dry-run`'s `matched` count,
  not just whether it raised** -- a converged-but-wrong pose is silent
  otherwise. If nothing converges usefully, the missing ingredient is a rough
  *orientation* guess (roughly which way each camera points), not more code:
  ask the user for it, or seed `init` from a similar rig's prior calibration
  if one exists (see "several runs sharing one calibration" in
  `docs/multi-folder-runs.md`) rather than a blind default.
- `calibrate_dataset` treats each camera independently: one camera failing
  to converge no longer aborts the whole dataset's report (`CamResult.error`
  is set instead of raising; `run`'s JSON report and printed summary show
  per-camera pass/fail so you can see which cameras need attention without
  losing the ones that worked).

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
- `init <dataset> [--distance MM]` — naive default `.ori`/`.addpar` for cameras
  missing one entirely (see "Truly fresh dataset" above for what it can and
  can't do).
- `render <dataset> --output-dir D` — cal-image grids + labeled 3D body.
- `pick <dataset> [--ids a,b,c,d]` — interactive mouse click-picker for the seed.
- `seed <dataset> --seed-json F` — write `man_ori.par` + `man_ori.dat` from JSON.
- `run <dataset> --output F [--dry-run]` — calibrate, overlays, report JSON.
- `snapshot-refine <dataset> [--tol-px N] [--frames F1,F2] [--dry-run]` — refine from tracking results.

Standalone scripts (each takes `<dataset>` as `sys.argv[1]`; run with
`uv run python skills/openptv-calibrate/scripts/<name>.py <dataset> ...` from
the openptv2 checkout — no per-machine path edits needed, unlike earlier
one-off copies of these that lived inside dataset folders):
- `detect_targets.py <dataset>` — detects calibration-plate targets for every
  camera (`detect_plate` thresholds from the dataset YAML, same code path as
  the GUI's calibration "Detection" button) and writes `cal/camN.tif_targets`.
  Run this first on a fresh dataset, or whenever existing `_targets` files
  turn out to be stale placeholders (see the pitfall below).
- `recalibrate_constrained.py <dataset>` — re-fit with addpar zeroed (no
  distortion, `cc` fixed), only exterior + `xh,yh` free. Diagnostic: does an
  odd-looking pose survive with distortion removed, or was it overfit?
- `recalibrate_full.py <dataset>` — staged full recalibration: runs
  `recalibrate_constrained` first, then re-enables `cc`+distortion from that
  converged pose. **Also the recovery path when `calib.py run` fails with
  "external_calibration did not converge"** (degenerate/reused man_ori seed)
  — it starts from the existing on-disk `.ori` instead of reseeding, so it
  works even when the seed itself is unusable, as long as a real (non-
  placeholder) prior calibration already exists on disk.
- `recalibrate_exterior_only.py <dataset> [--dry-run]` — refine ONLY the 6
  exterior DOF (position + angles); interior (cc/xh/yh) and distortion are
  left exactly as they are on disk. For when interior was set by hand (e.g.
  a known/measured focal length) and is trusted, and you just want a real
  bundle-adjustment polish against all detected targets rather than
  `calib.py run`'s always-reseed-with-cc/xh/yh-free behavior. Uses
  `full_calibration(..., flags=[])` -- orient()'s "raw-like" mode. If the
  on-disk pose is too far off for sortgrid's pixel-radius matching to find
  anything, it automatically also tries reseeding exterior from the YAML's
  `man_ori` seed (`external_calibration` solves pose directly from 4
  correspondences, no close starting guess needed) and keeps whichever pose
  sortgrid matches more points against -- verified on a real 4-camera
  splitter rig where the hand-set poses for 3 of 4 cameras reprojected
  hundreds of pixels off-frame; reseeding got all 4 to 72-81/135 matched,
  RMS 1.5-2.9px, cc/xh/yh unchanged throughout.
- `robust_calibrate.py <dataset> [--dry-run] [--target 1.0] [--min-keep 0.6] [--mad K]`
  — RANSAC/IRLS-style outlier rejection to reach a **sub-pixel** fit. A POLISH
  step run AFTER any calibration (`calib.py run`, `recalibrate_*`, GUI): starts
  from each camera's on-disk `.ori`/`.addpar`, sortgrids, then removes the
  worst-reprojecting correspondences (greedy: drop worst, refit, until inlier
  RMS ≤ `--target`; or `--mad K`: one-shot median+K·MAD cutoff) and refits the
  full model over the survivors. A plain bundle adjustment is dragged by the
  few points the pinhole+Brown model genuinely can't reproject (a mismatched
  sortgrid ID, a splitter-prism-distorted edge dot, an occluded blob); dropping
  them yields a clean fit over the points the model CAN represent. On a real
  4-cam splitter rig this took all four cameras from 1.2–2.4px to ~0.98px.
  **Trade-off (why it is not free accuracy, and why it is opt-in, not a silent
  default):** rejecting a point because the model can't fit it can mean hiding
  real model error at the frame edges rather than removing a bad measurement.
  `--min-keep` (default 0.6) guards against trimming the pose loose, and the
  printed `cover=` (surviving-points bbox as a fraction of the image) flags a
  camera that collapsed to a central band — that camera is then only trustworthy
  where its inliers are. **Always** re-run `reproject_on_combined.py` after: the
  FULL calblock must still land on the dots (global pose preserved) even for a
  trimmed camera. Greedy mode maximizes sub-pixel-ness (trims more); `--mad` is
  gentler (keeps coverage, may not reach sub-pixel). Backups: `*.robustbck`.
- `detection_params_demo.py` — interactive marimo explainer for the target-
  detection parameters (`gvthres`, `nnmin/nnmax`, `nxmax/nymax`, `sumg_min`,
  and the confusing `disco`/`tol_dis`). Self-contained (pure
  numpy/scipy/skimage, no compiled openptv2), so it runs in a sandbox; it
  models the SAME semantics as `target_recognition` on a synthetic
  calibration-like scene, one slider per parameter, live overlay of which
  blobs survive and which rule rejected each. Includes a dedicated `disco`
  demo (two overlapping dots) showing the split-vs-merge rule: a saddle deeper
  than `disco` splits two overlapping particles, shallower merges them —
  which is why `detect_plate` (high disco, plate dots) and `targ_rec` (low
  disco, crowded particles) detect the same image differently. Open with:
  `uv run marimo edit --sandbox skills/openptv-calibrate/scripts/detection_params_demo.py`
- `dump_matches.py <dataset>` — writes `cal/calib_matches/camN_matches.txt`
  (id, detected px, reprojected px) and `camN_overlay_ids.png` (like the
  usual overlay, but with the calibration-body point ID labeled next to each
  dot). Run this after any calibration to get overlays you can actually debug
  from — the default overlays (green/red dots, no IDs) don't let you tell
  *which* point failed to detect or match.
- `reproject_on_combined.py <dataset> <path-to-combined.tif> [--verify-order]`
  — projects the calblock through every camera's calibration and places each
  camera's points at the correct quadrant offset within the original
  un-split multiplexed frame (all 4 cameras' fit visible on one image).
  `--verify-order` empirically checks the current `ptv.splitter_order`
  against the raw pixel data (and the alternate ordering) rather than
  trusting the convention by assumption — **always run this once per new
  rig** before believing any quadrant labels; wrong order gives ~70 grey-
  level mean diff vs. ~1 for the correct one, so it's an unambiguous check.
- `tune_eps0.py <dataset> [frame]` — visual assistance for picking
  `criteria.par`'s `eps0` (epipolar-band half-width used by correspondence
  matching, `find_candidate()` in `algorithms/epi.py` — unrelated to
  calibration itself). Sweeps `eps0` against real detected targets from one
  frame (needs `img/camN.<frame>_targets` and an existing calibration) and
  plots quad/triplet/pair counts vs. `eps0`, colored to match the GUI's own
  pair(yellow)/triplet(green)/quad(red) overlay. Pick the value at the
  "knee" where the quad curve flattens — tighter throws away real matches
  for no precision gain (they show up as green/triplet in the GUI, one
  camera short of the 3D intersection); looser mainly inflates pairs/
  triplets with spurious matches, not more real quads. A good starting
  point is a few pixels (`eps0 = N * pix_x` for N in 2-3), not the ~1px
  default some datasets ship with, which is usually too tight relative to
  the calibration's real reprojection RMS.
- `estimate_eps0_from_calibration.py <dataset>` — same goal as `tune_eps0.py`
  but usable *immediately after calibration, before any sequence frame
  exists*: reuses `cal/calib_matches/camN_matches.txt` (calibration-plate
  detections, from `dump_matches.py`) instead of real particle data. For
  every calibration-body point ID detected in both cameras of a pair, computes
  camera A's actual epipolar line in camera B (`epi_mm()`) and measures how
  far camera B's real detection sits from it — the same residual real
  correspondence search sees, driven purely by the calibration's own
  cross-camera consistency. Prints per-pair and combined percentiles and
  suggests `eps0 ≈ 1.5x` the combined p95. On a real dataset this landed
  within ~10% of the value found by sweeping real particle data with
  `tune_eps0.py` — a useful starting point, not a replacement for confirming
  against real sequence data once it exists (the calibration plate's clean,
  well-separated dots don't capture particle-image noise/occlusion/overlap).
- `plot_calblock_3d.py` — interactive marimo notebook: **just** the 3D
  calibration body (fixp/calblock), IDs labeled, mouse-drag rotation. Needs
  only the calblock -- no camera `.ori`/`.addpar` required, so use this
  *first*, before any calibration exists, to help the user understand the
  target's physical layout (which points are corners/edges, whether it's a
  flat plate or a multi-plane staircase body, etc.) before picking manual-
  orientation seed points. Accepts a dataset dir, a `parameters_*.yaml`, or a
  calblock `.txt` directly:
  ```
  uv run marimo edit --sandbox --no-token \
      skills/openptv-calibrate/scripts/plot_calblock_3d.py \
      -- --target "<dataset-or-yaml-or-calblock.txt>"
  ```
  Once calibration exists (`.ori`/`.addpar` per camera), move on to
  `visualize_calibration_setup.py` below for the full setup with camera poses.
- `visualize_calibration_setup.py` — interactive marimo notebook: 3D setup
  (world frame, ID-labeled calibration body, camera poses labeled by
  splitter quadrant, mouse-drag rotation via `mo.mpl.interactive`) plus the
  same ID-labeled 2D overlays as `dump_matches.py`. Runs directly from this
  checkout via marimo's own `--` CLI-args convention -- **no per-dataset
  copy** (a copied notebook was itself, once, the last thing standing
  between a corrupted calibration and its only local backup — don't
  reintroduce that). Open with the marimo-pair skill:
  ```
  uv run marimo edit --sandbox --no-token \
      skills/openptv-calibrate/scripts/visualize_calibration_setup.py \
      -- --dataset "<dataset>"
  ```
  (Still falls back to `mo.notebook_dir()` if no `--dataset` is given, so a
  copy would work too — just don't make one on purpose.)

## Image-splitter datasets (4 views tiled into one multiplexed frame)

A splitter rig records one combined image (e.g. 1024x1024) that's split into
4 quadrants, one per camera — either live by openptv2 (`ptv.splitter: true`,
`image_split()` in `openptv2/gui/ptv.py`) or offline by an external script
(e.g. a MATLAB script writing `Cam1.NNNNNN.tif`..`Cam4.NNNNNN.tif` per frame,
which is what pre-split `cal/cam_N.tif` calibration images usually mean).
Recurring pitfalls found while calibrating two real splitter datasets:

- **Quadrant order is `[top-left, top-right, bottom-left, bottom-right]` at
  raw indices `[0,1,2,3]`** (`image_split`'s slice order), but
  `ptv.splitter_order` (default `[0,1,3,2]`) remaps that into
  cam1=TL, cam2=TR, **cam3=BR, cam4=BL** — a "clockwise" order, not simple
  reading order. Do not assume; verify with
  `reproject_on_combined.py ... --verify-order` against a real un-split
  frame if one exists.
- **Camera/calblock file paths come from the dataset YAML, not a fixed
  `camN.tif` convention.** Both real splitter datasets seen so far had
  `parameters/ptv.par`/`parameters_Run1.yaml` referencing `cam_N.tif`
  (underscore) — a different name than the classic `camN.tif` convention
  this skill's code originally assumed everywhere. Earlier versions of this
  skill worked around that by copying `cam_N.tif` → `camN.tif` (and the
  calblock to `cal/target_on_a_side.txt`) by hand, syncing results back
  after every calibration. **Don't do that anymore** — it's exactly the kind
  of fragile duplication that once caused real data loss (the adapter
  copies got cleaned up as apparent clutter mid-session, and with them the
  only local backup of a calibration that had just been corrupted by a
  diverged manual GUI fit). `openptv2.autocalibration.cam_files(base, cam)`
  and `resolve_calblock(base)` now resolve straight from the dataset YAML's
  `cal_ori.img_cal_name` / `img_ori` / `fixp_name` (falling back to the
  classic `camN.tif` convention only when no YAML exists), and every script
  in this skill uses them — so there is exactly one copy of every file, the
  one the GUI itself reads and writes. If you're extending this skill,
  always resolve camera paths through `cam_files()`/`resolve_calblock()`
  rather than hardcoding `cal/cam{N}.tif` — that hardcoding is the root
  cause this whole pitfall came from.
- **Existing `camN.tif_targets` are often stale placeholders** (e.g. 2 points
  from a manual dumbbell click, not real plate detection). `inspect`'s
  `has_targets` only checks file existence, not content — always look at the
  file content (`cat cal/cam1.tif_targets`) before trusting it. Real
  detection: `openptv2.segmentation.target_recognition` with
  `parameters/detect_plate.par` thresholds (not `targ_rec.par`, which is
  tuned for smaller/dimmer particle targets, not the calibration plate's
  larger dots).
- **`external_calibration did not converge`**: usually a degenerate man_ori
  seed — e.g. the same 4 point IDs reused unchanged across every camera, and
  those 4 points nearly collinear (weak/singular pose solve). If the dataset
  already has a real prior calibration on disk, skip reseeding entirely with
  `recalibrate_full.py` (see above) rather than trying to fix the seed.
- **A physically-implausible pose (huge, asymmetric camera-to-target
  distances) can still show good per-camera RMS.** A shallow calibration
  body (small z-depth vs. camera distance) has a weak constraint on
  depth-along-viewing-axis; `recalibrate_constrained.py` (addpar removed) is
  a useful diagnostic — if the asymmetry survives with cc/distortion fixed,
  it's a real pose issue (bad seed or ambiguous geometry), not overfitting.
- **Don't loosen detection thresholds to catch more points without
  re-testing the full calibration.** Tightening `detect_plate.par`'s
  `discont` from 20 down to 8 raised raw detection counts significantly, but
  running the actual calibration on the richer detections made RMS *much*
  worse (0.7px → 4px on one camera) — the extra detections were mostly noise
  blobs or fragmented dots that `sortgrid` mismatched, corrupting the bundle
  adjustment. If points are missing near the current thresholds, verify with
  the flood-fill size check below before changing anything project-wide.
- **Why a specific point isn't detected**: `detect_plate`'s flood-fill grows
  each candidate blob outward (up to `discont` grey-level change per step),
  then discards the result if its bounding box exceeds `nxmax`/`nymax`/
  `nnmax` (tuned for an isolated ~7px dot). A dot near glare/reflection/a
  bright fixture can flood into a much larger connected region and get
  rejected entirely — and since the flood also marks every pixel it touched
  as consumed, it silently steals neighboring dots' peaks too, so several
  adjacent points vanish together. This is a real image/lighting limitation,
  not a calibration bug — confirm it by re-flooding from that dot's expected
  pixel location and checking the resulting bbox against the configured
  bounds before concluding anything is broken.

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
