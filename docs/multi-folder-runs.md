# Multi-folder experiments

For an experiment that's really several runs sharing one calibration --
multiple workpieces, repeated trials, whatever your acquisition splits into
separate frame sequences. See `docs/cloud-batch.md` first for how a single
`parameters_*.yaml` runs; this doc is about organizing and running several of
them together, and the calibration/YAML-prep steps specific to that.

This grew out of a real multi-workpiece dataset and the mistakes made
setting it up -- the checklists below exist because each one bit us once.

## Layout

```
<experiment_root>/
  <shared_calibration>/        one calibration, reused by every run below
    cam_1.tif ... cam_N.tif            calibration images (what .ori/.addpar were fit to)
    cam_1.tif.ori ... .addpar          exterior + interior/distortion params, per camera
    <calblock>.txt                     3D calibration-body point coordinates (id x y z)
    calib_matches/cam{1..N}_matches.txt   detected-vs-reprojected residuals (optional, enables RMS checks)

  run1/ run2/ .../              one per independent run
    img/                               raw acquisition frames
    parameters_<name>_sample.yaml      small frame range, for a quick check
    parameters_<name>_batch.yaml       full frame range
    res/                               sequence/tracking output (created on first run)
```

**Exactly one YAML pair per run.** If a folder has more than
`*_sample.yaml` + `*_batch.yaml`, one is stale. Cross-check against that
folder's `parameters/*.par` (if present -- the GUI's own last-saved state,
independent of any YAML) to see which one the GUI actually agrees with, and
remove the other. An ambiguous duplicate left lying around means the next
person -- or the next agent -- has no way to know which one is live.

## Path resolution: relative to the YAML, not to cwd

Both the GUI and `openptv2-batch` `os.chdir()` into the YAML file's own
directory before doing anything else (`ParameterManager.from_yaml` itself
does zero path resolution -- it just parses the file; every relative path in
it is resolved by whatever the process's cwd happens to be when it's used).
That has two consequences:

- A YAML living in `run1/` referencing `../<shared_calibration>/cam_1.tif.ori`
  resolves correctly, one level up from `run1/`.
- **Output is also cwd-relative and NOT configurable.** `res/rt_is.*`,
  `ptv_is.*`, `added.*` are hardcoded relative paths (no YAML field
  overrides them). If two YAMLs ever share the same parent directory, they
  write to the *same* `res/` and silently clobber each other's results.
  This is why each run needs **its own folder** -- don't try to save
  "tidiness" by hoisting multiple runs' YAMLs into one shared directory;
  that trades a real correctness risk for a cosmetic win. (Parallel
  execution across runs is still fine -- separate folders means separate
  cwd means separate `res/`, safe to run concurrently.)

## Preparing one shared calibration for all runs

1. **Confirm it's actually the same physical rig setup.** Compare exterior
   camera positions (`cam_N.tif.ori`, first line: X Y Z in mm) against any
   calibration already in the target experiment, if one exists. Tens of mm
   apart in any camera means the rig moved (or it's a different rig/session
   entirely) -- do not share, calibrate that experiment separately. This is
   the single most consequential mistake to avoid: a calibration that loads
   without error and even produces plausible-looking correspondence counts
   can still be geometrically wrong for the wrong rig -- nothing in the
   pipeline itself will tell you, you have to check (see below).
2. **Match the acquisition modality**, not just the rig position. A
   calibration shot as 4 separate per-camera images (`ptv.splitter: false`)
   is not automatically compatible with a run that multiplexes 4 views onto
   one sensor (`ptv.splitter: true`) -- the "camera 1/2/3/4" identity in the
   splitter case comes from `splitter_order` demuxing a combined frame, and
   that quadrant-to-camera mapping has to agree with however the
   calibration images were assigned their camera numbers. If unsure, find
   evidence of the actual quadrant layout (a splitting script, a preview
   image, anything that shows which physical quadrant is "camera 1") rather
   than assuming the existing `splitter_order` value is correct.
3. **Same calibration body**, confirmed by diffing the calblock `.txt` files
   byte-for-byte, not just by name.
4. Once confirmed, point every run's YAML at the shared directory with a
   relative path (see the layout above) -- don't duplicate the calibration
   files into every run folder.

## Preparing the YAML files

Per run, starting from a copy of a known-good YAML in this dataset (or
`test_data/test_cavity/parameters_Run1.yaml` as a generic template):

- **`cal_ori`** (`fixp_name`, `img_cal_name`, `img_ori`) and **`ptv.img_cal`**
  -- point at the shared calibration directory, relative to this YAML's own
  location (see path resolution above).
- **`man_ori.nr`** + **`man_ori_coordinates`** -- 4 calibration-body point
  IDs visible in *every* camera (check each camera's
  `calib_matches/cam{N}_matches.txt` -- an ID only appears there if it was
  detected+matched in that camera; don't assume the same 4 IDs are visible
  everywhere just because they're a round set like 1/16/32/46), plus their
  actual per-camera pixel positions -- not copy-pasted from one camera to
  another. This is the easiest calibration mistake to make silently: the
  GUI's manual-orientation click doesn't validate that the point you clicked
  is even the one you meant.
- **`sequence.base_name` + `first`/`last`` -- check this before anything
  else if the sequence step fails.** These must match your *actual raw
  filenames*, not a template default. In splitter mode the loader does
  `Path(base_name % frame)`, so `base_name` needs a real `%06d`-style
  placeholder and `first`/`last` must be the literal numbers embedded in
  your filenames. A copy-pasted template value (e.g. a `100001`-based
  convention left over from a different naming scheme) fails with
  `FileNotFoundError` on the very first frame -- and only when you actually
  run the sequence step; the calibration/detection GUI panels won't catch
  it, since they only touch the calibration images. Verify with:
  ```bash
  ls run1/img | sort | sed -n '1p;$p'   # first and last real filename
  ```
  and match `sequence.first`/`last` to the numbers actually in those
  filenames. `shaking.shaking_first_frame`/`shaking_last_frame` use the same
  numbering -- keep them inside `[sequence.first, sequence.last]`.

### Sample vs. batch

Two YAMLs per run, identical except the frame range:

- `parameters_<name>_sample.yaml` -- ~10 frames, for a fast GUI or CLI check
  before committing to a full run.
- `parameters_<name>_batch.yaml` -- the full range.

Best-strategy order: fix the calibration and frame-naming issues above
*first* (they're silent and easy to miss), then run the sample through the
GUI and eyeball a tracked-particle overlay, *then* run the sample headlessly
with `openptv2-batch` to confirm the CLI path agrees with what the GUI
showed you, and only then run the full batch. A batch run just repeats
whatever the sample did, at scale -- if the sample is wrong, the batch is
the same wrong thing with more frames.

## Checking and visualizing the calibration

Don't rely on reprojection RMS alone -- a bundle adjustment can converge to
a self-consistent but physically wrong pose (the classic cause: a
manual-orientation seed click landed on the wrong calibration-body point,
or a splitter quadrant/camera-identity mismatch per above). Two tools, same
underlying checks (`openptv2.calibration_diagnostics`):

**Headless / CI / scripted** -- `scripts/calibration_diagnostics.py`:
```bash
uv run python scripts/calibration_diagnostics.py --models "current=<shared_calibration>"
```
Prints, per camera: distance to the calibration-body centroid, sight-line
angle (>15deg off is flagged -- a camera aimed away from the target even
with good RMS at its matched points), reprojection RMS + match count when
`calib_matches/` is present. Exits non-zero if anything is flagged, so it
can gate a script or CI step. Compare two models (e.g. before/after
recalibration, or two candidate calibrations) with a second `label=path` in
`--models`.

**Interactive** -- `src/openptv2/gui/visualize_calibration_nb.py` (marimo
notebook; needs the `viz` extra -- `uv run --extra viz` installs it on
demand, no permanent dependency change needed for a one-off look):
```bash
uv run --extra viz marimo -y run src/openptv2/gui/visualize_calibration_nb.py -- \
  --models "current=<shared_calibration>"
```
Orbit-able 3D plot of camera poses + calibration body, with the same
diagnostics rendered as a table below it. Also usable as a Claude Code skill
(`.claude/skills/visualize-calibration/`) -- ask to "visualize the
calibration" / "compare these two calibration models" / "visualize before
and after full calibration" and it resolves the natural-language request
into the right `--models`/`--calblock` args.

## Running

```bash
scripts/run_pipeline_multi.sh <experiment_root> --sample              # quick check, all runs
scripts/run_pipeline_multi.sh <experiment_root> --sample --folder run1  # one run
scripts/run_pipeline_multi.sh <experiment_root>                       # full batch, sequential
scripts/run_pipeline_multi.sh <experiment_root> --parallel            # full batch, concurrent
```

Auto-discovers run folders by looking for `parameters_Run1_<variant>.yaml`
one level under `<experiment_root>` (override the prefix with `--pattern` if
your YAMLs aren't named `parameters_Run1_*`). Each run gets its own
`<run>/pipeline_<variant>.log` -- check it ends with `Batch processing
completed successfully` and that per-frame correspondence counts look
reasonable (hundreds, not near-zero -- near-zero across every frame usually
means the calibration or detection thresholds are off, not just noise).
