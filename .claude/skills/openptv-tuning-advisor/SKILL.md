---
name: openptv-tuning-advisor
description: >-
  Guide a user through the chain of openptv2 decisions for a new or poorly
  performing 3D-PTV dataset: which tracker to use, how to set the image
  processing (targ_rec, detect_plate, masking), sequence/correspondence
  (criteria) and tracking (track:, two_phase quality rules, confirmation,
  smoothing) parameters, always by MEASURING the data first and changing one
  stage at a time. Use when the user asks "which tracker", "how do I set
  parameters", "tracks are short/noisy/full of ghosts", "what should dvxmax /
  q_seed / confirm_tol / gvthres be", or wants a parameter review of a run.
---

# openptv2 tuning advisor

Walk the user through the stages **in pipeline order**. An error in an early stage
cannot be repaired by a later one, so never tune tracking before the 3D points are
good. Ask for the run folder (it holds `parameters_*.yaml`, `cal/`, `res/run.zarr`)
and the camera frame rate. Talk in plain words: say what the number means and what
changes if it goes up or down. Change **one stage at a time**, re-measure, and keep
the old yaml (work on a scratch copy; never overwrite real data). Always `uv run`.

A change of a yaml value never needs a software release.

## Stage 0: what is the situation?

Ask (or read from the yaml): number of cameras, frame rate, pixel size, particle
density (3D points per frame), whether frames were skipped, whether the calibration
is trusted. If calibration is doubtful, stop and use the calibration docs / skill
`openptv-multiplane-calibrate` and `docs/calibration-rms-vs-rcm.md` first.

## Stage 1: image processing (`targ_rec`, `detect_plate`, `masking`, `unsharp_mask`)

Goal: one blob per particle per camera, no noise blobs, few merged particles.
Measure: blobs per camera per frame (stored targets in `res/run.zarr`,
`targets/cam_N/frame_*`), their size (`npix`) and brightness (`sumg`) distributions,
and the same blobs overlaid on an image (GUI or `docs/browser-gui.md`).

| Symptom | Likely cause | Change |
|---|---|---|
| far more blobs than particles, tiny blobs | grey threshold too low (`gvthres`), noise | raise `gvthres`, raise `nnmin` / `sumg_min` |
| particles missing, matched fraction low | threshold too high, or `nnmin` too big | lower `gvthres`, `nnmin` |
| big merged blobs, two particles as one | `nnmax`, `nxmax`, `nymax` too large or particles overlapping | tighten the maximum sizes; do not hide it by raising thresholds |
| uneven background | light gradient | `unsharp_mask`, background subtraction / `masking` |
| blobs shifted or wobbling | `disco` (discontinuity limit) too loose, or saturation | lower `disco`; reduce exposure for saturated images |

Rule: fix this stage until the blob counts per camera are similar to each other and
stable over frames (a drifting count means lighting changes or a threshold on the
edge).

## Stage 2: sequence / correspondences (`criteria`, `X_lay`, `Zmin_lay`, `Zmax_lay`)

Goal: a high share of 4-camera points and few ghosts. Measure from the run:
3D points per frame, share of points seen by 4 / 3 / 2 cameras, median ray-miss
distance (rcm) per camera count, per-camera reprojection RMS. Run
`scripts/tracking_advice.py` (it prints the 3-camera share and the flagged share).

| Symptom | Likely cause | Change |
|---|---|---|
| few points per frame | `eps0` (epipolar tolerance, mm) too small, volume limits too tight | raise `eps0` a little, check `X_lay`/`Z*_lay` cover the volume |
| many 2- and 3-camera points, many ghosts | `eps0` too large, `cn`/`cnx`/`cny`/`csumg` too loose, dense seeding | lower `eps0`, tighten `cn*`; ghosts are then handled by tracking (stage 4) |
| real particles seen by 3 cameras only | one camera misses the blob (stage 1) | fix stage 1 first |
| rcm large everywhere | calibration | stage 0 |

A ghost fraction of 7–8% of 3D points is typical for 4 cameras in dense data; the
tracking stage compensates for part of it, but only partly.

## Stage 3: which tracker?

Read `docs/trackers.md` (all presets), `docs/tracker-tutorials.md` (decision
guide, section 5) and the registry `src/openptv2/tracking_registry.py`.

| Situation | Choice |
|---|---|
| default: dense or noisy turbulence data, 4 cameras, want correct Lagrangian trajectories | **`two_phase`** (measured best accuracy in the benchmark; supports ghost marks) |
| need the original algorithm's behaviour, compatibility with legacy results | `default` (trackcorr, `full_multipass`) |
| very clean, dense data and speed matters most, ghosts unlikely | `priority_segment_3d` (Fast 3D / 3MA) |
| sparse, clean, reliable detection, not turbulence | `4be` |
| frames skipped / low frame rate | `two_phase` with `blob_gate`; or `hybrid_deltat_3d` |
| cameras unreliable, strong per-camera dropouts | `myptv_2d_tracking` |

State the reason in terms of the data (density, noise, frame rate), and say what the
alternative would cost. If unsure, run the candidates on a short stretch of frames
and compare with the answer-free checks of stage 5.

## Stage 4: tracking parameters

**First run the advisor, then explain its output:**

```bash
uv run python scripts/tracking_advice.py PATH_TO_RUN_FOLDER --fps FPS
```

It measures density, step, kink (the noise), 3-camera share and flagged share and
recommends `q_seed`/`q_young` (ghost rules), `confirm_tol` (automatic or fixed 0.3),
`blob_gate` and the smoothing window, each with its reason. Details:
`docs/tracking_advice.md`, `docs/tracking_parameters_guide.md`,
`docs/tracking_quality.md`.

Parameters you must also decide yourself (the advisor does not):
- `v_max` / `dvxmax…dvzmax` (mm per frame): about 3x the typical step and above the
  largest real step. Too small loses fast particles; too large adds candidates.
- `max_gap` (counts steps; 1 = no gap bridging, 0 finds no links): 2 bridges one
  missing frame.
- `leaf_weight` (0 = 3D only), `bidirectional`: leave the defaults unless a measured
  reason exists.
- If the yaml pins `confirm_tol: 0.3`, the automatic tolerance is off: tell the user.

Facts to keep honest (from the benchmark): the ghost rules gain 0.02 to 0.075 in
velocity error; the automatic tolerance is harmful in dense (>7 neighbours) or
frame-skipped data (the guard falls back to 0.3); cost-term tweaks do not matter
because only 3 to 6% of tracks face a contested decision.

## Stage 5: check the result without ground truth

```bash
uv run python scripts/real_retrack.py NAME 300 q_seed=0.15 q_young=6     # scratch copy
uv run python scripts/realism_metrics.py <scratch>/res/run.zarr --last 300
```

Better means: fewer tracks, longer mean length, about the same number of tracks of at
least 50 frames, lower `never4_share` and `jitter_z_p90`. **Do not use `jump_share`
to judge the confirmation tolerance** (it counts depth-noise outliers). For a real
accuracy number, build a synthetic copy with known truth (`scripts/synth_bench.py`,
see `docs/tracking_parameters_guide.md`, section 7) and never tune on real data
alone. A claim of improvement needs numbers before and after.

## Stage 6: output smoothing

`trajectories.smoothing_window` about 4 ms of frames (21 at 5000 fps);
`trim_doubtful` 0.3 (cuts doubtful end points) when accuracy matters more than the
number of points. Longer windows blur real accelerations.

## How to answer

1. State what you measured and what it means, in plain language.
2. Give the recommended value, the reason and the expected effect, and what to
   do if it does not help.
3. Offer the yaml lines; apply them only to a scratch copy unless told otherwise.
4. Say clearly when a recommendation is untested for this kind of data (one flow
   field and one camera geometry stand behind the benchmark numbers).
