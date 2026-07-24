# Report cross-camera ray-convergence (RCM) in the calibration output

> Item #1 of a three-part "make calibration cross-camera-aware" roadmap.
> **Status: #1 done (PR #21), #3 joint-plate BA + distortion shaking done
> (PR #22), #2 iterated tracer self-calibration done (PR #26).** Only the full-#3
> backward/epipolar residual + held-out RCM gate remain (see status block below).
> Do #1 first — you must *measure* RCM before #2/#3 can be shown to drive it down.

## Context — why this matters (measured, not hypothetical)

OpenPTV calibration (`orient` / `full_calibration`) is **per-camera space
resection against a known rigid calibration body**: each camera's `.ori` is
fitted independently to minimize *that camera's* reprojection error, with the
2D↔3D correspondence already known (sortgrid). There is **no cross-camera term**
in the cost — nothing forces camera A's and camera B's rays to actually converge
on the same 3D point. So per-camera reprojection RMS can look excellent while the
four cameras disagree badly *as a set*.

On the real 4-camera splitter dataset `TT13_aorta/calibration` (after the
`interf` glass-tilt fit, mean per-camera reprojection RMS **0.8–1.2 px ≈
0.02 mm**), triangulating the 39 calblock points seen in all four cameras gave a
**ray-convergence miss distance (RCM):**

| percentile | RCM |
|---|---|
| p50 | 0.21 mm |
| p90 | 0.26 mm |
| p95 | 0.26 mm |
| max | 0.31 mm |

That is **~10× larger** than the per-camera reprojection error implies. Cause:
the cameras view the volume at **shallow parallax** (short epipolar lines), which
amplifies a small in-image error into a large along-ray (depth) error. The rays
pass close *sideways* (epipolar 2D residual p95 ≈ 1.45 px, fine) but far
*lengthwise*. **Per-camera RMS hides this entirely.** A user reading only
"0.02 mm RMS" is misled about the true 3D consistency of their rig.

`src/openptv2/calibration_diagnostics.py::compute_diagnostics` already reports
per-camera sight-line angle, reprojection RMS, and centroid-distance spread — but
**no cross-camera RCM**. This plan adds it and surfaces it in the main flow.

## Goal

Surface a **cross-camera RCM summary** wherever calibration quality is reported:

- `skills/openptv-calibrate/scripts/calib.py run` printed output and JSON report.
- `compute_diagnostics` (so the GUI/marimo diagnostics viewer can show it too).

so a user sees *both* per-camera reprojection RMS **and** the cross-camera 3D
miss distance, and is warned when the latter is large.

## Design

### The computation (already validated in a scratch prototype)

RCM is exactly what `openptv2.orientation.multi_cam_point_positions` already
returns as its second output — no new geometry to write:

```python
pos, rcm = multi_cam_point_positions(targets_flat, cpar, cals)
# targets_flat: (n_points, n_cams, 2) FLAT metric coords
# rcm[i]: ray-convergence miss distance (mm) for point i
```

Building `targets_flat` from the per-camera matched calblock points:

1. Collect, per calblock point ID, its **detected pixel** position in each camera
   that matched it. (In `calibrate_dataset` these are `CamResult.ref`/`det`; the
   3D `ref` row identifies the point across cameras. `dump_matches.py` writes the
   same data to `cal/calib_matches/camN_matches.txt`.)
2. Keep points seen in **≥ 2 cameras** (report separately for the all-cameras
   subset, which is the stricter number).
3. For each (point, camera): `pixel_to_metric(px, py, cpar)` →
   `correct_arr_brown_affine(..., cal)` to get **flat** metric coords (the space
   `multi_cam_point_positions` expects; cameras that didn't see the point get the
   `-999` sentinel so they're skipped).
4. `multi_cam_point_positions(targets_flat, cpar, cals)` → `rcm`.

Report `median / p90 / p95 / max` of `rcm` in **mm** (RCM is a 3D distance — do
NOT relabel it "px"; only note pixel size for scale intuition).

### Where the code goes

- **`src/openptv2/autocalibration.py`**: new
  `cross_camera_rcm(results: list[CamResult], cpar) -> dict` returning
  `{n_points, n_common, median, p90, p95, max}` (empty/`None` when < 2 cameras or
  < a handful of common points). Call it at the end of `calibrate_dataset` and
  attach to the returned report (or return alongside `results`).
- **`skills/openptv-calibrate/scripts/calib.py` `cmd_run`**: after the per-camera
  lines, print one summary line, e.g.
  `cross-camera RCM (39 pts in 4 cams): p50=0.21mm p95=0.26mm max=0.31mm`
  and add the same numbers to the JSON report. Add a **warning** when
  `p95 > rcm_flag_mm` (default e.g. 0.1 mm, tunable) —
  `⚠ cross-camera RCM high relative to per-camera RMS; consider a tracer
  self-calibration / dumbbell pass (see roadmap).`
- **`src/openptv2/calibration_diagnostics.py`**: extend `compute_diagnostics`
  (or add a sibling `cross_camera_rcm_from_models`) so the diagnostics dict
  carries the RCM summary, and the marimo viewer
  (`visualize_calibration_setup.py`) can display it. Keep the pixel-size-only
  legacy path working when no matched-point data is available.

### Deliberately NOT in scope for #1

- Changing the fit (that is #2/#3). #1 only *measures and reports*.
- Per-point RCM overlays (nice, but a follow-up; the summary is the MVP).

## Implementation steps

1. Add `cross_camera_rcm(results, cpar)` to `autocalibration.py` (pure function,
   reuses `multi_cam_point_positions`). Handle the degenerate cases (0/1 camera,
   too few common points) by returning `None`.
2. Wire it into `calibrate_dataset` — compute after the per-camera loop from the
   `CamResult` matched points; include in the report structure.
3. Print + JSON in `calib.py cmd_run`; add the high-RCM warning + `--rcm-flag-mm`
   flag (default 0.1).
4. Extend `compute_diagnostics` to include the RCM summary; update the marimo
   viewer to show it.
5. Docs: one line in `skills/openptv-calibrate/SKILL.md` (calib.py run now
   reports cross-camera RCM) and a sentence in `CLAUDE.md`'s calibration section.

## Test plan

- **Unit** (`tests/unit/test_autocalibration.py` or a new
  `test_calibration_rcm.py`):
  - On a known-good fixture (e.g. `test_data/test_cavity` or the splitter
    fixture), `cross_camera_rcm` returns a finite dict with `n_common ≥ 2` and
    `median ≥ 0`.
  - **Monotonicity check (the real regression guard):** perturb one camera's
    `.ori` exterior by a small known offset, recompute RCM, assert it **increases**
    — RCM must respond to cross-camera inconsistency (a per-camera RMS test would
    not catch a camera nudged along its own viewing axis; RCM must).
  - Degenerate: 1 camera → `None`, no crash.
- **Manual verification** on `TT13_aorta/calibration`: `calib.py run` prints
  `p50≈0.21mm p95≈0.26mm` matching the scratch prototype, and the high-RCM
  warning fires (0.26 mm > 0.1 mm default).

## Verification checklist

- `uv run pytest tests/unit/test_calibration_rcm.py -q` green.
- `uv run pytest tests/unit/test_autocalibration.py -q` still green.
- `calib.py run <TT13>` shows both per-camera RMS **and** the RCM summary line +
  warning.
- No Cython rebuild needed (pure-Python additions; `multi_cam_point_positions`
  is already compiled).

## Roadmap — the follow-ups this unblocks

Item #1 gives the **metric**. #2 and #3 are the two ways to *drive it down*, and
OpenPTV already has the scaffolding for both (this is refinement, not a rewrite).

**Implementation status (2026-07-24):**
- **#1 — DONE** (`autocalibration.cross_camera_rcm`, `calib.py run`, PR #21).
- **#3 first cut — DONE** (`autocalibration.joint_plate_bundle_adjust`,
  `calib.py run --joint-ba`, PR #22): frees every camera's exterior + the shared
  3D plate points, holds distortion fixed, gauge-anchored to nominal. Drives the
  synthetic-rig median RCM 0.77 mm → ~0; noise-floored above 1e-3 with 0.3 px
  detection noise.
- **#3 distortion shaking — DONE** (`--shake-distortion`, PR #22): greedy
  one-group-at-a-time distortion (k1k2k3 / p1p2 / scaleshear / glass) gated on
  cross-camera RCM (point 4 below). On TT13 it accepts p1p2+glass but only moves
  RCM ~2% — that rig is at its plate floor.
- **#2 — DONE** (`autocalibration.tracer_self_calibrate`,
  `calib.py tracer-selfcal`, GUI "Tracer self-cal" button, PR #26): coupled joint
  fit over FREE tracer particles (from `res/ptv_is.*`) spanning the real volume,
  gauge-fixed by holding one camera; **iterated** (refine → re-match → repeat,
  accepting a pass only if median tracer RCM improves). On wp1 real data RCM
  95→72 µm / 102.8→68.7 µm (~24–33%) — exactly what the plate BA couldn't do.
  Driven by the `shaking` parameter block (+ `shaking_tol_px` / `shaking_hold_cam`).
- **Full #3 (below), remaining — TODO:** the backward/epipolar residual (make RCM
  itself a residual, points 1–2), a held-out fit/eval split for the RCM gate
  (point 4), and per-point RCM overlays.

### The full-blown bundle adjustment / self-calibration (target design)

The user's framing, which is the plan of record for finishing #2/#3. Today we
calibrate camera-by-camera on **reprojection only**. The full version treats the
calibration body as a cloud of **3D dots** and closes the loop both ways:

1. **Forward + backward residual.** Keep the forward reprojection residual
   (3D→image, `image_coordinates` vs detected metric — already in the #3 first
   cut) **and** add the backward one: back-project each camera's detection to a
   ray (`ray_tracing`), and penalize the **ray-to-point / ray-to-ray miss
   distance** — i.e. make RCM itself (from #1) a residual, not just a report.
   `multi_cam_point_positions` already returns that miss distance; assembling it
   per point is the new residual vector.
2. **Epipolar-distance testing.** For every plate point seen in ≥2 cameras, the
   epipolar residual (point-to-epipolar-line distance, `epi.py`) is a second,
   image-space check that is cheap and stabilizing; include it as a low-weight
   term so the fit stays consistent with the correspondence geometry that
   tracking/stereo-matching will later use.
3. **Full "shaking" of the calibration files** — the self-calibration proper:
   let the camera exteriors, the glass/interface vector, the distortion terms,
   **and** the 3D point cloud all float, minimizing (1)+(2) jointly. The plate
   nominal coords remain a soft prior (gauge fix + trust in the manufactured
   body); real tracer particles (#2) extend the cloud into the depth the plate
   can't reach.
4. **Add parameters ONE AT A TIME (stability).** This is the key discipline the
   user called out and it is correct: introducing all of exterior + glass +
   k1/k2/k3/p1/p2/scale/shear + 3D points at once is ill-conditioned (parameters
   trade off against each other — e.g. cc vs. depth, k-terms vs. principal
   point). Instead **greedily**: start from the per-camera fit, free one new
   parameter group, re-solve, and **accept it only if the #1 RCM (and a held-out
   split) actually improves** — otherwise roll it back. This mirrors the existing
   `CANDIDATE_FLAGS` greedy selection in `autocalibration.py`, but gated on
   cross-camera RCM instead of single-camera RMS. Same idea, cross-camera metric.
5. **Payoff.** Lower RCM ⇒ tighter epipolar bands ⇒ fewer false correspondences
   and better stereo-matching ⇒ longer, cleaner tracks and more accurate 3D
   positions/velocities/accelerations downstream.

**#2 — tracer self-calibration ("shaking"), iterated (Shake-the-Box lineage) —
DONE (PR #26).** `autocalibration.tracer_self_calibrate`: free tracer particles
(from `res/ptv_is.*`) as shared 3D points spanning the real volume, gauge-fixed by
holding one camera; iterated refine → re-match → repeat, accepting a pass only if
median tracer RCM improves (RCM from #1 is the stopping criterion). Scales via
particle subsampling + a sparse Jacobian. Exposed as `calib.py tracer-selfcal`
(`--iters`) and the GUI "Tracer self-cal" button, driven by the `shaking`
parameter block. wp1 real data: RCM ~95→72 µm / 102.8→68.7 µm.

**#3 remaining — extend the joint fits with the backward/epipolar residual terms
(1,2)** — make RCM itself a residual (`ray_tracing` ray-miss) plus the epipolar
term (`epi.py`) — and a **held-out fit/eval split** for the distortion-shaking RCM
gate (4) to prevent overfitting. Scaffolding: `joint_plate_bundle_adjust` +
`tracer_self_calibrate`, `ray_tracing`, `epi.py`, `multi_cam_point_positions`.

## Files touched (item #1)

- `src/openptv2/autocalibration.py` — new `cross_camera_rcm`, wired into
  `calibrate_dataset`.
- `skills/openptv-calibrate/scripts/calib.py` — print + JSON + `--rcm-flag-mm`.
- `src/openptv2/calibration_diagnostics.py` — RCM in the diagnostics dict.
- `skills/openptv-calibrate/scripts/visualize_calibration_setup.py` — show it.
- `tests/unit/test_calibration_rcm.py` — new.
- `skills/openptv-calibrate/SKILL.md`, `CLAUDE.md` — one line each.
