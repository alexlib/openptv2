# Report cross-camera ray-convergence (RCM) in the calibration output

> Item #1 of a three-part "make calibration cross-camera-aware" roadmap.
> Items #2 (tracer self-calibration / "shaking", iterated) and #3 (joint plate
> bundle adjustment) are scoped at the end as follow-ups. Do #1 first — you must
> *measure* RCM before #2/#3 can be shown to drive it down.

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
OpenPTV already has the scaffolding for both (this is refinement, not a rewrite):

- **#2 — tracer self-calibration ("shaking"), iterated (Shake-the-Box lineage).**
  plate-calibrate → track → self-calibrate on triangulation disparity/RCM at real
  depth → re-track → repeat. This is the highest-value fix for shallow-parallax
  rigs like this one, because it uses real particles spanning the actual volume
  the plate can't reach. Formalize the existing `calib.py snapshot-refine`
  (`cmd_snapshot_refine`) into an iterated, first-class stage. Reuse
  `weighted_dumbbell_precision` / `point_position`'s convergence measure as the
  objective. Uses the RCM metric from #1 as its convergence/stopping criterion.
- **#3 — joint plate bundle adjustment.** Optimize all four cameras' exterior
  (+ optionally `interf` glass) *simultaneously* against the shared calblock,
  minimizing total reprojection with the rigid-body constraint — coupling the
  cameras, unlike per-camera resection. Scaffolding exists:
  `src/openptv2/gui/ptv_calibration.py::full_scipy_calibration`,
  `dumbbell_ba_residuals`, and `dumbbell_target_residuals` already wrap
  `scipy.optimize`; this is a new residual assembling all cameras' reprojections
  into one vector, not new geometry. Validate by the #1 RCM metric
  (joint BA should lower RCM vs the per-camera fit at equal reprojection RMS).

## Files touched (item #1)

- `src/openptv2/autocalibration.py` — new `cross_camera_rcm`, wired into
  `calibrate_dataset`.
- `skills/openptv-calibrate/scripts/calib.py` — print + JSON + `--rcm-flag-mm`.
- `src/openptv2/calibration_diagnostics.py` — RCM in the diagnostics dict.
- `skills/openptv-calibrate/scripts/visualize_calibration_setup.py` — show it.
- `tests/unit/test_calibration_rcm.py` — new.
- `skills/openptv-calibrate/SKILL.md`, `CLAUDE.md` — one line each.
