# Calibration lab: verify the imaging assumptions one by one, then use what they teach

Date 2026-11-26. Status: layers 1-3 built, lab steps S0-S5 measured, end-to-end numbers below. Decision on #51 follows from it.

## Why

PR #51 (`align_4cam`) removed a symptom (positions jump when a track changes camera subset) silently and by default, for a
cause nobody had located. The study of that PR showed that the benefit grows with calibration error and vanishes for a perfect
calibration. So the useful question is not "should the symptom be patched" but **"is the calibration consistent, where does the
inconsistency come from, and can it be removed at the source?"** This plan turns the machinery into a measurement and a fix,
and builds a truth-known test bed so that each assumption of the imaging model is tested separately.

## What was built

| piece | where |
|---|---|
| Independent ray-traced truth rig (Snell's law in 3D, glass tilt / wedge / thickness / index) | `src/openptv2/synthetic_rig.py` |
| Layer 1: consistency report with verdict | `openptv2.calibration_consistency.consistency_report`, CLI `check` |
| Layer 2: re-calibration from the run's tracers, validated on held-out frames, never overwrites | `self_calibrate_from_store`, `write_calibration`, CLI `selfcal` |
| Layer 3: explicit, guarded, marked partial-point alignment | `align_partial_points`, CLI `align` |
| Self-calibration: coarse-to-fine robust loss (big errors converge), optional free glass thickness / fluid index / glass distance | `openptv2.selfcal.joint_self_calibrate` (`anneal`, `free_multimedia`, `free_glass`) |
| Lab scripts | `bench/calibration_lab/{ladder,selfcal_vs_align,real_selfcal,detection_bias}.py` |
| Tests | `tests/unit/test_synthetic_rig.py`, `tests/unit/test_calibration_consistency.py` |
| User page | `docs/calibration-consistency.md` |

## The ladder (rig = syn1 cameras; 845 plate dots on 5 planes; 3000 tracers; 0.1 px centroid noise; scores on 2000 held-out points)

| step | assumption tested | result |
|---|---|---|
| S0 | the library's radial-shift multimedia equals an exact 3D ray trace for flat parallel layers | max difference 0.0006 px (cavity rig). Verified; the tracer is a valid truth |
| S1 | with the model right, plate calibration of one camera recovers it | position error 3.3 µm, jumps 0.1-1.8 µm, LOCO 0.011 px (the floor) |
| S2 | a wrong glass thickness / fluid index / glass offset / tilt / wedge hurts a plate calibration | no: each is absorbed by the other camera parameters inside the plate volume: d+1 mm 3.6 µm, n3+0.01 7.5 µm, glass +1 mm 3.6 µm, 0.3° tilt 4.2 µm, 0.3° wedge 5.3 µm; jumps ≤ 3 µm |
| S2b | same with a plate spanning only 25 % of the depth, mismatch ×4 (d+4 mm, n3+0.04, 1.2° wedge) | still small: d 4.6 µm (glass nearly index-matched to the fluid in this rig), n3 26 µm, wedge 36 µm; jumps ≤ 3 µm. These errors are smooth: velocity impact negligible |
| S3 | self-calibration with free glass thickness, fluid index and glass distance closes the remainder | no: n3 fitted 1.416 for a true 1.45 and error unchanged: not identifiable at the level where it matters (and the effect is below 40 µm) |
| S4 | cameras moved after the plate (drift 0.09 mm / 0.3 mrad ×1 and ×3) | plate calibration leaves 1.14 mm / 3.4 mm position error, LOCO 1.8 / 5.5 px, jumps 130-600 / 400-1800 µm; **tracer self-calibration: LOCO 0.008 px, jumps < 2 µm** (3 seeds); absolute position moves 0.2-0.8 mm (the gauge: tracers cannot define it) |
| S5 | the blob detector returns the true sub-pixel centre | bias ≤ 0.002 px, pixel locking ≤ 0.027 px (faint small spots), centre convention (+0.5) confirmed |

Reading: inside the volume covered by a plate calibration the multimedia simplifications cost tens of µm at most and are not
identifiable from tracers. The large, removable error is **inconsistency between cameras after the plate was recorded**.

## End-to-end, rendered syn1 (truth, 400 frames), tracker two_phase

See the table appended by `bench/calibration_lab/selfcal_vs_align.py` runs (below, section "Measured").

## Decisions this supports

1. Layer 1 is cheap and informative everywhere: ship it, run it in `self_check` / GUI later.
2. Layer 2 is the fix when the report says `marginal` or `inconsistent`; it is validated on held-out frames and declines when it
   does not help.
3. Layer 3 only as an explicit stopgap. `align_4cam` as a silent default in the loader is not needed.
4. The multimedia parameters are not where quality is lost for this rig class; revisit only with a rig whose plate does not
   cover the volume or whose fluid index is uncertain by more than 0.04.

## Open

* Real datasets other than wp2 (wp1, lv_multi) and a longer wp2 range; a dumbbell anchor for the scale.
* Rigs with cameras on opposite sides (cavity): the ladder was run on the same-side syn1 geometry only.
* A GUI button for the check and for `selfcal`.
