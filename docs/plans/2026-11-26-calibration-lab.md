# Calibration lab: verify the imaging assumptions one by one, then use what they teach

Date 2026-11-26. Status: layers 1-3 built and tested on rendered syn1 (3 calibrations x 3 trackers) and on real wp1 / wp2 (two ranges).

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

## Measured end-to-end

**Scoring note.** `vel_err_rel` is computed on each run's own matched points, so it is not comparable between variants (a
perfect calibration shows ±15 % differences from set composition alone). The tables use the velocity error on the **common set**
(the points the first run tracked) and the position error.

### Rendered syn1 (truth, 400 frames), 3 calibrations x 3 trackers

Velocity error on the common set (position error in brackets); "off" = the store as matched, "align" = `align_partial_points`,
"selfcal" = `self_calibrate_from_store` then matched again:

| calibration | tracker | off | align | selfcal |
|---|---|---|---|---|
| real syn1 error (LOCO 0.305 px) | two_phase | (0.0785) | 0.120 (0.0693) | **0.107** (0.0711) |
| | fast_3d | 0.311 (0.0815) | 0.277 (0.0724) | **0.229** (0.0746) |
| | trackcorr (legacy) | (0.0806) | 0.191 (0.0712) | **0.161** (0.0738) |
| error doubled (LOCO 0.523 px) | two_phase | (0.0919) | 0.144 (0.0748) | **0.104** (0.0730) |
| | fast_3d | 0.405 (0.0955) | 0.423 (0.0787) | **0.234** (0.0766) |
| | trackcorr (legacy) | (0.0768) | 0.244 (0.0768) | **0.143** (0.0758) |
| perfect (LOCO 0.143 px) | two_phase | (0.0670) | 0.112 (0.0669) | 0.111 (0.0705) |
| | fast_3d | 0.212 (0.0710) | 0.207 (0.0711) | 0.212 (0.0744) |
| | trackcorr (legacy) | (0.0702) | 0.180 (0.0703) | 0.171 (0.0736) |

* Perfect calibration: nothing to gain, nothing lost (self-calibration declines in the CLI; forced here to show it is harmless on
  velocity). Position error is 4-5 % worse with self-calibration there because the volume shifts (see the gauge note).
* Self-calibration is at least as good as alignment in every case and the gap grows with the calibration error; alignment
  made `fast_3d` 5 % worse with the doubled error, and changed nothing for `trackcorr` there.
* Position error: alignment and self-calibration are about equal.
* The legacy `trackcorr` rows are valid only since the `tnr` fix (below).

### Real recordings (no truth), two_phase, 300 / 300 / 201 frames

| recording | report (worst jump) | tracks >= 50 frames | points in tracks >= 20 | held-out jump after selfcal | scale / shift of the volume |
|---|---|---|---|---|---|
| wp2 frames 1-300 | marginal, 0.14 px | align +3.3 %, **selfcal +9.8 %** | +2.7 % / **+7.8 %** | 22.9 -> 0.3 um | +0.31 % / 2.2 mm |
| wp1 frames 1-300 | marginal, 0.11 px | align 0 %, **selfcal +9.5 %** | 0 % / **+6.8 %** | 18.7 -> 0.4 um | +0.32 % / 2.3 mm |
| wp2 frames 700-900 | marginal, 0.15 px | align +2.7 %, **selfcal +13.7 %** | +1.7 % / **+8.3 %** | 24.6 -> 1.0 um | +0.25 % / 2.2 mm |

After self-calibration the report reads `consistent` on all three (0.02-0.03 px). Along tracks the jump at camera changes was
already small on wp1 (2-4 um), alignment removes it where it exists (wp2: 10-18 -> 2-5 um), and self-calibration leaves 1-10 um;
the gain in long tracks does not come from that jump alone.

### Limitation: the gauge
Tracers define consistency, not scale or frame. On all three recordings the fit moves the volume by +0.25...+0.32 % in scale and
about 2.2 mm in position, and tighter priors on the camera positions (sigma 0.3, 1.0, 3.0 times the default) leave it at
+0.27...+0.31 %: it is driven by the data, either a real scale error of the old calibration or the scale degeneracy of tracers
(the glass and fluid index couple to it). Velocity is shift-invariant but a 0.3 % scale change is a 0.3 % velocity bias. The
result prints the scale and warns; check it against a known length (dumbbell, plate) or pass an anchor
(`KnownBody` / `Dumbbell` in `openptv2.selfcal`) before using absolute velocities from new cameras. Soft gauge rows over all
points were tried and dropped (they make the sparse Jacobian dense, a 10x slowdown).

## A bug found on the way: re-matching left stale target pointers
`match_correspondences_batch_parallel(write_to_store=True)` wrote the new correspondences but not the per-target `tnr` pointers
(target -> 3D point) that the sequence loop writes. The legacy `trackcorr` follows those pointers: after any re-matching (new
calibration, new settings) it found no links (448 504 tracks for 449 168 points). Only 52 of 13 090 stored targets still pointed at
their point. `sync_target_tnr` now updates them with the correspondences (tests in `tests/unit/test_target_tnr_sync.py`);
re-matched self-calibrated syn1 gives 7 041 tracks of mean length 10.1 on 60 frames instead of 70 910 of length 1. This
also explains the unexplained `trackcorr` yield of 0.0001 in the earlier #51 controls.

## Decisions this supports

1. Layer 1 is cheap and informative everywhere: ship it, run it in `self_check` / GUI later.
2. Layer 2 is the fix when the report says `marginal` or `inconsistent`; it is validated on held-out frames and declines when it
   does not help.
3. Layer 3 only as an explicit stopgap. `align_4cam` as a silent default in the loader is not needed.
4. The multimedia parameters are not where quality is lost for this rig class; revisit only with a rig whose plate does not
   cover the volume or whose fluid index is uncertain by more than 0.04.

## Open

* lv_multi; a dumbbell or plate anchor for the scale; a longer wp1/wp2 range (all 1072 frames).
* Rigs with cameras on opposite sides (cavity): the ladder was run on the same-side syn1 geometry only.
* A GUI button for the check and for `selfcal`.
