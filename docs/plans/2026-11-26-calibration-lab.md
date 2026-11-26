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

### Full length (wp2, all 1072 frames, two_phase)

Same behaviour as the 300-frame ranges: report `marginal` (0.13 px), self-calibration accepted (held-out jump 23.5 -> 1.3 µm,
`consistent` afterwards), tracks of 50 frames or longer 5723 -> 5884 with align (+2.8 %) and **6283 with selfcal (+9.8 %)**, points in
tracks of 20 frames or longer +2.2 % / **+8.7 %**. The volume change did not shrink with more data: stretches 0.9958, 1.0042,
1.0110 (mean +0.37 %), shift 2.5 mm.

### Cavity geometry (`test_data/test_cavity`, cameras on opposite sides, real images, 5 frames, ~430 quads per frame)

Run with `bench/calibration_lab/cavity_selfcal.py` (detect and match all five frames with the production loop in a scratch copy).
Result: **self-calibration is possible but gives nothing measurable here**, and testing it exposed three defects, now fixed:

* The first run said INCONSISTENT with 198 µm jumps and the fit made things worse. The stored quads contain many wrong matches
  (reprojection residual median 0.78 px, 99th percentile 17 px, maximum 43 px). Wrong matches now are excluded from every number
  (`inlier_mask`: rms residual above max(1.5 px, 3 x median)); 12 % of the points were left out, the real systematic jump is about
  30 µm, and wp1 / wp2 / syn1 reports are unchanged (0 % left out). Test: `test_wrong_matches_do_not_spoil_the_report`.
* A verdict from one frame and 434 points was printed with confidence. The report now has an `inconclusive` verdict when the
  offsets are within 3 standard errors of the sample noise, shows the standard error, and prints "n/a" instead of "0 %" for the
  reproducibility when there is no held-out half.
* On the perfect syn1 control the self-calibration was ACCEPTED although the cameras were already consistent. It now declines
  (test `test_self_calibration_declines_when_there_is_nothing_to_gain`).

With those fixed: report `marginal` (0.28 px, jumps 1-27 µm, smooth field **not reproducible**, 0-4 %: there is no consistent camera
error to fit); self-calibration accepted (held-out jump 30 -> 12 µm) with stretches 0.9952 / 0.9989 / 0.9999 (-0.20 % mean, 0.26 mm
shift); free focal length, glass thickness, fluid index and glass distance change nothing (fitted d 5.98 mm for 6.0, n3 1.335 for 1.33;
looser priors: scale -0.5 %). Tracking over the five frames, good trajectories (>= 4 frames, quadratic-fit rms < 0.2 mm):

| tracker | shipped cameras | self-calibrated | |
|---|---|---|---|
| two_phase | 189 | 191 | +1 % |
| trackcorr | 408 | 416 | +2 % |
| fast_3d | 576 | 545 | -5 % |

That is noise. What limits `test_cavity` is the per-blob noise (residuals 0.6-0.75 px against 0.2 px on wp1 / wp2) and the ghosts in
the matches, not camera consistency; its calibration was already refined once by the particle bundle adjustment
(`scripts/particle_bundle_adjust.py`). Recommendation: do not self-calibrate `test_cavity`; more frames would let the report separate
noise from structure (it needs a held-out half of frames to judge a field).

### lv_multi (real, one side, 20 frames each): self-calibration made tracking worse, and why

Run with `bench/calibration_lab/redetect_selfcal.py` (the stores hold only trajectories, so all 20 frames are detected from the
multiplexed images and matched with the production loop). With the first version of the fit both sub-runs looked fine by the jump
criterion (held-out 24.2 -> 0.5 µm and 18.8 -> 1.8 µm) but good trajectories (>= 4 frames, quadratic-fit rms < 0.2 mm) fell:
two_phase 1430 -> 1292 (-9.7 %) and 1096 -> 1015 (-7.4 %), trackcorr and fast_3d -1.5 ... -3 %, and wp5 lost 10 % of its 4-camera
matches. A calibration-target anchor fixed the volume change but not the loss, so it was not the gauge. **Cause found:** the priors on
the lens terms.

1. **Which parameters.** Refitting with one group of parameters at a time and matching all 20 frames again (`lv_why.py`), 4-camera
   matches against the old cameras: exterior only 103.7 %, + principal point 104.8 %, **+ decentering (p1, p2) 108.5 %**, **+ radial
   (k1, k2) 77.3 %**, + principal point + radial 79.2 %, everything 90.9 % (wp2: 98.6 % with k1, k2, 112 % with p1, p2).
2. **Where the matches are lost** (`lv_lost.py`, lv5, everything free): 65 % of the matches lie within 3 mm of the sensor centre and only
   2.6 % beyond 5 mm. Of the old matches 8.6 % are lost in the 0-3 mm ring, 35 % in 3-5 mm and **97 % beyond 5 mm**; 85 % of the newly
   gained ones are in the centre. Exterior only: 4.5 % lost, 8.2 % gained. The fit improves the centre where it has data and wrecks
   the periphery where it has none.
3. **Why.** The distortion acts on the image-plane radius in mm: k1 ~ r^3, k2 ~ r^5, p ~ r^2. On this sensor (corner radius 9.8 mm) a
   change of k1 by 1e-4 moves the corner by about 8 px and a change of k2 by 1e-5 by about 75 px. The priors on k and p were fixed numbers
   (0.02 and 0.002), thousands of pixels at the edge: no prior at all. The data fix k1, k2 near the centre and the rest is extrapolation.
4. **Fix.** `joint_self_calibrate(edge_px=...)` sets every lens term's prior as the image shift (px, 1 sigma) it may cause at the sensor
   corner (sigma = edge_px * pixel / r_corner^n); `self_calibrate_from_store` uses 1 px. Matches against the old cameras, loose priors
   -> `edge_px` 1: lv5 90.9 % -> **108.5 %**, lv4 99.7 % -> **108.4 %**, wp2 105.9 % -> 111.7 %, wp1 104.9 % -> 109.8 %; flat from 0.25 to 2 px,
   worse above 5 px (lv4: 105.5 % at 5, 100.3 % at 20).
5. **End to end** (good trajectories before -> after, two_phase / trackcorr / fast_3d): lv5 1430 -> 1450 (+1.4 %) / 2033 -> 2142 (+5.4 %) /
   1495 -> 1568 (+4.9 %), 4-camera matches per frame 642 -> 687; lv4 1096 -> 1090 (-0.5 %) / 1646 -> 1730 (+5.1 %) / 1250 -> 1297 (+3.8 %),
   515 -> 548. The volume change is the same as before (-0.94 % / -1.09 %, 5-6 mm): a separate, still open, issue. On the real
   recordings the benefit grew: wp2 300 frames tracks >= 50 frames +11.7 % (was +9.8 %), points in tracks >= 20 frames +10.5 % (was +7.8 %);
   wp1 +15.2 % (was +9.5 %) and +10.9 % (was +6.8 %).

The matching-yield acceptance test stays (it is how this was noticed), but it no longer has to decline anything on these data.

### Synthetic cavity with truth (cameras on both sides): `docs/self-calibration-cavity-synthetic.md`

`cavity_rig()` (independent ray tracer, cameras at -z and +z, glass vectors of both signs) and `bench/calibration_lab/cavity_synth_lab.py`:
13 experiments, 5 invariance moves, noise and frame-count variations, and a reproduction of the lv_multi mechanism. Findings: the fit
works only if the old calibration still matches (image error below about 1.5-3 px); consistency is always restored but the error against
the truth depends on the reference camera (a held, drifted camera made it worse, 1.03 -> 1.60 px); glass distance / thickness are
not recoverable; the volume keeps its shape to 0.35 %; the half-turn invariance holds to three digits; the matcher cannot handle a rig
turned 90 degrees; with fit data concentrated at the axis the old loose lens priors destroy the outermost sensor ring (recall 51 % against
100 % for the true cameras) and the physical priors do not. `hold_cam="auto"` (the camera with the smallest residual) is the default
reference.

### Full length (wp1, all 1072 frames, two_phase)

Report marginal (0.11 px), selfcal held-out jump 19.4 -> 1.1 µm, stretches 0.9950 / 1.0036 / 1.0101 (+0.29 %, 2.1 mm). Tracks of 50 frames
or longer 5961 -> 6138 with align (+3.0 %) and 6513 with selfcal (+9.3 %); points in tracks of 20 frames or longer +2.1 % / +8.2 %.

### Limitation: the gauge
Tracers define consistency, not scale, shape or frame. On every real recording the fit moves the volume by +0.25...+0.37 % in
mean scale (principal stretches up to 0.995 / 1.011: not uniform) and about 2.2 mm in position (cavity: -0.20 %, 0.26 mm), and tighter priors on the camera positions (sigma 0.3, 1.0, 3.0 times the default) leave it at
+0.27...+0.31 %: it is driven by the data, either a real scale error of the old calibration or the scale degeneracy of tracers
(the glass and fluid index couple to it). Velocity is shift-invariant but a 0.3 % scale change is a 0.3 % velocity bias. The
result prints the scale and warns; check it against a known length (dumbbell, plate) or pass an anchor
(`KnownBody` / `Dumbbell` in `openptv2.selfcal`) before using absolute velocities from new cameras. Written up in
`docs/self-calibration-scale-bias.md`; from a perfect start on rendered syn1 the fit introduced +0.04...0.10 %. Soft gauge rows over all
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
2. Layer 2 can help (+9-14 % long tracks on wp1 / wp2) but is **not** a universal fix (lv_multi: -7 to -10 %). The report says
   whether there is a camera inconsistency, not whether removing it improves tracking: match again and compare your own tracking.
   The acceptance requires a held-out jump reduction and no loss of matching yield.
3. Layer 3 only as an explicit stopgap. `align_4cam` as a silent default in the loader is not needed.
4. The multimedia parameters are not where quality is lost for this rig class; revisit only with a rig whose plate does not
   cover the volume or whose fluid index is uncertain by more than 0.04.

## Open

* A dumbbell or plate anchor measured on a real recording (does it remove the deformation?); the full wp1 length.
* Rigs with cameras on opposite sides: tested on `test_cavity` (5 frames, above) but not with a truth-known rig; the ladder is same-side only.
* Whether the volume deformation (not the tracking loss) is a calibration correction or a bias: needs an independent length on a real recording.
* A GUI button for the check and for `selfcal`.
