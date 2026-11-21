# Holistic 3D-PTV: review and next phase (2026-11-21)

Continues `2026-11-20-holistic-ptv.md` (design, stages, results) and `2026-11-19-joint-tracker.md` (lab record).

## 1. Status against the goal
| target | result |
|---|---|
| velocity error against `two_phase` (L2 image pipeline, default density) | 0.140 -> 0.106 (-24 %); x4: -12 % |
| yield | 0.69 -> 0.74 (ceiling about 0.85) |
| long-track completeness | 0.54 -> 0.65-0.67 |
| real wp2 (300 frames) | mean track length 20.7 -> 30.4, jump share 1.44 -> 0.46 %, tracks of 50+ frames 1,533 -> 1,827 |
| real lv-multi wp4/wp5 (20 frames) | jitter about -20 %, wp5 jump share 2.79 -> 1.78 %, mean length +6-14 % |
| time | 0.58 s/frame (186 -> 116 s per 200 frames after the first speed pass); goal <= 3x sequential |
Product: plugin `holistic`, registry and GUI list, docs page, `unified` switch (`track.holistic: true`), final tracks
written to `trajectories/` (openptv-cloud `03_results` notebook), `scripts/export_trajectories.py`. PR #44.

Weak spot: leave-one-camera-out error does not improve (wp2 0.343 px raw against 0.343 corrected; lv-multi slightly worse
in the tail). The gain on real data comes from longer and cleaner linking and from births, not from better geometry.
The correction of merged blobs has no real-data support yet.

## 2. Why we are circling
- Every gain is 10-25 % on the same two benchmark metrics (velocity error, yield) on the same synthetic cases.
- The yield breakdown (`scripts/joint_yield_breakdown.py`) shows what is left, and blob attributes cannot reach it:
  14.6 % of particles have no blob in >= 2 cameras; at x4 the 2-clean group (21 % of particles) is matched 34 % of the
  time, the 1-/0-clean groups (53 %) are almost never recovered; extension from existing tracks does not help there
  because tracks do not start.
- The condition set on 2026-11-19 for opening the pixel option (result stays well above the ceiling: 0.106 against
  0.063) is met.

## 3. Next phase (one by one; each has a gate, a failed gate stops that step)
**N1. Score what is used: Eulerian statistics on full wp2.** Run the openptv-cloud analyze phase on the holistic output
and on `two_phase`, compare flow rate, VSS and eps against the MATLAB reference (`comparison_vs_legacy.json`).
Gate: holistic is closer to the reference on at least two of the three; otherwise find which part (links, births,
jitter) moves them.

**N2. Neighbour flow prior.** Predict a particle's next position (and a birth's start) from a local velocity field built
from tracked neighbours (relaxation-style), not only from its own history. Targets the track-start problem at x4 and
the births. Gate: x4 yield +5 points at unchanged velocity error and ghost share; default density not worse.

**N3. Pixel patches, only where needed.** Merged and missing blobs (about 10 % of observations): joint two-Gaussian fit
in a merged blob for both centres; fit at a track-predicted position with a lower threshold for missing blobs.
Gate: centre error of split particles below 0.2 px on the render (`syn1`), yield +4 points, leave-one-camera-out error
of corrected observations clearly below raw on real wp2.

**N4. Self-calibration from clean tracks.** Depth jitter 0.039 mm against 0.008 mm sideways (wp2); leave-one-camera-out
floor about 0.35 px. Estimate per-camera distortion corrections from long holistic tracks (volume self-calibration).
Gate: depth jitter -20 %, leave-one-camera-out median down, no change in the particle count.

**N5. Learned acceptance rules.** Small classifier over the cost components already computed (brightness ratio, ray
miss, track length, prediction residual, leave-one-camera-out residual) replacing hand thresholds (`min_resid`,
`partner_min_len`, confirmation tolerance). Labels: truth on the benchmark, leave-one-camera-out on real data.

**N6. A real-data score without truth beyond leave-one-camera-out.** Hold out every 5th frame, predict its blob centres
from the neighbouring frames, report the error. Use next to N1.

**Housekeeping.** Update the stale status table of the first plan; rebase and rerun the benchmark when the other
agent's `two_phase` changes land (holistic calls `two_phase`; `two_phase_tracking.py` has my noise-aware confirmation
edits, expect a conflict); speed: per-group loops, analytic Jacobian, Cython.

## 4. Results
(appended per step below)

### N1 result: Eulerian statistics on full wp1 + wp2 (1,072 frames, 3 rounds) - gate NOT met
Tracks (truth-free realism metrics, per run, wp1 / wp2): mean length 24.3 / 24.2 -> 34.9 / 34.3; jump share 1.13 / 1.16
-> 0.32 / 0.37 %; sideways jitter 0.0077 / 0.0076 -> 0.0051 / 0.0052 mm (-33 %), depth 0.0365 / 0.0363 -> 0.0267 /
0.0272 (-26 %); tracks of 50+ frames 5,961 / 5,723 -> 5,825 / 5,583 (about -2 %; fewer but longer tracks).

Downstream (identical post-processing for both: reconnect, repair, Savitzky-Golay 21/2, then the openptv-analysis
Eulerian stages; the `two_phase` baseline reproduces the existing `phaseRegionMetrics.csv` exactly), phases 5-10, global:
| metric | holistic vs `two_phase` | `two_phase` vs legacy (MATLAB) | holistic vs legacy |
|---|---|---|---|
| mean velocity | +0.0 % (-0.3..+0.4) | -11.1 % | -11.1 % |
| MKE | +0.1 % | -26.1 % | -26.0 % |
| VSS | -0.2 % | -23.9 % | -24.0 % |
| eps | -0.2 % (-1.7..+1.4) | -45.5 % | -45.6 % |
Reading: (1) the phase-averaged statistics do not see the tracker improvement at all, to within 1-2 %: a voxel average
over thousands of points, after reconnect/repair/smoothing, hides what the realism metrics show; (2) the systematic
gap to the legacy table (-11 % velocity, -26 % MKE, -45 % eps) is the same for both trackers, so it does not come
from linking: detection, calibration, mask/binning definitions, or legacy noise inflating VSS/eps (legacy eps is higher,
which is what noise does). The jet-region numbers differ by 10-100x between us and legacy (mask proxy 3,254 against
5,746 hand-mask voxels, documented earlier) and say nothing about trackers.
Consequences for the order: the tracker-side steps (N2, N3) improve tracks the downstream statistics cannot reward;
the next steps that can move these numbers are on the geometry side: N4 (self-calibration, depth jitter) and finding
the cause of the common gap to legacy (same frames: compare detections/positions of the legacy and the new tracks,
then binning on both).
Tooling: `scripts/eulerian_compare.py` (post-processing in the openptv-cloud tool environment, flowtracks 1.6.1; analysis
stages from openptv-analysis with its flowtracks zarr reader disabled, `postprocess` stage skipped, no `legacy_root`).

### N2 result: neighbour flow prior, feasibility on x4 (blob table, frame 30) - gate not met, stopped
`scripts/joint_flow_prior.py`: full pool of 2-camera epipolar pairs (105,214 rows, 16 % of them real), each moved by the
displacement of its 10 nearest tracked neighbours (final holistic tracks, within 3 mm), projected into all cameras at
t+1 and t-1, confirmed where a blob lies within 1.5 px.
- The prior itself is good: error against the true displacement median 0.026 mm, p90 0.066, p99 0.133 (true step 0.093 mm).
- Separation is not: accepting rows with >= 3 cameras confirmed in both directions gives 6,232 rows, **72.5 % real**,
  and 349 particles that the holistic output does not hold (**8.6 % of the 4,073 missing**); >= 2 cameras: 49.9 % real,
  526 new. A 28 % ghost share on the added points breaks the gate (ghost share unchanged).
- By clean cameras (missing particles with blobs in >= 2 cameras): 2+ clean 833, all reachable by a pair, 25 % confirmed
  (>= 2 cameras, both directions); 1 clean 1,304, 17 %; 0 clean 1,357, 15 %.
Why: (1) a ghost pair of two real blobs predicts into its own blobs in the forming cameras (smooth flow), so only the
other cameras can discriminate, and there the blob must be clean at t-1, t and t+1: merging changes frame to frame
(clean at t is not clean at t+1), so a true particle often fails a pixel-domain test; (2) the particles that are missing
are exactly the merged ones. A pixel-domain confirmation cannot be right without the blob model (predicted blob centre =
brightness-weighted mean of the members), i.e. it belongs inside the births acceptance, not next to it.
Possible reuse: the prior as one more term in the births acceptance (cost already includes brightness and path); not a
standalone stage. Not pursued; next is N4.

### N4 result, part 1: is the leave-one-camera-out floor systematic? (`scripts/loco_map.py`, wp2 holistic, 1072 frames)
208,818 clean 4-camera points (no shared blob). Residual rms per camera 0.30-0.31 px. A cubic polynomial in image position
fitted on odd frames explains 22-41 % of the variance on the even frames; adding depth terms 45-54 %: rms 0.30 -> 0.20-0.22 px
on held-out frames. So about half of the floor is a smooth, repeatable (calibration-type) error, not detection noise.
Gate (depth jitter -20 %) **not met**: correcting every observation with the fitted fields and re-triangulating leaves the track
metrics slightly worse (depth jitter 0.0393 -> 0.0410, jump share 4.8 -> 6.0 %; metrics are from re-triangulated raw
observations, not the refined holistic positions). Reasons: a smooth spatial error moves with a particle and barely
changes frame-to-frame jitter; and the leave-one-camera-out residual of camera c contains the errors of the three others,
so correcting all four cameras by their own residual field double-counts. A real volume self-calibration would solve for all
cameras together (bundle adjustment over the long tracks); its benefit is accuracy (spatial bias of positions and velocity),
which the jitter metrics cannot show. Not built.

### N4 result, part 2: the gap to the legacy table is mostly a pipeline asymmetry
Legacy tracks (MATLAB, imported in `CompleteTest-e2e-legacy`): 4.8 % of their points move faster than 2.5 m/s (p99 5.3 m/s,
max 18 m/s; ours p99 1.6, capped at 2.5 by the post-processing). The legacy table was computed without that cap.
Same analysis (phases 5-10, global), the legacy tracks with the same 2.5 m/s cap applied:
| metric | legacy as published | legacy, same cap | `two_phase` | holistic |
|---|---|---|---|---|
| mean velocity | 0.541 | 0.496 | 0.488 | 0.488 |
| MKE | 259.0 | 213.4 | 209.1 | 209.2 |
| VSS | 2.048 | 1.765 | 1.572 | 1.569 |
| eps | 469.5 | 328.5 | 265.2 | 264.7 |
The cap removes most of the difference (velocity -11 % -> -1.6 %, MKE -26 % -> -2 %, VSS -24 % -> -11 %, eps -45 % -> -19 %).
What remains, VSS +12 % and eps +24 % in legacy, is the sign noise gives (gradient statistics grow with position noise).
Conclusion: the apparent gap says nothing against the new trackers; compare with identical filtering. Both trackers agree to 0.3 %.

### N4 result, part 3: joint self-calibration (`src/openptv2/selfcal.py`, `scripts/selfcal_real.py`, `scripts/selfcal_synthetic.py`)
Bundle adjustment over 15,000 clean 4-camera tracers (odd frames), camera parameters and 3D positions free, camera 1 exterior
held, soft priors; scored by leave-one-camera-out on the 104,067 clean points of the even frames. Docs: `docs/self-calibration.md`.
- **Mechanism verified with a known answer** (rendered synthetic recording `syn1`, calibration perturbed by a known amount,
  leave-one-camera-out 0.67 px): extrinsics alone do nothing (0.50 px); with principal point, radial and decentering
  distortion the position error against truth after removing the gauge returns to that of the true calibration (median
  0.0183 mm, p90 0.0754), leave-one-camera-out 0.099 px, which is *below* the true calibration's 0.210 px (the render and the
  processing calibration of the synthetic data differ: raw position error against truth 0.168 -> 0.121 mm).
- **Real wp2:** held-out leave-one-camera-out 0.3036 -> 0.2488 px (-18 % with 1,500 iterations; -17.5 % at 800; -16 % at
  300; not fully converged). What helps: decentering p1, p2 (-12 %), with principal point and radial (-16 %); exteriors alone -2.6 %.
  Cubic image-field fit explained 45-54 % of the variance, so about a third of the floor is still unexplained by this model
  (refraction/glass terms are not in the parameter set).
- **Effect on tracks: none.** Re-triangulating all observations with the new calibration: depth jitter 0.0393 -> 0.0390 mm
  (-1 %), jump share 4.79 -> 4.64 %. As expected for a smooth error.
- **Absolute positions move**: median 0.090 mm (p99 0.37), non-affine part 0.067 mm; a global linear part of about 0.5 %
  (max coefficient 5e-3) that does not depend on the parametrisation (same with tight exterior priors, with radial terms
  frozen, with intrinsics only). Tracers constrain consistency between cameras, not the scale. Without a known length in the
  volume (dumbbell, plate) the accuracy gain on real data cannot be confirmed, and a scale change of 0.5 % would change every
  velocity by 0.5 %.
Decision: tool and evidence, not a default step. Gate (depth jitter -20 %) not met; the held-out consistency gain is real.
Next if wanted: add the known-length constraint (dumbbell or plate points) to the adjustment, then apply to the run and
recompute the downstream statistics.

## What we learned, end to end (2026-11-21)
Pipeline: images -> detection -> calibration -> stereo matching -> tracking -> smoothing -> binned statistics.

**Detection (images -> blobs).** The blob table is the information bottleneck. 27 % of benchmark observations (about half of
that on real wp2) sit in merged blobs with a shifted centre; 14.6 % of particles have no blob in two cameras. The ceiling when
every particle gets its own centre (velocity error 0.063 against 0.106 achieved) says the pixel level is the largest unused
source of information. Blob attributes (brightness, count, path) recover a lot of it, not all.

**Calibration.** The leave-one-camera-out floor of 0.30 px is about half systematic (smooth in image position and depth) and half
detection noise. Joint self-calibration finds it (known-answer synthetic: exact recovery; real wp2: -17 % held-out) but does not
change track jitter, moves absolute positions by 0.09 mm and cannot fix the scale (tracers constrain consistency, not length).
Even the "true" processing calibration of the synthetic data carries a hidden mismatch.

**Stereo matching.** The decision is taken once, without time. At x4 the matcher cloud holds 26-40 % of the particles; the 2-clean
group (21 % of particles) is matched 34 % of the time; 16 % of the 2-camera pair candidates are real. Using time afterwards
recovers part (births: 49 -> 72 % at default density). A pixel-domain flow prior cannot confirm candidates cleanly (72 %
precision) because merging changes frame to frame.

**Tracking.** The chain is accurate with exact data (0.0009). Noise-aware confirmation, births from residual brightness and
dropping unlinked births give longer, cleaner tracks: real wp2/wp1 (1,072 frames) mean length +42 %, jump share /3.5, jitter
-30 %; benchmark velocity error -24 %. Temporal persistence is the best ghost filter: linked births 95 % real, unlinked 53 %.

**Statistics.** Phase-averaged velocity, MKE, VSS and eps do not move (< 2 %) with any of this. The apparent gap to the legacy
table was a pipeline asymmetry (a 2.5 m/s cap applied to ours, not to legacy): with the same cap velocity -1.6 %, MKE -2 %,
VSS +12 %, eps +24 % (legacy noise). Reconnect, repair, Savitzky-Golay and voxel averaging already absorb random noise.

**The pattern.** Downstream outputs are insensitive to *noise* (smoothed and averaged away) and sensitive to *bias* and
*selection*: scale and calibration, which particles are kept (yield 0.74 is not random: fast, dense, merged regions lose more),
filters and masks. We optimised noise and yield; the leverage left is bias and selection, and detection upstream.

**Next, in this order.** (1) Known-length constraint (dumbbell/plate) in the self-calibration, then apply and recompute the
statistics: settles scale and absolute accuracy. (2) Selection audit: yield and velocity bias by speed, depth, voxel, jet region
against what a uniform seeding would give. (3) Pixel-level fit for merged and missing blobs (joint two-Gaussian), the root of
matching and tracking losses. (4) Calibration as one more latent variable inside the holistic loop (match -> calibrate ->
match). Housekeeping: merge PR #44, update docs, speed work.
