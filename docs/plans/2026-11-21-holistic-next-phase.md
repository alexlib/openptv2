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
