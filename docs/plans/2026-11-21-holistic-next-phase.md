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
