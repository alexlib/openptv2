# Merged-blob ideas for 3D-PTV (brainstorm)

Context: `two_phase` tracker is near-ceiling on reachable links (99 %). Ghosts cost 0.5–2 %. The real gap is
**merged blobs**: 27 % of blob observations hold ≥2 particles (≈13 % on real wp2). The correspondence step
assigns each blob to one 3D point, so a merged blob (a) shifts the kept particle, (b) drops the partner,
and (c) the dropped partner then steals the merged blob as a "clean" observation and corrupts its partner.
Membership inference is already ~98 % correct; the loss is from *hidden partners* not being recovered.
Bringing them back also makes the already-tracked particles more accurate.

## Theme 1 — Representation: a blob is a container, not a point

1. **Blob-as-container state.** Maintain per-blob a set/list of particle IDs it is believed to contain, updated by
   the tracking loop rather than thrown away after correspondence. The loop's unit of matching becomes
   blob↔blob (containers), with particles moving through containers; de-merging happens inside a container.
   This is the "wrapped in a loop that keeps track of what each blob contains" architecture.

2. **Multiplicity / super-particle points.** A 3D point produced from a merged blob is not one particle but a
   cluster with multiplicity m. Store m (and an estimated per-constituent intensity) on the point. Downstream
   tracking treats it as a soft observation with inflated covariance instead of a clean point, preventing
   the "partner steals merged blob as clean" corruption.

3. **Intensity/area budget per particle.** Track each particle's blob brightness (and area) as a conserved
   quantity. A merged blob's brightness ≈ sum of constituents. When a particle is occluded/merged, its
   brightness is absorbed into the host; recovery is triggered when a host's brightness exceeds its
   single-particle budget by a threshold.

## Theme 2 — De-merging at correspondence time

4. **Mixture-centroid solve.** For a merged blob in camera c with measured centre μ_c and known/estimated
   constituent intensities {I_k}, and candidate particles with predicted positions, solve for updated positions
   such that the intensity-weighted projection of the constituents reproduces μ_c (plus other cameras'
   constraints). Small nonlinear least squares per merged blob; recovers both positions jointly and also
   improves the kept particle (matches "bringing them back makes existing particles more accurate").

5. **All-subsets triangulation.** Allow the same 2D blob to participate in *multiple* candidate 3D points
   (with a penalty), instead of one-to-one assignment. Enumerate pairings of a merged blob with blobs in
   other cameras; generate candidate points for hidden partners. Let the tracker's consistency (trajectory
   continuity) select the subset. Cost: more candidates, but membership inference already filters.

6. **Residual peeling with known templates.** After assigning the best point to a merged blob, subtract the
   projected PSF of that particle and re-detect in the residual image. Unlike blind peeling, use the tracked
   particle's known brightness/shape as the template, so order dependence is reduced. Gate: only peel when
   residual exceeds noise floor.

7. **Temporal de-merge via interpolation.** A particle hidden in frame t by a merge can be interpolated from
   t-1 and t+1 (trajectory continuity). Use the interpolated position to (a) claim a sub-region of the merged
   blob and (b) seed a "virtual observation" for the hidden partner at reduced weight, feeding tracking.

## Theme 3 — Tracking loop changes

8. **Predicted-position splitting at match time.** When the tracker predicts two particles near a merged blob,
   do not compete for the single blob; instead split the blob into two pseudo-observations at the predicted
   positions (with covariance inflated by merge uncertainty) and feed both. The merged blob becomes a shared
   measurement for two tracks (extends the existing `allow_shared` idea but at 3D-point level).

9. **Merged-observation flag + covariance inflation.** Tag each 3D point that came from a merged blob. In the
   Kalman/track update, inflate its measurement noise and never use it alone to update velocity. Prevents
   the shifted centre from biasing trajectory and velocity (addresses corruption of partners).

10. **Bidirectional recovery pass.** Run forward tracking, then a backward pass; particles missed due to merges
    forward may be recovered backward (and vice versa). Merge trajectories, then re-run correspondence with
    the recovered hidden partners as known seeds to split previously merged blobs. Iterates toward closure.

11. **Membership-driven reassignment.** Use the 98 %-accurate membership inference to, post-correspondence,
    reassign merged blobs: if blob B is inferred to contain particles {p,q} but only p got a point, create a
    point for q from (a) other cameras, (b) predicted position, or (c) the merged centre split. This is the
    cheapest high-value change: it does not change the matcher, only adds missing points.

## Theme 4 — Global / closure

12. **Iterative EM over blob content.** E-step: given current tracks, infer blob membership and split positions.
    M-step: re-run correspondence + tracking with the split observations. Repeat until membership/positions
    converge. The "loop that keeps track of what each blob contains" is literally this outer loop.

13. **Multi-frame data association (min-cost flow).** Replace per-frame Hungarian matching with a windowed
    (e.g. 4-frame) assignment where a merged blob can serve multiple particles across frames with capacity
    constraints and trajectory-consistency costs. Expensive but directly models the one-blob-many-particles
    reality; the existing chunked parallel infra could host it.

14. **Deblending at segmentation.** Lower the merge rate at the source: adaptive threshold + watershed
    deblending on touching blobs, accepting extra noise then filtering by track continuity. Reduces the 27 %
    to ~13 % (real wp2 level) before correspondence even sees it. Complementary, not competing.

## Suggested first experiment (cheapest, highest expected gain)

Implement idea 11 (membership-driven reassignment) as a post-correspondence step:
- After correspondences are built, run membership inference per blob (already ~98 %).
- For each merged blob that was assigned to one point, if membership says it contains a second particle,
  attempt to create a point for that partner using (i) its predicted position from tracking (if a track
  exists) projected and matched to the merged blob's camera subset, else (ii) split the merged centre using
  intensity weights and triangulate with the other cameras.
- Feed both points into tracking with the merged flag (idea 9).
- Measure on the benchmark: vErr, yield, and whether existing (kept) particles get more accurate.

This isolates the "hidden partner" effect the user identified as the main finding, without rewriting the
matcher, and gives a measurable before/after on the same metrics used in bench/*.json.

## Investigation findings (from repo test cases / synth bench)

**Q1 — Does membership inference expose which particles, or just a count?**
- The repo stores no membership. In `synth_bench`, merged observations are single 3D points (the merged centre); truth
  identities are lost (`id=-1` in `proto_shared_validate`). The oracle linker (`_oracle`) assigns each point to its
  *single* nearest truth particle, so the second constituent becomes a "missing" truth point.
- Consequence: any recovery method must be benchmarked against an **oracle membership** first (truth particles within
  ~0.6 mm of a point ⇒ merged blob + its constituents). This gives the upper bound. A practical membership would use
  nearest predicted tracks / intensity; the existing `q_seed`/ghost gating already controls which new tracks may start.
- Implication: Idea 11 is implementable as *oracle-first* to bound gain, then swap in a predictor-based membership.

**Q2 — Cost budget.**
- Baseline `two_phase` ≈ 4 s / 200 frames / ~200 particles on `L1_d1`. Hidden partners touch ~27 % of blob
  observations, i.e. tens of points per frame. A per-merged-blob recovery pass (Ideas 4/11) adds at most a few seconds
  (well under the 10× cost of a windowed MFA). The mixture-centroid solve (Idea 4) is the only per-point nonlinear
  cost; keep it optional and measure.
- `allow_shared=True` (existing) ran in 4.3 s with zero metric change: it does not create points, so it cannot recover
  hidden partners — confirms recovery needs new points, not just shared links.

**Q3 — Should hidden partners start new trajectories or only attach to existing ones?**
- Evidence from `proto_shared_validate.score_tracks`: a merged row counts as 2 truths only when **two** tracks hold it;
  a single track covering it counts as 1. Starting new tracks from hidden partners risks ghost inflation, but the
  bench `ghost` metric will expose it; `q_seed` gating already prevents doubtful points from seeding.
- Recommendation: first allow hidden partners to **attach to existing tracks as shared observations** (low ghost risk,
  measures accuracy gain on kept particles); then, only if a particle has no nearby track, seed a new track gated by
  `q_seed`. Evaluate `ghost` and `yield` to decide.

## Experimental result (worktree, 2026-11-20)

Upper-bound experiment `scripts/exp_split_merged.py` on `L1_d1_c0_k1_n200`: replaced each merged
blob's 3D centre with its constituent truth particles (perfect split, leaves projected from 3D),
ran `two_phase`, matched predicted points to truth within 0.6 mm.

- baseline (merged centres): 215 548 / 300 000 truth points covered (71.8 %)
- oracle split:            218 348 / 300 000 (72.8 %)

Gain ≈ +2 800 points (+1.0 pp coverage) from recovering hidden partners alone. Small but positive;

vErr check (oracle split, L1_d1): vel 19.65 → 19.79 (slightly worse). Coverage improves but velocity does
not, because adding hidden partners at truth positions without correcting the kept particles' shifted
centres (and without 2D-leaf consistency) does not help tracking. This confirms Idea 4 (mixture-centroid
split of the merged centre) is required for vErr, not just Idea 11 (adding points).

Predictor-based two-pass (Idea 11 + 4, L1_d1): vel 19.65 → 19.35 (improvement). Membership via nearest
predicted track; split via merged_split. Integrated into src/openptv2/blob_recovery.py (recover()).
predicted track; split via merged_split. First vErr gain achieved.

Extended tuning (init-residual gate, exact-2 tracks, <1mm): sparse cases improve (L1_d1 19.65→19.35,
L1_d1 k8 3.23→3.19) but clustered/dense regress (L1_d2 12.20→12.55, L1_d4 92.80→94.31).
Root cause: predictor-based membership is too permissive on dense data (false merges); init-residual
gate never rejects (all 1614 splits accepted). Next: add intensity/area-based membership (merged blob
brightness > single-particle budget) and a stricter spatial gate (point must lie between constituents).
(vErr 0.308→0.339, ghost 0.036→0.068) despite conservative gating (exact-2 tracks, <1mm apart,
<0.5mm residual). Root cause: split points are treated as clean (Idea 9 noise inflation not yet
applied in tracker), and predictor-based membership is noisy. Tuning stopped per early-stop rule.
Next: apply per-point noise inflation in tracker (merged flag) before re-enabling.
predicted track; split via merged_split. First vErr gain achieved.

Full outer loop (feat/blob-ownership-loop, "blob_loop" plugin): tested on L1_d1 (n=12 and n=200).
Results much worse than two_phase (n=200: vErr 0.42 vs 0.16, ghost 0.07 vs 0.04, 80s vs 4s).
Not recommended as-is; needs tuning before it can replace two_phase. The predictor-based split_merged

Unified tracker (feat/unified+merged-blob merge): L1_d1 vErr 0.266 vs two_phase 0.160, 43s vs 4s.
Pre-existing test failures (7+3) confirmed not caused by merge. Unified tracker not yet better
than two_phase; merged-blob ideas integrated but not improving end-to-end yet.
flag (gated, default off) is the more promising incremental step.
predicted track; split via merged_split. First vErr gain achieved. Next: wire into plugin do_tracking.
not, because adding hidden partners at truth positions without correcting the kept particles' shifted
centres (and without 2D-leaf consistency) does not help tracking. This confirms Idea 4 (mixture-centroid
split of the merged centre) is required for vErr, not just Idea 11 (adding points).

Batch (synthetic L1 cases, oracle split, two_phase):
- L1_d1_c0_k1_n200: 215548→218348 /300000 (+0.9 pp)
- L1_d1 ..._s0.08: 215443→218216 /300000 (+0.9 pp)
- L1_d2_c0.3_k4_n150: 202051→211271 /450000 (+2.0 pp)
- L1_d4_c0_k1_n150: 282234→294979 /900000 (+1.4 pp)
- L1_d1_c0_k8_n150: 156733→159367 /225000 (+1.2 pp)
- L1_d1 ..._a0.18_f0.05: 210513→213383 /300000 (+0.9 pp)

Consistent +0.9–2.0 pp coverage gain across densities/shear/jitter. Real datasets (lv_multi,
Illmenau) have no ground truth, so the oracle split cannot be scored there; testing there requires
implementing a predictor-based membership (nearest predicted track) and measuring yield/vErr/ghost.
the larger vErr benefit (user's "bringing them back makes existing particles more accurate") requires
also fixing shifted centres (Idea 4, mixture split), which this upper bound includes for positions but
not for the 2D-leaf consistency. Next: measure vErr via a de-merged store case and test Idea 9.

## Suggested order to check (first → last)

1. **Idea 11 — membership-driven reassignment, oracle membership.** Cheapest; directly tests the user's main finding
   (hidden partners). Add missing partner points post-correspondence using oracle membership; measure vErr/yield/ghost
   on `L1_d1`/`L1_d2`/clustered cases. Bound on gain.
2. **Idea 9 — merged flag + covariance inflation.** Tag points that came from a (recovered) merged blob; inflate
   measurement noise and never update velocity from them. Prevents the "partner steals merged blob as clean" corruption.
   Combine with (1).
3. **Idea 7 — temporal interpolation of hidden partners.** For particles missing due to merges, interpolate from
   t-1/t+1 and seed a virtual observation. Low cost, no spatial split; complements (1) when membership is uncertain.
4. **Idea 4 — mixture-centroid solve.** Fix shifted centres of kept particles once (1) shows gain. Per-merged-blob
   nonlinear LS; measure cost.
5. **Idea 14 — deblending at segmentation.** Complementary pipeline change; reduces the 27 % merge rate toward the
   ~13 % real-wp2 level before correspondence. Evaluate separately.

Defer: **Idea 13 (windowed MFA)** — correct but ~10× slower; only if (1)–(4) plateau. **Idea 1 (blob-as-container
rearchitecture)** is the end state; build it after the above experiments quantify the gain.

## Open questions to settle before coding

- Does membership inference expose *which* particles, or just a count? If count only, intensity-weighted
  split (idea 4) needs a separate identity source (tracking prediction).
- What is the cost budget? Idea 13 (windowed MFA) is the "right" fix but likely 10x slower; the loop
  (idea 12) may be the pragmatic middle.
- Should hidden partners be allowed to start *new* trajectories, or only attach to existing ones? Starting
  new ones risks ghost inflation; attaching only recovers within-track accuracy.
