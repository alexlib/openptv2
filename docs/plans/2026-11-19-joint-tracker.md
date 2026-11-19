# Plan: a joint stereo-matching + tracking algorithm ("joint tracker")

Date: 2026-11-19. Branch: `feat/zarr-only-remove-legacy`. Status: plan, nothing implemented yet.
Related: `docs/plans/2026-09-30-tracker-plan.md` (benchmark facts),
`docs/plans/2026-10-09-trackcorr-two-phase-merge.md`,
`docs/plans/unified-tracker-implementation.md` (peeling, ILP, stitching).

## 1. Why

**Problem.** The baseline (`trackcorr` forward tracking) and every newer idea in this
repo work on a 3D point cloud that the stereo matcher fixed **once per frame, without
any knowledge of time**. Measured earlier (`2026-09-30-tracker-plan.md`):

- Ghost points cause about 60 % of the remaining velocity error and nearly all wrong
  links. Most ghost points sit in ghost-only trajectories, often 30+ frames long.
- Tracks rarely lose a contest (3–6 % contested, about one candidate per track). They
  fail because **the real particle is not in the cloud at all**: 12.8 % of true points
  are seen by ≥ 3 cameras but missing from the 3D cloud; 22 % "no candidate" at 4× density.
- Ghost cues per point saturate (AUC about 0.92; 46 % of flagged points are real), so
  deleting points on a mark costs real particles. Soft, claim-based rules work.
- Dropping whole doubtful tracks afterwards is worse than point rules, because a ghost
  track that formed has already taken part in the assignments.
- Feeding track knowledge back into matching (A11) failed: the matcher gives back the
  same combination, because nothing can forbid one and alternatives are not kept.

**Root cause in the code.** `algorithms/correspondences.py::correspondences` finds all
4-, 3- and 2-camera combinations that pass the epipolar band (`con0`), then
`take_best_candidates` keeps a greedy non-overlapping subset by `corr` score (4-camera
first) and **throws the rest away**. When a ghost combination wins, the real particles
whose blobs it used become 3-camera points or vanish. Peeling and the ILP modules can
only prune this greedy cloud; they never see the discarded alternatives.

**Goal.** A new standalone tracker that starts from the stored 2D blobs (targets) and
the calibration and decides matching and tracking together: fewer ghosts, more
recovered real points, longer smooth trajectories, no wrong links.

**Decisions taken (2026-11-19).**
- Blobs only (target space): no image subtraction, no optical model.
- New standalone tracker plugin; `trackcorr`, `two_phase`, `unified` stay as baselines.
- Judge: `scripts/synth_bench.py` (truth-based) + real CompleteTest wp2 / lv_multi.
- Measure the best possible gain of each ingredient first; build only what pays.

## 2. What exists today, and where it goes

| Idea | State in the repo | Role in the joint tracker |
|---|---|---|
| Peeling / Clean-and-Claim (`layered_tracking.peel_and_track`, tiers 2/3) | code; **on by default in `unified`, with the 3D-only engine (track3d), not trackcorr**; prunes the greedy cloud, never re-matches | its principle (a verified track owns its blobs) becomes the core rule, applied frame by frame on all candidates |
| Per-frame ILP (`ghost_ilp.ilp_uniqueness_frame`), conflict groups (`ghost_components`) | code; only with `ilp_mode: frame`; selects among the greedy cloud's points | bound B5; later only for tie groups |
| Trajectory ILP (`ghost_ilp.trajectory_ilp_uniqueness`) | code, default `ilp_mode` | promotion ties and final cleanup |
| Track-assisted claims (`track_assisted._claim_particle`, shared blobs, poly predictor) | code; a late corrective pass, one frame per pass at track ends | "tracks pick first", inside the loop |
| IRLS stitching (`temporal_smoothing.stitch_tracklets_irls`) | code, on by default; adjacent frames only, greedy | final step |
| Temporal smoothing (`temporal_smoothing.smooth_trajectories`) | code; a keep/drop filter per trajectory (the planned sliding-window joint solve was not built) | not used as a deleter; smoothness enters through probation |
| Ghost marks (`point_quality.ghost_probability`: relative rcm, cameras, brightness) | code | soft score for candidates and seeds |
| Two-hop confirmation, dead-end rule (`tracking_postprocess.confirm_links`) | code | probation |
| Multi-hypothesis tracking | idea only (docs) | probation tracklets that may share blobs |
| Epipolar iterations / several camera orders | idea only (docs) | not needed: the pool keeps all combinations whatever the camera order |
| Conflict-group Hungarian (`layered_tracking.resolve_conflict_components`) | code, never called | contested picks among confirmed tracks |

**No measured numbers exist** for peeling, frame ILP, trajectory ILP, smoothing or IRLS
stitching on the truth benchmark; they were only checked on
`tests/benchmarks/test_unified_gt_scorecard.py` (floors: precision ≥ 0.80 clean, ≥ 0.40 others).
That scorecard's `dense_ghost` ghosts are random 3D points with their own blobs in every
camera and identity correspondences (`benchmarking/scenario.py`, `datawriter.py`); the
matcher never runs, so they are **not** epipolar ghosts and the peeling result there
(precision 0.999) does not carry over to real ghosts.

**The stored cloud is already unique.** `take_best_candidates` lets each blob serve at
most one stored 3D point. So the per-frame ILP on stored correspondences has nothing to
decide (every conflict group is a single point), and "ghosts reuse blobs of real
particles" is true only inside the matcher, among the candidates that are discarded.

**What the matcher throws away** (and the joint tracker keeps): all losing 4/3/2-camera
combinations; the match score `corr`; the number of candidates per blob; 2-camera
combinations (pairs are off by default on rigs with 3+ cameras). There is no way today
to ask the matcher for alternatives, to forbid a combination, or to give it a prior.
On the batch path `rcm/` is not written and `tnr` is not stamped into stored targets,
and later stages rewrite `correspondences/` without updating `rcm/`; the joint tracker
therefore computes rcm itself from the blobs it uses.

**Defects found in the baseline while reading** (they make `unified` look worse or
different than designed; see step 0a):
1. `plugins/unified_tracking.py` (about L879) imports `compute_correspondence_rcm` from
   `layered_tracking`, which does not define it, and calls `store.has_targets(f)` with
   one argument; the exception is swallowed, so the trajectory ILP's rcm factor is always 1.
2. `_load_linkage_chains` ignores gap-bridged links (step > 1), so a bridged index is
   read in the wrong frame (affects trajectory ILP and temporal smoothing).
3. `dangle` is gon in trackcorr and degrees in track3d, IRLS stitching and `prune_kinky_links`.
4. Peeling default: on in code, off in `PRESET_CONFIGS["unified"]` and in `docs/tracking.md`.
5. Peeling pass 2 mixes `full_forward_3d` with the epipolar `full_backward`.

## 3. The idea in plain words

1. **Keep every candidate.** Per frame, keep all blob combinations that pass the
   epipolar test, each with its 3D position, ray miss distance (rcm), number of cameras
   and brightness spread. Choose nothing yet.
2. **Tracks pick first.** A track with history predicts where its particle is in the
   next frame. Near that prediction it takes the fitting candidate; if there is none,
   it projects the prediction into the cameras and takes the nearest blobs directly.
   Its blobs are then taken.
3. **New particles come from the leftovers.** Only blobs that no track took are
   matched for new particles. With most blobs gone, accidental combinations (ghosts)
   become rare: their number grows much faster than the number of blobs.
4. **Newborns are on probation.** A new candidate becomes a track only after N frames
   (start with 4) of smooth motion. During probation, rival newborns may use the same
   blob; nothing is decided. At promotion the winner takes its blobs through its whole
   past and the rivals that needed them are removed. A ghost made of blobs from two
   real particles dies here, because both real particles get promoted and take their blobs.
5. **Sharing is explicit.** Two real particles can overlap in one camera. A blob may
   serve two confirmed tracks when both predictions fall on it (lower weight in the
   triangulation). So exclusivity does not delete real overlapping particles.
6. **Sweep forward, then backward, and repeat.** The backward sweep extends tracks to
   before their birth and takes blobs there, which removes ghosts that lived early.
   Stop when the set of tracks no longer changes (expected 2–3 sweeps). The first
   frames have no tracks, so the start uses strict seeds: 4 cameras, low relative rcm,
   no rival candidate on the same blobs.
7. **Final cleanup over whole trajectories.** Trajectory-level blob uniqueness (ILP),
   stitching of pieces, quality marks: all exist and are reused.

Difference from today's peeling: it works on all candidates instead of pruning the
greedy cloud; matching and linking alternate frame by frame instead of running the
whole tracker several times; decisions about new points are delayed.

**Known limit.** A ghost whose blobs are all false targets that are consistent across
cameras (a static bright object) looks like a real stationary particle to any
geometric method. It is out of scope; do not tune on speed (real data has slow real flow).

## 4. Step 0: baselines and truth-based bounds (no new algorithm yet)

Cases (exist in `~/Downloads/HiDImaging/tracker-bench-2026-09-29/cases`), all with
real jitter, per-camera brightness and flicker (`_s0.08_a0.18_f0.05`):
`L1_d1_c0_k1_n200` (sparse), `L1_d2_c0_k1_n150`, `L1_d4_c0_k1_n150`,
`L1_d2_c0.3_k4_n150` (clustered, frame skip 4). L1 cases hold synthetic blobs run
through the real matcher, so a tracker that starts from blobs is tested fairly.

**0a. Baseline hygiene (separate small commit, before B0).** Fix defects 1 and 2 of
section 2 (a few lines each, with a test), so `unified` is judged as designed. Record
defects 3–5; do not change them here.

**0b. Harness.** `scripts/synth_bench.py::build_case` already computes, per true
particle and camera, the blob index (`tid`); save it in `truth.npz` (today only `ndet`).
Rebuild the four cases. Every bound below needs this map.

| # | Experiment | Question | Go / no-go (proposed) |
|---|---|---|---|
| B0 | `trackcorr@fwd`, `trackcorr`, `two_phase`, `unified` (default = peeling), `unified+peeling=false`, `unified+ilp_mode=frame` | Where do we stand on the truth benchmark? | reference numbers |
| B1 | Pool coverage: share of true points (seen by ≥ 2 / ≥ 3 cameras) whose true combination is in the full pool vs in the greedy output; pool size; ghost : real ratio | How many lost real points does "keep every candidate" bring back, at what cost? | pool recovers at least half of the 12.8 % |
| B2 | Perfect selection from the pool (truth picks), then `two_phase` and the perfect linker | Ceiling of the whole approach | clearly below the "ghosts removed" bound (0.1551 against 0.1934) or more points kept |
| B3 | Predict-then-claim with true histories (true positions of t−1, t−2 plus measured noise): recovered true points and wrong claims, **split by 4 / 3 / 2 cameras** | Does step 2 work at real prediction error and blob density? trackcorr's 2–3-camera adds were much worse (velocity error 0.3047 → 0.5105) | ≥ 95 % correct claims; cameras below that level are not allowed to claim |
| B4 | Remove the blobs of a share f of the true particles (f = 0.5, 0.8, 0.95), match the rest | How fast do ghosts disappear as tracks take their blobs? | ghosts fall faster than (1 − f) |
| B1b | B1 again with 2-camera combinations in the pool (2-camera matches were 64 % ghosts on test_cavity, triplets 38 %, quads 16 %) | Are pairs worth carrying as probation candidates? | keep pairs only if they add true points that B3-style claims cannot recover |
| B5 | Exact per-frame selection on the full pool without time (`ghost_ilp.solve_component_ilp` on the pool's conflict groups) vs greedy | Is time needed, or is the greedy choice the problem? | expected small gain; confirms time is the lever |
| B6 | Life time of ghost vs real candidate chains in the pool (smooth-motion linking) | Probation length N | N where ≥ 90 % of ghost chains are dead and < 2 % of real ones |

Scripts: `scripts/joint_bounds.py` (one sub-command per bound). Numbers:
`bench/step23_joint_bounds_2026-11.json` and a results section appended to this file.
If B1/B2 fail, drop the pool and build only steps 2–4 of section 3 on the greedy cloud
plus direct claims.

## 5. Step 1: candidate pool

- `src/openptv2/algorithms/correspondences.py`: new
  `correspondence_candidates(frm, corrected, vpar, cpar, calib)`: `match_pairs` plus
  `four_camera_matching`, `three_camera_matching`, `consistent_pair_matching` with an
  all-zero `tusage`, **without** `take_best_candidates`; returns
  `(ids (M, num_cams), corr (M,))`. `correspondences()` is not touched (liboptv parity tests).
  Pairs in the pool are an option (B1b). Candidate lists are cut at `MAXCAND = 200`
  per blob; count how often that happens on the dense case.
- `src/openptv2/correspondences.py`: `match_frame_candidates(detections, cpar, cals,
  vpar)` → `ids, pos, rcm, corr`, built like `match_frame_correspondences`
  (`MatchedCoords`, `point_positions(method="lsq")`).
- Quality per candidate: `point_quality.ghost_probability`.
- Tests (`tests/unit/test_correspondences.py`): the greedy output is a subset of the
  pool; a 2-particle crossing gives both real combinations and the ghost.
- Cython rebuild, then the pure-Python fallback tests (CLAUDE.md).

## 6. Step 2: the tracker (Python prototype first)

`src/openptv2/joint_tracking.py` (algorithm) and a thin plugin
`src/openptv2/plugins/joint_tracking.py`, registered as `joint` in `plugins/loader.py`
and `tracking_registry.py`. Output in the usual run-store layout
(`RunStore.write_correspondences`, `write_linkage_many`, `write_point_quality`), so
`synth_bench`, post-processing and export work unchanged. Angles in degrees.

State: confirmed tracks, probation tracklets, blob ownership per frame (with a share
count), the pool per frame (cached).

Per frame, forward:
1. **Confirmed tracks.** Predict (`track_assisted._predict_position`, or
   `temporal_smoothing.irls_predict_state` for long history); gate; pick pool
   candidates; contested groups by `layered_tracking.resolve_conflict_components`
   (Hungarian per group). No candidate → direct claim with
   `track_assisted._claim_particle` (cameras allowed per B3, shared blobs, ray-miss and
   position gates). One or two missed frames are tolerated, then the track ends.
2. **Probation tracklets.** Same search on blobs not owned by confirmed tracks; no
   exclusivity among tracklets; a tracklet without continuation dies (dead-end rule).
3. **Seeds.** Every remaining candidate with no owned blob starts a tracklet.
4. **Promotion** at age N when the kink is within the data-driven tolerance
   (`tracking_postprocess.kink_statistics`, as `confirm_auto`). The track takes its
   blobs in its past frames; rivals on those blobs are removed; ties between
   promotable rivals go to `ghost_ilp.trajectory_ilp_uniqueness` for that group.

Then the backward sweep (same code, time reversed), repeated until stable. Final:
`trajectory_ilp_uniqueness`, `stitch_tracklets_irls`, positions re-triangulated from
the final blob sets (`lsq`), ghost marks stored.

Build order, each measured on the four cases against B0 and the bounds:
2a forward sweep (steps 1–4, no sharing) → 2b sharing → 2c backward sweep and
iteration → 2d final cleanup. Revert any part that gains only by dropping true points.

## 7. Step 3: real data, speed, product

- Real wp2 (300 frames) and lv_multi wp4/wp5 on scratch copies
  (`scripts/real_retrack.py`, `scripts/realism_metrics.py`): trajectories, mean
  length, ≥ 50-frame tracks, roughness, 3-camera share. Never write into the source runs.
- Profile; move hot loops to `algorithms/` (Cython) only after the accuracy is settled.
- Docs: `docs/trackers.md` entry and parameters.

## 8. How to check

- Unit: `uv run pytest tests/unit/test_correspondences.py tests/unit/test_joint_tracking.py -v`
  (new: pool superset; crossing with a ghost; overlap with a shared blob; one-frame dropout).
- Fallback tests from CLAUDE.md after touching `algorithms/`.
- Benchmark: `uv run python scripts/synth_bench.py track --case <case> --tracker joint`,
  then `eval --case <case> --trackers oracle trackcorr two_phase unified joint`.
- Success: on all four cases lower velocity error than `two_phase` and `unified`,
  ghost share at most half, true points kept not lower, long-track completeness up;
  on real data more ≥ 50-frame tracks and no rise in roughness.
- `uv run ruff check src scripts`.

## 9. Open points

- The go/no-go numbers in section 4 are proposals, not measurements.
- Whether to fix baseline defects 3–5 is a separate decision. Further small defects seen
  in `unified`: `read_correspondence_confidence` is called but is not a `RunStore`
  method (ILP weights fall back to uniform); `ghost_filter.filter_ghosts_rcm` only calls
  the low-support filter.
- Density limit: epipolar matching is known to break down above about 0.005–0.01
  particles per pixel (`docs/plans/archive/2026-08-24-shake-the-box-ipr-roadmap.md`).
  The 4× density case shows how far "tracks pick first" pushes that limit in blob space.
- Speed budget: prototype may be several times slower than `two_phase`; target after
  Cython is at most 2× trackcorr on the dense case.

## 10. Results of step 0 (2026-11-19)

Cases: `L1_d{1,2,4}_c0_k1_n{200,150,150}` and `L1_d2_c0.3_k4_n150`, all
`_s0.08_a0.18_f0.05`. Raw numbers: `bench/step23_joint_bounds_2026-11.json`;
scripts: `scripts/joint_bounds.py` (B1, B2 inputs), `scripts/synth_bench.py` (0b: `truth.npz`
now has `blob_tid`, the target index of every truth point per camera; the rebuilt cases
are identical to the old ones otherwise).

**Step 0a (commit 2f33ef73):** gap-bridged links followed in `_load_linkage_chains`; trajectory
ILP uses `point_quality.frame_rcm`.

### B0: baselines (vErr / link precision / yield / ghost share)

| Case | oracle | two_phase | trackcorr | trackcorr@fwd | unified (peeling default) | unified+peeling=false |
|---|---|---|---|---|---|---|
| sparse | 0.1435 | **0.1825** / 0.961 / 0.680 / 3.8 % | 0.2968 / 0.922 / 0.684 / 7.3 % | 0.2939 | 0.5051 / 0.895 / 0.745 / 9.1 % | = trackcorr |
| ×2 | 0.3336 | **0.3111** / 0.957 / 0.504 / 4.3 % | 0.5084 / 0.906 / 0.521 / 7.7 % | 0.5043 | 0.8766 / 0.856 / 0.614 / 10.8 % | = trackcorr |
| ×4 | 0.5675 | **0.4091** / 0.928 / 0.225 / 7.1 % | 1.0504 / 0.819 / 0.247 / 14.9 % | 1.0495 | 2.0785 / 0.672 / 0.308 / 25.3 % | = trackcorr |
| clustered, skip 4 | 0.1605 | **0.1697** / 0.941 / 0.375 / 5.3 % | 0.7581 / 0.867 / 0.403 / 8.1 % | 0.7593 | 0.8250 / 0.785 / 0.432 / 11.8 % | = trackcorr |

(oracle is the perfect linker on the same input; its vErr can exceed two_phase's because it
keeps more, noisier points.) Findings:
1. **`unified` with the default peeling is clearly worse than `trackcorr` and `two_phase`** on every
   case (link precision 0.67–0.89, ghost share 9–25 %). Peeling is the cause: `peeling=false` is
   identical to plain `trackcorr`. (More long tracks and yield, lower accuracy.)
2. **`two_phase` is the tracker to beat**; trackcorr and unified are far behind it.
3. Headroom of two_phase to the perfect linker on the sparse case: 0.183 → 0.144.

### B1: pool coverage (truth points seen by ≥ 3 cameras, 10 frames per case)

| Case | lost by greedy | pool recovers | "clean" truth lost by greedy (blob not shared) |
|---|---|---|---|
| sparse | 13.6 % | 57 % | 0.2 % |
| ×2 | 31.5 % | 49 % | 1.0 % |
| ×4 | 61.6 % | 32 % | 2.9 % |
| clustered, skip 4 | 43.2 % | 23 % | 0.5 % |

- The sparse figure reproduces the earlier 12.8 %. Pool size is 4.7–5.8× the greedy cloud
  (5300 rows per frame, no `MAXCAND` saturation); every greedy row is in the pool (sanity check passed).
- **Almost all of the loss is blob sharing (merged blobs), not ghosts stealing blobs:** truth points
  whose blobs are not shared lose only 0.2–2.9 %. The pool recovers the part whose merged-blob
  centroid still passes the epipolar band; it recovers little at high density.
- B1b (2-camera combinations): no change to the recall of points seen by ≥ 3 cameras.
- Verdict against the proposed go/no-go (≥ half of the lost points): passes sparse (57 %),
  borderline ×2 (49 %), fails ×4 and clustered.

### B2: ceiling of perfect selection from the pool (two_phase on the input; vErr / yield / long completeness / ghost share)

| Case | greedy cloud | greedy, ghosts removed | perfect pool selection |
|---|---|---|---|
| sparse | 0.1825 / 0.680 / 0.503 / 3.8 % | 0.1743 / 0.678 / 0.501 / 0.8 % | **0.2412** / 0.714 / 0.516 / 2.6 % |
| ×2 | 0.3111 / 0.504 / 0.344 / 4.3 % | 0.3054 / 0.501 / 0.342 / 2.1 % | **0.3964** / 0.575 / 0.372 / 5.2 % |
| ×4 | 0.4091 / 0.225 / 0.157 / 7.1 % | 0.3913 / 0.222 / 0.156 / 4.0 % | **0.4924** / 0.302 / 0.196 / 8.3 % |

- Removing every ghost from the input gains only 0.005–0.018 for two_phase (it already has the
  ghost rules).
- **Perfect pool selection gains points (+3 to +8 points yield, +1.5 to +4 points long-track
  completeness) but makes the velocity error worse by 0.06–0.10 on all three cases.** The recovered
  points are merged-blob points whose centroid is shifted; their 3D positions are noisy. Under the
  rule "accuracy ranks above density", this is not a win.

### Decision point (stopped here, as the plan says)
B1/B2 fail for the pool as designed. Plan section 4 says: drop the pool and build only the claim
logic. Options and the recommendation are in the report to the user; nothing beyond step 0 is built.
