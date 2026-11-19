# Plan: a joint stereo-matching + tracking algorithm ("joint tracker")

Date: 2026-11-19. Branch: `feat/zarr-only-remove-legacy`. Status: step 0 measured; **plan revised the
same day, and again after merging `feat/ilp-temporal-smoothing-impl`: read sections 10–14 first; section 14 is the current plan.** Sections 3–9 are the original design and are kept for
the record; where they differ from section 11, section 11 wins.
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

## 3. The idea in plain words (original design; revised in section 11)

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

### Extra check: merged blobs (2026-11-19, one-off script, 8 frames per case)

A blob is "merged" when two or more true particles share it in that camera
(`truth.npz['blob_tid']`). For each true particle, the pool combination with the most cameras
(the B2 pick), by the number of merged blobs it uses:

| Merged blobs in the pick | Share of picks (sparse / ×4) | RMS position error, mm (sparse / ×4) | More than 0.4 mm off (sparse / ×4) |
|---|---|---|---|
| 0 | 62 % / 25 % | 0.068 / 0.076 | 0 % / 0 % |
| 1 | 24 % / 27 % | 0.21 / 0.25 | 6 % / 11 % |
| 2 | 9 % / 25 % | 0.42 / 0.48 | 27 % / 32 % |
| 3 | 5 % / 23 % | 0.68 / 0.79 | 43 % / 52 % |

- 38 % (sparse) and 75 % (×4) of true particles have at least one merged blob in a frame.
- Where a combination of ≥ 3 cameras without merged blobs exists (34 % of the merged picks on
  the sparse case, 9 % at ×4), using it cuts the RMS error from 0.18 to 0.08 mm.
- Greedy cloud, sparse case: 7.3 % of rows are made of blobs of different particles; 8.4 % are
  more than 0.4 mm from truth. Of the rows made of one particle's own blobs, 1.5 % (sparse) and
  10.4 % (×4) are still more than 0.4 mm off: these are merged-blob points.

### What step 0 means

1. **The main enemy on this benchmark is overlapping particle images (merged blobs), not ghosts
   stealing blobs.** A merged blob has a shifted centre; a 3D point that uses it is 3–10× less
   accurate, and with 2–3 merged blobs it is often far enough off to count as a ghost.
2. **B2 did not measure the pool's ceiling.** Its "perfect" pick preferred more cameras, so it took
   4-camera combinations containing merged blobs over clean 3-camera ones. And its velocity error
   was averaged over a larger, noisier set of points, so it rose even where no original point
   changed. The same run shows what the extra points buy: perfect-linker long-track completeness
   0.553 → 0.628 and pieces per long track 1.83 → 1.53 (sparse).
3. **`two_phase` is the baseline to beat.** `trackcorr` and `unified` are far behind it.
4. **Peeling should not be the default in `unified`** (worse than plain `trackcorr` on all four
   cases). Separate decision for the user; not part of this plan.
5. Not yet known: whether real data has as many merged blobs as the benchmark's merge model
   (`synth_bench._target_rows`), and whether track breaks happen at the overlap events.

## 11. Revised plan (2026-11-19): merge-aware joint tracker

### 11.1 The idea in plain words

The first design said: tracks own their blobs, so ghosts lose. Step 0 shows the larger loss is
elsewhere: two real particles overlap in one camera, the detector gives one blob with a shifted
centre, and the matcher builds an inaccurate point or none. The matcher cannot know a blob is
merged. **A tracker can: two tracks that are predicted onto the same blob tell it.** So:

1. **Tracks predict first.** Every confirmed track predicts its 3D position and its position in
   each camera.
2. **Find merged blobs from the predictions.** A blob is merged when two or more predictions fall
   on it (within the blob radius). Blob size is a second hint.
3. **Triangulate around the merge.** A track uses only its clean cameras when enough remain
   (how many is decided by bound B3).
4. **Otherwise de-merge the centre.** For a merged blob with centre `c` and predictions `p_i`,
   particle `i` gets the 2D position `c + (p_i − mean(p))` (brightness-weighted mean when the
   brightness is known), used with a lower weight. Blobs only; no images.
5. **Keep the track through the overlap.** One or two frames without a usable position are
   coasted, so the track does not break at the overlap.
6. **New particles come from the leftover blobs.** Blobs taken by confirmed tracks are removed
   (a merged blob counts as taken); the existing matcher runs on the rest. This needs no change
   to `algorithms/correspondences.py`: `match_frame_correspondences` is called on the reduced
   target lists.
7. **Newborns are on probation**, as in section 3 (point 4): smooth motion for N frames, dead-end
   rule, and at promotion the track takes its blobs through its past. The existing `two_phase`
   rules (`q_seed`, `q_young`, `confirm_auto`) are the starting point.
8. **Forward, then backward, until stable; final cleanup** as in section 3 (points 6–7).

The candidate pool (section 5) is no longer a required part. It comes back only if bound M1 shows
a gain that the steps above cannot reach.

### 11.2 Bounds to run before building (all truth-based, in `scripts/joint_bounds.py`)

Results go to `bench/step23_joint_bounds_2026-11.json` (new keys `M0`…`B6`) and to a new
section 12 of this file. Cases: the three `_s0.08_a0.18_f0.05` cases sparse, ×2, ×4 (the
clustered case when time allows).

| # | Experiment | How | Go / no-go (proposed, not measured) |
|---|---|---|---|
| M0 | Do real recordings have as many overlaps as the benchmark? | No truth on real data, so use a proxy on tracked output: project every linked 3D point of a frame into each camera; share of points with another projected point within r = 1, 1.5, 2, 3 px. Compute on real wp2 (`~/Downloads/HiDImaging/CompleteTest-e2e-local/wp2/test/res/run.zarr`, read-only, 100 frames) and, the same way, on the sparse case's `two_phase` output. Also: share of 3-camera points whose projection into the missing camera lies within r px of a blob used by another point | If the real shares are below a third of the synthetic ones, the benchmark overstates merging: report and stop before building |
| M1 | Fair ceiling of selection | Per true particle take: (a) the best pool row without merged blobs and ≥ 3 cameras; else (b) a row from its clean cameras only when ≥ 2 are clean (build the id row from `blob_tid`, triangulate with `triangulate`); else (c) the most-camera row as in B2. Write the cloud (`<case>_M1`), track with `two_phase` and `oracle`. Report variants a, a+b, a+b+c | Position error of the added points below 2× the clean error; common-set velocity error (see 11.5) not worse than B2ng; long completeness higher |
| M2 | Ceiling of de-merging | For every merged blob give each of its true particles the 2D position `c + (p_i − mean(p))`, with `p_i` = projection of the true position plus Gaussian noise of 0.3 px and of 0.6 px (two runs; equal weights and brightness weights). Triangulate each particle from its clean and de-merged positions (`transforms.convert_arr_pixel_to_metric`, `distorted_to_flat`, `orientation.point_positions(method="lsq")`). Report position error by number of merged blobs (table above as reference), then track (`<case>_M2`) | RMS error of merged picks at least halved at 0.3 px noise |
| B3 | Predict-then-claim | For every true particle with truth at t−2, t−1: predict t with `track_assisted._predict_position` from the true positions plus the measured 3D noise (0.019 mm in-plane, 0.094 mm depth); project; take the nearest blob within `radius_px` (1.5, 2, 3) per camera. Report share of correct blobs, wrong blobs and misses, split by cameras hit (4 / 3 / 2) and by merged / clean blob; and the prediction error in px (this sets the M2 noise level and the merge radius) | ≥ 95 % correct claims for a camera count to be allowed to claim |
| B4 | Leftover matching | Remove the blobs of a share f = 0.5, 0.8, 0.95 of the true particles (merged blobs removed when any owner is in the share), run `match_frame_correspondences` on the rest, count identity ghosts and recovered true points | Ghost rows fall faster than (1 − f) |
| B6 | Probation length | Link the leftover candidates of B4 (f = 0.8) with `two_phase`; life-time histogram of ghost chains vs real chains | N where ≥ 90 % of ghost chains are dead and < 2 % of real ones |
| F1 | Do tracks break at overlaps? | On the sparse `two_phase` output: for each end of a track piece whose true particle continues, is a merged blob involved in the next true point (yes / no) against the base rate (38 %) | Informative only |

B5 (per-frame exact selection) is dropped: the stored cloud is already unique and the pool is no
longer central.

### 11.3 Build (only what the bounds support)

New module `src/openptv2/joint_tracking.py`, plugin `src/openptv2/plugins/joint_tracking.py`,
name `joint` in `plugins/loader.py` and `tracking_registry.py`. Python first. Output in the usual
run-store layout (`write_correspondences`, `write_linkage_many`), cam ids may repeat for a merged
blob. `trackcorr`, `two_phase`, `unified` are not changed.

Reuse, do not re-implement:
- linking, ghost rules, confirmation: `plugins/two_phase_tracking.py`
  (`_match_two_phase_frame`, `ghost_probabilities`, `_merge_bidirectional_links`) and
  `tracking_postprocess` (`confirm_link_tuples`, `kink_statistics`);
- prediction and claims: `track_assisted._predict_position`, `_claim_particle`,
  `_triangulate_weighted_lsq` (already supports a lower weight for shared blobs);
- leftover matching: `correspondences.match_frame_correspondences(..., return_rcm=True)`;
- cleanup: `ghost_ilp.trajectory_ilp_uniqueness`, `temporal_smoothing.stitch_tracklets_irls`,
  `reconnect.reconnect_pieces`, `smoothness_filter`.

Stages, one change each, each measured on the cases against `two_phase`:
- **J1 Merge-aware re-triangulation only.** Run `two_phase` as it is; then, per frame, project the
  linked points' predictions, mark merged blobs, and recompute the position of every linked point
  that uses a merged blob (clean cameras, or de-merged centre with lower weight). No link changes.
  This isolates the accuracy gain of points 2–4.
- **J2 Claims through overlaps.** Tracks that end while their prediction falls on a blob used by
  another track claim it as shared and continue (points 3–5); camera counts per B3.
- **J3 Tracks first, leftovers second.** The frame loop of 11.1 (points 1, 6, 7): confirmed tracks
  take blobs, the matcher runs on the leftovers, newborns on probation.
- **J4 Backward sweep and iteration; final cleanup.**

Rules: revert a stage that gains only by dropping true points; one stage per commit; numbers into
section 12; pure-Python fallback tests and a Cython rebuild only if `algorithms/` is touched
(not expected before J4).

### 11.4 Then

Real wp2 (300 frames) and lv_multi on scratch copies (`scripts/real_retrack.py`,
`scripts/realism_metrics.py`); profile; docs (`docs/trackers.md`).

### 11.5 How to score (changed)

`synth_bench.evaluate_pred` averages the velocity error over whatever points a tracker outputs,
so a tracker that keeps more (harder) points looks worse. Add to it:
- `subset`: an optional set of truth keys `(particle_id, frame)`; when given, also report
  `vel_err_rel_subset`, the velocity error over output points matched to those keys. Use the keys
  matched by plain `two_phase` on the same case (the **common set**).
- report `pos_err_mm` for the common set and for the extra points separately.

Success for `joint` on sparse, ×2, ×4:
- common-set velocity error not higher than `two_phase`;
- long-track completeness higher and pieces per long track lower;
- true points kept (yield) higher; position error of the extra points below 2× the common-set error;
- ghost share not higher.
On real data: more tracks of ≥ 50 frames, roughness not higher.

### 11.6 Notes for whoever implements this

- Work in the worktree `/Users/alex/Documents/Github/openptv2-joint` (branch
  `feat/zarr-only-remove-legacy`). The main checkout `/Users/alex/Documents/Github/openptv2` is on
  `main` with someone else's untracked files; do not switch it. The worktree has its own
  environment (`uv sync --extra dev --python 3.13` and `uv run python setup.py build_ext --inplace`
  were run on 2026-11-19; repeat the build after changing anything in `algorithms/`; the main
  checkout's environment imports the main checkout's sources, not the worktree's).
  Gitignored test data exists only in the main checkout (e.g. `test_data/test_cavity/run.zarr`).
- Benchmark data: `~/Downloads/HiDImaging/tracker-bench-2026-09-29` (`cases/`, `runs/`); source
  truth `~/Downloads/HiDImaging/Synthetic-CompleteTest`. Paths are constants at the top of
  `scripts/synth_bench.py`. Never write into the real runs
  (`~/Downloads/HiDImaging/CompleteTest-e2e-local`, `openptv-cloud/examples/lv-multi`).
- `scripts/joint_bounds.py` has `case_setup`, `raw_targets`, `pool_frame`, `purity` (identity
  match of rows to truth; a row is pure for every particle whose own blobs it uses),
  `triangulate`, `b1`, `b2`, `save`. Add each bound as a sub-command.
- Timing on this machine: `two_phase` 4–5 s per case, `trackcorr` 45–95 s, `unified` 70–110 s,
  `eval` about a minute per tracker; sweeps with `-j 3`. zsh: pass lists as arrays.
- Positions carry a calibration bias (about −0.17 mm in depth); subtract the median offset before
  comparing with truth, as `evaluate_pred` does.
- Earlier findings that still hold: soft rules beat deletion; link-cost weights have no effect;
  never tune a rule on particle speed; 2–3-camera adds without a check hurt trackcorr badly.

## 12. After merging `feat/ilp-temporal-smoothing-impl` (2026-11-19)

The merge (`5139900d`) brought in a parallel line of work on ghost suppression. What it added,
what it does on the truth benchmark, and what it changes in this plan. Raw numbers:
`bench/step23_joint_bounds_2026-11.json` (`merged_unified`, `merged_ablation_sparse`); run labels
in `runs/` carry `+merged=1` / `+m=1`.

### 12.1 What came in

| Module | What it does | Default in the merged code |
|---|---|---|
| `trajectory_stitch.stitch_trajectories` | after tracking: merge fragments across a gap, split at kinks of ≥ 60° | on (`stitch: auto`), epipolar engine only |
| `temporal_smoothing.trajectory_level_ghost_filter` (+ `extract_trajectories`, velocity statistics `vel_mean`/`vel_std`, point-level smoothing) | drop or trim trajectories by smoothness and velocity statistics | on when `resolve_smoothing_config` says so |
| `epipolar_rerank.rerank_trajectories` | soft 2D-consistency score (camera count, rcm) over trajectories | on (`epipolar_rerank: auto`) |
| `mht_linker.mht_relink` | sliding-window Hungarian re-linking | **off** (the author: over-merges) |
| `frame_quality` | find frames with few correspondences, interpolate them | on |
| `calibration_aware` | rcm normalisation by distance from the centre, multi-cycle consistency, per-region threshold | modules only, partly wired |
| hybrid `trackcorr` | 3D priority cascade (track3d kernel) plus epipolar soft score `w_epi` (rcm-based) in the kernel; runs forward + backward + postprocess | default engine of `trackcorr` (`trackcorr_engine: 3d_hybrid`; the old engine is `epipolar`) |
| `unified` engine choice | dense (ghost rate > 0.2 or density ≥ 500) → 3D engine, else epipolar; corrective passes 2; per-frame ILP removed | on |

`docs/software_map.md` and `docs/agent_memory_ilp_tracker.md` describe this line of work. Their
benchmark is the scenario scorecard (`tests/benchmarks`), not the truth benchmark used here; their
claim "priority_segment_3d is better for dense scenes" is not tested on the truth benchmark.

### 12.2 Re-baseline on the truth benchmark (sparse case, 200 frames; vErr / link precision / link recall* / ghost / long completeness / pieces per long track / yield)

| Tracker | vErr | lnkP | lnkR* | ghost | longC | frag | yield | time |
|---|---|---|---|---|---|---|---|---|
| oracle | 0.1435 | 1.000 | 1.000 | 0.5 % | 0.556 | 1.83 | 0.691 | 7 s |
| `two_phase` | **0.1825** | 0.961 | 0.957 | 3.8 % | 0.503 | 2.29 | 0.680 | 4 s |
| `trackcorr`, old engine (`trackcorr_engine=epipolar`) | 0.2968 | 0.922 | 0.878 | 7.3 % | 0.439 | 2.99 | 0.684 | 52 s |
| **`trackcorr`, hybrid (new default)** | 0.2648 | 0.921 | **0.999** | 7.4 % | **0.556** | **1.84** | 0.694 | 37 s |
| `unified` before the merge (= merged with `stitch=false`) | 0.5051 | 0.895 | 0.938 | 9.1 % | 0.605 | 1.76 | 0.745 | 108 s |
| `unified` merged (default) | 1.5171 | 0.913 | 0.660 | 9.4 % | 0.387 | 4.45 | 0.731 | 198 s |

Merged `unified` at the other densities (vErr / lnkR* / frag): ×2 2.834 / 0.633 / 3.82 (was
0.877 / 0.926 / 2.30); ×4 5.266 / 0.603 / 2.31 (was 2.079 / 0.929 / 2.04); with `peeling=false`
even worse (1.96, 3.43, 8.33). Time 2–4× longer.

Findings:
1. **The regression of the merged `unified` is `stitch_trajectories` alone.** Ablation on the sparse
   case: `stitch=false` reproduces the pre-merge numbers to all digits; `temporal_smoothing=false`
   and `epipolar_rerank=false` change nothing (identical output: they are no-ops or not triggered
   here). The stitch pass raises acceleration error from 30 to 177 and cuts link recall from 0.94
   to 0.66: it makes wrong joins and wrong kink splits. It costs about 90 s.
2. **The hybrid `trackcorr` is a real step forward in continuity:** link recall 0.88 → 0.999, pieces
   per long track 2.99 → 1.84 (better than `two_phase`'s 2.29), long completeness 0.556 (equal to the
   perfect linker's), velocity error 0.297 → 0.265. It still keeps the ghosts (7.4 % against 3.8 %
   for `two_phase`) and has a higher velocity error than `two_phase`.
3. **Two baselines now, not one:** `two_phase` is best on accuracy and ghost share; hybrid `trackcorr`
   is best on continuity. A joint tracker has to be compared with both.
4. The B0 table in section 10 stays valid for the pre-merge code; rows for `trackcorr` and `unified`
   there are superseded by the table above for the merged code.

### 12.3 Consequences for the plan

- **Linking core.** Section 11.3 starts from `two_phase`. The hybrid `trackcorr` links nearly every
  reachable pair, `two_phase` rejects ghosts better. Add stage **J0**: apply the merge-aware
  re-triangulation of J1 as a post-step to *both* trackers and keep the one that scores better on
  the common set; decide the linking core of J2–J4 from that. Run J1 on `two_phase` and on
  hybrid `trackcorr` output.
- **Reuse list (11.3), changed:**
  - do **not** reuse `trajectory_stitch.stitch_trajectories` as it is (harmful on this benchmark);
    keep `temporal_smoothing.stitch_tracklets_irls` and `reconnect.reconnect_pieces`;
  - `temporal_smoothing.trajectory_level_ghost_filter` and `epipolar_rerank` showed no effect here;
    use them only if a later stage shows they help;
  - candidates worth testing in J3/J4: `frame_quality` (low-correspondence frames) and
    `calibration_aware.normalize_rcm` as an input to the ghost mark (it matches the earlier
    finding that rcm grows with distance from the centre);
  - the kernel's `w_epi` soft score uses the stored `rcm/` array; `rcm/` is not updated after
    later rewrites of `correspondences/` (see section 2), so a merged-blob correction must write
    `rcm/` again if the hybrid engine is used on its output.
- **Scoring (11.5):** the comparison set is `two_phase`, hybrid `trackcorr` and `unified+stitch=false`
  (not the merged `unified` default). Success additionally: long completeness not below the hybrid
  `trackcorr`'s 0.556 on the sparse case.
- **Decisions for the user (not part of this plan's work):**
  1. `stitch` should not be on by default in `unified` (finding 1); fixing `stitch_trajectories` is a
     separate task: its merge test accepts joins that raise acceleration error sixfold.
  2. `peeling` default in `unified` (section 10, finding 4) is still open.
  3. `docs/software_map.md` says dense data should use the 3D engine and sparse the epipolar one;
     that has not been checked on the truth benchmark (only sparse/×2/×4 L1 cases exist here;
     B0 and 12.2 show `two_phase` ahead of both at every density).

### 12.4 State of the repository

- Merge commit `5139900d` is on `feat/zarr-only-remove-legacy` and on its origin (committed and
  pushed by someone else while the merge was being finished). Conflicts were resolved in
  `unified_tracking.py` (both post-processing chains kept: IRLS stitching, trajectory ILP,
  smoothing, then the other branch's ghost filter, re-rank, stitch, MHT), `temporal_smoothing.py`,
  `ghost_ilp.py` (our `trajectory_ilp_uniqueness`, auto thresholds ghost rate > 0.15 or density ≥ 500
  kept), `track3d` kernels (`w_epi`), `gui/ptv.py` and `run_store.py` (Zarr-only versions).
- Unit tests after the merge: 1975 passed, 3 failed, 95 skipped (excluding slow, gui,
  `test_determinism.py`). The failures are not caused by the merge: two in
  `tests/unit/test_cython_epipolar_direction.py` (the default `trackcorr` engine is the hybrid, which
  runs the 3D engine and the backward pass; the tests assume the old forward-only epipolar run) and
  `test_sequence_triangulation_method.py` (needs the gitignored `test_data/test_cavity/run.zarr`,
  which only the main checkout has). The two direction tests should be updated to the hybrid default
  by whoever owns that change.
- Not run after the merge: the benchmark tests (`tests/benchmarks`), the clustered case with the merged
  code, the pure-Python fallback tests.

## 13. Where the error comes from: bottleneck analysis (2026-11-19)

Goal restated: improve the combined stereo-matching + tracking result (accurate velocities, few
ghosts, long trajectories, many particles kept) with **no constraint on the order** of matching and
tracking. The question of this section: which stage limits the result, measured with truth? Method: an
**error-budget ladder**. Each rung gives the perfect linker (`oracle`) a different input, so the
difference between rungs is the cost of one thing. Cases: sparse (`L1_d1_c0_k1_n200`) and ×4
(`L1_d4_c0_k1_n150`), both `_s0.08_a0.18_f0.05`. Raw numbers: `bench/step23_joint_bounds_2026-11.json`
(`error_budget_ladder`, `M3_*`, `B3_*`). Scripts: `scripts/joint_bounds.py` (`ladder`, `demerge`, `m3`, `b3`).

### 13.1 The ladder (velocity error vErr; yield = share of true points output; longC = long-track completeness; frag = pieces per long track)

Sparse case:

| Rung | Input to the linker | vErr | yield | longC | frag |
|---|---|---|---|---|---|
| L0 | exact projections, no noise, identity matching | **0.0009** | 0.995 | 0.973 | 1.16 |
| E1c σ=0.1 px | perfect association, every merged blob replaced by each particle's own centre ± 0.1 px | 0.0436 | 0.825 | 0.824 | 0.91 |
| E1c σ=0.3 px | same, ± 0.3 px | 0.0634 | 0.824 | 0.822 | 0.91 |
| E1c σ=0.6 px | same, ± 0.6 px | 0.1060 | 0.790 | 0.786 | 0.92 |
| E1a | perfect association, **clean blobs only** (≥ 3 cameras) | 0.1034 | 0.573 | 0.422 | 2.29 |
| greedy cloud | what the matcher gives (with ghosts), perfect linker | 0.1435 | 0.691 | 0.556 | 1.83 |
| E1c σ=1.0 px | same, ± 1.0 px | 0.2090 | 0.703 | 0.642 | 1.29 |
| E1b | perfect association, **all detected blobs, merged ones used as they are** | 0.2297 | 0.726 | 0.588 | 1.79 |
| `two_phase` | greedy cloud, real linker | **0.1825** | 0.680 | 0.503 | 2.29 |
| hybrid `trackcorr` | greedy cloud, real linker | 0.2648 | 0.694 | 0.556 | 1.84 |

×4 density:

| Rung | vErr | yield | longC | frag |
|---|---|---|---|---|
| L0 | 0.0009 | 0.996 | 0.980 | 1.11 |
| E1c σ=0.1 | 0.0589 | 0.857 | 0.850 | 1.03 |
| E1c σ=0.3 | 0.1311 | 0.850 | 0.843 | 1.02 |
| E1a (clean blobs only) | 0.2559 | 0.143 | 0.111 | 0.90 |
| E1c σ=0.6 | 0.2590 | 0.739 | 0.717 | 1.08 |
| `two_phase` | **0.4091** | 0.225 | 0.157 | 1.48 |
| E1c σ=1.0 | 0.5180 | 0.571 | 0.409 | 2.22 |
| greedy cloud, perfect linker | 0.5675 | 0.258 | 0.183 | 1.61 |
| E1b (merged blobs as they are) | 0.8676 | 0.462 | 0.275 | 2.83 |

(vErr is averaged over each rung's own output points, so rungs with different yield are not exactly
comparable; the yield column says how much of that is selection.)

### 13.2 What the ladder says

1. **The estimator and the tracker are not the limit.** With exact data the whole chain is accurate
   to 0.0009 (L0), and `two_phase` links L0 almost as well as the perfect linker. Everything above
   that is the noise of the measurements and the association.
2. **The measurement is the bottleneck, and the measurement problem is merged blobs.** Of all blob
   observations of particles seen by ≥ 3 cameras, **26 % (sparse) and 66 % (×4) are merged blobs**
   (two or more particles in one blob, with a shifted centre). Using them as they are (E1b) is
   2–3× worse than the greedy cloud; leaving them out (E1a) is better in accuracy but throws away
   43 % (sparse) and 86 % (×4) of the particles.
3. **The ceiling of correcting merged blobs is large.** If each particle in a merged blob gets its own
   centre to within 0.3 px (E1c): vErr 0.063 and 0.131, which is **2.9× and 3.1× lower than `two_phase`**,
   while 82–85 % of the particles stay (against 68 % and 22 %), long-track completeness 0.82 and
   0.84 (against 0.50 and 0.16) and about one piece per long track (against 2.3 and 1.5). This
   ceiling beats `two_phase` as long as the de-merge error stays below about 0.9 px (sparse) and 0.75 px (×4).
4. **A track can predict a blob centre well enough** (B3, sparse, clean particles, linear
   extrapolation from 2 positions): 0.27 px RMS, median 0.21, 90 % below 0.40, 99 % below 0.62.
   Same at ×4: 0.28 px RMS (median 0.23, 99 % below 0.65). A quadratic extrapolation from 3
   positions is *worse* (0.39 px sparse, 0.43 px ×4; 99 % below 1.0 px): it amplifies the noise,
   as in the earlier reconnect finding, so use linear prediction (or a two-sided smoothed position
   in a post-step). A de-merge error of 0.3–0.5 px is therefore plausible, well inside the break-even.
5. **Association (matching + linking) is the second, smaller bottleneck.** Sparse: greedy cloud with
   a perfect linker 0.1435, `two_phase` 0.1825: association costs 0.039 (21 % of the error);
   removing every ghost from the input gains `two_phase` only 0.005–0.018 (section 10). Hybrid
   `trackcorr` has the best continuity but keeps 7.4 % ghosts and has the higher error.
6. **Depth noise is real but is not a lever.** The depth axis carries most of the velocity error
   (RMS 26 mm/s against 4–5 mm/s in-plane for clean particles, sparse, 150 particles). Fitting one
   trajectory directly to all blob centres (image space, `m3`) instead of fitting 3D points
   improves it by only 5 % (0.0501 → 0.0476), 8 % at ×4 (0.0668 → 0.0615) and 3 % with merged
   blobs (0.277 → 0.269). Merged blobs make the same particles 5.5× worse (0.050 → 0.277). So the
   per-frame triangulation + polynomial smoothing is already close to the information in the
   blobs; improving the blobs is what pays. (The ×4 run with merged blobs gave no result.)
7. **Speed.** The matcher costs 0.10 s per frame (sparse, 1150 points) and 0.44 s per frame (×4,
   2560 points); `two_phase` links at 0.02–0.03 s per frame, hybrid `trackcorr` 0.19, merged
   `unified` 0.5–1.0. So the matcher is the cost of the fast pipeline. Any scheme that runs the
   matcher more than once per frame, or builds a large candidate pool, multiplies the dominant cost.
   De-merging is cheap (a projection and a small solve per merged blob).

### 13.3 Levers that the measurements rule out (do not spend time here)
- A candidate pool / perfect selection from it (B1, B2): recovers part of the lost points, but the
  recovered ones are the merged-blob points and they are noisy; the same points are better served by correcting the blob.
- Image-space trajectory fitting as such (3–8 %).
- Ghost removal on `two_phase` (0.005–0.018). (Ghosts do matter for hybrid `trackcorr`: 7.4 %.)
- Cost weights, wider gates, re-matching freed blobs (earlier findings, `2026-09-30-tracker-plan.md`).
- `trajectory_stitch.stitch_trajectories` in its present form (section 12).

### 13.4 Caveats (read before trusting the ceiling)
- The merge model is the benchmark's (`synth_bench._target_rows`, calibrated to reproduce the image
  pipeline's 2D recall to ~2 %); **whether real recordings have as many merged blobs is not
  measured (bound M0)**. Everything in 13.2 points 2–3 depends on it.
- E1c assumes the particles in each merged blob are known, and that each gets a centre with
  independent error σ. A real implementation must find the particles from tracks, which exist
  only after the particle was tracked before or after the merge (cold start, long overlaps), and must
  split the blob between them using brightness. The ceiling is for the case where this works.
- B3 uses clean particles with perfectly associated history; real histories contain errors.

## 14. Current plan (2026-11-19, after the bottleneck analysis): merge-aware joint tracker

Replaces sections 11.1–11.3 where they differ; 11.5 (scoring) and 12 (merged branch,
baselines) still apply.

### 14.1 Principle (no fixed order)
The information must flow both ways. Detections → matcher → 3D points → tracks is the old direction.
The new one: **tracks → which blobs are merged and where each particle sits inside them → better
3D points and particles the matcher could not form → better tracks.** It is one loop, started from
the existing matcher and linker, iterated until stable, and it can run forward in time, backward, or
both. The matcher still runs once per frame (it is the cost); no candidate pool.

### 14.2 The loop
1. **Start:** greedy cloud from the existing matcher; link with the better of `two_phase` and hybrid
   `trackcorr` (decided by J0 below). These tracks are the first predictions.
2. **Track-informed measurement correction (J1).** For every track point and every camera, project
   the track's predicted position (two-sided smoothed position when run as a post-step: its error is
   smaller than the causal prediction's 0.27 px). Several predictions (of different particles) on one
   blob, or a blob much brighter or larger than its track's usual (brightness is persistent per
   particle and camera: linked pairs change it by a median 0.06), mark the blob as merged. The blob has
   one measured centre `c`, the brightness-weighted mean of the spots' centres `s_j = p_j + δ_j`
   (`p_j` predicted). The minimum-norm solution of "weighted mean of `s_j` equals `c`" is
   `δ_j = w_j r / Σ w_k²` with `r = c − Σ w_j p_j` and `w_j` the brightness shares. Each particle gets
   the pixel `s_j`, with a larger variance than a clean blob. Then re-triangulate (least squares, per-camera
   weights: `track_assisted._triangulate_weighted_lsq`); when ≥ 2 clean cameras remain and the
   merged ones disagree, drop the merged ones.
3. **Recovery (J2).** A particle whose blob was merged is often missing from the cloud (the matcher
   could not form it). Tracks that end or have a gap, and whose prediction falls on a merged blob,
   take the blob with the same split (a claim in blob space, `track_assisted._claim_particle` with
   shared blobs), add the point and continue. New particles in merged blobs have no track: they enter
   from the leftover blobs after the tracks have taken theirs (existing matcher on the reduced target
   lists), on probation (smoothness for N frames, dead-end rule, taking blobs through the past at promotion).
4. **Iterate (J3).** Better positions give better predictions and better links; repeat 2–3 with a
   forward and a backward sweep until the set of tracks is stable (expected 2–3 rounds).
5. **Final cleanup** with what exists and has been measured to help: `reconnect.reconnect_pieces`,
   `smoothness_filter`, `quality_post` trimming, IRLS stitching. Not `stitch_trajectories` (section 12).

### 14.3 Gate and bounds
- **M0 (gate, do first):** the real merge rate. No truth on real data, so a proxy on tracked output:
  project every linked 3D point into each camera and count the share of points whose projection lies
  within r = 1, 1.5, 2, 3 px of another point's projection; also the share of 3-camera points whose
  missing-camera projection falls within r of a blob used by another point. Compute on real wp2
  (`~/Downloads/HiDImaging/CompleteTest-e2e-local/wp2/test/res/run.zarr`, read-only, 100 frames) and on
  the benchmark's `two_phase` output (same numbers on the case with known truth). **If the real share is
  below a third of the benchmark's, stop and report**: the ceiling in 13.2 would not apply.
- B3: done (linear 0.27 / 0.28 px, quadratic 0.39 / 0.43 px; sparse / ×4).
- Done: B0, B1, B1b, B2, the ladder (L0, E1a, E1b), M2 (E1c) for σ = 0.1, 0.3, 0.6, 1.0, M3, B3 (sparse,
  order 2), timing. Dropped: B4, B5, B6 (the pool and the probation length are only needed in J2; B6
  moves into J2), M1.

### 14.4 Stages (one change each, each measured on sparse, ×2, ×4 against `two_phase`, hybrid `trackcorr` and `unified+stitch=false`)

| Stage | What | Expected (from 13.1, not a promise) | Stop / go |
|---|---|---|---|
| J0 | Which linker first. Run J1 on the output of both `two_phase` and hybrid `trackcorr` | choose the one with the lower common-set vErr after J1 | decides the core of J2–J4 |
| J1 | Track-informed measurement correction as a **post-step** on a finished run (no change to links, no new points): `src/openptv2/blob_demerge.py` (find merged blobs from predicted pixel positions; minimum-norm offsets with brightness weights; weights) and a post step `merge_aware_refine(store, linkage_name, ...)` that rewrites correspondences, positions and `rcm/` | vErr sparse 0.18 → ~0.10–0.12 (E1c σ=0.6 is 0.106, σ=0.3 is 0.063); ×4 0.41 → 0.20–0.26 | go to J2 if common-set vErr falls ≥ 30 %; if < 15 %, stop and re-measure the de-merge error against B3 |
| J2 | Recovery: tracks claim merged blobs; leftover matching and probation for newborns | yield sparse 0.68 → 0.75–0.80; ×4 0.23 → 0.5–0.8 (ceiling 0.82 / 0.85); longC up | go if the extra points' position error is below 2× the common-set error |
| J3 | Forward + backward rounds until stable | frag 2.3 → ~1.3 | |
| J4 | Real data: wp2 (300 frames), lv_multi; speed; docs | more tracks of ≥ 50 frames, roughness not higher | |

Success (as 11.5): on sparse, ×2, ×4 common-set vErr not above `two_phase`'s (target: ≥ 30 % below),
yield and long completeness higher, ghost share not higher, position error of extra points below 2×
the common-set error; time of the correction below the matcher's.

### 14.5 Implementation notes for the next model
- Use the worktree `/Users/alex/Documents/Github/openptv2-joint` (section 11.6). `scripts/joint_bounds.py`
  has the helpers: `case_setup`, `raw_targets`, `triangulate`, `shared_mask` (truth: blobs shared by
  several particles), `purity`, and the sub-commands `ladder`, `demerge`, `m3`, `b3`, `b1`, `b2`. The
  truth blob map is `truth.npz['blob_tid']`.
- The correction needs, per output point, the blob (camera → target index): `correspondences/` has it
  (`cam_target_ids`); the blob's brightness is `sumg` (column 6) and its pixel count `n` (column 3) of
  `targets/cam_*/frame_*`. Positions of tracks: `linkage/<name>/` (prev, next, pos).
- Write `rcm/` again after changing positions (the hybrid engine and the ghost marks read it), and keep
  row order of `correspondences/` equal to the linkage rows.
- Test on the benchmark with a variant label, e.g. `two_phase` then the post step, and check with
  `synth_bench eval`; the truth check of the correction itself: compare the corrected blob centres with
  `render` projections as in `demerge_inputs`.
- Cython is not needed until J3 (the loop is per blob, cheap); do not touch `correspondences()`.

## 15. Reframing (2026-11-19): one space–time–camera inference, not a pipeline

### 15.1 The assumption that was wrong
Detection → stereo matching → tracking assumes every blob is one particle, every stage is decided
once, and each stage only passes its result on. Section 13 showed that the first assumption fails
for 26 % (sparse) to 66 % (×4) of the blob observations, and that this, not the linker, limits the
accuracy. The deeper point: **the image plane, the 3D position and time are one object.** A blob
is evidence about particles that exist in 3D and move through time; and what a blob contains is
visible only from its context:
- **in time:** particles enter and leave a blob; a particle that was alone in the blob before and
  after is inside it now;
- **across cameras:** a 3D particle seen cleanly by 2–3 cameras predicts where it must be in the
  others; a blob that is larger/brighter or off-centre there is a merged one;
- **in appearance:** each particle has a persistent brightness per camera; a blob's grey sum is
  (about) the sum of its particles' brightnesses, so the count is readable.
A single frame knows none of this; it can only learn "what a typical particle image looks like".

### 15.2 What the data say about the three cues (truth-based, 2026-11-19)
Scripts: `scripts/joint_info_coverage.py` (coverage), the appearance numbers from the rendered
frames of `Synthetic-CompleteTest` (openptv2's own detections, truth from the render).

**Can a merged observation be resolved? (share of merged observations)**

| Cue | sparse | ×4 |
|---|---|---|
| same frame: the particle is clean (single) in ≥ 2 other cameras | 64 % | 21 % |
| same camera: the particle was alone in its blob within ± 3 frames | 31 % | 18 % |
| 3D view: the particle has ≥ 2 clean cameras (a 3D position) at the same or a frame within ± 3 | **76 %** | 38 % |
| … within ± 6 frames | 80 % | 48 % |
| … clean on both sides (interpolation, not extrapolation), ± 3 | 64 % | 20 % |
| not resolvable by any of these within ± 6 | 20 % | 52 % |

- Per-camera history alone is weak (31 %): overlapping particles stay overlapped for several
  frames. The 3D view (any camera, any nearby frame) is much stronger (76 %): this is the argument
  for treating camera, space and time together.
- At ×4 more than half of the merged observations have no clean 3D moment within ±6 frames. There
  the cues must be combined over a whole cluster of particles and frames (joint hypotheses), and the
  method meets the known density limit of blob-based matching.

**Can the blob itself say it is merged? (single vs merged, rendered frames, blobs with ≥ 1 particle)**

| Information | AUC |
|---|---|
| the blob alone: pixel count / grey sum | 0.78 / 0.73 |
| the blob **and the particle's own typical appearance** (grey sum / its median over its alone-in-blob frames) | **0.95** |

With the context ratio: a single particle's grey sum stays within 0.90–1.11 of its typical value
(5–95 %); merged blobs have a median ratio 1.99 (two particles). Threshold 1.25: finds 74 % of the
merged observations, 96 % of the alarms are right, 1.1 % false alarms among single blobs; threshold
1.5: 62 % found, 99 % right. (Truth identities were used to define "the particle"; real data needs a track.
Real linked pairs change blob brightness per camera by a median 0.06 on wp2, so the signature exists there.)

**Does the blob's shape change more when it is merged?** (rendered frames 100–130, intensity-weighted
second moments of the blob's pixels; scripts `scripts/joint_blob_shape.py`, `joint_blob_context.py`)
Yes, but weakly. Same particle and camera, the blob at frame `f` and `f+L`; both single against both merged:

| Lag | minor-axis variance, median change single / merged | elongation, single / merged | grey sum, single / merged | AUC (minor variance / grey sum) |
|---|---|---|---|---|
| 1 frame | 0.048 / 0.074 | 0.098 / 0.143 | 0.023 / 0.025 | 0.57 / 0.53 |
| 3 frames | 0.065 / 0.139 | 0.164 / 0.260 | 0.027 / 0.040 | 0.61 / 0.59 |
| 6 frames | 0.078 / 0.204 | 0.235 / 0.349 | 0.030 / 0.056 | 0.62 / 0.64 |

- Merged blobs do change 1.5–2.6× more, and the gap grows with the lag (partners drift apart, arrive
  or leave). But single blobs also change a lot (pixel noise, thresholding, neighbours), so as a stand-alone
  test the shape change is weak (AUC 0.55–0.62). With blob attributes only (`n`, `nx`, `ny`) it is
  invisible at one frame (AUC 0.47–0.57): extents move in whole pixels.
- Static shape is also only moderate: AUC 0.67–0.71 (variances), 0.73 (low peakiness); relative to
  the particle's own typical shape no better (0.44–0.66). **Brightness mass is the strong appearance
  cue (0.95), shape is a weak supporting one.**
- Arrivals and departures show clearly in the grey sum: single → merged, next/previous ratio median 1.48
  (5–95 %: 0.96–9.3); merged → single median 0.68 (0.10–1.03), against ±2–3 % for an unchanged blob.
  Merged episodes are long: median 5 frames, mean 8, 90 % up to 19 frames.
- Use: weak cues add up when combined. Each camera and each frame contributes a small log-likelihood
  for "this blob holds m particles"; four cameras times several frames, tied together by the same
  particles' 3D track, is the point of one joint model. Shape terms go into the model as soft evidence,
  not as a classifier on their own.

### 15.3 The model
Latent: particles `k` with smooth 3D trajectories `X_k(t)` and per-camera brightness `a_{k,c}`
(persistent, slowly varying). Observations: per camera and frame, blobs with centre, pixel count,
extent and grey sum. Blob model: a blob holds `m ≥ 1` spots; its centre is the brightness-weighted
mean of the spots' centres, its grey sum the sum of their brightnesses, its size grows with `m` and
the spots' separation. Detection → matching → tracking is a greedy coordinate ascent on this model with
the wrong `m = 1`. The better procedure is an **order-free loop** (EM / coordinate ascent):
1. **Explain blob mass.** For each blob: the tracked particles that project into it and their
   brightness `a_{k,c}` explain a part of its grey sum and shift its centre; the split of the
   centre uses the minimum-norm rule of section 14.2 point 2.
2. **Update** trajectories and brightness from the de-merged measurements (re-triangulation with
   per-camera weights; clean cameras dominate).
3. **Look at the residual.** Grey sum not explained by the tracked particles in a blob, in a camera
   where the 3D evidence says there should be exactly one more particle (or two cameras see a clean
   particle that is absent here), is a **birth hypothesis**; a track whose blob is empty or too weak
   is an **occlusion** (coast) or a death. Births and deaths are what enter/leave events are.
4. **Repeat** in any order, forward and backward in time, until stable.
The matcher keeps its role as a cheap generator of first hypotheses (once per frame), not as the
authority.

### 15.4 What changes in the plan (section 14)
- **J1 becomes the blob-content model, not just a re-triangulation:** (a) per track and camera a
  brightness signature `a_{k,c}` (median over alone-in-blob frames, updated); (b) mass accounting
  to count the members of a blob and to flag merged blobs (the 0.95 AUC cue) instead of using only
  "two predictions on one blob"; (c) the minimum-norm centre split; (d) camera selection by the
  cross-camera evidence (use clean cameras when ≥ 2 exist: 64 % of the merged observations at sparse density).
- **J2 (recovery) is the birth/death step of 15.3:** unexplained blob mass + cross-camera evidence
  create a particle; the matcher's leftovers are one source of hypotheses.
- **A new first step, J1-oracle:** run the J1 algorithm on **perfectly associated** data (the
  ladder inputs: truth identities, merged blobs left as they are, E1b) and compare with the ceilings
  E1c (0.063 sparse, 0.131 ×4) and E1b (0.230 / 0.868). It measures how much of the ceiling the
  estimation captures without the association problem. Then replace truth identities by tracks
  (`two_phase`, hybrid `trackcorr`) and measure the loss of association (J1 proper).
- **×4 needs joint hypotheses over clusters** (more than half of merged observations lack a clean 3D
  moment within ± 6 frames). Plan it as stage J3b after J1–J3 are measured, not before.
- **M0 stays the gate** (real merge rate), now with a second part: the stability of real blob
  brightness per particle (the appearance cue) on wp2 tracks: median |Δ log sumg| of linked pairs
  and its 95th percentile.

### 15.5 Open decision: blobs only, or patches?
The plan keeps the earlier rule "target space only, no image subtraction". Everything above uses
only blob attributes (centre, pixel count, extent, grey sum). The pixels of a merged blob carry
more (its shape, the positions of the peaks); fitting 2–3 known-brightness spots to the small
patch, with the tracks supplying the start positions, would sharpen the split. It is local to
the merged blobs (no full-image OTF). Decision: first measure with blob attributes only (J1); open
the patch option only if J1 stays well above the E1c ceiling and the shortfall comes from the split,
not from the association.
