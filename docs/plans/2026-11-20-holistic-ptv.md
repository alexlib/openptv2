# Holistic 3D-PTV: one inference over cameras, space and time

Date: 2026-11-20. Branch: `feat/zarr-only-remove-legacy`. Status: approved; implementation follows the stages below
(progress log at the end).
The measurements it rests on are in `docs/plans/2026-11-19-joint-tracker.md`, sections 10-16 (lab record: bounds,
error-budget ladder, merged-branch baselines, cue measurements, M0 gate).
Scripts: `scripts/joint_bounds.py`, `joint_info_coverage.py`, `joint_blob_context.py`, `joint_blob_shape.py`,
`joint_m0.py`, `synth_bench.py` (`MERGE_SCALE`). Numbers: `bench/step23_joint_bounds_2026-11.json`.
Work in the worktree `/Users/alex/Documents/Github/openptv2-joint` (own environment, Python 3.13, Cython built).

## Context

openptv2 runs detection → stereo matching → tracking, each decided once, each blind to the
others. We measured where that loses (truth benchmark, 2026-11-19):

- **The chain itself is fine:** with exact data it is accurate to 0.0009 (velocity error).
- **The limit is the measurement, and the measurement problem is merged blobs:** two or
  more particles in one detected blob with a shifted centre. 27 % of blob observations on
  the benchmark, 66 % at 4× density; on real wp2 roughly half to equal the benchmark's rate.
- **Ceiling if each particle in a merged blob gets its own centre to 0.3 px:** velocity
  error 0.063 against 0.183 for the best tracker today (`two_phase`), 82 % of particles
  kept against 68 %, long-track completeness 0.82 against 0.50. At real-like merge rates
  still 2× lower error, +4–7 points of particles, +8–12 points of completeness.
- **The information to do it exists, but only jointly:** per-camera history resolves 31 %
  of merged observations, other cameras 64 %, cameras + nearby frames in 3D 76 %. A blob's
  brightness against its particle's own typical brightness tells merged from single with
  AUC 0.95 (weaker on real data); shape change is a weak cue; a track predicts a blob
  centre to 0.27 px (linear; quadratic is worse).
- **Association is the other half** at real-like merge rates, and 11 % of true particles
  have no blob at all (too faint): the yield ceiling in blob space is about 0.82–0.86.
- Ruled out: candidate pool, image-space trajectory fit as such (3–8 %), ghost removal
  for `two_phase`, cost weights, `stitch_trajectories` (harmful), peeling as default.

**Goal:** a breakthrough against the sequential baselines, not an increment: one
procedure in which detection evidence, stereo geometry and motion resolve each other's
ambiguities. Decision already taken: blob attributes first; pixel patches only if the
split of merged centres is shown to be the shortfall.

## The idea in plain words

Stop asking "which blobs match?" and then "which points link?". Ask one question:
**which set of moving particles explains every blob in every camera in every frame?**

- A particle is a smooth 3D path plus a brightness in each camera (stable in time).
- A blob is explained by the particles that project into it: its brightness is the sum of
  theirs, its centre their brightness-weighted mean.
- Everything is scored in one currency (a cost):
  1. distance between each blob centre and what its particles predict;
  2. brightness of each blob against the sum of its particles' brightness;
  3. smoothness of each path (acceleration);
  4. a price for a blob nobody explains (false target), and a price for a particle that
     should be visible in a camera but has no blob (cheap when the particle is faint).
- Matching, linking, counting particles in a blob, starting and ending tracks are all the
  same kind of move: accept it if the total cost goes down. A ghost dies by itself: it
  claims brightness that real particles already account for. An overlap is not a failure:
  both particles keep their paths and share the blob.
- Order is free: the existing matcher and tracker give the first guess; then the loop
  runs forward and backward in time until nothing improves.

This is the Shake-the-Box principle (predict, then resolve the image) moved to blob space
with brightness accounting, so it stays fast and works without an optical model.

## Stages (each measured; stop rules are part of the plan)

Benchmark cases for every stage: sparse, ×2, ×4 at the default merge rate and sparse at
real-like rates (`MERGE_SCALE=0.75`, `0.6`; build 150–200-frame versions). Baselines:
`two_phase`, hybrid `trackcorr`, `unified+stitch=false`. Score on the common set of points
(below) plus yield, long-track completeness, pieces per long track, ghost share.

**H0. Foundations (small, first)**
- Observation table: per 3D point and camera the pixel position actually used, its **variance** and
  the number of particles sharing the blob. New optional arrays in `RunStore` (`storage/run_store.py`),
  written next to `correspondences/`. Consumers: link confirmation and ghost marks use the per-point
  variance (below), `point_quality.rcm_from_arrays` the corrected pixels.
  *Reviewed assumption (2026-11-20):* the first guess was that later steps recompute ray miss from the
  uncorrected blob centres, so corrected points look like ghosts. Measured: switching the ghost rules
  off changes `two_phase` on the corrected points from 0.138 to 0.141 (no effect); switching off
  its **link confirmation** as well gives 0.076 (perfect linker 0.060). So the cause is the
  confirmation tolerance (8x the median kink of the run), which cuts true links of the noisier
  corrected points. Fix to build in H2: confirmation tolerance per link from the points' own
  variances instead of one run-wide value.
- Scoring in `scripts/synth_bench.py::evaluate_pred`: `subset` (truth keys matched by
  plain `two_phase`) → velocity and position error on the common set and on the extra
  points separately.
- Truth-free score for real data: **leave-one-camera-out**. Reconstruct each particle
  from three cameras, project into the fourth, compare with the blob there
  (`scripts/joint_m0.py` grows this). Works on real wp2 and on the benchmark, so the two
  can be compared directly.

**H1. The blob model with perfect association (isolates the estimation)**
New module `src/openptv2/blob_model.py`; driver in `scripts/joint_bounds.py`. Input: truth
identities, merged blobs as detected (the `E1b` input). Implement:
- brightness signature per particle and camera (median grey sum over frames where it is
  alone in its blob; updated in the loop);
- membership and count per blob from projected paths and brightness accounting;
- centre split: for blob centre `c`, predictions `p_j`, brightness shares `w_j`:
  `δ_j = w_j·r / Σw_k²`, `r = c − Σ w_j p_j`; each particle gets `p_j + δ_j` with a
  larger variance than a clean blob;
- camera choice: clean cameras only when ≥ 2 exist, else weighted
  (`track_assisted._triangulate_weighted_lsq`);
- path refinement and repeat (2–3 rounds).
Gate: closes at least 60 % of the gap between "merged blobs as they are" and the ceiling
(sparse 0.230 → 0.063, ×4 0.868 → 0.131; real-like 0.106 → 0.060). If it falls short,
compare the centre error after the split with the 0.27 px prediction error: if equal, the
split is at its limit and the patch option opens (user decision of 2026-11-19); if not,
fix the model.

**H2. The same on real tracks, as a post-step (no link changes)**
`merge_aware_refine(store, linkage_name)` on finished `two_phase` and hybrid `trackcorr`
runs; pick the better linker as the core for H3. Gate: common-set velocity error −30 % at
the default merge rate and −20 % at `MERGE_SCALE=0.75`; under −15 %: stop and find where
association loses it.

**H3. Births, deaths and links by the same cost**
- Through an overlap: a track whose prediction lands on a shared blob keeps going
  (`track_assisted._claim_particle` with shared blobs), instead of ending.
- Cross-camera completion: a particle seen cleanly by 2–3 cameras takes its share of the
  blob in the remaining camera.
- Births from unexplained brightness and from leftover blobs (existing
  `match_frame_correspondences` on the reduced target lists; no change to
  `algorithms/correspondences.py`); newborns on probation (smooth for N frames,
  dead-end rule).
- Links: join two pieces when the joint path lowers the cost (`reconnect.reconnect_pieces`
  as the candidate generator; not `trajectory_stitch.stitch_trajectories`).
Gate: yield +10 points at the default rate and +5 at 0.75; extra points' position error
below 2× the common set's; ghost share not higher.

**H4. The full loop**
Sliding window in time, forward and backward, all moves under one cost, until stable.
For dense data (more than half of merged observations have no clean 3D moment within
±6 frames at 4×): hypotheses over a whole cluster of overlapping particles and frames
(small exact search per cluster, `ghost_ilp.solve_component_ilp` style). Plugin `holistic`
(`plugins/loader.py`, `tracking_registry.py`); hot loops to Cython only now.

**H5. Proof on images and real data**
- Rendered images, not only synthetic blob tables: the native-density render exists
  (`Synthetic-CompleteTest/syn1`); render ×2 with openptv-analysis
  `synthetic_experiment` for a dense image-level case.
- Real wp2 (300 frames, then all 1072) and lv_multi on scratch copies: leave-one-camera-out
  error, tracks ≥ 50 frames, roughness, and the Eulerian statistics against the MATLAB
  reference (`comparison_vs_legacy.json`).

## What counts as the breakthrough

| | `two_phase` today | Target |
|---|---|---|
| Sparse, default merge rate: velocity error (common set) | 0.183 | ≤ 0.09 |
| Sparse: particles kept / long-track completeness | 0.68 / 0.50 | ≥ 0.78 / ≥ 0.75 |
| Real-like merge rate (0.75): velocity error | 0.122 | ≤ 0.075 |
| 4× density: particles kept, at velocity error not above today's 0.41 | 0.22 | ≥ 0.50 |
| Real wp2: leave-one-camera-out error; tracks ≥ 50 frames | measured in H0 | clearly lower; more |
| Time per frame | matcher 0.10 s + linker 0.02 s | ≤ 3× that |

Targets are set from the measured ceilings; they are goals, not promises.

## Open points
- `unified` defaults (`stitch` on, `peeling` on) hurt on the truth benchmark; changing
  them is a separate decision for the user.
- The 11 % of particles with no blob need the images (track-guided detection with a lower
  local threshold). Outside this plan unless the user opens the image option.
- Real-data brightness is less stable than synthetic (linked pairs: 95 % within ×1.3
  against ×1.16), so thresholds come from each run's own spread.

## Verification
- Unit tests for `blob_model` (two crossing particles: counts, split, shared blob;
  brightness signature), run-store round trip of the observation table.
- `uv run pytest tests/unit -q -m "not slow and not gui"`; pure-Python fallback tests and
  a Cython rebuild if `algorithms/` is touched (H4).
- Per stage: `scripts/synth_bench.py track/eval` on the cases above; numbers appended to
  the repo plan and `bench/`; a stage that gains only by dropping true points is reverted.
- `uv run ruff check src scripts`.

## Progress log

- 2026-11-20, H0a: cause of "two_phase worse on corrected points" found: link confirmation, not ghost marks
  (sparse `_m0.75_E1c0.3`: default 0.138; ghost rules off 0.141; ghost rules and confirmation off 0.076; perfect linker 0.060).
- 2026-11-20, H0 done: `RunStore.write/read/has_observations` (group `obs`, shape (N, C, 4) = x_px, y_px, var_px2,
  n_share; tests in `tests/unit/test_run_store.py`); common-set scoring `synth_bench eval --ref two_phase`
  (`common_set_errors`, tests `test_synth_bench_common_set.py`); `src/openptv2/loco.py` (leave-one-camera-out residual,
  tests `test_loco.py`). LOCO baseline on 4-camera points, 20 frames: real wp2 median 0.34 px / p90 0.68 / rms 0.44;
  rendered synthetic 0.31 / 0.54 / 0.37 (`scripts/joint_m0.py`). Caveat: LOCO only sees points the matcher accepted, so
  strongly merged blobs (rejected by the epipolar band) are under-represented; it is a consistency check, not a
  full measure.
- 2026-11-20, H1 first result (`scripts/joint_bounds.py h1`, module `src/openptv2/blob_model.py`, tests `test_blob_model.py`):
  blob model on perfectly associated data = leave-one-out path prediction + per-particle *effective observation*
  of a shared blob (blob centre minus the other members' predictions, over its brightness share; the particle's own
  prediction never enters) + Gauss-Newton triangulation with weights 1/variance, 2 rounds. Case `_m0.75` (60 frames),
  3D error vs truth (mm), raw blob centres -> H1: all 0.143 -> 0.074; 1 merged camera 0.178 -> 0.067; 2: 0.243 -> 0.096;
  3+: 0.321 -> 0.236; clean particles unchanged (0.050). Velocity error with the perfect linker 0.0603 against 0.1061
  (merged blobs as they are, E1b) and 0.0595 (ideal centres, E1c): the gap is closed to 98 % (gate was 60 %).
  `two_phase` on the H1 input: 0.0918 (greedy cloud 0.1219), link precision 0.990, long completeness 0.775 (0.699).
  Reviewed: the variance claimed for corrected observations is about 1.8x too small (z^2 = 1.76), and the pixel
  error of the corrected centres has a heavy tail (median 0.29 px, rms 0.91 px; raw merged centre rms 0.87 px,
  prediction rms 0.36 px) coming from dim members whose weight is small. Assumptions that make this an upper
  bound: truth association, brightness signature from truth "alone" frames, centre noise var_c = 0.01 px^2 set to the
  benchmark's. Next: other cases, then H2 (real tracks).
- 2026-11-20, H1 completed and reviewed. Module `src/openptv2/merge_refine.py` (truth-free `refine_merged`), tests
  `test_merge_refine.py`, `test_blob_model.py`; benchmark driver `scripts/joint_bounds.py h1` is a thin wrapper.
  *Review finding:* the first version (all predictions trusted, fixed variance) diverged: particles merged in >= 3
  cameras got worse each round (sparse: 0.52 -> 0.92 -> 1.13 mm; x4: 0.85 -> 1.8 -> 13.7 mm) and the variance was
  2-3x too small (z^2 5-8), because poor positions gave bad predictions, which spoiled the partners' effective
  observations. The velocity-error numbers had hidden it (medians improved). *Fix:* predictions are made from frames
  with >= 2 clean cameras only and trusted only with >= 3 such frames in the window; an effective observation more
  than 4 sigma from the particle's own trusted prediction is dropped; the trusted path prediction enters the
  position fit as pseudo-observations. Now stable (rounds 2 and 3 agree), calibrated (z^2 1.1-1.2).
  Gate (perfect linker, velocity error; gap closed = (E1b - H1) / (E1b - E1c)):
  | case (60-frame dev cases) | greedy | E1b raw merged | **H1** | E1c ceiling | gap closed |
  |---|---|---|---|---|---|
  | real-like `_m0.6` | 0.061 | 0.067 | 0.050 | 0.051 | ~100 % |
  | real-like `_m0.75` | 0.095 | 0.106 | 0.060 | 0.060 | ~100 % (earlier version) |
  | default merge rate | 0.166 | 0.245 | 0.092 | 0.065 | **85 %** |
  | x4 density | 0.581 | 0.834 | 0.562 | 0.134 | **39 %** (gate 60 % not met) |
  `two_phase` on the H1 input: default rate 0.202 -> 0.105 (common set 0.175 -> 0.083), yield 0.684 -> 0.773, long
  completeness 0.60 -> 0.76, ghost 3.9 % -> 1.6 %; x4: yield 0.227 -> 0.590, long completeness 0.18 -> 0.48, but
  velocity error 0.433 -> 0.670 and link precision 0.929 -> 0.876 (the noisier extra points). By number of merged
  cameras (3D error mm, default rate, round 3): clean 0.052 -> 0.035; 1 merged camera 0.250 -> 0.060; 2: 0.353 ->
  0.161; 3+: 0.498 -> 0.506 (not helped). At x4 the 3+ group is 57 % of the particles and 49 % of predictions are
  trusted: information-limited, which is what the coverage analysis said (H4: cluster hypotheses).
  Consequences for the plan: (1) H2 must give the linkers noise-aware gating (per-link tolerance from the points'
  variances); (2) x4 targets move to H4; (3) particles with no clean camera and no trusted path should carry a
  large variance, not a confident wrong position.
- 2026-11-20, H2 first results (`scripts/joint_bounds.py h2`, `src/openptv2/merge_post.py`, tests `test_merge_post.py`):
  merge-aware refinement of real `two_phase` tracks (60-frame default-merge-rate case; membership inferred from the
  tracks: a point without a view in a camera attaches to the blob its projection falls on, radius 2 px; links
  unchanged). Velocity error 0.1909 -> 0.1608 (-16 %; common set 0.172 -> 0.143); attachment radius 1.2 / 2 / 3 px
  gives 0.164 / 0.161 / 0.167 (insensitive). Alternating loop (link on the corrected positions, then correct again
  from the new tracks): 0.1566 after one and 0.1526 after two rounds (-20 %; common set 0.1294, -25 %). Link recall
  was already 0.991 -> 0.997: linking is not the bottleneck at this merge rate; the missing yield is (0.66 against
  0.80 with perfect association): particles absent from the cloud. Plan gate for H2 (-30 %) not met by the post-step
  alone, above the stop line (-15 %); proceeding to H3 (recovery). Cost: 11 s for 69k rows (0.18 s/frame), python
  loops; to be vectorised later (matcher 0.10 s/frame).
