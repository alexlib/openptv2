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
- Observation table: per 3D point and camera the pixel position actually used, its weight
  and the number of particles sharing the blob. New optional arrays in `RunStore`
  (`storage/run_store.py`), written next to `correspondences/`. `point_quality.rcm_from_arrays`,
  `rcm/` and the hybrid kernel's `w_epi` read it when present. Reason: `two_phase` on
  corrected points scored worse than on raw ones although its links were 99.9 % right;
  the likely cause is that later steps recompute ray miss from the uncorrected blob
  centres. Verify that cause first.
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

