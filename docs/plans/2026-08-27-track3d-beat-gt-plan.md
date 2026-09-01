# Plan: make `track3d` (level-3 cascade) beat 3dptv.exe's own GT, not just match it

> **Update 2026-09-01:** After zarr-only cutover (`archive/2026-09-01-zarr-only-final-cutover-plan.md`, `6a1e81aa`), the visual harness and `wp1_10_images/scripts/` examples that read `res/rt_is.*` / `img/*_targets` should use `RunStore` (`res/run.zarr`, `targets/cam_*/frame_*` + `correspondences/frame_*`).

Context: wp1 10-frame dataset, `C:\Users\alex\Downloads\wp1_10_images`. GT
(`run_ground_truth.zarr`) is 3dptv.exe's own tracked output (`rt_is`/`ptv_is`),
almost certainly produced by `track3d.c` (`track_mode: 1` in this dataset's
params) — confirmed by feeding GT's own point cloud into our `track3d`
translation with liboptv-parity settings and getting 90% precision / 95.6%
recall against it (see `scripts/ab_dist_weight.py`,
`scripts/compare_track3d_vs_trackcorr.py` in the wp1 folder). The earlier
`cold_start_gate` tuning session (jumps 322→159) was measured against a
different, contaminated baseline (our own detection/correspondence output,
not GT's cloud) and understated true tracker quality by conflating
correspondence-stage error with tracking-stage error — that comparison should
not be reused; the clean control is GT's own `rt_is` cloud fed straight into
the tracker.

## Where we are now (this session's findings)

Best config so far on GT's own cloud (`v_max=1.9, a_max=1.9, dist_weight=1.827,
cold_start_gate=1.5`): prec 0.918, rec 0.958, jumps 119 (down from 229 at
liboptv parity). Per-cascade-level breakdown
(`wp1_10_images/scripts/classify_by_level.py`):

| level | volume (TP+FN) | TP | FP | FN | prec | rec |
|---|---|---|---|---|---|---|
| 1 (has own velocity history) | 6384 | 6240 | 333 | 144 | 0.949 | 0.977 |
| 2 (neighbor-average velocity) | 322 | 246 | 108 | 76 | 0.695 | 0.764 |
| 3 (cold-start / gated static) | 1312 | 1198 | 241 | 114 | 0.833 | 0.913 |

Level 1 is high-volume and already good. **Level 2 is the worst performer by
precision** (0.695) despite tiny volume. Level 3 is the second-worst and by
far the largest error contributor in absolute FP/FN counts.

FN root-cause split (`classify_by_level.py`'s `fn_reason` table):

- **Level 2**: 43/76 FN are "unlinked; target already claimed" — the correct
  candidate existed but a competing particle (almost always processed in
  Level 1, which runs and claims exclusively *before* Level 2 even starts)
  took it first, and Level 2 has no fallback. This is a **cascade-ordering
  loss**, not a prediction-quality problem.
- **Level 3**: 66/114 FN are "unlinked; target free" — the correct candidate
  was available and unclaimed, but `cold_start_gate=1.5` rejected it as
  inconsistent with the local-flow prediction. This is a **direct gate
  false-reject**, the explicit cost of the jump-suppression win.
- **Level 1**: 78/144 FN are "linked elsewhere" — the true continuation was
  available but a different, wrong candidate outranked it under the current
  cost function (`acc + dist_weight * dist_from_curr`). This is a **ranking
  problem**, not an availability problem.

## The goal, precisely stated

"Better than 3dptv.exe's GT" cannot be measured as pure agreement with GT —
GT is itself `track3d.c`/`trackcorr` output, a heuristic tracker, not
ground truth in the metrological sense. Agreement-with-GT caps us at
*reproducing* a possibly-flawed reference. Two separate objectives, don't
conflate them:

1. **Faithful, well-tuned reproduction on real data** (wp1): agreement with
   GT is still the right proxy here because it's the only real-world
   correctness signal we have for this dataset — but the target is "as high
   as the algorithm family allows," not "byte-identical to a specific run."
2. **Genuine correctness on synthetic ground truth**
   (`tests/unit/test_synthetic_tracking.py`, exact known trajectories): this
   is the only place "better than GT" is even measurable, because there the
   true continuation is known independent of any tracker's output. Every
   change below must be checked against the synthetic suite, not just wp1 —
   a change that improves wp1 agreement by exploiting a quirk of this
   dataset's density/geometry is worthless (or harmful) if it regresses
   synthetic recovery rate.

## Concrete next steps, in priority order (highest expected win first)

1. **Cross-level losers-retry for Level 2** (mirrors the fix already proven
   for `trackcorr_c_loop`'s Phase 3, documented in
   `docs/algorithms/tracking.md`). When a Level 1 edge and a Level 2 edge
   compete for the same Level-2-eligible target and Level 1 wins, currently
   Level 2's particle is simply dropped. Instead: after Level 1 finishes,
   let Level 2 also consider candidates Level 1 rejected in *its own*
   contested set (fallback to 2nd/3rd-best), same way `track.py`'s Phase 3
   works. Expected to recover a meaningful slice of the 43 "target claimed"
   Level-2 FN without touching Level 1 at all (Level 1 is already claiming
   correctly in these cases — it's Level 2 that has nowhere to go).
   Implement in `track_kernels_track3d.track3d_loop_fast`, verify against
   both wp1 (per-level breakdown must show Level 2 FN drop, other levels
   unchanged) and the synthetic suite (recovery rate must not regress).

2. **Per-particle cold_start_gate that adapts to local neighbor density**,
   not a single global constant. The gate's 66 false-rejects vs its jump
   suppression is a precision/recall knob tuned on one dataset; a gate
   scaled by the local point spacing (e.g. `gate = f(local NN distance)`
   instead of a flat 1.5mm) should let dense regions stay conservative while
   sparse regions (where the true continuation is farther away and jumps are
   less ambiguous anyway) loosen up. Test the current flat value against a
   density-scaled variant on both wp1 (does the FN-66 bucket shrink without
   jumps climbing back) and synthetic (recovery rate).

3. **Level 1 ranking**: test adding a mild angle-consistency term to the
   Level 1 cost (currently `acc + dist_weight * dist_from_curr`, no angular
   term at all — `track.c`'s equivalent uses angle+acc jointly, our port
   dropped angle entirely per the "Fixed same day" notes about avoiding
   decoy candidates). A candidate physically consistent in direction as well
   as magnitude should out-rank the current 78 "linked elsewhere" wrong
   picks. Needs a synthetic-suite regression check since angle terms are the
   documented culprit `track_kernels_track3d.py` warns against
   over-constraining (comment blocks at top of Level 1 cost computation).

4. **Re-validate on a second real dataset** (burgers, or a different wp1
   frame window) before locking in any parameter changes from steps 1-3 —
   everything found this session is from a single 10-frame window; avoid
   overfitting `cold_start_gate`/`dist_weight` to it.

5. **Only after 1-4**: revisit whether `cold_start_gate` should default to
   something in production `Cython3DTracker`/`tracking_presets.py` (it's
   currently opt-in, default 0.0). Don't flip that default until the
   cascade-order and gate-adaptivity fixes above are in, since today's
   1.5mm value was picked to compensate for a fixable structural loss
   (item 1), not because 1.5mm is intrinsically the right number.

## Visual comparison harness (build first, before step 1)

Every change in steps 1-5 above must be judged visually, not just by the
TP/FP/FN table. Build a reusable 4-panel figure and regenerate it after each
change:

- **3 orthogonal projections** (XY, XZ, YZ) + **1 isometric 3D view**, one
  figure, e.g. a 2x2 matplotlib grid.
- **GT trajectories in blue**, **our tracker's trajectories in red**, same
  axes/limits across all 4 panels so the eye can compare directly.
- Target look: red should overlay blue as closely as possible — the visual
  read is "did this change reduce noise (jagged red segments where blue is
  smooth), lengthen red trajectories (fewer premature breaks vs blue's
  longer runs), and kill wrong-link jumps (a red segment that suddenly darts
  away from its blue counterpart)."
- Save each run's figure with a name tied to the config that produced it
  (e.g. `viz_gate1.5_dw1.827.png`) so consecutive changes can be flipped
  through/diffed, not just eyeballed once and discarded. `wp1_10_images/`
  already has `viz_data*/` and `make_viz_data*.py` scripts from earlier
  sessions — check whether one of those already builds most of this before
  writing a new one from scratch.

## Trajectory trees for cross-tracker decision comparison

Alongside the visual harness, build/extend a **trajectory tree** extractor:
for each trajectory, walk it frame by frame and record, per node, the 3D
position + row id, and the per-camera 2D target (pixel coords + target id)
that the 3D point resolved from. `wp1_10_images/scripts/analyze_trajectory_trees.py`
already does most of this (`extract_trees()` — frame, rt_row, pos, per-cam
`cam_ids`/2D target lookup) — extend it rather than rewriting:

- Run it once per tracker config (GT, current best `track3d` config, any
  candidate change from steps 1-3) to get one tree set each.
- At each frame step, diff trees node-by-node: same particle (matched by 3D
  position within tolerance, or by shared 2D target ids where available)
  but a different `next` link between two tracker outputs marks exactly the
  frame/particle where the decision diverged.
- This is the mechanism for turning "FN reason: linked elsewhere" (from
  `classify_by_level.py`) into "linked WHERE, TO WHAT, and why that
  candidate outranked the correct one" — pull the actual competing
  candidates' costs at that decision point, not just the aggregate count.
- Needs the real 2D targets to be meaningful for the per-camera layer (see
  the `compare_track3d_vs_trackcorr.py` caveat below — GT's own 2D targets,
  not a mismatched detection run's, or the per-camera comparison is noise).

## Tooling already built this session (wp1_10_images/scripts/)

- `ab_dist_weight.py`, `gate_sweep_gt_cloud.py` — config sweeps on GT's own
  cloud (the correct control).
- `classify_by_level.py` — per-level TP/FP/FN + FN-reason breakdown; reuse
  and extend this for steps 1-3 above (add a level-2-losers-retry variant,
  a density-scaled-gate variant, and an angle-term variant, each re-run
  through this same classifier to see which FN bucket actually shrinks).
- `compare_track3d_vs_trackcorr.py` — do **not** trust its trackcorr number;
  it feeds mismatched 2D targets (copied by frame-index from an unrelated
  detection run) into `trackcorr_c_loop`, so its trackcorr row (rec 0.001) is
  meaningless. Fix if trackcorr comparison is needed again: either find
  GT's actual original 2D targets (check `res_ground_truth_backup/` for
  `_targets` files) or drop the trackcorr comparison until real targets are
  available.
