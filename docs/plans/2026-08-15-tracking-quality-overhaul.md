# Tracking Quality Overhaul: Warmup Auto-Calibration + Corrective Backward Pass

**Date:** 2026-08-15
**Status:** approved, ready for implementation
**Implementer notes:** every edit under `src/openptv2/algorithms/` requires a Cython rebuild:
`uv run python setup.py build_ext --inplace`. Always `uv run pytest ...`, never bare python.
Ignore duplicate module copies under `.claude/worktrees/` and `Lib/site-packages/`.

## Context: why this plan exists

Despite porting/adapting several trackers (trackcorr C-translation, track3d, MyPTV, proPTV-GMM,
Kalman/Hungarian) and extensive parameter sweeps, tracking quality remains below the legacy
standard: tracks are short, fragmented, and on real data polluted by wrong links. Exploration of
the codebase, docs, and git history established the actual root causes — mostly **not** in the
candidate-ranking logic that previous attempts kept tuning:

1. **Confirmed bug** in the backward-pass kernel `src/openptv2/algorithms/track_kernels_corr.py`
   lines ~1723–1742: the acceptance test
   `if (acc < dacc and angle < dangle) or acc < dacc * 0.1:` guards **only** the `d13`
   assignment; the rest of the body (`d01`, `dl`, `rr`, and the `path_decis_1` /
   `path_linkdecis_1` append) is de-dented and executes unconditionally for every candidate that
   passed the velocity box, using a stale `d13` from a previous loop iteration. The forward-pass
   equivalent (L825–844) and the C original keep the whole body inside the `if`. Consequence:
   the backward pass accepts candidates that fail the acceleration/angle test and ranks them with
   garbage `rr`; this also corrupts `enforce_reciprocity` in `tracking_postprocess.py` (backward
   field vetoes/confirms forward links). Every forward+backward run is degraded.
2. **Ghost correspondences dominate real-data wrong links.** The synthetic benchmark with *real*
   triangulation (`test_data/tracking_synthetic_dense`) shows wrong links ≈ 0 for ALL trackers in
   all density/noise regimes — accuracy loss there is entirely lost links. On real test_cavity,
   2-camera pairs are 64% ghosts, triplets 38%, quads 16% (short-baseline sub-rigs, z-sensitivity
   ~7× worse than in-plane; see `docs/plans/two-subrig-calibration.md`). Kalman and global
   Hungarian both failed to help (commit b367c5d) because the problem is upstream of linking.
3. **Density loss is lost links, not wrong links** — every engine drops below 0.30 precision at
   5k particles/frame; fixed buffers (`max_targets=10000`, `POSI_K=80`, `MAX_CANDS=32`) overflow
   at 20k.
4. **The one-way pipeline is the ceiling**: detection → correspondence (frozen) → triangulation →
   tracking. Modern codes (Shake-the-Box, proPTV, OpenLPT) couple tracking with correspondence.
5. **Parameter tuning is manual.** The test_cavity gate was committed 50–100× looser than the true
   flow (commit 07f1fc1) and nobody noticed for a long time; `tracking_feasibility.py` warns but
   nothing auto-tunes.

## The leading tracking concept

Three phases, replacing today's manual tune-then-run workflow:

1. **Smart warmup (auto-calibration)** — on the first N frames, run forward tracking AND backward
   tracking (first+N → first) in parallel, compare the two link fields, iterate. The
   forward/backward *disagreement rate* is a ground-truth-free quality signal. From it: estimate
   positional noise empirically, auto-tune tracker parameters (dv box, dacc, dangle), and
   auto-select the best tracker algorithm.
2. **Full forward run** with the warmup-chosen tracker + parameters.
3. **Corrective backward pass** — NOT a mere reciprocity check: a backward sweep merged with
   track-assisted re-correspondence (STB-lite). Walking backward it predicts from established
   tracks, claims unused 2D targets, re-triangulates, fixes wrong links, bridges gaps, adds
   particles the combinatorial correspondence missed, and prunes ghosts. One pass that *repairs*
   both tracks and the particle field; iterable (N = 1–2) while links still change.

The online sequence+tracking merge (restructuring `py_sequence_loop`) is deferred to a contingent
Stage 5 — only if the corrective-pass iteration measurably saturates.

## Benchmarks (user-approved tiers)

| Tier | Dataset | Role |
|---|---|---|
| Primary | ghost-inclusive synthetic (Stage 0.5 extends `test_data/tracking_synthetic_dense/generate.py`) | gate for every stage; reproduces the real failure mode with pid ground truth |
| Secondary | proPTV 500_25 / 500_30 (`C:/Users/alex/Github/proPTV/data/`) | cross-software comparison, PMP > 99% |
| Sanity | test_cavity (real) | directional checks only — rig is flagged poorly conditioned (z-noise 0.474 mm ≥ motion 0.337 mm/frame) |
| Future | new well-conditioned real experiment | promoted to real-data gate when available |

---

## Stage 0 — Bug fix + honest quality gates (0.5–1 day, FIRST)

### 0a. Fix the backward-pass guard bug
- `src/openptv2/algorithms/track_kernels_corr.py` L1723–1742: indent the body (`d01`, `dl`, `rr`,
  the appends, and the `quali` read) into the `if` guard, mirroring forward L825–844.
- Rebuild Cython.
- New `tests/unit/test_trackback_candidate_guard.py`: construct a 4-frame scenario with one
  candidate inside the velocity box but failing acc/angle; assert `path_inlist` stays 0 after
  `trackback_loop_fast`. End-to-end on `test_data/synthetic_turbulent_1k`: record backward link
  count and reciprocity-confirmed count before/after the fix.
- Existing parity tests (`tests/unit/test_track3d.py`, `test_track.py`) pin burgers/test_cavity
  counts — backward results will intentionally change; re-pin with a comment citing this plan.

### 0b. Quality-floor gates for ALL trackers
- Extend `tests/unit/test_tracker_quality.py` (currently fast_3d only, `@slow`) to parametrize
  over `tracking_registry.py` entries on `synthetic_turbulent_1k`, using
  `src/openptv2/tracking_metrics.py` precision / recall / ghost_capture. Record current values
  per tracker, gate at (current − 2%) as ratchet floors. Keep total runtime < ~60 s so it runs in
  normal CI (drop `@slow` if it fits).

**Success:** backward-pass precision ≥ forward precision on synthetic_turbulent_1k (pre-fix it
should be worse); reciprocity postprocess stops deleting correct forward links.

## Stage 0.5 — Ghost-inclusive primary benchmark — DONE (2026-08-15)

`test_data/tracking_synthetic_dense/generate.py::build_fixture_with_correspondence` (new,
alongside the untouched ghost-free `build_fixture`) runs the *real* combinatorial correspondence
matcher (`src/openptv2/algorithms/correspondences.py`) on noisy projected 2D targets instead of
writing `rt_is` from known identity, so ghosts arise from genuine epipolar ambiguity as density
increases — no injected false detections needed. Returns `row_gt: frame -> [true_pid_or_-1]`
since row index no longer equals particle id once real matching runs. New
`tests/unit/test_tracking_synthetic_dense_ghosts.py`: confirms the fixture produces ghosts at
test_cavity-like density, and that track3d tracking on it produces measurable wrong links
(30 wrong / 511 correct in the checked-in regression) — the ghost-free fixture cannot show this
by construction.

**Significant finding along the way (flagged, not fixed — out of scope for this plan):**
`algorithms/correspondences.py`'s `four_camera_matching`/`three_camera_matching` cross-reference
one camera pair's adjacency table using another pair's candidate `pnr` value *as an array index*
(e.g. `p2_arr[1, 2, p2]` where `p2` came from the (0,1) pair's `find_candidate` output). That
table's row dimension is actually built from each target's **x-sorted position** in
`corrected[cam]`, not its `pnr`. The two only coincide when `pnr` happens to already be assigned
in x-sorted-rank order — true of the existing `test_correspondences.py` grid fixture (with its
odd-camera index reversal) purely by construction, but false in general, and false for the
y-sorted `pnr` convention the *tracker* requires (`candsearch_in_pix`'s binary search). Assigning
`pnr` in any other order (e.g. y-sorted, or real detection order) was measured to collapse
correspondence matching to near-total ghosts *regardless of true particle separation* — verified
directly: even 2 widely-separated random points fail to match; switching only the `pnr` order to
x-sorted-rank fixed it to 100% correct at every density tested (2 to 100 particles, both the
`tracking_synthetic` sub-rig calibration and `openptv2.benchmarking`'s standard rig). This
generator works around it internally (matches on an x-sorted-`pnr` `Frame`, translates results
back to the on-disk y-sorted `pnr` the tracker needs). **Not fixed here**: whether the real
production pipeline (`gui/ptv.py::py_correspondences_proc_c` → `MatchedCoords`) is also exposed
depends on whatever order real target detection assigns `pnr` in, which needs its own
investigation — this could be a material, previously-undocumented contributor to real-data ghost
rates (e.g. test_cavity's measured 64%/38%/16% ghost rates) independent of rig conditioning.
Recommended as a dedicated Stage 0.5b or early Stage 2 investigation before trusting ghost-rate
numbers as purely a rig-conditioning signal.

Deferred: promoting the proPTV comparison scripts (`scratch/benchmark_all_trackers_500_25.py` /
`_30.py`) into a repeatable `scripts/` harness — lower-priority sub-bullet, not blocking Stage 1.

**Success (met):** the benchmark reproduces the real-data failure signature (nonzero ghost
capture, wrong links > 0) that the ghost-free synthetic misses.

## Stage 1 — Smart warmup: auto-calibration on the first N frames (~1 week)

New module `src/openptv2/tracking_warmup.py`. **Warmup is a separate, standalone step that runs
BEFORE the tracker — not a hidden phase inside the tracking run.** Exposed as `openptv warmup` in
`cli.py` (and a GUI action later). The natural workflow:

1. Run warmup one or more times — each run searches for the best parameter set and reports it.
2. The result is **persisted** (RunStore `stats/warmup` + written back into the run's YAML
   tracking config as the chosen tracker + `TrackPar` values) so the user can inspect and
   **manually adjust** any value they care about.
3. Full tracking then runs with the stored config and does NOT re-run warmup — no warmup cost on
   the production run, and re-runs are reproducible from the stored parameters.

`plugins/default_tracking.py` only *checks* for a stored warmup result: if
`track.warmup.require: true` and none exists, it fails with "run `openptv warmup` first";
otherwise it uses whatever config is present (stored-warmup or manual). Config
`track.warmup: {frames: N, require: bool}` (N default 20–30).

Loop (2–4 cycles):
1. **Seed** parameters from `tracking_recommender.compute_dataset_stats` + `_suggest_params`
   (displacement percentiles, NN spacing — `src/openptv2/tracking_recommender.py`) and
   `tracking_feasibility.assess_tracking_conditioning` (analytic z-noise floor vs motion scale).
2. **Dual-direction tracking on frames [first, first+N]**: `Tracker.full_forward` over the
   window, and independently `trackback_c` from first+N down to first (backward-only field, not
   primed by forward links). Both write to scratch linkage groups in the zarr RunStore
   (`linkage/warmup_fwd`, `linkage/warmup_bwd` — extend `storage/run_store.py` naming).
3. **Compare**: per-frame forward/backward link agreement (link i→j confirmed when the backward
   field contains j→i), disagreement taxonomy (missing vs conflicting), and the positional
   scatter of confirmed links → an **empirical positional-noise estimate** (validates/replaces
   the analytic `z_noise_floor_mm`).
4. **Adjust**: set the dv box and dacc from the confirmed-link displacement distribution (e.g.
   dv = p99 of confirmed displacements × margin); re-run the window. Promote the search logic
   from `scripts/tune_tracker_params.py::adaptive_sweep` into the module.
5. **Select algorithm**: run the 2–3 registry candidates that
   `tracking_recommender.recommend_tracker` shortlists over the window with tuned params; pick by
   highest (fwd/bwd agreement × mean confirmed-track length). Persist choice + params + noise
   estimate to RunStore `stats/warmup`.

Output: chosen tracker, tuned `TrackPar`, empirical noise estimate — persisted to
`stats/warmup` and the run's tracking config, optionally hand-edited by the user, then consumed
by the full forward run without another warmup.

**Success criteria:**
- Ghost-inclusive synthetic + proPTV 500_30: warmup-chosen config reaches ≥ 95% of the
  precision/recall of the best hand-tuned configuration found by exhaustive sweep.
- test_cavity: reproduces or beats the manually retuned gate from commit 07f1fc1
  (±0.6 mm / dacc 0.24) with zero human input.
- Warmup wall time < ~1 min at 1k particles/frame.

**Key insight:** fwd/bwd agreement is a ground-truth-free quality metric, so warmup works on real
data where no pid ground truth exists.

## Stage 2 — Corrective backward pass with track-assisted re-correspondence (main event, 1–2 weeks)

The backward pass and STB-lite re-correspondence become ONE mechanism. New module
`src/openptv2/track_assisted.py` (~300–400 lines orchestration, **zero new kernels**), invoked
from `plugins/default_tracking.py` behind `track.corrective_passes: N` (default 0 = off, fully
backward compatible; N = 1–2 typical).

Walking backward frame t+1 → t, using forward tracks as prior:
1. **Load** tracks + correspondences from RunStore (`tracking_postprocess.read_linkage`, readers
   in `storage/run_store.py`).
2. **Predict backward** per track: linear from the two later points; the vendored GMM predictor
   (`src/openptv2/plugins/proptv/prediction.py`) for tracks of length ≥ 4.
3. **Verify/fix links**: where the backward prediction disagrees with the forward link beyond the
   warmup noise estimate, re-rank candidates using the (Stage-0-fixed) `trackback_loop_fast`
   decision logic and rewire. This replaces the plain `enforce_reciprocity` veto with actual
   correction.
4. **Claim missed particles**: for tracks that end at t+1 going forward (= start here going
   backward), call `assess_new_position_fast`
   (`src/openptv2/algorithms/track_kernels_transform.py` ~L720) at the backward-predicted 3D
   position — it already projects to all cameras, runs `candsearch_in_pix_fast` over *unused*
   targets, and triangulates via `point_position_fast` (~L516). Accept at ≥ 2 camera hits within
   `tpar.add` radius; extend the track and add a particle the combinatorial correspondence
   missed.
5. **Optional shake**: wire `src/openptv2/plugins/stb_4d_refinement.py::shake_particle_position_3d`
   on claimed triangulations (opt-in `track.shake: true`; only when frame images exist — skip for
   existing-targets runs).
6. **Prune ghosts**: an untracked particle with only 2-cam support that no forward or backward
   track claims decays; a track-claimed particle sustained by only 2 cameras for ≥ 3 consecutive
   frames without regaining a 3rd is killed. Directly attacks the 64%-ghost pair population.
7. **Bridge gaps**: where a backward extension meets a forward track fragment within the
   noise-scaled search volume, merge the fragments (subsumes
   `tracking_postprocess.relink_trajectory_gaps` for the tracked case).
8. **Write back** corrected correspondences + linkage with provenance flags
   (forward / backward-corrected / track-claimed / pruned). Repeat the pass while links change
   > ~1%.

**Success criteria:**
- Ghost-inclusive synthetic (primary): wrong-link rate halved vs forward-only baseline; mean
  track length up ≥ 25%; ghost_capture_rate down ≥ 50%; NO precision regression on the
  ghost-free dense benchmark (guards against claimed-ghost feedback).
- test_cavity (sanity): ghost_capture down, mean track length up; no hard threshold.
- proPTV 500_30: PMP stays > 99%, track completeness up.

**Risks:** ghost-track self-reinforcement (mitigated by the 2-cam decay rule + pass cap);
rewriting correspondences invalidates prior linkage (acceptable — the pass rewrites both
atomically per frame; provenance flags keep it auditable).

## Stage 3 — Better in-tracker cost (independent, ~1 week)

All inside `track_kernels_corr.py`, using data already in the 4-frame buffer:
- **Per-track adaptive search radius**: with history ≥ 3, scale the dv box per particle from
  local velocity + k·(recent velocity variance) instead of the global box.
- **Multi-frame residual term**: proPTV-style polynomial-fit residual over the buffered X[0..3]
  added to the `rr` cost (currently `dl/lmax + acc/dacc + angle/dangle`, divided by camera-vote
  count `quali`).
- **In-tracker 1-frame coast** for detection dropouts — but FIRST measure whether
  `relink_trajectory_gaps` / the Stage 2 pass already recovers these; skip if so (YAGNI).

**Success:** at ≥ 5k particles/frame on the dense benchmark, recall up with precision ≥ baseline
(target 0.30 → ≥ 0.5).

## Stage 4 — Capacity + performance (1–2 days, mechanical, parallel to Stage 2)

- Replace fixed `max_targets=10000`, `POSI_K=80`, `MAX_CANDS=32`
  (`src/openptv2/algorithms/constants.py` and kernel buffer allocations) with sizes computed from
  actual frame target counts at `TrackingRun` setup (`src/openptv2/algorithms/tracking_run.py`).
  Buffers are allocated per run, so this is parameter plumbing, not algorithm change.

**Success:** 20k particles/frame synthetic run completes without overflow; 1k results
bitwise-identical to baseline.

## Stage 5 (contingent) — Physics validation + optional online merge

- Implement track-lifetime distribution + acceleration-PDF kurtosis from
  `docs/lagrangian_turbulence_quality_guide.md` into `src/openptv2/benchmarking/metrics.py`; run
  per stage — these catch ghost contamination that precision alone misses.
- Online sequence+tracking merge (restructure `src/openptv2/gui/ptv.py::py_sequence_loop`,
  ~L980): ONLY if the corrective-pass iteration measurably saturates below target. Big lift
  (touches ptv.py, Tracker/framebuf lifecycle, GUI, batch); explicitly deferred.
- When the new well-conditioned real dataset arrives, promote it to the real-data gate.

**Skipped deliberately:** a standalone reciprocity-only backward feature (Stage 2 subsumes it);
Kalman/Hungarian retreads (shown information-limited, commit b367c5d).

## Sequencing

Stage 0 → measure → Stage 0.5 → Stage 1 (warmup) → Stage 2 (corrective pass, ‖ Stage 4) →
measure → Stage 3 → Stage 5 metrics throughout. Every stage gates on the ratcheted
`test_tracker_quality.py` floors plus its own criterion. Warmup lands before the corrective pass
because its noise estimate parametrizes the corrective pass's disagreement thresholds.

## Critical files

- `src/openptv2/algorithms/track_kernels_corr.py` — bug fix L1723–1742; backward decision logic
  reused in Stage 2; Stage 3 cost changes (Cython rebuild after each edit)
- `src/openptv2/algorithms/track_kernels_transform.py` — `assess_new_position_fast` (~L720),
  `point_position_fast` (~L516): the Stage 2 claim primitive, reused as-is
- `src/openptv2/algorithms/correspondences.py` — real correspondence run for Stage 0.5 generator
- `src/openptv2/tracking_warmup.py` — NEW, Stage 1 (reuses `tracking_recommender.py`,
  `tracking_feasibility.py`, `scripts/tune_tracker_params.py::adaptive_sweep`)
- `src/openptv2/track_assisted.py` — NEW, Stage 2 corrective backward pass
- `src/openptv2/plugins/default_tracking.py` — integration point: reads stored warmup config
  (never runs warmup itself) → full forward → corrective passes
- `src/openptv2/cli.py` — new `warmup` subcommand (standalone pre-tracking step)
- `src/openptv2/tracking_postprocess.py` — linkage read/write (Stage 0 verification, Stage 2
  orchestration; reciprocity/gap-relink subsumed by Stage 2 for tracked particles)
- `src/openptv2/storage/run_store.py` — scratch linkage groups (`linkage/warmup_fwd`/`_bwd`),
  `stats/warmup`, provenance flags
- `test_data/tracking_synthetic_dense/generate.py` — ghost injection (Stage 0.5)
- `tests/unit/test_tracker_quality.py` — parametrized ratchet floors (Stage 0b)

## Verification

- `uv run pytest tests/unit/test_trackback_candidate_guard.py tests/unit/test_tracker_quality.py -v`
  after each stage.
- Parity guard: `uv run pytest tests/unit/test_track3d.py tests/unit/test_track.py` — burgers
  parity must hold except where the Stage 0 fix intentionally changes backward results; re-pin
  those with a comment citing this plan.
- Benchmark sweep: ghost-inclusive dense generator at 1k/5k/20k + proPTV 500_30, compared against
  recorded baselines via `scripts/benchmark_utils.py`.
- Real-data sanity: test_cavity through GUI/batch; inspect trajectory-length distribution from
  the RunStore `traj/` index.
- Warmup acceptance: exhaustive sweep vs warmup-chosen config on the primary benchmark (≥ 95%).
