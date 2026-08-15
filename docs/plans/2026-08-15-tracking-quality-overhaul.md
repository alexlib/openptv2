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

**Docstring/footgun found and fixed along the way (verified NOT a production bug):**
`algorithms/epi.py::find_candidate` writes each candidate's **index into the x-sorted `crd`
list** into its `cand_pnr` output array — not the particle's `pnr` (`p2 = crd[j].pnr` is computed
and used only for an internal quality-ratio lookup; the value actually stored is `j`, the loop
index). That output flows straight through `algorithms/correspondences.py`'s adjacency tables into
`NTupel.p[cam]`, so **every** camera's component of a `NTupel` — not just camera 0 — is an
x-sorted index, never a `pnr`. `find_candidate`'s docstring/parameter comment and `NTupel`'s own
docstring were wrong (or misleadingly named) about this; both are fixed now to say so explicitly,
and point at the correct translation.

This was initially misdiagnosed as a live production bug (assigning `pnr` in y-sorted order
appeared to collapse correspondence matching to near-total ghosts regardless of separation). That
diagnosis was wrong: the code path used to reach that conclusion called the raw `NTupel.p[cam]`
values directly as if they were already `pnr` for cameras 1–3 (only camera 0 was translated). The
**real** production entry point, `openptv2.correspondences.correspondences()` (what
`gui/ptv.py::py_correspondences_proc_c` actually calls) already performs
`corrected[cam][geo_id].pnr` for **every** camera before using the result — verified directly: run
end-to-end through this exact wrapper with y-sorted `pnr` (matching real detection: `gui/ptv.py`'s
`_detect` calls `targs.sort_y()` before `MatchedCoords` assigns `pnr = i`), 100% correct
quads at every density tested (2–200 particles, multiple seeds, two different calibrations). New
`tests/unit/test_correspondences.py::TestNTupelIdentityTranslation` /
`TestProductionCorrespondencesWrapperIdentity` lock this in: one test proves the untranslated
low-level interpretation genuinely scrambles identities on a random scatter (the footgun), the
other proves the actual production wrapper does not. **No live bug, no effect on test_cavity's
measured ghost rates** — those remain purely a rig-conditioning signal, as originally documented
in `docs/plans/two-subrig-calibration.md`. `test_data/tracking_synthetic_dense/generate.py`'s
generator now uses the same (correct, uniform-per-camera) translation as the production wrapper,
which simplified it considerably — the earlier two-numbering-scheme workaround was unnecessary.

Deferred: promoting the proPTV comparison scripts (`scratch/benchmark_all_trackers_500_25.py` /
`_30.py`) into a repeatable `scripts/` harness — lower-priority sub-bullet, not blocking Stage 1.

**Success (met):** the benchmark reproduces the real-data failure signature (nonzero ghost
capture, wrong links > 0) that the ghost-free synthetic misses.

## Stage 1 — Smart warmup: auto-calibration on the first N frames — DONE (2026-08-15)

New module `src/openptv2/tracking_warmup.py` + `openptv warmup` CLI subcommand
(`src/openptv2/cli.py`). Warmup is a standalone step run *before* tracking (`run_warmup(...)`
never gets called implicitly by the tracker); results print to the console and, with `--write`,
get persisted into the run's YAML `track:`/`plugins.selected_tracking` and the RunStore's
`meta.attrs["warmup"]` — a plain `openptv track` run afterward picks the tuned config up with no
warmup-awareness of its own.

**Design deviation from the original spec (deliberate, documented in the module docstring):**
- *Dual-direction measurement*, not two independent passes. `trackback_c` fundamentally requires
  pre-existing forward `next` links on disk to find track heads (`prev==-1, next>=0`) to extend
  backward — there's no way to run it "not primed by forward links" as originally written. Reused
  `Tracker.full_forward()` + `Tracker.full_backward()` + `Tracker.postprocess(reciprocity=True)`
  (which already wraps `tracking_postprocess.enforce_reciprocity`/`count_links`) as-is: the
  reciprocity pass's `severed_next`/`links_before`/`links_after` **are** the forward/backward
  agreement signal, computed with zero new comparison code. Confirmed-link (post-reciprocity)
  displacement distribution gives the noise estimate, same idea as originally specified.
- *Scratch linkage groups* needed no `storage/run_store.py` extension — `Tracker`'s existing
  `naming["linkage"]` basename already becomes the store's group name (`Tracker(..., naming={
  "linkage": "warmup/cycle1", ...}, store=store)` writes to `linkage/warmup/cycle1` and nothing
  else), so the plan's `linkage/warmup_fwd`/`_bwd` scratch-group goal is met without any storage
  changes.
- *Persistence* uses `store.root["meta"].attrs["warmup"]`, not a `stats/warmup` group —
  `RunStore.write_stats` has a fixed, tracking-telemetry-specific schema (frame-keyed
  n_targets/n_links/etc.), not a generic key/value store; `meta.attrs` already holds
  schema_version/sealed the same way.
- *Algorithm selection* scope-cut to the two engines directly reachable through `Tracker`
  (`priority_segment_3d`/track3d via `full_forward_3d`, `full_multipass`/trackcorr via
  `full_forward`+`full_backward`), scored by mean forward-only trajectory length (computed
  straight from prev/next chains, no ground truth needed). The other `TRACKER_REGISTRY` entries
  are plugin-based and need a constructed experiment object (`pm`, target files, the plugin
  loader) — a materially bigger integration, not done here; extend `_CANDIDATE_ENGINES` if
  warmup's pick needs to go beyond these two.

**Verified end-to-end** (`tests/unit/test_tracking_warmup.py` + a manual CLI smoke test on
`test_data/synthetic_turbulent_1k`): on a realistic 8-frame window, warmup tuned a seed dv box of
±15.5 mm (an intentionally-loose recommender default) down to ±13.3 mm automatically with 100%
forward/backward agreement, picked track3d (mean trajectory length 6.5 vs trackcorr's 3.3), and
`--write` correctly persisted both the tuned `track:` block and `plugins.selected_tracking` into
the YAML — a `TrackPar.from_yaml` reload afterward picks up the exact tuned values. On a denser
test_cavity-like synthetic scene the tuning loop pulled a seed dv box of 15.5 mm down to ~3.1 mm
in 2 cycles (same direction and rough magnitude as the real, manually-discovered fix in commit
07f1fc1 — ±15.5 mm was 50-100x too loose there too), which is the qualitative validation the
original "reproduces or beats the manual retune" success criterion was after; a literal apples-to-
apples number against 07f1fc1's exact fixture/dataset was not re-run.

**Not yet validated against the originally-specified quantitative gates** (ghost-inclusive
synthetic / proPTV 500_30 precision-recall vs. exhaustive sweep; wall-time budget at 1k
particles/frame) — those need Stage 0.5's ghost-inclusive benchmark and a real exhaustive-sweep
baseline to compare against, which is follow-up work, not blocking Stage 2 (Stage 2's corrective
pass consumes warmup's noise estimate as a threshold parameter, not a pass/fail gate on warmup
itself).

**Key insight confirmed:** fwd/bwd agreement (via the existing reciprocity postprocess) is a
ground-truth-free quality signal, so warmup works identically on synthetic and real data.

## Stage 2 — Corrective backward pass with track-assisted re-correspondence — DONE (2026-08-15, reduced scope)

New module `src/openptv2/track_assisted.py`, invoked from `plugins/default_tracking.py` behind
`track.corrective_passes: N` (default 0 = off, fully backward compatible). Implements items 1, 4
(the core claim mechanism), 7, and 8 (partially) of the original 8-step spec; items 2/3/5/6 are
scope cuts, each forced by a primitive that turned out not to exist as reusable plain-Python code
(verified before writing any code, not discovered by trial and error):

- **`assess_new_position`/`assess_new_position_fast`** (the item-4 primitive the plan named) has
  **zero callers anywhere in the codebase** and is unusable outside the full compiled-kernel
  `TrackingRun`/`FrameBuf`/packed-calibration setup — building that setup from orchestration code
  is not "zero new kernels." Used `candsearch_in_pix_rest` + `orientation.point_position` +
  `openptv2.correspondences.MatchedCoords` instead (all genuinely plain-Python, all already
  proven elsewhere this session) to get the same functional result: project the prediction into
  every camera, search each camera's *unclaimed* 2D targets, triangulate on ≥ 2 hits.
- **Item 3** (re-rank disagreeing links against `trackback_loop_fast`'s exact decision logic) has
  no plain-Python equivalent either — that 4-point test + `rr` cost formula lives only inside the
  compiled nogil kernel. Not reimplemented (a hand-copied duplicate of fixed decision logic is a
  correctness liability, not a scope win); `enforce_reciprocity` (already proven, unchanged) is
  the disagreement signal used instead, same as before.
- **Item 2's GMM predictor** and **item 5's shake refinement** are deferred: GMM
  (`plugins/proptv/prediction.py`) needs track chains of length ≥ 4 assembled from the prev/next
  graph, which the reduced scope doesn't build; shake needs real per-camera image arrays this
  disk-level pass doesn't have. Linear 2-point backward extrapolation only.
- **Item 6's per-track running 2-cam-decay prune** is deferred — not yet load-bearing until Stage
  2 is run at a density where it matters; the claim mechanism already records each claim's camera
  count (`CorrectiveStats.claimed_2cam`) so the decay rule can be added without touching the claim
  path itself.
- **Item 8's provenance flags**: no storage field for this exists (`RunStore.write_linkage`'s
  optional `prio` int32 column is the closest precedent, but it's a single generic slot already
  named for a different legacy purpose) — not added; claimed vs. original rows are currently
  distinguishable only by re-deriving from `CorrectiveStats`, not from a persisted per-row flag.

What's implemented, walking backward frame t+1 → t: for every track head at t+1 (`prev == -1`)
with a known forward velocity (a `next` link to t+2), predicts its position at t by linear
backward extrapolation, projects into every camera (`algorithms.track.point_to_pixel`), searches
each camera's targets not already referenced by t's correspondences
(`algorithms.track.candsearch_in_pix_rest`), triangulates when ≥ 2 cameras hit
(`algorithms.orientation.point_position`), appends the new row to the frame's correspondences
(`RunStore.write_correspondences`), and rewires the track's `prev` pointer to the new row
(`tracking_postprocess.write_linkage`). After the walk, reuses (unchanged)
`tracking_postprocess.relink_trajectory_gaps` and `enforce_reciprocity` — satisfying item 7
entirely via reuse, no new gap-bridging code. Iterates while total link count keeps changing by
more than `min_change_frac` (default 1%, matching the original "repeat while links change > ~1%"
spec), capped at `max_passes`.

**Verified correct** (`tests/unit/test_track_assisted.py`): a hand-built scenario where one
particle's 2D targets exist in every camera at one frame but its 3D correspondence row is
deliberately absent there (simulating exactly the "combinatorial correspondence missed it"
failure mode), while its track resumes the next frame with a known forward velocity. The
corrective pass recovers the exact dropped particle (triangulated position within 2×10⁻⁵ mm of
ground truth) and correctly rewires the downstream track's `prev` pointer to the recovered row.

**Not yet measured against the original quantitative success criteria** (wrong-link rate halved,
mean track length +25%, ghost_capture -50% on the ghost-inclusive synthetic; test_cavity and
proPTV 500_30 directional checks) — a smoke test on the Stage 0.5 ghost-inclusive fixture at
moderate density ran cleanly (no crash) with both track3d and trackcorr forward engines, but
neither run had a genuine correspondence dropout for the pass to fix (track3d linked 100% of
particles already; a separate trackcorr/this-fixture interaction produced 0 forward links,
unrelated to Stage 2 and not investigated further this session) — so the smoke test confirms the
pass is safe to run, not that it moves the needle at scale. Running the quantitative gates is
follow-up work.

**Risks:** ghost-track self-reinforcement is NOT YET mitigated (the decay rule from item 6 isn't
built) — `corrective_passes` should stay opt-in (default 0) until that's measured; rewriting
correspondences invalidates prior linkage assumptions elsewhere in the same frame (acceptable in
the tested scope — the pass appends one row and rewires one pointer per claim, atomically per
frame, but wasn't stress-tested against concurrent/other postprocess passes touching the same
frame in the same run).

## Stage 3 — Better in-tracker cost — INVESTIGATED, deferred (2026-08-15)

All three sub-items were evaluated before writing any code (same discipline as every prior
stage); one is resolved, two are deliberately deferred rather than rushed into the riskiest code
path this plan touches.

- **In-tracker 1-frame coast for detection dropouts — SKIPPED (YAGNI), confirmed redundant.** The
  plan itself said to check `relink_trajectory_gaps` first. It already handles exactly this case:
  `tests/unit/test_tracking_postprocess.py::test_relink_trajectory_gaps_bridges_missing_frame`
  builds a particle undetected for one whole frame (not just missing from correspondence — no 2D
  targets either) and confirms the gap gets bridged by constant-velocity extrapolation. Already
  wired into Stage 2's corrective pass. Nothing to build.
- **Per-track adaptive search radius** and **multi-frame residual term — deferred, not
  implemented.** Both require modifying `track_kernels_corr.py`'s compiled nogil decision logic —
  the exact code Stage 0 found and fixed a real, subtle bug in (the backward-pass acceptance
  guard), and the code every pinned-link-count regression test in the suite depends on for its
  exact candidate tie-breaking. A rushed change here risks silently regressing dozens of already-
  validated tests in ways that are easy to miss without dedicated, focused verification time this
  stage did not get. User decision (2026-08-15): defer both to a dedicated follow-up session
  rather than compress them into the tail of this one.
  - The *concept* behind the adaptive-radius item already exists as reusable code:
    `src/openptv2/tracking_cost.py::compute_velocity_aligned_search_radius` (velocity-aligned
    anisotropic search ellipsoids), already used by the Hungarian/assignment-based tracker family
    (`kalman_hungarian_3d` etc.). The gap is specifically that trackcorr's compiled kernel doesn't
    use it — extending it there needs new per-particle velocity-history state threaded through
    `TrackingRun`/`FrameBuf`, which is the actual size of the remaining work, not a green-field
    design problem.

**Not measured** (no code changed, no baseline shift expected): recall at ≥ 5k particles/frame on
the dense benchmark remains at the Stage 0-era numbers documented in `master-plan.md`'s density
sweep.

## Stage 4 — Capacity + performance — DONE (2026-08-15, reduced scope)

`src/openptv2/tracker.py::_estimate_max_targets` replaces `Tracker.restart()`'s hardcoded
`max_targets=10000` with a value computed from the actual correspondence counts across the run's
frames (peeked via the attached `RunStore` if present, else the `rt_is` ASCII line counts),
floored at the old 10000 default and margined ×1.5 over the observed maximum. Only `max_targets`
needed fixing, not `POSI_K`/`MAX_CANDS` (`algorithms/constants.py`) as the plan's bullet also
named: those bound the number of *candidates considered per particle* (camera-geometry/consensus
limited, not particle-count limited — `path_decis_1[h, inlist]` is sized `(max_targets, POSI_K)`,
so `max_targets` already governs the dimension that actually scales with density), confirmed by
reproducing the real failure and checking it disappears with only the `max_targets` fix.

**Verified, not just claimed:** reproduced the exact pre-fix failure at 20k particles/frame —
`Tracker`/`TrackingRun` with the old hardcoded `max_targets=10000` raises
`ValueError: frame 10001: 20000 particles exceeds max_targets=10000` (a clean, pre-existing
defensive check in `Frame.read`, not silent corruption — better than the plan's "overflow"
framing implied). With the new dynamic sizing (`_estimate_max_targets` → 30000 for this dataset),
the same 20k-particle, 3-frame run completes cleanly (60000 total particles, 39961 links, no
error). All 291 existing tests, including every exact pinned-link-count regression, still pass
unchanged — confirms the 1k-and-below baseline is bitwise-identical (the estimator never returns
less than the old 10000 floor).

**Success (met):** 20k particles/frame synthetic run completes without overflow; 1k results
bitwise-identical to baseline (all existing pinned-count tests unchanged).

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
