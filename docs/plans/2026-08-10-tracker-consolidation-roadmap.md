# Tracker consolidation roadmap: two presets, honest measurement, high-density scaling

**Date:** 2026-08-10
**Status:** Stage A, Stage 0, and Stage 1 (1a–1d) done and measured. Stage 2
in design (2a revised to a Kalman-filter predictor). Stages 3–4 not started.
**Supersedes:** `docs/plans/2026-08-04-tracking-improvement-metrics-plan.md` (its
A1–A6 deliverables shipped, but its results doc benchmarks a tracker that no
longer exists — see Context)

## Progress log

- **2026-08-10 — Stage A**: fixed `fast_3d`'s Level 1/2 acceleration-residual
  sign bug (candidates were ranked by proximity to a point *behind* the
  particle). `test_cavity` link count 1765 → 1753 (fewer but correct).
- **2026-08-10 — Stage 0**: pid-exact one-to-one identity metrics, ghost-capture
  rate, `benchmark_utils.combined_metrics` (both metric systems in one row),
  `scripts/bench_trackers.py` single entry point, density-sweep dataset
  generation (fixed an O(n²) bug in the origin-file writer along the way),
  `tests/unit/test_tracker_quality.py` ground-truth CI floor. Also found and
  fixed a **segfault** at 20k particles/frame (`Tracker`/plugins hardcode
  `max_targets=10000`; `Frame.read()` now raises `ValueError` instead of
  silently overflowing fixed-size buffers).
- **2026-08-10 — Stage 1**: 1b global cost-ordered claiming (measured:
  precision 0.718→0.871, recall 0.648→0.812 at 1k density; fast_3d now
  nearly matches `myptv_3d_tracking`'s accuracy while staying faster). 1c
  postprocess wiring landed but **measured no benefit** on this dataset (net
  0–1 links for 5–13× runtime) — left off by default, contradicts the
  assumed root-cause table until re-measured on other data. 1d (bubble
  sorts) fell out of 1b for free.
- **Committed, not yet in this doc's Stage 2 text until this edit**: decided
  to replace Stage 2a's Savitzky-Golay prediction plan with a per-track
  constant-acceleration Kalman filter — see 2a below for the reasoning
  (O(1) vs proPTV's O(track length) per-link refit, and the innovation
  covariance subsumes Stage 3's adaptive search volume).

**Resume here:** Stage 2 (`quality_3d`) is designed but no code written yet.
Start with 2a (KF predictor) or 2c (cluster-local assignment, has a direct
reference in `_assignment.py` to port) — either is a clean starting point.

## Context

openptv2 currently ships **six** selectable tracking engines that overlap heavily,
and there is no trustworthy way to say which one is better. `docs/tracking-benchmark-results.md`
— the only comparison document — credits an engine (`hybrid_3d_corr`) that was
deleted in commit `1d52e12`, and quotes an 800k particle/s figure measured on an
isolated kernel call rather than an end-to-end run. Ground-truth particle IDs
exist on disk and are never used by any metric.

The goal is to end with **two** supported trackers plus reference implementations:

| Preset | Role | Constraint |
|---|---|---|
| `fast_3d` | throughput-optimal default | only cost-neutral quality changes; must hold 5k–20k particles/frame |
| `quality_3d` | accuracy-optimal | multi-frame prediction + global assignment + post-processing; speed secondary but must not be Python-bound |

Target operating point is **5k–20k particles/frame** (high-density LPT). That is
the decisive constraint: it rules the pure-Python plugins out as products (they
stay as reference implementations for parity tests), and it makes the O(N²)
loops in the Cython kernel the binding limit rather than an academic concern.

---

## What exists today

**Compiled (Cython, `src/openptv2/algorithms/`, built by `setup.py:67-80`):**

- **`trackcorr`** — the classic Willneff/OpenPTV 4-frame tracker. Predicts
  `2·X₁ − X₀`, projects a velocity quader's 8 corners into every camera
  (`track.py:694` `searchquader`), searches 2D pixel candidates per camera
  (`track_kernels_search.py:407`), ranks candidates by frequency across cameras,
  applies a 4-frame second-order criterion, and scores with
  `rr = (dl/lmax + acc/dacc + angle/dangle)/quali` (`track_kernels_corr.py:651`).
  Three-phase greedy conflict resolution at `track_kernels_corr.py:1411-1460`.
  Has a backward pass (`track.py:1216`) and new-particle seeding. Slowest, most
  information used (it is the only tracker that sees the images at all).
- **`fast_3d` / `track3d`** — current default. 3-frame, position-space only, three
  cascading levels (`track_kernels_track3d.py:69`). No backward pass, no seeding,
  no post-processing. Fastest.

**Pure Python plugins (`src/openptv2/plugins/`):**

- **`fast_3d_smooth`** — Savitzky-Golay order-3 velocity over ≤5 samples, vectorised
  across tracks in one `tensordot` (`fast_3d_smooth_tracking.py:62`).
- **`myptv_3d_tracking`** — constant-velocity prediction, two-tier radius.
- **`proptv_tracking`** — Gaussian-basis (GMM) fit of the whole track history,
  smoothed X/V/A, multi-term cost `CostWeights(1.0, 0.6, 0.3)`, optional backtrack.
- **`myptv_2d_tracking`** — per-camera 2D linking with cross-camera vote fusion.

The middle three are **the same tracker with three different velocity estimators**:
predict → `match_within_radius` → gap-bridge → seed unmatched → remap to frame
slots by `argmin`. That is the consolidation target.

---

## Findings that drive the plan

**F1 — `fast_3d`'s candidate ranking is provably wrong.**
`track_kernels_track3d.py:179-182` computes `X_curr − 2·X_cand + X_prev`. The
acceleration residual is `X_cand − 2·X_curr + X_prev`. The written expression
equals `−2·(X_cand − midpoint(X_prev, X_curr))`, so among candidates inside a box
centred on the *forward* prediction it ranks by proximity to a point *behind* the
particle — anti-correlated with the correct ordering. Level 2 (`:269-271`) has the
same structure with a half-step backward bias. Level 3 (`:332-334`) reduces to
`2·|pred − cand|` and is accidentally correct. Present since the original
translation (`e5750f6`). Fixing it costs nothing.

**F2 — Linking is first-come-first-served in particle-index order.**
`track_kernels_track3d.py:200-208`: candidates are sorted per particle, then the
first whose target is unclaimed wins. Particle 0 beats particle 500 regardless of
cost. `trackcorr` at least does three-phase conflict resolution; the plugins do a
per-component Hungarian. At 20k particles/frame this ordering bias is a
first-order error source, not a tie-breaker.

**F3 — Two O(N²) loops, no spatial index anywhere in the compiled path.**
`_find_closest_in_3d` (`:45`) linearly scans every frame-2 particle per query;
Level 2's neighbour-velocity averaging (`:226-239`) is a naked double loop over
`orig_parts`. At 700 particles this is free; at 20k it is ~400M distance tests per
level, ~1.2G per frame. The only KD-tree in the repo is in `_assignment.py:89`,
on the Python side.

**F4 — `MAX_CANDS = 32` becomes binding exactly at the target density.**
With 20k particles in a 100³ mm volume and a ±6 mm box, the expected candidate
count in the box is `20000 · (12³/100³) ≈ 35`. The top-32 buffer silently
truncates, and the insertion sort (O(32) per accepted candidate) plus the bubble
sort of the decision list (`:187-198`, O(32²)) become ~60M swaps/frame.

**F5 — `fast_3d` ignores the post-processing that already exists.**
`src/openptv2/tracking_postprocess.py` has `relink_trajectory_gaps` (`:244`),
`seed_cold_start` (`:173`), `enforce_reciprocity` (`:123`) — all tracker-agnostic,
operating on `ptv_is.#` on disk. It is wired only for the trackcorr path
(`plugins/default_tracking.py:67`, under `postprocess=True`). The repo's own
root-cause table attributes 60.4% of wrong links to detection dropouts and 19.8%
to cold-start ambiguity — precisely what these two functions address.

**F6 — No metric uses the ground truth that is on disk.**
`test_data/synthetic_turbulent/res/origin_<frame>.txt` column 0 is the true
particle id, and `img/camN.<frame>_targets` column 8 (`tnr`) carries it too, with
ghosts marked `−999` / `pid = −1`. `benchmarking/metrics.py:75` (`_match_frame`)
instead re-derives identity by unconstrained nearest-neighbour at `eps=1.0` mm —
which is *many-to-one* (two predicted tracks can claim the same true particle), so
it is not even a valid assignment. Ghosts are dropped silently by
`benchmark_utils.build_true_tracks:57` and never counted as captures.
Separately, `tracking_metrics.py:77` implements a genuine link-identity metric but
runs on a different generator and a different CLI subcommand; the two systems
never meet.

**F7 — Nothing pins tracker quality in CI.** `tests/unit/test_track3d.py:195`
asserts `npart == 2082` / `nlinks == 1765` — link *count*, not correctness. Every
change proposed below is unfalsifiable until a ground-truth regression exists.

**F8 — `dacc` means two incompatible things.** In `trackcorr` it is a cost
denominator (`track_kernels_corr.py:651`); in `fast_3d` it is the search-box
half-width (`track_kernels_track3d.py:133-135`). Same YAML key, so a user cannot
tune both. Minor but a real trap.

**F9 — Density benchmark does not exist.** `synthetic_turbulent` is 220 particles
in a 100³ mm volume (~0.0002 ppp) over 30 frames. `test_cavity` is 4 frames —
too short to show fragmentation at all. Neither exercises the target regime.

---

## Stage A — Land the ranking fix now (F1), ahead of everything else

Do this first, on its own, so it is a clean isolated diff.

1. `track_kernels_track3d.py:179-182` (Level 1) →
   `path_x_2[k] − 2·path_x_1[i] + path_x_0[prev_idx]` per component.
2. `track_kernels_track3d.py:269-271` (Level 2) → `path_x_2[k] − pred`, since
   `pred = curr + v̄` already carries the neighbour-averaged velocity, so the
   residual *is* the acceleration.
3. `track_kernels_track3d.py:332-334` (Level 3) → simplify to `path_x_2[k] − pred`
   (currently `2·(pred − cand)`, i.e. already distance up to a factor of 2 — this
   is a readability change, not a behaviour change).
4. Rebuild: `uv run python setup.py build_ext --inplace`.
5. New unit test in `tests/unit/test_track3d.py`: three particles in frame 1, each
   with a prev link giving a known velocity; in frame 2 place, inside the search
   box, both the velocity-continuing point and a decoy sitting behind the particle
   (nearer the `midpoint(prev, curr)`). Assert the link goes to the continuing
   point. This test fails on the current kernel and passes after the fix — that is
   the whole proof.
6. `tests/unit/test_track3d.py:195,240` assert `nlinks == 1765` on cavity and will
   likely move. Re-measure and update the constants, noting the old/new values in
   the commit message; do **not** relax them to inequalities.
7. `tests/unit/test_track3d.py:270` (`test_track3d_burgers_parity_with_cython`)
   compares the pure-Python path against Cython — the pure-Python
   `find_candidates_in_3d` (`track3d.py:16`) does candidate *generation* only, so
   parity should hold, but confirm it rather than assume.

Everything below then quantifies how much this was costing.

---

## Stage 0 — Honest measurement (blocking for Stages 1–4)

**0a. Exact pid-keyed metrics.** Add a pid-aware path to
`src/openptv2/benchmarking/metrics.py`: when the ground truth carries particle
ids (it always does for generated data), match predicted points to true points by
**stored pid** via the nearest-true-point-with-that-pid, and make the frame
matching **one-to-one** (`scipy.optimize.linear_sum_assignment`) instead of the
current unconstrained `tree.query`. Keep the eps-based path for data without ids.

**0b. Unify the two metric universes.** `benchmarking/metrics.py` (F/C/purity/pmt)
and `tracking_metrics.py:77` (yield/precision/FCR/gap-recovery) are both correct
and both wanted. Have the benchmark emit one row containing both, computed from
one run. Do not build a third metric.

**0c. Count ghosts.** Ghosts are `pid < 0`. Report **ghost-capture rate** = links
whose endpoint is a ghost / total links. This is the metric that will justify the
multi-term cost in Stage 2, and it is currently invisible.

**0d. Density sweep datasets.** `benchmarking/scenario.py` already parameterises
particle count; generate `synthetic_turbulent_{1k,5k,20k}` (same seed, same flow,
same volume) and check the smallest into `test_data/`, generating the larger two
on demand in the benchmark. This is the only way F3/F4 become visible.

**0e. One benchmark entry point.** Fold `scripts/benchmark_synthetic_turbulent.py`,
`benchmark_head_to_head.py`, and `benchmark_all_trackers_fair.py` into one
`scripts/bench_trackers.py` driven by `scripts/benchmark_utils.py`, emitting a
markdown table: `tracker | density | link_precision | link_recall | ghost_rate |
F | C | purity | pmt | ms/frame`. Delete the three absolute-path-bound scripts
(`benchmark_all_trackers_fair.py`, `synthetic_test_multi_density.py`,
`synthetic_test_trackcorr_vs_3d.py` — all hardcode `C:\Users\alex\Downloads\...`).

**0f. CI regression.** One `tests/unit/test_tracker_quality.py`, marked `slow`,
asserting `link_precision` and `link_recall` floors for `fast_3d` on the 1k case.
Floors set from the measured baseline, not aspirationally.

**Deliverable:** a table of where all six trackers actually stand at 1k / 5k / 20k.
Everything after this is measured against it.

---

## Stage 1 — Cost-neutral correctness in `fast_3d` (DONE)

These change no complexity class and are the whole of what `fast_3d` gets.

**1a.** Done in Stage A.

**1b. Global cost-ordered claiming (F2).** DONE. Per level (still a strict
1→2→3 cascade), candidate generation stays per-particle but claiming no longer
happens inside that loop: `(cost, i, k)` triples go into one edge buffer, sorted
once per level with `np.argsort`, then claimed in ascending-cost order with a
`path_next_1[i] < 0` / `path_prev_2[k] < 0` bitmap check. Measured effect on
`synthetic_turbulent` (1k density): precision 0.718 → 0.871, yield_recall
0.648 → 0.812, fragmentation 10.04 → 5.72. At the tuned 220-density point,
fast_3d now nearly matches `myptv_3d_tracking`'s precision/purity (0.974/0.970
vs 0.984/0.982) while still running faster (197ms vs 312ms/frame). Also fixed
a previously-documented hazard: `track3d` no longer mislinks under a too-tight
`dvxmax` in `test_tracking_synthetic.py`'s scenario (was a known "fails unsafe"
case; now fails safe like `trackcorr`) — see
`test_track3d_fails_safe_under_tight_dvxmax`.

**1c. Wire post-processing (F5).** DONE, but **the assumed payoff did not
hold** — measure before trusting root-cause tables. `plugins/default_tracking.py`
now runs `Tracker.postprocess()` (`seed_cold_start` → `relink_trajectory_gaps`
→ `enforce_reciprocity`) for `fast_3d` too, gated by `track.postprocess`
(reused `tracking_postprocess.py` unchanged, per plan). Measured on
`synthetic_turbulent` at both 220 and 1k density: **net effect was 0-1 extra
links total, while runtime went 5-13× (10s→57s at 220; 6.6s→89s at 1k)**. The
turbulent flow's `velocity_jitter=1.0` (Ornstein-Uhlenbeck) apparently makes
constant-velocity extrapolation — which `seed_cold_start`/`relink_trajectory_gaps`
both rely on — rarely land within their acceptance tolerance. Left **off by
default** (matches `tracking_presets.PRESET_CONFIGS["fast_3d"]["postprocess"] =
False`), opt-in via `track.postprocess: true`. Regression test:
`tests/unit/test_default_tracking_postprocess.py` (pins the wiring, not a
quality claim). Root-cause table's 60.4%/19.8% dropout/cold-start attribution
in `docs/tracking-benchmark-results.md` should be re-measured before being
used to justify further postprocess investment — it may be dataset-specific
(a slower-varying flow, or a real experimental dataset, might behave
differently) or the tolerance/guard conditions in `tracking_postprocess.py`
may need loosening; either needs actual measurement, not assumption.

**1d. Replace the bubble sorts (F4).** DONE as a side effect of 1b — the three
per-particle `decis_vals`/`decis_inds` bubble sorts in
`track_kernels_track3d.py` are gone; each level now does one `np.argsort` over
its whole edge list instead.

Verified against Stage 0's harness throughout (see measurements above), not
assumed.

---

## Stage 2 — `quality_3d`: the plugins' brain, in the kernel

New compiled tracker, selected as `quality_3d`, sharing Stage 1's candidate
generation and spatial index (Stage 3) but a smarter predictor and matcher.

**2a. Multi-frame prediction — constant-acceleration Kalman filter (revised
2026-08-10, superseding the earlier SG-smoothing plan below).** A per-track
KF (state `[x,y,z,vx,vy,vz,ax,ay,az]`, 9×9 covariance) replaces both fast_3d's
2-point extrapolation and the SG-smoothing idea:
  - **O(1) per track per frame** (a 9×9 predict + update), not O(track length)
    — this is what keeps `quality_3d` compiled-fast where proPTV's per-link
    GMM refit (`_smooth_history`, re-fit on every accepted link over the
    *whole* track) is the reason it's 20–30× slower than fast_3d at the same
    density (measured in Stage 0/1: 1470ms vs 216ms/frame at 220/frame, and
    proptv was excluded from the 5k/20k sweep entirely as impractical).
  - **The innovation covariance sizes the search radius directly** — replaces
    the fixed `dvxmax`/`dacc` box with an ellipsoid that's tight when a track's
    motion has been consistent and widens automatically after a gap or a
    turn. This *is* Stage 3's "adaptive search volume," not a separate feature
    to build twice.
  - Tracks with <2 history points fall back to a wide isotropic gate (no
    velocity estimate yet) — same cold-start handling as today's Level 3.
  - The KF replaces the *prediction* half only. Candidate generation (spatial
    box/grid) and assignment (2c below) are unchanged — a KF does not by
    itself resolve multi-track competition for one candidate.
  - Reference to validate against: `fast_3d_smooth`'s SG-velocity plugin
    (`fast_3d_smooth_tracking.py:47`) stays as-is per Stage 4 (a reference
    implementation for parity testing), since it already demonstrates the
    "smoothed prediction beats 2-point" effect cheaply in Python; the KF is
    the compiled, principled version of that same idea plus the free adaptive
    radius.

**2b. Multi-term cost.** Implement the three terms of
`tracking_cost.compute_multi_term_cost_matrix:67` inline (distance +
velocity-continuity + acceleration), weights from YAML with the proPTV plugin's
defaults `(1.0, 0.6, 0.3)` as the starting point. This is what suppresses ghost
captures — Stage 0c makes the effect measurable.

**2c. Cluster-local optimal assignment.** After candidate generation, the
prediction↔candidate graph decomposes into connected components exactly as
`_assignment.py:104-153` argues. Port that decomposition to Cython: components of
size 1 (the large majority) taken in bulk; components up to ~8 solved with a small
Hungarian; anything larger falls back to Stage 1b's cost-ordered greedy so the
worst case stays bounded. This is the concrete "brain from proPTV/MyPTV" — and
`_assignment.py` is already the reference to match, so parity is testable.

**2d. Backward pass.** Reuse `trackback_c`'s structure (`track.py:1216`) — run the
same forward kernel over the reversed sequence and keep links that both passes
agree on. Reciprocal agreement is a strong precision filter and costs one extra
pass, which is acceptable for `quality_3d`.

**Acceptance:** `quality_3d` must beat every Python plugin on link precision *and*
recall at 1k, and must still run at 5k/20k where the plugins do not.

---

## Stage 3 — Scale to 5k–20k particles/frame

**3a. Uniform grid spatial hash (F3).** Replace `_find_closest_in_3d`'s linear
scan. Per frame: cell size `h` = the largest search-box half-width ×2, compute
`cell = (ix, iy, iz)` per particle, counting-sort into a CSR layout
(counts → prefix sum → fill) in O(N), then query the 27 neighbouring cells. For
20k particles in 100³ mm at `h = 12` mm that is 729 cells — trivial memory. The
same index serves Level 2's neighbour-velocity query, killing the second O(N²)
loop. ~40 lines of Cython, no new dependency.

**3b. Adaptive candidate cap (F4).** `MAX_CANDS = 32` truncates at exactly the
target density. Either size the buffer from measured density at run time, or
shrink the search box when the cell occupancy says the box will overflow. Report
a truncation counter in the benchmark so silent truncation stops being silent.

**3c. Parallel candidate generation.** `prange` the candidate-generation phase
(pure read, embarrassingly parallel with the grid). Claiming stays serial after
the global sort — or reuse the `__sync_bool_compare_and_swap` atomic-claim pattern
already established in `track_kernels_corr.py:24-40`.

**3d. Re-measure the density curve.** The whole point of Stage 0d. Expect the
crossover where the grid pays off somewhere around 2k particles/frame; confirm it
rather than assuming it.

---

## Stage 4 — Consolidation and deletion

**4a. Collapse the three predictive plugins into one.** `fast_3d_smooth`,
`myptv_3d_tracking`, and `proptv_tracking` differ only in `estimate_velocity(history)`
(2-point / SG / GMM). One `plugins/_predictive_tracker.py` with a pluggable
estimator; the three names become thin configs kept as **reference
implementations for parity tests against `quality_3d`**, not as products.

**4b. Fix the frame remap.** All three plugins map trajectory points back to frame
slots with `argmin` over positions (`fast_3d_smooth_tracking.py:331-334`,
`myptv_3d_tracking.py:262-266`, `proptv_tracking.py:404-405`). This is O(N) per
link and mis-assigns when two particles coincide. Carry the candidate index
through the tracker instead of recovering it by search.

**4c. Small fixes.** `myptv_2d_tracking.py:203` reads `track.dvmx`, a key that
exists nowhere — it has always silently used the 20 px fallback. `proptv_tracking.py:233`
has `np.where(seeded, cfg.maxvel, cfg.maxvel)` — a dead branch where the seeded
case should use a tighter acceleration-scale radius. Delete the orphaned
`src/openptv2/algorithms/track_kernels_hybrid.{c,pyd,html}` left behind by
commit `1d52e12`.

**4d. Split the `dacc` overload (F8).** Introduce `dacc_search` for `fast_3d`'s
box half-width, keep `dacc` as `trackcorr`'s cost denominator, and default the new
key to the old one so existing YAML keeps working.

**4e. Rewrite `docs/tracking-benchmark-results.md`** from Stage 0's measured table.
It currently benchmarks a deleted tracker.

---

## Critical files

| File | Role in this plan |
|---|---|
| `src/openptv2/algorithms/track_kernels_track3d.py` | Stage A, Stage 1, Stage 3a/3b/3c — the main edit target |
| `src/openptv2/algorithms/track3d.py` | driver; parameter plumbing for new keys |
| `src/openptv2/benchmarking/metrics.py` | Stage 0a/0b/0c — pid-exact, one-to-one matching |
| `src/openptv2/tracking_metrics.py` | link-level metric to fold in (do not reimplement) |
| `src/openptv2/benchmarking/scenario.py`, `datawriter.py` | Stage 0d density datasets |
| `scripts/benchmark_utils.py` | harness to extend; `bench_trackers.py` replaces the three ad-hoc scripts |
| `src/openptv2/tracking_postprocess.py` | Stage 1c — reuse unchanged |
| `src/openptv2/plugins/_assignment.py` | Stage 2c reference implementation to port and parity-test against |
| `src/openptv2/plugins/fast_3d_smooth_tracking.py:47` | Stage 2a SG coefficients to hardcode |
| `src/openptv2/tracking_cost.py:67` | Stage 2b cost terms to inline |
| `src/openptv2/plugins/loader.py:39-52`, `tracking_presets.py:34-97` | register `quality_3d`, retire aliases |
| `src/openptv2/tracking_registry.py` | `speed_ranking`/`accuracy_ranking` are hand-assigned labels; replace with measured values |

---

## Verification

```bash
cd C:\Users\alex\projects\openptv2

# after any algorithms/ change
uv run python setup.py build_ext --inplace

# correctness gates
uv run pytest tests/unit/test_track3d.py tests/unit/test_track.py \
              tests/unit/test_tracking_synthetic.py -v
uv run pytest tests/unit/test_benchmarking.py -v

# ground-truth quality (Stage 0f)
uv run pytest tests/unit/test_tracker_quality.py -v -m slow

# the table that decides everything
uv run python scripts/bench_trackers.py --density 1000,5000,20000

uv run ruff check . && uv run mypy src/openptv2/
```

Per-stage gates:

- **Stage A** — the new decoy test fails before and passes after; cavity/burgers
  link counts re-measured and pinned.
- **Stage 0** — `pmt`/purity computed by pid must match the eps-based value on
  `synthetic_turbulent` (sparse enough that both agree); they must diverge on the
  5k case, which is the proof the old metric was approximating.
- **Stage 1** — link precision and recall both up at 1k; ms/frame within noise of
  baseline. `test_track3d.py`'s `nlinks` assertion **will** move; pair it with a
  quality floor rather than replacing it with an inequality.
- **Stage 2** — `quality_3d` ≥ best Python plugin on precision and recall at 1k;
  Stage 2c must produce assignments matching `_assignment.match_within_radius` on
  the same inputs (parity test).
- **Stage 3** — 20k case completes; measured ms/frame scales ~linearly in N, not
  quadratically; candidate-truncation counter is zero.
- **Stage 4** — plugin outputs unchanged after the collapse (byte-identical
  `ptv_is.#` on `synthetic_turbulent`).

---

## Out of scope

- Porting proPTV's 2D triangulation pipeline (the plugin deliberately avoids it).
- Shake-The-Box iterative position refinement — `plugins/stb_4d_refinement.py`
  exists but is a position refiner, not a tracker, and has no `Tracking` class.
- `myptv_2d_tracking`'s cross-camera vote fusion — its dense `linear_sum_assignment`
  over N×N is O(N³) and cannot reach the target density. Fix 4c and leave it.
- `trackcorr` algorithm changes. It stays as the image-space tracker for datasets
  where 2D information matters; only the shared post-processing touches it.
