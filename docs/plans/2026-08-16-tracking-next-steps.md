# Tracking work — state and next steps (2026-08-16)

Continues `2026-08-15-tracking-quality-overhaul.md`. That document's Stage 1
(warmup) and Stage 2 (corrective pass) conclusions still stand; this one
records what changed on 2026-08-15/16 and what to do next.

Read the **Corrections** section before trusting any number in the older
plan docs — two metrics were being read wrongly, and some earlier
conclusions were built on them.

---

## 1. What landed

| commit | change |
|---|---|
| `ca06720` | `read_correspondences` on flat zero-particle frames; `write_linkage` no longer writes ASCII when a store is attached |
| `0e55c4f` → `c3a30ff` | GUI "Measure distances" (later removed) |
| `34dbf3f` | Removed Measure distances; added **Tracking → Visualize 3D positions** (interactive multi-frame 3D scatter, hover to identify, click to measure displacement/velocity/acceleration) |
| `4bd4473` | Right-click handling no longer sticks after an early return (`rclicked` reset moved to the top of `right_click_process`) |
| `433c5f6` | `Tracker.restart()` clears stale linkage via new `RunStore.clear_linkage()`; removed 11 dead test files |
| `6a081e0` | Epipolar plugin honoured its preset's direction; `_CORE_PRESETS` gained the epipolar keys |
| `bfccfa9` | **`relink_trajectory_gaps` placeholder bridging — to be reverted, see §3.1** |
| `cb7d113` | **New `4be` tracker** (Ouellette four-frame best estimate), stereo-3D only |
| `a3a44e2` | 4BE: 3MA fallback for candidates unsupported at n+2 (default; `strict_support=1` restores the literal paper rule) |
| `b6ed53b` | Compiled the 4BE wrapper (was interpreted); `_find_closest_in_3d_grid` → `@cython.cfunc` |

---

## 2. Corrections — read this first

**2.1 `pmt` is not a track-quality rate.** `benchmarking/metrics.py:206-223` computes
it over *predicted* tracks: the fraction whose points are ≥2/3 one true
particle, divided by the number of predicted tracks. A 2-point fragment
scores "correct" automatically, so a fragmenting tracker inflates it.
`E_track = 1 - pmt` is **wrong** and was used in earlier reports.

Use Ouellette's definition instead — fraction of **true** tracks not
reproduced perfectly. **DONE:** promoted as
`openptv2.benchmarking.metrics.e_track()`, wired into
`scripts/benchmark_utils.combined_metrics`, with
`tests/unit/test_e_track.py`.

Note the parenthetical above ("no spurious points, same start frame") is
itself incomplete, and §5's snippet implemented exactly that — it never
checks the predicted track *covers* the true one, so a 2-point fragment
scores as a perfect reproduction of a 30-point trajectory. That is the same
inflation this section identifies in `pmt`. The shipped `e_track` requires
all three conditions: one fragment, nothing foreign or unmatched in it, and
exactly the true track's frames. Six of the ten tests fail against the
`(one id + same start frame)` form, verified by mutating the implementation
back to it.

It is a strict, all-or-nothing measure, so it returns a failure breakdown
(`n_fragmented` / `n_contaminated` / `n_incomplete` / `n_missed`) alongside
the scalar, and it is only informative with **gap bridging enabled** — see
§3.6.

**2.2 The synthetic ground truth contains gaps.** 5796 step-1 links, but
476 links of step 2-4, and **214 of 236 tracks (91%) contain at least one
gap**. Any comparison that assumes contiguous truth is wrong.

**2.3 track3d diverges from the C original.** The C (`track3d.c`) passes
`dvxmax/dvymax/dvzmax` to `find_candidates_in_3d` at **all three levels** and
never uses `dacc` as a gate. The Cython port substitutes `ax = dacc if
dacc > 0` at levels 1 and 2 (`track_kernels_track3d.py:251`). The port also
*fixed* a C cost bug: C computes `curr - 2*next + prev`, the port computes
`next - 2*curr + prev`, which is the true residual from the prediction.

**2.4 The tracking pipeline is I/O-bound.** Kernel cost in isolation is
0.213 ms/step (3MA) and 1.573 ms/step (4BE), against ~270 ms/frame end to
end. **The kernel is under 1% of runtime.** Optimising kernels will not make
tracking faster; frame read/write, the zarr store and the SoA↔AoS sync are
where the time goes.

---

## 3. Next steps, in order

### 3.1 Revert placeholder gap bridging; restore cross-frame gap links — DONE

Landed. `relink_trajectory_gaps` writes a single cross-frame link again; the
step is recovered by the reciprocal-pointer search below, exposed as
`tracking_postprocess.link_step` / `back_link_step` (cap `MAX_LINK_STEP = 3`).
All three consumers are gap-aware: `enforce_reciprocity` (searches
`1..max_step` instead of comparing k against k+1 only), `benchmarking/runner.py
::read_trajectories`, and `storage/seal.py` (keeps a 3-frame trajid/next
history). The two walkers fall back to step 1 when nothing reciprocates, so
non-reciprocal linkage (no postprocess pass) behaves exactly as before.

Measured, `priority_segment_3d` on `synthetic_turbulent`:

| postprocess | tracks | points | links | precision | yield | mean len |
|---|---|---|---|---|---|---|
| off | 946 | 6748 | 5802 | 0.9667 | 0.894 | 7.13 |
| on | 660 | 6756 | 6096 | 0.9642 | 0.937 | 10.24 |

Links up 294, yield up, precision flat (it was 0.878 with placeholders). The
+8 points come from `enforce_reciprocity`/`seed_cold_start` changing
reachability, not from fabricated particles — relink now writes no particles
at all.

Original write-up follows.

`bfccfa9` made `relink_trajectory_gaps` fill a bridged gap with an
*interpolated placeholder particle* per skipped frame. This is wrong on two
counts:

- It fabricates a measurement at a frame where the particle was never
  observed. Those points then enter Lagrangian velocity/acceleration
  statistics, which is this project's actual output.
- Ground truth (and `calculate_tracking_metrics`' `gap_recovery_rate`
  machinery) represents a gap as a **link with step > 1**. The placeholder
  form emits `(k,k+1)+(k+1,k+2)` where truth has `(k,k+2)`, so 589 of 593
  bridge links scored as wrong — precision fell 0.967 → 0.878 for a bridging
  pass that was 97% correct about *which particle* to join.

**Do:** restore the cross-frame `next` pointer, and make the three consumers
that assume `next[k][i]` indexes frame `k+1` gap-aware:

1. `tracking_postprocess.enforce_reciprocity` — currently compares `next_k`
   against frame `k+1` only, so it severs every bridge it sees.
2. `benchmarking/runner.py::read_trajectories` — walks with an unconditional
   `cur_frame += 1`.
3. `storage/seal.py` — carries `prev_trajids` only from the immediately
   previous frame (guard 1, ~line 100), so a bridged particle starts a new
   trajectory id.

The linkage format cannot express "which frame does this index point into".
Resolve it **without a format change** by searching for the reciprocal:
accept the smallest step `s` in `1..max_gap+1` where `prev[k+s][j] == i`.
Relink already sets that reciprocal pointer, so it is well-defined; take the
smallest `s` so a genuine step-1 link always wins.

**Success:** `postprocess` on `priority_segment_3d` raises links without
lowering precision; `test_relink_trajectory_gaps_bridges_missing_frame` is
rewritten to assert a cross-frame link (not a placeholder); no new points
appear in any trajectory.

### 3.2 `max_velocity_err` is the wrong parameter — DONE

Landed. `relink_trajectory_gaps`' `max_velocity_err` is now `max_accel_err`,
the tolerance is gap-scaled as `0.5 · max_accel_err · (gap+1)²/2`, and all
three callers (`tracker.py`, `track_assisted.py`,
`fast_3d_smooth_tracking.py`) pass `dacc` instead of `dvxmax`.

The extra 0.5 is measured, not assumed — the plan's literal `dacc·(gap+1)²/2`
puts the gap-1 tolerance at 12 mm for `dacc=6`, past the knee. Bridges scored
against ground-truth identity (this is the sweep re-run on the restored
cross-frame representation, so the numbers differ slightly from §3.2's
original table):

| max_accel_err | tol @ gap 1 | bridges | % correct | true gaps recovered |
|---|---|---|---|---|
| 1.5 | 3.0 | 137 | 92.0% | 26.5% |
| 2.0 | 4.0 | 219 | 92.7% | 42.6% |
| **3.0** | **6.0** | **309** | **92.6%** | **60.1%** |
| 4.0 | 8.0 | 330 | 90.9% | 63.0% |
| 6.0 | 12.0 | 347 | 85.6% | 62.4% |

`0.5·dacc·4/2 = dacc` lands the gap-1 tolerance on the knee. End to end
(`priority_segment_3d`): 946 → 637 tracks, links 5802 → 6119, precision
0.9667 → 0.9596, yield 0.894 → 0.936, mean length 7.13 → 10.61.

Also fixed in passing: the velocity estimate read `frames[k-1]`
unconditionally, which is the wrong frame when the incoming link is itself a
bridge from an earlier pass. It now resolves the real source frame with
`back_link_step` and divides the displacement by that step.

Original write-up follows.

`Tracker.postprocess` passes `max_velocity_err=float(tpar.dvxmax)`. The
tolerance is applied to a position that has **already been
velocity-extrapolated**, so the residual is acceleration-scale, not
velocity-scale. It only looks sane on the synthetic set because
`dvxmax == dacc == 6.0` there.

This is a live hazard: warmup suggested `dvxmax ≈ 52 mm` on the JHU data,
against ~9 mm particle spacing. Relink would accept 52 mm bridges.

Measured sweep (bridge correctness scored against ground-truth identity,
independent of representation):

| mve | bridges | %correct | true gaps recovered |
|---|---|---|---|
| 2.0 | 48 | 100% | 10.1% |
| 3.0 | 112 | 99.1% | 22.9% |
| 4.0 | 188 | 98.9% | 38.7% |
| **6.0** | 286 | 97.2% | 57.8% |
| 8.0 | 314 | 95.2% | 62.0% |
| 12.0 | 336 | 91.1% | 63.0% |

**Do:** derive the tolerance from `dacc` and the gap length (about
`dacc·(gap+1)²/2`), not from `dvxmax`. Knee is at 6-8 mm on this dataset,
which such a formula should roughly reproduce for `dacc = 6`.

### 3.3 C gate fidelity for track3d — RESOLVED: keep the port, don't restore C

The framing below ("restore the C behaviour or make the choice explicit") was
a false choice, on two counts.

**1. `dacc = 0` already IS the C behaviour**, today, with no code change:
`track_kernels_track3d.py`'s `ax = dacc if dacc > 0.0 else dx` falls back to
`dx/dy/dz` at levels 1 and 2, and `dacc` feeds nothing else in that kernel
(only `ax/ay/az` and the grid cell size). So the two are the *same code path*
whenever `dacc == dvxmax` — measured bit-identical:

| config | tracks | prec | yield |
|---|---|---|---|
| port `dacc=6 dvxmax=6` | 946 | 0.9667 | 0.8943 |
| C `dacc=0 dvxmax=6` | 946 | 0.9667 | 0.8943 |

That is why the divergence was invisible on this dataset.

**2. The port is a strict superset, and modestly better.** It decouples the
seeded-step box (levels 1-2) from the unseeded box (level 3); C forces them
equal. Link precision / yield (both endpoints matched — see the metric
warning below for why E_track is *not* used here), `priority_segment_3d`,
postprocess off:

| config | 220/f prec | 220/f yield | 970/f prec | 970/f yield |
|---|---|---|---|---|
| port `dacc=3 dvxmax=6` | 0.9766 | 0.8852 | 0.9443 | 0.8738 |
| **port `dacc=4 dvxmax=6`** | 0.9748 | **0.8949** | 0.9346 | 0.8708 |
| C `dacc=0 dvxmax=4` | 0.9770 | 0.8665 | 0.9448 | 0.8765 |
| C `dacc=0 dvxmax=6` | 0.9667 | 0.8943 | 0.9157 | 0.8670 |
| C `dacc=0 dvxmax=10` | 0.9427 | 0.8914 | 0.8764 | 0.8622 |

What drives quality is the *size* of the seeded box, not which parameter
names it — the wide-box C rows are the worst at both densities. The port's
edge is that it can tighten level 1 without starving level 3's cold search:
worth ~3 points of yield at 220/frame (0.895 vs 0.867 at equal precision),
and roughly parity at 970/frame. Modest, but free.

**Metric warning — §5's `e_track` snippet is wrong, do not promote it as
written.** It checks "all points map to one true id" + "same start frame" but
never checks that the predicted track *covers* the true one, so a 2-point
fragment starting on the right frame scores as a perfect reproduction. This
is the same inflation §2.1 identifies in `pmt`, and it is not theoretical: on
this data it ranks `dacc=1` best (E=0.068, 220 of 236 "perfect") at
yield 0.55 with 3054 predicted tracks for 236 true ones. Adding the coverage
check (same end frame and same point count) makes E_track saturate at
0.94-0.99 for *every* config, because 91% of true tracks contain a gap (§2.2)
and a gap is unbridgeable with postprocess off — so it stops discriminating
instead. Before promoting `e_track()` into `benchmarking/metrics.py`: add the
coverage check, and evaluate it only with gap bridging enabled.

Caveats on the evidence: **no real-data discrimination was possible.**
`test_cavity` is 4 frames (mean track length 1.1, 415 links) — far too short,
consistent with this repo's own "directional checks only" note; the real-data
gate still awaits a well-conditioned experiment. And mean track length moves
*against* quality: `dvxmax=10` gives the longest tracks and the worst
precision.

**Done:** the required documentation ("`dacc` is the knob that controls
seeded-step search") is now in `_track3d_full_loop`'s docstring with these
numbers.

Original write-up follows.

Restore the C behaviour (`dvx/dvy/dvz` gate at all levels) or make the choice
explicit in config. Measured impact is modest, so this is fidelity work, not
a quality fix:

| gate | tracks | mean len | precision |
|---|---|---|---|
| `dacc`=5.5 (current port) | 915 | 7.37 | 0.960 |
| `dvxmax`=10 (C-equivalent) | 817 | 8.26 | 0.943 |

Whatever is decided, document that in the current port **`dacc` is the knob
that controls seeded-step search** — a user carrying C-era intuition will
tune `dvxmax` and change nothing for particles already being tracked.

### 3.4 4BE follow-ups — DONE

**Tests landed.** `tests/unit/test_track4be.py`, 11 kernel-level tests
covering eq. 12, `strict_support`, both conflict rules, the NN fallback for
unseeded particles, the 3MA degradation when n+2 is missing, and empty
frames. The eq. 12 test was verified by mutating the kernel to
`q + (x1 - x0)`: the obvious one-candidate scene passes under that mutant
(with a wide velocity window the wrong estimate still finds support), so the
test uses two candidates whose costs invert between the two formulas.

**A silent bug found while measuring:** `four_be_tracking.py` never called
`tracker.postprocess()`, so `track.postprocess` was a no-op for 4BE and the
"4BE + gap bridging" measurement below was impossible. Fixed with the same
opt-in hook `default_tracking`'s priority_segment_3d branch has.

**Measured** (both ground-truth sets, `BASE_OVERRIDES`):

| run | 220/f tracks | prec | yield | 970/f tracks | prec | yield |
|---|---|---|---|---|---|---|
| 3MA | 946 | 0.9667 | 0.8943 | 3553 | 0.9157 | 0.8670 |
| **3MA + bridging** | 637 | 0.9596 | **0.9362** | 2635 | **0.9059** | **0.8890** |
| 4BE paper (give-up) | 1573 | **0.9851** | 0.8128 | 10130 | 0.9412 | 0.6616 |
| 4BE greedy conflicts | 1007 | 0.9700 | 0.8879 | 4215 | 0.8533 | 0.7870 |
| 4BE paper + bridging | 1115 | 0.9335 | 0.8371 | 6733 | 0.8147 | 0.6758 |
| 4BE greedy + bridging | 701 | 0.9625 | 0.9281 | 3212 | 0.8367 | 0.8034 |

**1. Cost-ordered greedy claiming: rejected as a default, kept as a flag.**
It buys a lot of yield (0.813 → 0.888 at 220/f; 0.662 → 0.787 at 970/f) but
pays in precision, and the price scales with density: -1.5 pp at 220/f,
**-8.8 pp** at 970/f. Ouellette's result (conflict-breaking degrades every
heuristic but NN) holds. Available as
`track4be_loop_fast(greedy_conflicts=1)` / `track4be.GREEDY_CONFLICTS`.

**2. "4BE's clean-but-short output is the ideal input for gap bridging" —
refuted.** Bridging *4BE paper* output costs 5.2 pp precision at 220/f and
12.7 pp at 970/f, far worse than bridging 3MA output (-0.7 / -1.0 pp).

The mechanism is visible in the last row. 4BE-paper fragments end at two
different kinds of place: genuine detection gaps, and *conflicts it
deliberately declined*. `relink_trajectory_gaps` looks for an unlinked end
facing an unlinked start — which is exactly what a declined conflict looks
like — so it re-creates the very links 4BE refused, on the evidence 4BE
judged too ambiguous. Resolving conflicts inside the tracker first
(`greedy + bridging`) leaves only detection gaps for the bridger, and
precision recovers (0.9335 → 0.9625 at 220/f). Give-up-on-conflict and gap
bridging are working against each other; do not pair them naively.

**3. `priority_segment_3d` + gap bridging remains the best default** at both
densities. 4BE's edge is precision without bridging at moderate density
(0.9851 at yield 0.813) — a "few but trustworthy tracks" operating point. It
degrades badly with density (mean length 2.87, yield 0.66 at 970/f).

Original write-up follows.

4BE is markedly more accurate but more fragmented than 3MA:

| tracker | tracks | mean len | perfect/236 | E_track | precision | yield |
|---|---|---|---|---|---|---|
| `priority_segment_3d` (3MA) | 946 | 7.13 | 158 | 0.331 | 0.967 | 0.894 |
| `4be` | 1573 | 4.29 | 225 | **0.047** | 0.985 | 0.813 |

Worth separating the two remaining causes of its fragmentation:

- **Give-up-on-conflict.** The paper prefers it (Munkres degraded every
  heuristic but NN), but it does end tracks. Try cost-ordered greedy claiming
  (what track3d does) as a variant and measure.
- **Real gaps.** 4BE cannot bridge a missing frame at all. §3.1 is the fix,
  and 4BE's clean-but-short output is the ideal input for gap bridging —
  bridging is only safe when the fragments being joined are correct.

Also: 4BE has **no dedicated tests**. It needs at least a kernel-level test
pinning the eq. 12 estimate (`x̃ = 2q − x1`) and the give-up-on-conflict rule.

### 3.6 Track-level results, once E_track works

With the corrected `e_track` and gap bridging on, the metric discriminates
(0.67–0.99, versus pinned at 0.94–0.99 with bridging off):

| run | E_track 220/f | perfect | E_track 970/f | perfect |
|---|---|---|---|---|
| 3MA `dacc=6` no bridging | 0.9407 | 14 | 0.9626 | 38 |
| 3MA `dacc=6` + bridging | 0.6907 | 73 | 0.8819 | 120 |
| 3MA `dacc=3.6` + bridging | 0.7754 | 53 | **0.8189** | **184** |
| 4BE paper + bridging | 0.9153 | 20 | 0.9911 | 9 |
| 4BE greedy + bridging | **0.6737** | **77** | 0.9429 | 58 |

Three things this shows that the link-level metrics could not:

1. **Gap bridging is a far bigger win than yield suggested.** Perfectly
   reproduced trajectories go 14 → 73 (5.2x) at 220/frame and 38 → 120
   (3.2x) at 970/frame. Yield moved 4 points; whole trajectories multiplied.
   This is the real justification for turning it on by default.
2. **The `dacc = 0.6 x dvxmax` change is confirmed where it matters and is a
   genuine trade elsewhere** — at 970/frame it is the best row by a wide
   margin (184 vs 120 perfect), at 220/frame it is worse on E_track (53 vs
   73) while better on precision. Same density dependence the §3.3 sweep
   found; the single constant favours the dense case.
3. **Essentially all remaining failure is fragmentation, not wrong links.**
   Across every row: `n_fragmented` is 156–1006 while `n_contaminated` is
   1–15 and `n_incomplete`/`n_missed` are ~0. The trackers are producing
   *correct but split* trajectories. Precision is already ~0.96; chasing it
   further is chasing the wrong number. The remaining quality is in
   stitching — which is what master-plan's "Track repair & stitching" row
   already anticipates as a long-gap / spatio-temporal stitcher.

### 3.5 Where the speed actually is — PROFILED

Measured, not inferred. `priority_segment_3d`, `synthetic_turbulent`,
30 frames, 225 particles/frame, postprocess off; store methods wrapped
directly rather than read off a profiler's inclusive times (zarr's async
bridge double-counts those).

| component | calls | total | ms/call | % wall |
|---|---|---|---|---|
| `store.write_targets` | 120 | 1.520 s | 12.67 | 26.3% |
| `store.write_linkage` | 30 | 1.327 s | 44.23 | 23.0% |
| `store.read_linkage` | 30 | 0.428 s | 14.26 | 7.4% |
| `store.write_correspondences` | 30 | 0.294 s | 9.79 | 5.1% |
| `_sync_soa_to_aos` | 58 | 0.023 s | 0.40 | **0.4%** |
| `store.read_correspondences` | 30 | 0.020 s | 0.67 | 0.4% |

**1. `_sync_soa_to_aos` is not a suspect.** §3.5 named it alongside the
frame read/write; it is 0.4% of wall. Drop it from the list.

**2. The unit cost is zarr *array creation*, ~11-12 ms each.** It is
strikingly stable — `write_targets` creates 1 array (12.67 ms/call),
`write_linkage` creates 4 (prev/next/pos/prio; 44.23 ms/call ≈ 11 ms each),
`write_correspondences` creates 1 (9.79 ms). The same ~12 ms/array shows up
on `test_cavity_small` through `pyptv_batch`. Each creation is a directory,
a `zarr.json`, and a chunk file, behind zarr v3's sync-over-async bridge.

**3. Nothing is redundantly re-created.** Instrumenting `create_array`:
270 calls, **100% new**, 0 re-creations of an existing same-shape array
(`Tracker.restart()` clears linkage by design, per `433c5f6`). So array
reuse / write-in-place buys nothing here.

**4. Correction — the benchmark path overstates tracking I/O.** That 26%
`write_targets` row is an artefact of `benchmarking/runner.run_tracker`,
which skips the sequence stage, so the frame buffer imports ASCII targets
into a fresh store *during tracking*. Instrumenting a real
`pyptv_batch` sequence→tracking run: the tracking phase issues **zero**
`write_targets` calls (they all happen in sequence, via `gui/ptv.py:249`).
Profile `pyptv_batch`, not `run_tracker`, for anything speed-related.

**What would actually help** — both change the on-disk format, so neither is
a drive-by fix, and both need a `docs/releases/` runbook:

- Pack `prev`/`next`/`pos`/`prio` into one array per frame instead of four:
  4x fewer creations, ~33 ms/frame off `write_linkage`.
- Go further and store one array per *stream* for the whole run plus an
  offset index — the layout `seal` already builds for `trajectories/` —
  making creations O(1) in frames rather than O(frames).

Not implemented here: §3.5 asks for the profile, and the payoff is gated on a
format decision rather than on a local optimisation.

Original write-up follows.

Per §2.4 the kernel is <1% of runtime. If tracking speed matters, profile
`fb.write_frame_from_start` / `read_frame_at_end`, the zarr store write path,
and `_sync_soa_to_aos` — not the tracking kernels.

---

## 4. Deferred / unchanged

- Warmup's `p99 × 3.0` margin still overshoots (see the 2026-08-15 plan,
  Stage 5 part 2 item 1). Its tuned `dvxmax`/`dacc` degraded precision on
  every measured synthetic condition. Keep recommending dry-run + compare.
- proPTV 500_25/500_30 comparison still needs a format adapter.
- Hungarian-based trackers (`nearest_hungarian_3d`, `kalman_hungarian_3d`,
  `fast_3d_smooth`) are working against Ouellette's conflict-breaking result;
  keep them deprioritised.

---

## 5. Verification snippets

Ouellette E_track: **shipped** as
`openptv2.benchmarking.metrics.e_track(true_tracks, pred_tracks, eps)`, and
included in `scripts/benchmark_utils.combined_metrics`. Use that; do not
re-derive it.

> The snippet that used to sit here was wrong and is deleted rather than
> corrected in place, because it was copied once already. It checked only
> "every point maps to one true id" plus "same start frame", with no
> coverage check, so a 2-point fragment counted as a perfect reproduction of
> a 30-point trajectory — the same inflation §2.1 identifies in `pmt`, which
> is exactly what it was introduced to avoid. It ranked a configuration
> producing 3054 tracks for 236 true ones as the best of its sweep, at a link
> yield of 0.55. See §2.1 and §3.6.

Commands:

```bash
uv run python setup.py build_ext --inplace      # after any algorithms/*.py change
uv run pytest tests/unit/test_track3d.py tests/unit/test_track.py -q
uv run pytest tests/unit/test_tracking_postprocess.py tests/unit/test_run_store.py -q
uv run python scripts/benchmark_all_gui_trackers.py
```

Dataset used throughout: `test_data/synthetic_turbulent`, frames
10001-10030, 220 particles/frame, 236 true trajectories, mean spacing
9.18 mm, mean displacement 2.0 mm/frame (ξ = 0.218 — the hard end of
Ouellette Fig. 9).

Reference: Ouellette, Xu & Bodenschatz, *A quantitative study of
three-dimensional Lagrangian particle tracking algorithms*, Exp. Fluids
40:301-313 (2006).
