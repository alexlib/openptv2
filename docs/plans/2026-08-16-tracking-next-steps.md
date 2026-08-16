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
reproduced perfectly (no spurious points, same start frame). Reference
implementation: `scripts/` has none yet; the throwaway used during the
investigation is reproduced in §5 and should be promoted into
`openptv2/benchmarking/metrics.py` as `e_track()`.

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

### 3.3 C gate fidelity for track3d

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

### 3.4 4BE follow-ups

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

### 3.5 Where the speed actually is

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

Ouellette E_track, to be promoted into `benchmarking/metrics.py`:

```python
# a true track is "perfect" if some predicted track matches only it and
# starts on the same frame; E_track = 1 - perfect / n_true
perfect = set()
for pid, pts in pred.items():
    pts = sorted(pts, key=lambda p: p[0])
    ids = {match(f, (x, y, z)) for (f, x, y, z) in pts}   # nearest true id, tol 1.0
    if len(ids) != 1:
        continue
    tid = ids.pop()
    if tid is not None and pts[0][0] == true_start[tid]:
        perfect.add(tid)
e_track = 1.0 - len(perfect) / len(true_tracks)
```

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
