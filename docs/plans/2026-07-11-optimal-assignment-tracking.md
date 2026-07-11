# Optimal Link Assignment for Tracking (Auction Algorithm) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the per-particle *greedy* link decision in the tracker with a *globally optimal* min-cost bipartite assignment per frame pair, using the auction algorithm, so that contested candidates are resolved optimally instead of first-come-first-served.

**Architecture:** The Cython kernels already compute, for each source particle, a short list of candidate targets and a scalar cost `rr = (dl/lmax + acc/dacc + angle/dangle) / quali` (see `track_kernels_corr.py:630`). Today those are stored in `path_decis` (costs) / `path_linkdecis` (target indices) and reduced greedily. This plan keeps the candidate generation and cost function unchanged and swaps only the *reduction* step for an optimal assignment over the sparse cost matrix. It is contained to one new kernel + one call site per frame pair; the 3D and 2D loops share it.

**Tech stack:** Python/NumPy/Cython 3 (same runtime model as the rest of `algorithms/`). Pure-Python auction first (correctness + tests), then a `nogil` Cython port for speed.

---

## Background: why this is the highest-ceiling small change

- **Greedy is locally optimal, globally wrong under contention.** When two source particles both have their best candidate = target *t*, greedy gives *t* to whichever is processed first; the other takes its 2nd choice or is dropped. The auction/Hungarian assignment minimizes the *total* cost across all particles simultaneously, so both get their globally-best consistent pairing.
- **Everything expensive is already done.** Candidate search (epipolar / 3D neighborhood) and the `rr` cost are computed. Assignment only re-decides among already-scored candidates — cheap and self-contained.
- **Sparse & near-linear.** Each particle has ≤ `MAXCAND` candidates, so the cost matrix is very sparse. Bertsekas' auction algorithm on sparse bipartite graphs is near-linear in practice and trivially parallelizable over the bidding phase.

---

## Current-state anchors (read these first)

- Cost + greedy store: `src/openptv2/algorithms/track_kernels_corr.py:630` and `:722` (`rr` formula), `path_decis`/`path_linkdecis` arrays.
- Link SoA fields: `src/openptv2/algorithms/tracking_frame_buf.py:533` (`path_prev`, `path_next`), `:560` (init).
- Forward driver: `src/openptv2/algorithms/track.py` `trackcorr_c_loop` (2D), `track3d.py` `track3d_loop` (3D).
- Constants: `PREV_NONE=-1`, `NEXT_NONE=-2`, `MAXCAND` (in `epi.py`).

---

## File structure

- Create: `src/openptv2/algorithms/assignment.py` — the auction solver (`auction_assign`) + sparse cost-matrix builder from candidate lists. Standalone, no tracker deps.
- Create: `tests/unit/test_assignment.py` — solver correctness + the greedy-vs-optimal contention case.
- Modify: `src/openptv2/algorithms/track_kernels_corr.py` — after candidate scoring, call `auction_assign` over the collected `(source, target, cost)` triples instead of the greedy reduction; write resulting links.
- Modify: `src/openptv2/algorithms/track3d.py` — same swap for the 3D loop (shared helper).
- Modify: `setup.py` — add `assignment.py` to the Cython build list (annotate=True).

---

## Task 1: Pure-Python auction solver (correctness first)

**Files:** Create `src/openptv2/algorithms/assignment.py`; Test `tests/unit/test_assignment.py`

- [ ] **Step 1: Write the failing test — optimal beats greedy on contention**

```python
import numpy as np
from openptv2.algorithms.assignment import auction_assign

def test_auction_beats_greedy_on_contention():
    # 2 sources, 2 targets. Source 0 slightly prefers t0; source 1 STRONGLY
    # prefers t0. Greedy by source-order gives t0 to source 0 -> source 1 forced
    # to costly t1. Optimal gives t0 to source 1 (big saving), t1 to source 0.
    # cost[s, t]:
    cost = np.array([[1.0, 2.0],
                     [0.1, 9.0]], dtype=np.float64)
    # sparse triples (src, tgt, cost) — all present here
    src = np.array([0, 0, 1, 1], dtype=np.int32)
    tgt = np.array([0, 1, 0, 1], dtype=np.int32)
    c   = np.array([1.0, 2.0, 0.1, 9.0], dtype=np.float64)
    assign = auction_assign(2, 2, src, tgt, c)   # returns tgt per src, -1 if none
    assert assign[1] == 0            # source 1 wins the contested cheap target
    assert assign[0] == 1
    total = cost[0, assign[0]] + cost[1, assign[1]]
    assert total == 2.1              # optimal; greedy would give 1.0+9.0=10.0
```

- [ ] **Step 2: Run it — fails (module missing).** `uv run pytest tests/unit/test_assignment.py -v` → ImportError.

- [ ] **Step 3: Implement the sparse auction solver**

```python
"""Sparse bipartite minimum-cost assignment via Bertsekas' auction algorithm.

Costs are MINIMIZED (tracker `rr`: lower = better). Auction natively maximizes
value, so we bid on value = -cost. Unassigned sources/targets are allowed:
a source stays unassigned (-1) if all its candidate bids are pushed above the
`max_cost` reservation (i.e. every candidate got too expensive)."""
import numpy as np
import cython


@cython.ccall
def auction_assign(n_src: int, n_tgt: int, src, tgt, cost, eps_frac: float = 0.01):
    src = np.ascontiguousarray(src, dtype=np.int32)
    tgt = np.ascontiguousarray(tgt, dtype=np.int32)
    value = -np.ascontiguousarray(cost, dtype=np.float64)   # maximize value

    # Adjacency: per-source candidate slices (CSR-like) for cache-friendly bids.
    order = np.argsort(src, kind="stable")
    src_s, tgt_s, val_s = src[order], tgt[order], value[order]
    starts = np.searchsorted(src_s, np.arange(n_src + 1))

    prices = np.zeros(n_tgt, dtype=np.float64)
    owner = np.full(n_tgt, -1, dtype=np.int32)     # target -> source
    assign = np.full(n_src, -1, dtype=np.int32)    # source -> target
    if len(val_s):
        span = float(val_s.max() - val_s.min()) or 1.0
    else:
        span = 1.0
    eps = eps_frac * span

    unassigned = [s for s in range(n_src) if starts[s + 1] > starts[s]]
    while unassigned:
        s = unassigned.pop()
        a, b = starts[s], starts[s + 1]
        # best and second-best net value (value - price) among candidates
        best_j = -1; best_v = -1e18; second_v = -1e18
        for k in range(a, b):
            j = tgt_s[k]
            net = val_s[k] - prices[j]
            if net > best_v:
                second_v = best_v; best_v = net; best_j = j
            elif net > second_v:
                second_v = net
        if best_j == -1:
            continue
        if second_v <= -1e17:
            second_v = best_v            # single candidate
        bid = (best_v - second_v) + eps  # raise price by the winning margin
        prices[best_j] += bid
        prev = owner[best_j]
        if prev != -1:
            assign[prev] = -1
            unassigned.append(prev)
        owner[best_j] = s
        assign[s] = best_j
    return assign
```

- [ ] **Step 4: Run — passes.** `uv run pytest tests/unit/test_assignment.py -v`.

- [ ] **Step 5: Add solver-property tests** — (a) permutation identity cost → identity assignment; (b) each target assigned ≤ once; (c) empty candidates → all -1. Run, then commit.

```bash
git add src/openptv2/algorithms/assignment.py tests/unit/test_assignment.py
git commit -m "feat(tracking): sparse auction assignment solver"
```

## Task 2: Cost-reservation / thresholding

**Files:** Modify `assignment.py`; Test `tests/unit/test_assignment.py`

- [ ] **Step 1: Failing test** — a candidate whose cost exceeds `max_cost` must never be assigned (source stays -1). Add `max_cost` param; internally drop triples with `cost > max_cost` before the auction.
- [ ] **Step 2–4:** implement the filter, run, commit. This preserves the tracker's accept/reject gate (`rr` acceptance threshold) so the assignment never manufactures links the greedy path would have rejected.

## Task 3: Wire into the 3D loop (smaller surface than 2D)

**Files:** Modify `src/openptv2/algorithms/track3d.py`; Test `tests/unit/test_track3d.py`

- [ ] **Step 1: Characterization test** — run `track3d_loop` on `test_data/track` (synthetic, deterministic) and record current link count as the baseline (`npart==nlinks==2` per step must be preserved — optimal assignment must not regress the trivial case).
- [ ] **Step 2:** In the per-frame-pair section, instead of the greedy reduction, collect `(source_idx, target_idx, rr)` triples for all in-threshold candidates, call `auction_assign(n_src, n_tgt, ...)`, and set `path_next[s]=assign[s]` / `path_prev[assign[s]]=s`.
- [ ] **Step 3:** Run the characterization test — must still pass (synthetic case has no contention, so assignment == greedy).
- [ ] **Step 4:** Measure on `test_cavity` (fresh sequence): link count and mean track length before/after. Expect links ≥ greedy and fewer swaps. Commit.

## Task 4: Wire into the 2D loop

**Files:** Modify `src/openptv2/algorithms/track_kernels_corr.py`

- [ ] Mirror Task 3 for `trackcorr_c_loop`. The candidate list + `rr` are already in `path_decis`/`path_linkdecis`; build triples from them and call the solver. Keep the existing conflict-resolution code path behind a flag for one release so results are comparable.
- [ ] Measure vs greedy on `test_cavity` (2D). Commit.

## Task 5: Cython port of the solver (performance)

**Files:** Modify `assignment.py` (add `nogil` typed kernel), `setup.py`

- [ ] Add typed memoryview signatures + `@cython.boundscheck(False)`; keep the pure-Python path as reference. Add `assignment` to `setup.py` build list.
- [ ] Rebuild (`uv run python setup.py build_ext --inplace`), check the annotation HTML for yellow in the bid loop, re-run all assignment + tracking tests. Commit.

---

## Testing & verification

- **Unit:** solver correctness (contention, permutation, thresholding, empty).
- **Parity:** synthetic `test_data/track` must keep `npart==nlinks==2`/step (no regression where there is no contention).
- **Quality metric on `test_cavity`:** report (a) total links, (b) mean/median track length after chaining links into trajectories, (c) reciprocity (should stay 100%), (d) count of assignments that differ from greedy (the contention cases the auction fixed). A net increase in mean track length with equal-or-higher link count is the success signal.
- **Perf:** time per frame-pair greedy vs auction; the Cython port must be within ~2x of greedy.

## Risks / notes

- **Ties** in `rr` → non-unique optima; `eps` scaling makes auction deterministic. Seed order must not matter — assert with a shuffled-input test.
- **Non-square** (n_src ≠ n_tgt) and unassigned particles are first-class (`-1`); do NOT pad with huge-cost dummies (blows up sparsity).
- **Threshold coupling:** the `max_cost` reservation must equal the tracker's existing `rr` acceptance bound, or link counts shift for the wrong reason.

---

## Future "secret sauce": Lie-group hooks (from 2026-07-11-lie-group-plans)

The auction step is cost-driven, which makes it the natural insertion point for a
better *predictor* — the cost `rr` depends on the predicted position via `acc`
and `angle`. Two items from the Lie-group notes plug in with **no change to the
assignment code**, only to how `rr` is computed:

- **Vorticity predictor (𝖘𝖔(3), Plan 2).** Replace the constant-velocity
  `predict()` (`track.py:350`) with a local-vorticity exponential-map step
  `U(t+Δt) = exp([ω]_× Δt)·U(t) + A·Δt`. In rotating flow (the cavity) this
  sharpens the predicted position, lowering `rr` for the *true* candidate and
  raising it for wrong ones — the auction then optimally exploits the cleaner
  costs. This is the single highest-value predictor upgrade and is independent
  of Task 1–5.
- **Plücker fast epipolar (correspondence side).** The reciprocal-product ray
  distance `|v1·m2 + v2·m1| / ‖v1×v2‖` gives an O(1) coplanarity test for
  candidate generation in `epi.py`, feeding cleaner candidates into the same
  assignment. Correspondence-phase change, complementary to this plan.

Sequencing suggestion: land the auction assignment (Tasks 1–5) first (pure
algorithmic win, measurable), then the 𝖘𝖔(3) predictor (physics win that the
assignment amplifies), then Plücker on the correspondence side.
