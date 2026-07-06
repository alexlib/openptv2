# Stereo Matching Algorithmic Upgrade Plan

**Status:** Strategic Planning Phase
**Target Modules:** `correspondences.py`, `epi.py`

This document details the algorithmic shifts required to break past the O(n⁴) scaling wall in the multi-camera correspondence matching engine. It provides strict, step-by-step instructions for implementation. 

**Core Philosophy:** 
1. The current Cython implementation is fast but algorithmically naive (nested loops).
2. We will implement these algorithms one by one in isolated feature branches.
3. **Ground Truth Over Legacy Parity:** We cannot assume the current nested-loop algorithm is perfectly optimal. Therefore, we will design explicit **Synthetic Ground Truth Unit Tests** for every new component. It is not enough that the existing `correspondences.py` tests pass (they might pass for the wrong reasons or enforce suboptimal legacy behavior). We must prove via synthetic scenarios that the new algorithm captures the correct physical reality.

---

## 0. Prerequisite: Synthetic Ground Truth Testing Framework

**Goal:** Establish a baseline where we know the exact, mathematically correct answer before changing any algorithms. Since the legacy code might have hidden flaws, replicating its exact output is not a sufficient test of correctness.

**Implementation Steps:**
1. **Generate Synthetic Cameras & Particles:** Create a dedicated test suite (e.g., `tests/unit/test_stereo_synthetic.py`) that initializes 4 virtual cameras looking at a mathematically defined 3D volume.
2. **Project Ground Truth:** Generate 100-500 known 3D points. Project these into the 4 virtual cameras using strict pinhole math (with known `Calibration` objects).
3. **Inject Noise:** Add controlled sub-pixel jitter to the projected 2D coordinates to simulate real-world sensor noise.
4. **Inject Ambiguity (Ghosts):** Add "fake" 2D particles in one or two cameras that lie on the epipolar lines of real particles, intentionally creating ambiguous cliques.
5. **The Ultimate Test:** The stereo matching algorithm must ingest the 2D coordinates and return the exact original 100-500 3D points. The metric of success is **Recall** and **Precision** against the Ground Truth, not just "matching what the legacy code did."

Every feature branch below MUST first prove it passes this Synthetic Ground Truth suite before it is allowed to merge.

---

## 1. Feature Branch 1: Early Pruning (The Low-Hanging Fruit)

**Goal:** Modify the nested `for` loops in `four_camera_matching` and `three_camera_matching` to calculate correlation scores incrementally, aborting branches that are doomed to fail before exploring deeper.

**Current State (`four_camera_matching`):**
The loops go from `i` → `j` → `k` → `l` → `m` → `n` → `o`, generating a complete quadruplet. Only at the very bottom is the 6-way correlation computed and checked against `accept_corr`.

**Implementation Steps:**
1. **Incremental Correlation:** In `correspondences.py`, inside `four_camera_matching`, define intermediate correlation sums and distance sums.
2. **First Cutoff (After Camera 3):**
   * After the `m` loop (which verifies C2↔C3 candidate matching), calculate the partial correlation for the 3-camera triplet: `corr_123 = (corr_12 + corr_13 + corr_23) / (dist_12 + dist_13 + dist_23)`.
   * **Crucial Logic Check:** Because tolerances and correlations are additive, we must define a safe proxy threshold for `corr_123`. A strict implementation: if `corr_123` is significantly worse than `accept_corr` (allowing for some buffer because the 4th camera might have a perfect match), we `continue` and skip the `l`, `n`, and `o` loops entirely.
3. **Unit Test Validation:** Run `uv run pytest tests/unit/test_correspondences.py -v`. The output list of correspondences and the `match_counts` array MUST be perfectly identical to the baseline.
4. **Repeat for `three_camera_matching`.**

---

## 2. Feature Branch 2: Spatial Grid Hashing for Candidate Search

**Goal:** Replace the 1D X-sorted linear scan in `find_candidate` with an O(1) 2D Spatial Hash lookup.

**Current State (`find_candidate` in `epi.py`):**
We do a binary search on the `x` coordinate to find a starting index, then linearly iterate `j` forward, calculating the perpendicular distance `d` to the epipolar line `y = m*x + b`.

**Implementation Steps:**
1. **Define the Grid:** In `tracking_frame_buf.py`, when coordinates are flattened and corrected, map each particle to a 2D grid index. 
   * `cell_size = vpar.eps0 * 2` (or a multiple thereof).
   * Arrays needed: `grid_head[num_cells_x, num_cells_y]` (points to first particle index), `grid_next[num_particles]` (points to next particle index in that cell).
2. **Epipolar Line Rasterization:** In `epi.py:find_candidate`, calculate which grid cells the epipolar line segment (from `(xa, ya)` to `(xb, yb)`) intersects.
   * Calculate bounding box cells: `min_cx, min_cy, max_cx, max_cy`.
3. **Grid Traversal:** 
   * Instead of looping `j` from `j0` to `num`, loop over the bounding box cells.
   * If a cell intersects the epipolar line (expanded by `vpar.eps0`), follow the linked list (`grid_head` → `grid_next`) to evaluate exactly those particles.
4. **Unit Test Validation:** Run `uv run pytest tests/unit/test_epi.py tests/unit/test_correspondences.py -v`. The returned candidate arrays (`cand_pnr`, `cand_tol`, `cand_corr`) and `count` MUST be perfectly identical.

---

## 3. Feature Branch 3: Bron-Kerbosch Maximal Clique Finding

**Goal:** Replace the hardcoded, nested O(n⁴) camera loops with an abstract, highly optimized graph search algorithm.

**Current State:**
`correspondences.py` has separate, duplicated logic for 4-camera, 3-camera, and 2-camera matching.

**Implementation Steps:**
1. **Adjacency Matrix Translation:** The flat arrays `p2_arr` and `n_arr` essentially represent a sparse Adjacency Matrix where edges are valid candidates.
2. **Bron-Kerbosch Kernel:** Create a new Cython kernel: `bron_kerbosch(R, P, X, max_depth)`.
   * `R`: Currently growing clique.
   * `P`: Prospective nodes (targets) that are connected to all nodes in `R`.
   * `X`: Excluded nodes (already processed).
   * Implement pivoting to minimize recursive branches.
3. **Integration:** 
   * Pass the epipolar adjacency data into the Bron-Kerbosch kernel.
   * The kernel will naturally yield maximal cliques of size 4, 3, or 2 without needing separate functions.
   * Once a clique is yielded, evaluate the aggregate correlation (as done previously) and accept/reject it.
4. **Unit Test Validation:** Run `uv run pytest tests/unit/test_correspondences.py -v`. The algorithm MUST yield the same cliques. Sorting/ordering of the final list might differ depending on graph traversal order, so the test assertions may need to be order-independent (e.g., comparing sets of sets).

---

## 4. Feature Branch 4: Global Optimization via Max-Flow (Optional / Future)

**Goal:** Replace the greedy `take_best_candidates` selection with a globally optimal assignment.

**Current State:**
When particles share candidates, the engine sorts all valid cliques by correlation and assigns them greedily. A slightly lower-correlation clique might be chosen, ruining two other excellent cliques.

**Implementation Steps:**
1. **Bipartite Graph Definition:** Map the resulting cliques and their component particles to a flow network or use the Hungarian algorithm / Maximum Weight Matching to resolve conflicts.
2. **Integration:** Replace the `take_best_candidates` function in `correspondences.py`.
3. **Unit Test Validation:** Note: This *will* change the output of the matching algorithm. This branch requires a synthetic tracking test with known ground-truth ambiguities to mathematically prove that the Max-Flow assignment yields fewer ghost particles than the Greedy assignment.
