# Refactoring Plan: Python `algorithms/` Bug Hunting via C-to-Python Comparison

## Problem Statement

The Python `algorithms/` package is a direct translation of the C library in `lib/src/`. Integration tests that exercise the full tracking pipeline fail (wrong `npart`/`nlinks` counts in cavity, burgers, track tests). The bugs are hard to locate because tracking sits at the end of a long pipeline:

```
parameters → calibration → image_processing → segmentation → trafo → imgcoord →
multimed → ray_tracing → epi → correspondences → tracking_frame_buf →
tracking_run → track → track3d
```

A bug in any earlier stage silently corrupts downstream results.

**Current test status: 37 failed, 64 passed** out of 101 tests in `algorithms/tests/`.

## Strategy

Compare every Python module and test against its C counterpart using the C code as ground truth. Work **bottom-up** through the dependency chain — fix foundation modules first so that downstream modules can be tested reliably.

The C tests live in `lib/tests/check_*.c` and use the Check framework. The C test data lives in `lib/tests/testing_fodder/`. The Python tests use `testing_fodder/` relative paths but this directory doesn't exist in the Python test context — many tests fail with `FileNotFoundError` before logic is even tested. Fixing test data paths is a prerequisite.

---

## Phase 0: Test Infrastructure

### 0.1 Create shared test data fixture
The C tests run from `lib/tests/` where `testing_fodder/` exists. The Python tests reference `testing_fodder/` but run from the repo root where it doesn't exist.

**Action:**
- Create a conftest.py fixture or symlink that makes `testing_fodder/` → `lib/tests/testing_fodder/` available to Python tests
- Alternatively, use `test_data/` (which already exists) and update Python test paths to point there
- Audit which test data files the C tests use vs. what's in `test_data/`

**Verify:** `pytest algorithms/tests/test_parameters.py -v` should find all parameter files.

### 0.2 Fix import of `MAX_CANDS` constant
`test_track.py:228` references `MAX_CANDS` which is not imported. The C `track.h` defines `MAX_CANDS 4`.

**Action:** Ensure `MAX_CANDS = 4` is defined in `algorithms/constants.py` (or `algorithms/track.py`) and imported in the test.

**Verify:** `test_sort_candidates_by_freq` no longer raises `NameError`.

---

## Phase 1: Foundation Modules (no inter-module dependencies)

### 1.1 `vec_utils` — C: `check_vec_utils.c` (206 lines) vs Python: `test_vec_utils.py` (88 lines)

**Status:** All 7 Python tests pass. Good coverage.

**Action:** Spot-check for missing C test cases. The C test has `test_vec_init`, `test_vec_cmp`, `test_vec_approx_cmp`, `test_dot`, `test_cross`, `test_norm`, `test_is_empty`. Compare function signatures and behavior 1:1.

**Verify:** No action needed unless discrepancies found.

### 1.2 `lsqadj` — C: `check_lsqadj.c` (153 lines) vs Python: `test_lsqadj.py` (43 lines)

**Status:** All 2 Python tests pass.

**Action:** The C test is 153 lines vs Python's 43 — likely missing test cases. Read `check_lsqadj.c` and port missing tests (e.g., `test_matmul`, `test_ata`).

**Verify:** All ported tests pass.

### 1.3 `calibration` — C: `check_calibration.c` (151 lines) vs Python: `test_calibration.py` (102 lines)

**Status:** All 5 Python tests pass.

**Action:** Compare `read_calibration()` parsing, especially the rotation matrix computation. Verify `Calibration.from_file()` produces identical `ext_par`, `int_par`, `glass_par`, `added_par` values to C's `read_calibration()`.

**Verify:** Pass.

### 1.4 `parameters` — C: `check_parameters.c` (194 lines) vs Python: `test_parameters.py` (141 lines)

**Status:** All 5 tests FAIL — `FileNotFoundError` for `testing_fodder/parameters/*.par`.

**Action:**
1. Fix test data paths (Phase 0.1)
2. Compare `read_control_par`, `read_sequence_par`, `read_track_par`, `read_volume_par`, `read_target_par` against C implementations field by field
3. **Critical:** Verify `track_par` struct mapping. C defines it as `{dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax, dangle, dacc, add, dsumg, dn, dnx, dny}` — 13 fields. Confirm Python's `TrackParTuple` has identical field order and semantics.
4. **Critical:** Verify `control_par.mm` (multimedia parameters) is populated correctly, especially `nlay`, `n1`, `n2[]`, `d[]`, `n3`.

**C test values to match:**
- `track_par`: `{0.4, 120, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 0., 0., 0., 0., 1.}`
- `volume_par`: `{X_lay={-250., 250.}, Zmin_lay={-100., -100.}, Zmax_lay={100., 100.}, cnx=0.01, cny=0.3, csumg=0.3, corrmin=0.01, eps0=1, cn=33}`
- `control_par`: `num_cams=4, imx=1280, imy=1024, pix_x=0.017, pix_y=0.017, ...`

**Verify:** All parameter tests pass with correct values.

---

## Phase 2: Coordinate Transforms and Image Processing

### 2.1 `trafo` — C: `check_trafo.c` (375 lines) vs Python: `test_trafo.py` (181 lines)

**Status:** All 9 Python tests pass. Good.

**Action:** Verify exact numerical match for:
- `metric_to_pixel` / `pixel_to_metric` round-trip: C expects `(512.0, 504.0)` for `(xc=0, yc=0, imx=1024, imy=1008, pix_x=0.01, pix_y=0.01)`
- `distort_brown_affin`: C expects `(0.158529, 0.540302)` for `(x=1, y=1, ap={0,0,0,0,0,1,1})`
- `dist_to_flat` / `flat_to_dist` round-trip with tolerance 1e-3

**Verify:** All 9 pass with C-identical values.

### 2.2 `image_processing` — C: `check_image_proc.c` (406 lines) vs Python: `test_image_processing.py` (186 lines)

**Status:** 2 tests FAIL (`test_box_blur`, `test_highpass`) with shape broadcast errors.

**Action:**
1. Debug `fast_box_blur` — the Python implementation likely has an array shape mismatch (shape `(4,)` into shape `(5,)`)
2. Debug `prepare_image` (highpass) — same broadcast issue
3. Compare C `filter_3`, `lowpass_3`, `fast_box_blur`, `highpass_3`, `subtract_img`, `split` against Python implementations
4. **C test values:** `highpass` of 3x3 white block in 5x5: inner pixels should be `{142, 85, 142, 85, 0, 85, 142, 85, 142}`

**Verify:** All 8 Python tests pass.

---

## Phase 3: Core Geometry (multimed, ray_tracing, imgcoord)

These modules are **critical** — they compute where 3D points project onto cameras and vice versa. Any bug here corrupts correspondences and tracking.

### 3.1 `multimed` — C: `check_multimed.c` (448 lines) vs Python: `test_multimed.py` (217 lines)

**Status:** All 6 tests FAIL — mix of `FileNotFoundError` and `ValueError` (wrong return count from `trans_Cam_Point`).

**Action:**
1. Fix test data paths
2. **Critical bug:** `trans_Cam_Point` returns 3 values but test expects 4. C signature: `trans_Cam_Point(Exterior, mm_np, Glass, pos, &Ex_t, pos_t, cross_p, cross_c)` — it outputs `Ex_t`, `pos_t`, `cross_p`, `cross_c`. Python likely returns `(Ex_t, pos_t, cross_p)` missing `cross_c`.
3. Compare `multimed_nlay` — C expects `Xq=0.74811917, Yq=0.75977975` for `pos={1.23, 1.23, 1.23}` with cam1 calibration
4. Compare `init_mmlut` — C expects `origin = (0, 0, -250.00001105)`, `nr=130`, `nz=177`, `rw=2`
5. Compare `get_mmf_from_mmlut` — C expects `mmf = 1.00382` for `pos={1,1,1}`
6. Compare `volumedimension` — C expects `xmax=73.02053752, xmin=-46.80667189, ymax=51.04924925, ymin=-62.91848990`
7. Compare `back_trans_Point` round-trip — must recover original position to within 1e-6

**Verify:** All 6 tests pass with C-matching values.

### 3.2 `ray_tracing` — C: `check_ray_tracing.c` (93 lines) vs Python: `test_ray_tracing.py` (41 lines)

**Status:** 1 test FAILS — `AttributeError: 'Calibration' object has no attribute 'set_pos'`

**Action:**
1. Fix test setup — use `Calibration` API correctly instead of calling nonexistent `set_pos`
2. Compare `ray_tracing(x=100, y=100, ...)` — C expects:
   - `X = (110.406944, 88.325788, 0.988076)`
   - `a = (0.387960, 0.310405, -0.867834)`
3. Verify the Calibration object is correctly initialized with `Exterior`, `Interior`, `Glass`, `ap_52`

**Verify:** Pass with C-identical output.

### 3.3 `imgcoord` — C: `check_imgcoord.c` (194 lines) vs Python: `test_imgcoord.py` (132 lines)

**Status:** All 5 Python tests pass.

**Action:** Verify exact numerical match for:
- `flat_image_coord(pos={10,5,-20}, cam_at_origin)` = `(10/6, 5/6)` (all-air case)
- `img_coord` with barrel distortion: `x = (10/6)*(1 - 0.01*r^2)` where `r = norm(10/6, 5/6, 0)`
- Decentered camera looking through origin: result should be `(0, 0)`
- Multilayer with glass: result should be `(0, 0)` when camera axis passes through point

**Verify:** Already passing — confirm values match C.

---

## Phase 4: Epipolar Geometry and Correspondences

### 4.1 `epi` — C: `check_epi.c` (340 lines) vs Python: `test_epi.py` (116 lines)

**Status:** 1 test FAILS (`test_epi_mm`) with `ValueError: Total internal reflection: arcsin argument out of bounds`.

**Action:**
1. **Critical bug:** The `epi_mm` function triggers total internal reflection. This means refractive indices or geometry are being passed in wrong order. C test uses `mm = {nlay=1, n1=1.0, n2={1.49}, d={5.0}, n3=1.33}`. Check whether Python swaps `n1`/`n3` or `n2`/`d`.
2. Compare `epi_mm_2D(x=1, y=10)` — C expects `out = (0.85858163, 8.58581626, 0.0)`
3. Compare `epi_mm(x=10, y=10)` — C expects `xmin=26.44927852, xmax=10.08218486, ymin=51.60078764, ymax=10.04378909`
4. Compare `find_candidate` — C expects: `cand[0].pnr=0`, `cand[0].tol<EPS`, `sum_corr=3301.0`, `count=5`, `cand[3].tol=0.636396`
5. Check the perpendicular camera case: `xmin=-100, xmax=0, ymin=100, ymax=0`

**Verify:** All 4 Python tests pass with C-matching values.

### 4.2 `correspondences` — C: `check_correspondences.c` (528 lines) vs Python (no dedicated test file exists yet)

**Status:** No `test_correspondences.py` exists in `algorithms/tests/`! This is a major gap.

**Action:**
1. **Create `test_correspondences.py`** by porting tests from `check_correspondences.c`
2. Port `test_qs_target_y` — quicksort targets by y-coordinate
3. Port `test_quicksort_coord2d_x` — quicksort coord_2d by x
4. Port `test_quicksort_con` — quicksort n_tupel by correlation
5. Port `test_pairwise_matching` — generate a 4x4 grid test set, verify each target has correct matches as candidates
6. Port `test_four_camera_matching` — expects `matched == 16`
7. Port `test_three_camera_matching` — darken cam2, expects `matched == 16` triplets
8. Port `test_two_camera_matching` — expects `matched == 16` pairs
9. Port `test_correspondences` — full pipeline, expects `match_counts = [16, 0, 0, 16]`
10. **Critical:** The C test generates targets by back-projecting a 4x4 grid through each camera's calibration, then runs `correct_frame()` (pixel→metric→flat→x-sort) before matching. Verify the Python equivalent does the same pipeline.

**Verify:** All 9 tests pass.

---

## Phase 5: Frame Buffer and Tracking Infrastructure

### 5.1 `tracking_frame_buf` — C: `check_fb.c` (319 lines) vs Python: `test_tracking_frame_buf.py` (210 lines)

**Status:** 2 tests FAIL — `compare_path_info` returns False; shape broadcast error on corres `p[4]` into `p[2]`.

**Action:**
1. **Bug:** `compare_path_info` fails — likely the `Pathinfo` object fields don't match C's `P` struct. C `P` has: `x[3]`, `prev`, `next`, `prio`, `decis[POSI]`, `finaldecis`, `linkdecis[POSI]`, `inlist`. Verify Python matches.
2. **Bug:** `corres.p` is shape `(2,)` but C `corres.p[4]` always has 4 camera slots. Even for 2-camera setups, the array must be length 4.
3. Compare `read_targets` — C expects 2 targets from `sample_0042_targets` with specific `pnr`, `x`, `y`, `n`, `nx`, `ny`, `sumg`, `tnr` values
4. Compare `read_path_frame` — C expects 80 particles from `rt_is.818`, particle 3 at `x={45.219, -20.269, 25.946}`, `prev=-1`, `next=-2`, `prio=4`
5. Compare `write_targets` / `write_path_frame` round-trip
6. Verify `frame_init` allocates correct sizes for `path_info`, `correspond`, `targets[num_cams]`

**C target values:**
- `t1 = {pnr=0, x=1127.0, y=796.0, n=13320, nx=111, ny=120, sumg=828903, tnr=1}`
- `t2 = {pnr=1, x=796.0, y=809.0, n=13108, nx=113, ny=116, sumg=658928, tnr=0}`

**Verify:** All 7 tests pass.

### 5.2 `tracking_run` — C: `tracking_run.c/h` vs Python: `tracking_run.py`

**Status:** `tr_new()` fails with `AttributeError: 'str' object has no attribute 'num_cams'` — the Python `tr_new` accepts filename strings instead of parsed parameter objects but tries to access `.num_cams` on the string.

**Action:**
1. Fix `tr_new()` signature — C's `tr_new` takes parsed parameter objects (`sequence_par*`, `track_par*`, etc.). C's `tr_new_legacy` takes filenames. The Python code seems to mix these up.
2. Verify the `tracking_run` data structure: `fb`, `seq_par`, `tpar`, `vpar`, `cpar`, `cal`, `flatten_tol`, `ymin`, `ymax`, `lmax`, `npart`, `nlinks`
3. Verify `fb_init` creates the 4-frame ring buffer correctly with proper virtual dispatch

**Verify:** `tr_new` creates a valid tracking_run object.

---

## Phase 6: Segmentation

### 6.1 `segmentation` — C: `check_segmentation.c` (146 lines) vs Python: `test_segmentation.py` (86 lines)

**Status:** All 3 Python tests pass.

**Action:** Verify exact match for:
- `targ_rec` on 3x3 white block in 5x5: expects `ntargets=1, pix[0].n=9, pix[0].tnr=CORRES_NONE`
- `targ_rec` on two isolated pixels with threshold 250: expects `ntargets=2`
- `targ_rec` with raised threshold 252: expects `ntargets=1`
- `peak_fit` — same inputs as `targ_rec` but a different code path

**Verify:** Already passing.

---

## Phase 7: Sortgrid and Orientation

### 7.1 `sortgrid` — C: `check_sortgrid.c` (153 lines) vs Python: `test_sortgrid.py` (65 lines)

**Status:** 1 test FAILS — `len(Targets(0 targets))` is 0 (target reading fails).

**Action:**
1. Fix test data paths
2. Compare `nearest_neighbour_pix` — C expects `pnr=-999` for zero/negative epsilon, `pnr=0` for exact match
3. Compare `read_sortgrid_par` — C expects `eps=25`
4. Compare `sortgrid` — with `eps=25`: `sorted_pix[0].pnr == -999`, with `eps=120`: `sorted_pix[1].pnr == 1, sorted_pix[1].x == 796`

**Verify:** All 4 tests pass.

### 7.2 `orientation` — C: `check_orientation.c` (397 lines) vs Python: `test_orientation.py` (255 lines)

**Status:** All 7 tests FAIL — FileNotFoundError and `vec_cmp` logic error.

**Action:**
1. Fix test data paths
2. **Bug in `skew_midpoint`:** C returns distance=1.0 and midpoint=(0,0,0.5) for perpendicular unit rays. Python `vec_cmp` comparison fails — likely the midpoint computation is wrong.
3. Compare `raw_orient` — should recover original calibration to within 1e-3 total error
4. Compare `orient` — should recover original calibration to within 1e-6
5. Compare `point_position` — skew distance should be < 1e-10 for exact case, with jigged cameras error should be < 0.05
6. Compare `weighted_dumbbell_precision` — should be < 1e-10 for exact case

**Verify:** All 7 tests pass.

---

## Phase 8: Track Core Functions

### 8.1 Unit-level track functions — C: first half of `check_track.c`

**Status:** `test_predict` and `test_search_volume_center_moving` pass. `test_pos3d_in_bounds` and `test_angle_acc` pass. `test_candsearch_in_pix` FAILS (returns 0 instead of 2). `test_candsearch_in_pix_rest` FAILS (returns 0 instead of 1).

**Action:**
1. **Critical bug in `candsearch_in_pix`:** Returns 0 candidates instead of 2. Compare C implementation line-by-line. C checks `cent_x - dl < pix.x < cent_x + dr` AND `cent_y - du < pix.y < cent_y + dd`. The search window is `[0.1, 0.3] x [0.1, 0.3]` — targets at `(0.2, 0.2)` and `(0.2, 0.3)` should match.
2. **Critical bug in `candsearch_in_pix_rest`:** Same pattern — window logic likely wrong. Additionally, this function should only match targets with `tnr == -1` (unlinked).
3. Compare `searchquader` — C projects a 3D point onto each camera and computes search rectangles. Depends on correct `img_coord` + `metric_to_pixel` + `multimed`.
4. Compare `sort_candidates_by_freq` — C counts frequency of each `ftnr` across cameras and sorts descending.

**Verify:** All unit track tests pass with C-matching values.

### 8.2 `track3d` — C: `check_track3d.c` (454 lines) vs Python: `test_track3d.py` (166 lines)

**Status:** All 3 tests FAIL — FileNotFoundError and `tr_new` signature issues.

**Action:**
1. Fix `tr_new` (Phase 5.2)
2. Port `find_candidates_in_3d` tests from C: empty frame (0), single match (1), outside box (0), multiple (2), max limit (3), boundary excluded (0)
3. Verify `track3d_loop` produces same results as `trackcorr_c_loop` for the simple 2-camera track case

**Verify:** All tests pass.

---

## Phase 9: Integration Tests (End-to-End Tracking)

These tests exercise the entire pipeline. They should pass only after all upstream modules are fixed.

### 9.1 `test_trackcorr_no_add` — 2 cameras, 1 particle, no additions

**C expects:** `npart/range = 0.8`, `nlinks/range = 0.8`

**Action:** Run after all upstream fixes. If it still fails, add print statements at each pipeline stage to compare intermediate values with C.

### 9.2 `test_trackcorr_with_add` — 2 cameras, 1 particle, with additions

**C expects:** `npart/range = 1.0`, `nlinks/range = 0.7`

### 9.3 `test_cavity` — 4 cameras, ~200 particles per frame

**C expects:**
- Without add: `npart = 2082`, `nlinks = 452`
- With add: `npart = 2086`, `nlinks = 461`

### 9.4 `test_burgers` — 4 cameras, Burgers vortex flow

**C expects:**
- Without add: `npart = 19`, `nlinks = 17`
- With add: `npart = 20`, `nlinks = 20`

### 9.5 `test_trackback` — backward tracking

**C test has this assertion commented out:** `nlinks ≈ 1.043062`. Just verify it runs without error.

---

## Execution Order (Priority-Ranked)

```
Phase 0: Test infrastructure (paths, constants)           → unblocks 22 tests
Phase 1: Foundation (vec_utils, lsqadj, calibration, parameters) → unblocks 5 tests
Phase 2: Transforms (trafo, image_processing)              → unblocks 2 tests
Phase 3: Core geometry (multimed, ray_tracing, imgcoord)   → unblocks 7 tests
Phase 4: Epipolar + correspondences (epi, correspondences) → unblocks 1 test + new tests
Phase 5: Frame buffer + tracking_run                       → unblocks 5 tests
Phase 6: Segmentation (verify only)                        → 0 tests
Phase 7: Sortgrid + orientation                            → unblocks 8 tests
Phase 8: Track core functions                              → unblocks 5 tests
Phase 9: Integration tests                                 → unblocks 5 tests
```

## How to Verify Each Fix

For each module:
1. Read the C test (`lib/tests/check_<module>.c`) and note exact expected values
2. Read the C source (`lib/src/<module>.c`) for algorithm logic
3. Read the Python source (`algorithms/<module>.py`) and find divergences
4. Update or create the Python test (`algorithms/tests/test_<module>.py`) to match C test exactly
5. Run `uv run pytest algorithms/tests/test_<module>.py -v` and verify all pass
6. After each phase, run `uv run pytest algorithms/tests/ -v` to check for regressions

## Key Patterns to Watch For

1. **Field order in structs/tuples** — C structs have fixed field order. Python NamedTuples or dataclasses may have different field order, causing silent data corruption when fields are passed positionally.
2. **Array size conventions** — C always uses 4-camera arrays (`p[4]`, `whichcam[4]`). Python may dynamically size these to `num_cams`, breaking assumptions.
3. **Sentinel values** — C uses `-999` for unused targets (`PT_UNUSED`), `-1` for `CORRES_NONE`, `-1`/`-2` for `PREV_NONE`/`NEXT_NONE`. Verify Python uses identical sentinels.
4. **Integer vs float division** — C integer division truncates. Python 3 `/` always returns float.
5. **In-place vs return-value** — C functions often modify output via pointer arguments. Python translations may return values instead. Verify the caller handles this correctly.
6. **Multimedia parameter passing** — `mm_np` has `nlay`, `n1`, `n2[3]`, `d[3]`, `n3`. Wrong field mapping here corrupts all geometry.
7. **Rotation matrix** — C's `rotation_matrix()` computes `dm[3][3]` from `(omega, phi, kappa)`. Verify Python produces identical matrix for same angles.
8. **Coordinate system conventions** — pixel (0,0) is top-left in C. Verify Python agrees.
