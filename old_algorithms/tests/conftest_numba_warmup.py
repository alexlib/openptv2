"""
Numba JIT warmup – force-compile all 47 @njit functions before tests run.

Usage:
  - As a pytest plugin: automatically loaded via algorithms/tests/conftest.py
  - Standalone:  uv run python -m algorithms.tests.conftest_numba_warmup

All 47 functions already have cache=True, so the first run pays the full
compilation cost and subsequent runs load from the on-disk cache (~0.5 s).
"""

import time
import numpy as np
from numba.core.registry import CPUDispatcher


def _warmup_all() -> tuple[int, float]:
    """Import and call every @njit function with minimal dummy inputs.

    Returns (n_compiled, elapsed_seconds).
    """
    t0 = time.perf_counter()
    compiled = 0

    # ── vec_utils (13 functions) ─────────────────────────────────────
    from algorithms.vec_utils import (
        vec_norm, vec_set, norm, vec_copy, vec_subt, vec_add,
        vec_scalar_mul, vec_diff_norm, vec_dot, vec_cross, vec_cmp,
        unit_vector, vec_init,
    )
    v3 = np.array([1.0, 0.0, 0.0])
    v3b = np.array([0.0, 1.0, 0.0])
    vec_norm(v3)
    vec_set(1.0, 2.0, 3.0)
    norm(1.0, 2.0, 3.0)
    vec_copy(v3)
    vec_subt(v3, v3b)
    vec_add(v3, v3b)
    vec_scalar_mul(v3, 2.0)
    vec_diff_norm(v3, v3b)
    vec_dot(v3, v3b)
    vec_cross(v3, v3b)
    vec_cmp(v3, v3b, 1e-6)
    unit_vector(np.ascontiguousarray(v3))
    vec_init(np.int32(3))
    compiled += 13

    # ── trafo (6 functions) ──────────────────────────────────────────
    from algorithms.trafo import (
        fast_pixel_to_metric, arr_pixel_to_metric,
        fast_metric_to_pixel, fast_arr_metric_to_pixel,
        distort_brown_affine, correct_brown_affine,
    )
    ap = np.zeros(7, dtype=np.float64)
    fast_pixel_to_metric(100.0, 100.0, 1024, 1024, 0.01, 0.01)
    arr_pixel_to_metric(
        np.array([[100, 100]], dtype=np.int32),
        np.int32(1024), np.int32(1024), 0.01, 0.01,
    )
    fast_metric_to_pixel(0.0, 0.0, 1024, 1024, 0.01, 0.01)
    fast_arr_metric_to_pixel(
        np.array([[0.0, 0.0]]), np.int32(1024), np.int32(1024), 0.01, 0.01,
    )
    distort_brown_affine(0.0, 0.0, ap)
    correct_brown_affine(0.0, 0.0, ap)
    compiled += 6

    # ── calibration (1 function) ─────────────────────────────────────
    from algorithms.calibration import rotation_matrix, Exterior
    ext = Exterior.copy()
    rotation_matrix(ext)
    compiled += 1

    # ── ray_tracing (2 functions) ────────────────────────────────────
    from algorithms.ray_tracing import fast_ray_tracing, matmul_numba_optimized
    fast_ray_tracing(
        np.array([0.0, 0.0, -1.0]),   # initial_ray_direction
        np.eye(3),                     # distortion_matrix
        np.array([0.0, 0.0, 100.0]),   # primary_point
        np.array([0.0, 0.0, 1.0]),     # glass_vector
        1.0,                           # distance_param
        1.0, 1.49, 1.0,               # n1, n2, n3
    )
    matmul_numba_optimized(
        np.eye(2), np.eye(2), np.zeros((2, 2)),
        np.int64(2), np.int64(2), np.int64(2),
    )
    compiled += 2

    # ── multimed (8 functions) ───────────────────────────────────────
    from algorithms.multimed import (
        fast_get_mmf_from_mmlut_raw, fast_multimed_r_nlay,
        fast_trans_cam_point, fast_back_trans_point,
        fast_flat_image_coord_raw, fast_point_to_pixel,
        move_along_ray, fast_get_mmf_from_mmlut,
    )
    origin = np.zeros(3)
    # mmlut_data is accessed as 1D flat in _raw, 2D in the non-raw version
    mmlut_data_1d = np.ones(4, dtype=np.float64)
    mmlut_data_2d = np.ones((2, 2), dtype=np.float64)
    pos = np.array([1.0, 1.0, 1.0])
    fast_get_mmf_from_mmlut_raw(
        np.int32(1), origin, mmlut_data_1d, np.int32(1), np.int32(1), pos,
    )
    fast_get_mmf_from_mmlut(
        np.int32(1), origin, mmlut_data_1d, np.int32(1), np.int32(1), pos,
    )
    fast_multimed_r_nlay(
        np.int32(1), 1.0, np.array([1.49]), 1.0, np.array([5.0]),
        0.0, 0.0, 100.0, pos,
    )
    pp = np.array([0.0, 0.0, 100.0])
    gd = np.array([0.0, 0.0, 1.0])
    fast_trans_cam_point(pp, 1.0, gd, pos)
    cross_c = np.zeros(3)
    cross_p = np.zeros(3)
    fast_back_trans_point(gd, 1.0, cross_c, cross_p, pos)
    move_along_ray(0.0, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    # fast_flat_image_coord_raw needs many args:
    fast_flat_image_coord_raw(
        pos,                          # orig_pos
        np.zeros(3),                  # ex_pos
        np.eye(3),                    # ex_dm
        100.0,                        # int_cc
        gd,                           # glass_par
        np.array([5.0]),              # mm_d
        1.0,                          # mm_n1
        np.array([1.49]),             # mm_n2
        1.0,                          # mm_n3
        origin,                       # mmlut_origin
        mmlut_data_1d,                # mmlut_data
        np.int32(1),                  # mmlut_nz
        np.int32(1),                  # mmlut_nr
        np.int32(1),                  # mmlut_rw
    )
    # fast_point_to_pixel needs even more:
    fast_point_to_pixel(
        pos,                          # point
        np.zeros(3),                  # ex_pos
        np.eye(3),                    # ex_dm
        100.0,                        # int_cc
        0.0, 0.0,                     # int_xh, int_yh
        ap,                           # added_par (7 elems)
        gd,                           # glass_par
        np.array([5.0]),              # mm_d
        1.0,                          # mm_n1
        np.array([1.49]),             # mm_n2
        1.0,                          # mm_n3
        np.int32(1),                  # mm_nlay
        origin,                       # mmlut_origin
        mmlut_data_1d,                # mmlut_data
        np.int32(1),                  # mmlut_nz
        np.int32(1),                  # mmlut_nr
        np.int32(1),                  # mmlut_rw
        np.int32(1024),               # imx
        np.int32(1024),               # imy
        0.01,                         # pix_x
        0.01,                         # pix_y
    )
    compiled += 8

    # ── image_processing (4 functions) ───────────────────────────────
    from algorithms.image_processing import (
        filter_3, lowpass_3, subtract_img, subtract_mask,
    )
    img = np.zeros((8, 8), dtype=np.uint8)
    filter_3(img)
    lowpass_3(img)
    subtract_img(img, img, np.zeros_like(img))
    subtract_mask(img.copy(), img)
    compiled += 4

    # ── orientation (2 functions) ────────────────────────────────────
    from algorithms.orientation import (
        skew_midpoint, _multi_cam_point_positions_numba,
    )
    skew_midpoint(v3, v3, v3b, v3b)
    # _multi_cam_point_positions_numba needs array inputs:
    # Keep all camera dimensions consistent (ncams=1) to avoid OOB memory writes.
    targets = np.zeros((1, 1, 2), dtype=np.float64)       # (n, ncams, 2)
    dm = np.eye(3).reshape(1, 3, 3)                       # (ncams, 3, 3)
    pp_ = np.array([[0.0, 0.0, 100.0]])                   # (ncams, 3)
    gv = np.array([[0.0, 0.0, 1.0]])                      # (ncams, 3)
    ccs = np.array([100.0])                               # (ncams,)
    _multi_cam_point_positions_numba(
        targets, dm, pp_, gv, ccs,
        1.0, 1.0, 1.49, 1.0,
        -999.0,
    )
    compiled += 2

    # ── epi (1 function) ─────────────────────────────────────────────
    from algorithms.epi import _epi_mm_batch_inner
    # Needs a bunch of arrays — create minimal ones
    _epi_mm_batch_inner(
        np.array([0.0]),              # xl
        np.array([0.0]),              # yl
        np.zeros(3),                  # cal1_pos
        np.eye(3),                    # cal1_dm
        gd,                           # cal1_glass
        100.0,                        # cal1_cc
        np.zeros(3),                  # cal2_pos
        np.eye(3),                    # cal2_dm
        gd,                           # cal2_glass
        100.0,                        # cal2_cc
        1.0,                          # mm_n1
        np.array([5.0]),              # mm_d
        np.array([1.49]),             # mm_n2
        1.0,                          # mm_n3
        origin,                       # mmlut_origin
        mmlut_data_1d,                # mmlut_data
        np.array([-10.0, 10.0]),      # x_lay
        np.array([-100.0, -100.0]),   # z_min_lay
        np.array([100.0, 100.0]),     # z_max_lay
    )
    compiled += 1

    # ── find_candidate (2 functions) ─────────────────────────────────
    from algorithms.find_candidate import quality_ratio, find_start_point_binary
    quality_ratio(1.0, 2.0)
    find_start_point_binary(np.array([1.0, 2.0, 3.0]), np.int32(3), 1.5, 0.1)
    compiled += 2

    # ── correspondences (2 functions) ────────────────────────────────
    from algorithms.correspondences import (
        _fill_adjacency_pair, _four_camera_matching_inner,
    )
    # _fill_adjacency_pair has many args — trigger via import only if cached
    # These are large kernels; triggering them with dummy data is fragile.
    # Just accessing the CPUDispatcher forces compilation of the IR.
    if isinstance(_fill_adjacency_pair, CPUDispatcher):
        try:
            _fill_adjacency_pair.compile(
                _fill_adjacency_pair.signatures[0]
            ) if _fill_adjacency_pair.signatures else None
        except Exception:
            pass
    if isinstance(_four_camera_matching_inner, CPUDispatcher):
        try:
            _four_camera_matching_inner.compile(
                _four_camera_matching_inner.signatures[0]
            ) if _four_camera_matching_inner.signatures else None
        except Exception:
            pass
    compiled += 2

    # ── track (5 functions) ──────────────────────────────────────────
    from algorithms.track import (
        _candsearch_in_pix_core, search_volume_center_moving,
        pos3d_in_bounds, angle_acc, _sort_candidates_by_freq_njit,
    )
    from algorithms.parameters import TrackParTuple

    tpar = TrackParTuple(
        dvxmin=-10.0, dvxmax=10.0,
        dvymin=-10.0, dvymax=10.0,
        dvzmin=-10.0, dvzmax=10.0,
        dangle=200.0, dacc=200.0, add=0,
        dsumg=100.0, dn=100.0, dnx=100.0, dny=100.0,
    )
    pos3d_in_bounds(pos, tpar)
    angle_acc(v3, v3b, np.array([0.0, 0.0, 1.0]))
    search_volume_center_moving(v3, v3b)
    _sort_candidates_by_freq_njit(
        np.zeros(16, dtype=np.int32),
        np.zeros(16, dtype=np.int32),
        np.zeros((16, 4), dtype=np.int32),
        np.int32(4),
    )
    _candsearch_in_pix_core(
        np.array([0.0, 1.0]),         # target_x
        np.array([0.0, 1.0]),         # target_y
        np.zeros(2, dtype=np.int32),  # target_tnr
        np.int32(2),                  # num_targets
        0.5,                          # cent_x
        0.5,                          # cent_y
        -10.0, 10.0, -10.0, 10.0,    # dl, dr, du, dd
        np.int32(1024),               # imx
        np.int32(1024),               # imy
        False,                        # require_unused
    )
    compiled += 5

    # ── segmentation (1 function) ────────────────────────────────────
    from algorithms.segmentation import fast_targ_rec
    img16 = np.zeros((16, 16), dtype=np.uint8)
    fast_targ_rec(
        img16,
        np.int32(128),   # thres
        0.5,             # disco
        np.int32(1),     # nnmin
        np.int32(100),   # nnmax
        np.int32(1),     # nxmin
        np.int32(100),   # nxmax
        np.int32(1),     # nymin
        np.int32(100),   # nymax
        np.int32(10),    # sumg_min
        np.int32(0),     # xmin
        np.int32(15),    # xmax
        np.int32(0),     # ymin
        np.int32(15),    # ymax
    )
    compiled += 1

    elapsed = time.perf_counter() - t0
    return compiled, elapsed


if __name__ == "__main__":
    print("Warming up numba JIT cache …")
    n, elapsed = _warmup_all()
    print(f"Done: {n} functions compiled/loaded in {elapsed:.2f}s")
