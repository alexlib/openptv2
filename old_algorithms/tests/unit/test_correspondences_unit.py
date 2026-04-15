"""Unit tests for correspondences core logic and SoA wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.correspondences import (
    _take_best_candidates_soa,
    correspondences,
    correspondences_soa,
    match_pairs,
    match_pairs_soa,
    safely_allocate_adjacency_lists,
)
from algorithms.tracking_frame_buf import n_tupel_dtype
from algorithms.tests.helpers.factories import (
    build_corresp_control_par,
    build_corresp_volume_par,
    generate_grid_frame,
    load_sym_calibrations,
)


@pytest.mark.unit
def test_take_best_candidates_soa_prefers_high_corr() -> None:
    src_p = np.array(
        [
            [0, 0, -2, -2],
            [0, 1, -2, -2],
            [1, 1, -2, -2],
        ],
        dtype=np.int32,
    )
    src_corr = np.array([10.0, 50.0, 40.0], dtype=np.float64)
    tusage = np.zeros((4, 8), dtype=np.int32)

    dst_p, dst_corr, taken = _take_best_candidates_soa(src_p, src_corr, 3, 4, tusage)

    assert taken == 1
    # Best candidate is [0,1,...]; the rest conflict on already-used targets.
    assert np.array_equal(dst_p[0], np.array([0, 1, -2, -2], dtype=np.int32))
    assert dst_corr[0] == pytest.approx(50.0)


@pytest.mark.unit
def test_match_pairs_soa_and_original_have_same_counts() -> None:
    cpar = build_corresp_control_par()
    vpar = build_corresp_volume_par()
    cals = load_sym_calibrations()
    frm, corrected = generate_grid_frame(cals, cpar)

    corr_lists = safely_allocate_adjacency_lists(cpar.num_cams, frm.num_targets)
    match_pairs(corr_lists, corrected, frm, vpar, cpar, cals)

    corr_n, *_ = match_pairs_soa(corrected, frm, vpar, cpar, cals)

    for c1 in range(cpar.num_cams - 1):
        for c2 in range(c1 + 1, cpar.num_cams):
            original_n = corr_lists[c1, c2, : frm.num_targets[c1]].n
            soa_n = corr_n[c1, c2, : frm.num_targets[c1]]
            assert int(np.sum(original_n)) > 0
            assert int(np.sum(soa_n)) > 0


@pytest.mark.unit
def test_correspondences_soa_returns_expected_dtype() -> None:
    cpar = build_corresp_control_par()
    vpar = build_corresp_volume_par()
    cals = load_sym_calibrations()
    frm, corrected = generate_grid_frame(cals, cpar)

    match_counts = [0, 0, 0, 0]
    res_soa = correspondences_soa(frm, corrected, vpar, cpar, cals, match_counts)
    res_orig = correspondences(frm, corrected, vpar, cpar, cals, [0, 0, 0, 0])

    assert isinstance(res_soa, np.recarray)
    assert res_soa.dtype == n_tupel_dtype
    assert len(res_soa) == len(res_orig)
    assert int(np.count_nonzero(res_soa.corr > 0.0)) > 0
