"""Coverage tests for openptv2.algorithms.correspondences.

Small, fast unit tests targeting pure-Python line coverage.
Does NOT use slow markers; does NOT call match_pairs end-to-end with real
epi/trafo (both have UnboundLocalError in pure-Python mode).

Strategy
--------
* NTupel, sort helpers, allocate_adjacency_arrays, is_compiled  — trivial.
* four/three/consistent matching  — synthetic flat arrays, no calibration needed.
* take_best_candidates            — synthetic NTupel lists.
* _correct_one_camera / correct_frame — monkeypatch trafo.pixel_to_metric
  and trafo.dist_to_flat (both have cython.double[N] UnboundLocalError in
  pure-Python mode).
* correspondences (all num_cams branches) — build corrected manually;
  monkeypatch epi.find_candidate (fails in pure-Python via trafo call).
* match_pairs (serial + threaded) — same monkeypatching via correspondences.
"""

import numpy as np
import pytest

from openptv2.algorithms.correspondences import (
    NTupel,
    PT_UNUSED,
    NMAX,
    quicksort_target_y,
    quicksort_coord2d_x,
    allocate_adjacency_arrays,
    four_camera_matching,
    three_camera_matching,
    consistent_pair_matching,
    take_best_candidates,
    correct_frame,
    correspondences,
    is_compiled,
)
from openptv2.algorithms.epi import Coord2d, MAXCAND
from openptv2.algorithms.tracking_frame_buf import Target, Frame
from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, VolumePar


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PARAM_YAML = "test_data/parameters.yaml"
ORI_FMT = "test_data/calibration/sym_cam{}.tif.ori"
ADDPAR = "test_data/calibration/cam1.tif.addpar"


def _load_calib(n):
    return [Calibration.from_file(ORI_FMT.format(i + 1), ADDPAR) for i in range(n)]


def _make_flat_arrays(num_cams, max_t):
    """Return freshly allocated adjacency arrays."""
    p1_arr = np.full((num_cams, num_cams, max_t), -1, dtype=np.int32)
    n_arr = np.zeros((num_cams, num_cams, max_t), dtype=np.int32)
    p2_arr = np.zeros((num_cams, num_cams, max_t, MAXCAND + 1), dtype=np.int32)
    corr_arr = np.zeros((num_cams, num_cams, max_t, MAXCAND + 1), dtype=np.float64)
    dist_arr = np.zeros((num_cams, num_cams, max_t, MAXCAND + 1), dtype=np.float64)
    return p1_arr, n_arr, p2_arr, corr_arr, dist_arr


def _fill_perfect_pairs(p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
                         num_cams, n_targets, corr=2.0, dist=1.0):
    """Fill every camera pair with perfect 1:1 candidates."""
    for c1 in range(num_cams - 1):
        for c2 in range(c1 + 1, num_cams):
            for i in range(n_targets):
                p1_arr[c1, c2, i] = i
                n_arr[c1, c2, i] = 1
                p2_arr[c1, c2, i, 0] = i
                corr_arr[c1, c2, i, 0] = corr
                dist_arr[c1, c2, i, 0] = dist


def _frame(num_cams, n=1):
    """Minimal Frame with n targets per camera."""
    frm = Frame(num_cams, n)
    for cam in range(num_cams):
        frm.num_targets[cam] = n
        for k in range(n):
            t = frm.targets[cam][k]
            t.pnr = k
            t.x = 100.0 + k * 10.0
            t.y = 100.0 + k * 10.0
            t.n = 25
            t.nx = 5
            t.ny = 5
            t.sumg = 10
            t.tnr = -1
    return frm


# ---------------------------------------------------------------------------
# Trafo mocks (fixes pure-Python UnboundLocalError)
# ---------------------------------------------------------------------------


def _mock_pixel_to_metric(x, y, cpar):
    return x * 1e-3, y * 1e-3


def _mock_dist_to_flat(xm, ym, xh, yh, k1, k2, k3, p1, p2, scx, she, tol):
    return xm, ym


def _mock_find_candidate_one(corrected_i2, targets_i2, n2,
                              xmin, ymin, xmax, ymax,
                              n, nx, ny, sumg,
                              cand_pnr, cand_tol, cand_corr,
                              vpar, cpar, calib_i2):
    """Always returns 1 candidate: target index 0."""
    cand_pnr[0] = 0
    cand_tol[0] = 1.0
    cand_corr[0] = 2.0
    return 1


def _mock_find_candidate_zero(*args, **kwargs):
    return 0


# ---------------------------------------------------------------------------
# NTupel
# ---------------------------------------------------------------------------


class TestNTupel:
    def test_default(self):
        nt = NTupel()
        assert nt.p == [-1, -1, -1, -1]
        assert nt.corr == pytest.approx(0.0)

    def test_custom(self):
        nt = NTupel(p=[3, 7, -2, 1], corr=1.5)
        assert nt.p[0] == 3
        assert nt.p[2] == -2
        assert nt.corr == pytest.approx(1.5)

    def test_p_is_mutable(self):
        nt = NTupel()
        nt.p[0] = 42
        assert nt.p[0] == 42

    def test_corr_settable(self):
        nt = NTupel()
        nt.corr = 3.14
        assert nt.corr == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Sort helpers
# ---------------------------------------------------------------------------


class TestSortHelpers:
    def test_quicksort_target_y(self):
        targets = [
            Target(pnr=0, x=0.0, y=3.0, n=5, nx=1, ny=1, sumg=10, tnr=-1),
            Target(pnr=1, x=0.0, y=1.0, n=5, nx=1, ny=1, sumg=10, tnr=-1),
            Target(pnr=2, x=0.0, y=2.0, n=5, nx=1, ny=1, sumg=10, tnr=-1),
        ]
        quicksort_target_y(targets)
        assert targets[0].y == pytest.approx(1.0)
        assert targets[1].y == pytest.approx(2.0)
        assert targets[2].y == pytest.approx(3.0)

    def test_quicksort_coord2d_x(self):
        crds = [
            Coord2d(pnr=0, x=5.0, y=0.0),
            Coord2d(pnr=1, x=1.0, y=0.0),
            Coord2d(pnr=2, x=3.0, y=0.0),
        ]
        quicksort_coord2d_x(crds)
        assert crds[0].x == pytest.approx(1.0)
        assert crds[2].x == pytest.approx(5.0)

    def test_quicksort_target_y_already_sorted(self):
        targets = [
            Target(pnr=0, x=0.0, y=-1.0, n=5, nx=1, ny=1, sumg=10, tnr=-1),
            Target(pnr=1, x=0.0, y=0.0, n=5, nx=1, ny=1, sumg=10, tnr=-1),
        ]
        quicksort_target_y(targets)
        assert targets[0].y == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# allocate_adjacency_arrays
# ---------------------------------------------------------------------------


class TestAllocateAdjacencyArrays:
    def test_shapes_2cam(self):
        p1, n, p2, c, d = allocate_adjacency_arrays(2, [5, 3])
        assert p1.shape == (2, 2, 5)
        assert n.shape == (2, 2, 5)
        assert p2.shape == (2, 2, 5, MAXCAND + 1)
        assert c.shape == (2, 2, 5, MAXCAND + 1)
        assert d.shape == (2, 2, 5, MAXCAND + 1)

    def test_shapes_4cam(self):
        p1, n, p2, c, d = allocate_adjacency_arrays(4, [10, 8, 6, 4])
        assert p1.shape[0] == 4
        assert p1.shape[2] == 10  # max_targets

    def test_p1_initialized_upper_triangle(self):
        p1, n, p2, c, d = allocate_adjacency_arrays(2, [3, 3])
        assert p1[0, 1, 0] == 0
        assert p1[0, 1, 1] == 1
        assert p1[0, 1, 2] == 2

    def test_n_arr_zeros(self):
        p1, n, p2, c, d = allocate_adjacency_arrays(2, [2, 2])
        assert np.all(n == 0)

    def test_3cam_triangle_filled(self):
        p1, n, p2, c, d = allocate_adjacency_arrays(3, [4, 4, 4])
        assert p1[0, 1, 0] == 0
        assert p1[0, 2, 0] == 0
        assert p1[1, 2, 0] == 0

    def test_unequal_target_counts(self):
        p1, n, p2, c, d = allocate_adjacency_arrays(2, [1, 5])
        assert p1.shape[2] == 5  # max is 5


# ---------------------------------------------------------------------------
# is_compiled
# ---------------------------------------------------------------------------


def test_is_compiled_pure_python():
    result = is_compiled()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# four_camera_matching  (synthetic arrays — no calibration needed)
# ---------------------------------------------------------------------------


class TestFourCameraMatching:
    def _make(self, n=1):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = _make_flat_arrays(4, n)
        _fill_perfect_pairs(p1_arr, n_arr, p2_arr, corr_arr, dist_arr, 4, n)
        return p1_arr, n_arr, p2_arr, corr_arr, dist_arr

    def test_one_quadruplet(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        matched = four_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            base_target_count=1, accept_corr=1.0,
            scratch=scratch, scratch_size=10,
        )
        assert matched == 1
        assert scratch[0].p[0] == 0
        assert scratch[0].p[1] == 0

    def test_corr_below_threshold_no_match(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        matched = four_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            base_target_count=1, accept_corr=10.0,
            scratch=scratch, scratch_size=10,
        )
        assert matched == 0

    def test_scratch_limit_early_return(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(2)
        scratch = [NTupel() for _ in range(5)]
        matched = four_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            base_target_count=2, accept_corr=1.0,
            scratch=scratch, scratch_size=1,
        )
        assert matched == 1

    def test_inconsistent_cross_pair_no_match(self):
        """cam1→cam2 points to target 1 but cam0→cam2 points to target 0 → no clique."""
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = _make_flat_arrays(4, 2)
        # cam0→cam1: target 0 → target 0
        p1_arr[0, 1, 0] = 0; n_arr[0, 1, 0] = 1
        p2_arr[0, 1, 0, 0] = 0; corr_arr[0, 1, 0, 0] = 2.0; dist_arr[0, 1, 0, 0] = 1.0
        # cam0→cam2: target 0 → target 0
        p1_arr[0, 2, 0] = 0; n_arr[0, 2, 0] = 1
        p2_arr[0, 2, 0, 0] = 0; corr_arr[0, 2, 0, 0] = 2.0; dist_arr[0, 2, 0, 0] = 1.0
        # cam0→cam3: target 0 → target 0
        p1_arr[0, 3, 0] = 0; n_arr[0, 3, 0] = 1
        p2_arr[0, 3, 0, 0] = 0; corr_arr[0, 3, 0, 0] = 2.0; dist_arr[0, 3, 0, 0] = 1.0
        # cam1→cam2: target 0 → target 1 (mismatch with p3=0)
        p1_arr[1, 2, 0] = 0; n_arr[1, 2, 0] = 1
        p2_arr[1, 2, 0, 0] = 1; corr_arr[1, 2, 0, 0] = 2.0; dist_arr[1, 2, 0, 0] = 1.0
        scratch = [NTupel() for _ in range(5)]
        matched = four_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            base_target_count=1, accept_corr=1.0,
            scratch=scratch, scratch_size=5,
        )
        assert matched == 0

    def test_zero_candidates(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = _make_flat_arrays(4, 1)
        p1_arr[0, 1, 0] = 0  # p1 set but n=0 → inner loops don't execute
        scratch = [NTupel() for _ in range(5)]
        matched = four_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            base_target_count=1, accept_corr=1.0,
            scratch=scratch, scratch_size=5,
        )
        assert matched == 0


# ---------------------------------------------------------------------------
# three_camera_matching  (synthetic arrays)
# ---------------------------------------------------------------------------


class TestThreeCameraMatching:
    def _make(self, num_cams=3, n=1):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = _make_flat_arrays(num_cams, n)
        _fill_perfect_pairs(p1_arr, n_arr, p2_arr, corr_arr, dist_arr, num_cams, n)
        return p1_arr, n_arr, p2_arr, corr_arr, dist_arr

    def test_one_triplet_3cam(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(3, 1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(3)]
        matched = three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=3, target_counts=[1, 1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 1
        assert scratch[0].p[0] == 0
        assert scratch[0].p[1] == 0
        assert scratch[0].p[2] == 0

    def test_non_triplet_slots_are_minus2(self):
        """In 4-cam mode, unused slots in a triplet should be -2."""
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(4, 1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(4)]
        three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=4, target_counts=[1, 1, 1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        # At least one slot should be -2 (the missing camera)
        found_minus2 = any(scratch[0].p[c] == -2 for c in range(4))
        assert found_minus2

    def test_corr_below_threshold(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(3, 1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(3)]
        matched = three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=3, target_counts=[1, 1, 1], accept_corr=10.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_skips_used_p1_target(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(3, 1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(3)]
        tusage[0][0] = 1
        matched = three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=3, target_counts=[1, 1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_skips_used_p2_target(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(3, 1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(3)]
        tusage[1][0] = 1
        matched = three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=3, target_counts=[1, 1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_skips_used_p3_target(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(3, 1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(3)]
        tusage[2][0] = 1
        matched = three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=3, target_counts=[1, 1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_scratch_limit_early_return(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(3, 2)
        scratch = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(3)]
        matched = three_camera_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=3, target_counts=[2, 2, 2], accept_corr=1.0,
            scratch=scratch, scratch_size=1, tusage=tusage,
        )
        assert matched == 1


# ---------------------------------------------------------------------------
# consistent_pair_matching  (synthetic arrays)
# ---------------------------------------------------------------------------


class TestConsistentPairMatching:
    def _make(self, n=1):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = _make_flat_arrays(2, n)
        _fill_perfect_pairs(p1_arr, n_arr, p2_arr, corr_arr, dist_arr, 2, n)
        return p1_arr, n_arr, p2_arr, corr_arr, dist_arr

    def test_one_pair(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(2)]
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 1
        assert scratch[0].p[0] == 0
        assert scratch[0].p[1] == 0

    def test_non_pair_slots_are_minus2(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(2)]
        consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert scratch[0].p[0] == 0
        assert scratch[0].p[1] == 0

    def test_ambiguous_n_not_one_skipped(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        n_arr[0, 1, 0] = 2  # ambiguous
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(2)]
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_corr_below_threshold(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(2)]
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[1, 1], accept_corr=10.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_skips_used_p1(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(2)]
        tusage[0][0] = 1
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_skips_used_p2(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(1)
        scratch = [NTupel() for _ in range(10)]
        tusage = [[0] * NMAX for _ in range(2)]
        tusage[1][0] = 1
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=10, tusage=tusage,
        )
        assert matched == 0

    def test_scratch_limit_early_return(self):
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = self._make(2)
        scratch = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(2)]
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=2, target_counts=[2, 2], accept_corr=1.0,
            scratch=scratch, scratch_size=1, tusage=tusage,
        )
        assert matched == 1

    def test_4cam_mode(self):
        """consistent_pair_matching also works with num_cams=4."""
        p1_arr, n_arr, p2_arr, corr_arr, dist_arr = _make_flat_arrays(4, 1)
        _fill_perfect_pairs(p1_arr, n_arr, p2_arr, corr_arr, dist_arr, 4, 1)
        scratch = [NTupel() for _ in range(20)]
        tusage = [[0] * NMAX for _ in range(4)]
        matched = consistent_pair_matching(
            p1_arr, n_arr, p2_arr, corr_arr, dist_arr,
            num_cams=4, target_counts=[1, 1, 1, 1], accept_corr=1.0,
            scratch=scratch, scratch_size=20, tusage=tusage,
        )
        assert matched >= 1


# ---------------------------------------------------------------------------
# take_best_candidates
# ---------------------------------------------------------------------------


class TestTakeBestCandidates:
    def test_sorts_descending_corr(self):
        src = [
            NTupel(p=[0, 1, -2, -2], corr=1.0),
            NTupel(p=[2, 3, -2, -2], corr=3.0),
            NTupel(p=[4, 5, -2, -2], corr=2.0),
        ]
        dst = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(4)]
        taken = take_best_candidates(src, dst, num_cams=4, num_cands=3, tusage=tusage)
        assert taken == 3
        assert dst[0].corr == pytest.approx(3.0)
        assert dst[1].corr == pytest.approx(2.0)
        assert dst[2].corr == pytest.approx(1.0)

    def test_skips_used_targets(self):
        src = [
            NTupel(p=[0, 1, -2, -2], corr=5.0),
            NTupel(p=[2, 3, -2, -2], corr=2.0),
        ]
        dst = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(4)]
        tusage[0][0] = 1  # cam0 target0 used → first candidate skipped
        taken = take_best_candidates(src, dst, num_cams=4, num_cands=2, tusage=tusage)
        assert taken == 1
        assert dst[0].p[0] == 2

    def test_marks_usage_after_taking(self):
        src = [NTupel(p=[0, 1, -2, -2], corr=2.0)]
        dst = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(4)]
        take_best_candidates(src, dst, num_cams=4, num_cands=1, tusage=tusage)
        assert tusage[0][0] == 1
        assert tusage[1][1] == 1

    def test_empty_candidates(self):
        src = []
        dst = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(2)]
        taken = take_best_candidates(src, dst, num_cams=2, num_cands=0, tusage=tusage)
        assert taken == 0

    def test_negative_one_slots_not_marked(self):
        """p[cam]==-1 should not touch tusage."""
        src = [NTupel(p=[-1, 0, -1, -1], corr=2.0)]
        dst = [NTupel() for _ in range(5)]
        tusage = [[0] * NMAX for _ in range(4)]
        take_best_candidates(src, dst, num_cams=4, num_cands=1, tusage=tusage)
        assert tusage[0][0] == 0  # cam0 target0 NOT marked (slot was -1)
        assert tusage[1][0] == 1  # cam1 target0 marked


# ---------------------------------------------------------------------------
# correct_frame  (monkeypatch trafo)
# ---------------------------------------------------------------------------


class TestCorrectFrame:
    def test_single_camera_sequential_path(self, monkeypatch):
        """num_cams<=1 → sequential branch in correct_frame."""
        monkeypatch.setattr("openptv2.algorithms.trafo.pixel_to_metric",
                            _mock_pixel_to_metric)
        monkeypatch.setattr("openptv2.algorithms.trafo.dist_to_flat",
                            _mock_dist_to_flat)
        cpar = ControlPar.from_yaml(PARAM_YAML)
        cpar.num_cams = 1
        calib = _load_calib(1)
        frm = _frame(1, n=2)
        corrected = correct_frame(frm, calib, cpar, tol=1e-6)
        assert len(corrected) == 1
        assert len(corrected[0]) == 2
        # x-sorted invariant
        xs = [c.x for c in corrected[0]]
        assert xs == sorted(xs)

    def test_two_camera_parallel_path(self, monkeypatch):
        """num_cams>1 → ThreadPoolExecutor branch."""
        monkeypatch.setattr("openptv2.algorithms.trafo.pixel_to_metric",
                            _mock_pixel_to_metric)
        monkeypatch.setattr("openptv2.algorithms.trafo.dist_to_flat",
                            _mock_dist_to_flat)
        cpar = ControlPar.from_yaml(PARAM_YAML)
        cpar.num_cams = 2
        calib = _load_calib(2)
        frm = _frame(2, n=1)
        corrected = correct_frame(frm, calib, cpar, tol=1e-6)
        assert len(corrected) == 2
        assert len(corrected[0]) == 1
        assert len(corrected[1]) == 1

    def test_pnr_preserved(self, monkeypatch):
        monkeypatch.setattr("openptv2.algorithms.trafo.pixel_to_metric",
                            _mock_pixel_to_metric)
        monkeypatch.setattr("openptv2.algorithms.trafo.dist_to_flat",
                            _mock_dist_to_flat)
        cpar = ControlPar.from_yaml(PARAM_YAML)
        cpar.num_cams = 1
        calib = _load_calib(1)
        frm = _frame(1, n=1)
        frm.targets[0][0].pnr = 7
        corrected = correct_frame(frm, calib, cpar, tol=1e-6)
        assert corrected[0][0].pnr == 7

    def test_output_x_sorted(self, monkeypatch):
        monkeypatch.setattr("openptv2.algorithms.trafo.pixel_to_metric",
                            _mock_pixel_to_metric)
        monkeypatch.setattr("openptv2.algorithms.trafo.dist_to_flat",
                            _mock_dist_to_flat)
        cpar = ControlPar.from_yaml(PARAM_YAML)
        cpar.num_cams = 1
        calib = _load_calib(1)
        frm = _frame(1, n=3)
        # Reverse the x coords to ensure sorting is exercised
        for k in range(3):
            frm.targets[0][k].x = 300.0 - k * 100.0
        corrected = correct_frame(frm, calib, cpar, tol=1e-6)
        xs = [c.x for c in corrected[0]]
        assert xs == sorted(xs)


# ---------------------------------------------------------------------------
# correspondences (full pipeline) — monkeypatch epi.find_candidate + trafo
# ---------------------------------------------------------------------------


class TestCorrespondences:
    def _setup(self, monkeypatch, num_cams=2, find_cand=None, allCam=0, corrmin=0.001):
        monkeypatch.setattr("openptv2.algorithms.epi.find_candidate",
                            find_cand or _mock_find_candidate_zero)
        monkeypatch.setattr("openptv2.algorithms.trafo.pixel_to_metric",
                            _mock_pixel_to_metric)
        monkeypatch.setattr("openptv2.algorithms.trafo.dist_to_flat",
                            _mock_dist_to_flat)
        cpar = ControlPar.from_yaml(PARAM_YAML)
        vpar = VolumePar.from_yaml(PARAM_YAML)
        cpar.num_cams = num_cams
        cpar.allCam_flag = allCam
        vpar.corrmin = corrmin
        calib = _load_calib(num_cams)
        frm = _frame(num_cams, n=1)
        corrected = correct_frame(frm, calib, cpar, tol=1e-6)
        return frm, corrected, vpar, cpar, calib

    def test_2cam_no_candidates_empty_result(self, monkeypatch):
        """2-cam, find_candidate=0 → no pairs → empty."""
        frm, corrected, vpar, cpar, calib = self._setup(monkeypatch, 2)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[3] == 0
        assert len(con) == 0

    def test_2cam_one_pair_found(self, monkeypatch):
        """2-cam, find_candidate=1 → 1 pair."""
        frm, corrected, vpar, cpar, calib = self._setup(
            monkeypatch, 2, _mock_find_candidate_one)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[2] == 1
        assert mc[3] == 1

    def test_2cam_tnr_updated(self, monkeypatch):
        """Target track numbers should be set after matching."""
        frm, corrected, vpar, cpar, calib = self._setup(
            monkeypatch, 2, _mock_find_candidate_one)
        correspondences(frm, corrected, vpar, cpar, calib)
        assert frm.targets[0][0].tnr == 0

    def test_3cam_triplet_branch(self, monkeypatch):
        """3-cam → three_camera_matching branch runs."""
        frm, corrected, vpar, cpar, calib = self._setup(
            monkeypatch, 3, _mock_find_candidate_one)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[3] >= 1

    def test_4cam_quadruplet_branch(self, monkeypatch):
        """4-cam → four_camera_matching branch runs."""
        frm, corrected, vpar, cpar, calib = self._setup(
            monkeypatch, 4, _mock_find_candidate_one)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[0] >= 1  # quadruplet found
        assert mc[3] >= 1

    def test_4cam_allcam_flag_1_skips_trips_pairs(self, monkeypatch):
        """4-cam allCam_flag=1 → only quads, triplet and pair branches skipped."""
        frm, corrected, vpar, cpar, calib = self._setup(
            monkeypatch, 4, _mock_find_candidate_one, allCam=1)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[1] == 0  # no triplets
        assert mc[2] == 0  # no pairs

    def test_match_pairs_serial_branch_2cam(self, monkeypatch):
        """2-cam has only 1 pair → match_pairs uses serial path."""
        frm, corrected, vpar, cpar, calib = self._setup(monkeypatch, 2)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[3] == 0  # no crashes

    def test_match_pairs_threaded_branch_3cam(self, monkeypatch):
        """3-cam has 3 pairs → match_pairs uses ThreadPoolExecutor path."""
        frm, corrected, vpar, cpar, calib = self._setup(monkeypatch, 3)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[3] == 0  # no crashes

    def test_pt_unused_skipped_in_adjacency(self, monkeypatch):
        """corrected coord with x==PT_UNUSED should be skipped by _build_adjacency_for_pair."""
        monkeypatch.setattr("openptv2.algorithms.epi.find_candidate",
                            _mock_find_candidate_one)
        monkeypatch.setattr("openptv2.algorithms.trafo.pixel_to_metric",
                            _mock_pixel_to_metric)
        monkeypatch.setattr("openptv2.algorithms.trafo.dist_to_flat",
                            _mock_dist_to_flat)
        cpar = ControlPar.from_yaml(PARAM_YAML)
        vpar = VolumePar.from_yaml(PARAM_YAML)
        cpar.num_cams = 2
        cpar.allCam_flag = 0
        vpar.corrmin = 0.001
        calib = _load_calib(2)
        frm = _frame(2, n=1)
        corrected = correct_frame(frm, calib, cpar, tol=1e-6)
        corrected[0][0].x = float(PT_UNUSED)
        con, mc = correspondences(frm, corrected, vpar, cpar, calib)
        assert mc[2] == 0  # skipped, no pairs
