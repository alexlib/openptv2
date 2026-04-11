"""Targeted unit tests for track.py bug fixes and perf improvements.

Each test exercises a single function in isolation — no integration data needed.
"""

import numpy as np
import pytest

from algorithms.constants import (
    CORRES_NONE,
    MAX_CANDS,
    PT_UNUSED,
    TR_MAX_CAMS,
    TR_UNUSED,
)
from algorithms.tracking_frame_buf import Corres_dtype, Frame, Pathinfo, Target
from algorithms.track import (
    _candsearch_in_pix_core,
    _sort_candidates_by_freq_njit,
    add_particle,
    angle_acc,
    candsearch_in_pix_rest,
    copy_foundpix_array,
    Foundpix_dtype,
    sort_candidates_by_freq,
)


# ---------------------------------------------------------------------------
# 1. add_particle — must write ONLY the new record, not corrupt others
# ---------------------------------------------------------------------------
class TestAddParticle:
    def _make_frame(self, num_cams=4, num_existing=3):
        """Create a frame with some existing particles."""
        frm = Frame(num_cams=num_cams, max_targets=20)
        # Simulate existing particles
        for i in range(num_existing):
            frm.path_info[i].x = np.array([float(i), float(i), float(i)])
            frm.correspond[i].nr = i
            frm.correspond[i].p[:] = i * 10 + np.arange(4)
        frm.num_parts = num_existing
        return frm

    def test_add_particle_does_not_corrupt_existing(self):
        """Bug #1: old code broadcast to ALL correspond records."""
        frm = self._make_frame(num_cams=4, num_existing=3)

        # Snapshot existing records before the call
        old_corr = frm.correspond[:3].copy()

        pos = np.array([10.0, 20.0, 30.0])
        cand_inds = np.full((4, MAX_CANDS), PT_UNUSED, dtype=np.int32)
        cand_inds[0, 0] = 5  # cam 0 has candidate at target index 5
        cand_inds[2, 0] = 7  # cam 2 has candidate at target index 7

        add_particle(frm, pos, cand_inds)

        # Existing records must be untouched
        for i in range(3):
            assert frm.correspond[i].nr == old_corr[i].nr, (
                f"correspond[{i}].nr corrupted: {frm.correspond[i].nr} != {old_corr[i].nr}"
            )
            np.testing.assert_array_equal(
                frm.correspond[i].p, old_corr[i].p,
                err_msg=f"correspond[{i}].p corrupted",
            )

    def test_add_particle_sets_new_record_correctly(self):
        """New record at num_parts should have correct cam mappings."""
        frm = self._make_frame(num_cams=4, num_existing=3)

        pos = np.array([10.0, 20.0, 30.0])
        cand_inds = np.full((4, MAX_CANDS), PT_UNUSED, dtype=np.int32)
        cand_inds[1, 0] = 8  # cam 1 sees target 8

        add_particle(frm, pos, cand_inds)

        new_rec = frm.correspond[3]
        assert new_rec.p[0] == CORRES_NONE  # cam 0 — no candidate
        assert new_rec.p[1] == 8            # cam 1 — target index 8
        assert new_rec.p[2] == CORRES_NONE  # cam 2 — no candidate
        assert new_rec.p[3] == CORRES_NONE  # cam 3 — no candidate
        assert new_rec.nr == 3

    def test_add_particle_increments_num_parts(self):
        frm = self._make_frame(num_cams=4, num_existing=3)
        pos = np.array([1.0, 2.0, 3.0])
        cand_inds = np.full((4, MAX_CANDS), PT_UNUSED, dtype=np.int32)
        add_particle(frm, pos, cand_inds)
        assert frm.num_parts == 4

    def test_add_particle_sets_pathinfo(self):
        frm = self._make_frame(num_cams=2, num_existing=0)
        pos = np.array([5.0, 6.0, 7.0])
        cand_inds = np.full((2, MAX_CANDS), PT_UNUSED, dtype=np.int32)
        add_particle(frm, pos, cand_inds)
        np.testing.assert_array_equal(frm.path_info[0].x, pos)


# ---------------------------------------------------------------------------
# 2. _candsearch_in_pix_core — sentinel must be PT_UNUSED (-999), not TR_UNUSED (-1)
# ---------------------------------------------------------------------------
class TestCandsearchSentinel:
    def test_unused_slots_are_pt_unused(self):
        """Returned p[] must use PT_UNUSED=-999, not TR_UNUSED=-1."""
        # Put one target in bounds
        target_x = np.array([100.0], dtype=np.float64)
        target_y = np.array([100.0], dtype=np.float64)
        target_tnr = np.array([0], dtype=np.int32)  # not TR_UNUSED

        p, _ = _candsearch_in_pix_core(
            target_x, target_y, target_tnr,
            num_targets=1,
            cent_x=100.0, cent_y=100.0,
            dl=50.0, dr=50.0, du=50.0, dd=50.0,
            imx=1024.0, imy=1024.0,
            require_unused=False,
        )
        # Slot 0 should be 0 (found target index 0)
        assert p[0] == 0
        # Slots 1-3 should be PT_UNUSED, not TR_UNUSED
        for i in range(1, MAX_CANDS):
            assert p[i] == PT_UNUSED, f"p[{i}]={p[i]}, expected PT_UNUSED={PT_UNUSED}"

    def test_no_candidates_all_pt_unused(self):
        """When no target is in the search box, all slots should be PT_UNUSED."""
        target_x = np.array([500.0], dtype=np.float64)
        target_y = np.array([500.0], dtype=np.float64)
        target_tnr = np.array([0], dtype=np.int32)

        p, _ = _candsearch_in_pix_core(
            target_x, target_y, target_tnr,
            num_targets=1,
            cent_x=100.0, cent_y=100.0,
            dl=10.0, dr=10.0, du=10.0, dd=10.0,
            imx=1024.0, imy=1024.0,
            require_unused=False,
        )
        for i in range(MAX_CANDS):
            assert p[i] == PT_UNUSED


# ---------------------------------------------------------------------------
# 3. candsearch_in_pix_rest — must reset p[0] to PT_UNUSED on entry
# ---------------------------------------------------------------------------
class TestCandsearchRestInit:
    def test_p0_reset_when_no_match(self):
        """p[0] must be PT_UNUSED if no candidate found, regardless of input."""
        from algorithms.parameters import ControlPar

        targets = [Target() for _ in range(5)]
        for i, t in enumerate(targets):
            t.x = float(i * 100)
            t.y = float(i * 100)
            t.tnr = 0  # all matched — rest search wants tnr == TR_UNUSED

        cpar = ControlPar()
        cpar.imx = 1024
        cpar.imy = 1024

        # Pass p with stale value (not -999)
        p = [42]  # stale
        count = candsearch_in_pix_rest(
            targets, 5,
            cent_x=50.0, cent_y=50.0,
            dl=10.0, dr=10.0, du=10.0, dd=10.0,
            p=p, cpar=cpar,
        )
        assert count == 0
        assert p[0] == PT_UNUSED, f"p[0]={p[0]}, expected PT_UNUSED"


# ---------------------------------------------------------------------------
# 5. copy_foundpix_array — slice copy vs. element loop
# ---------------------------------------------------------------------------
class TestCopyFoundpixArray:
    def test_copy_matches(self):
        """copy_foundpix_array must faithfully copy all fields."""
        n = 6
        src = np.zeros(n, dtype=Foundpix_dtype).view(np.recarray)
        dest = np.zeros(n, dtype=Foundpix_dtype).view(np.recarray)

        for i in range(n):
            src[i].ftnr = i * 10
            src[i].freq = i + 1
            src[i].whichcam[:] = [i, i + 1, i + 2, i + 3]

        copy_foundpix_array(dest, src, n, 4)

        for i in range(n):
            assert dest[i].ftnr == src[i].ftnr
            assert dest[i].freq == src[i].freq
            np.testing.assert_array_equal(dest[i].whichcam, src[i].whichcam)


# ---------------------------------------------------------------------------
# 6. angle_acc — correctness sanity
# ---------------------------------------------------------------------------
class TestAngleAcc:
    def test_same_direction_zero_angle(self):
        start = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 0.0, 0.0])
        cand = np.array([1.0, 0.0, 0.0])
        angle, acc = angle_acc(start, pred, cand)
        assert angle == pytest.approx(0.0)
        assert acc == pytest.approx(0.0)

    def test_opposite_direction_200_gon(self):
        start = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 0.0, 0.0])
        cand = np.array([-1.0, 0.0, 0.0])
        angle, acc = angle_acc(start, pred, cand)
        assert angle == pytest.approx(200.0)

    def test_right_angle(self):
        start = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 0.0, 0.0])
        cand = np.array([0.0, 1.0, 0.0])
        angle, acc = angle_acc(start, pred, cand)
        assert angle == pytest.approx(100.0)  # 90° = 100 gon

    def test_returns_tuple_not_array(self):
        """angle_acc must return a tuple (no array allocation)."""
        start = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 0.0, 0.0])
        cand = np.array([0.5, 0.5, 0.0])
        result = angle_acc(start, pred, cand)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2

    def test_angle_acc_benchmark(self):
        """Benchmark angle_acc — should be very fast with tuple return."""
        import time
        start = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 0.0, 0.0])
        cand = np.array([0.5, 0.5, 0.0])
        # warmup
        angle_acc(start, pred, cand)

        N = 100000
        t0 = time.perf_counter()
        for _ in range(N):
            angle_acc(start, pred, cand)
        elapsed = time.perf_counter() - t0
        per_call_us = elapsed / N * 1e6
        print(f"\nangle_acc: {per_call_us:.2f} µs/call ({N} iters)")
        assert per_call_us < 5, f"Too slow: {per_call_us:.2f} µs"


# ---------------------------------------------------------------------------
# 7. sort_candidates_by_freq — @njit version correctness
# ---------------------------------------------------------------------------
class TestSortCandidatesByFreq:
    def test_single_cam_single_target(self):
        """One camera, one target → freq=1, kept (only pruned when dup or lower freq exists)."""
        num_cams = 4
        n = num_cams * MAX_CANDS  # 16
        ftnr = np.full(n, TR_UNUSED, dtype=np.int32)
        freq = np.zeros(n, dtype=np.int32)
        whichcam = np.zeros((n, num_cams), dtype=np.int32)

        ftnr[0] = 42  # cam 0 sees target 42
        result = _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams)
        assert result == 1  # freq=1, single entry, kept
        assert ftnr[0] == 42
        assert freq[0] == 1

    def test_two_cams_same_target(self):
        """Target seen in 2 cameras → freq=2, kept."""
        num_cams = 4
        n = num_cams * MAX_CANDS
        ftnr = np.full(n, TR_UNUSED, dtype=np.int32)
        freq = np.zeros(n, dtype=np.int32)
        whichcam = np.zeros((n, num_cams), dtype=np.int32)

        ftnr[0] = 42  # cam 0 slot 0
        ftnr[4] = 42  # cam 1 slot 0

        result = _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams)
        assert result == 1
        assert ftnr[0] == 42
        assert freq[0] == 2

    def test_three_cams_same_target(self):
        """Target seen in 3 cameras → freq=3."""
        num_cams = 4
        n = num_cams * MAX_CANDS
        ftnr = np.full(n, TR_UNUSED, dtype=np.int32)
        freq = np.zeros(n, dtype=np.int32)
        whichcam = np.zeros((n, num_cams), dtype=np.int32)

        ftnr[0] = 10  # cam 0
        ftnr[4] = 10  # cam 1
        ftnr[8] = 10  # cam 2

        result = _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams)
        assert result == 1
        assert ftnr[0] == 10
        assert freq[0] == 3

    def test_two_distinct_targets(self):
        """Two targets each seen by 2 cameras → 2 results sorted by freq."""
        num_cams = 4
        n = num_cams * MAX_CANDS
        ftnr = np.full(n, TR_UNUSED, dtype=np.int32)
        freq = np.zeros(n, dtype=np.int32)
        whichcam = np.zeros((n, num_cams), dtype=np.int32)

        # target 10: cams 0,1,2 → freq 3
        ftnr[0] = 10
        ftnr[4] = 10
        ftnr[8] = 10
        # target 20: cams 1,3 → freq 2
        ftnr[5] = 20
        ftnr[13] = 20

        result = _sort_candidates_by_freq_njit(ftnr, freq, whichcam, num_cams)
        assert result == 2
        assert ftnr[0] == 10
        assert freq[0] == 3
        assert ftnr[1] == 20
        assert freq[1] == 2

    def test_recarray_wrapper(self):
        """sort_candidates_by_freq wrapper works with Foundpix_dtype recarray."""
        num_cams = 4
        n = num_cams * MAX_CANDS
        fp = np.zeros(n, dtype=Foundpix_dtype).view(np.recarray)
        fp['ftnr'] = TR_UNUSED
        fp['ftnr'][0] = 5
        fp['ftnr'][4] = 5
        fp['ftnr'][8] = 5

        result = sort_candidates_by_freq(fp, num_cams)
        assert result == 1
        assert fp['ftnr'][0] == 5
        assert fp['freq'][0] == 3

    def test_sort_benchmark(self):
        """Benchmark sort_candidates_by_freq @njit vs old Python."""
        import time
        num_cams = 4
        n = num_cams * MAX_CANDS

        # Warm up JIT
        ftnr = np.full(n, TR_UNUSED, dtype=np.int32)
        freq = np.zeros(n, dtype=np.int32)
        wc = np.zeros((n, num_cams), dtype=np.int32)
        ftnr[0] = 1; ftnr[4] = 1; ftnr[8] = 1
        _sort_candidates_by_freq_njit(ftnr, freq, wc, num_cams)

        N_ITER = 10000
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            ftnr[:] = TR_UNUSED
            freq[:] = 0
            wc[:] = 0
            ftnr[0] = 10; ftnr[4] = 10; ftnr[8] = 10
            ftnr[1] = 20; ftnr[5] = 20
            _sort_candidates_by_freq_njit(ftnr, freq, wc, num_cams)
        elapsed = time.perf_counter() - t0
        per_call_us = elapsed / N_ITER * 1e6
        print(f"\n_sort_candidates_by_freq_njit: {per_call_us:.1f} µs/call ({N_ITER} iters)")
        assert per_call_us < 50, f"Too slow: {per_call_us:.1f} µs"
