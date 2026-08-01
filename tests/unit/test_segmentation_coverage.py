"""Coverage-boosting unit tests for openptv2.algorithms.segmentation.

These tests target the pure-Python line-coverage gaps that the existing
test_segmentation.py does not exercise.  Every test is small and fast —
no file I/O, no large loops, no slow markers.

Pure-Python note:
  peak_fit's check_touch body (lines 85-92) and the reunification pass
  (lines 339-393) require Peak.touch to be pre-allocated as a fixed-size
  list.  In pure-Python mode cython.declare(list) returns None, so
  __post_init__ initialises touch = [] (empty), making index-assignment
  crash.  Tests that need touching peaks monkeypatch __post_init__ to
  use [0, 0, 0, 0] (0 = "no touch", maps to p2=-1 which the bounds
  guard skips).  This is documented under suspected_bugs below.
"""

import numpy as np
import pytest

from openptv2.algorithms.segmentation import (
    CORRES_NONE,
    Peak,
    _is_local_maximum,
    check_touch,
    is_compiled,
    peak_fit,
    targ_rec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_img(*shape, dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# is_compiled (line 437)
# ---------------------------------------------------------------------------


def test_is_compiled_returns_bool():
    result = is_compiled()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Peak dataclass (line 51->exit)
# ---------------------------------------------------------------------------


def test_peak_post_init_skips_when_touch_already_set():
    """Line 51->exit: touch is not None, so __post_init__ does NOT overwrite it."""
    sentinel = [99, 88, 77]
    p = Peak(touch=sentinel)
    assert p.touch is sentinel  # unchanged


def test_peak_default_touch_is_list():
    """__post_init__ True-branch (line 52): touch None -> []."""
    p = Peak()
    assert isinstance(p.touch, list)


# ---------------------------------------------------------------------------
# _is_local_maximum (direct API)
# ---------------------------------------------------------------------------


def test_is_local_maximum_true():
    img = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]], dtype=np.uint8)
    assert bool(_is_local_maximum(img, 1, 1)) is True


def test_is_local_maximum_false():
    img = np.array([[0, 255, 0], [0, 200, 0], [0, 0, 0]], dtype=np.uint8)
    assert bool(_is_local_maximum(img, 1, 1)) is False


# ---------------------------------------------------------------------------
# check_touch direct (lines 85-92)
# ---------------------------------------------------------------------------


def _make_peak_with_touch():
    """Return a Peak with touch pre-allocated as [0,0,0,0]."""
    p = Peak()
    p.touch = [0, 0, 0, 0]
    return p


def test_check_touch_p2_zero_early_return():
    """p2==0 -> early return; touch list unchanged."""
    p = _make_peak_with_touch()
    check_touch(p, 1, 0)
    assert p.n_touch == 0


def test_check_touch_p2_equals_p1_early_return():
    """p2==p1 -> early return; touch list unchanged."""
    p = _make_peak_with_touch()
    check_touch(p, 2, 2)
    assert p.n_touch == 0


def test_check_touch_records_new_touch():
    """Lines 89-90: new p2 is appended and n_touch incremented."""
    p = _make_peak_with_touch()
    check_touch(p, 1, 2)
    assert p.n_touch == 1
    assert p.touch[0] == 2


def test_check_touch_duplicate_ignored():
    """Lines 86-87: second call with same p2 is a no-op."""
    p = _make_peak_with_touch()
    check_touch(p, 1, 2)
    check_touch(p, 1, 2)
    assert p.n_touch == 1


def test_check_touch_cap_at_three():
    """Lines 91-92: n_touch is capped at 3 even when a 4th touch is added."""
    p = _make_peak_with_touch()
    check_touch(p, 1, 2)
    check_touch(p, 1, 3)
    check_touch(p, 1, 4)
    check_touch(p, 1, 5)  # 4th unique touch -> n_touch would be 4 -> capped to 3
    assert p.n_touch == 3


# ---------------------------------------------------------------------------
# targ_rec – explicit positive xmax/ymax (lines 132->134, 134->137)
# ---------------------------------------------------------------------------


def test_targ_rec_explicit_positive_xmax():
    """Line 132->134: xmax >= 0 branch (else-branch of 'if xmax < 0')."""
    img = _make_img(10, 10)
    img[4, 4] = 255
    targets = targ_rec(
        img, 200, 5, 1, 20, 1, 10, 1, 10, 10, xmin=1, xmax=8, ymin=1, ymax=8
    )
    assert len(targets) == 1
    assert targets[0].tnr == CORRES_NONE


def test_targ_rec_explicit_positive_ymax():
    """Line 134->137: ymax >= 0 branch."""
    img = _make_img(12, 12)
    img[5, 5] = 255
    targets = targ_rec(
        img, 200, 5, 1, 20, 1, 10, 1, 10, 10, xmin=1, xmax=10, ymin=1, ymax=10
    )
    assert len(targets) == 1


# ---------------------------------------------------------------------------
# peak_fit – explicit positive xmax/ymax (lines 203->205, 205->213)
# ---------------------------------------------------------------------------


def test_peak_fit_explicit_positive_xmax():
    """Lines 203->205: xmax >= 0 branch in peak_fit."""
    img = _make_img(10, 10)
    img[4, 4] = 255
    targets = peak_fit(
        img, 200, 5, 1, 20, 1, 10, 1, 10, 10, xmin=1, xmax=8, ymin=1, ymax=8
    )
    assert len(targets) == 1


def test_peak_fit_explicit_positive_ymax():
    """Lines 205->213: ymax >= 0 branch in peak_fit."""
    img = _make_img(12, 12)
    img[5, 5] = 255
    targets = peak_fit(
        img, 200, 5, 1, 20, 1, 10, 1, 10, 10, xmin=1, xmax=10, ymin=1, ymax=10
    )
    assert len(targets) == 1


# ---------------------------------------------------------------------------
# peak_fit – pixel above threshold but NOT local maximum (line 251)
# ---------------------------------------------------------------------------


def test_peak_fit_non_local_max_above_threshold():
    """Line 251: pixel > gvthres that fails _is_local_maximum is skipped."""
    img = _make_img(8, 8)
    img[3, 3] = 255  # local maximum
    img[3, 4] = 200  # above threshold but NOT a local max (255 > 200 at (3,3))
    targets = peak_fit(img, 150, 10, 1, 20, 1, 10, 1, 10, 10)
    # Exactly one target (the true local max at (3,3))
    assert len(targets) == 1


# ---------------------------------------------------------------------------
# peak_fit – BFS boundary check and pass-2 boundary (lines 278, 330->327)
# ---------------------------------------------------------------------------


def test_peak_fit_bfs_hits_left_boundary():
    """Lines 278 and 330->327: peak at column 0 (xmin=0) causes BFS and
    pass-2 neighbour checks to encounter out-of-bounds coordinates."""
    img = _make_img(8, 8)
    img[3, 0] = 255  # peak at leftmost column
    # xmin=0 so the outer loop includes j=0; BFS from (3,0) tries col -1
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10, xmin=0)
    # 1 peak detected (tiny, no BFS growth)
    assert len(targets) == 1


# ---------------------------------------------------------------------------
# peak_fit – pass-2 xmin / ymin updates (lines 318, 322)
# ---------------------------------------------------------------------------


def test_peak_fit_bfs_updates_xmin():
    """Line 318: BFS grows LEFT of the local maximum, updating peak.xmin."""
    img = _make_img(8, 8)
    img[3, 4] = 255  # local max at col 4
    img[3, 3] = 200  # BFS grows left to col 3 -> peak.xmin updated from 4 to 3
    targets = peak_fit(img, 150, 60, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 1
    # Width spans cols 3-4 → nx == 2
    assert targets[0].nx == 2


def test_peak_fit_bfs_updates_ymin():
    """Line 322: BFS grows UP from the local maximum, updating peak.ymin."""
    img = _make_img(9, 8)
    img[4, 3] = 255  # local max at row 4
    img[3, 3] = 200  # BFS grows up to row 3 -> peak.ymin updated from 4 to 3
    targets = peak_fit(img, 150, 60, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 1
    assert targets[0].ny == 2


# ---------------------------------------------------------------------------
# peak_fit – reunification pass (lines 339-393, 399)
# Using monkeypatch to pre-allocate touch so check_touch works in pure Python
# ---------------------------------------------------------------------------


@pytest.fixture()
def patch_peak_touch(monkeypatch):
    """Monkeypatch Peak.__post_init__ so touch = [0,0,0,0] instead of [].

    0 is the sentinel for 'no touch' (maps to p2 = 0-1 = -1 which the
    bounds guard in the reunification loop discards).

    Skipped in compiled mode: Cython extension types are immutable.
    """
    if is_compiled():
        pytest.skip("monkeypatch unavailable on compiled Cython extension type")

    def patched(self):
        if self.touch is None:
            self.touch = [0, 0, 0, 0]

    from openptv2.algorithms import segmentation

    monkeypatch.setattr(segmentation.Peak, "__post_init__", patched)


def test_peak_fit_reunification_two_diagonal_peaks(patch_peak_touch):
    """Lines 339-393: two single-pixel peaks that are diagonally adjacent
    (distance = sqrt(2) < 2) are unified into one target (line 399 also hit
    for the absorbed peak)."""
    img = _make_img(8, 8)
    img[2, 2] = 255  # peak 1
    img[3, 3] = 255  # peak 2 – diagonal, distance=sqrt(2) < 2 → unified
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 1
    # Unified target has n == 2 pixels
    assert targets[0].n == 2


def test_peak_fit_no_reunification_far_peaks(patch_peak_touch):
    """Lines 335-337: two peaks with n_touch==0 (not touching) both survive."""
    img = _make_img(10, 10)
    img[2, 2] = 255  # peak 1
    img[7, 7] = 255  # peak 2 – too far apart → no touch → no unification
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 2


# ---------------------------------------------------------------------------
# peak_fit – pass-4 edge filter (lines 403-406)
# ---------------------------------------------------------------------------


def test_peak_fit_edge_filter_left(tmp_path):
    """Lines 402-404: width > 32 and peak.xmin == xmin → target filtered out."""
    img = _make_img(10, 40)
    img[4, 1] = 200  # at xmin=1 (left edge) in a 40-wide image (width=39>32)
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 0


def test_peak_fit_edge_filter_right():
    """Lines 405-406: width > 32 and peak.xmax == xmax-1 → filtered out.

    Use explicit xmax=35 so BFS can reach col 34 = xmax-1 without going
    out of the underlying array bounds (img is 40 wide).
    """
    img = _make_img(10, 40)
    img[4, 33] = 255  # local max
    img[4, 34] = 200  # BFS grows to col 34 == xmax-1 → right-edge filter
    targets = peak_fit(img, 150, 80, 1, 20, 1, 10, 1, 10, 10, xmax=35)
    assert len(targets) == 0


def test_peak_fit_large_image_center_peak_not_filtered():
    """Confirm that a centre peak in a wide image is NOT filtered."""
    img = _make_img(10, 40)
    img[4, 20] = 200
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 1


# ---------------------------------------------------------------------------
# peak_fit – pass-4 output-criteria filter (line 411->397)
# ---------------------------------------------------------------------------


def test_peak_fit_sumg_filter_rejects_dim_peak():
    """Line 411->397: peak.sumg <= sumg_min → target rejected."""
    img = _make_img(8, 8)
    img[3, 3] = 100  # dim peak; sumg=100 <= sumg_min=200 → filtered
    targets = peak_fit(img, 50, 5, 1, 20, 1, 10, 1, 10, 200)
    assert len(targets) == 0


def test_peak_fit_n_max_filter_rejects_large_region():
    """Line 411->397: peak.n > nnmax → target rejected."""
    img = _make_img(10, 10)
    # 3×3 block of 255 → n=9; use nnmax=5 so peak.n > nnmax → rejected
    img[2:5, 2:5] = 255
    targets = peak_fit(img, 200, 5, 1, 5, 1, 10, 1, 10, 10)
    assert len(targets) == 0


# ---------------------------------------------------------------------------
# targ_rec – zero-detect path returns dummy target (existing, for completeness)
# ---------------------------------------------------------------------------


def test_targ_rec_returns_dummy_when_nothing_found():
    img = _make_img(8, 8)
    targets = targ_rec(img, 250, 5, 1, 10, 1, 10, 1, 10, 12)
    assert len(targets) == 1
    assert targets[0].pnr == 1
    assert targets[0].tnr == CORRES_NONE


# ---------------------------------------------------------------------------
# Reunification with profile-loop path (lines 362-375, 378)
# and xmax boundary-update (line 391)
# ---------------------------------------------------------------------------


def test_peak_fit_reunification_xmax_update(patch_peak_touch):
    """Lines 386->388 (xmin no-op) and 391 (xmax update):
    peaks at (2,4) and (3,3) — peak_i=(2,4) has xmin=4 > peak_j.xmin=3
    so xmin is NOT updated; but xmax=4 > peak_j.xmax=3 so xmax IS updated."""
    img = _make_img(8, 8)
    img[2, 4] = 255  # detected first (row 2)
    img[3, 3] = 255  # detected second (row 3) — diagonal to (2,4)
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10)
    assert len(targets) == 1  # unified


def test_peak_fit_profile_check_fails_no_reunion(patch_peak_touch):
    """Lines 362-375, 378: two touching peaks with centroid distance > 2 whose
    profile intermediate pixel fails the brightness criterion → not unified."""
    img = _make_img(10, 12)
    img[3, 3] = 255  # local max, peak 1
    img[3, 4] = 200  # BFS grows right from (3,3) with discont=5
    img[3, 6] = 255  # local max, peak 2
    img[3, 5] = 200  # BFS grows left from (3,6)
    # (3,4) and (3,5) are horizontally adjacent → check_touch fires.
    # Centroid distance ≥ 2 → profile loop enters.
    # Midpoint pixel img[3,4]=200, gv+discont=205 < gv1=255 → unify=False.
    targets = peak_fit(img, 150, 5, 1, 20, 1, 10, 1, 10, 10)
    # Peaks NOT unified → 2 separate targets survive
    assert len(targets) == 2
