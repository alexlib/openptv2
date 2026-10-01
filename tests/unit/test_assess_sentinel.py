"""Sentinel contract of the add-particle candidate indexes (trackcorr kernel).

``assess_new_position_fast_nogil`` marks a camera without a candidate with
``tr_unused`` (-1), NOT with PT_UNUSED (-999). Its consumers in track_kernels_corr
must therefore test ``index >= 0``: ``index != PT_UNUSED`` lets -1 through and indexes
``targ_tnr[ci, -1]``, an out-of-bounds read and WRITE in the compiled build (no
wraparound). That corrupted the heap and made the add-particle decision, and so the
trajectories, depend on whatever memory sat before the array (a run-to-run flake in the
openptv-cloud end-to-end test; found with libgmalloc, see the tracker plan).
"""

from pathlib import Path

import numpy as np

import openptv2.algorithms.track_kernels_corr as corr_mod
from openptv2.algorithms.track_kernels_position import assess_new_position_fast_nogil
from tests.unit.test_track_kernels_tracking_coverage import _make_cal_arr, _make_mmlut

NC = 4
TR_UNUSED = -1
COORD_UNUSED = -1e10


def _assess(blob_cams):
    """Run the assess kernel with one unclaimed blob at the projection in ``blob_cams``."""
    cal = _make_cal_arr(NC)
    mo, mnr, mnz, mrw = _make_mmlut(NC)
    targ_x = np.zeros((NC, 2))
    targ_y = np.zeros((NC, 2))
    targ_tnr = np.full((NC, 2), TR_UNUSED, dtype=np.int32)
    num_targets = np.zeros(NC, dtype=np.int32)
    proj_x = np.full(NC, 50.0)
    proj_y = np.full(NC, 50.0)
    for c in blob_cams:
        targ_x[c, 0], targ_y[c, 0] = 50.0, 50.0
        num_targets[c] = 1
    targ_pos = np.zeros((NC, 2))
    inds = np.full(NC, -999, dtype=np.int32)  # as allocated by the callers
    quali = assess_new_position_fast_nogil(
        np.zeros(3),
        NC,
        3.0,
        cal,
        mo,
        mnr,
        mnz,
        mrw,
        targ_x,
        targ_y,
        targ_tnr,
        num_targets,
        50.0,
        50.0,
        1.0,
        1.0,
        0,
        100,
        100,
        0.01,
        0.01,
        0.001,
        TR_UNUSED,
        COORD_UNUSED,
        proj_x,
        proj_y,
        targ_pos,
        inds,
        np.zeros(4),
    )
    return quali, inds


def test_cameras_without_a_candidate_are_marked_minus_one_not_minus_999():
    quali, inds = _assess([0, 1, 2])  # camera 3 has no blob
    assert quali == 3
    assert inds.tolist() == [0, 0, 0, TR_UNUSED]


def test_no_candidate_anywhere_gives_minus_one_in_every_camera():
    quali, inds = _assess([])
    assert quali == 0
    assert inds.tolist() == [TR_UNUSED] * NC


def test_kernel_never_guards_assessed_indexes_against_pt_unused():
    """-999 is not what the assess kernel writes; guards must be ``>= 0``."""
    src = (
        (Path(corr_mod.__file__).parent / "track_kernels_corr.py")
        .read_text()
        .splitlines()
    )
    bad = [
        (n, line.strip())
        for n, line in enumerate(src, 1)
        if "!= PT_UNUSED" in line and not line.lstrip().startswith("#")
    ]
    assert not bad, (
        f"candidate indexes must be tested with >= 0, not != PT_UNUSED: {bad}"
    )
