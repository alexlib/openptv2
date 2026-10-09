"""A particle created by the tracker is marked prio = 2.

3DPTV (Tcl/Tk track.c) and liboptv mark a particle the tracker creates from
unused 2D blobs with prio = 2 ("added"); particles read from the
correspondences keep prio = 4. The added.* files carry this column, so it is
how a user tells measured from added points. openptv2's kernels wrote 4 for
both.

Scenes use the four test_cavity cameras and blobs placed exactly at the
projections of the expected positions:

* frame-3 add: a good frame-2 candidate, and free blobs where the look-ahead
  predicts frame 3;
* frame-2 add: the only frame-2 candidate fails the acceleration gate, and
  free blobs sit at the prediction, so the particle is created there.
"""

import numpy as np

from openptv2.algorithms.track_kernels_corr import trackcorr_loop_fast
from openptv2.algorithms.track_kernels_geom import point_to_pixel_fast
from tests.unit.test_trackback_add import CAP, _cams, _frame, _run

PRIO_ADDED = 2


def _project(cams, x, ci):
    cpar, cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr, half = cams
    return point_to_pixel_fast(
        np.asarray(x, dtype=np.float64),
        cal_arr[ci],
        md_arr[ci],
        mo_arr[ci],
        mnr_arr[ci],
        mnz_arr[ci],
        mrw_arr[ci],
        int(mnr_arr[ci] > 0),
        *half,
        cpar.chfield,
    )


def _put_targets(frame, cams, blobs):
    """blobs: list of (x, tnr). One target per blob per camera, y-sorted."""
    nc = cams[0].num_cams
    for ci in range(nc):
        rows = sorted(
            ((*_project(cams, x, ci), tnr, k) for k, (x, tnr) in enumerate(blobs)),
            key=lambda r: r[1],
        )
        for j, (px, py, tnr, k) in enumerate(rows):
            frame["targ_x"][ci, j], frame["targ_y"][ci, j] = px, py
            frame["targ_tnr"][ci, j] = tnr
            if tnr >= 0:
                frame["corres_p"][tnr, ci] = j
        frame["num_targets"][ci] = len(blobs)


def _forward(f0, f1, f2, f3, cams, dacc):
    cpar, cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr, half = cams
    nc = cpar.num_cams
    targ_sumg = np.zeros((nc, CAP))
    trackcorr_loop_fast(
        1,
        f0["path_x"],
        f1["path_x"],
        f1["path_prev"],
        f1["path_next"],
        f1["path_inlist"],
        f1["path_finaldecis"],
        f1["path_decis"],
        f1["path_linkdecis"],
        f1["corres_p"],
        f1["targ_x"],
        f1["targ_y"],
        f1["targ_tnr"],
        targ_sumg,
        f2["path_x"],
        f2["path_prev"],
        f2["path_next"],
        f2["path_inlist"],
        f2["path_prio"],
        f2["path_finaldecis"],
        f2["path_decis"],
        f2["path_linkdecis"],
        f2["corres_p"],
        f2["corres_nr"],
        f2["targ_x"],
        f2["targ_y"],
        f2["targ_tnr"],
        targ_sumg,
        np.full(CAP, -1.0),
        f2["num_targets"],
        f2["num_parts"],
        f3["path_x"],
        f3["path_prev"],
        f3["path_next"],
        f3["path_inlist"],
        f3["path_prio"],
        f3["path_finaldecis"],
        f3["path_decis"],
        f3["path_linkdecis"],
        f3["corres_p"],
        f3["corres_nr"],
        f3["targ_x"],
        f3["targ_y"],
        f3["targ_tnr"],
        f3["num_targets"],
        f3["num_parts"],
        cal_arr,
        md_arr,
        mo_arr,
        mnr_arr,
        mnz_arr,
        mrw_arr,
        -5.0,
        5.0,
        -5.0,
        5.0,
        -5.0,
        5.0,  # dv bounds
        dacc,
        30.0,  # dacc, dangle
        1,  # add
        np.sqrt(3 * 10.0**2),  # lmax
        -100.0,
        100.0,
        -100.0,
        100.0,
        -100.0,
        100.0,  # volume
        nc,
        *half,
        cpar.chfield,
        float(cpar.imx),
        float(cpar.imy),
        cpar.pix_x,
        cpar.pix_y,
        1e-5,
        1,  # num_threads
        1,  # loser_retry
        1,  # cold_start_neighbour
        0.0,  # app_weight
        np.ones(CAP),  # gate_scale
        0,  # use_grid
    )


def _scene(cams):
    nc = cams[0].num_cams
    f0, f1, f2, f3 = (_frame(nc) for _ in range(4))
    x0, x1 = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    f0["path_x"][0], f1["path_x"][0] = x0, x1
    f1["path_prev"][0] = 0
    return f0, f1, f2, f3, x1, 2 * x1 - x0


def test_forward_add_in_frame3_is_marked_added():
    cams = _cams()
    f0, f1, f2, f3, x1, x2 = _scene(cams)
    f2["path_x"][0] = x2  # a perfect frame-2 candidate
    f2["num_parts"][0] = 1
    _put_targets(f2, cams, [(x2, 0)])
    x5 = 2 * x2 - x1  # look-ahead prediction for constant velocity
    _put_targets(f3, cams, [(x5, -1)])  # free blobs only
    _forward(f0, f1, f2, f3, cams, dacc=1.0)
    assert f3["num_parts"][0] == 1, "no particle added in frame 3"
    assert f3["path_prio"][0] == PRIO_ADDED


def test_forward_add_in_frame2_is_marked_added():
    cams = _cams()
    f0, f1, f2, f3, x1, x2 = _scene(cams)
    p = x2 + [0.8, 0.0, 0.0]  # candidate that fails dacc = 0.5
    f2["path_x"][0] = p
    f2["num_parts"][0] = 1
    _put_targets(f2, cams, [(p, 0), (x2, -1)])  # free blobs at the prediction
    for ci in range(cams[0].num_cams):  # the two blobs are told apart
        assert np.hypot(*np.subtract(_project(cams, p, ci), _project(cams, x2, ci))) > 3
    _forward(f0, f1, f2, f3, cams, dacc=0.5)
    assert f2["num_parts"][0] == 2, "no particle added in frame 2"
    assert f2["path_prio"][1] == PRIO_ADDED
    assert f2["path_prio"][0] != PRIO_ADDED  # the candidate itself is untouched


def test_backward_add_is_marked_added():
    f2 = _run((0, 1, 2, 3))
    assert f2["num_parts"][0] == 1
    assert f2["path_prio"][0] == PRIO_ADDED
