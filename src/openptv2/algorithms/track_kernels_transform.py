# ruff: noqa: E402,F842
"""Compiled kernels for the tracking hot path.

Auto-generated split from track_kernels.py.
"""

import cython
import numpy as np

if cython.compiled:
    pass
else:
    pass

_M_PI: cython.double = 3.141592653589793


if cython.compiled:
    from cython.cimports.openptv2.algorithms.track_kernels_pixel import (
        _dist_to_flat_out,
        _pixel_to_metric_out,
        _point_to_pixel_out,
    )
else:
    from .track_kernels_pixel import (
        _dist_to_flat_out,
        _pixel_to_metric_out,
        _point_to_pixel_out,
    )
from .track_kernels_search import (
    candsearch_in_pix_rest_fast,
)

# Sentinel values — typed C int/double
cython.declare(
    PT_UNUSED=cython.int,
    COORD_UNUSED=cython.double,
)
PT_UNUSED = -999
COORD_UNUSED = -1e10


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def assess_new_position_fast(
    pos: cython.double[:],
    num_cams: cython.int,
    add_part: cython.double,
    cal_arr: cython.double[:, ::1],
    md_arr: object,
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    targ_x: cython.double[:, ::1],
    targ_y: cython.double[:, ::1],
    targ_tnr: cython.int[:, ::1],
    num_targets,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
    tr_unused: cython.int,
    coord_unused: cython.double,
    use_proj: cython.bint,
    proj_x: cython.double[:],
    proj_y: cython.double[:],
    targ_pos_out: cython.double[:, ::1] = None,
    cand_inds_out: cython.int[:] = None,
    scratch: cython.double[:] = None,
):
    """Assess new position: project, find unused targets, undistort.

    When use_proj=True, proj_x[cam] and proj_y[cam] provide pre-computed
    pixel projections (avoids redundant _point_to_pixel_out calls).
    When use_proj=False, proj_x/proj_y are unused (can be empty arrays).

    When targ_pos_out, cand_inds_out, and scratch are provided, they are
    used as pre-allocated output buffers instead of allocating new arrays.

    Returns (targ_pos, cand_inds, valid_cams).
    """
    cam: cython.int
    valid_cams: cython.int
    best: cython.int
    count: cython.int
    has_mmlut: cython.int
    px: cython.double
    py: cython.double
    mx: cython.double
    my: cython.double
    fx: cython.double
    fy: cython.double
    _pp_mv: cython.double[:]
    targ_pos: cython.double[:, ::1]
    cand_inds: cython.int[:]
    if scratch is not None:
        _pp_mv = scratch
    else:
        _pp_mv = np.empty(2, dtype=np.float64)
    if targ_pos_out is not None:
        targ_pos = targ_pos_out
    else:
        targ_pos = np.full((num_cams, 2), coord_unused, dtype=np.float64)
    if cand_inds_out is not None:
        cand_inds = cand_inds_out
    else:
        cand_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)

    for cam in range(num_cams):
        if use_proj:
            # Use pre-computed projection (caller already projected this pos)
            px = proj_x[cam]
            py = proj_y[cam]
        else:
            has_mmlut = mnr_arr[cam] > 0
            _point_to_pixel_out(
                pos,
                cal_arr[cam],
                md_arr[cam],
                mo_arr[cam],
                mnr_arr[cam],
                mnz_arr[cam],
                mrw_arr[cam],
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            px = _pp_mv[0]
            py = _pp_mv[1]

        best, count = candsearch_in_pix_rest_fast(
            targ_x[cam],
            targ_y[cam],
            targ_tnr[cam],
            num_targets[cam],
            px,
            py,
            add_part,
            add_part,
            add_part,
            add_part,
            imx,
            imy,
            tr_unused,
        )

        if count > 0:
            cand_inds[cam] = best
            targ_pos[cam, 0] = targ_x[cam, best]
            targ_pos[cam, 1] = targ_y[cam, best]

    valid_cams = 0
    for cam in range(num_cams):
        if targ_pos[cam, 0] != coord_unused:
            _pixel_to_metric_out(
                targ_pos[cam, 0],
                targ_pos[cam, 1],
                imx,
                imy,
                pix_x,
                pix_y,
                chfield,
                _pp_mv,
            )
            mx = _pp_mv[0]
            my = _pp_mv[1]

            cal = cal_arr[cam]
            _dist_to_flat_out(
                mx,
                my,
                cal[13],
                cal[14],
                cal[24],
                cal[25],
                cal[26],
                cal[27],
                cal[28],
                cal[29],
                cal[30],
                flatten_tol,
                _pp_mv,
            )

            targ_pos[cam, 0] = _pp_mv[0]
            targ_pos[cam, 1] = _pp_mv[1]
            valid_cams += 1

    return targ_pos, cand_inds, valid_cams


POSI_K = 80
MAX_CANDS_K = 32
TR_UNUSED_K = -1
CORRES_NONE_K = -1
PREV_NONE_K = -1
NEXT_NONE_K = -2
COORD_UNUSED_K = -1e10
ADD_PART_K = 3.0

