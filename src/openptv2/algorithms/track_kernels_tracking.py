"""Compiled kernels for the tracking hot path.

Auto-generated split from track_kernels.py.
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        sqrt as c_sqrt,
        sin as c_sin,
        cos as c_cos,
        tan as c_tan,
        asin as c_asin,
        acos as c_acos,
        atan as c_atan,
    )
else:
    from math import (
        sqrt as c_sqrt,
        sin as c_sin,
        cos as c_cos,
        tan as c_tan,
        asin as c_asin,
        acos as c_acos,
        atan as c_atan,
    )

_M_PI: cython.double = 3.141592653589793


from .track_kernels_geom import (
    _angle_acc_out,
    _point_to_pixel_out,
)
from .track_kernels_search import (
    _sorted_candidates_fast_out,
)
from .track_kernels_transform import (
    assess_new_position_fast,
    point_position_fast,
)

# Constants for tracking kernels — typed C int/double via cython.declare()
# to avoid Python int boxing in every comparison inside the hot particle loop.
cython.declare(
    PT_UNUSED=cython.int,
    POSI_K=cython.int,
    MAX_CANDS_K=cython.int,
    TR_UNUSED_K=cython.int,
    CORRES_NONE_K=cython.int,
    PREV_NONE_K=cython.int,
    NEXT_NONE_K=cython.int,
    COORD_UNUSED_K=cython.double,
    ADD_PART_K=cython.double,
)
PT_UNUSED = -999
POSI_K = 80
MAX_CANDS_K = 4
TR_UNUSED_K = -1
CORRES_NONE_K = -1
PREV_NONE_K = -1
NEXT_NONE_K = -2
COORD_UNUSED_K = -1e10
ADD_PART_K = 3.0


def trackcorr_loop_fast(
    orig_parts_1: cython.int,
    # Frame 0 (prev — read only)
    path_x_0: cython.double[:, :],
    # Frame 1 (curr — read/write)
    path_x_1: cython.double[:, :],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:],
    path_decis_1: cython.double[:, :],
    path_linkdecis_1: cython.int[:, :],
    corres_p_1: cython.int[:, :],
    targ_x_1: object,
    targ_y_1: object,
    # Frame 2 (next — read/write)
    path_x_2: cython.double[:, :],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:],
    path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, :],
    path_linkdecis_2: cython.int[:, :],
    corres_p_2: cython.int[:, :],
    corres_nr_2: cython.int[:],
    targ_x_2: object,
    targ_y_2: object,
    targ_tnr_2: object,
    num_targets_2: cython.int[:],
    num_parts_2: cython.int[:],
    # Frame 3 (next-next — read/write)
    path_x_3: cython.double[:, :],
    path_prev_3: cython.int[:],
    path_next_3: cython.int[:],
    path_inlist_3: cython.int[:],
    path_prio_3: cython.int[:],
    path_finaldecis_3: cython.double[:],
    path_decis_3: cython.double[:, :],
    path_linkdecis_3: cython.int[:, :],
    corres_p_3: cython.int[:, :],
    corres_nr_3: cython.int[:],
    targ_x_3: object,
    targ_y_3: object,
    targ_tnr_3: object,
    num_targets_3: cython.int[:],
    num_parts_3: cython.int[:],
    # Calibration
    cal_t: tuple,
    md_t: tuple,
    mo_t: tuple,
    mnr_t: tuple,
    mnz_t: tuple,
    mrw_t: tuple,
    # Tracking params
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    dacc: cython.double,
    dangle: cython.double,
    add_flag: cython.int,
    lmax: cython.double,
    # Volume bounds
    X_lay_0: cython.double,
    X_lay_1: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    Zmin_lay_0: cython.double,
    Zmax_lay_1: cython.double,
    # Pixel params
    num_cams: cython.int,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
):
    """Full per-particle tracking loop + link resolution — single compiled entry.

    All internal calls (sorted_candidates, angle_acc, assess_new_position,
    point_position) are compiled with zero dispatch overhead.

    Args:
        num_parts_2, num_parts_3: (1,) int32 arrays — mutable particle counts.

    Returns:
        (count1, num_added) — number of links established and particles added.
    """
    count1: cython.int
    num_added: cython.int
    n_sc: cython.int
    h: cython.int
    j: cython.int
    mm: cython.int
    kk: cython.int
    prev_h: cython.int
    ftnr_mm: cython.int
    ftnr_kk: cython.int
    ki: cython.int
    ci: cython.int
    inlist: cython.int
    np2: cython.int
    np3: cython.int
    in_volume: cython.int
    quali: cython.int
    i: cython.int
    ti: cython.int
    cand: cython.int
    has_mmlut: cython.int
    px: cython.double
    py: cython.double
    dp0: cython.double
    dp1: cython.double
    dp2: cython.double
    angle1: cython.double
    acc1: cython.double
    angle0: cython.double
    acc0: cython.double
    acc: cython.double
    angle: cython.double
    rr: cython.double
    d13: cython.double = 0.0
    d43: cython.double = 0.0
    dl: cython.double = 0.0
    d01: cython.double = 0.0
    quali_f: cython.int
    count1 = 0
    num_added = 0
    n_sc = num_cams * MAX_CANDS_K

    # Pre-allocated output buffers for _sorted_candidates_fast_out
    _n_ftnr1 = np.empty(n_sc, dtype=np.int32)
    _n_freq1 = np.empty(n_sc, dtype=np.int32)
    _n_wc1 = np.empty((n_sc, num_cams), dtype=np.int32)
    _n_ftnr2 = np.empty(n_sc, dtype=np.int32)
    _n_freq2 = np.empty(n_sc, dtype=np.int32)
    _n_wc2 = np.empty((n_sc, num_cams), dtype=np.int32)
    _ftnr_buf1: cython.int[:] = _n_ftnr1
    _freq_buf1: cython.int[:] = _n_freq1
    _wc_buf1: cython.int[:, :] = _n_wc1
    _ftnr_buf2: cython.int[:] = _n_ftnr2
    _freq_buf2: cython.int[:] = _n_freq2
    _wc_buf2: cython.int[:, :] = _n_wc2

    _cpx = np.empty(num_cams, dtype=np.float64)
    _cpy = np.empty(num_cams, dtype=np.float64)
    _x2_cpx_save = np.empty(num_cams, dtype=np.float64)  # saved X[2] projection
    _x2_cpy_save = np.empty(num_cams, dtype=np.float64)
    _X = np.zeros((6, 3), dtype=np.float64)
    cpx: cython.double[:] = _cpx
    cpy: cython.double[:] = _cpy
    x2_cpx: cython.double[:] = _x2_cpx_save
    x2_cpy: cython.double[:] = _x2_cpy_save
    X: cython.double[:, :] = _X
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp

    # Pre-allocated output buffers for assess_new_position_fast
    _assess_targ = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)
    _assess_pp = np.empty(2, dtype=np.float64)
    _assess_targ2 = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds2 = np.full(num_cams, PT_UNUSED, dtype=np.int32)

    # Convert calibration tuples to flat arrays for C-speed access
    nc_local = len(cal_t)
    cal_arr = np.asarray(list(cal_t), dtype=np.float64)
    md_arr = list(md_t)
    mo_arr = np.asarray(list(mo_t), dtype=np.float64)
    mnr_arr = np.array(list(mnr_t), dtype=np.int32)
    mnz_arr = np.array(list(mnz_t), dtype=np.int32)
    mrw_arr = np.array(list(mrw_t), dtype=np.float64)

    for h in range(orig_parts_1):
        path_inlist_1[h] = 0

        X[1, 0] = path_x_1[h, 0]
        X[1, 1] = path_x_1[h, 1]
        X[1, 2] = path_x_1[h, 2]

        prev_h = path_prev_1[h]

        if prev_h >= 0:
            X[0, 0] = path_x_0[prev_h, 0]
            X[0, 1] = path_x_0[prev_h, 1]
            X[0, 2] = path_x_0[prev_h, 2]
            X[2, 0] = 2.0 * X[1, 0] - X[0, 0]
            X[2, 1] = 2.0 * X[1, 1] - X[0, 1]
            X[2, 2] = 2.0 * X[1, 2] - X[0, 2]

            for j in range(num_cams):
                has_mmlut = mnr_arr[j] > 0
                _point_to_pixel_out(
                    X[2],
                    cal_arr[j],
                    md_arr[j],
                    mo_arr[j],
                    mnr_arr[j],
                    mnz_arr[j],
                    mrw_arr[j],
                    has_mmlut,
                    imx_half,
                    imy_half,
                    inv_pix_x,
                    inv_pix_y,
                    chfield,
                    _pp_mv,
                )
                cpx[j] = _pp_mv[0]
                cpy[j] = _pp_mv[1]
        else:
            X[2, 0] = X[1, 0]
            X[2, 1] = X[1, 1]
            X[2, 2] = X[1, 2]

            for j in range(num_cams):
                if corres_p_1[h, j] == CORRES_NONE_K:
                    has_mmlut = mnr_arr[j] > 0
                    _point_to_pixel_out(
                        X[2],
                        cal_arr[j],
                        md_arr[j],
                        mo_arr[j],
                        mnr_arr[j],
                        mnz_arr[j],
                        mrw_arr[j],
                        has_mmlut,
                        imx_half,
                        imy_half,
                        inv_pix_x,
                        inv_pix_y,
                        chfield,
                        _pp_mv,
                    )
                    cpx[j] = _pp_mv[0]
                    cpy[j] = _pp_mv[1]
                else:
                    _ix = corres_p_1[h, j]
                    cpx[j] = targ_x_1[j][_ix]
                    cpy[j] = targ_y_1[j][_ix]

        # Save X[2] projections for later use by assess_new_position_fast
        for j in range(num_cams):
            x2_cpx[j] = cpx[j]
            x2_cpy[j] = cpy[j]

        # --- sorted_candidates for frame 2 ---
        w_nc = _sorted_candidates_fast_out(
            X[2],
            cpx,
            cpy,
            num_cams,
            MAX_CANDS_K,
            cal_t,
            md_t,
            mo_t,
            mnr_t,
            mnz_t,
            mrw_t,
            targ_x_2,
            targ_y_2,
            targ_tnr_2,
            num_targets_2,
            dvxmin,
            dvxmax,
            dvymin,
            dvymax,
            dvzmin,
            dvzmax,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            imx,
            imy,
            TR_UNUSED_K,
            _ftnr_buf1,
            _freq_buf1,
            _wc_buf1,
        )

        if w_nc == 0:
            continue

        mm = 0
        while mm < w_nc:
            ftnr_mm = _ftnr_buf1[mm]
            X[3, 0] = path_x_2[ftnr_mm, 0]
            X[3, 1] = path_x_2[ftnr_mm, 1]
            X[3, 2] = path_x_2[ftnr_mm, 2]

            if prev_h >= 0:
                for j in range(3):
                    X[5, j] = 0.5 * (5.0 * X[3, j] - 4.0 * X[1, j] + X[0, j])
            else:
                X[5, 0] = 2.0 * X[3, 0] - X[1, 0]
                X[5, 1] = 2.0 * X[3, 1] - X[1, 1]
                X[5, 2] = 2.0 * X[3, 2] - X[1, 2]

            for j in range(num_cams):
                has_mmlut = mnr_arr[j] > 0
                _point_to_pixel_out(
                    X[5],
                    cal_arr[j],
                    md_arr[j],
                    mo_arr[j],
                    mnr_arr[j],
                    mnz_arr[j],
                    mrw_arr[j],
                    has_mmlut,
                    imx_half,
                    imy_half,
                    inv_pix_x,
                    inv_pix_y,
                    chfield,
                    _pp_mv,
                )
                cpx[j] = _pp_mv[0]
                cpy[j] = _pp_mv[1]

            # --- sorted_candidates for frame 3 ---
            wn_nc = _sorted_candidates_fast_out(
                X[5],
                cpx,
                cpy,
                num_cams,
                MAX_CANDS_K,
                cal_t,
                md_t,
                mo_t,
                mnr_t,
                mnz_t,
                mrw_t,
                targ_x_3,
                targ_y_3,
                targ_tnr_3,
                num_targets_3,
                dvxmin,
                dvxmax,
                dvymin,
                dvymax,
                dvzmin,
                dvzmax,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                imx,
                imy,
                TR_UNUSED_K,
                _ftnr_buf2,
                _freq_buf2,
                _wc_buf2,
            )

            if wn_nc > 0:
                kk = 0
                while kk < wn_nc:
                    ftnr_kk = _ftnr_buf2[kk]
                    X[4, 0] = path_x_3[ftnr_kk, 0]
                    X[4, 1] = path_x_3[ftnr_kk, 1]
                    X[4, 2] = path_x_3[ftnr_kk, 2]

                    dp0 = X[4, 0] - X[3, 0]
                    dp1 = X[4, 1] - X[3, 1]
                    dp2 = X[4, 2] - X[3, 2]

                    if (
                        dvxmin < dp0 < dvxmax
                        and dvymin < dp1 < dvymax
                        and dvzmin < dp2 < dvzmax
                    ):
                        _angle_acc_out(
                            X[3, 0],
                            X[3, 1],
                            X[3, 2],
                            X[4, 0],
                            X[4, 1],
                            X[4, 2],
                            X[5, 0],
                            X[5, 1],
                            X[5, 2],
                            _pp_mv,
                        )
                        angle1 = _pp_mv[0]
                        acc1 = _pp_mv[1]
                        if prev_h >= 0:
                            _angle_acc_out(
                                X[1, 0],
                                X[1, 1],
                                X[1, 2],
                                X[2, 0],
                                X[2, 1],
                                X[2, 2],
                                X[3, 0],
                                X[3, 1],
                                X[3, 2],
                                _pp_mv,
                            )
                            angle0 = _pp_mv[0]
                            acc0 = _pp_mv[1]
                        else:
                            acc0 = acc1
                            angle0 = angle1

                        acc = (acc0 + acc1) * 0.5
                        angle = (angle0 + angle1) * 0.5
                        quali = _freq_buf2[kk] + _freq_buf1[mm]

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = c_sqrt(
                                (X[1, 0] - X[3, 0]) ** 2
                                + (X[1, 1] - X[3, 1]) ** 2
                                + (X[1, 2] - X[3, 2]) ** 2
                            )
                            d43 = c_sqrt(
                                (X[4, 0] - X[3, 0]) ** 2
                                + (X[4, 1] - X[3, 1]) ** 2
                                + (X[4, 2] - X[3, 2]) ** 2
                            )
                            dl = (d13 + d43) * 0.5
                            rr = (dl / lmax + acc / dacc + angle / dangle) / quali

                            inlist = path_inlist_1[h]
                            if inlist < POSI_K:
                                path_decis_1[h, inlist] = rr
                                path_linkdecis_1[h, inlist] = ftnr_mm
                                path_inlist_1[h] = inlist + 1

                    kk += 1

            # --- assess_new_position for X[5] in frame 3 ---
            # Use cached pixel projection (cpx/cpy just computed from X[5] above)
            targ_pos, cand_inds, quali = assess_new_position_fast(
                X[5],
                num_cams,
                ADD_PART_K,
                cal_t,
                md_t,
                mo_t,
                mnr_t,
                mnz_t,
                mrw_t,
                targ_x_3,
                targ_y_3,
                targ_tnr_3,
                num_targets_3,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                int(imx),
                int(imy),
                pix_x,
                pix_y,
                flatten_tol,
                TR_UNUSED_K,
                COORD_UNUSED_K,
                use_proj=True,
                proj_x=cpx,
                proj_y=cpy,
                targ_pos_out=_assess_targ,
                cand_inds_out=_assess_inds,
                scratch=_assess_pp,
            )

            if quali >= 2:
                in_volume = 0
                pos_new, dl_pp = point_position_fast(targ_pos, num_cams, cal_t)
                X[4, 0] = pos_new[0]
                X[4, 1] = pos_new[1]
                X[4, 2] = pos_new[2]

                if (
                    X_lay_0 < X[4, 0] < X_lay_1
                    and ymin < X[4, 1] < ymax
                    and Zmin_lay_0 < X[4, 2] < Zmax_lay_1
                ):
                    in_volume = 1

                dp0 = X[3, 0] - X[4, 0]
                dp1 = X[3, 1] - X[4, 1]
                dp2 = X[3, 2] - X[4, 2]

                if (
                    in_volume == 1
                    and dvxmin < dp0 < dvxmax
                    and dvymin < dp1 < dvymax
                    and dvzmin < dp2 < dvzmax
                ):
                    _angle_acc_out(
                        X[3, 0],
                        X[3, 1],
                        X[3, 2],
                        X[4, 0],
                        X[4, 1],
                        X[4, 2],
                        X[5, 0],
                        X[5, 1],
                        X[5, 2],
                        _pp_mv,
                    )
                    angle = _pp_mv[0]
                    acc = _pp_mv[1]

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = c_sqrt(
                            (X[1, 0] - X[3, 0]) ** 2
                            + (X[1, 1] - X[3, 1]) ** 2
                            + (X[1, 2] - X[3, 2]) ** 2
                        )
                        d43 = c_sqrt(
                            (X[4, 0] - X[3, 0]) ** 2
                            + (X[4, 1] - X[3, 1]) ** 2
                            + (X[4, 2] - X[3, 2]) ** 2
                        )
                        dl = (d13 + d43) * 0.5
                        rr = (dl / lmax + acc / dacc + angle / dangle) / (
                            quali + _freq_buf1[mm]
                        )

                        inlist = path_inlist_1[h]
                        if inlist < POSI_K:
                            path_decis_1[h, inlist] = rr
                            path_linkdecis_1[h, inlist] = ftnr_mm
                            path_inlist_1[h] = inlist + 1

                        if add_flag:
                            np3 = num_parts_3[0]
                            path_x_3[np3, 0] = X[4, 0]
                            path_x_3[np3, 1] = X[4, 1]
                            path_x_3[np3, 2] = X[4, 2]
                            path_prev_3[np3] = PREV_NONE_K
                            path_next_3[np3] = NEXT_NONE_K
                            path_inlist_3[np3] = 0
                            path_prio_3[np3] = 4
                            path_finaldecis_3[np3] = 1000000.0
                            for ki in range(POSI_K):
                                path_decis_3[np3, ki] = 0.0
                                path_linkdecis_3[np3, ki] = PT_UNUSED
                            for ci in range(num_cams):
                                corres_p_3[np3, ci] = CORRES_NONE_K
                            corres_nr_3[np3] = np3
                            for ci in range(num_cams):
                                if cand_inds[ci] != PT_UNUSED:
                                    idx = cand_inds[ci]
                                    targ_tnr_3[ci][idx] = np3
                                    corres_p_3[np3, ci] = idx
                            num_parts_3[0] = np3 + 1
                            num_added += 1

                in_volume = 0
            quali = 0

            # --- fallback: direct link if no links and prev >= 0 ---
            if path_inlist_1[h] == 0 and prev_h >= 0:
                dp0 = X[3, 0] - X[1, 0]
                dp1 = X[3, 1] - X[1, 1]
                dp2 = X[3, 2] - X[1, 2]

                if (
                    dvxmin < dp0 < dvxmax
                    and dvymin < dp1 < dvymax
                    and dvzmin < dp2 < dvzmax
                ):
                    _angle_acc_out(
                        X[1, 0],
                        X[1, 1],
                        X[1, 2],
                        X[2, 0],
                        X[2, 1],
                        X[2, 2],
                        X[3, 0],
                        X[3, 1],
                        X[3, 2],
                        _pp_mv,
                    )
                    angle = _pp_mv[0]
                    acc = _pp_mv[1]

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        quali_f = _freq_buf1[mm]
                        d13 = c_sqrt(
                            (X[1, 0] - X[3, 0]) ** 2
                            + (X[1, 1] - X[3, 1]) ** 2
                            + (X[1, 2] - X[3, 2]) ** 2
                        )
                        d01 = c_sqrt(
                            (X[0, 0] - X[1, 0]) ** 2
                            + (X[0, 1] - X[1, 1]) ** 2
                            + (X[0, 2] - X[1, 2]) ** 2
                        )
                        dl = (d13 + d01) * 0.5
                        rr = (dl / lmax + acc / dacc + angle / dangle) / quali_f

                        inlist = path_inlist_1[h]
                        if inlist < POSI_K:
                            path_decis_1[h, inlist] = rr
                            path_linkdecis_1[h, inlist] = ftnr_mm
                            path_inlist_1[h] = inlist + 1

            mm += 1

        # --- add_particle to frame 2 if no links found ---
        if add_flag:
            if path_inlist_1[h] == 0 and prev_h >= 0:
                # Use cached X[2] projection (saved before the inner candidate loop)
                targ_pos2, cand_inds2, quali2 = assess_new_position_fast(
                    X[2],
                    num_cams,
                    ADD_PART_K,
                    cal_t,
                    md_t,
                    mo_t,
                    mnr_t,
                    mnz_t,
                    mrw_t,
                    targ_x_2,
                    targ_y_2,
                    targ_tnr_2,
                    num_targets_2,
                    imx_half,
                    imy_half,
                    inv_pix_x,
                    inv_pix_y,
                    chfield,
                    int(imx),
                    int(imy),
                    pix_x,
                    pix_y,
                    flatten_tol,
                    TR_UNUSED_K,
                    COORD_UNUSED_K,
                    use_proj=True,
                    proj_x=x2_cpx,
                    proj_y=x2_cpy,
                    targ_pos_out=_assess_targ2,
                    cand_inds_out=_assess_inds2,
                    scratch=_assess_pp,
                )

                if quali2 >= 2:
                    in_volume = 0
                    pos_new2, dl_pp2 = point_position_fast(targ_pos2, num_cams, cal_t)
                    X[3, 0] = pos_new2[0]
                    X[3, 1] = pos_new2[1]
                    X[3, 2] = pos_new2[2]

                    if (
                        X_lay_0 < X[3, 0] < X_lay_1
                        and ymin < X[3, 1] < ymax
                        and Zmin_lay_0 < X[3, 2] < Zmax_lay_1
                    ):
                        in_volume = 1

                    dp0 = X[2, 0] - X[3, 0]
                    dp1 = X[2, 1] - X[3, 1]
                    dp2 = X[2, 2] - X[3, 2]

                    if (
                        in_volume == 1
                        and dvxmin < dp0 < dvxmax
                        and dvymin < dp1 < dvymax
                        and dvzmin < dp2 < dvzmax
                    ):
                        _angle_acc_out(
                            X[1, 0],
                            X[1, 1],
                            X[1, 2],
                            X[2, 0],
                            X[2, 1],
                            X[2, 2],
                            X[3, 0],
                            X[3, 1],
                            X[3, 2],
                            _pp_mv,
                        )
                        angle = _pp_mv[0]
                        acc = _pp_mv[1]

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = c_sqrt(
                                (X[1, 0] - X[3, 0]) ** 2
                                + (X[1, 1] - X[3, 1]) ** 2
                                + (X[1, 2] - X[3, 2]) ** 2
                            )
                            d01 = c_sqrt(
                                (X[0, 0] - X[1, 0]) ** 2
                                + (X[0, 1] - X[1, 1]) ** 2
                                + (X[0, 2] - X[1, 2]) ** 2
                            )
                            dl = (d13 + d01) * 0.5
                            rr = (dl / lmax + acc / dacc + angle / dangle) / quali2

                            np2 = num_parts_2[0]
                            inlist = path_inlist_1[h]
                            if inlist < POSI_K:
                                path_decis_1[h, inlist] = rr
                                path_linkdecis_1[h, inlist] = np2
                                path_inlist_1[h] = inlist + 1

                            path_x_2[np2, 0] = X[3, 0]
                            path_x_2[np2, 1] = X[3, 1]
                            path_x_2[np2, 2] = X[3, 2]
                            path_prev_2[np2] = PREV_NONE_K
                            path_next_2[np2] = NEXT_NONE_K
                            path_inlist_2[np2] = 0
                            path_prio_2[np2] = 4
                            path_finaldecis_2[np2] = 1000000.0
                            for ki in range(POSI_K):
                                path_decis_2[np2, ki] = 0.0
                                path_linkdecis_2[np2, ki] = PT_UNUSED
                            for ci in range(num_cams):
                                corres_p_2[np2, ci] = CORRES_NONE_K
                            corres_nr_2[np2] = np2
                            for ci in range(num_cams):
                                if cand_inds2[ci] != PT_UNUSED:
                                    idx = cand_inds2[ci]
                                    targ_tnr_2[ci][idx] = np2
                                    corres_p_2[np2, ci] = idx
                            num_parts_2[0] = np2 + 1
                            num_added += 1

                    in_volume = 0

    # ========== LINK RESOLUTION ==========
    # Phase 1: Sort decis/linkdecis, set finaldecis and next
    for h in range(orig_parts_1):
        inlist = path_inlist_1[h]
        if inlist > 0:
            flag = True
            while flag:
                flag = False
                for i in range(inlist - 1):
                    if path_decis_1[h, i] > path_decis_1[h, i + 1]:
                        path_decis_1[h, i], path_decis_1[h, i + 1] = (
                            path_decis_1[h, i + 1],
                            path_decis_1[h, i],
                        )
                        path_linkdecis_1[h, i], path_linkdecis_1[h, i + 1] = (
                            path_linkdecis_1[h, i + 1],
                            path_linkdecis_1[h, i],
                        )
                        flag = True
            path_finaldecis_1[h] = path_decis_1[h, 0]
            path_next_1[h] = path_linkdecis_1[h, 0]

    # Phase 2: Resolve conflicts (original single-pass)
    for h in range(orig_parts_1):
        if path_inlist_1[h] > 0:
            next_h = path_next_1[h]
            if path_prev_2[next_h] == PREV_NONE_K:
                path_prev_2[next_h] = h
            else:
                prev_of_next = path_prev_2[next_h]
                if path_finaldecis_1[prev_of_next] > path_finaldecis_1[h]:
                    path_next_1[prev_of_next] = NEXT_NONE_K
                    path_prev_2[next_h] = h
                else:
                    path_next_1[h] = NEXT_NONE_K

    # Phase 3: Losers retry with fallback candidates (claim unclaimed only)
    for h in range(orig_parts_1):
        if path_inlist_1[h] > 1 and path_next_1[h] == NEXT_NONE_K:
            for ti in range(1, path_inlist_1[h]):
                cand = path_linkdecis_1[h, ti]
                if path_prev_2[cand] == PREV_NONE_K:
                    path_next_1[h] = cand
                    path_finaldecis_1[h] = path_decis_1[h, ti]
                    path_prev_2[cand] = h
                    break

    for h in range(orig_parts_1):
        if path_next_1[h] != NEXT_NONE_K:
            count1 += 1

    return count1, num_added


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def trackback_loop_fast(
    num_parts_1: cython.int,
    # Frame 0 (forward/next in time — read only)
    path_x_0: cython.double[:, :],
    # Frame 1 (current — read/write)
    path_x_1: cython.double[:, :],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:],
    path_decis_1: cython.double[:, :],
    path_linkdecis_1: cython.int[:, :],
    # Frame 2 (backward/prev in time — read/write)
    path_x_2: cython.double[:, :],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int[:],
    targ_x_2: object,
    targ_y_2: object,
    targ_tnr_2: object,
    num_targets_2: cython.int[:],
    corres_p_2: cython.int[:, :],
    corres_nr_2: cython.int[:],
    path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:],
    path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, :],
    path_linkdecis_2: cython.int[:, :],
    # Frame 3 (further backward — read only, for extra angle check)
    path_x_3: cython.double[:, :],
    path_prev_3: cython.int[:],
    # Calibration
    cal_t: tuple,
    md_t: tuple,
    mo_t: tuple,
    mnr_t: tuple,
    mnz_t: tuple,
    mrw_t: tuple,
    # Tracking params
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    dacc: cython.double,
    dangle: cython.double,
    add_flag: cython.int,
    lmax: cython.double,
    # Volume bounds
    X_lay_0: cython.double,
    X_lay_1: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    Zmin_lay_0: cython.double,
    Zmax_lay_1: cython.double,
    # Pixel params
    num_cams: cython.int,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    pix_x: cython.double,
    pix_y: cython.double,
    flatten_tol: cython.double,
):
    """Backward tracking loop — compiled compiled.

    For each particle in buf[1] with next >= 0 and prev == -1,
    searches for candidates in buf[2] (backward in time).
    """
    count1: cython.int
    num_added: cython.int
    h: cython.int
    i: cython.int
    j: cython.int
    ki: cython.int
    ci: cython.int
    next_h: cython.int
    prev_h: cython.int
    ftnr_i: cython.int
    inlist: cython.int
    best_cand: cython.int
    has_mmlut: cython.int
    prev_of_cand: cython.int
    np2: cython.int
    in_volume: cython.int
    quali: cython.int
    ti: cython.int
    px: cython.double
    py: cython.double
    dp0: cython.double
    dp1: cython.double
    dp2: cython.double
    angle: cython.double
    acc: cython.double
    rr: cython.double
    d13: cython.double = 0.0
    d01: cython.double = 0.0
    dl: cython.double = 0.0
    idx: cython.int
    flag: cython.bint
    count1 = 0
    num_added = 0
    n_sc = num_cams * MAX_CANDS_K
    _n_ftnr = np.empty(n_sc, dtype=np.int32)
    _n_freq = np.empty(n_sc, dtype=np.int32)
    _n_wc = np.empty((n_sc, num_cams), dtype=np.int32)
    _ftnr_buf: cython.int[:] = _n_ftnr
    _freq_buf: cython.int[:] = _n_freq
    _wc_buf: cython.int[:, :] = _n_wc
    _cpx = np.empty(num_cams, dtype=np.float64)
    _cpy = np.empty(num_cams, dtype=np.float64)
    _X = np.zeros((6, 3), dtype=np.float64)
    cpx: cython.double[:] = _cpx
    cpy: cython.double[:] = _cpy
    X: cython.double[:, :] = _X
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp

    # Pre-allocated output buffers for assess_new_position_fast
    _assess_targ = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)
    _assess_pp = np.empty(2, dtype=np.float64)

    # Convert calibration tuples to flat arrays for C-speed access
    cal_arr = np.asarray(list(cal_t), dtype=np.float64)
    md_arr = list(md_t)
    mo_arr = np.asarray(list(mo_t), dtype=np.float64)
    mnr_arr = np.array(list(mnr_t), dtype=np.int32)
    mnz_arr = np.array(list(mnz_t), dtype=np.int32)
    mrw_arr = np.array(list(mrw_t), dtype=np.float64)

    for h in range(num_parts_1):
        next_h = path_next_1[h]
        prev_h = path_prev_1[h]

        if (next_h < 0) or (prev_h != -1):
            continue

        path_inlist_1[h] = 0

        X[1, 0] = path_x_1[h, 0]
        X[1, 1] = path_x_1[h, 1]
        X[1, 2] = path_x_1[h, 2]

        X[0, 0] = path_x_0[next_h, 0]
        X[0, 1] = path_x_0[next_h, 1]
        X[0, 2] = path_x_0[next_h, 2]

        # Predict backward: 2*curr - next (mirror of forward prediction)
        X[2, 0] = 2.0 * X[1, 0] - X[0, 0]
        X[2, 1] = 2.0 * X[1, 1] - X[0, 1]
        X[2, 2] = 2.0 * X[1, 2] - X[0, 2]

        for j in range(num_cams):
            has_mmlut = mnr_arr[j] > 0
            _point_to_pixel_out(
                X[2],
                cal_arr[j],
                md_arr[j],
                mo_arr[j],
                mnr_arr[j],
                mnz_arr[j],
                mrw_arr[j],
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            cpx[j] = _pp_mv[0]
            cpy[j] = _pp_mv[1]

        w_nc = _sorted_candidates_fast_out(
            X[2],
            cpx,
            cpy,
            num_cams,
            MAX_CANDS_K,
            cal_t,
            md_t,
            mo_t,
            mnr_t,
            mnz_t,
            mrw_t,
            targ_x_2,
            targ_y_2,
            targ_tnr_2,
            num_targets_2,
            dvxmin,
            dvxmax,
            dvymin,
            dvymax,
            dvzmin,
            dvzmax,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            imx,
            imy,
            TR_UNUSED_K,
            _ftnr_buf,
            _freq_buf,
            _wc_buf,
        )

        if w_nc > 0:
            i = 0
            while i < w_nc:
                ftnr_i = _ftnr_buf[i]
                X[3, 0] = path_x_2[ftnr_i, 0]
                X[3, 1] = path_x_2[ftnr_i, 1]
                X[3, 2] = path_x_2[ftnr_i, 2]

                dp0 = X[1, 0] - X[3, 0]
                dp1 = X[1, 1] - X[3, 1]
                dp2 = X[1, 2] - X[3, 2]

                if (
                    dvxmin < dp0 < dvxmax
                    and dvymin < dp1 < dvymax
                    and dvzmin < dp2 < dvzmax
                ):
                    _angle_acc_out(
                        X[1, 0],
                        X[1, 1],
                        X[1, 2],
                        X[2, 0],
                        X[2, 1],
                        X[2, 2],
                        X[3, 0],
                        X[3, 1],
                        X[3, 2],
                        _pp_mv,
                    )
                    angle = _pp_mv[0]
                    acc = _pp_mv[1]

                    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                        d13 = c_sqrt(
                            (X[1, 0] - X[3, 0]) ** 2
                            + (X[1, 1] - X[3, 1]) ** 2
                            + (X[1, 2] - X[3, 2]) ** 2
                        )
                    d01 = c_sqrt(
                        (X[0, 0] - X[1, 0]) ** 2
                        + (X[0, 1] - X[1, 1]) ** 2
                        + (X[0, 2] - X[1, 2]) ** 2
                    )
                    dl = (d13 + d01) * 0.5
                    quali = _freq_buf[i]
                    rr = (dl / lmax + acc / dacc + angle / dangle) / quali

                    inlist = path_inlist_1[h]
                    if inlist < POSI_K:
                        path_decis_1[h, inlist] = rr
                        path_linkdecis_1[h, inlist] = ftnr_i
                        path_inlist_1[h] = inlist + 1

                i += 1

        if add_flag:
            if path_inlist_1[h] == 0:
                targ_pos, cand_inds, quali = assess_new_position_fast(
                    X[2],
                    num_cams,
                    ADD_PART_K,
                    cal_t,
                    md_t,
                    mo_t,
                    mnr_t,
                    mnz_t,
                    mrw_t,
                    targ_x_2,
                    targ_y_2,
                    targ_tnr_2,
                    num_targets_2,
                    imx_half,
                    imy_half,
                    inv_pix_x,
                    inv_pix_y,
                    chfield,
                    int(imx),
                    int(imy),
                    pix_x,
                    pix_y,
                    flatten_tol,
                    TR_UNUSED_K,
                    COORD_UNUSED_K,
                    use_proj=False,
                    proj_x=cpx,
                    proj_y=cpy,
                    targ_pos_out=_assess_targ,
                    cand_inds_out=_assess_inds,
                    scratch=_assess_pp,
                )

                if quali >= 2:
                    in_volume = 0
                    pos_new, dl_pp = point_position_fast(targ_pos, num_cams, cal_t)
                    X[3, 0] = pos_new[0]
                    X[3, 1] = pos_new[1]
                    X[3, 2] = pos_new[2]

                    if (
                        X_lay_0 < X[3, 0] < X_lay_1
                        and ymin < X[3, 1] < ymax
                        and Zmin_lay_0 < X[3, 2] < Zmax_lay_1
                    ):
                        in_volume = 1

                    dp0 = X[1, 0] - X[3, 0]
                    dp1 = X[1, 1] - X[3, 1]
                    dp2 = X[1, 2] - X[3, 2]

                    if (
                        in_volume == 1
                        and dvxmin < dp0 < dvxmax
                        and dvymin < dp1 < dvymax
                        and dvzmin < dp2 < dvzmax
                    ):
                        _angle_acc_out(
                            X[1, 0],
                            X[1, 1],
                            X[1, 2],
                            X[2, 0],
                            X[2, 1],
                            X[2, 2],
                            X[3, 0],
                            X[3, 1],
                            X[3, 2],
                            _pp_mv,
                        )
                        angle = _pp_mv[0]
                        acc = _pp_mv[1]

                        if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                            d13 = c_sqrt(
                                (X[1, 0] - X[3, 0]) ** 2
                                + (X[1, 1] - X[3, 1]) ** 2
                                + (X[1, 2] - X[3, 2]) ** 2
                            )
                            d01 = c_sqrt(
                                (X[0, 0] - X[1, 0]) ** 2
                                + (X[0, 1] - X[1, 1]) ** 2
                                + (X[0, 2] - X[1, 2]) ** 2
                            )
                            dl = (d13 + d01) * 0.5
                            rr = (dl / lmax + acc / dacc + angle / dangle) / quali

                            np2 = num_parts_2[0]
                            inlist = path_inlist_1[h]
                            if inlist < POSI_K:
                                path_decis_1[h, inlist] = rr
                                path_linkdecis_1[h, inlist] = np2
                                path_inlist_1[h] = inlist + 1

                            path_x_2[np2, 0] = X[3, 0]
                            path_x_2[np2, 1] = X[3, 1]
                            path_x_2[np2, 2] = X[3, 2]
                            path_prev_2[np2] = PREV_NONE_K
                            path_next_2[np2] = NEXT_NONE_K
                            path_inlist_2[np2] = 0
                            path_prio_2[np2] = 4
                            path_finaldecis_2[np2] = 1000000.0
                            for ki in range(POSI_K):
                                path_decis_2[np2, ki] = 0.0
                                path_linkdecis_2[np2, ki] = PT_UNUSED
                            for ci in range(num_cams):
                                corres_p_2[np2, ci] = CORRES_NONE_K
                            corres_nr_2[np2] = np2
                            for ci in range(num_cams):
                                if cand_inds[ci] != PT_UNUSED:
                                    idx = cand_inds[ci]
                                    targ_tnr_2[ci][idx] = np2
                                    corres_p_2[np2, ci] = idx
                            num_parts_2[0] = np2 + 1
                            num_added += 1

                    in_volume = 0

    # Sort candidates
    for h in range(num_parts_1):
        inlist = path_inlist_1[h]
        if inlist > 0:
            flag = True
            while flag:
                flag = False
                for i in range(inlist - 1):
                    if path_decis_1[h, i] > path_decis_1[h, i + 1]:
                        path_decis_1[h, i], path_decis_1[h, i + 1] = (
                            path_decis_1[h, i + 1],
                            path_decis_1[h, i],
                        )
                        path_linkdecis_1[h, i], path_linkdecis_1[h, i + 1] = (
                            path_linkdecis_1[h, i + 1],
                            path_linkdecis_1[h, i],
                        )
                        flag = True

    # Link resolution — trackback style
    for h in range(num_parts_1):
        if path_inlist_1[h] > 0:
            best_cand = path_linkdecis_1[h, 0]

            # Case 1: candidate has no links at all
            if (
                path_prev_2[best_cand] == PREV_NONE_K
                and path_next_2[best_cand] == NEXT_NONE_K
            ):
                path_finaldecis_1[h] = path_decis_1[h, 0]
                path_prev_1[h] = best_cand
                path_next_2[best_cand] = h
                num_added += 1

            # Case 2: candidate has a prev but no next — extra angle check
            elif (
                path_prev_2[best_cand] != PREV_NONE_K
                and path_next_2[best_cand] == NEXT_NONE_K
            ):
                X[0, 0] = path_x_0[path_next_1[h], 0]
                X[0, 1] = path_x_0[path_next_1[h], 1]
                X[0, 2] = path_x_0[path_next_1[h], 2]
                X[1, 0] = path_x_1[h, 0]
                X[1, 1] = path_x_1[h, 1]
                X[1, 2] = path_x_1[h, 2]
                X[3, 0] = path_x_2[best_cand, 0]
                X[3, 1] = path_x_2[best_cand, 1]
                X[3, 2] = path_x_2[best_cand, 2]

                prev_of_cand = path_prev_2[best_cand]
                X[4, 0] = path_x_3[prev_of_cand, 0]
                X[4, 1] = path_x_3[prev_of_cand, 1]
                X[4, 2] = path_x_3[prev_of_cand, 2]

                for j in range(3):
                    X[5, j] = 0.5 * (5.0 * X[3, j] - 4.0 * X[1, j] + X[0, j])

                _angle_acc_out(
                    X[3, 0],
                    X[3, 1],
                    X[3, 2],
                    X[4, 0],
                    X[4, 1],
                    X[4, 2],
                    X[5, 0],
                    X[5, 1],
                    X[5, 2],
                    _pp_mv,
                )
                angle = _pp_mv[0]
                acc = _pp_mv[1]

                if (acc < dacc and angle < dangle) or acc < dacc * 0.1:
                    path_finaldecis_1[h] = path_decis_1[h, 0]
                    path_prev_1[h] = best_cand
                    path_next_2[best_cand] = h
                    num_added += 1

        if path_prev_1[h] != PREV_NONE_K:
            count1 += 1

    return count1, num_added


@cython.boundscheck(False)
@cython.wraparound(False)
def _find_closest_in_3d(
    path_x_2: cython.double[:, :],
    np2: cython.int,
    pred_x: cython.double,
    pred_y: cython.double,
    pred_z: cython.double,
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
    cand_inds: cython.int[:],
    cand_dists: cython.double[:],
):
    """Find up to max_cands closest candidates by distance within a 3D box.

    Maintains a running top-N by distance, matching candsearch_in_pix logic.
    Writes into pre-allocated cand_inds/cand_dists arrays.
    Returns the number of candidates found.
    """
    s: cython.int
    k: cython.int
    slot: cython.int
    ddx: cython.double
    ddy: cython.double
    ddz: cython.double
    d: cython.double
    n_found = 0
    for s in range(max_cands):
        cand_inds[s] = -1
        cand_dists[s] = 1e20

    for k in range(np2):
        ddx = path_x_2[k, 0] - pred_x
        ddy = path_x_2[k, 1] - pred_y
        ddz = path_x_2[k, 2] - pred_z
        if abs(ddx) < dx and abs(ddy) < dy and abs(ddz) < dz:
            d = c_sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            for slot in range(max_cands):
                if d < cand_dists[slot]:
                    for s in range(max_cands - 1, slot, -1):
                        cand_inds[s] = cand_inds[s - 1]
                        cand_dists[s] = cand_dists[s - 1]
                    cand_inds[slot] = k
                    cand_dists[slot] = d
                    break

    for s in range(max_cands):
        if cand_inds[s] >= 0:
            n_found += 1
    return n_found


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def track3d_loop_fast(
    orig_parts: cython.int,
    # Frame 0 (prev) — read only
    path_x_0: cython.double[:, :],
    path_prev_0: cython.int[:],
    num_parts_0: cython.int,
    # Frame 1 (curr) — read/write
    path_x_1: cython.double[:, :],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    num_parts_1: cython.int,
    # Frame 2 (next) — read/write
    path_x_2: cython.double[:, :],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int,
    # Tracking params
    dx: cython.double,
    dy: cython.double,
    dz: cython.double,
    max_cands: cython.int,
):
    """Full track3d loop (3 levels) — single compiled entry.

    Level 1: particles with previous links — predict from velocity.
    Level 2: no prev link — average velocity from neighbors.
    Level 3: no prev link, no neighbor info — use current position.

    Returns count1 (number of links established).
    """
    count1: cython.int
    np2: cython.int
    i: cython.int
    j: cython.int
    ci: cython.int
    prev_idx: cython.int
    pred_x: cython.double
    pred_y: cython.double
    pred_z: cython.double
    n_cands: cython.int
    n_decis: cython.int
    k: cython.int
    d0: cython.double
    d1: cython.double
    d2: cython.double
    acc: cython.double
    si: cython.int
    sj: cython.int
    vel_x: cython.double
    vel_y: cython.double
    vel_z: cython.double
    nvel: cython.int
    cx: cython.double
    cy: cython.double
    cz: cython.double
    pj: cython.int
    inv_nvel: cython.double
    count1 = 0
    np2 = num_parts_2
    _cand_inds = np.empty(max_cands, dtype=np.int32)
    _cand_dists = np.empty(max_cands, dtype=np.float64)
    _decis_vals = np.empty(max_cands, dtype=np.float64)
    _decis_inds = np.empty(max_cands, dtype=np.int32)

    cand_inds: cython.int[:] = _cand_inds
    cand_dists: cython.double[:] = _cand_dists
    decis_vals: cython.double[:] = _decis_vals
    decis_inds: cython.int[:] = _decis_inds

    # ===== Level 1: Particles with previous links =====
    for i in range(orig_parts):
        if path_prev_1[i] < 0:
            continue
        prev_idx = path_prev_1[i]
        if prev_idx < 0 or prev_idx >= num_parts_0:
            continue

        pred_x = 2.0 * path_x_1[i, 0] - path_x_0[prev_idx, 0]
        pred_y = 2.0 * path_x_1[i, 1] - path_x_0[prev_idx, 1]
        pred_z = 2.0 * path_x_1[i, 2] - path_x_0[prev_idx, 2]

        n_cands = _find_closest_in_3d(
            path_x_2,
            np2,
            pred_x,
            pred_y,
            pred_z,
            dx,
            dy,
            dz,
            max_cands,
            cand_inds,
            cand_dists,
        )
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = path_x_1[i, 0] - 2.0 * path_x_2[k, 0] + path_x_0[prev_idx, 0]
            d1 = path_x_1[i, 1] - 2.0 * path_x_2[k, 1] + path_x_0[prev_idx, 1]
            d2 = path_x_1[i, 2] - 2.0 * path_x_2[k, 2] + path_x_0[prev_idx, 2]
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = (
                            decis_vals[sj],
                            decis_vals[sj - 1],
                        )
                        decis_inds[sj - 1], decis_inds[sj] = (
                            decis_inds[sj],
                            decis_inds[sj - 1],
                        )

        if path_prev_2[decis_inds[0]] < 0:
            path_next_1[i] = decis_inds[0]
            path_prev_2[decis_inds[0]] = i
            count1 += 1
        else:
            path_next_1[i] = -1

    # ===== Level 2: No previous link, neighbor velocity =====
    for i in range(orig_parts):
        if path_prev_1[i] >= 0 or path_next_1[i] >= 0:
            continue

        vel_x = 0.0
        vel_y = 0.0
        vel_z = 0.0
        nvel = 0
        cx = path_x_1[i, 0]
        cy = path_x_1[i, 1]
        cz = path_x_1[i, 2]

        for j in range(orig_parts):
            if j == i:
                continue
            if (
                abs(path_x_1[j, 0] - cx) < dx
                and abs(path_x_1[j, 1] - cy) < dy
                and abs(path_x_1[j, 2] - cz) < dz
                and path_prev_1[j] >= 0
            ):
                pj = path_prev_1[j]
                vel_x += path_x_1[j, 0] - path_x_0[pj, 0]
                vel_y += path_x_1[j, 1] - path_x_0[pj, 1]
                vel_z += path_x_1[j, 2] - path_x_0[pj, 2]
                nvel += 1

        if nvel == 0:
            continue

        inv_nvel = 1.0 / nvel
        pred_x = cx + vel_x * inv_nvel
        pred_y = cy + vel_y * inv_nvel
        pred_z = cz + vel_z * inv_nvel

        n_cands = _find_closest_in_3d(
            path_x_2,
            np2,
            pred_x,
            pred_y,
            pred_z,
            dx,
            dy,
            dz,
            max_cands,
            cand_inds,
            cand_dists,
        )
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = cx - 2.0 * path_x_2[k, 0] + pred_x
            d1 = cy - 2.0 * path_x_2[k, 1] + pred_y
            d2 = cz - 2.0 * path_x_2[k, 2] + pred_z
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = (
                            decis_vals[sj],
                            decis_vals[sj - 1],
                        )
                        decis_inds[sj - 1], decis_inds[sj] = (
                            decis_inds[sj],
                            decis_inds[sj - 1],
                        )

        if path_prev_2[decis_inds[0]] < 0:
            path_next_1[i] = decis_inds[0]
            path_prev_2[decis_inds[0]] = i
            count1 += 1
        else:
            path_next_1[i] = -1

    # ===== Level 3: No previous link, no neighbors — static prediction =====
    for i in range(orig_parts):
        if path_prev_1[i] >= 0 or path_next_1[i] >= 0:
            continue

        pred_x = path_x_1[i, 0]
        pred_y = path_x_1[i, 1]
        pred_z = path_x_1[i, 2]

        n_cands = _find_closest_in_3d(
            path_x_2,
            np2,
            pred_x,
            pred_y,
            pred_z,
            dx,
            dy,
            dz,
            max_cands,
            cand_inds,
            cand_dists,
        )
        if n_cands == 0:
            path_next_1[i] = -1
            continue

        n_decis = 0
        for ci in range(n_cands):
            k = cand_inds[ci]
            d0 = pred_x - 2.0 * path_x_2[k, 0] + pred_x
            d1 = pred_y - 2.0 * path_x_2[k, 1] + pred_y
            d2 = pred_z - 2.0 * path_x_2[k, 2] + pred_z
            acc = c_sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            decis_vals[n_decis] = acc
            decis_inds[n_decis] = k
            n_decis += 1

        if n_decis > 1:
            for si in range(n_decis - 1):
                for sj in range(n_decis - 1, si, -1):
                    if decis_vals[sj - 1] > decis_vals[sj]:
                        decis_vals[sj - 1], decis_vals[sj] = (
                            decis_vals[sj],
                            decis_vals[sj - 1],
                        )
                        decis_inds[sj - 1], decis_inds[sj] = (
                            decis_inds[sj],
                            decis_inds[sj - 1],
                        )

        if path_prev_2[decis_inds[0]] < 0:
            path_next_1[i] = decis_inds[0]
            path_prev_2[decis_inds[0]] = i
            count1 += 1
        else:
            path_next_1[i] = -1

    return count1


# ============================================================
# Batch kernels for standalone API acceleration
# ============================================================
