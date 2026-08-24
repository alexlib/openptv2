"""Forward and backward 2D→3D tracking loops with deterministic particle linking."""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        acos as c_acos,
    )
    from cython.cimports.libc.math import (
        sqrt as c_sqrt,
    )
else:
    from math import (
        acos as c_acos,
    )
    from math import (
        sqrt as c_sqrt,
    )

if cython.compiled:
    from cython.cimports.openptv2.algorithms.track_kernels_pixel import (
        _point_to_pixel_out,
        _sorted_candidates_fast_out_nogil,
    )
    from cython.cimports.openptv2.algorithms.track_kernels_position import (
        _point_position_out,
        assess_new_position_fast_nogil,
    )
else:
    from .track_kernels_pixel import (
        _point_to_pixel_out,
        _sorted_candidates_fast_out_nogil,
    )
    from .track_kernels_position import (
        _point_position_out,
        assess_new_position_fast_nogil,
    )
from .track_kernels_search import _sorted_candidates_fast_out
from .track_kernels_transform import assess_new_position_fast

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
MAX_CANDS_K = 32
TR_UNUSED_K = -1
CORRES_NONE_K = -1
PREV_NONE_K = -1
NEXT_NONE_K = -2
COORD_UNUSED_K = -1e10
ADD_PART_K = 3.0


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.profile(False)
@cython.nogil
def _angle_acc_out(
    start_x: cython.double,
    start_y: cython.double,
    start_z: cython.double,
    pred_x: cython.double,
    pred_y: cython.double,
    pred_z: cython.double,
    cand_x: cython.double,
    cand_y: cython.double,
    cand_z: cython.double,
    out: cython.double[:],
) -> cython.int:
    """Write angle and acc to out[0], out[1] — no tuple creation."""
    v0x: cython.double
    v0y: cython.double
    v0z: cython.double
    v1x: cython.double
    v1y: cython.double
    v1z: cython.double
    angle: cython.double
    norm0: cython.double
    norm1: cython.double
    dot: cython.double
    dx: cython.double
    dy: cython.double
    dz: cython.double
    acc: cython.double
    v0x = pred_x - start_x
    v0y = pred_y - start_y
    v0z = pred_z - start_z
    v1x = cand_x - start_x
    v1y = cand_y - start_y
    v1z = cand_z - start_z

    if v0x == -v1x and v0y == -v1y and v0z == -v1z:
        angle = 200.0
    elif v0x == v1x and v0y == v1y and v0z == v1z:
        angle = 0.0
    else:
        norm0 = c_sqrt(v0x * v0x + v0y * v0y + v0z * v0z)
        norm1 = c_sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if norm0 == 0.0 or norm1 == 0.0:
            angle = 0.0
        else:
            dot = (v0x * v1x + v0y * v1y + v0z * v1z) / (norm0 * norm1)
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            angle = c_acos(dot) * 200.0 / 3.141592653589793

    dx = v1x - v0x
    dy = v1y - v0y
    dz = v1z - v0z
    acc = c_sqrt(dx * dx + dy * dy + dz * dz)
    out[0] = angle
    out[1] = acc
    return 0


@cython.cfunc
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def _trackcorr_particle_fast(
    h: cython.int,
    # Scratch buffers
    X: cython.double[:, ::1],
    cpx: cython.double[:],
    cpy: cython.double[:],
    x2_cpx: cython.double[:],
    x2_cpy: cython.double[:],
    _pp_mv: cython.double[:],
    _assess_targ: cython.double[:, ::1],
    _assess_inds: cython.int[:],
    _assess_pp: cython.double[:],
    _assess_targ2: cython.double[:, ::1],
    _assess_inds2: cython.int[:],
    _pos_mv: cython.double[:],
    _ftnr_buf1: cython.int[:],
    _freq_buf1: cython.int[:],
    _wc_buf1: cython.int[:, ::1],
    _ftnr_buf2: cython.int[:],
    _freq_buf2: cython.int[:],
    _wc_buf2: cython.int[:, ::1],
    scratch_ray: cython.double[:],
    pt_buf: cython.double[:],
    # Added particle staging buffers
    added_count_3: cython.int[:],
    added_x_3: cython.double[:, ::1],
    added_cand_3: cython.int[:, ::1],
    added_count_2: cython.int[:],
    added_h_2: cython.int[:],
    added_x_2: cython.double[:, ::1],
    added_cand_2: cython.int[:, ::1],
    added_rr_2: cython.double[:],
    # Unpacked md_arr
    md0: cython.double[:],
    md1: cython.double[:],
    md2: cython.double[:],
    md3: cython.double[:],
    md4: cython.double[:],
    md5: cython.double[:],
    md6: cython.double[:],
    md7: cython.double[:],
    # Input arrays and parameters
    path_inlist_1: cython.int[:],
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_x_0: cython.double[:, ::1],
    orig_parts_1: cython.int,
    num_cams: cython.int,
    mnr_arr: cython.int[:],
    cal_arr: cython.double[:, ::1],
    mo_arr: cython.double[:, ::1],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    corres_p_1: cython.int[:, ::1],
    targ_x_1: cython.double[:, ::1],
    targ_y_1: cython.double[:, ::1],
    targ_x_2: cython.double[:, ::1],
    targ_y_2: cython.double[:, ::1],
    targ_tnr_2: cython.int[:, ::1],
    num_targets_2: cython.int[:],
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    imx: cython.double,
    imy: cython.double,
    path_x_2: cython.double[:, ::1],
    targ_x_3: cython.double[:, ::1],
    targ_y_3: cython.double[:, ::1],
    targ_tnr_3: cython.int[:, ::1],
    num_targets_3: cython.int[:],
    path_x_3: cython.double[:, ::1],
    dacc: cython.double,
    dangle: cython.double,
    lmax: cython.double,
    path_decis_1: cython.double[:, ::1],
    path_linkdecis_1: cython.int[:, ::1],
    flatten_tol: cython.double,
    pix_x: cython.double,
    pix_y: cython.double,
    X_lay_0: cython.double,
    X_lay_1: cython.double,
    ymin: cython.double,
    ymax: cython.double,
    Zmin_lay_0: cython.double,
    Zmax_lay_1: cython.double,
    add_flag: cython.int,
) -> cython.int:
    prev_h: cython.int
    j: cython.int
    has_mmlut: cython.int
    md_j: cython.double[:]
    w_nc: cython.int
    mm: cython.int
    ftnr_mm: cython.int
    wn_nc: cython.int
    kk: cython.int
    ftnr_kk: cython.int
    dp0: cython.double
    dp1: cython.double
    dp2: cython.double
    angle1: cython.double
    acc1: cython.double
    angle0: cython.double
    acc0: cython.double
    acc: cython.double
    angle: cython.double
    d13: cython.double
    d43: cython.double
    dl: cython.double
    rr: cython.double
    inlist: cython.int
    quali: cython.int
    in_volume: cython.int
    idx_add: cython.int
    ci: cython.int
    cand_idx: cython.int
    quali_f: cython.int
    d01: cython.double
    quali2: cython.int
    claimed_ok: cython.int
    nb_j: cython.int
    nb_vx: cython.double
    nb_vy: cython.double
    nb_vz: cython.double
    nb_n: cython.int
    nb_inv: cython.double

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
            md_j = None
            if j == 0:
                md_j = md0
            elif j == 1:
                md_j = md1
            elif j == 2:
                md_j = md2
            elif j == 3:
                md_j = md3
            elif j == 4:
                md_j = md4
            elif j == 5:
                md_j = md5
            elif j == 6:
                md_j = md6
            elif j == 7:
                md_j = md7

            _point_to_pixel_out(
                X[2],
                cal_arr[j],
                md_j,
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
        # Cold start (no established previous link): borrow a velocity
        # estimate from a nearby particle in frame 1 that DOES have one,
        # same as track3d_loop_fast's Level 2 (track_kernels_track3d.py) --
        # a real turbulent flow's velocity field is spatially correlated, so
        # a neighbour's velocity is a far better prior than assuming zero
        # motion. Falls back to the original zero-velocity guess (Level 3)
        # only when no such neighbour exists. Ported 2026-08-17 after
        # finding trackcorr's zero-velocity cold start was the dominant
        # bottleneck on real-turbulence data (see docs/plans/
        # 2026-08-17-lagrangian-accuracy-program.md, next-steps item 2) --
        # every other engine already does something like this (4BE: nearest
        # neighbour; proptv_tracking: an explicit NN init phase), trackcorr
        # was the outlier assuming stationarity instead.
        nb_vx = 0.0
        nb_vy = 0.0
        nb_vz = 0.0
        nb_n = 0
        for nb_j in range(orig_parts_1):
            if nb_j == h:
                continue
            if path_prev_1[nb_j] < 0:
                continue
            if (
                path_x_1[nb_j, 0] - path_x_1[h, 0] < dvxmax
                and path_x_1[h, 0] - path_x_1[nb_j, 0] < dvxmax
                and path_x_1[nb_j, 1] - path_x_1[h, 1] < dvymax
                and path_x_1[h, 1] - path_x_1[nb_j, 1] < dvymax
                and path_x_1[nb_j, 2] - path_x_1[h, 2] < dvzmax
                and path_x_1[h, 2] - path_x_1[nb_j, 2] < dvzmax
            ):
                prev_h = path_prev_1[nb_j]
                nb_vx += path_x_1[nb_j, 0] - path_x_0[prev_h, 0]
                nb_vy += path_x_1[nb_j, 1] - path_x_0[prev_h, 1]
                nb_vz += path_x_1[nb_j, 2] - path_x_0[prev_h, 2]
                nb_n += 1
        prev_h = path_prev_1[h]  # restore -- overwritten by the scan above

        if nb_n > 0:
            nb_inv = 1.0 / nb_n
            X[2, 0] = X[1, 0] + nb_vx * nb_inv
            X[2, 1] = X[1, 1] + nb_vy * nb_inv
            X[2, 2] = X[1, 2] + nb_vz * nb_inv
        else:
            X[2, 0] = X[1, 0]
            X[2, 1] = X[1, 1]
            X[2, 2] = X[1, 2]

        for j in range(num_cams):
            if corres_p_1[h, j] == CORRES_NONE_K:
                has_mmlut = mnr_arr[j] > 0
                md_j = None
                if j == 0:
                    md_j = md0
                elif j == 1:
                    md_j = md1
                elif j == 2:
                    md_j = md2
                elif j == 3:
                    md_j = md3
                elif j == 4:
                    md_j = md4
                elif j == 5:
                    md_j = md5
                elif j == 6:
                    md_j = md6
                elif j == 7:
                    md_j = md7

                _point_to_pixel_out(
                    X[2],
                    cal_arr[j],
                    md_j,
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
                cpx[j] = targ_x_1[j, _ix]
                cpy[j] = targ_y_1[j, _ix]

    # Save X[2] projections for later use by assess_new_position_fast
    for j in range(num_cams):
        x2_cpx[j] = cpx[j]
        x2_cpy[j] = cpy[j]

    # --- sorted_candidates for frame 2 ---
    w_nc = _sorted_candidates_fast_out_nogil(
        X[2],
        cpx,
        cpy,
        num_cams,
        MAX_CANDS_K,
        cal_arr,
        md0,
        md1,
        md2,
        md3,
        md4,
        md5,
        md6,
        md7,
        mo_arr,
        mnr_arr,
        mnz_arr,
        mrw_arr,
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
        pt_buf,
        _pp_mv,
    )

    if w_nc == 0:
        return 0

    for mm in range(w_nc):
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
            md_j = None
            if j == 0:
                md_j = md0
            elif j == 1:
                md_j = md1
            elif j == 2:
                md_j = md2
            elif j == 3:
                md_j = md3
            elif j == 4:
                md_j = md4
            elif j == 5:
                md_j = md5
            elif j == 6:
                md_j = md6
            elif j == 7:
                md_j = md7

            _point_to_pixel_out(
                X[5],
                cal_arr[j],
                md_j,
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
        wn_nc = _sorted_candidates_fast_out_nogil(
            X[5],
            cpx,
            cpy,
            num_cams,
            MAX_CANDS_K,
            cal_arr,
            md0,
            md1,
            md2,
            md3,
            md4,
            md5,
            md6,
            md7,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
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
            pt_buf,
            _pp_mv,
        )

        if wn_nc > 0:
            for kk in range(wn_nc):
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

        # --- assess_new_position for X[5] in frame 3 ---
        quali = assess_new_position_fast_nogil(
            X[5],
            num_cams,
            ADD_PART_K,
            cal_arr,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
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
            cpx,
            cpy,
            _assess_targ,
            _assess_inds,
            _assess_pp,
        )

        if quali >= 2:
            in_volume = 0
            _point_position_out(_assess_targ, num_cams, cal_arr, _pos_mv, scratch_ray)
            X[4, 0] = _pos_mv[0]
            X[4, 1] = _pos_mv[1]
            X[4, 2] = _pos_mv[2]

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
                        claimed_ok = 1
                        for ci in range(num_cams):
                            cand_idx = _assess_inds[ci]
                            if cand_idx != PT_UNUSED:
                                if targ_tnr_3[ci, cand_idx] != TR_UNUSED_K:
                                    claimed_ok = 0
                                    break

                        if claimed_ok:
                            idx_add = added_count_3[0]
                            if idx_add < added_x_3.shape[0]:
                                added_x_3[idx_add, 0] = X[4, 0]
                                added_x_3[idx_add, 1] = X[4, 1]
                                added_x_3[idx_add, 2] = X[4, 2]
                                for ci in range(num_cams):
                                    cand_idx = _assess_inds[ci]
                                    added_cand_3[idx_add, ci] = cand_idx
                                    if cand_idx != PT_UNUSED:
                                        targ_tnr_3[ci, cand_idx] = -100
                                added_count_3[0] = idx_add + 1

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

    # --- add_particle to frame 2 if no links found ---
    if add_flag:
        if path_inlist_1[h] == 0 and prev_h >= 0:
            quali2 = assess_new_position_fast_nogil(
                X[2],
                num_cams,
                ADD_PART_K,
                cal_arr,
                mo_arr,
                mnr_arr,
                mnz_arr,
                mrw_arr,
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
                x2_cpx,
                x2_cpy,
                _assess_targ2,
                _assess_inds2,
                _assess_pp,
            )

            if quali2 >= 2:
                in_volume = 0
                _point_position_out(
                    _assess_targ2, num_cams, cal_arr, _pos_mv, scratch_ray
                )
                X[3, 0] = _pos_mv[0]
                X[3, 1] = _pos_mv[1]
                X[3, 2] = _pos_mv[2]

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

                        claimed_ok = 1
                        for ci in range(num_cams):
                            cand_idx = _assess_inds2[ci]
                            if cand_idx != PT_UNUSED:
                                if targ_tnr_2[ci, cand_idx] != TR_UNUSED_K:
                                    claimed_ok = 0
                                    break

                        if claimed_ok:
                            idx_add = added_count_2[0]
                            if idx_add < added_x_2.shape[0]:
                                added_h_2[idx_add] = h
                                added_x_2[idx_add, 0] = X[3, 0]
                                added_x_2[idx_add, 1] = X[3, 1]
                                added_x_2[idx_add, 2] = X[3, 2]
                                added_rr_2[idx_add] = rr
                                for ci in range(num_cams):
                                    cand_idx = _assess_inds2[ci]
                                    added_cand_2[idx_add, ci] = cand_idx
                                    if cand_idx != PT_UNUSED:
                                        targ_tnr_2[ci, cand_idx] = -100
                                added_count_2[0] = idx_add + 1

                in_volume = 0

    return 0


def trackcorr_loop_fast(
    orig_parts_1: cython.int,
    # Frame 0 (prev — read only)
    path_x_0: cython.double[:, ::1],
    # Frame 1 (curr — read/write)
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:],
    path_decis_1: cython.double[:, ::1],
    path_linkdecis_1: cython.int[:, ::1],
    corres_p_1: cython.int[:, ::1],
    targ_x_1: cython.double[:, ::1],
    targ_y_1: cython.double[:, ::1],
    targ_tnr_1: cython.int[:, ::1],
    # Frame 2 (next — read/write)
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:],
    path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, ::1],
    path_linkdecis_2: cython.int[:, ::1],
    corres_p_2: cython.int[:, ::1],
    corres_nr_2: cython.int[:],
    targ_x_2: cython.double[:, ::1],
    targ_y_2: cython.double[:, ::1],
    targ_tnr_2: cython.int[:, ::1],
    num_targets_2: cython.int[:],
    num_parts_2: cython.int[:],
    # Frame 3 (next-next — read/write)
    path_x_3: cython.double[:, ::1],
    path_prev_3: cython.int[:],
    path_next_3: cython.int[:],
    path_inlist_3: cython.int[:],
    path_prio_3: cython.int[:],
    path_finaldecis_3: cython.double[:],
    path_decis_3: cython.double[:, ::1],
    path_linkdecis_3: cython.int[:, ::1],
    corres_p_3: cython.int[:, ::1],
    corres_nr_3: cython.int[:],
    targ_x_3: cython.double[:, ::1],
    targ_y_3: cython.double[:, ::1],
    targ_tnr_3: cython.int[:, ::1],
    num_targets_3: cython.int[:],
    num_parts_3: cython.int[:],
    # Calibration — pre-flattened arrays
    cal_arr: cython.double[:, ::1],
    md_arr: object,
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
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
    num_threads: cython.int = 1,
):
    """Full per-particle tracking loop + link resolution — single compiled entry.

    All internal calls (sorted_candidates, angle_acc, assess_new_position,
    point_position) are compiled with zero dispatch overhead.

    Args:
        num_parts_2, num_parts_3: (1,) int32 arrays — mutable particle counts.

    Returns:
        (count1, num_added) — number of links established and particles added.
    """
    count1: cython.int = 0
    num_added: cython.int = 0
    h: cython.int
    idx_add: cython.int
    np3: cython.int
    np2: cython.int
    ci: cython.int
    cand_idx: cython.int
    ki: cython.int
    inlist: cython.int
    next_h: cython.int
    prev_of_next: cython.int
    ti: cython.int
    cand: cython.int
    flag: cython.bint

    n_sc: cython.int = num_cams * MAX_CANDS_K

    # Unpack md_arr to individual memoryviews for GIL-free access
    dummy_empty = np.empty(0, dtype=np.float64)
    md0: cython.double[:] = dummy_empty
    md1: cython.double[:] = dummy_empty
    md2: cython.double[:] = dummy_empty
    md3: cython.double[:] = dummy_empty
    md4: cython.double[:] = dummy_empty
    md5: cython.double[:] = dummy_empty
    md6: cython.double[:] = dummy_empty
    md7: cython.double[:] = dummy_empty

    if num_cams > 0:
        md0 = md_arr[0]
    if num_cams > 1:
        md1 = md_arr[1]
    if num_cams > 2:
        md2 = md_arr[2]
    if num_cams > 3:
        md3 = md_arr[3]
    if num_cams > 4:
        md4 = md_arr[4]
    if num_cams > 5:
        md5 = md_arr[5]
    if num_cams > 6:
        md6 = md_arr[6]
    if num_cams > 7:
        md7 = md_arr[7]

    # Pre-allocated output buffers for _sorted_candidates_fast_out_nogil
    _n_ftnr1 = np.empty(n_sc, dtype=np.int32)
    _n_freq1 = np.empty(n_sc, dtype=np.int32)
    _n_wc1 = np.empty((n_sc, num_cams), dtype=np.int32)
    _n_ftnr2 = np.empty(n_sc, dtype=np.int32)
    _n_freq2 = np.empty(n_sc, dtype=np.int32)
    _n_wc2 = np.empty((n_sc, num_cams), dtype=np.int32)

    ftnr_buf1: cython.int[:] = _n_ftnr1
    freq_buf1: cython.int[:] = _n_freq1
    wc_buf1: cython.int[:, ::1] = _n_wc1
    ftnr_buf2: cython.int[:] = _n_ftnr2
    freq_buf2: cython.int[:] = _n_freq2
    wc_buf2: cython.int[:, ::1] = _n_wc2

    _cpx = np.empty(num_cams, dtype=np.float64)
    _cpy = np.empty(num_cams, dtype=np.float64)
    _x2_cpx = np.empty(num_cams, dtype=np.float64)
    _x2_cpy = np.empty(num_cams, dtype=np.float64)
    _X = np.zeros((6, 3), dtype=np.float64)
    _pp = np.empty(2, dtype=np.float64)

    cpx: cython.double[:] = _cpx
    cpy: cython.double[:] = _cpy
    x2_cpx: cython.double[:] = _x2_cpx
    x2_cpy: cython.double[:] = _x2_cpy
    X: cython.double[:, ::1] = _X
    pp: cython.double[:] = _pp

    # Pre-allocated output buffers for assess_new_position_fast_nogil
    _assess_targ = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)
    _assess_pp = np.empty(2, dtype=np.float64)
    _assess_targ2 = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds2 = np.full(num_cams, PT_UNUSED, dtype=np.int32)

    assess_targ: cython.double[:, ::1] = _assess_targ
    assess_inds: cython.int[:] = _assess_inds
    assess_pp: cython.double[:] = _assess_pp
    assess_targ2: cython.double[:, ::1] = _assess_targ2
    assess_inds2: cython.int[:] = _assess_inds2

    # Pre-allocated output buffer for _point_position_out
    _pos = np.empty(3, dtype=np.float64)
    pos: cython.double[:] = _pos
    _scratch_ray = np.empty(6, dtype=np.float64)
    scratch_ray: cython.double[:] = _scratch_ray
    _pt_buf = np.empty(3, dtype=np.float64)
    pt_buf: cython.double[:] = _pt_buf

    # Added particle staging buffers
    max_cap3: cython.int = path_x_3.shape[0]
    max_cap2: cython.int = path_x_2.shape[0]

    _added_count_3 = np.zeros(1, dtype=np.int32)
    _added_x_3 = np.empty((max_cap3, 3), dtype=np.float64)
    _added_cand_3 = np.empty((max_cap3, num_cams), dtype=np.int32)

    added_count_3: cython.int[:] = _added_count_3
    added_x_3: cython.double[:, ::1] = _added_x_3
    added_cand_3: cython.int[:, ::1] = _added_cand_3

    _added_count_2 = np.zeros(1, dtype=np.int32)
    _added_h_2 = np.empty(max_cap2, dtype=np.int32)
    _added_x_2 = np.empty((max_cap2, 3), dtype=np.float64)
    _added_cand_2 = np.empty((max_cap2, num_cams), dtype=np.int32)
    _added_rr_2 = np.empty(max_cap2, dtype=np.float64)

    added_count_2: cython.int[:] = _added_count_2
    added_h_2: cython.int[:] = _added_h_2
    added_x_2: cython.double[:, ::1] = _added_x_2
    added_cand_2: cython.int[:, ::1] = _added_cand_2
    added_rr_2: cython.double[:] = _added_rr_2

    # Serial particle loop
    for h in range(orig_parts_1):
        _trackcorr_particle_fast(
            h,
            X,
            cpx,
            cpy,
            x2_cpx,
            x2_cpy,
            pp,
            assess_targ,
            assess_inds,
            assess_pp,
            assess_targ2,
            assess_inds2,
            pos,
            ftnr_buf1,
            freq_buf1,
            wc_buf1,
            ftnr_buf2,
            freq_buf2,
            wc_buf2,
            scratch_ray,
            pt_buf,
            added_count_3,
            added_x_3,
            added_cand_3,
            added_count_2,
            added_h_2,
            added_x_2,
            added_cand_2,
            added_rr_2,
            md0,
            md1,
            md2,
            md3,
            md4,
            md5,
            md6,
            md7,
            path_inlist_1,
            path_x_1,
            path_prev_1,
            path_x_0,
            orig_parts_1,
            num_cams,
            mnr_arr,
            cal_arr,
            mo_arr,
            mnz_arr,
            mrw_arr,
            imx_half,
            imy_half,
            inv_pix_x,
            inv_pix_y,
            chfield,
            corres_p_1,
            targ_x_1,
            targ_y_1,
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
            imx,
            imy,
            path_x_2,
            targ_x_3,
            targ_y_3,
            targ_tnr_3,
            num_targets_3,
            path_x_3,
            dacc,
            dangle,
            lmax,
            path_decis_1,
            path_linkdecis_1,
            flatten_tol,
            pix_x,
            pix_y,
            X_lay_0,
            X_lay_1,
            ymin,
            ymax,
            Zmin_lay_0,
            Zmax_lay_1,
            add_flag,
        )

    # Sequential post-loop actual appending to global arrays
    for idx_add in range(added_count_3[0]):
        np3 = num_parts_3[0]
        if np3 < path_x_3.shape[0]:
            path_x_3[np3, 0] = added_x_3[idx_add, 0]
            path_x_3[np3, 1] = added_x_3[idx_add, 1]
            path_x_3[np3, 2] = added_x_3[idx_add, 2]
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
                cand_idx = added_cand_3[idx_add, ci]
                if cand_idx != PT_UNUSED:
                    if 0 <= cand_idx < targ_tnr_3.shape[1]:
                        targ_tnr_3[ci, cand_idx] = np3
                        corres_p_3[np3, ci] = cand_idx
            num_parts_3[0] = np3 + 1
            num_added += 1

    for idx_add in range(added_count_2[0]):
        h = added_h_2[idx_add]
        np2 = num_parts_2[0]
        if np2 < path_x_2.shape[0]:
            inlist = path_inlist_1[h]
            if inlist < POSI_K:
                path_decis_1[h, inlist] = added_rr_2[idx_add]
                path_linkdecis_1[h, inlist] = np2
                path_inlist_1[h] = inlist + 1

            path_x_2[np2, 0] = added_x_2[idx_add, 0]
            path_x_2[np2, 1] = added_x_2[idx_add, 1]
            path_x_2[np2, 2] = added_x_2[idx_add, 2]
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
                cand_idx = added_cand_2[idx_add, ci]
                if cand_idx != PT_UNUSED:
                    if 0 <= cand_idx < targ_tnr_2.shape[1]:
                        targ_tnr_2[ci, cand_idx] = np2
                        corres_p_2[np2, ci] = cand_idx
            num_parts_2[0] = np2 + 1
            num_added += 1

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
            # next_h comes from path_linkdecis_1[h, 0] (Phase 1) -- normally
            # a valid frame-2 candidate index, but a wide enough search cone
            # (large dvxmax/dacc) was observed to leave it at the PT_UNUSED
            # sentinel here, which -- unguarded -- indexed path_prev_2 out of
            # bounds and crashed (segfault with boundscheck compiled out,
            # confirmed as this exact line via a temporary boundscheck=True
            # rebuild). Same defensive pattern already used a few lines up
            # for cand_idx into targ_tnr_2.
            if next_h == PT_UNUSED or not (0 <= next_h < path_prev_2.shape[0]):
                path_next_1[h] = NEXT_NONE_K
                continue
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
    path_x_0: cython.double[:, ::1],
    # Frame 1 (current — read/write)
    path_x_1: cython.double[:, ::1],
    path_prev_1: cython.int[:],
    path_next_1: cython.int[:],
    path_inlist_1: cython.int[:],
    path_finaldecis_1: cython.double[:],
    path_decis_1: cython.double[:, ::1],
    path_linkdecis_1: cython.int[:, ::1],
    # Frame 2 (backward/prev in time — read/write)
    path_x_2: cython.double[:, ::1],
    path_prev_2: cython.int[:],
    path_next_2: cython.int[:],
    num_parts_2: cython.int[:],
    targ_x_2: cython.double[:, ::1],
    targ_y_2: cython.double[:, ::1],
    targ_tnr_2: cython.int[:, ::1],
    num_targets_2: cython.int[:],
    corres_p_2: cython.int[:, ::1],
    corres_nr_2: cython.int[:],
    path_inlist_2: cython.int[:],
    path_prio_2: cython.int[:],
    path_finaldecis_2: cython.double[:],
    path_decis_2: cython.double[:, ::1],
    path_linkdecis_2: cython.int[:, ::1],
    # Frame 3 (further backward — read only, for extra angle check)
    path_x_3: cython.double[:, ::1],
    path_prev_3: cython.int[:],
    # Calibration — pre-flattened arrays
    cal_arr: cython.double[:, ::1],
    md_arr: object,
    mo_arr: cython.double[:, ::1],
    mnr_arr: cython.int[:],
    mnz_arr: cython.int[:],
    mrw_arr: cython.double[:],
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
    _wc_buf: cython.int[:, ::1] = _n_wc
    _cpx = np.empty(num_cams, dtype=np.float64)
    _cpy = np.empty(num_cams, dtype=np.float64)
    _X = np.zeros((6, 3), dtype=np.float64)
    cpx: cython.double[:] = _cpx
    cpy: cython.double[:] = _cpy
    X: cython.double[:, ::1] = _X
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp

    # Pre-allocated output buffers for assess_new_position_fast
    _assess_targ = np.full((num_cams, 2), COORD_UNUSED_K, dtype=np.float64)
    _assess_inds = np.full(num_cams, PT_UNUSED, dtype=np.int32)
    _assess_pp = np.empty(2, dtype=np.float64)

    _pos_buf = np.zeros(3, dtype=np.float64)
    _pos_mv: cython.double[:] = _pos_buf
    _scratch_ray = np.zeros(6, dtype=np.float64)
    scratch_ray: cython.double[:] = _scratch_ray

    # cal_arr, md_arr, mo_arr, mnr_arr, mnz_arr, mrw_arr pre-flattened by caller
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
            cal_arr,
            md_arr,
            mo_arr,
            mnr_arr,
            mnz_arr,
            mrw_arr,
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
                        quali = _freq_buf[i]
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
                    cal_arr,
                    md_arr,
                    mo_arr,
                    mnr_arr,
                    mnz_arr,
                    mrw_arr,
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
                    proj_x=cpx,
                    proj_y=cpy,
                    targ_pos_out=_assess_targ,
                    cand_inds_out=_assess_inds,
                    scratch=_assess_pp,
                )

                if quali >= 2:
                    in_volume = 0
                    _point_position_out(
                        targ_pos, num_cams, cal_arr, _pos_mv, scratch_ray
                    )
                    X[3, 0] = _pos_buf[0]
                    X[3, 1] = _pos_buf[1]
                    X[3, 2] = _pos_buf[2]

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
                                    targ_tnr_2[ci, idx] = np2
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
