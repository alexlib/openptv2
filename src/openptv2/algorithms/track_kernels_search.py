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
    _point_to_pixel_out,
)

# Sentinel values for unused particle/candidate indices — typed C int
cython.declare(
    PT_UNUSED=cython.int,
    TR_UNUSED_K=cython.int,
)
PT_UNUSED = -999
TR_UNUSED_K = -1


def candsearch_in_pix_fast(
    targ_x: cython.double[:],
    targ_y: cython.double[:],
    targ_tnr: cython.int[:],
    num_targets: cython.int,
    cent_x: cython.double,
    cent_y: cython.double,
    dl: cython.double,
    dr: cython.double,
    du: cython.double,
    dd: cython.double,
    imx: cython.double,
    imy: cython.double,
    tr_unused: cython.int,
):
    """Find up to 4 closest candidates in pixel search area.

    Args:
        targ_x, targ_y: float64 arrays of target coordinates.
        targ_tnr: int32 array of target numbers (TR_UNUSED = unused).
        num_targets: number of valid targets.
        cent_x, cent_y: search center.
        dl, dr, du, dd: search margins (left, right, up, down).
        imx, imy: image dimensions.
        tr_unused: TR_UNUSED sentinel value.

    Returns:
        (p0, p1, p2, p3) — indices of up to 4 closest candidates,
        PT_UNUSED for empty slots.
    """
    xmin: cython.double
    xmax: cython.double
    ymin: cython.double
    ymax: cython.double
    p1: cython.int
    p2: cython.int
    p3: cython.int
    p4: cython.int
    d1: cython.double
    d2: cython.double
    d3: cython.double
    d4: cython.double
    j0: cython.int
    dj: cython.int
    j: cython.int
    ty: cython.double
    tx: cython.double
    dx: cython.double
    dy: cython.double
    d: cython.double
    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > imx:
        xmax = imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > imy:
        ymax = imy

    p1 = PT_UNUSED
    p2 = PT_UNUSED
    p3 = PT_UNUSED
    p4 = PT_UNUSED
    d1 = 1e20
    d2 = 1e20
    d3 = 1e20
    d4 = 1e20

    if not (0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy):
        return p1, p2, p3, p4

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if targ_y[j0] < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        ty = targ_y[j]
        if targ_tnr[j] != tr_unused:
            if ty > ymax:
                break
            tx = targ_x[j]
            if tx > xmin and tx < xmax and ty > ymin and ty < ymax:
                dx = cent_x - tx
                dy = cent_y - ty
                d = c_sqrt(dx * dx + dy * dy)

                if d < d1:
                    p4 = p3
                    p3 = p2
                    p2 = p1
                    p1 = j
                    d4 = d3
                    d3 = d2
                    d2 = d1
                    d1 = d
                elif d < d2:
                    p4 = p3
                    p3 = p2
                    p2 = j
                    d4 = d3
                    d3 = d2
                    d2 = d
                elif d < d3:
                    p4 = p3
                    p3 = j
                    d4 = d3
                    d3 = d
                elif d < d4:
                    p4 = j
                    d4 = d

    return p1, p2, p3, p4


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def candsearch_in_pix_rest_fast(
    targ_x: cython.double[:],
    targ_y: cython.double[:],
    targ_tnr: cython.int[:],
    num_targets: cython.int,
    cent_x: cython.double,
    cent_y: cython.double,
    dl: cython.double,
    dr: cython.double,
    du: cython.double,
    dd: cython.double,
    imx: cython.double,
    imy: cython.double,
    tr_unused: cython.int,
):
    """Find closest unused candidate.

    Returns:
        (index, count) — index of closest candidate with tnr==TR_UNUSED, count (0 or 1).
    """
    xmin: cython.double
    xmax: cython.double
    ymin: cython.double
    ymax: cython.double
    best: cython.int
    dmin: cython.double
    counter: cython.int
    j0: cython.int
    dj: cython.int
    j: cython.int
    ty: cython.double
    tx: cython.double
    dx: cython.double
    dy: cython.double
    d: cython.double
    xmin = cent_x - dl
    xmax = cent_x + dr
    ymin = cent_y - du
    ymax = cent_y + dd

    if xmin < 0.0:
        xmin = 0.0
    if xmax > imx:
        xmax = imx
    if ymin < 0.0:
        ymin = 0.0
    if ymax > imy:
        ymax = imy

    best = PT_UNUSED
    dmin = 1e20
    counter = 0

    if not (0.0 <= cent_x <= imx and 0.0 <= cent_y <= imy):
        return best, 0

    j0 = num_targets // 2
    dj = num_targets // 4
    while dj > 1:
        if targ_y[j0] < ymin:
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num_targets):
        ty = targ_y[j]
        if targ_tnr[j] == tr_unused:
            if ty > ymax:
                break
            tx = targ_x[j]
            if tx > xmin and tx < xmax and ty > ymin and ty < ymax:
                dx = cent_x - tx
                dy = cent_y - ty
                d = c_sqrt(dx * dx + dy * dy)
                if d < dmin:
                    dmin = d
                    best = j
                    counter = 1

    return best, counter


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def sort_candidates_by_freq_fast(
    ftnr: cython.int[:],
    freq: cython.int[:],
    whichcam: cython.int[:, ::1],
    n: cython.int,
    num_cams: cython.int,
    max_cands: cython.int,
):
    """Sort candidates by frequency, matches C algorithm.

    Args:
        ftnr: (n,) int32 — candidate target numbers (TR_UNUSED = -1).
        freq: (n,) int32 — frequency counts (zeroed on entry).
        whichcam: (n, num_cams) int32 — camera flags.
        n: total number of entries (num_cams * max_cands).
        num_cams: number of cameras.
        max_cands: candidates per camera (4).

    Returns:
        num_valid: number of valid candidates after sort.
    """
    i: cython.int
    j: cython.int
    m: cython.int
    k: cython.int
    ftnr_i: cython.int
    num_valid: cython.int
    tr_unused = -1

    for i in range(n):
        ftnr_i = ftnr[i]
        if ftnr_i == tr_unused:
            continue
        for j in range(num_cams):
            for m in range(max_cands):
                if ftnr_i == ftnr[max_cands * j + m]:
                    whichcam[i, j] = 1

    for i in range(n):
        if ftnr[i] != tr_unused:
            for j in range(num_cams):
                if whichcam[i, j] == 1:
                    freq[i] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq[j - 1] < freq[j]:
                ftnr[j - 1], ftnr[j] = ftnr[j], ftnr[j - 1]
                freq[j - 1], freq[j] = freq[j], freq[j - 1]
                for k in range(num_cams):
                    whichcam[j - 1, k], whichcam[j, k] = (
                        whichcam[j, k],
                        whichcam[j - 1, k],
                    )

    for i in range(n):
        ftnr_i = ftnr[i]
        for j in range(i + 1, n):
            if ftnr[j] == ftnr_i or freq[j] < 2:
                freq[j] = 0
                ftnr[j] = tr_unused

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq[j - 1] < freq[j]:
                ftnr[j - 1], ftnr[j] = ftnr[j], ftnr[j - 1]
                freq[j - 1], freq[j] = freq[j], freq[j - 1]
                for k in range(num_cams):
                    whichcam[j - 1, k], whichcam[j, k] = (
                        whichcam[j, k],
                        whichcam[j - 1, k],
                    )

    num_valid = 0
    for i in range(n):
        if freq[i] != 0:
            num_valid += 1
    return num_valid


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def sorted_candidates_fast(
    center: cython.double[:],
    center_proj_x: cython.double[:],
    center_proj_y: cython.double[:],
    num_cams: cython.int,
    max_cands: cython.int,
    cal_arrays: tuple,
    mmlut_datas: tuple,
    mmlut_origins: tuple,
    mmlut_nrs: tuple,
    mmlut_nzs: tuple,
    mmlut_rws: tuple,
    targ_x_tuple: object,
    targ_y_tuple: object,
    targ_tnr_tuple: object,
    num_targets: cython.int[:],
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    tr_unused: cython.int,
):
    """Fused searchquader + candsearch + sort — single compiled entry.

    Returns (ftnr, freq, whichcam, num_valid).
    """
    n: cython.int
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp
    n = num_cams * max_cands
    ftnr = np.full(n, tr_unused, dtype=np.int32)
    freq = np.zeros(n, dtype=np.int32)
    whichcam = np.zeros((n, num_cams), dtype=np.int32)
    num_valid = _sorted_candidates_fast_out(
        center,
        center_proj_x,
        center_proj_y,
        num_cams,
        max_cands,
        cal_arrays,
        mmlut_datas,
        mmlut_origins,
        mmlut_nrs,
        mmlut_nzs,
        mmlut_rws,
        targ_x_tuple,
        targ_y_tuple,
        targ_tnr_tuple,
        num_targets,
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
        tr_unused,
        ftnr,
        freq,
        whichcam,
    )
    return ftnr, freq, whichcam, num_valid


@cython.ccall
@cython.inline
@cython.boundscheck(False)
@cython.wraparound(False)
def _sorted_candidates_fast_out(
    center: cython.double[:],
    center_proj_x: cython.double[:],
    center_proj_y: cython.double[:],
    num_cams: cython.int,
    max_cands: cython.int,
    cal_arrays: tuple,
    mmlut_datas: tuple,
    mmlut_origins: tuple,
    mmlut_nrs: tuple,
    mmlut_nzs: tuple,
    mmlut_rws: tuple,
    targ_x_tuple: object,
    targ_y_tuple: object,
    targ_tnr_tuple: object,
    num_targets: cython.int[:],
    dvxmin: cython.double,
    dvxmax: cython.double,
    dvymin: cython.double,
    dvymax: cython.double,
    dvzmin: cython.double,
    dvzmax: cython.double,
    imx_half: cython.double,
    imy_half: cython.double,
    inv_pix_x: cython.double,
    inv_pix_y: cython.double,
    chfield: cython.int,
    imx: cython.double,
    imy: cython.double,
    tr_unused: cython.int,
    ftnr_out: cython.int[:],
    freq_out: cython.int[:],
    whichcam_out: cython.int[:, :],
) -> cython.int:
    """Fused searchquader + candsearch + sort — _out variant.
    Returns num_valid.
    """
    n: cython.int
    px: cython.double
    py: cython.double
    pz: cython.double
    i: cython.int
    pt: cython.int
    xr_i: cython.double
    xl_i: cython.double
    yd_i: cython.double
    yu_i: cython.double
    cx: cython.double
    cy: cython.double
    corner_x: cython.double
    corner_y: cython.double
    mrw: cython.double
    mnr: cython.int
    mnz: cython.int
    has_mmlut: cython.int
    cam: cython.int
    base: cython.int
    ci: cython.int
    idx: cython.int
    ftnr_i: cython.int
    num_valid: cython.int
    j: cython.int
    m: cython.int
    k: cython.int
    p0: cython.int
    p1: cython.int
    p2: cython.int
    p3: cython.int
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp
    n = num_cams * max_cands

    # --- searchquader inlined ---
    px = center[0]
    py = center[1]
    pz = center[2]
    quader = np.empty((8, 3), dtype=np.float64)
    for pt in range(8):
        quader[pt, 0] = px + (dvxmax if pt & 1 else dvxmin)
        quader[pt, 1] = py + (dvymax if pt & 2 else dvymin)
        quader[pt, 2] = pz + (dvzmax if pt & 4 else dvzmin)

    xr = np.zeros(num_cams, dtype=np.float64)
    xl = np.zeros(num_cams, dtype=np.float64)
    yd = np.zeros(num_cams, dtype=np.float64)
    yu = np.zeros(num_cams, dtype=np.float64)

    for i in range(num_cams):
        cal = cal_arrays[i]
        md = mmlut_datas[i]
        mo = mmlut_origins[i]
        mnr = mmlut_nrs[i]
        mnz = mmlut_nzs[i]
        mrw = mmlut_rws[i]
        has_mmlut = mnr > 0

        xr_i = 0.0
        xl_i = float(imx)
        yd_i = 0.0
        yu_i = float(imy)
        # Use pre-computed center projection (caller already projected it)
        cx = center_proj_x[i]
        cy = center_proj_y[i]
        for pt in range(8):
            _point_to_pixel_out(
                quader[pt],
                cal,
                md,
                mo,
                mnr,
                mnz,
                mrw,
                has_mmlut,
                imx_half,
                imy_half,
                inv_pix_x,
                inv_pix_y,
                chfield,
                _pp_mv,
            )
            corner_x = _pp_mv[0]
            corner_y = _pp_mv[1]
            if corner_x < xl_i:
                xl_i = corner_x
            if corner_y < yu_i:
                yu_i = corner_y
            if corner_x > xr_i:
                xr_i = corner_x
            if corner_y > yd_i:
                yd_i = corner_y
        if xl_i < 0.0:
            xl_i = 0.0
        if yu_i < 0.0:
            yu_i = 0.0
        if xr_i > imx:
            xr_i = imx
        if yd_i > imy:
            yd_i = imy
        xr[i] = xr_i - cx
        xl[i] = cx - xl_i
        yd[i] = yd_i - cy
        yu[i] = cy - yu_i

    # --- initialize output buffers ---
    for i in range(n):
        ftnr_out[i] = tr_unused
        freq_out[i] = 0
        for j in range(num_cams):
            whichcam_out[i, j] = 0

    # --- candsearch per camera, write directly into ftnr_out/whichcam_out ---
    for cam in range(num_cams):
        p0, p1, p2, p3 = candsearch_in_pix_fast(
            targ_x_tuple[cam],
            targ_y_tuple[cam],
            targ_tnr_tuple[cam],
            num_targets[cam],
            center_proj_x[cam],
            center_proj_y[cam],
            xl[cam],
            xr[cam],
            yu[cam],
            yd[cam],
            imx,
            imy,
            tr_unused,
        )

        base = cam * max_cands
        cands = (p0, p1, p2, p3)
        for ci in range(4):
            idx = cands[ci]
            if idx != PT_UNUSED:
                whichcam_out[base + ci, cam] = 1
                ftnr_out[base + ci] = int(targ_tnr_tuple[cam][idx])

    # --- sort_candidates_by_freq inlined ---
    for i in range(n):
        ftnr_i = ftnr_out[i]
        if ftnr_i == tr_unused:
            continue
        for j in range(num_cams):
            for m in range(max_cands):
                if ftnr_i == ftnr_out[max_cands * j + m]:
                    whichcam_out[i, j] = 1

    for i in range(n):
        if ftnr_out[i] != tr_unused:
            for j in range(num_cams):
                if whichcam_out[i, j] == 1:
                    freq_out[i] += 1

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq_out[j - 1] < freq_out[j]:
                ftnr_out[j - 1], ftnr_out[j] = ftnr_out[j], ftnr_out[j - 1]
                freq_out[j - 1], freq_out[j] = freq_out[j], freq_out[j - 1]
                for k in range(num_cams):
                    whichcam_out[j - 1, k], whichcam_out[j, k] = (
                        whichcam_out[j, k],
                        whichcam_out[j - 1, k],
                    )

    for i in range(n):
        ftnr_i = ftnr_out[i]
        for j in range(i + 1, n):
            if ftnr_out[j] == ftnr_i or freq_out[j] < 2:
                freq_out[j] = 0
                ftnr_out[j] = tr_unused

    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            if freq_out[j - 1] < freq_out[j]:
                ftnr_out[j - 1], ftnr_out[j] = ftnr_out[j], ftnr_out[j - 1]
                freq_out[j - 1], freq_out[j] = freq_out[j], freq_out[j - 1]
                for k in range(num_cams):
                    whichcam_out[j - 1, k], whichcam_out[j, k] = (
                        whichcam_out[j, k],
                        whichcam_out[j - 1, k],
                    )

    num_valid = 0
    for i in range(n):
        if freq_out[i] != 0:
            num_valid += 1
    return num_valid
