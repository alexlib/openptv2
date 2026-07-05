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
    _multimed_r_nlay_1layer,
    _ray_tracing_out,
)
from .track_kernels_transform import (
    _metric_to_pixel_out,
    _pixel_to_metric_out,
    point_position_fast,
)


def ray_tracing_batch_fast(xy: cython.double[:, :], cal: cython.double[:]):
    """Trace N rays through multi-media interface.

    Args:
        xy: (N, 2) float64 — metric image coordinates.
        cal: (31,) float64 — packed calibration.

    Returns:
        (positions, directions) each (N, 3) float64.
    """
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    _ray_out = np.empty(6, dtype=np.float64)
    _ray_out_mv: cython.double[:] = _ray_out
    n = xy.shape[0]
    positions = np.empty((n, 3), dtype=np.float64)
    directions = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        _ray_tracing_out(xy[i, 0], xy[i, 1], cal, _ray_out_mv)
        positions[i, 0] = _ray_out_mv[0]
        positions[i, 1] = _ray_out_mv[1]
        positions[i, 2] = _ray_out_mv[2]
        directions[i, 0] = _ray_out_mv[3]
        directions[i, 1] = _ray_out_mv[4]
        directions[i, 2] = _ray_out_mv[5]
    return positions, directions


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def point_position_batch_fast(
    all_targets: cython.double[:, :, :],
    num_pts: cython.int,
    num_cams: cython.int,
    cal_arrays,
):
    """Triangulate M targets from N cameras.

    Args:
        all_targets: (M, num_cams, 2) float64.
        num_pts: M.
        num_cams: N.
        cal_arrays: tuple of (31,) float64 arrays.

    Returns:
        (positions, distances) — (M, 3) and (M,) float64.
    """
    i: cython.Py_ssize_t
    dist: cython.double
    positions = np.empty((num_pts, 3), dtype=np.float64)
    distances = np.empty(num_pts, dtype=np.float64)
    for i in range(num_pts):
        _cal_arr = np.asarray(list(cal_arrays), dtype=np.float64)
        pos, dist = point_position_fast(all_targets[i], num_cams, _cal_arr)
        positions[i, 0] = pos[0]
        positions[i, 1] = pos[1]
        positions[i, 2] = pos[2]
        distances[i] = dist
    return positions, distances


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def pixel_to_metric_batch_fast(
    xy: cython.double[:, :],
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    chfield: cython.int,
):
    """Convert N pixel coordinates to metric."""
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp
    n = xy.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        _pixel_to_metric_out(
            xy[i, 0],
            xy[i, 1],
            imx,
            imy,
            pix_x,
            pix_y,
            chfield,
            _pp_mv,
        )
        result[i, 0] = _pp_mv[0]
        result[i, 1] = _pp_mv[1]
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def metric_to_pixel_batch_fast(
    xy: cython.double[:, :],
    imx: cython.int,
    imy: cython.int,
    pix_x: cython.double,
    pix_y: cython.double,
    chfield: cython.int,
):
    """Convert N metric coordinates to pixel."""
    n: cython.Py_ssize_t
    i: cython.Py_ssize_t
    _pp = np.empty(2, dtype=np.float64)
    _pp_mv: cython.double[:] = _pp
    n = xy.shape[0]
    result = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        _metric_to_pixel_out(
            xy[i, 0],
            xy[i, 1],
            imx,
            imy,
            pix_x,
            pix_y,
            chfield,
            _pp_mv,
        )
        result[i, 0] = _pp_mv[0]
        result[i, 1] = _pp_mv[1]
    return result


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def targ_rec_fast(
    img: cython.uchar[:, :],
    img0: cython.uchar[:, :],
    gvthres: cython.int,
    discont: cython.int,
    nnmin: cython.int,
    nnmax: cython.int,
    nxmin: cython.int,
    nxmax: cython.int,
    nymin: cython.int,
    nymax: cython.int,
    sumg_min: cython.int,
    xmin: cython.int,
    ymin: cython.int,
    xmax: cython.int,
    ymax: cython.int,
    max_targets: cython.int,
):
    """BFS flood-fill target recognition.

    Args:
        img: original uint8 image (imy, imx) — read-only for neighbor checks.
        img0: writable copy (imy, imx) — pixels zeroed as they are consumed.
        gvthres: grey value threshold.
        discont: maximum grey value discontinuity for BFS growth.
        nnmin, nnmax: min/max pixel count per target.
        nxmin, nxmax, nymin, nymax: min/max bounding box per target.
        sumg_min: minimum sum of grey values.
        xmin, ymin, xmax, ymax: search window (inclusive on ymin/xmin side).
        max_targets: pre-allocated output capacity.

    Returns:
        (n_targets, out_x, out_y, out_n, out_nx, out_ny, out_sumg)
        First n_targets elements of each output array are valid.
    """
    n_targets: cython.int
    queue_size: cython.int
    i: cython.int
    j: cython.int
    d: cython.int
    xa: cython.int
    xb: cython.int
    ya: cython.int
    yb: cython.int
    x_weighted: cython.double
    y_weighted: cython.double
    head: cython.int
    tail: cython.int
    wj: cython.int
    wi: cython.int
    xn4: cython.int
    yn4: cython.int
    nx: cython.int
    ny: cython.int
    gv: cython.int
    sumg: cython.int
    numpix: cython.int
    gvref: cython.int
    gv4: cython.int
    sumg_adj: cython.int
    out_x = np.empty(max_targets, dtype=np.float64)
    out_y = np.empty(max_targets, dtype=np.float64)
    out_n = np.empty(max_targets, dtype=np.int64)
    out_nx = np.empty(max_targets, dtype=np.int64)
    out_ny = np.empty(max_targets, dtype=np.int64)
    out_sumg = np.empty(max_targets, dtype=np.int64)

    # BFS circular queue — bounded by nnmax (we stop adding beyond that).
    queue_size = nnmax + 16
    qx = np.empty(queue_size, dtype=np.int32)
    qy = np.empty(queue_size, dtype=np.int32)

    # Offsets for 4-connectivity
    dx4 = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy4 = np.array([0, 0, -1, 1], dtype=np.int32)

    n_targets = 0

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            gv = int(img[i, j])
            if gv <= gvthres:
                continue

            # 8-neighbor local maximum check
            if not (
                gv >= img[i, j - 1]
                and gv >= img[i, j + 1]
                and gv >= img[i - 1, j]
                and gv >= img[i + 1, j]
                and gv >= img[i - 1, j - 1]
                and gv >= img[i + 1, j - 1]
                and gv >= img[i - 1, j + 1]
                and gv >= img[i + 1, j + 1]
            ):
                continue

            if img0[i, j] <= gvthres:
                continue

            # Start BFS from this peak
            sumg = gv
            img0[i, j] = 0
            xa = j
            xb = j
            ya = i
            yb = i
            x_weighted = float(j) * float(gv - gvthres)
            y_weighted = float(i) * float(gv - gvthres)
            numpix = 1

            head = np.int32(0)
            tail = np.int32(1)
            qx[0] = np.int32(j)
            qy[0] = np.int32(i)

            while head != tail:
                wj = qx[head]
                wi = qy[head]
                head += 1
                if head >= queue_size:
                    head = 0
                gvref = int(img[wi, wj])

                for d in range(4):
                    xn4 = wj + dx4[d]
                    yn4 = wi + dy4[d]

                    if xn4 < xmin or xn4 >= xmax or yn4 < ymin or yn4 >= ymax:
                        continue

                    gv4 = int(img0[yn4, xn4])
                    if (
                        gv4 > gvthres
                        and gv4 <= gvref + discont
                        and gvref + discont >= img[yn4 - 1, xn4]
                        and gvref + discont >= img[yn4 + 1, xn4]
                        and gvref + discont >= img[yn4, xn4 - 1]
                        and gvref + discont >= img[yn4, xn4 + 1]
                    ):
                        sumg += gv4
                        img0[yn4, xn4] = 0
                        if xn4 < xa:
                            xa = xn4
                        if xn4 > xb:
                            xb = xn4
                        if yn4 < ya:
                            ya = yn4
                        if yn4 > yb:
                            yb = yn4
                        x_weighted += float(xn4) * float(gv4 - gvthres)
                        y_weighted += float(yn4) * float(gv4 - gvthres)
                        numpix += 1
                        if numpix <= nnmax:
                            qx[tail] = xn4
                            qy[tail] = yn4
                            tail += 1
                            if tail >= queue_size:
                                tail = 0

            # Skip particles whose bounding box would extend outside the
            # search area (mirrors the original C border check convention).
            if xa == xmin - 1 or ya == ymin - 1 or xb == xmax + 1 or yb == ymax + 1:
                continue

            nx = xb - xa + 1
            ny = yb - ya + 1

            if (
                nnmin <= numpix <= nnmax
                and nxmin <= nx <= nxmax
                and nymin <= ny <= nymax
                and sumg > sumg_min
            ):
                if n_targets >= max_targets:
                    break

                sumg_adj = sumg - numpix * gvthres
                if sumg_adj > 0:
                    out_x[n_targets] = x_weighted / float(sumg_adj) + 0.5
                    out_y[n_targets] = y_weighted / float(sumg_adj) + 0.5
                else:
                    out_x[n_targets] = float(j) + 0.5
                    out_y[n_targets] = float(i) + 0.5
                out_n[n_targets] = numpix
                out_nx[n_targets] = nx
                out_ny[n_targets] = ny
                out_sumg[n_targets] = sumg
                n_targets += 1

    return n_targets, out_x, out_y, out_n, out_nx, out_ny, out_sumg


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def init_mmlut_data_fast(
    nr: cython.int,
    nz: cython.int,
    rw: cython.double,
    cal_t_x0: cython.double,
    cal_t_y0: cython.double,
    cal_t_z0: cython.double,
    Zmin_t: cython.double,
    mm_n1: cython.double,
    mm_n2_0: cython.double,
    mm_n3: cython.double,
    mm_d0: cython.double,
):
    """Fill mmlut data grid in parallel — single-layer multimedia.

    Computes the radial shift factor for every (R, Z) grid point using
    _multimed_r_nlay_1layer. Rows are R-indices, columns are Z-indices.

    Args:
        nr: number of R grid points.
        nz: number of Z grid points.
        rw: grid cell size (mm).
        cal_t_x0, cal_t_y0, cal_t_z0: transformed camera center.
        Zmin_t: minimum Z of translated grid.
        mm_n1, mm_n2_0, mm_n3, mm_d0: single-layer multimedia parameters.

    Returns:
        data: (nr * nz,) float64 array of radial shift factors.
    """
    i: cython.int
    j: cython.int
    R: cython.double
    Z: cython.double
    data = np.empty(nr * nz, dtype=np.float64)
    for i in range(nr):
        R = i * rw + cal_t_x0
        for j in range(nz):
            Z = Zmin_t + j * rw
            data[i * nz + j] = _multimed_r_nlay_1layer(
                R,
                cal_t_y0,
                Z,
                cal_t_x0,
                cal_t_y0,
                cal_t_z0,
                mm_n1,
                mm_n2_0,
                mm_n3,
                mm_d0,
            )
    return data
