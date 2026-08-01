"""Compiled kernels for the tracking hot path.

Auto-generated split from track_kernels.py.
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import (
        asin as c_asin,
    )
    from cython.cimports.libc.math import (
        atan as c_atan,
    )
    from cython.cimports.libc.math import (
        sin as c_sin,
    )
    from cython.cimports.libc.math import (
        tan as c_tan,
    )
else:
    from math import (
        asin as c_asin,
    )
    from math import (
        atan as c_atan,
    )
    from math import (
        sin as c_sin,
    )
    from math import (
        tan as c_tan,
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

    # BFS circular queue — typed memoryviews for zero-overhead C indexing
    queue_size = nnmax + 16
    qx: cython.int[:] = np.empty(queue_size, dtype=np.int32)
    qy: cython.int[:] = np.empty(queue_size, dtype=np.int32)

    # 4-connectivity offsets — typed memoryviews, avoids buffer-protocol on each BFS step
    dx4: cython.int[:] = np.array([-1, 1, 0, 0], dtype=np.int32)
    dy4: cython.int[:] = np.array([0, 0, -1, 1], dtype=np.int32)

    # Typed memoryview aliases so the nogil block can write outputs without GIL
    out_x_mv: cython.double[:] = out_x
    out_y_mv: cython.double[:] = out_y
    out_n_mv: cython.longlong[:] = out_n
    out_nx_mv: cython.longlong[:] = out_nx
    out_ny_mv: cython.longlong[:] = out_ny
    out_sumg_mv: cython.longlong[:] = out_sumg

    n_targets = 0

    with cython.nogil:
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

                head = 0
                tail = 1
                qx[0] = j
                qy[0] = i

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
                        out_x_mv[n_targets] = x_weighted / float(sumg_adj) + 0.5
                        out_y_mv[n_targets] = y_weighted / float(sumg_adj) + 0.5
                    else:
                        out_x_mv[n_targets] = float(j) + 0.5
                        out_y_mv[n_targets] = float(i) + 0.5
                    out_n_mv[n_targets] = numpix
                    out_nx_mv[n_targets] = nx
                    out_ny_mv[n_targets] = ny
                    out_sumg_mv[n_targets] = sumg
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


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def init_mmlut_data_nlay_fast(
    nr: cython.int,
    nz: cython.int,
    rw: cython.double,
    cal_t_x0: cython.double,
    cal_t_y0: cython.double,
    cal_t_z0: cython.double,
    Zmin_t: cython.double,
    mm_n1: cython.double,
    mm_n3: cython.double,
    n2: cython.double[:],
    d: cython.double[:],
    nlay: cython.int,
):
    """Fill the mmlut data grid — multi-layer multimedia (nlay > 1).

    Compiled counterpart of init_mmlut_data_fast for the general n-layer
    case. Inlines the iterative Snell solve (multimed_r_nlay_iterative) with
    typed n2[]/d[] memoryviews so the whole nr*nz grid is built in C instead
    of a Python double loop that boxed every cell through the object API.

    Args:
        nr, nz: grid dimensions.
        rw: grid cell size (mm).
        cal_t_x0, cal_t_y0, cal_t_z0: transformed camera center.
        Zmin_t: minimum Z of the translated grid.
        mm_n1, mm_n3: outer refractive indices.
        n2: (nlay,) refractive index per layer.
        d: (nlay,) thickness per layer (d[0] is the glass distance).
        nlay: number of layers.

    Returns:
        data: (nr * nz,) float64 array of radial shift factors.
    """
    i: cython.int
    j: cython.int
    k: cython.int
    it: cython.int
    R: cython.double
    Z: cython.double
    pos_x: cython.double
    zout: cython.double
    r: cython.double
    rq: cython.double
    beta1: cython.double
    sin_beta1: cython.double
    arg: cython.double
    beta3: cython.double
    rbeta: cython.double
    rdiff: cython.double
    d0: cython.double = d[0]
    n_iter: cython.int = 40
    tol: cython.double = 0.001

    data = np.empty(nr * nz, dtype=np.float64)
    for i in range(nr):
        R = i * rw + cal_t_x0
        pos_x = R
        # dx = pos_x - cal_t_x0 (== i*rw), dy = cal_t_y0 - cal_t_y0 == 0
        r = pos_x - cal_t_x0
        for j in range(nz):
            Z = Zmin_t + j * rw

            zout = Z
            for k in range(1, nlay):
                zout += d[k]

            rq = r
            data[i * nz + j] = 1.0
            for it in range(n_iter):
                beta1 = c_atan(rq / (cal_t_z0 - Z))
                sin_beta1 = c_sin(beta1)

                arg = sin_beta1 * mm_n1 / mm_n3
                if arg > 1.0:
                    arg = 1.0
                elif arg < -1.0:
                    arg = -1.0
                beta3 = c_asin(arg)

                rbeta = (cal_t_z0 - d0) * c_tan(beta1) - zout * c_tan(beta3)
                for k in range(nlay):
                    arg = sin_beta1 * mm_n1 / n2[k]
                    if arg > 1.0:
                        arg = 1.0
                    elif arg < -1.0:
                        arg = -1.0
                    rbeta += d[k] * c_tan(c_asin(arg))

                rdiff = r - rbeta
                rq += rdiff
                if rdiff < 0.0:
                    rdiff = -rdiff
                if rdiff < tol:
                    if r != 0.0:
                        data[i * nz + j] = rq / r
                    break
            else:
                data[i * nz + j] = 1.0
    return data
