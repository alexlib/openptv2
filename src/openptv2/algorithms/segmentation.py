"""Particle detection via thresholding and peak fitting.

Translation of lib/src/segmentation.c and lib/include/segmentation.h.

Provides:
- targ_rec: thresholding and center-of-gravity with peak fitting (delegates to
  track_kernels._targ_rec_fast for the hot BFS — that function compiles to
  near-pure C with typed memoryviews).
- peak_fit: two-pass component labeling with reunification (alternative
  implementation, used only in tests).
"""

import cython
from dataclasses import dataclass

import numpy as np

from .tracking_frame_buf import Target
from .track_kernels import targ_rec_fast as _targ_rec_fast

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt

# Constant for no correspondence assigned
CORRES_NONE = -1


@cython.cclass
@dataclass
class Peak:
    """Detected peak for connectivity analysis."""

    pos: cython.int = 0
    status: cython.int = 0
    xmin: cython.int = 0
    xmax: cython.int = 0
    ymin: cython.int = 0
    ymax: cython.int = 0
    n: cython.int = 0
    sumg: cython.int = 0
    x: cython.double = 0.0
    y: cython.double = 0.0
    unr: cython.int = 0
    touch: list = cython.declare(list, visibility="public")
    n_touch: cython.int = 0


@cython.ccall
def check_touch(tpeak: Peak, p1: cython.int, p2: cython.int) -> None:
    """Check whether p1, p2 are already marked as touching and mark them otherwise."""
    if p2 == 0 or p2 == p1:
        return

    m: cython.int
    for m in range(tpeak.n_touch):
        if tpeak.touch[m] == p2:
            return

    tpeak.touch[tpeak.n_touch] = p2
    tpeak.n_touch += 1
    if tpeak.n_touch > 3:
        tpeak.n_touch = 3


@cython.ccall
def _is_local_maximum(
    img: cython.uchar[:, :], i: cython.int, j: cython.int
) -> cython.bint:
    """Check if pixel at (i, j) is an 8-neighbor local maximum.

    Pure C pointer arithmetic when compiled — no Python overhead.
    """
    gv = img[i, j]
    return (
        gv >= img[i, j - 1]
        and gv >= img[i, j + 1]
        and gv >= img[i - 1, j]
        and gv >= img[i + 1, j]
        and gv >= img[i - 1, j - 1]
        and gv >= img[i + 1, j - 1]
        and gv >= img[i - 1, j + 1]
        and gv >= img[i + 1, j + 1]
    )


@cython.ccall
def targ_rec(
    img: cython.uchar[:, :],
    gvthres: cython.int,
    discont: cython.int,
    nnmin: cython.int,
    nnmax: cython.int,
    nxmin: cython.int,
    nxmax: cython.int,
    nymin: cython.int,
    nymax: cython.int,
    sumg_min: cython.int,
    xmin: cython.int = 1,
    xmax: cython.int = -1,
    ymin: cython.int = 1,
    ymax: cython.int = -1,
) -> list:
    """Thresholding and center-of-gravity with peak fitting (C targ_rec translation).

    Delegates to the compiled BFS in track_kernels._targ_rec_fast.

    Args:
        img: input image (2D uint8, shape (imy, imx)).
        gvthres: grey value threshold for binarization.
        discont: maximum discontinuity for peak growth.
        nnmin, nnmax: min/max number of pixels per target.
        nxmin, nxmax: min/max extent in x.
        nymin, nymax: min/max extent in y.
        sumg_min: minimum sum of grey values.
        xmin, xmax, ymin, ymax: search area (defaults to image bounds).

    Returns:
        List of detected targets.
    """
    imy: cython.int = img.shape[0]
    imx: cython.int = img.shape[1]
    if xmax < 0:
        xmax = imx - 1
    if ymax < 0:
        ymax = imy - 1

    xmin = max(xmin, 1)
    ymin = max(ymin, 1)
    xmax = min(xmax, imx - 1)
    ymax = min(ymax, imy - 1)

    img0 = np.asarray(img, dtype=np.uint8).copy()
    max_targets = (xmax - xmin) * (ymax - ymin)

    n_found, ox, oy, on, onx, ony, osumg = _targ_rec_fast(
        np.asarray(img, dtype=np.uint8),
        img0,
        gvthres,
        discont,
        nnmin,
        nnmax,
        nxmin,
        nxmax,
        nymin,
        nymax,
        sumg_min,
        xmin,
        ymin,
        xmax,
        ymax,
        max_targets,
    )
    if n_found == 0:
        return [Target(pnr=1, x=1, y=1, n=1, nx=1, ny=1, sumg=1, tnr=CORRES_NONE)]
    return [
        Target(
            pnr=k,
            x=float(ox[k]),
            y=float(oy[k]),
            n=int(on[k]),
            nx=int(onx[k]),
            ny=int(ony[k]),
            sumg=int(osumg[k]),
            tnr=CORRES_NONE,
        )
        for k in range(n_found)
    ]


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def peak_fit(
    img: cython.uchar[:, :],
    gvthres: cython.int,
    discont: cython.int,
    nnmin: cython.int,
    nnmax: cython.int,
    nxmin: cython.int,
    nxmax: cython.int,
    nymin: cython.int,
    nymax: cython.int,
    sumg_min: cython.int,
    xmin: cython.int = 1,
    xmax: cython.int = -1,
    ymin: cython.int = 1,
    ymax: cython.int = -1,
) -> list:
    """Two-pass component labeling with peak fitting and reunification."""
    imy: cython.int = img.shape[0]
    imx: cython.int = img.shape[1]
    if xmax < 0:
        xmax = imx
    if ymax < 0:
        ymax = imy

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    gv: cython.int

    # Pre-allocated typed arrays for BFS queue (maximally sized)
    _qx = np.empty(imy * imx, dtype=np.int32)
    _qy = np.empty(imy * imx, dtype=np.int32)
    qx: cython.int[:] = _qx
    qy: cython.int[:] = _qy

    # Static direction lookup arrays — compile to C constant arrays
    dx4 = [-1, 1, 0, 0]
    dy4 = [0, 0, -1, 1]
    di8 = [-1, -1, -1, 0, 0, 1, 1, 1]
    dj8 = [-1, 0, 1, -1, 1, -1, 0, 1]

    qhead: cython.int
    qtail: cython.int
    wx: cython.int
    wy: cython.int
    nx_pos: cython.int
    ny_pos: cython.int
    gvref: cython.int
    neighbor_gv: cython.int
    d: cython.int

    # Label image
    label_img = np.zeros((imy, imx), dtype=np.int32)
    peaks: list[Peak] = []

    # ---- Pass 1: Connectivity analysis with peak search ----
    for i in range(ymin, ymax - 1):
        for j in range(xmin, xmax):
            n = i * imx + j
            gv = img[i, j]

            if gv <= gvthres:
                continue
            if label_img[i, j] != 0:
                continue

            # Check local maximum
            if not _is_local_maximum(img, i, j):
                continue

            # New peak
            n_peaks = len(peaks) + 1
            label_img[i, j] = n_peaks

            peak = Peak(pos=n, status=1, xmin=j, xmax=j, ymin=i, ymax=i)
            peaks.append(peak)

            # BFS region growing — typed array queue
            qhead = 0
            qtail = 0
            qx[qtail] = j
            qy[qtail] = i
            qtail += 1
            label_img[i, j] = n_peaks

            while qhead < qtail:
                wx = qx[qhead]
                wy = qy[qhead]
                qhead += 1
                gvref = img[wy, wx]

                for d in range(4):
                    nx_pos = wx + dx4[d]
                    ny_pos = wy + dy4[d]
                    if not (0 <= nx_pos < imx and 0 <= ny_pos < imy):
                        continue
                    if label_img[ny_pos, nx_pos] != 0:
                        continue

                    neighbor_gv = img[ny_pos, nx_pos]

                    if (
                        neighbor_gv > gvthres
                        and xmin <= nx_pos < xmax
                        and ymin <= ny_pos < ymax - 1
                        and neighbor_gv <= gvref + discont
                        and gvref + discont >= img[ny_pos - 1, nx_pos]
                        and gvref + discont >= img[ny_pos + 1, nx_pos]
                        and gvref + discont >= img[ny_pos, nx_pos - 1]
                        and gvref + discont >= img[ny_pos, nx_pos + 1]
                    ):
                        label_img[ny_pos, nx_pos] = n_peaks
                        qx[qtail] = nx_pos
                        qy[qtail] = ny_pos
                        qtail += 1

    # ---- Pass 2: Collect data and detect touches ----
    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            n = i * imx + j
            label = label_img[i, j]

            if label <= 0:
                continue

            pnr = label - 1
            peak = peaks[pnr]
            gv = img[i, j]

            peak.n += 1
            peak.sumg += gv
            peak.x += float(j) * gv
            peak.y += float(i) * gv

            if j < peak.xmin:
                peak.xmin = j
            if j > peak.xmax:
                peak.xmax = j
            if i < peak.ymin:
                peak.ymin = i
            if i > peak.ymax:
                peak.ymax = i

            # Check 8-neighbors for touches
            for d in range(8):
                ni: cython.int = i + di8[d]
                nj: cython.int = j + dj8[d]
                if 0 <= ni < imy and 0 <= nj < imx:
                    neighbor_label = label_img[ni, nj]
                    check_touch(peak, label, neighbor_label)

    # ---- Pass 3: Reunification test ----
    for peak_i in peaks:
        if peak_i.n_touch == 0 or peak_i.unr != 0:
            continue

        x1 = peak_i.x / peak_i.sumg
        y1 = peak_i.y / peak_i.sumg
        pi: cython.int = peak_i.pos // imx
        pj: cython.int = peak_i.pos % imx
        gv1 = img[pi, pj]

        for j_idx in peak_i.touch:
            p2 = j_idx - 1
            if p2 < 0 or p2 >= len(peaks) or peaks[p2].unr != 0:
                continue

            peak_j = peaks[p2]
            x2 = peak_j.x / peak_j.sumg
            y2 = peak_j.y / peak_j.sumg
            pj2: cython.int = peak_j.pos // imx
            pj3: cython.int = peak_j.pos % imx
            gv2 = img[pj2, pj3]

            s12 = c_sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Profile criterion
            unify: cython.bint = s12 < 2.0
            if not unify:
                unify = True
                l: cython.int
                for l in range(1, int(s12)):
                    intx1 = int(x1 + l * (x2 - x1) / s12)
                    inty1 = int(y1 + l * (y2 - y1) / s12)

                    if 0 <= inty1 < imy and 0 <= intx1 < imx:
                        gv = img[inty1, intx1] + discont
                        if gv < gv1 + l * (gv2 - gv1) / s12 or gv < gv1 or gv < gv2:
                            unify = False
                            break
                    else:
                        unify = False
                        break

            if not unify:
                continue

            # Unify targets
            peak_i.unr = p2 + 1  # 1-indexed
            peak_j.x += peak_i.x
            peak_j.y += peak_i.y
            peak_j.sumg += peak_i.sumg
            peak_j.n += peak_i.n
            if peak_i.xmin < peak_j.xmin:
                peak_j.xmin = peak_i.xmin
            if peak_i.ymin < peak_j.ymin:
                peak_j.ymin = peak_i.ymin
            if peak_i.xmax > peak_j.xmax:
                peak_j.xmax = peak_i.xmax
            if peak_i.ymax > peak_j.ymax:
                peak_j.ymax = peak_i.ymax

    # ---- Pass 4: Output targets ----
    targets = []
    for peak in peaks:
        if peak.unr != 0:
            continue

        width = xmax - xmin
        if width > 32:
            if peak.xmin == xmin or peak.ymin == ymin:
                continue
            if peak.xmax == xmax - 1 or peak.ymax == ymax - 1:
                continue

        nx = peak.xmax - peak.xmin + 1
        ny = peak.ymax - peak.ymin + 1

        if (
            peak.sumg > sumg_min
            and nxmin <= nx <= nxmax
            and nymin <= ny <= nymax
            and nnmin <= peak.n <= nnmax
        ):
            x_final = 0.5 + peak.x / peak.sumg
            y_final = 0.5 + peak.y / peak.sumg

            targets.append(
                Target(
                    pnr=len(targets),
                    x=x_final,
                    y=y_final,
                    n=peak.n,
                    nx=nx,
                    ny=ny,
                    sumg=peak.sumg,
                    tnr=CORRES_NONE,
                )
            )

    return targets


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
