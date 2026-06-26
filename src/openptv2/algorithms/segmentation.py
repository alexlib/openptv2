"""Particle detection via thresholding and peak fitting.

Translation of lib/src/segmentation.c and lib/include/segmentation.h.

Provides:
- targ_rec: thresholding and center-of-gravity with peak fitting
- peak_fit: two-pass component labeling with reunification
- check_touch: detect touching peaks

Uses NumPy vectorized operations where possible, with clear Python
logic for the connected component labeling algorithms.
"""
import cython


import numpy as np
from dataclasses import dataclass, field

# Constant for no correspondence assigned
CORRES_NONE = -1

from .track_kernels import targ_rec_fast as _targ_rec_fast




@dataclass
class Target:
    """Detected particle target.
    pnr: particle number (index)
    x, y: centroid coordinates
    n: number of pixels in target
    nx, ny: extent in x and y
    sumg: sum of grey values
    tnr: correspondence number (-1 = unassigned)
    """
    pnr: int = 0
    x: float = 0.0
    y: float = 0.0
    n: int = 0
    nx: int = 0
    ny: int = 0
    sumg: int = 0
    tnr: int = CORRES_NONE


@dataclass
class Peak:
    """Detected peak for connectivity analysis."""
    pos: int = 0
    status: int = 0
    xmin: int = 0
    xmax: int = 0
    ymin: int = 0
    ymax: int = 0
    n: int = 0
    sumg: int = 0
    x: float = 0.0
    y: float = 0.0
    unr: int = 0
    touch: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    n_touch: int = 0


@cython.ccall
def check_touch(tpeak: Peak, p1: int, p2: int) -> None:
    """Check whether p1, p2 are already marked as touching and mark them otherwise.

    Matches C implementation exactly, including the cap on touches.
    """
    if p2 == 0:
        return
    if p2 == p1:
        return

    m: cython.int
    # check whether p1, p2 are already marked as touching
    for m in range(tpeak.n_touch):
        if tpeak.touch[m] == p2:
            return

    # mark touch event
    tpeak.touch[tpeak.n_touch] = p2
    tpeak.n_touch += 1

    # don't allow for more than 4 touches (C caps at 3, meaning index 3 is max)
    if tpeak.n_touch > 3:
        tpeak.n_touch = 3


def _is_local_maximum(img: np.ndarray, i: cython.int, j: cython.int, imx: cython.int, imy: cython.int) -> bool:
    """Check if pixel at (i, j) is an 8-neighbor local maximum."""
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
    img: np.ndarray,
    gvthres: int,
    discont: int,
    nnmin: int,
    nnmax: int,
    nxmin: int,
    nxmax: int,
    nymin: int,
    nymax: int,
    sumg_min: int,
    xmin: int = 1,
    xmax: int = -1,
    ymin: int = 1,
    ymax: int = -1,
) -> list[Target]:
    """Thresholding and center-of-gravity with peak fitting (C targ_rec translation).

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
        List of detected targets (structure-of-arrays, like C target pix[]).
    """
    imy, imx = img.shape
    if xmax < 0:
        xmax = imx - 1
    if ymax < 0:
        ymax = imy - 1

    xmin = max(xmin, 1)
    ymin = max(ymin, 1)
    xmax = min(xmax, imx - 1)
    ymax = min(ymax, imy - 1)

    img_u8 = np.asarray(img, dtype=np.uint8)

    img0 = img_u8.copy()
    max_targets = (xmax - xmin) * (ymax - ymin)
    n_found, ox, oy, on, onx, ony, osumg = _targ_rec_fast(
        img_u8, img0,
        np.int64(gvthres), np.int64(discont),
        np.int64(nnmin), np.int64(nnmax),
        np.int64(nxmin), np.int64(nxmax),
        np.int64(nymin), np.int64(nymax),
        np.int64(sumg_min),
        np.int32(xmin), np.int32(ymin),
        np.int32(xmax), np.int32(ymax),
        np.int64(max_targets),
    )
    if n_found == 0:
        return [Target(pnr=1, x=1, y=1, n=1, nx=1, ny=1, sumg=1, tnr=CORRES_NONE)]
    return [
        Target(pnr=k, x=float(ox[k]), y=float(oy[k]),
               n=int(on[k]), nx=int(onx[k]), ny=int(ony[k]),
               sumg=int(osumg[k]), tnr=CORRES_NONE)
        for k in range(n_found)
    ]


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
def peak_fit(
    img: np.ndarray,
    gvthres: int,
    discont: int,
    nnmin: int,
    nnmax: int,
    nxmin: int,
    nxmax: int,
    nymin: int,
    nymax: int,
    sumg_min: int,
    xmin: int = 1,
    xmax: int = -1,
    ymin: int = 1,
    ymax: int = -1,
) -> list[Target]:
    """Two-pass component labeling with peak fitting and reunification."""
    imy, imx = img.shape
    if xmax < 0:
        xmax = imx
    if ymax < 0:
        ymax = imy

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    gv: cython.int

    # Pre-allocated typed arrays for BFS queue (maximally sized to image dimensions)
    _qx = np.empty(imy * imx, dtype=np.int32)
    _qy = np.empty(imy * imx, dtype=np.int32)
    qx: cython.int[:] = _qx
    qy: cython.int[:] = _qy
    qhead: cython.int
    qtail: cython.int
    wx: cython.int
    wy: cython.int
    nx_pos: cython.int
    ny_pos: cython.int
    gvref: cython.int
    neighbor_gv: cython.int
    dx: cython.int
    dy: cython.int

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
            if not _is_local_maximum(img, i, j, imx, imy):
                continue

            # New peak
            n_peaks = len(peaks) + 1
            label_img[i, j] = n_peaks

            peak = Peak(
                pos=n,
                status=1,
                xmin=j,
                xmax=j,
                ymin=i,
                ymax=i,
            )
            peaks.append(peak)

            # BFS region growing — typed array queue (faster than deque in compiled mode)
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
                gvref = int(img[wy, wx])

                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx_pos = wx + dx
                    ny_pos = wy + dy
                    if not (0 <= nx_pos < imx and 0 <= ny_pos < imy):
                        continue
                    if label_img[ny_pos, nx_pos] != 0:
                        continue

                    neighbor_gv = int(img[ny_pos, nx_pos])

                    if (
                        neighbor_gv > gvthres
                        and xmin <= nx_pos < xmax
                        and ymin <= ny_pos < ymax - 1
                        and neighbor_gv <= gvref + discont
                        and gvref + discont >= int(img[ny_pos - 1, nx_pos])
                        and gvref + discont >= int(img[ny_pos + 1, nx_pos])
                        and gvref + discont >= int(img[ny_pos, nx_pos - 1])
                        and gvref + discont >= int(img[ny_pos, nx_pos + 1])
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
            peak.sumg += int(gv)
            peak.x += float(j) * gv
            peak.y += float(i) * gv

            peak.xmin = min(peak.xmin, j)
            peak.xmax = max(peak.xmax, j)
            peak.ymin = min(peak.ymin, i)
            peak.ymax = max(peak.ymax, i)

            # Check 8-neighbors for touches
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < imy and 0 <= nj < imx:
                    neighbor_label = label_img[ni, nj]
                    check_touch(peak, label, neighbor_label)

    # ---- Pass 3: Reunification test ----
    for i, peak_i in enumerate(peaks):
        if peak_i.n_touch == 0 or peak_i.unr != 0:
            continue

        x1 = peak_i.x / peak_i.sumg
        y1 = peak_i.y / peak_i.sumg
        gv1 = img.flat[peak_i.pos]

        for j_idx in peak_i.touch:
            p2 = j_idx - 1
            if p2 < 0 or p2 >= len(peaks) or peaks[p2].unr != 0:
                continue

            peak_j = peaks[p2]
            x2 = peak_j.x / peak_j.sumg
            y2 = peak_j.y / peak_j.sumg
            gv2 = img.flat[peak_j.pos]

            s12 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Profile criterion
            unify = s12 < 2.0
            if not unify:
                unify = True
                for l in range(1, int(s12)):
                    intx1 = int(x1 + l * (x2 - x1) / s12)
                    inty1 = int(y1 + l * (y2 - y1) / s12)
                    
                    # Ensure indices are within bounds
                    if 0 <= inty1 < imy and 0 <= intx1 < imx:
                        gv = int(img[inty1, intx1]) + discont
                        if (
                            gv < gv1 + l * (gv2 - gv1) / s12
                            or gv < gv1
                            or gv < gv2
                        ):
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
            peak_j.xmin = min(peak_j.xmin, peak_i.xmin)
            peak_j.ymin = min(peak_j.ymin, peak_i.ymin)
            peak_j.xmax = max(peak_j.xmax, peak_i.xmax)
            peak_j.ymax = max(peak_j.ymax, peak_i.ymax)

    # ---- Pass 4: Output targets ----
    targets = []
    for i, peak in enumerate(peaks):
        # Skip if unified into another
        if peak.unr != 0:
            continue

        # Check border touching
        width = xmax - xmin
        if width > 32:
            if peak.xmin == xmin or peak.ymin == ymin:
                continue
            if peak.xmax == xmax - 1 or peak.ymax == ymax - 1:
                continue

        # Acceptance criteria
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
