"""Streamlined target recognition and binarization."""

from collections import deque
import numpy as np
from openptv2.algorithms.tracking_frame_buf import TargetArray, Target

CORRES_NONE = -1


def _empty_target() -> Target:
    return Target(pnr=1, x=1.0, y=1.0, n=1, nx=1, ny=1, sumg=1, tnr=CORRES_NONE)


def _vectorized_targ_rec(
    img,
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
    xmax,
    ymin,
    ymax,
):
    img_u8 = np.asarray(img, dtype=np.uint8)
    imy, imx = img_u8.shape

    if xmax < 0:
        xmax = imx - 1
    if ymax < 0:
        ymax = imy - 1

    xmin = max(int(xmin), 1)
    ymin = max(int(ymin), 1)
    xmax = min(int(xmax), imx - 1)
    ymax = min(int(ymax), imy - 1)

    if xmin >= xmax or ymin >= ymax:
        return [_empty_target()]

    img0 = img_u8.copy()
    interior = img_u8[ymin:ymax, xmin:xmax]
    local_max = (
        (interior > gvthres)
        & (interior >= img_u8[ymin:ymax, xmin - 1 : xmax - 1])
        & (interior >= img_u8[ymin:ymax, xmin + 1 : xmax + 1])
        & (interior >= img_u8[ymin - 1 : ymax - 1, xmin:xmax])
        & (interior >= img_u8[ymin + 1 : ymax + 1, xmin:xmax])
        & (interior >= img_u8[ymin - 1 : ymax - 1, xmin - 1 : xmax - 1])
        & (interior >= img_u8[ymin + 1 : ymax + 1, xmin - 1 : xmax - 1])
        & (interior >= img_u8[ymin - 1 : ymax - 1, xmin + 1 : xmax + 1])
        & (interior >= img_u8[ymin + 1 : ymax + 1, xmin + 1 : xmax + 1])
    )
    peak_coords = np.argwhere(local_max)
    if peak_coords.size == 0:
        return [_empty_target()]

    targets = []
    for py, px in peak_coords:
        i = int(py + ymin)
        j = int(px + xmin)
        gv = int(img_u8[i, j])
        if int(img0[i, j]) <= gvthres:
            continue

        sumg = gv
        img0[i, j] = 0
        xa = xb = j
        ya = yb = i
        x_weighted = float(j) * float(gv - gvthres)
        y_weighted = float(i) * float(gv - gvthres)
        numpix = 1
        waitlist = deque([(j, i)])

        while waitlist:
            wx, wy = waitlist.popleft()
            gvref = int(img_u8[wy, wx])

            for xn, yn in ((wx - 1, wy), (wx + 1, wy), (wx, wy - 1), (wx, wy + 1)):
                if xn < xmin or xn >= xmax or yn < ymin or yn >= ymax:
                    continue

                gv4 = int(img0[yn, xn])
                if (
                    gv4 > gvthres
                    and gv4 <= gvref + discont
                    and gvref + discont >= int(img_u8[yn - 1, xn])
                    and gvref + discont >= int(img_u8[yn + 1, xn])
                    and gvref + discont >= int(img_u8[yn, xn - 1])
                    and gvref + discont >= int(img_u8[yn, xn + 1])
                ):
                    sumg += gv4
                    img0[yn, xn] = 0
                    xa = min(xa, xn)
                    xb = max(xb, xn)
                    ya = min(ya, yn)
                    yb = max(yb, yn)
                    x_weighted += float(xn) * float(gv4 - gvthres)
                    y_weighted += float(yn) * float(gv4 - gvthres)
                    numpix += 1
                    if numpix <= nnmax:
                        waitlist.append((xn, yn))

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
            sumg_adj = sumg - numpix * gvthres
            if sumg_adj > 0:
                x = x_weighted / float(sumg_adj) + 0.5
                y = y_weighted / float(sumg_adj) + 0.5
            else:
                x = float(j) + 0.5
                y = float(i) + 0.5

            targets.append(
                Target(
                    pnr=len(targets),
                    x=x,
                    y=y,
                    n=numpix,
                    nx=nx,
                    ny=ny,
                    sumg=sumg,
                    tnr=CORRES_NONE,
                )
            )

    return targets if targets else [_empty_target()]


def target_recognition(img, tpar, cam, cpar, subrange_x=None, subrange_y=None):
    """Recognize targets in image using segmentation."""
    imx, imy = cpar.get_image_size()

    if subrange_x is None:
        xmin, xmax = 1, imx - 1
    else:
        xmin, xmax = subrange_x

    if subrange_y is None:
        ymin, ymax = 1, imy - 1
    else:
        ymin, ymax = subrange_y

    targets = _vectorized_targ_rec(
        img=img,
        gvthres=int(tpar.get_grey_thresholds()[cam]),
        discont=int(tpar.get_max_discontinuity()),
        nnmin=int(tpar.get_pixel_count_bounds()[0]),
        nnmax=int(tpar.get_pixel_count_bounds()[1]),
        nxmin=int(tpar.get_xsize_bounds()[0]),
        nxmax=int(tpar.get_xsize_bounds()[1]),
        nymin=int(tpar.get_ysize_bounds()[0]),
        nymax=int(tpar.get_ysize_bounds()[1]),
        sumg_min=int(tpar.get_min_sum_grey()),
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    return TargetArray(targets)


__all__ = ["target_recognition"]
