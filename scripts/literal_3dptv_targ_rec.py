"""Literal, unoptimized Python transliteration of 3dptv.exe's
segmentation.c::targ_rec (the real C source, read from 3dptv/src_c on disk
during the 2026-08-27 investigation). Deliberately naive and slow -- this is
a verification ORACLE for comparing against openptv2's compiled
targ_rec_fast, not production code. Kept line-for-line faithful to the C,
including its variable roles (img = mutable, zeroed as pixels are consumed;
img0 = read-only original), even where that reads as unusual, so behavior
divergences are attributable to a genuine algorithm difference rather than a
transliteration slip.

See docs/plans/2026-08-27-verified-pipeline-ghost-particle-study-plan.md,
Phase 1 step 4 (revised approach).
"""
from __future__ import annotations

import numpy as np


def literal_3dptv_targ_rec(
    image: np.ndarray,
    gvthres: int,
    disco: int,
    nnmin: int,
    nnmax: int,
    nxmin: int,
    nxmax: int,
    nymin: int,
    nymax: int,
    sumg_min: int,
    xmin: int = 1,
    xmax: int | None = None,
    ymin: int = 1,
    ymax: int | None = None,
):
    """Direct port of segmentation.c's targ_rec loop body (lines ~136-234).

    Returns a list of (x, y, n, nx, ny, sumg) tuples, x/y in the same
    sub-pixel convention as the C (+0.5 offset already applied).
    """
    imy, imx = image.shape
    if xmax is None:
        xmax = imx - 1
    if ymax is None:
        ymax = imy - 1

    img = image.astype(np.int32).copy()  # mutable, zeroed as consumed (C's `img`)
    img0 = image.astype(np.int32).copy()  # read-only original (C's `img0`)

    thres = gvthres
    targets = []

    for i in range(ymin + 1, ymax - 1):
        for j in range(xmin + 1, xmax - 1):
            gv = int(img[i, j])
            if not (
                gv > thres
                and gv >= img[i, j - 1]
                and gv >= img[i, j + 1]
                and gv >= img[i - 1, j]
                and gv >= img[i + 1, j]
                and gv >= img[i - 1, j - 1]
                and gv >= img[i + 1, j - 1]
                and gv >= img[i - 1, j + 1]
                and gv >= img[i + 1, j + 1]
            ):
                continue

            # local maximum -> new peak
            yn, xn = i, j
            sumg = gv
            img[i, j] = 0
            xa = xb = xn
            ya = yb = yn
            gv_rel = gv - thres
            x_acc = xn * gv_rel
            y_acc = yn * gv_rel
            numpix = 1

            waitlist = [(j, i)]
            head = 0
            while head < len(waitlist):
                wx, wy = waitlist[head]
                head += 1
                gvref = int(img0[wy, wx])

                for xn4, yn4 in (
                    (wx - 1, wy),
                    (wx + 1, wy),
                    (wx, wy - 1),
                    (wx, wy + 1),
                ):
                    if not (0 <= xn4 < imx and 0 <= yn4 < imy):
                        continue
                    gv4 = int(img[yn4, xn4])

                    if (
                        gv4 > thres
                        and xn4 >= xmin and xn4 < xmax - 1
                        and yn4 >= ymin and yn4 < ymax - 1
                        and gv4 <= gvref + disco
                        and gvref + disco >= img0[yn4 - 1, xn4]
                        and gvref + disco >= img0[yn4 + 1, xn4]
                        and gvref + disco >= img0[yn4, xn4 - 1]
                        and gvref + disco >= img0[yn4, xn4 + 1]
                    ):
                        sumg += gv4
                        img[yn4, xn4] = 0
                        if xn4 < xa:
                            xa = xn4
                        if xn4 > xb:
                            xb = xn4
                        if yn4 < ya:
                            ya = yn4
                        if yn4 > yb:
                            yb = yn4
                        x_acc += xn4 * (gv4 - thres)
                        y_acc += yn4 * (gv4 - thres)
                        numpix += 1
                        waitlist.append((xn4, yn4))

            if xa == xmin or ya == ymin or xb == xmax - 1 or yb == ymax - 1:
                continue

            nx = xb - xa + 1
            ny = yb - ya + 1

            if (
                numpix >= nnmin and numpix <= nnmax
                and nx >= nxmin and nx <= nxmax
                and ny >= nymin and ny <= nymax
                and sumg > sumg_min
            ):
                sumg_for_centroid = sumg - numpix * thres
                x = x_acc / sumg_for_centroid + 0.5
                y = y_acc / sumg_for_centroid + 0.5
                targets.append((x, y, numpix, nx, ny, sumg))

    return targets
