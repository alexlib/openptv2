"""Epipolar geometry for multi-camera correspondence matching.

Translation of lib/src/epi.c and lib/include/epi.h.

Computes epipolar lines and candidate matching between cameras.
"""

import numpy as np
from dataclasses import dataclass

MAXCAND = 200


@dataclass
class Candidate:
    pnr: int
    tol: float
    corr: float


@dataclass
class Coord2d:
    pnr: int
    x: float
    y: float


def epi_mm(xl, yl, cal1, cal2, mmp, vpar):
    """Compute epipolar line endpoints in second camera.

    Args:
        xl, yl: point position in origin camera's image space [mm].
        cal1: origin camera Calibration object.
        cal2: target camera Calibration object.
        mmp: MmNp multimedia parameters (n1, n2, d, n3).
        vpar: VolumePar volume parameters (X_lay, Zmin_lay, Zmax_lay).

    Returns:
        (xmin, ymin, xmax, ymax) endpoints of epipolar line in second camera.
    """
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray
    from .imgcoord import flat_image_coord

    pos, v = ray_tracing(
        xl, yl,
        cal1.ext_par.dm, cal1.ext_par.x0, cal1.ext_par.y0, cal1.ext_par.z0,
        cal1.int_par.cc,
        cal1.glass_par.vec_x, cal1.glass_par.vec_y, cal1.glass_par.vec_z,
        mmp.n1, mmp.n2[0], mmp.n3, mmp.d[0],
    )

    Zmin = (vpar.Zmin_lay[0]
        + (pos[0] - vpar.X_lay[0])
        * (vpar.Zmin_lay[1] - vpar.Zmin_lay[0])
        / (vpar.X_lay[1] - vpar.X_lay[0]))

    Zmax = (vpar.Zmax_lay[0]
        + (pos[0] - vpar.X_lay[0])
        * (vpar.Zmax_lay[1] - vpar.Zmax_lay[0])
        / (vpar.X_lay[1] - vpar.X_lay[0]))

    X_at_Zmin = move_along_ray(Zmin, pos, v)
    xmin, ymin = flat_image_coord(
        X_at_Zmin,
        cal2.ext_par.x0, cal2.ext_par.y0, cal2.ext_par.z0,
        cal2.ext_par.dm, cal2.int_par.cc,
        cal2.glass_par.vec_x, cal2.glass_par.vec_y, cal2.glass_par.vec_z,
        mmp.n1, mmp.n2[0], mmp.n3, mmp.d[0],
    )

    X_at_Zmax = move_along_ray(Zmax, pos, v)
    xmax, ymax = flat_image_coord(
        X_at_Zmax,
        cal2.ext_par.x0, cal2.ext_par.y0, cal2.ext_par.z0,
        cal2.ext_par.dm, cal2.int_par.cc,
        cal2.glass_par.vec_x, cal2.glass_par.vec_y, cal2.glass_par.vec_z,
        mmp.n1, mmp.n2[0], mmp.n3, mmp.d[0],
    )

    return xmin, ymin, xmax, ymax


def epi_mm_2d(xl, yl, cal, mmp, vpar):
    """Compute 3D position for single-camera multimedia case."""
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray

    pos, v = ray_tracing(
        xl, yl,
        cal.ext_par.dm, cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
        cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mmp.n1, mmp.n2[0], mmp.n3, mmp.d[0],
    )

    Zmin = (vpar.Zmin_lay[0]
        + (pos[0] - vpar.X_lay[0])
        * (vpar.Zmin_lay[1] - vpar.Zmin_lay[0])
        / (vpar.X_lay[1] - vpar.X_lay[0]))

    Zmax = (vpar.Zmax_lay[0]
        + (pos[0] - vpar.X_lay[0])
        * (vpar.Zmax_lay[1] - vpar.Zmax_lay[0])
        / (vpar.X_lay[1] - vpar.X_lay[0]))

    return move_along_ray(0.5 * (Zmin + Zmax), pos, v)


def _quality_ratio(a, b):
    if a < b:
        return float(a) / float(b)
    return float(b) / float(a)


def find_candidate(crd, pix, num, xa, ya, xb, yb, n, nx, ny, sumg,
                   cand_out, vpar, cpar, cal):
    """Find candidates along epipolar line.

    Matches C find_candidate exactly.

    Args:
        crd: x-sorted Coord2d array of detected points (flat-image coords).
        pix: Target array indexed by pnr.
        num: number of particles in image.
        xa, ya, xb, yb: endpoints of epipolar line [mm].
        n, nx, ny, sumg: typical target properties.
        cand_out: output list of Candidate (appended to).
        vpar: VolumePar.
        cpar: ControlPar.
        cal: Calibration of the camera seeing candidates.

    Returns:
        int count - number of candidates found. Negative if epipolar line
        is outside sensor array.
    """
    from .trafo import correct_brown_affin

    tol_band_width = vpar.eps0
    count = 0

    xmin = -cpar.pix_x * cpar.imx / 2
    xmax = cpar.pix_x * cpar.imx / 2
    ymin = -cpar.pix_y * cpar.imy / 2
    ymax = cpar.pix_y * cpar.imy / 2
    xmin -= cal.int_par.xh
    ymin -= cal.int_par.yh
    xmax -= cal.int_par.xh
    ymax -= cal.int_par.yh

    ap = cal.added_par
    xmin, ymin = correct_brown_affin(xmin, ymin, ap.k1, ap.k2, ap.k3,
                                     ap.p1, ap.p2, ap.scx, ap.she)
    xmax, ymax = correct_brown_affin(xmax, ymax, ap.k1, ap.k2, ap.k3,
                                     ap.p1, ap.p2, ap.scx, ap.she)

    if xa == xb:
        xb += 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    if xa > xb:
        xa, xb = xb, xa
    if ya > yb:
        ya, yb = yb, ya

    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        return -1

    j0 = num // 2
    dj = num // 4
    while dj > 1:
        if crd[j0].x < (xa - tol_band_width):
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    j0 -= 12
    if j0 < 0:
        j0 = 0

    for j in range(j0, num):
        if crd[j].x > xb + tol_band_width:
            return count

        if crd[j].y <= ya - tol_band_width or crd[j].y >= yb + tol_band_width:
            continue
        if crd[j].x <= xa - tol_band_width or crd[j].x >= xb + tol_band_width:
            continue

        d = abs((crd[j].y - m * crd[j].x - b) / np.sqrt(m * m + 1))
        if d >= tol_band_width:
            continue

        p2 = crd[j].pnr
        if p2 >= num:
            return -1

        qn = _quality_ratio(n, pix[p2].n)
        qnx = _quality_ratio(nx, pix[p2].nx)
        qny = _quality_ratio(ny, pix[p2].ny)
        qsumg = _quality_ratio(sumg, pix[p2].sumg)

        if qn < vpar.cn or qnx < vpar.cnx or qny < vpar.cny or qsumg <= vpar.csumg:
            continue
        if count >= MAXCAND:
            return count

        corr = (4 * qsumg + 2 * qn + qnx + qny)
        corr *= float(sumg + pix[p2].sumg)

        cand_out.append(Candidate(pnr=j, tol=d, corr=corr))
        count += 1

    return count
