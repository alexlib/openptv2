"""Epipolar geometry for multi-camera correspondence matching.

Translation of lib/src/epi.c and lib/include/epi.h.

Computes epipolar lines and candidate matching between cameras.
"""

import cython
import numpy as np

if cython.compiled:
    from cython.cimports.libc.math import sqrt as c_sqrt
else:
    from math import sqrt as c_sqrt
from dataclasses import dataclass

MAXCAND: cython.int = 200


@cython.cclass
@dataclass
class Candidate:
    pnr: cython.int = cython.declare(cython.int, 0, visibility="public")
    tol: cython.double = cython.declare(cython.double, 0.0, visibility="public")
    corr: cython.double = cython.declare(cython.double, 0.0, visibility="public")


@cython.cclass
@dataclass
class Coord2d:
    pnr: cython.int = cython.declare(cython.int, 0, visibility="public")
    x: cython.double = cython.declare(cython.double, 0.0, visibility="public")
    y: cython.double = cython.declare(cython.double, 0.0, visibility="public")


@cython.ccall
@cython.locals(
    xp=cython.double,
    yp=cython.double,
    xf=cython.double,
    yf=cython.double,
    i=cython.int,
    Z=cython.double,
    xm=cython.double,
    ym=cython.double,
    n1=cython.double,
    n2_0=cython.double,
    n3=cython.double,
    d0=cython.double,
)
def epipolar_curve(
    image_point, origin_cal, project_cal, num_points: cython.int, cpar, vpar
) -> object:
    """Generate points along the epipolar line projected into a second camera."""
    from .trafo import pixel_to_metric, metric_to_pixel, dist_to_flat
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray
    from .imgcoord import img_coord

    # Extract multimedia parameters once — local doubles compile to C registers
    n1 = cpar.mm.n1
    n2_0 = cpar.mm.n2[0]
    n3 = cpar.mm.n3
    d0 = cpar.mm.d[0]

    xp, yp = pixel_to_metric(image_point[0], image_point[1], cpar)

    ap = origin_cal.added_par
    xf, yf = dist_to_flat(
        xp,
        yp,
        origin_cal.int_par.xh,
        origin_cal.int_par.yh,
        ap.k1,
        ap.k2,
        ap.k3,
        ap.p1,
        ap.p2,
        ap.scx,
        ap.she,
    )

    pos, direct = ray_tracing(
        xf,
        yf,
        origin_cal.ext_par.dm,
        origin_cal.ext_par.x0,
        origin_cal.ext_par.y0,
        origin_cal.ext_par.z0,
        origin_cal.int_par.cc,
        origin_cal.glass_par.vec_x,
        origin_cal.glass_par.vec_y,
        origin_cal.glass_par.vec_z,
        n1,
        n2_0,
        n3,
        d0,
    )

    line_points = np.empty((num_points, 2))
    zvals = np.linspace(vpar.Zmin_lay[0], vpar.Zmax_lay[0], num_points)

    i: cython.int
    for i in range(num_points):
        Z = zvals[i]
        pt3d = move_along_ray(Z, pos, direct)
        xm, ym = img_coord(pt3d, project_cal, cpar.mm)
        line_points[i, 0], line_points[i, 1] = metric_to_pixel(xm, ym, cpar)

    return line_points


@cython.ccall
@cython.locals(
    xl=cython.double,
    yl=cython.double,
    Zmin=cython.double,
    Zmax=cython.double,
    xmin=cython.double,
    ymin=cython.double,
    xmax=cython.double,
    ymax=cython.double,
    n1=cython.double,
    n2_0=cython.double,
    n3=cython.double,
    d0=cython.double,
)
def epi_mm(xl: cython.double, yl: cython.double, cal1, cal2, mmp, vpar):
    """Compute epipolar line endpoints in second camera."""
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray
    from .imgcoord import flat_image_coord

    n1 = mmp.n1
    n2_0 = mmp.n2[0]
    n3 = mmp.n3
    d0 = mmp.d[0]

    pos, v = ray_tracing(
        xl,
        yl,
        cal1.ext_par.dm,
        cal1.ext_par.x0,
        cal1.ext_par.y0,
        cal1.ext_par.z0,
        cal1.int_par.cc,
        cal1.glass_par.vec_x,
        cal1.glass_par.vec_y,
        cal1.glass_par.vec_z,
        n1,
        n2_0,
        n3,
        d0,
    )

    Zmin_lay0: cython.double = vpar.Zmin_lay[0]
    Zmin_lay1: cython.double = vpar.Zmin_lay[1]
    Zmax_lay0: cython.double = vpar.Zmax_lay[0]
    Zmax_lay1: cython.double = vpar.Zmax_lay[1]
    X_lay0: cython.double = vpar.X_lay[0]
    X_lay1: cython.double = vpar.X_lay[1]
    Zmin = Zmin_lay0 + (pos[0] - X_lay0) * (Zmin_lay1 - Zmin_lay0) / (X_lay1 - X_lay0)

    Zmax = Zmax_lay0 + (pos[0] - X_lay0) * (Zmax_lay1 - Zmax_lay0) / (X_lay1 - X_lay0)

    X_at_Zmin = move_along_ray(Zmin, pos, v)
    xmin, ymin = flat_image_coord(
        X_at_Zmin,
        cal2.ext_par.x0,
        cal2.ext_par.y0,
        cal2.ext_par.z0,
        cal2.ext_par.dm,
        cal2.int_par.cc,
        cal2.glass_par.vec_x,
        cal2.glass_par.vec_y,
        cal2.glass_par.vec_z,
        n1,
        n2_0,
        n3,
        d0,
    )

    X_at_Zmax = move_along_ray(Zmax, pos, v)
    xmax, ymax = flat_image_coord(
        X_at_Zmax,
        cal2.ext_par.x0,
        cal2.ext_par.y0,
        cal2.ext_par.z0,
        cal2.ext_par.dm,
        cal2.int_par.cc,
        cal2.glass_par.vec_x,
        cal2.glass_par.vec_y,
        cal2.glass_par.vec_z,
        n1,
        n2_0,
        n3,
        d0,
    )

    return xmin, ymin, xmax, ymax


@cython.ccall
@cython.locals(
    xl=cython.double,
    yl=cython.double,
    Zmin=cython.double,
    Zmax=cython.double,
    n1=cython.double,
    n2_0=cython.double,
    n3=cython.double,
    d0=cython.double,
)
def epi_mm_2d(xl: cython.double, yl: cython.double, cal, mmp, vpar) -> object:
    """Compute 3D position for single-camera multimedia case."""
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray

    n1 = mmp.n1
    n2_0 = mmp.n2[0]
    n3 = mmp.n3
    d0 = mmp.d[0]

    pos, v = ray_tracing(
        xl,
        yl,
        cal.ext_par.dm,
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        cal.int_par.cc,
        cal.glass_par.vec_x,
        cal.glass_par.vec_y,
        cal.glass_par.vec_z,
        n1,
        n2_0,
        n3,
        d0,
    )

    Zmin = vpar.Zmin_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmin_lay[1] - vpar.Zmin_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    Zmax = vpar.Zmax_lay[0] + (pos[0] - vpar.X_lay[0]) * (
        vpar.Zmax_lay[1] - vpar.Zmax_lay[0]
    ) / (vpar.X_lay[1] - vpar.X_lay[0])

    return move_along_ray(0.5 * (Zmin + Zmax), pos, v)


@cython.ccall
def _quality_ratio(a: cython.double, b: cython.double) -> cython.double:
    if a < b:
        return a / b
    return b / a


# ---------------------------------------------------------------------------
# Candidate search — hot path for epipolar matching.
#
# Instead of appending Python Candidate objects to a list, we write directly
# into pre-allocated output arrays (cand_pnr, cand_tol, cand_corr).  The
# crd parameter is typed as list[Coord2d] so that crd[j].x compiles to a
# single C field load.
# ---------------------------------------------------------------------------


@cython.ccall
@cython.locals(
    num=cython.int,
    xa=cython.double,
    ya=cython.double,
    xb=cython.double,
    yb=cython.double,
    n=cython.int,
    nx=cython.int,
    ny=cython.int,
    sumg=cython.int,
    tol_band_width=cython.double,
    count=cython.int,
    xmin=cython.double,
    xmax=cython.double,
    ymin=cython.double,
    ymax=cython.double,
    m=cython.double,
    b=cython.double,
    j0=cython.int,
    dj=cython.int,
    j=cython.int,
    p2=cython.int,
    d=cython.double,
    qn=cython.double,
    qnx=cython.double,
    qny=cython.double,
    qsumg=cython.double,
    corr=cython.double,
    k1=cython.double,
    k2=cython.double,
    k3=cython.double,
    p1=cython.double,
    p2d=cython.double,
    scx=cython.double,
    she=cython.double,
    xh=cython.double,
    yh=cython.double,
)
def find_candidate(
    crd: list,  # list[Coord2d] — typed in docstring, Cython Pure Python doesn't support generic list[T]
    pix,
    num: cython.int,
    xa: cython.double,
    ya: cython.double,
    xb: cython.double,
    yb: cython.double,
    n: cython.int,
    nx: cython.int,
    ny: cython.int,
    sumg: cython.int,
    cand_pnr: cython.int[:],  # output: candidate pnr indices
    cand_tol: cython.double[:],  # output: candidate tolerances
    cand_corr: cython.double[:],  # output: candidate correlations
    vpar,
    cpar,
    cal,
) -> cython.int:
    """Find candidates along epipolar line.

    Writes candidate data into pre-allocated arrays cand_pnr/cand_tol/cand_corr
    instead of appending Python objects to a list.  Returns the number of
    candidates found (≤ MAXCAND).

    Args:
        crd: x-sorted Coord2d list of detected points (flat-image coords).
        pix: Target array indexed by pnr.
        num: number of particles in image.
        xa, ya, xb, yb: endpoints of epipolar line [mm].
        n, nx, ny, sumg: typical target properties.
        cand_pnr, cand_tol, cand_corr: pre-allocated output arrays (size ≥ MAXCAND).
        vpar: VolumePar.
        cpar: ControlPar.
        cal: Calibration of the camera seeing candidates.

    Returns:
        int count — number of candidates found. -1 if epipolar line is
        outside the sensor array.
    """
    from .trafo import correct_brown_affin

    tol_band_width = vpar.eps0
    count = 0

    # Extract calibration parameters to locals (avoid attribute chain overhead)
    k1 = cal.added_par.k1
    k2 = cal.added_par.k2
    k3 = cal.added_par.k3
    p1 = cal.added_par.p1
    p2d = cal.added_par.p2
    scx = cal.added_par.scx
    she = cal.added_par.she
    xh = cal.int_par.xh
    yh = cal.int_par.yh
    pix_x = cpar.pix_x
    pix_y = cpar.pix_y
    imx = cpar.imx
    imy = cpar.imy

    xmin_val = -pix_x * imx / 2
    xmax_val = pix_x * imx / 2
    ymin_val = -pix_y * imy / 2
    ymax_val = pix_y * imy / 2
    xmin_val -= xh
    ymin_val -= yh
    xmax_val -= xh
    ymax_val -= yh

    xmin_val, ymin_val = correct_brown_affin(
        xmin_val, ymin_val, k1, k2, k3, p1, p2d, scx, she
    )
    xmax_val, ymax_val = correct_brown_affin(
        xmax_val, ymax_val, k1, k2, k3, p1, p2d, scx, she
    )

    if xa == xb:
        xb += 1e-10

    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    if xa > xb:
        xa, xb = xb, xa
    if ya > yb:
        ya, yb = yb, ya

    if xb <= xmin_val or xa >= xmax_val or yb <= ymin_val or ya >= ymax_val:
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

        d = abs((crd[j].y - m * crd[j].x - b) / c_sqrt(m * m + 1))
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

        corr = 4 * qsumg + 2 * qn + qnx + qny
        corr *= float(sumg + pix[p2].sumg)

        cand_pnr[count] = j
        cand_tol[count] = d
        cand_corr[count] = corr
        count += 1

    return count


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
