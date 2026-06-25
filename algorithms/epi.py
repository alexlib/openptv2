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


@cython.ccall
@cython.locals(
    xp=cython.double, yp=cython.double,
    xf=cython.double, yf=cython.double,
    i=cython.int, Z=cython.double,
    xm=cython.double, ym=cython.double
)
def epipolar_curve(image_point, origin_cal, project_cal, num_points: cython.int, cpar, vpar) -> np.ndarray:
    """Generate points along the epipolar line projected into a second camera.

    Takes a distorted pixel coordinate in one camera and produces the
    epipolar curve as seen in a second camera, sampled at num_points
    evenly-spaced Z values between the observed volume's Zmin and Zmax.

    Args:
        image_point: (2,) array, distorted pixel coordinates in origin camera.
        origin_cal: Calibration of the camera seeing the point.
        project_cal: Calibration of the camera onto which the line is projected.
        num_points: number of samples along the line (minimum 2 for endpoints).
        cpar: ControlPar with image size and multimedia parameters.
        vpar: VolumePar with observed volume Z limits.

    Returns:
        (num_points, 2) array of pixel coordinates in the projection camera.
    """
    from .trafo import pixel_to_metric, metric_to_pixel, dist_to_flat
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray
    from .imgcoord import img_coord

    xp, yp = pixel_to_metric(image_point[0], image_point[1], cpar)

    ap = origin_cal.added_par
    xf, yf = dist_to_flat(
        xp, yp,
        origin_cal.int_par.xh, origin_cal.int_par.yh,
        ap.k1, ap.k2, ap.k3, ap.p1, ap.p2, ap.scx, ap.she,
    )

    pos, direct = ray_tracing(
        xf, yf,
        origin_cal.ext_par.dm,
        origin_cal.ext_par.x0, origin_cal.ext_par.y0, origin_cal.ext_par.z0,
        origin_cal.int_par.cc,
        origin_cal.glass_par.vec_x, origin_cal.glass_par.vec_y,
        origin_cal.glass_par.vec_z,
        cpar.mm.n1, cpar.mm.n2[0], cpar.mm.n3, cpar.mm.d[0],
    )

    line_points = np.empty((num_points, 2))
    for i, Z in enumerate(np.linspace(vpar.Zmin_lay[0], vpar.Zmax_lay[0],
                                      num_points)):
        pt3d = move_along_ray(Z, pos, direct)
        xm, ym = img_coord(pt3d, project_cal, cpar.mm)
        line_points[i, 0], line_points[i, 1] = metric_to_pixel(xm, ym, cpar)

    return line_points


@cython.ccall
@cython.locals(
    xl=cython.double, yl=cython.double,
    Zmin=cython.double, Zmax=cython.double,
    xmin=cython.double, ymin=cython.double,
    xmax=cython.double, ymax=cython.double
)
def epi_mm(xl: cython.double, yl: cython.double, cal1, cal2, mmp, vpar):
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


@cython.ccall
@cython.locals(
    xl=cython.double, yl=cython.double,
    Zmin=cython.double, Zmax=cython.double
)
def epi_mm_2d(xl: cython.double, yl: cython.double, cal, mmp, vpar) -> np.ndarray:
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


@cython.ccall
def _quality_ratio(a: cython.double, b: cython.double) -> cython.double:
    if a < b:
        return a / b
    return b / a


@cython.ccall
@cython.locals(
    num=cython.int, xa=cython.double, ya=cython.double, xb=cython.double, yb=cython.double,
    n=cython.int, nx=cython.int, ny=cython.int, sumg=cython.int,
    tol_band_width=cython.double, count=cython.int,
    xmin=cython.double, xmax=cython.double, ymin=cython.double, ymax=cython.double,
    m=cython.double, b=cython.double, temp=cython.double,
    j0=cython.int, dj=cython.int, j=cython.int, p2=cython.int,
    d=cython.double, qn=cython.double, qnx=cython.double, qny=cython.double, qsumg=cython.double,
    corr=cython.double
)
def find_candidate(crd, pix, num: cython.int, xa: cython.double, ya: cython.double,
                   xb: cython.double, yb: cython.double, n: cython.int, nx: cython.int,
                   ny: cython.int, sumg: cython.int, cand_out, vpar, cpar, cal) -> cython.int:
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

        corr = (4 * qsumg + 2 * qn + qnx + qny)
        corr *= float(sumg + pix[p2].sumg)

        cand_out.append(Candidate(pnr=j, tol=d, corr=corr))
        count += 1

    return count


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
