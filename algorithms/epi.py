"""Epipolar geometry for multi-camera correspondence matching.

Translation of lib/src/epi.c and lib/include/epi.h.

Computes epipolar lines and candidate matching between cameras.
"""

import numpy as np
from dataclasses import dataclass

# Maximum number of candidates per search
MAXCAND = 200


@dataclass
class Candidate:
    """A candidate match from epipolar search.

    Attributes:
        pnr: particle number (index into sorted coordinate array).
        tol: distance from epipolar line.
        corr: correlation score.
    """
    pnr: int
    tol: float
    corr: float


@dataclass
class Coord2d:
    """2D coordinate with particle reference.

    Attributes:
        pnr: particle number.
        x, y: flat-image (undistorted) coordinates.
    """
    pnr: int
    x: float
    y: float


def epi_mm(
    xl: float,
    yl: float,
    cal1: dict,
    cal2: dict,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
    vpar_X_lay: tuple[float, float],
    vpar_Zmin_lay: tuple[float, float],
    vpar_Zmax_lay: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Compute epipolar line endpoints in second camera.

    Takes a point in image space of one camera and returns the
    epipolar line (in mm) projected onto the second camera.

    Args:
        xl, yl: point position in origin camera's image space [mm].
        cal1: origin camera calibration (dm, x0, y0, z0, cc, glass, etc.).
        cal2: target camera calibration.
        mm_n1, mm_n2_0, mm_n3, mm_d0: multimedia parameters.
        vpar_X_lay: (X_left, X_right) volume boundaries.
        vpar_Zmin_lay: (Zmin_left, Zmin_right).
        vpar_Zmax_lay: (Zmax_left, Zmax_right).

    Returns:
        (xmin, ymin, xmax, ymax) endpoints of epipolar line in second camera.
    """
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray
    from .imgcoord import flat_image_coord

    # Ray trace from origin camera
    pos, v = ray_tracing(
        xl, yl,
        cal1["dm"], cal1["x0"], cal1["y0"], cal1["z0"], cal1["cc"],
        cal1["gx"], cal1["gy"], cal1["gz"],
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    # Calculate Z min/max for position
    X_lay_0, X_lay_1 = vpar_X_lay
    Zmin_lay_0, Zmin_lay_1 = vpar_Zmin_lay
    Zmax_lay_0, Zmax_lay_1 = vpar_Zmax_lay

    Zmin = Zmin_lay_0 + (pos[0] - X_lay_0) * (Zmin_lay_1 - Zmin_lay_0) / (X_lay_1 - X_lay_0)
    Zmax = Zmax_lay_0 + (pos[0] - X_lay_0) * (Zmax_lay_1 - Zmax_lay_0) / (X_lay_1 - X_lay_0)

    # Project endpoints onto second camera
    X_at_Zmin = move_along_ray(Zmin, pos, v)
    xmin, ymin = flat_image_coord(
        X_at_Zmin,
        cal2["x0"], cal2["y0"], cal2["z0"], cal2["dm"], cal2["cc"],
        cal2["gx"], cal2["gy"], cal2["gz"],
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    X_at_Zmax = move_along_ray(Zmax, pos, v)
    xmax, ymax = flat_image_coord(
        X_at_Zmax,
        cal2["x0"], cal2["y0"], cal2["z0"], cal2["dm"], cal2["cc"],
        cal2["gx"], cal2["gy"], cal2["gz"],
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    return xmin, ymin, xmax, ymax


def epi_mm_2d(
    xl: float,
    yl: float,
    cal: dict,
    mm_n1: float,
    mm_n2_0: float,
    mm_n3: float,
    mm_d0: float,
    vpar_X_lay: tuple[float, float],
    vpar_Zmin_lay: tuple[float, float],
    vpar_Zmax_lay: tuple[float, float],
) -> np.ndarray:
    """Compute 3D position for single-camera multimedia case.

    Ray traces through multimedia to the mid-plane between Zmin and Zmax.

    Args:
        xl, yl: point in camera image space [mm].
        cal: camera calibration.
        mm_n1, mm_n2_0, mm_n3, mm_d0: multimedia parameters.
        vpar_X_lay: (X_left, X_right).
        vpar_Zmin_lay: (Zmin_left, Zmin_right).
        vpar_Zmax_lay: (Zmax_left, Zmax_right).

    Returns:
        3D position at mid-plane.
    """
    from .ray_tracing import ray_tracing
    from .multimed import move_along_ray

    pos, v = ray_tracing(
        xl, yl,
        cal["dm"], cal["x0"], cal["y0"], cal["z0"], cal["cc"],
        cal["gx"], cal["gy"], cal["gz"],
        mm_n1, mm_n2_0, mm_n3, mm_d0,
    )

    X_lay_0, X_lay_1 = vpar_X_lay
    Zmin_lay_0, Zmin_lay_1 = vpar_Zmin_lay
    Zmax_lay_0, Zmax_lay_1 = vpar_Zmax_lay

    Zmin = Zmin_lay_0 + (pos[0] - X_lay_0) * (Zmin_lay_1 - Zmin_lay_0) / (X_lay_1 - X_lay_0)
    Zmax = Zmax_lay_0 + (pos[0] - X_lay_0) * (Zmax_lay_1 - Zmax_lay_0) / (X_lay_1 - X_lay_0)

    return move_along_ray(0.5 * (Zmin + Zmax), pos, v)


def _quality_ratio(a: int, b: int) -> float:
    """Compute quality ratio between two values."""
    if a < b:
        return a / b
    else:
        return b / a


def find_candidate(
    crd: list[Coord2d],
    pix: list[dict],
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    n: int,
    nx: int,
    ny: int,
    sumg: int,
    vpar_eps0: float,
    vpar_cn: float,
    vpar_cnx: float,
    vpar_cny: float,
    vpar_csumg: float,
    cpar_imx: int,
    cpar_imy: int,
    cpar_pix_x: float,
    cpar_pix_y: float,
    cal_int_xh: float,
    cal_int_yh: float,
    cal_k1: float,
    cal_k2: float,
    cal_k3: float,
    cal_p1: float,
    cal_p2: float,
    cal_scx: float,
    cal_she: float,
) -> list[Candidate]:
    """Find candidates along epipolar line using binary search.

    Searches in x-sorted coordinate array for candidates near the
    epipolar line, exploiting shape information.

    Args:
        crd: x-sorted array of detected points (flat-image coords).
        pix: target information (size, grey value) indexed by pnr.
        xa, ya, xb, yb: endpoints of epipolar line [mm].
        n, nx, ny: typical target size parameters.
        sumg: typical sum of grey values.
        vpar_eps0: tolerance band width.
        vpar_cn, vpar_cnx, vpar_cny, vpar_csumg: minimum quality thresholds.
        cpar_imx, cpar_imy: image dimensions.
        cpar_pix_x, cpar_pix_y: pixel sizes.
        cal_int_xh, cal_int_yh: principal point.
        cal_k1, cal_k2, cal_k3, cal_p1, cal_p2, cal_scx, cal_she: distortion.

    Returns:
        List of candidates. Negative count means epipolar line out of sensor.
    """
    from .trafo import correct_brown_affin

    tol_band_width = vpar_eps0
    num = len(crd)

    # Define sensor bounds
    xmin = -cpar_pix_x * cpar_imx / 2
    xmax = cpar_pix_x * cpar_imx / 2
    ymin = -cpar_pix_y * cpar_imy / 2
    ymax = cpar_pix_y * cpar_imy / 2
    xmin -= cal_int_xh
    ymin -= cal_int_yh
    xmax -= cal_int_xh
    ymax -= cal_int_yh

    # Correct bounds for distortion
    xmin, ymin = correct_brown_affin(xmin, ymin, cal_k1, cal_k2, cal_k3, cal_p1, cal_p2, cal_scx, cal_she)
    xmax, ymax = correct_brown_affin(xmax, ymax, cal_k1, cal_k2, cal_k3, cal_p1, cal_p2, cal_scx, cal_she)

    # Handle vertical line case
    if xa == xb:
        xb += 1e-10

    # Line equation: y = m*x + b
    m = (yb - ya) / (xb - xa)
    b = ya - m * xa

    # Ensure xa <= xb, ya <= yb
    if xa > xb:
        xa, xb = xb, xa
    if ya > yb:
        ya, yb = yb, ya

    # Check if epipolar line is outside sensor
    if xb <= xmin or xa >= xmax or yb <= ymin or ya >= ymax:
        return []

    # Binary search for start point
    j0 = num // 2
    dj = num // 4
    while dj > 1:
        if crd[j0].x < (xa - tol_band_width):
            j0 += dj
        else:
            j0 -= dj
        dj //= 2

    # Shift back for truncation safety
    j0 -= 12
    if j0 < 0:
        j0 = 0

    candidates = []

    for j in range(j0, num):
        # X-sorted: out of bounds means past last candidate
        if crd[j].x > xb + tol_band_width:
            return candidates

        # Check Y bounds
        if crd[j].y <= ya - tol_band_width or crd[j].y >= yb + tol_band_width:
            continue
        # Check X bounds
        if crd[j].x <= xa - tol_band_width or crd[j].x >= xb + tol_band_width:
            continue

        # Distance from epipolar line
        d = abs(crd[j].y - m * crd[j].x - b) / np.sqrt(m * m + 1)
        if d >= tol_band_width:
            continue

        p2 = crd[j].pnr
        if p2 >= num:
            return []  # Invalid pnr

        # Quality ratios
        qn = _quality_ratio(n, pix[p2]["n"])
        qnx = _quality_ratio(nx, pix[p2]["nx"])
        qny = _quality_ratio(ny, pix[p2]["ny"])
        qsumg = _quality_ratio(sumg, pix[p2]["sumg"])

        # Check minimum quality
        if qn < vpar_cn or qnx < vpar_cnx or qny < vpar_cny or qsumg <= vpar_csumg:
            continue

        # Max candidates check
        if len(candidates) >= MAXCAND:
            return candidates

        # Correlation score
        corr = 4 * qsumg + 2 * qn + qnx + qny
        corr *= (sumg + pix[p2]["sumg"])

        candidates.append(Candidate(pnr=j, tol=d, corr=corr))

    return candidates
