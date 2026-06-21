"""Sortgrid operations for PTV calibration.

Translation of lib/src/sortgrid.c and lib/include/sortgrid.h.

Provides:
- sortgrid: matches calibration 3D targets to detected 2D points.
- nearest_neighbour_pix: helper to find the closest pixel.
- read_sortgrid_par: reads search radius.
- read_calblock: reads 3D calibration point coordinates.
"""
import cython


import math
import numpy as np
from typing import Tuple
from pathlib import Path

from .tracking_frame_buf import Target


@cython.ccall
@cython.boundscheck(False)
def nearest_neighbour_pix(pix: list, x: float, y: float, eps: float) -> int:
    """Search for the nearest target in the image space within epsilon distance."""
    if eps < 0:
        return -999

    j: cython.Py_ssize_t
    d: cython.double
    dmin: cython.double
    pnr: cython.int

    pnr = -999
    dmin = 1e20

    xmin, xmax = x - eps, x + eps
    ymin, ymax = y - eps, y + eps

    for j, p in enumerate(pix):
        if ymin < p.y < ymax and xmin < p.x < xmax:
            d = math.sqrt((x - p.x)**2 + (y - p.y)**2)
            if d < dmin:
                dmin = d
                pnr = j

    return pnr


def _nearest_neighbour_arr(
    pix_x: np.ndarray, pix_y: np.ndarray,
    x: cython.double, y: cython.double, eps: cython.double,
) -> int:
    """Vectorized nearest-neighbour search over pre-extracted coordinate arrays."""
    if eps < 0:
        return -999

    mask = (
        (pix_x > x - eps) & (pix_x < x + eps)
        & (pix_y > y - eps) & (pix_y < y + eps)
    )
    if not mask.any():
        return -999

    dx = pix_x[mask] - x
    dy = pix_y[mask] - y
    dist_sq = dx * dx + dy * dy
    idx_in_mask = dist_sq.argmin()
    return int(np.flatnonzero(mask)[idx_in_mask])


@cython.ccall
def read_sortgrid_par(filename: str | Path) -> int:
    """Read search radius for sortgrid."""
    path = Path(filename)
    if not path.exists():
        print(f"Error: {filename} does not exist.")
        return 0
    try:
        return int(path.read_text().strip())
    except (ValueError, IndexError):
        print(f"Error reading sortgrid parameter from {filename}")
        return 0

@cython.ccall
def read_calblock(filename: str | Path) -> Tuple[np.ndarray, int]:
    """Read calibration target 3D coordinates.

    Returns:
        (fix, num_points) where fix is an (N, 3) array.
    """
    path = Path(filename)
    if not path.exists():
        print(f"Can't open calibration block file: {filename}")
        return np.empty((0, 3)), 0

    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        print(f"Empty or badly formatted file: {filename}")
        return np.empty((0, 3)), 0

    num_points = data.shape[0]
    fix = data[:, 1:4]

    return fix, num_points


def sortgrid(
    cal: "Calibration",
    cpar: "ControlPar",
    nfix: int,
    fix: np.ndarray,
    num: int,
    eps: int,
    pix: list,
) -> list:
    """Sort detected target points by back-projection.

    Returns:
        List of Target objects sorted by calibration point ID.
        Unmatched entries have pnr=-999.
    """
    from .imgcoord import img_coord
    from .trafo import metric_to_pixel

    i: cython.Py_ssize_t
    j: cython.int

    sorted_pix = [Target(pnr=-999) for _ in range(nfix)]

    imx, imy = cpar.imx, cpar.imy
    pix_size_x, pix_size_y = cpar.pix_x, cpar.pix_y
    chfield = cpar.chfield
    mm = cpar.mm
    feps = float(eps)

    use_vec = len(pix) >= 16
    if use_vec:
        pix_x = np.array([p.x for p in pix], dtype=np.float64)
        pix_y = np.array([p.y for p in pix], dtype=np.float64)
        nn_func = lambda px, py: _nearest_neighbour_arr(pix_x, pix_y, px, py, feps)
    else:
        nn_func = lambda px, py: nearest_neighbour_pix(pix, px, py, feps)

    for i in range(nfix):
        xp, yp = img_coord(fix[i], cal, mm)
        px, py = metric_to_pixel(xp, yp, imx, imy, pix_size_x, pix_size_y, chfield)

        if (px > -eps and py > -eps
                and px < imx + eps and py < imy + eps):

            j = nn_func(px, py)

            if j != -999:
                t = pix[j]
                sorted_pix[i] = Target(
                    pnr=i, x=t.x, y=t.y, n=t.n,
                    nx=t.nx, ny=t.ny, sumg=t.sumg, tnr=t.tnr,
                )

    return sorted_pix


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
