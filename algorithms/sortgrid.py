"""Sortgrid operations for PTV calibration.

Translation of lib/src/sortgrid.c and lib/include/sortgrid.h.

Provides:
- sortgrid: matches calibration 3D targets to detected 2D points.
- nearest_neighbour_pix: helper to find the closest pixel.
- read_sortgrid_par: reads search radius.
- read_calblock: reads 3D calibration point coordinates.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

# Placeholder for Target struct until tracking_frame_buf is translated
# In actual code, this would be imported from the correct module
# from .tracking_frame_buf import Target 

def nearest_neighbour_pix(pix: list, x: float, y: float, eps: float) -> int:
    """Search for the nearest target in the image space within epsilon distance.

    Args:
        pix: List of target objects (must have .x and .y attributes).
        x, y: Search coordinates.
        eps: Search radius.

    Returns:
        Index of the nearest target or -999 if none found.
    """
    if eps < 0:
        return -999
        
    pnr = -999
    dmin = 1e20
    
    xmin, xmax = x - eps, x + eps
    ymin, ymax = y - eps, y + eps

    for j, p in enumerate(pix):
        if ymin < p.y < ymax and xmin < p.x < xmax:
            d = np.sqrt((x - p.x)**2 + (y - p.y)**2)
            if d < dmin:
                dmin = d
                pnr = j
                
    return pnr

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

def read_calblock(filename: str | Path) -> Tuple[np.ndarray, int]:
    """Read calibration target 3D coordinates.

    Returns:
        (fix, num_points) where fix is an (N, 3) array.
    """
    path = Path(filename)
    if not path.exists():
        print(f"Can't open calibration block file: {filename}")
        return np.empty((0, 3)), 0
        
    data = np.loadtxt(path)
    if data.size == 0:
        print(f"Empty or badly formatted file: {filename}")
        return np.empty((0, 3)), 0
        
    # Calibration target points are indices (ignore), x, y, z
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

    Args:
        cal: Calibration object.
        cpar: Control parameters.
        nfix: Number of points in calibration file.
        fix: (N, 3) array of calibration target 3D positions.
        num: Number of detected dots.
        eps: Search radius in pixels.
        pix: List of detected targets.

    Returns:
        List of targets sorted by their ID (pnr), unassigned marked with pnr=-999.
    """
    from .imgcoord import img_coord
    from .trafo import metric_to_pixel

    sorted_pix = [None] * nfix
    # Initialize with placeholder
    for i in range(nfix):
        # We need a way to represent an unassigned target
        # Using a dummy Target if needed, or keeping it as None
        sorted_pix[i] = None

    for i in range(nfix):
        xp, yp = img_coord(fix[i], cal, cpar.mm)
        px, py = metric_to_pixel(xp, yp, cpar.imx, cpar.imy, cpar.pix_x, cpar.pix_y, cpar.chfield)
        
        if (px > -eps and py > -eps and 
            px < cpar.imx + eps and py < cpar.imy + eps):
            
            j = nearest_neighbour_pix(pix, px, py, float(eps))
            
            if j != -999:
                sorted_pix[i] = pix[j]
                sorted_pix[i].pnr = i
                
    return sorted_pix
