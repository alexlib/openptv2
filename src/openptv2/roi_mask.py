"""Per-camera polygon ROI masks, drawn in the GUI's Mask window (mask_gui.py).

A camera with a saved ``mask_{i_cam}.txt`` (x y pixel points, one per line,
written by ``MaskGUI``) only detects particles inside that polygon. A camera
with no such file has no restriction -- the default is the full frame,
``0..imx, 0..imy``.
"""

from pathlib import Path

import numpy as np
from skimage.draw import polygon as sk_polygon

from openptv2.algorithms.image_processing import subtract_mask


def mask_file_path(i_cam: int, working_folder: Path | str | None = None) -> Path:
    folder = Path(working_folder) if working_folder is not None else Path.cwd()
    return folder / f"mask_{i_cam}.txt"


def load_mask_polygon(
    i_cam: int, working_folder: Path | str | None = None
) -> np.ndarray | None:
    """Read camera ``i_cam``'s saved polygon, or None if unset (= full frame)."""
    path = mask_file_path(i_cam, working_folder)
    if not path.exists():
        return None
    pts = np.atleast_2d(np.loadtxt(path))
    if pts.shape[0] < 3:
        return None
    return pts


def rasterize_mask(polygon_xy: np.ndarray, imx: int, imy: int) -> np.ndarray:
    """Build a uint8 raster mask (255 inside the polygon, 0 outside)."""
    mask = np.zeros((imy, imx), dtype=np.uint8)
    rr, cc = sk_polygon(polygon_xy[:, 1], polygon_xy[:, 0], shape=(imy, imx))
    mask[rr, cc] = 255
    return mask


def apply_roi_mask(
    img: np.ndarray, i_cam: int, working_folder: Path | str | None = None
) -> np.ndarray:
    """Zero out ``img`` outside camera ``i_cam``'s saved polygon.

    No saved polygon -> ``img`` is returned unchanged (default: full frame).
    """
    polygon_xy = load_mask_polygon(i_cam, working_folder)
    if polygon_xy is None:
        return img
    mask = rasterize_mask(polygon_xy, img.shape[1], img.shape[0])
    return subtract_mask(img, mask)
