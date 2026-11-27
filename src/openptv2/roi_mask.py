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


# One run's mask never changes frame to frame, but apply_roi_mask is called
# once per camera per frame -- caching the rasterized mask (keyed by mtime,
# so an edited-and-resaved mask still invalidates it) turns the per-frame
# cost back into one cheap os.stat() instead of a re-parse + re-rasterize.
_rasterized_cache: dict[tuple[str, int, int], tuple[float, np.ndarray | None]] = {}


def _cached_rasterized_mask(
    i_cam: int, working_folder: Path | str | None, imx: int, imy: int
) -> np.ndarray | None:
    path = mask_file_path(i_cam, working_folder)
    key = (str(path), imx, imy)
    mtime = path.stat().st_mtime if path.exists() else -1.0

    cached = _rasterized_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    polygon_xy = load_mask_polygon(i_cam, working_folder)
    mask = rasterize_mask(polygon_xy, imx, imy) if polygon_xy is not None else None
    _rasterized_cache[key] = (mtime, mask)
    return mask


def apply_roi_mask(
    img: np.ndarray, i_cam: int, working_folder: Path | str | None = None
) -> np.ndarray:
    """Zero out ``img`` outside camera ``i_cam``'s saved polygon.

    No saved polygon -> ``img`` is returned unchanged (default: full frame).
    """
    mask = _cached_rasterized_mask(i_cam, working_folder, img.shape[1], img.shape[0])
    if mask is None:
        return img
    return subtract_mask(img, mask)
