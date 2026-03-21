"""
Tracking frame buffer module.

This module provides the Target and Targets classes for particle detection
and tracking.
"""

from typing import Optional, List, Tuple
import numpy as np

__all__ = ["Target", "Targets", "read_targets", "write_targets", "Frame"]

# Try to import from Cython bindings, fall back to Python implementation
try:
    from optv.tracking_framebuf import Target as _C_Target
    from optv.tracking_framebuf import TargetArray as _C_TargetArray
    from optv.tracking_framebuf import Frame as _C_Frame
    from optv.tracking_framebuf import read_targets as _c_read_targets
    from optv.tracking_framebuf import write_targets as _c_write_targets
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


class Target:
    """
    Represents a single detected particle target.
    
    Wraps the Cython Target class if available, otherwise uses pure Python.

    Attributes:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        n: Number of pixels in target
        nx: Width in pixels
        ny: Height in pixels
        sumg: Sum of grey values
        pnr: Particle number (used by tracking)
        tnr: Target number (used by tracking)
    """

    def __init__(
        self,
        pnr: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        n: int = 0,
        nx: int = 0,
        ny: int = 0,
        sumg: float = 0.0,
        tnr: int = 0
    ):
        if _CYTHON_AVAILABLE:
            # Use Cython implementation
            self._target = _C_Target(
                pnr=pnr, x=x, y=y, n=n, nx=nx, ny=ny, sumg=sumg, tnr=tnr
            )
        else:
            self._target = None
            self.pnr = pnr
            self.x = x
            self.y = y
            self.n = n
            self.nx = nx
            self.ny = ny
            self.sumg = sumg
            self.tnr = tnr

    @property
    def x(self) -> float:
        """X coordinate in pixels."""
        if _CYTHON_AVAILABLE and self._target is not None:
            return self._target.pos()[0]
        return getattr(self, '_x', 0.0)

    @x.setter
    def x(self, value: float):
        if _CYTHON_AVAILABLE and self._target is not None:
            pos = list(self._target.pos())
            pos[0] = value
            self._target.set_pos(pos)
        else:
            self._x = value

    @property
    def y(self) -> float:
        """Y coordinate in pixels."""
        if _CYTHON_AVAILABLE and self._target is not None:
            return self._target.pos()[1]
        return getattr(self, '_y', 0.0)

    @y.setter
    def y(self, value: float):
        if _CYTHON_AVAILABLE and self._target is not None:
            pos = list(self._target.pos())
            pos[1] = value
            self._target.set_pos(pos)
        else:
            self._y = value

    @property
    def pos(self) -> Tuple[float, float]:
        """Get target position as (x, y) tuple."""
        if _CYTHON_AVAILABLE and self._target is not None:
            return self._target.pos()
        return (self.x, self.y)

    def __repr__(self):
        if _CYTHON_AVAILABLE and self._target is not None:
            return f"Target(x={self.x:.2f}, y={self.y:.2f}, pnr={self._target.pnr()})"
        return f"Target(x={self.x:.2f}, y={self.y:.2f}, pnr={self.pnr})"


class Targets:
    """
    Container for multiple detected targets.
    
    Wraps the Cython TargetArray class if available.

    Attributes:
        targets: List of Target objects
    """

    def __init__(self, size: int = 0):
        if _CYTHON_AVAILABLE:
            self._target_array = _C_TargetArray(size)
        else:
            self._target_array = None
            self._targets: List[Target] = []

    def add(self, target: Target) -> None:
        """Add a target to the collection."""
        if _CYTHON_AVAILABLE and self._target_array is not None:
            # Cython implementation handles this internally
            pass
        else:
            self._targets.append(target)

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        """Get coordinates as numpy array (Nx2)."""
        if _CYTHON_AVAILABLE and self._target_array is not None:
            coords = np.array([t.pos for t in self], dtype=np.float64)
            return coords
        elif not _CYTHON_AVAILABLE:
            if self._targets:
                return np.array([(t.x, t.y) for t in self._targets], dtype=np.float64)
        return None

    def sort_y(self) -> None:
        """Sort targets by Y coordinate."""
        if _CYTHON_AVAILABLE and self._target_array is not None:
            self._target_array.sort_y()

    def write(self, file_base: str, frame_num: int) -> None:
        """Write targets to file."""
        if _CYTHON_AVAILABLE and self._target_array is not None:
            c_string = file_base.encode('utf-8')
            self._target_array.write(c_string, frame_num)

    def __len__(self):
        if _CYTHON_AVAILABLE and self._target_array is not None:
            return len(self._target_array)
        return len(self._targets)

    def __getitem__(self, idx):
        if _CYTHON_AVAILABLE and self._target_array is not None:
            c_target = self._target_array[idx]
            # Wrap in Python Target
            t = Target()
            t._target = c_target
            return t
        return self._targets[idx]

    def __repr__(self):
        n = len(self)
        return f"Targets({n} targets)"


class Frame:
    """
    Holds a frame of particles with 3D positions and tracking information.
    """

    def __init__(
        self,
        num_cams: int = 4,
        corres_file_base: Optional[str] = None,
        linkage_file_base: Optional[str] = None,
        prio_file_base: Optional[str] = None,
        target_file_base: Optional[List[str]] = None,
        frame_num: Optional[int] = None
    ):
        if _CYTHON_AVAILABLE:
            if corres_file_base is not None and target_file_base is not None:
                self._frame = _C_Frame(
                    num_cams,
                    corres_file_base.encode('utf-8') if corres_file_base else None,
                    linkage_file_base.encode('utf-8') if linkage_file_base else None,
                    prio_file_base.encode('utf-8') if prio_file_base else None,
                    [f.encode('utf-8') for f in target_file_base] if target_file_base else None,
                    frame_num
                )
            else:
                self._frame = None
        else:
            self._frame = None
            self.num_cams = num_cams

    @property
    def positions(self) -> Optional[np.ndarray]:
        """Get 3D positions as (n,3) array."""
        if _CYTHON_AVAILABLE and self._frame is not None:
            return self._frame.positions()
        return None

    def target_positions_for_camera(self, cam: int) -> Optional[np.ndarray]:
        """Get 2D target positions for a specific camera."""
        if _CYTHON_AVAILABLE and self._frame is not None:
            return self._frame.target_positions_for_camera(cam)
        return None

    @property
    def num_parts(self) -> int:
        """Get number of 3D particles in frame."""
        if _CYTHON_AVAILABLE and self._frame is not None:
            return self._frame._frm.num_parts if self._frame._frm is not None else 0
        return 0


def read_targets(basename: str, frame_num: int) -> Targets:
    """
    Read targets from a file.

    Args:
        basename: Base name of the file (without frame number)
        frame_num: Frame number

    Returns:
        Targets object containing the read targets
    """
    if _CYTHON_AVAILABLE:
        c_string = basename.encode('utf-8')
        return _c_read_targets(c_string, frame_num)
    else:
        # Fallback: return empty Targets
        return Targets()


def write_targets(targets: Targets, basename: str, frame_num: int) -> None:
    """
    Write targets to a file.

    Args:
        targets: Targets object to write
        basename: Base name of the file
        frame_num: Frame number
    """
    if _CYTHON_AVAILABLE and hasattr(targets, '_target_array'):
        c_string = basename.encode('utf-8')
        targets._target_array.write(c_string, frame_num)


def detect_targets(
    image: np.ndarray,
    threshold: float = 50.0,
    min_area: int = 3,
    max_area: int = 100,
    engine: Optional[str] = None
) -> Targets:
    """
    Detect particle targets in an image.

    Args:
        image: Input image (2D numpy array)
        threshold: Intensity threshold for detection
        min_area: Minimum particle area
        max_area: Maximum particle area
        engine: Engine to use ("optv" or "python", None for default)

    Returns:
        Targets object containing detected particles
    """
    from .engine import select_engine

    eng = select_engine(engine)
    
    # Try to use engine's detect_targets
    if hasattr(eng, 'detect_targets'):
        return eng.detect_targets(image, threshold, min_area, max_area)
    else:
        # Fallback to image_processing module
        try:
            from optv.image_processing import detect_targets as _detect
            return _detect(image, threshold, min_area, max_area)
        except (ImportError, AttributeError):
            # Return empty targets if nothing works
            return Targets()
