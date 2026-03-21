"""
Correspondence matching module for multi-camera PTV.

Provides algorithms for matching particles across camera views.
"""

from typing import Optional, List, Tuple, Union
import numpy as np

__all__ = ["Correspondence", "find_correspondences"]

# Try to import from Cython bindings
try:
    from optv.epipolar import find_epipolar_matches
    from optv.correspondences import find_correspondences as _c_find_correspondences
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


class Correspondence:
    """
    Multi-camera correspondence matching.
    
    Finds matching particles across multiple camera views using
    epipolar geometry constraints.

    Example:
        >>> corr = Correspondence(calibration)
        >>> matches = corr.find(targets_cam1, targets_cam2)
    """

    def __init__(
        self,
        calibration,
        epipolar_tolerance: float = 1.0,
        max_matches: int = 10
    ):
        """
        Initialize correspondence matcher.

        Args:
            calibration: Camera calibration object
            epipolar_tolerance: Tolerance for epipolar constraint (pixels)
            max_matches: Maximum number of matches per target
        """
        self.calibration = calibration
        self.epipolar_tolerance = epipolar_tolerance
        self.max_matches = max_matches

    def find(
        self,
        targets1: np.ndarray,
        targets2: np.ndarray,
        camera1: int = 0,
        camera2: int = 1
    ) -> np.ndarray:
        """
        Find correspondences between two camera views.

        Args:
            targets1: Targets in camera 1 (Nx2 array)
            targets2: Targets in camera 2 (Mx2 array)
            camera1: First camera index
            camera2: Second camera index

        Returns:
            Correspondence matrix (NxM), 1 where match found
        """
        if _CYTHON_AVAILABLE:
            try:
                # Use Cython implementation
                return find_epipolar_matches(
                    targets1, targets2,
                    self.calibration,
                    camera1, camera2,
                    self.epipolar_tolerance
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Cython correspondence matching failed: {e}")
        
        # Fallback to Python implementation
        return self._find_epipolar_python(targets1, targets2, camera1, camera2)

    def _find_epipolar_python(
        self,
        targets1: np.ndarray,
        targets2: np.ndarray,
        camera1: int,
        camera2: int
    ) -> np.ndarray:
        """Python fallback for epipolar matching."""
        n1 = len(targets1)
        n2 = len(targets2)
        
        corr_matrix = np.zeros((n1, n2), dtype=np.int32)
        
        # Simple epipolar constraint check
        for i, (x1, y1) in enumerate(targets1):
            # Get epipolar line in camera2 for point in camera1
            # This is a simplified version - full implementation would use
            # the calibration's fundamental matrix
            
            best_j = None
            best_dist = self.epipolar_tolerance
            
            for j, (x2, y2) in enumerate(targets2):
                # Simplified distance check (not true epipolar constraint)
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            
            if best_j is not None:
                corr_matrix[i, best_j] = 1
        
        return corr_matrix

    def find_multi(
        self,
        targets_list: List[np.ndarray],
        cameras: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Find correspondences across multiple cameras.

        Args:
            targets_list: List of target arrays for each camera
            cameras: Camera indices (default: 0, 1, 2, ...)

        Returns:
            List of matched particle indices
        """
        if cameras is None:
            cameras = list(range(len(targets_list)))
        
        if len(targets_list) < 2:
            return []
        
        # Start with first two cameras
        matches = []
        corr_01 = self.find(targets_list[0], targets_list[1], 
                           cameras[0], cameras[1])
        
        # Find matches in first two cameras
        for i in range(len(targets_list[0])):
            js = np.where(corr_01[i] == 1)[0]
            for j in js:
                matches.append([i, j])
        
        # Add matches from additional cameras
        for cam_idx in range(2, len(targets_list)):
            new_matches = []
            for match in matches:
                # Get position from first camera
                i0 = match[0]
                pos0 = targets_list[0][i0]
                
                # Find match in current camera
                corr = self.find(
                    pos0.reshape(1, -1),
                    targets_list[cam_idx],
                    cameras[0], cameras[cam_idx]
                )
                
                js = np.where(corr[0] == 1)[0]
                for j in js:
                    new_matches.append(match + [j])
            
            matches = new_matches
        
        return matches


def find_correspondences(
    targets1: np.ndarray,
    targets2: np.ndarray,
    calibration,
    tolerance: float = 1.0,
    engine: Optional[str] = None
) -> np.ndarray:
    """
    Convenience function to find correspondences.

    Args:
        targets1: Targets in first camera (Nx2 array)
        targets2: Targets in second camera (Mx2 array)
        calibration: Camera calibration
        tolerance: Epipolar tolerance (pixels)
        engine: Engine to use ("optv" or "python")

    Returns:
        Correspondence matrix (NxM)
    """
    from .engine import select_engine

    eng = select_engine(engine)
    
    # Try to use engine's correspondence matching
    if hasattr(eng, 'find_correspondences'):
        return eng.find_correspondences(
            targets1, targets2, calibration, tolerance
        )
    
    # Fallback to class-based implementation
    corr = Correspondence(calibration, tolerance)
    return corr.find(targets1, targets2)


def find_multi_correspondences(
    targets_list: List[np.ndarray],
    calibration,
    tolerance: float = 1.0,
    cameras: Optional[List[int]] = None,
    engine: Optional[str] = None
) -> List[np.ndarray]:
    """
    Find correspondences across multiple cameras.

    Args:
        targets_list: List of target arrays for each camera
        calibration: Camera calibration
        tolerance: Epipolar tolerance
        cameras: Camera indices
        engine: Engine to use

    Returns:
        List of matched particle indices
    """
    from .engine import select_engine

    eng = select_engine(engine)
    
    # Try to use engine's multi-camera matching
    if hasattr(eng, 'find_multi_correspondences'):
        return eng.find_multi_correspondences(
            targets_list, calibration, tolerance, cameras
        )
    
    # Fallback to class-based implementation
    corr = Correspondence(calibration, tolerance)
    return corr.find_multi(targets_list, cameras)
