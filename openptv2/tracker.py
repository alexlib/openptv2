"""
Particle tracking module.

Provides the Tracker class and tracking parameters for particle tracking
velocimetry.
"""

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import numpy as np

__all__ = ["Tracker", "TrackingParameters", "track"]

# Try to import from Cython bindings
try:
    from optv.tracker import Tracker as _C_Tracker
    from optv.parameters import ControlParams, TrackingParams, SequenceParams, VolumeParams
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


@dataclass
class TrackingParameters:
    """
    Parameters for particle tracking.

    Attributes:
        dx: Maximum displacement in x direction (pixels)
        dy: Maximum displacement in y direction (pixels)
        dz: Maximum displacement in z direction (for 3D)
        n_neighbors: Number of neighbors to consider
        n_frames: Number of frames to track
        predictor_order: Order of predictor for tracking
        gap_closing: Whether to enable gap closing
        max_gap: Maximum gap to close
    """
    dx: float = 10.0
    dy: float = 10.0
    dz: float = 5.0
    n_neighbors: int = 3
    n_frames: int = 2
    predictor_order: int = 1
    gap_closing: bool = False
    max_gap: int = 1
    
    # Additional parameters for Cython binding
    dmax_2d: float = 15.0
    dmax_3d: float = 10.0
    n_predictor: int = 3
    add_nearest: bool = False
    nn_radius: float = 2.0


class Tracker:
    """
    Particle tracker for PTV.
    
    Wraps the Cython Tracker class if available, otherwise uses pure Python.

    Example:
        >>> tracker = Tracker()
        >>> tracks = tracker.track(targets_sequence)
    """

    def __init__(
        self,
        parameters: Optional[TrackingParameters] = None,
        engine: Optional[str] = None,
        control_params: Optional['ControlParams'] = None,
        volume_params: Optional['VolumeParams'] = None,
        sequence_params: Optional['SequenceParams'] = None,
        calibrations: Optional[List] = None,
        naming: Optional[Dict] = None
    ):
        """
        Initialize the tracker.

        Args:
            parameters: Tracking parameters
            engine: Engine to use ("optv" or "python")
            control_params: Control parameters (for Cython engine)
            volume_params: Volume parameters (for Cython engine)
            sequence_params: Sequence parameters (for Cython engine)
            calibrations: List of Calibration objects (for Cython engine)
            naming: File naming dictionary (for Cython engine)
        """
        self.parameters = parameters or TrackingParameters()
        self._engine_pref = engine
        self._cython_tracker = None
        
        # Initialize Cython tracker if dependencies are available
        if _CYTHON_AVAILABLE and all([
            control_params, volume_params, sequence_params, calibrations
        ]):
            try:
                self._cython_tracker = _C_Tracker(
                    control_params, volume_params, 
                    self._to_cython_tracking_params(self.parameters),
                    sequence_params, calibrations, naming
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize Cython tracker: {e}")

    def _to_cython_tracking_params(self, params: TrackingParameters) -> 'TrackingParams':
        """Convert Python TrackingParameters to Cython TrackingParams."""
        if not _CYTHON_AVAILABLE:
            return None
        
        try:
            c_params = TrackingParams()
            c_params.dmax = params.dmax_2d
            c_params.dzmax = params.dmax_3d
            c_params.nneighbors = params.n_neighbors
            c_params.npredictor = params.n_predictor
            c_params.addnearest = params.add_nearest
            c_params.nnradius = params.nn_radius
            return c_params
        except Exception:
            return None

    def track(
        self,
        targets_sequence: List[np.ndarray],
        engine: Optional[str] = None
    ) -> List[Dict[int, Any]]:
        """
        Track particles through a sequence of frames.

        Args:
            targets_sequence: List of target coordinates per frame
            engine: Engine to use (overrides default)

        Returns:
            List of tracks, each track is a dict mapping frame number to target
        """
        from .engine import select_engine

        eng = select_engine(engine or self._engine_pref)
        
        # Use engine-specific implementation
        if hasattr(eng, 'track'):
            return eng.track(targets_sequence, self.parameters)
        
        # Fallback: simple nearest-neighbor tracking
        return self._simple_track(targets_sequence)

    def _simple_track(
        self,
        targets_sequence: List[np.ndarray]
    ) -> List[Dict[int, Any]]:
        """Simple nearest-neighbor tracking fallback."""
        if len(targets_sequence) < 2:
            return []
        
        tracks = []
        max_disp_sq = self.parameters.dx ** 2 + self.parameters.dy ** 2
        
        # Initialize tracks from first frame
        for i, pos in enumerate(targets_sequence[0]):
            tracks.append({0: (i, pos)})
        
        # Track through sequence
        for frame_idx in range(1, len(targets_sequence)):
            prev_targets = targets_sequence[frame_idx - 1]
            curr_targets = targets_sequence[frame_idx]
            
            used = set()
            
            for track in tracks:
                if frame_idx - 1 not in track:
                    continue
                    
                prev_idx, prev_pos = track[frame_idx - 1]
                
                # Find nearest neighbor in current frame
                best_idx = None
                best_dist_sq = max_disp_sq
                
                for curr_idx, curr_pos in enumerate(curr_targets):
                    if curr_idx in used:
                        continue
                    
                    dist_sq = np.sum((curr_pos - prev_pos) ** 2)
                    if dist_sq < best_dist_sq:
                        best_dist_sq = dist_sq
                        best_idx = curr_idx
                
                if best_idx is not None:
                    track[frame_idx] = (best_idx, curr_targets[best_idx])
                    used.add(best_idx)
        
        # Filter out incomplete tracks
        tracks = [t for t in tracks if len(t) >= 2]
        
        return tracks

    def track_frame(
        self,
        prev_targets: np.ndarray,
        curr_targets: np.ndarray,
        engine: Optional[str] = None
    ) -> np.ndarray:
        """
        Track particles between two consecutive frames.

        Args:
            prev_targets: Targets in previous frame (Nx2 array)
            curr_targets: Targets in current frame (Mx2 array)
            engine: Engine to use

        Returns:
            Correspondence matrix (NxM)
        """
        from .engine import select_engine

        eng = select_engine(engine or self._engine_pref)
        
        if hasattr(eng, 'track_frame'):
            return eng.track_frame(prev_targets, curr_targets, self.parameters)
        
        # Fallback: simple correlation matrix
        return self._simple_track_frame(prev_targets, curr_targets)

    def _simple_track_frame(
        self,
        prev_targets: np.ndarray,
        curr_targets: np.ndarray
    ) -> np.ndarray:
        """Simple frame-to-frame tracking fallback."""
        n_prev = len(prev_targets)
        n_curr = len(curr_targets)
        
        corr = np.zeros((n_prev, n_curr), dtype=np.int32)
        max_disp_sq = self.parameters.dx ** 2 + self.parameters.dy ** 2
        
        used = set()
        
        for i, prev_pos in enumerate(prev_targets):
            best_j = None
            best_dist_sq = max_disp_sq
            
            for j, curr_pos in enumerate(curr_targets):
                if j in used:
                    continue
                
                dist_sq = np.sum((curr_pos - prev_pos) ** 2)
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_j = j
            
            if best_j is not None:
                corr[i, best_j] = 1
                used.add(best_j)
        
        return corr

    def restart(self):
        """Restart tracking (for Cython engine)."""
        if self._cython_tracker is not None:
            self._cython_tracker.restart()

    def step_forward(self) -> bool:
        """Step forward one frame (for Cython engine)."""
        if self._cython_tracker is not None:
            return self._cython_tracker.step_forward()
        return False

    def finalize(self):
        """Finalize tracking (for Cython engine)."""
        if self._cython_tracker is not None:
            self._cython_tracker.finalize()

    def full_forward(self):
        """Run full forward tracking (for Cython engine)."""
        if self._cython_tracker is not None:
            self._cython_tracker.full_forward()


def track(
    targets_sequence: List[np.ndarray],
    parameters: Optional[TrackingParameters] = None,
    engine: Optional[str] = None
) -> List[Dict[int, Any]]:
    """
    Convenience function for particle tracking.

    Args:
        targets_sequence: List of target coordinates per frame
        parameters: Tracking parameters
        engine: Engine to use

    Returns:
        List of tracks
    """
    tracker = Tracker(parameters=parameters, engine=engine)
    return tracker.track(targets_sequence, engine)
