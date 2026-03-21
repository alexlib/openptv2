"""
Calibration module for camera calibration and 3D reconstruction.

Provides Calibration class and related utilities.
"""

from typing import Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np

__all__ = ["Calibration", "Orientation", "calibrate"]

# Try to import from Cython bindings
try:
    from optv.calibration import Calibration as _C_Calibration
    from optv.orientation import Orientation as _C_Orientation
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


@dataclass
class Orientation:
    """
    Camera orientation parameters.
    
    Attributes:
        omega: Rotation around x-axis (radians)
        phi: Rotation around y-axis (radians)
        kappa: Rotation around z-axis (radians)
        x0: Principal point x (pixels)
        y0: Principal point y (pixels)
        z0: Camera position z (world units)
    """
    omega: float = 0.0  # Rotation around x-axis
    phi: float = 0.0    # Rotation around y-axis
    kappa: float = 0.0  # Rotation around z-axis
    x0: float = 0.0     # Principal point x
    y0: float = 0.0     # Principal point y
    z0: float = 0.0     # Camera position z
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [omega, phi, kappa]."""
        return np.array([self.omega, self.phi, self.kappa])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Orientation':
        """Create from numpy array [omega, phi, kappa]."""
        return cls(omega=arr[0], phi=arr[1], kappa=arr[2])


class Calibration:
    """
    Camera calibration for PTV.
    
    Wraps the Cython Calibration class if available.
    
    Handles camera parameters, distortion coefficients, and
    2D-3D coordinate transformations.

    Example:
        >>> cal = Calibration()
        >>> cal.load_from_file("calibration.yaml")
        >>> x, y = cal.world_to_pixel(X, Y, Z)
    """

    def __init__(
        self,
        pos: Optional[np.ndarray] = None,
        angs: Optional[np.ndarray] = None,
        prim_point: Optional[np.ndarray] = None,
        rad_dist: Optional[np.ndarray] = None,
        decent: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        glass: Optional[np.ndarray] = None
    ):
        """
        Initialize calibration.

        Args:
            pos: Camera external position (3-element array)
            angs: Rotation angles [omega, phi, kappa] in radians (3-element array)
            prim_point: Primary point position (3-element array)
            rad_dist: Radial distortion coefficients (3-element array)
            decent: Decentering parameters (2-element array)
            affine: Affine transform parameters [scale, shear] (2-element array)
            glass: Glass vector (3-element array)
        """
        self._calibration = None
        
        if _CYTHON_AVAILABLE:
            try:
                self._calibration = _C_Calibration(
                    pos=pos, angs=angs, prim_point=prim_point,
                    rad_dist=rad_dist, decent=decent, affine=affine, glass=glass
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to initialize Cython calibration: {e}")
        
        # Store parameters for fallback
        if pos is None:
            pos = np.zeros(3)
        if angs is None:
            angs = np.zeros(3)
        if prim_point is None:
            prim_point = np.zeros(3)
        if rad_dist is None:
            rad_dist = np.zeros(3)
        if decent is None:
            decent = np.zeros(2)
        if affine is None:
            affine = np.array([1.0, 0.0])
        if glass is None:
            glass = np.zeros(3)
        
        self._pos = np.asarray(pos, dtype=np.float64)
        self._angs = np.asarray(angs, dtype=np.float64)
        self._prim_point = np.asarray(prim_point, dtype=np.float64)
        self._rad_dist = np.asarray(rad_dist, dtype=np.float64)
        self._decent = np.asarray(decent, dtype=np.float64)
        self._affine = np.asarray(affine, dtype=np.float64)
        self._glass = np.asarray(glass, dtype=np.float64)

    @classmethod
    def from_file(
        cls,
        ori_file: str,
        add_file: Optional[str] = None,
        fallback_file: Optional[str] = None
    ) -> 'Calibration':
        """
        Load calibration from files.

        Args:
            ori_file: Path to orientation file (.ori)
            add_file: Path to additional parameters file (.addpar)
            fallback_file: Fallback file if add_file fails

        Returns:
            Calibration object
        """
        cal = cls()
        
        if _CYTHON_AVAILABLE and cal._calibration is not None:
            try:
                cal._calibration.from_file(ori_file, add_file, fallback_file)
                return cal
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load Cython calibration: {e}")
        
        # Fallback: try to parse YAML file
        try:
            cal.load_from_file(ori_file)
        except Exception:
            pass
        
        return cal

    def load_from_file(self, filepath: str) -> None:
        """Load calibration from YAML file."""
        import yaml

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Parse calibration data
        if 'num_cameras' in data:
            self.num_cameras = data['num_cameras']
        
        if 'orientations' in data:
            for i, o_data in enumerate(data['orientations']):
                if i < len(self._orientations):
                    self._orientations[i].omega = o_data.get('omega', 0.0)
                    self._orientations[i].phi = o_data.get('phi', 0.0)
                    self._orientations[i].kappa = o_data.get('kappa', 0.0)
                    self._orientations[i].x0 = o_data.get('x0', 0.0)
                    self._orientations[i].y0 = o_data.get('y0', 0.0)

    def save_to_file(self, filepath: str) -> None:
        """Save calibration to YAML file."""
        import yaml

        data = {
            'num_cameras': getattr(self, 'num_cameras', 1),
            'orientations': [
                {
                    'omega': o.omega,
                    'phi': o.phi,
                    'kappa': o.kappa,
                    'x0': o.x0,
                    'y0': o.y0
                }
                for o in getattr(self, '_orientations', [Orientation()])
            ],
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f)

    @property
    def position(self) -> np.ndarray:
        """Get camera position."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_pos()
        return self._pos.copy()

    @position.setter
    def position(self, value: np.ndarray):
        """Set camera position."""
        value = np.asarray(value, dtype=np.float64)
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_pos(value)
        self._pos = value

    @property
    def angles(self) -> np.ndarray:
        """Get camera angles [omega, phi, kappa]."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_angles()
        return self._angs.copy()

    @angles.setter
    def angles(self, value: np.ndarray):
        """Set camera angles."""
        value = np.asarray(value, dtype=np.float64)
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_angles(value)
        self._angs = value

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_rotation_matrix()
        
        # Fallback: compute from angles
        omega, phi, kappa = self._angs
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(omega), -np.sin(omega)],
                       [0, np.sin(omega), np.cos(omega)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                       [0, 1, 0],
                       [-np.sin(phi), 0, np.cos(phi)]])
        Rz = np.array([[np.cos(kappa), -np.sin(kappa), 0],
                       [np.sin(kappa), np.cos(kappa), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx

    @property
    def primary_point(self) -> np.ndarray:
        """Get primary point position."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_primary_point()
        return self._prim_point.copy()

    @primary_point.setter
    def primary_point(self, value: np.ndarray):
        """Set primary point position."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_primary_point(value)
        self._prim_point = np.asarray(value, dtype=np.float64)

    @property
    def radial_distortion(self) -> np.ndarray:
        """Get radial distortion coefficients [k1, k2, k3]."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_radial_distortion()
        return self._rad_dist.copy()

    @radial_distortion.setter
    def radial_distortion(self, value: np.ndarray):
        """Set radial distortion coefficients."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_radial_distortion(value)
        self._rad_dist = np.asarray(value, dtype=np.float64)

    @property
    def decentering(self) -> np.ndarray:
        """Get decentering parameters [p1, p2]."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_decentering()
        return self._decent.copy()

    @decentering.setter
    def decentering(self, value: np.ndarray):
        """Set decentering parameters."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_decentering(value)
        self._decent = np.asarray(value, dtype=np.float64)

    @property
    def affine(self) -> np.ndarray:
        """Get affine parameters [scale, shear]."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_affine()
        return self._affine.copy()

    @affine.setter
    def affine(self, value: np.ndarray):
        """Set affine parameters."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_affine_trans(value)
        self._affine = np.asarray(value, dtype=np.float64)

    @property
    def glass_vector(self) -> np.ndarray:
        """Get glass vector."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            return self._calibration.get_glass_vec()
        return self._glass.copy()

    @glass_vector.setter
    def glass_vector(self, value: np.ndarray):
        """Set glass vector."""
        if _CYTHON_AVAILABLE and self._calibration is not None:
            self._calibration.set_glass_vec(value)
        self._glass = np.asarray(value, dtype=np.float64)

    def world_to_pixel(
        self,
        X: Union[np.ndarray, float],
        Y: Union[np.ndarray, float],
        Z: Union[np.ndarray, float],
        camera: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform world coordinates to pixel coordinates.

        Args:
            X, Y, Z: World coordinates (scalars or arrays)
            camera: Camera index

        Returns:
            (x, y) pixel coordinates
        """
        from .engine import select_engine

        eng = select_engine()
        return eng.world_to_pixel(X, Y, Z, self, camera)

    def pixel_to_world(
        self,
        x: Union[np.ndarray, float],
        y: Union[np.ndarray, float],
        camera: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform pixel coordinates to world ray.

        Args:
            x, y: Pixel coordinates
            camera: Camera index

        Returns:
            (X, Y, Z) ray direction
        """
        from .engine import select_engine

        eng = select_engine()
        return eng.pixel_to_world(x, y, self, camera)

    def calibrate(
        self,
        image_points: List[np.ndarray],
        world_points: np.ndarray,
        initial_guess: Optional['Calibration'] = None
    ) -> float:
        """
        Calibrate cameras from correspondence points.

        Args:
            image_points: 2D points in each camera image
            world_points: 3D world coordinates
            initial_guess: Initial calibration guess

        Returns:
            RMS reprojection error
        """
        from .engine import select_engine

        eng = select_engine()
        return eng.calibrate(image_points, world_points, initial_guess or self)


def calibrate(
    image_points: List[np.ndarray],
    world_points: np.ndarray,
    initial_guess: Optional[Calibration] = None,
    engine: Optional[str] = None
) -> Calibration:
    """
    Convenience function for camera calibration.

    Args:
        image_points: 2D points in each camera image
        world_points: 3D world coordinates
        initial_guess: Initial calibration guess
        engine: Engine to use

    Returns:
        Calibrated Calibration object
    """
    from .engine import select_engine

    eng = select_engine(engine)
    
    if initial_guess is None:
        initial_guess = Calibration()
    
    # Perform calibration
    rms_error = eng.calibrate(image_points, world_points, initial_guess)
    
    return initial_guess
