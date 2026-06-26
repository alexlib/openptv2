"""
Calibration compatibility wrapper providing optv-like API.
"""

import numpy as np
from openptv2.algorithms.calibration import Calibration as AlgoCalibration


class Calibration:
    """Wrapper for algorithms.calibration.Calibration with optv-compatible API."""

    def __init__(self, pos=None, angs=None, prim_point=None, rad_dist=None,
                 decent=None, affine=None, glass=None, cal=None):
        """
        Initialize Calibration wrapper (matching optv signature).

        All arguments are optional arrays, default for all is zeros except
        affine that defaults to [1, 0].
        """
        if cal is not None and isinstance(cal, AlgoCalibration):
            self._cal = cal
            return

        self._cal = AlgoCalibration()

        if pos is not None:
            self.set_pos(pos)
        if angs is not None:
            self.set_angles(angs)
        if prim_point is not None:
            self.set_primary_point(prim_point)
        if rad_dist is not None:
            self.set_radial_distortion(rad_dist)
        if decent is not None:
            self.set_decentering(decent)
        if affine is not None:
            self.set_affine_trans(affine)
        if glass is not None:
            self.set_glass_vec(glass)

    def get_pos(self):
        """Get camera position as ndarray[3]."""
        return np.array([self._cal.ext_par.x0, self._cal.ext_par.y0, self._cal.ext_par.z0])

    def set_pos(self, pos):
        """Set camera position from ndarray[3]."""
        self._cal.ext_par.x0 = float(pos[0])
        self._cal.ext_par.y0 = float(pos[1])
        self._cal.ext_par.z0 = float(pos[2])

    def get_angles(self):
        """Get rotation angles (omega, phi, kappa) as ndarray[3]."""
        return np.array([self._cal.ext_par.omega, self._cal.ext_par.phi, self._cal.ext_par.kappa])

    def set_angles(self, angles):
        """Set rotation angles from ndarray[3]."""
        self._cal.ext_par.omega = float(angles[0])
        self._cal.ext_par.phi = float(angles[1])
        self._cal.ext_par.kappa = float(angles[2])
        # Recompute rotation matrix
        self._cal.ext_par.compute_rotation_matrix()

    def get_primary_point(self):
        """Get primary point (xh, yh, cc) as ndarray[3]."""
        return np.array([self._cal.int_par.xh, self._cal.int_par.yh, self._cal.int_par.cc])

    def set_primary_point(self, pp):
        """Set primary point from ndarray (2 or 3 elements)."""
        self._cal.int_par.xh = float(pp[0])
        self._cal.int_par.yh = float(pp[1])
        if len(pp) > 2:
            self._cal.int_par.cc = float(pp[2])

    def get_radial_distortion(self):
        """Get radial distortion coefficients (k1, k2, k3) as ndarray[3]."""
        return np.array([self._cal.added_par.k1, self._cal.added_par.k2, self._cal.added_par.k3])

    def set_radial_distortion(self, dist):
        """Set radial distortion from ndarray[3]."""
        self._cal.added_par.k1 = float(dist[0])
        self._cal.added_par.k2 = float(dist[1])
        self._cal.added_par.k3 = float(dist[2])

    def get_decentering(self):
        """Get decentering parameters (p1, p2) as ndarray[2]."""
        return np.array([self._cal.added_par.p1, self._cal.added_par.p2])

    def set_decentering(self, decent):
        """Set decentering from ndarray[2]."""
        self._cal.added_par.p1 = float(decent[0])
        self._cal.added_par.p2 = float(decent[1])

    def get_affine(self):
        """Get affine parameters (scx, she) as ndarray[2]."""
        return np.array([self._cal.added_par.scx, self._cal.added_par.she])

    def set_affine_trans(self, affine):
        """Set affine transform from ndarray[2]."""
        self._cal.added_par.scx = float(affine[0])
        self._cal.added_par.she = float(affine[1])

    def get_glass_vec(self):
        """Get glass vector (vec_x, vec_y, vec_z) as ndarray[3]."""
        return np.array([self._cal.glass_par.vec_x, self._cal.glass_par.vec_y, self._cal.glass_par.vec_z])

    def set_glass_vec(self, gvec):
        """Set glass vector from ndarray[3]."""
        self._cal.glass_par.vec_x = float(gvec[0])
        self._cal.glass_par.vec_y = float(gvec[1])
        self._cal.glass_par.vec_z = float(gvec[2])

    def get_rotation_matrix(self):
        """Get rotation matrix as ndarray[3, 3]."""
        return self._cal.ext_par.dm.copy()

    def write(self, ori_file, add_file=None):
        """Write calibration to file(s)."""
        if isinstance(ori_file, bytes):
            ori_file = ori_file.decode('utf-8')
        if isinstance(add_file, bytes):
            add_file = add_file.decode('utf-8')
        self._cal.to_file(ori_file, add_file)

    def from_file(self, ori_file, add_file=None, fallback_file=None):
        """
        Populate calibration from .ori and .addpar files (in-place, like optv).

        Args:
            ori_file: Path to .ori file (str or bytes)
            add_file: Path to .addpar file (optional, str or bytes)
            fallback_file: Fallback path for .addpar (optional, str or bytes)
        """
        def _to_str(v):
            if isinstance(v, bytes):
                return v.decode('utf-8')
            return v

        ori_file = _to_str(ori_file)
        add_file = _to_str(add_file) if add_file is not None else None
        fallback_file = _to_str(fallback_file) if fallback_file is not None else None

        add_path = add_file
        if add_path is None and fallback_file is not None:
            add_path = fallback_file

        self._cal = AlgoCalibration.from_file(ori_file, add_path, fallback_file)
