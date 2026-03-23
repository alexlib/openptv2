"""Adapter layer to provide Cython-compatible Calibration API for Python engine."""

from pathlib import Path
from typing import Optional, Union

from .calibration import Calibration as CalibrationImpl


class Calibration:
    """
    Cython-compatible Calibration wrapper for Python engine.

    Provides the same API as bindings/optv/calibration.pyx Calibration class:
    - __init__()
    - from_file(ori_file, add_file=None) -> classmethod (matches Cython instance method)
    - write(filename, add_file)
    - get_pos(), set_pos()
    - get_angles(), set_angles()
    - get_rotation_matrix()
    - get_radial_distortion(), set_radial_distortion()
    - get_decentering(), set_decentering()
    - get_affine(), set_affine_trans()
    - get_glass_vec(), set_glass_vec()
    """

    def __init__(self):
        """Create an empty calibration object."""
        self._impl = CalibrationImpl()

    @classmethod
    def from_file(
        cls, ori_file: Union[str, bytes], add_file: Optional[Union[str, bytes]] = None
    ):
        """
        Read calibration from files.

        Arguments:
        ori_file: path of file containing interior and exterior orientation data.
        add_file: path of file containing added (distortions) parameters.
        """
        # Convert bytes to str if needed
        if isinstance(ori_file, bytes):
            ori_file = ori_file.decode("utf-8")
        if isinstance(add_file, bytes):
            add_file = add_file.decode("utf-8")

        # Handle None or string for add_file
        if add_file is not None and isinstance(add_file, str):
            add_file = Path(add_file)
        elif add_file is None:
            add_file = None

        if isinstance(ori_file, str):
            ori_file = Path(ori_file)

        return CalibrationImpl.from_file(ori_file, add_file)

    def from_file(
        self, ori_file: Union[str, bytes], add_file: Optional[Union[str, bytes]] = None
    ):
        """
        Instance method version for compatibility with Cython API.
        """
        # Convert bytes to str if needed
        if isinstance(ori_file, bytes):
            ori_file = ori_file.decode("utf-8")
        if isinstance(add_file, bytes):
            add_file = add_file.decode("utf-8")

        # Handle None or string
        if add_file is not None and isinstance(add_file, str):
            add_file = Path(add_file)
        elif add_file is None:
            add_file = None

        if isinstance(ori_file, str):
            ori_file = Path(ori_file)

        self._impl = CalibrationImpl.from_file(ori_file, add_file)
        return self

    def write(
        self, filename: Union[str, bytes], add_file: Optional[Union[str, bytes]] = None
    ):
        """Write calibration to file."""
        if isinstance(filename, bytes):
            filename = filename.decode("utf-8")
        if isinstance(add_file, bytes):
            add_file = add_file.decode("utf-8")

        self._impl.write(filename, add_file)

    def get_pos(self):
        """Get position (x, y, z)."""
        return self._impl.get_pos()

    def set_pos(self, pos):
        """Set position (x, y, z)."""
        self._impl.set_pos(pos)

    def get_angles(self):
        """Get angles (omega, phi, kappa)."""
        return self._impl.get_angles()

    def set_angles(self, angles):
        """Set angles (omega, phi, kappa)."""
        self._impl.set_angles(angles)

    def get_rotation_matrix(self):
        """Get rotation matrix."""
        return self._impl.get_rotation_matrix()

    def set_primary_point(self, point):
        """Set primary point."""
        self._impl.set_primary_point(point)

    def get_primary_point(self):
        """Get primary point."""
        return self._impl.get_primary_point()

    def get_radial_distortion(self):
        """Get radial distortion coefficients."""
        return self._impl.get_radial_distortion()

    def set_radial_distortion(self, dist):
        """Set radial distortion coefficients."""
        self._impl.set_radial_distortion(dist)

    def get_decentering(self):
        """Get decentering distortion."""
        return self._impl.get_decentering()

    def set_decentering(self, dec):
        """Set decentering distortion."""
        self._impl.set_decentering(dec)

    def get_affine(self):
        """Get affine transformation parameters."""
        return self._impl.get_affine()

    def set_affine_trans(self, affine):
        """Set affine transformation parameters."""
        self._impl.set_affine_trans(affine)

    def get_glass_vec(self):
        """Get glass vector."""
        return self._impl.get_glass_vec()

    def set_glass_vec(self, glass):
        """Set glass vector."""
        self._impl.set_glass_vec(glass)
