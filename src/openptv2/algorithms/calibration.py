"""Camera calibration data structures and I/O.

Translation of lib/src/calibration.c and lib/include/calibration.h.

Provides:
- Exterior: camera position (x0, y0, z0) and angles (omega, phi, kappa) + rotation matrix
- Interior: principal point (xh, yh) and camera constant (cc)
- Glass: glass interface normal vector and refractive properties
- AddedPar: Brown distortion parameters (k1, k2, k3, p1, p2, scx, she)
- MmLut: multimedia look-up table
- Calibration: aggregates all above
"""
from __future__ import annotations

import cython

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@cython.cclass
@dataclass
class Exterior:
    """Exterior orientation: camera position and orientation.

    Attributes:
        x0, y0, z0: camera center position [mm].
        omega, phi, kappa: rotation angles [radians].
        dm: 3x3 rotation matrix (computed from angles).
    """
    x0: cython.double = 0.0
    y0: cython.double = 0.0
    z0: cython.double = 0.0
    omega: cython.double = 0.0
    phi: cython.double = 0.0
    kappa: cython.double = 0.0
    dm: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    def compute_rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix from omega, phi, kappa angles.

        Matches the C `rotation_matrix()` function.

        Returns:
            3x3 rotation matrix.
        """
        cp = np.cos(self.phi)
        sp = np.sin(self.phi)
        co = np.cos(self.omega)
        so = np.sin(self.omega)
        ck = np.cos(self.kappa)
        sk = np.sin(self.kappa)

        dm = np.array([
            [cp * ck, -cp * sk, sp],
            [co * sk + so * sp * ck, co * ck - so * sp * sk, -so * cp],
            [so * sk - co * sp * ck, so * ck + co * sp * sk, co * cp],
        ], dtype=np.float64)

        self.dm = dm
        return dm


@cython.cclass
@dataclass
class Interior:
    """Interior orientation: principal point and camera constant.

    Attributes:
        xh, yh: principal point (sensor shift) [mm].
        cc: camera constant (focal length) [mm].
    """
    xh: cython.double = 0.0
    yh: cython.double = 0.0
    cc: cython.double = 0.0


@cython.cclass
@dataclass
class Glass:
    """Glass interface parameters.

    Attributes:
        vec_x, vec_y, vec_z: normal vector to glass surface.
        n1, n2, n3: refractive indices (not used directly, stored for reference).
        d: glass thickness [mm].
    """
    vec_x: cython.double = 0.0
    vec_y: cython.double = 0.0
    vec_z: cython.double = 0.0
    n1: cython.double = 0.0
    n2: cython.double = 0.0
    n3: cython.double = 0.0
    d: cython.double = 0.0


@cython.cclass
@dataclass
class AddedPar:
    """Brown distortion parameters.

    Attributes:
        k1, k2, k3: radial distortion coefficients.
        p1, p2: decentering distortion coefficients.
        scx: scale factor.
        she: shear angle.
        field: unused field (legacy).
    """
    k1: cython.double = 0.0
    k2: cython.double = 0.0
    k3: cython.double = 0.0
    p1: cython.double = 0.0
    p2: cython.double = 0.0
    scx: cython.double = 1.0
    she: cython.double = 0.0
    field: cython.int = 0


@cython.cclass
@dataclass
class MmLut:
    """Multimedia Look-Up Table.

    Attributes:
        origin: (x0, y0, z0) origin of the LUT grid.
        nr: number of radial grid points.
        nz: number of axial grid points.
        rw: grid spacing.
        data: 1D array of size nr * nz with multimedia factors.
    """
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    nr: cython.int = 0
    nz: cython.int = 0
    rw: cython.int = 2
    data: Optional[np.ndarray] = None  # 1D array of size nr * nz

    @property
    def is_initialized(self) -> bool:
        """Check if the LUT has been populated."""
        return self.data is not None


class hybrid_from_file:
    """A descriptor that allows from_file to be called as a classmethod or instance method."""
    def __get__(self, obj, cls):
        if obj is None:
            def _classmethod(ori_file, add_file=None, add_fallback=None):
                return cls._from_file_class(ori_file, add_file, add_fallback)
            return _classmethod
        else:
            def _instancemethod(ori_file, add_file=None, fallback_file=None):
                new_cal = cls._from_file_class(ori_file, add_file, fallback_file)
                obj.ext_par = new_cal.ext_par
                obj.int_par = new_cal.int_par
                obj.glass_par = new_cal.glass_par
                obj.added_par = new_cal.added_par
                obj.mmlut = new_cal.mmlut
                return obj
            return _instancemethod

@cython.cclass
@dataclass
class Calibration:
    """Complete calibration for a single camera.

    Aggregates exterior, interior, glass, distortion, and multimedia LUT.

    Attributes:
        ext_par: exterior orientation.
        int_par: interior orientation.
        glass_par: glass parameters.
        added_par: distortion parameters.
        mmlut: multimedia look-up table.
    """
    ext_par: Exterior = field(default_factory=Exterior)
    int_par: Interior = field(default_factory=Interior)
    glass_par: Glass = field(default_factory=Glass)
    added_par: AddedPar = field(default_factory=AddedPar)
    mmlut: MmLut = field(default_factory=MmLut)

    def __init__(
        self,
        ext_par = None,
        int_par = None,
        glass_par = None,
        added_par = None,
        mmlut = None,
        pos=None,
        angs=None,
        prim_point=None,
        rad_dist=None,
        decent=None,
        affine=None,
        glass=None,
        cal=None,
    ):
        """Initialize Calibration supporting both dataclass and legacy optv args."""
        # Legacy optv-style positional constructor:
        #   Calibration(pos, angs, prim_point, rad_dist, decent, affine, glass)
        # Those arrays would otherwise bind positionally to the dataclass fields
        # (ext_par, int_par, ...), so pos would receive `affine` and fail.
        # Detect this by the first positional being array-like (not an Exterior)
        # and remap to the legacy keyword arguments.
        if ext_par is not None and not isinstance(ext_par, Exterior):
            (pos, angs, prim_point, rad_dist, decent, affine, glass) = (
                ext_par, int_par, glass_par, added_par, mmlut, pos, angs,
            )
            ext_par = int_par = glass_par = added_par = mmlut = None

        if cal is not None and hasattr(cal, "ext_par"):
            self.ext_par = cal.ext_par
            self.int_par = cal.int_par
            self.glass_par = cal.glass_par
            self.added_par = cal.added_par
            self.mmlut = cal.mmlut
            return

        self.ext_par = ext_par if ext_par is not None else Exterior()
        self.int_par = int_par if int_par is not None else Interior()
        self.glass_par = glass_par if glass_par is not None else Glass()
        self.added_par = added_par if added_par is not None else AddedPar()
        self.mmlut = mmlut if mmlut is not None else MmLut()

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

        if self.ext_par is not None:
            self.ext_par.compute_rotation_matrix()

    def __post_init__(self):
        """Ensure rotation matrix is computed after initialization."""
        if self.ext_par is not None:
            self.ext_par.compute_rotation_matrix()

    # --- Backward Compatibility OOP Methods (No wrappers or indirection) ---

    def get_pos(self) -> np.ndarray:
        """Get camera position as ndarray[3]."""
        return np.array([self.ext_par.x0, self.ext_par.y0, self.ext_par.z0], dtype=np.float64)

    def set_pos(self, pos) -> None:
        """Set camera position from ndarray[3]."""
        if len(pos) != 3:
            raise ValueError("pos must have exactly 3 elements")
        self.ext_par.x0 = float(pos[0])
        self.ext_par.y0 = float(pos[1])
        self.ext_par.z0 = float(pos[2])

    def get_angles(self) -> np.ndarray:
        """Get rotation angles (omega, phi, kappa) as ndarray[3]."""
        return np.array([self.ext_par.omega, self.ext_par.phi, self.ext_par.kappa], dtype=np.float64)

    def set_angles(self, angles) -> None:
        """Set rotation angles from ndarray[3] and compute rotation matrix."""
        if len(angles) != 3:
            raise ValueError("angles must have exactly 3 elements")
        self.ext_par.omega = float(angles[0])
        self.ext_par.phi = float(angles[1])
        self.ext_par.kappa = float(angles[2])
        self.ext_par.compute_rotation_matrix()

    def get_primary_point(self) -> np.ndarray:
        """Get primary point (xh, yh, cc) as ndarray[3]."""
        return np.array([self.int_par.xh, self.int_par.yh, self.int_par.cc], dtype=np.float64)

    def set_primary_point(self, pp) -> None:
        """Set primary point from ndarray[3]."""
        if len(pp) != 3:
            raise ValueError("primary point must have exactly 3 elements")
        self.int_par.xh = float(pp[0])
        self.int_par.yh = float(pp[1])
        self.int_par.cc = float(pp[2])

    def get_radial_distortion(self) -> np.ndarray:
        """Get radial distortion coefficients (k1, k2, k3) as ndarray[3]."""
        return np.array([self.added_par.k1, self.added_par.k2, self.added_par.k3], dtype=np.float64)

    def set_radial_distortion(self, dist) -> None:
        """Set radial distortion from ndarray[3]."""
        if len(dist) != 3:
            raise ValueError("radial distortion must have exactly 3 elements")
        self.added_par.k1 = float(dist[0])
        self.added_par.k2 = float(dist[1])
        self.added_par.k3 = float(dist[2])

    def get_decentering(self) -> np.ndarray:
        """Get decentering parameters (p1, p2) as ndarray[2]."""
        return np.array([self.added_par.p1, self.added_par.p2], dtype=np.float64)

    def set_decentering(self, decent) -> None:
        """Set decentering from ndarray[2]."""
        if len(decent) != 2:
            raise ValueError("decentering parameters must have exactly 2 elements")
        self.added_par.p1 = float(decent[0])
        self.added_par.p2 = float(decent[1])

    def get_affine(self) -> np.ndarray:
        """Get affine parameters (scx, she) as ndarray[2]."""
        return np.array([self.added_par.scx, self.added_par.she], dtype=np.float64)

    def set_affine_trans(self, affine) -> None:
        """Set affine transform from ndarray[2]."""
        if len(affine) != 2:
            raise ValueError("affine parameters must have exactly 2 elements")
        self.added_par.scx = float(affine[0])
        self.added_par.she = float(affine[1])

    def get_glass_vec(self) -> np.ndarray:
        """Get glass vector (vec_x, vec_y, vec_z) as ndarray[3]."""
        return np.array([self.glass_par.vec_x, self.glass_par.vec_y, self.glass_par.vec_z], dtype=np.float64)

    def set_glass_vec(self, gvec) -> None:
        """Set glass vector from ndarray[3]."""
        if len(gvec) != 3:
            raise ValueError("glass vector must have exactly 3 elements")
        self.glass_par.vec_x = float(gvec[0])
        self.glass_par.vec_y = float(gvec[1])
        self.glass_par.vec_z = float(gvec[2])

    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix as ndarray[3, 3]."""
        return self.ext_par.dm.copy()

    def write(self, ori_file, add_file=None) -> None:
        """Write calibration to file(s)."""
        if isinstance(ori_file, bytes):
            ori_file = ori_file.decode('utf-8')
        if isinstance(add_file, bytes):
            add_file = add_file.decode('utf-8')
        self.to_file(ori_file, add_file)

    from_file = hybrid_from_file()

    @classmethod
    def _from_file_class(
        cls,
        ori_file: str | Path,
        add_file: str | Path | None = None,
        add_fallback: str | Path | None = None,
    ) -> Calibration:
        """Read calibration from orientation files.

        Args:
            ori_file: path to file with exterior, interior, glass parameters.
            add_file: path to file with distortion parameters.
            add_fallback: fallback path if add_file doesn't exist.

        Returns:
            Calibration instance.

        Raises:
            FileNotFoundError: if ori_file doesn't exist.
            ValueError: if file format is invalid.
        """
        # Accept bytes paths (the legacy liboptv/GUI API passes char* filenames,
        # e.g. cal_file.encode()). Path() rejects bytes, so decode first — mirrors
        # the symmetric handling already done in write().
        if isinstance(ori_file, bytes):
            ori_file = ori_file.decode('utf-8')
        if isinstance(add_file, bytes):
            add_file = add_file.decode('utf-8')
        if isinstance(add_fallback, bytes):
            add_fallback = add_fallback.decode('utf-8')

        ori_path = Path(ori_file)
        if not ori_path.exists():
            raise FileNotFoundError(f"ORI file not found: {ori_file}")

        lines = ori_path.read_text().strip().splitlines()
        idx = 0

        # Parse exterior: x0 y0 z0 omega phi kappa (may span 2 lines)
        first_line = lines[idx].strip()
        idx += 1
        parts = first_line.split()

        if len(parts) >= 6:
            x0, y0, z0, omega, phi, kappa = [float(p) for p in parts[:6]]
        elif len(parts) == 3:
            # Values split across lines
            x0, y0, z0 = [float(p) for p in parts]
            second_line = lines[idx].strip()
            idx += 1
            omega, phi, kappa = [float(p) for p in second_line.split()[:3]]
        else:
            raise ValueError(f"Invalid exterior orientation format: {first_line}")

        # Skip empty line if present
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        # Parse rotation matrix: 3 rows of 3 values
        dm = np.eye(3, dtype=np.float64)
        for i in range(3):
            row_parts = lines[idx].strip().split()
            idx += 1
            dm[i] = [float(row_parts[0]), float(row_parts[1]), float(row_parts[2])]

        # Skip empty lines
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        # Parse interior: xh yh cc (may span 2 lines)
        interior_line = lines[idx].strip()
        idx += 1
        interior_parts = interior_line.split()

        if len(interior_parts) >= 3:
            xh, yh, cc = [float(p) for p in interior_parts[:3]]
        elif len(interior_parts) == 2:
            xh, yh = [float(p) for p in interior_parts]
            cc_line = lines[idx].strip()
            idx += 1
            cc = float(cc_line.split()[0])
        else:
            raise ValueError(f"Invalid interior orientation format: {interior_line}")

        # Skip empty lines
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1

        # Parse glass: vec_x vec_y vec_z (may span lines)
        glass_line = lines[idx].strip()
        idx += 1
        glass_parts = glass_line.split()

        if len(glass_parts) >= 3:
            vec_x, vec_y, vec_z = [float(p) for p in glass_parts[:3]]
        else:
            raise ValueError(f"Invalid glass format: {glass_line}")

        # Parse additional parameters
        added_par = AddedPar()
        if add_file is not None:
            add_path = Path(add_file)
            if add_path.exists():
                add_lines = add_path.read_text().strip().split()
                if len(add_lines) >= 7:
                    added_par = AddedPar(
                        k1=float(add_lines[0]),
                        k2=float(add_lines[1]),
                        k3=float(add_lines[2]),
                        p1=float(add_lines[3]),
                        p2=float(add_lines[4]),
                        scx=float(add_lines[5]),
                        she=float(add_lines[6]),
                    )
            elif add_fallback is not None:
                fallback_path = Path(add_fallback)
                if fallback_path.exists():
                    fb_lines = fallback_path.read_text().strip().split()
                    if len(fb_lines) >= 7:
                        added_par = AddedPar(
                            k1=float(fb_lines[0]),
                            k2=float(fb_lines[1]),
                            k3=float(fb_lines[2]),
                            p1=float(fb_lines[3]),
                            p2=float(fb_lines[4]),
                            scx=float(fb_lines[5]),
                            she=float(fb_lines[6]),
                        )

        ext = Exterior(x0=x0, y0=y0, z0=z0, omega=omega, phi=phi, kappa=kappa, dm=dm)
        int_par = Interior(xh=xh, yh=yh, cc=cc)
        glass = Glass(vec_x=vec_x, vec_y=vec_y, vec_z=vec_z)

        return cls(
            ext_par=ext,
            int_par=int_par,
            glass_par=glass,
            added_par=added_par,
        )

    def to_file(
        self,
        ori_file: str | Path,
        add_file: str | Path | None = None,
    ) -> None:
        """Write calibration to orientation files.

        Args:
            ori_file: path to output file for exterior, interior, glass.
            add_file: path to output file for distortion parameters.
        """
        ori_path = Path(ori_file)
        ext = self.ext_par
        int_par = self.int_par
        glass = self.glass_par

        lines = [
            f"{ext.x0:11.8f} {ext.y0:11.8f} {ext.z0:11.8f}",
            f"    {ext.omega:10.8f}  {ext.phi:10.8f}  {ext.kappa:10.8f}",
            "",
        ]

        for i in range(3):
            lines.append(
                f"    {ext.dm[i, 0]:10.7f} {ext.dm[i, 1]:10.7f} {ext.dm[i, 2]:10.7f}"
            )

        lines.append("")
        lines.append(f"    {int_par.xh:8.4f} {int_par.yh:8.4f}")
        lines.append(f"    {int_par.cc:8.4f}")
        lines.append("")
        lines.append(
            f"    {glass.vec_x:20.15f} {glass.vec_y:20.15f}  {glass.vec_z:20.15f}"
        )

        ori_path.write_text("\n".join(lines) + "\n")

        if add_file is not None:
            add_path = Path(add_file)
            ap = self.added_par
            add_lines = [
                f"{ap.k1:.8f} {ap.k2:.8f} {ap.k3:.8f} "
                f"{ap.p1:.8f} {ap.p2:.8f} {ap.scx:.8f} {ap.she:.8f}"
            ]
            add_path.write_text("\n".join(add_lines) + "\n")


@cython.ccall
def compare_exterior(e1: Exterior, e2: Exterior) -> bool:
    """Compare two Exterior objects for equality (all fields, including dm)."""
    if not np.allclose(e1.dm, e2.dm):
        return False
    return (
        e1.x0 == e2.x0 and
        e1.y0 == e2.y0 and
        e1.z0 == e2.z0 and
        e1.omega == e2.omega and
        e1.phi == e2.phi and
        e1.kappa == e2.kappa
    )

@cython.ccall
def compare_interior(i1: Interior, i2: Interior) -> bool:
    """Compare two Interior objects for equality."""
    return (
        i1.xh == i2.xh and
        i1.yh == i2.yh and
        i1.cc == i2.cc
    )

@cython.ccall
def compare_glass(g1: Glass, g2: Glass) -> bool:
    """Compare two Glass objects for equality (only normal vector)."""
    return (
        g1.vec_x == g2.vec_x and
        g1.vec_y == g2.vec_y and
        g1.vec_z == g2.vec_z
    )

@cython.ccall
def compare_addpar(a1: AddedPar, a2: AddedPar) -> bool:
    """Compare two AddedPar (distortion) objects for equality."""
    return (
        a1.k1 == a2.k1 and
        a1.k2 == a2.k2 and
        a1.k3 == a2.k3 and
        a1.p1 == a2.p1 and
        a1.p2 == a2.p2 and
        a1.scx == a2.scx and
        a1.she == a2.she
    )

@cython.ccall
def compare_calib(c1: Calibration, c2: Calibration) -> bool:
    """Deep comparison of two Calibration objects (all fields except mmlut)."""
    return (
        compare_exterior(c1.ext_par, c2.ext_par)
        and compare_interior(c1.int_par, c2.int_par)
        and compare_glass(c1.glass_par, c2.glass_par)
        and compare_addpar(c1.added_par, c2.added_par)
    )

# read_calibration and write_calibration are already covered by from_file and to_file methods.
# The rotation_matrix logic is implemented in Exterior.compute_rotation_matrix().


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
