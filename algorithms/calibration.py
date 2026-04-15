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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Exterior:
    """Exterior orientation: camera position and orientation.

    Attributes:
        x0, y0, z0: camera center position [mm].
        omega, phi, kappa: rotation angles [radians].
        dm: 3x3 rotation matrix (computed from angles).
    """
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    omega: float = 0.0
    phi: float = 0.0
    kappa: float = 0.0
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


@dataclass
class Interior:
    """Interior orientation: principal point and camera constant.

    Attributes:
        xh, yh: principal point (sensor shift) [mm].
        cc: camera constant (focal length) [mm].
    """
    xh: float = 0.0
    yh: float = 0.0
    cc: float = 0.0


@dataclass
class Glass:
    """Glass interface parameters.

    Attributes:
        vec_x, vec_y, vec_z: normal vector to glass surface.
        n1, n2, n3: refractive indices (not used directly, stored for reference).
        d: glass thickness [mm].
    """
    vec_x: float = 0.0
    vec_y: float = 0.0
    vec_z: float = 0.0
    n1: float = 0.0
    n2: float = 0.0
    n3: float = 0.0
    d: float = 0.0


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
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    scx: float = 1.0
    she: float = 0.0
    field: int = 0


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
    nr: int = 0
    nz: int = 0
    rw: float = 2.0
    data: Optional[np.ndarray] = None  # 1D array of size nr * nz

    @property
    def is_initialized(self) -> bool:
        """Check if the LUT has been populated."""
        return self.data is not None


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

    def __post_init__(self):
        """Ensure rotation matrix is computed after initialization."""
        if self.ext_par is not None:
            self.ext_par.compute_rotation_matrix()

    @classmethod
    def from_file(
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
