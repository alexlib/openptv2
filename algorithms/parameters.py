"""PTV configuration parameters as clean dataclasses.

Translation of lib/src/parameters.c and lib/include/parameters.h.

Design principles:
- Pure dataclasses with direct field access (no getters/setters)
- File I/O separated into class methods (from_file / to_file)
- No coupling to C structures or adapter layers
- Named constants for defaults and magic numbers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class SequencePar:
    """Sequence parameters: image files and frame range.

    Attributes:
        num_cams: number of cameras.
        img_base_name: base names for image files (one per camera).
        first: first frame number.
        last: last frame number.
    """
    num_cams: int
    img_base_name: list[str] = field(default_factory=list)
    first: int = 0
    last: int = 0

    @classmethod
    def from_file(cls, filename: str | Path, num_cams: int) -> SequencePar:
        """Read sequence parameters from file.

        File format: first num_cams lines are image base names,
        then first frame, then last frame.

        Args:
            filename: path to parameter file.
            num_cams: number of cameras.

        Returns:
            SequencePar instance.

        Raises:
            FileNotFoundError: if file doesn't exist.
            ValueError: if file format is invalid.
        """
        path = Path(filename)
        lines = path.read_text().strip().splitlines()

        if len(lines) < num_cams + 2:
            raise ValueError(
                f"Expected at least {num_cams + 2} lines, got {len(lines)}"
            )

        img_base_name = [lines[i].strip() for i in range(num_cams)]
        first = int(lines[num_cams].strip())
        last = int(lines[num_cams + 1].strip())

        return cls(num_cams=num_cams, img_base_name=img_base_name, first=first, last=last)


@dataclass
class TrackPar:
    """Tracking parameters: search bounds and thresholds.

    Attributes:
        dvxmin, dvxmax: search bounds in x direction.
        dvymin, dvymax: search bounds in y direction.
        dvzmin, dvzmax: search bounds in z direction.
        dangle: maximum angle change threshold.
        dacc: maximum acceleration threshold.
        add: additional flag.
    """
    dvxmin: float = 0.0
    dvxmax: float = 0.0
    dvymin: float = 0.0
    dvymax: float = 0.0
    dvzmin: float = 0.0
    dvzmax: float = 0.0
    dangle: float = 0.0
    dacc: float = 0.0
    add: int = 0

    # Derived fields (set to 0 by default, not read from file)
    dsumg: int = 0
    dn: int = 0
    dnx: int = 0
    dny: int = 0

    @classmethod
    def from_file(cls, filename: str | Path) -> TrackPar:
        """Read tracking parameters from file.

        File format: 9 lines with values in order:
        dvxmin, dvxmax, dvymin, dvymax, dvzmin, dvzmax, dangle, dacc, add

        Args:
            filename: path to parameter file.

        Returns:
            TrackPar instance.

        Raises:
            FileNotFoundError: if file doesn't exist.
            ValueError: if file format is invalid.
        """
        path = Path(filename)
        lines = path.read_text().strip().splitlines()

        if len(lines) < 9:
            raise ValueError(f"Expected 9 lines, got {len(lines)}")

        return cls(
            dvxmin=float(lines[0]),
            dvxmax=float(lines[1]),
            dvymin=float(lines[2]),
            dvymax=float(lines[3]),
            dvzmin=float(lines[4]),
            dvzmax=float(lines[5]),
            dangle=float(lines[6]),
            dacc=float(lines[7]),
            add=int(lines[8]),
        )


@dataclass
class VolumePar:
    """Illuminated volume boundaries and correspondence criteria.

    Attributes:
        X_lay: leftmost and rightmost X boundaries [mm].
        Zmin_lay: closest Z points for each side [mm].
        Zmax_lay: farthest Z points for each side [mm].
        cnx: correlation limit for nx size.
        cny: correlation limit for ny size.
        cn: correlation limit for n particle size.
        csumg: correlation limit for sum of grey scale.
        corrmin: minimum overall correlation.
        eps0: flat coordinates tolerance [mm].
    """
    X_lay: tuple[float, float] = (0.0, 0.0)
    Zmin_lay: tuple[float, float] = (0.0, 0.0)
    Zmax_lay: tuple[float, float] = (0.0, 0.0)
    cnx: float = 0.0
    cny: float = 0.0
    cn: float = 0.0
    csumg: float = 0.0
    corrmin: float = 0.0
    eps0: float = 0.0

    @classmethod
    def from_file(cls, filename: str | Path) -> VolumePar:
        """Read volume parameters from file.

        File format: 12 lines with values in order:
        X_lay[0], Zmin_lay[0], Zmax_lay[0], X_lay[1], Zmin_lay[1], Zmax_lay[1],
        cnx, cny, cn, csumg, corrmin, eps0

        Args:
            filename: path to parameter file.

        Returns:
            VolumePar instance.

        Raises:
            FileNotFoundError: if file doesn't exist.
            ValueError: if file format is invalid.
        """
        path = Path(filename)
        lines = path.read_text().strip().splitlines()

        if len(lines) < 12:
            raise ValueError(f"Expected 12 lines, got {len(lines)}")

        return cls(
            X_lay=(float(lines[0]), float(lines[3])),
            Zmin_lay=(float(lines[1]), float(lines[4])),
            Zmax_lay=(float(lines[2]), float(lines[5])),
            cnx=float(lines[6]),
            cny=float(lines[7]),
            cn=float(lines[8]),
            csumg=float(lines[9]),
            corrmin=float(lines[10]),
            eps0=float(lines[11]),
        )


@dataclass
class MmNp:
    """Multimedia model: refractive indices and layer thicknesses.

    Attributes:
        nlay: number of layers.
        n1: refractive index of first medium (air ~ 1.0).
        n2: refractive indices of second medium (glass windows), up to 3 layers.
        d: thicknesses of second medium layers, up to 3 layers.
        n3: refractive index of third medium (flowing fluid).
    """
    nlay: int = 1
    n1: float = 1.0
    n2: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    d: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    n3: float = 1.0


@dataclass
class ControlPar:
    """General control parameters: image dimensions, multimedia, camera count.

    Attributes:
        num_cams: number of cameras.
        img_base_name: image file base names (one per camera).
        cal_img_base_name: calibration image file base names.
        hp_flag: high-pass filter flag (0/1).
        allCam_flag: flag for using particles matched in all cameras.
        tiff_flag: use TIFF headers (1) or raw images (0).
        imx: horizontal image size in pixels.
        imy: vertical image size in pixels.
        pix_x: pixel width [mm].
        pix_y: pixel height [mm].
        chfield: interlaced mode (0=whole, 1=upper half, 2=lower half).
        mm: multimedia model.
    """
    num_cams: int = 0
    img_base_name: list[str] = field(default_factory=list)
    cal_img_base_name: list[str] = field(default_factory=list)
    hp_flag: int = 0
    allCam_flag: int = 0
    tiff_flag: int = 0
    imx: int = 0
    imy: int = 0
    pix_x: float = 0.0
    pix_y: float = 0.0
    chfield: int = 0
    mm: MmNp = field(default_factory=MmNp)

    @classmethod
    def from_file(cls, filename: str | Path) -> ControlPar:
        """Read control parameters from file.

        File format (21 lines regardless of camera count):
        1. num_cams
        2n. img_base_name for camera n
        2n+1. cal_img_base_name for camera n
        10. hp_flag
        11. allCam_flag
        12. tiff_flag
        13. imx
        14. imy
        15. pix_x
        16. pix_y
        17. chfield
        18. mm.n1
        19. mm.n2[0]
        20. mm.n3
        21. mm.d[0]

        Args:
            filename: path to parameter file.

        Returns:
            ControlPar instance.

        Raises:
            FileNotFoundError: if file doesn't exist.
            ValueError: if file format is invalid.
        """
        path = Path(filename)
        lines = path.read_text().strip().splitlines()

        if len(lines) < 1:
            raise ValueError("Empty control parameter file")

        idx = 0
        num_cams = int(lines[idx].strip())
        idx += 1

        img_base_name = []
        cal_img_base_name = []
        for cam in range(num_cams):
            img_base_name.append(lines[idx].strip())
            idx += 1
            cal_img_base_name.append(lines[idx].strip())
            idx += 1

        hp_flag = int(lines[idx].strip())
        idx += 1
        allCam_flag = int(lines[idx].strip())
        idx += 1
        tiff_flag = int(lines[idx].strip())
        idx += 1
        imx = int(lines[idx].strip())
        idx += 1
        imy = int(lines[idx].strip())
        idx += 1
        pix_x = float(lines[idx].strip())
        idx += 1
        pix_y = float(lines[idx].strip())
        idx += 1
        chfield = int(lines[idx].strip())
        idx += 1
        n1 = float(lines[idx].strip())
        idx += 1
        n2_0 = float(lines[idx].strip())
        idx += 1
        n3 = float(lines[idx].strip())
        idx += 1
        d0 = float(lines[idx].strip())
        idx += 1

        mm = MmNp(nlay=1, n1=n1, n2=[n2_0, 1.0, 1.0], d=[d0, 0.0, 0.0], n3=n3)

        return cls(
            num_cams=num_cams,
            img_base_name=img_base_name,
            cal_img_base_name=cal_img_base_name,
            hp_flag=hp_flag,
            allCam_flag=allCam_flag,
            tiff_flag=tiff_flag,
            imx=imx,
            imy=imy,
            pix_x=pix_x,
            pix_y=pix_y,
            chfield=chfield,
            mm=mm,
        )


@dataclass
class TargetPar:
    """Target recognition parameters: detection thresholds and size limits.

    Attributes:
        discont: maximum discontinuity.
        gvthres: grey value thresholds for binarization (one per camera, up to 4).
        nnmin, nnmax: min/max number of pixels per target.
        nxmin, nxmax: min/max pixels in x direction.
        nymin, nymax: min/max pixels in y direction.
        sumg_min: minimum sum of grey values.
        cr_sz: size of crosses.
    """
    discont: int = 0
    gvthres: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    nnmin: int = 0
    nnmax: int = 0
    nxmin: int = 0
    nxmax: int = 0
    nymin: int = 0
    nymax: int = 0
    sumg_min: int = 0
    cr_sz: int = 0

    @classmethod
    def from_file(cls, filename: str | Path) -> TargetPar:
        """Read target recognition parameters from file.

        File format:
        gvthres[0..3] (4 lines)
        discont (1 line)
        nnmin nnmax (1 line)
        nxmin nxmax (1 line)
        nymin nymax (1 line)
        sumg_min (1 line)
        cr_sz (1 line)

        Args:
            filename: path to parameter file.

        Returns:
            TargetPar instance.

        Raises:
            FileNotFoundError: if file doesn't exist.
            ValueError: if file format is invalid.
        """
        path = Path(filename)
        lines = path.read_text().strip().splitlines()

        if len(lines) < 9:
            raise ValueError(f"Expected 9 lines, got {len(lines)}")

        gvthres = [int(lines[i].strip()) for i in range(4)]
        discont = int(lines[4].strip())

        nn_parts = lines[5].strip().split()
        nnmin, nnmax = int(nn_parts[0]), int(nn_parts[1])

        nx_parts = lines[6].strip().split()
        nxmin, nxmax = int(nx_parts[0]), int(nx_parts[1])

        ny_parts = lines[7].strip().split()
        nymin, nymax = int(ny_parts[0]), int(ny_parts[1])

        sumg_min = int(lines[8].strip())
        cr_sz = int(lines[9].strip()) if len(lines) > 9 else 0

        return cls(
            discont=discont,
            gvthres=gvthres,
            nnmin=nnmin,
            nnmax=nnmax,
            nxmin=nxmin,
            nxmax=nxmax,
            nymin=nymin,
            nymax=nymax,
            sumg_min=sumg_min,
            cr_sz=cr_sz,
        )

    def to_file(self, filename: str | Path) -> None:
        """Write target recognition parameters to file.

        Args:
            filename: path to output file.
        """
        path = Path(filename)
        lines = [
            str(self.gvthres[0]),
            str(self.gvthres[1]),
            str(self.gvthres[2]),
            str(self.gvthres[3]),
            str(self.discont),
            f"{self.nnmin} {self.nnmax}",
            f"{self.nxmin} {self.nxmax}",
            f"{self.nymin} {self.nymax}",
            str(self.sumg_min),
            str(self.cr_sz),
        ]
        path.write_text("\n".join(lines) + "\n")
