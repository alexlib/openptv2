# -*- coding: utf-8 -*-
"""Camera and tracking sequence parameters and file I/O.

Translation of parameters from old C/C++ representation into modern Python.
"""
from __future__ import annotations

import cython
from collections import namedtuple
from pathlib import Path
from typing import Optional, List, Union

import numpy as np

# TrackParTuple for test compatibility
TrackParTuple = namedtuple('TrackParTuple', [
    'dvxmin', 'dvxmax', 'dvymin', 'dvymax', 'dvzmin', 'dvzmax',
    'dangle', 'dacc', 'add', 'dsumg', 'dn', 'dnx', 'dny'
])


# Convert TrackPar to TrackParTuple for test compatibility
@cython.ccall
def convert_track_par_to_tuple(track_par: TrackPar) -> TrackParTuple:
    return TrackParTuple(
        track_par.dvxmin, track_par.dvxmax, track_par.dvymin, track_par.dvymax,
        track_par.dvzmin, track_par.dvzmax, track_par.dangle, track_par.dacc,
        track_par.add, getattr(track_par, 'dsumg', 0.0), getattr(track_par, 'dn', 0.0),
        getattr(track_par, 'dnx', 0.0), getattr(track_par, 'dny', 0.0)
    )


class SequencePar:
    num_cams: int
    img_base_name: list[str]
    first: int
    last: int

    def __init__(self, num_cams: int = 0, img_base_name: list[str] | None = None, first: int = 0, last: int = 0) -> None:
        self.num_cams = num_cams
        self.img_base_name = img_base_name if img_base_name is not None else []
        self.first = first
        self.last = last

    @staticmethod
    def from_file(filename: str | Path, num_cams: int) -> SequencePar:
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        if len(lines) < num_cams + 2:
            raise ValueError(f"Expected at least {num_cams + 2} lines, got {len(lines)}")
        img_base_name = [lines[i].strip() for i in range(num_cams)]
        first = int(lines[num_cams].strip())
        last = int(lines[num_cams + 1].strip())
        return SequencePar(num_cams, img_base_name, first, last)


class TrackPar:
    dvxmin: float
    dvxmax: float
    dvymin: float
    dvymax: float
    dvzmin: float
    dvzmax: float
    dangle: float
    dacc: float
    add: int
    track_mode: int
    dsumg: int
    dn: int
    dnx: int
    dny: int

    def __init__(
        self,
        dvxmin: float = 0.0,
        dvxmax: float = 0.0,
        dvymin: float = 0.0,
        dvymax: float = 0.0,
        dvzmin: float = 0.0,
        dvzmax: float = 0.0,
        dangle: float = 0.0,
        dacc: float = 0.0,
        add: int = 0,
        track_mode: int = 0
    ) -> None:
        self.dvxmin = dvxmin
        self.dvxmax = dvxmax
        self.dvymin = dvymin
        self.dvymax = dvymax
        self.dvzmin = dvzmin
        self.dvzmax = dvzmax
        self.dangle = dangle
        self.dacc = dacc
        self.add = add
        self.track_mode = track_mode
        self.dsumg = 0
        self.dn = 0
        self.dnx = 0
        self.dny = 0

    @staticmethod
    def from_file(filename: str | Path) -> TrackPar:
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        if len(lines) < 9:
            raise ValueError(f"Expected 9 lines, got {len(lines)}")
        track_mode = 0
        if len(lines) >= 10:
            try:
                track_mode = int(lines[9].strip())
            except (ValueError, IndexError):
                track_mode = 0
        return TrackPar(
            float(lines[0]), float(lines[1]), float(lines[2]), float(lines[3]),
            float(lines[4]), float(lines[5]), float(lines[6]), float(lines[7]), int(lines[8]),
            track_mode
        )


class VolumePar:
    X_lay: np.ndarray
    Zmin_lay: np.ndarray
    Zmax_lay: np.ndarray
    cnx: float
    cny: float
    cn: float
    csumg: float
    corrmin: float
    eps0: float

    def __init__(
        self,
        X_lay: list[float] | np.ndarray | None = None,
        Zmin_lay: list[float] | np.ndarray | None = None,
        Zmax_lay: list[float] | np.ndarray | None = None,
        cnx: float = 0.0,
        cny: float = 0.0,
        cn: float = 0.0,
        csumg: float = 0.0,
        corrmin: float = 0.0,
        eps0: float = 0.0
    ) -> None:
        self.X_lay = np.zeros(2) if X_lay is None else np.array(X_lay, dtype=np.float64)
        self.Zmin_lay = np.zeros(2) if Zmin_lay is None else np.array(Zmin_lay, dtype=np.float64)
        self.Zmax_lay = np.zeros(2) if Zmax_lay is None else np.array(Zmax_lay, dtype=np.float64)
        self.cnx = cnx
        self.cny = cny
        self.cn = cn
        self.csumg = csumg
        self.corrmin = corrmin
        self.eps0 = eps0

    @staticmethod
    def from_file(filename: str | Path) -> VolumePar:
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        if len(lines) < 12:
            raise ValueError(f"Expected 12 lines, got {len(lines)}")
        X_lay = [float(lines[0]), float(lines[3])]
        Zmin_lay = [float(lines[1]), float(lines[4])]
        Zmax_lay = [float(lines[2]), float(lines[5])]
        cnx = float(lines[6])
        cny = float(lines[7])
        cn = float(lines[8])
        csumg = float(lines[9])
        corrmin = float(lines[10])
        eps0 = float(lines[11])
        return VolumePar(X_lay, Zmin_lay, Zmax_lay, cnx, cny, cn, csumg, corrmin, eps0)


class MmNp:
    nlay: int
    n1: float
    n2: np.ndarray
    d: np.ndarray
    n3: float

    def __init__(
        self,
        nlay: int = 1,
        n1: float = 1.0,
        n2: list[float] | np.ndarray | None = None,
        d: list[float] | np.ndarray | None = None,
        n3: float = 1.0
    ) -> None:
        self.nlay = nlay
        self.n1 = n1
        self.n2 = np.ones(3) if n2 is None else np.array(n2, dtype=np.float64)
        self.d = np.zeros(3) if d is None else np.array(d, dtype=np.float64)
        self.n3 = n3


class ControlPar:
    num_cams: int
    img_base_name: list[str]
    cal_img_base_name: list[str]
    hp_flag: int
    allCam_flag: int
    tiff_flag: int
    imx: int
    imy: int
    pix_x: float
    pix_y: float
    chfield: int
    mm: MmNp

    def __init__(
        self,
        num_cams: int = 0,
        img_base_name: list[str] | None = None,
        cal_img_base_name: list[str] | None = None,
        hp_flag: int = 0,
        allCam_flag: int = 0,
        all_cam_flag: int | None = None,
        tiff_flag: int = 0,
        imx: int = 0,
        imy: int = 0,
        pix_x: float = 0.0,
        pix_y: float = 0.0,
        chfield: int = 0,
        mm: MmNp | None = None
    ) -> None:
        self.num_cams = num_cams
        self.img_base_name = img_base_name if img_base_name is not None else []
        self.cal_img_base_name = cal_img_base_name if cal_img_base_name is not None else []
        self.hp_flag = hp_flag
        # Accept both allCam_flag and all_cam_flag for compatibility
        if all_cam_flag is not None:
            self.allCam_flag = all_cam_flag
        else:
            self.allCam_flag = allCam_flag
        self.tiff_flag = tiff_flag
        self.imx = imx
        self.imy = imy
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.chfield = chfield
        self.mm = mm if mm is not None else MmNp()

    @staticmethod
    def from_file(filename: str | Path) -> ControlPar:
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
        return ControlPar(
            num_cams=num_cams, img_base_name=img_base_name,
            cal_img_base_name=cal_img_base_name, hp_flag=hp_flag,
            allCam_flag=allCam_flag, tiff_flag=tiff_flag,
            imx=imx, imy=imy, pix_x=pix_x, pix_y=pix_y,
            chfield=chfield, mm=mm,
        )


class TargetPar:
    gvthres: np.ndarray
    discont: int
    nnmin: int
    nnmax: int
    nxmin: int
    nxmax: int
    nymin: int
    nymax: int
    sumg_min: int
    cr_sz: int

    def __init__(
        self,
        gvthres: list[int] | np.ndarray | None = None,
        discont: int = 0,
        nnmin: int = 0,
        nnmax: int = 0,
        nxmin: int = 0,
        nxmax: int = 0,
        nymin: int = 0,
        nymax: int = 0,
        sumg_min: int = 0,
        cr_sz: int = 0
    ) -> None:
        self.gvthres = np.zeros(4, dtype=int) if gvthres is None else np.array(gvthres, dtype=int)
        self.discont = discont
        self.nnmin = nnmin
        self.nnmax = nnmax
        self.nxmin = nxmin
        self.nxmax = nxmax
        self.nymin = nymin
        self.nymax = nymax
        self.sumg_min = sumg_min
        self.cr_sz = cr_sz

    @staticmethod
    def from_file(filename: str | Path) -> TargetPar:
        path = Path(filename)
        tokens = path.read_text().split()
        if len(tokens) < 12:
            raise ValueError(f"Expected at least 12 values, got {len(tokens)}")
        gvthres = [int(tokens[i]) for i in range(4)]
        discont = int(tokens[4])
        nnmin, nnmax = int(tokens[5]), int(tokens[6])
        nxmin, nxmax = int(tokens[7]), int(tokens[8])
        nymin, nymax = int(tokens[9]), int(tokens[10])
        sumg_min = int(tokens[11])
        cr_sz = int(tokens[12]) if len(tokens) > 12 else 0
        return TargetPar(gvthres, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min, cr_sz)

    def to_file(self, filename: str | Path) -> None:
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


class OrientPar:
    useflag: int
    ccflag: int
    xhflag: int
    yhflag: int
    k1flag: int
    k2flag: int
    k3flag: int
    p1flag: int
    p2flag: int
    scxflag: int
    sheflag: int
    interfflag: int

    def __init__(
        self,
        useflag: int = 0,
        ccflag: int = 0,
        xhflag: int = 0,
        yhflag: int = 0,
        k1flag: int = 0,
        k2flag: int = 0,
        k3flag: int = 0,
        p1flag: int = 0,
        p2flag: int = 0,
        scxflag: int = 0,
        sheflag: int = 0,
        interfflag: int = 0
    ) -> None:
        self.useflag = useflag
        self.ccflag = ccflag
        self.xhflag = xhflag
        self.yhflag = yhflag
        self.k1flag = k1flag
        self.k2flag = k2flag
        self.k3flag = k3flag
        self.p1flag = p1flag
        self.p2flag = p2flag
        self.scxflag = scxflag
        self.sheflag = sheflag
        self.interfflag = interfflag

    @staticmethod
    def from_file(filename: str | Path) -> OrientPar:
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        flags = [int(line.strip()) for line in lines[:12]]
        while len(flags) < 12:
            flags.append(0)
        return OrientPar(*flags)


class MultimediaPar:
    n1: float
    n2: np.ndarray
    d: np.ndarray
    n3: float
    nlay: int

    def __init__(
        self,
        n1: float = 1.0,
        n2: list[float] | np.ndarray | None = None,
        d: list[float] | np.ndarray | None = None,
        n3: float = 1.0,
        nlay: int = 1
    ) -> None:
        self.n1 = n1
        self.n2 = np.ones(3) if n2 is None else np.array(n2, dtype=np.float64)
        self.d = np.zeros(3) if d is None else np.array(d, dtype=np.float64)
        self.n3 = n3
        self.nlay = nlay


class CalibrationPar:
    """Calibration parameters for calibration workflow."""
    fixp_name: str
    img_cal_name: list[str]
    img_ori: list[str]
    tiff_flag: int
    pair_flag: int
    chfield: int

    def __init__(
        self,
        fixp_name: str = "",
        img_cal_name: list[str] | None = None,
        img_ori: list[str] | None = None,
        tiff_flag: int = 0,
        pair_flag: int = 0,
        chfield: int = 0
    ) -> None:
        self.fixp_name = fixp_name
        self.img_cal_name = img_cal_name if img_cal_name is not None else []
        self.img_ori = img_ori if img_ori is not None else []
        self.tiff_flag = tiff_flag
        self.pair_flag = pair_flag
        self.chfield = chfield

    @staticmethod
    def from_file(file_path: str | Path, num_cams: int) -> CalibrationPar:
        """Read from cal_ori.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            fixp_name = file.readline().strip()
            tmp = [file.readline().strip() for _ in range(num_cams * 2)]
            img_cal_name = tmp[0::2]
            img_ori = tmp[1::2]
            tiff_flag = int(file.readline().strip())
            pair_flag = int(file.readline().strip())
            chfield = int(file.readline().strip())

        return CalibrationPar(fixp_name, img_cal_name, img_ori, tiff_flag, pair_flag, chfield)


class MultiPlanesPar:
    """Multiplanes parameters."""
    num_planes: int
    filename: list[str]

    def __init__(self, num_planes: int = 0, filename: list[str] | None = None) -> None:
        self.num_planes = num_planes
        self.filename = filename if filename is not None else []

    @staticmethod
    def from_file(file_path: str | Path) -> MultiPlanesPar:
        """Read from multiplanes.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            num_planes = int(file.readline().strip())
            filename = [file.readline().strip() for _ in range(num_planes)]
        return MultiPlanesPar(num_planes, filename)


class ExaminePar:
    """Examine parameters."""
    examine_flag: bool
    combine_flag: bool

    def __init__(self, examine_flag: bool = False, combine_flag: bool = False) -> None:
        self.examine_flag = examine_flag
        self.combine_flag = combine_flag

    @staticmethod
    def from_file(file_path: str | Path) -> ExaminePar:
        """Read from examine.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            examine_flag = bool(int(file.readline().strip()))
            combine_flag = bool(int(file.readline().strip()))
        return ExaminePar(examine_flag, combine_flag)


class PftVersionPar:
    """PFT version parameters."""
    existing_target_flag: bool

    def __init__(self, existing_target_flag: bool = False) -> None:
        self.existing_target_flag = existing_target_flag

    @staticmethod
    def from_file(file_path: str | Path) -> PftVersionPar:
        """Read from pft_version.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            existing_target_flag = bool(int(file.readline().strip()))
        return PftVersionPar(existing_target_flag)


# Aliases for compatibility with legacy test code (must be after all class definitions)
read_control_par = ControlPar.from_file
read_volume_par = VolumePar.from_file
read_sequence_par = SequencePar.from_file
read_track_par = TrackPar.from_file


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
