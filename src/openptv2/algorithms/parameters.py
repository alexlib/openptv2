# -*- coding: utf-8 -*-
"""Camera and tracking sequence parameters and file I/O.

Translation of parameters from old C/C++ representation into modern Python.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import cython
import numpy as np


def _load_yaml_params(filename: str | Path) -> dict:
    """Load an experiment YAML into a plain dict (YAML-only parameter I/O).

    Used by the ``from_yaml`` constructors below so parameter objects can be
    built directly from the single experiment YAML, without any legacy .par
    files.
    """
    import yaml

    with open(Path(filename), "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{filename} is not a valid parameter YAML")
    return data


def _clean_name_list(names, num_cams: int) -> list[str]:
    """Drop placeholder entries ('---'/'--') and pad to num_cams."""
    out = [str(n) for n in (names or []) if str(n) not in ("---", "--", "")]
    if not out and num_cams > 0:
        out = [""] * num_cams
    return out


# TrackParTuple for test compatibility
TrackParTuple = namedtuple(
    "TrackParTuple",
    [
        "dvxmin",
        "dvxmax",
        "dvymin",
        "dvymax",
        "dvzmin",
        "dvzmax",
        "dangle",
        "dacc",
        "add",
        "dsumg",
        "dn",
        "dnx",
        "dny",
    ],
)


# Convert TrackPar to TrackParTuple for test compatibility
@cython.ccall
def convert_track_par_to_tuple(track_par: object) -> object:
    return TrackParTuple(
        track_par.dvxmin,
        track_par.dvxmax,
        track_par.dvymin,
        track_par.dvymax,
        track_par.dvzmin,
        track_par.dvzmax,
        track_par.dangle,
        track_par.dacc,
        track_par.add,
        getattr(track_par, "dsumg", 0.0),
        getattr(track_par, "dn", 0.0),
        getattr(track_par, "dnx", 0.0),
        getattr(track_par, "dny", 0.0),
    )


@cython.cclass
class SequencePar:
    num_cams: cython.int = cython.declare(cython.int, visibility="public")
    img_base_name: object = cython.declare(object, visibility="public")
    first: cython.int = cython.declare(cython.int, visibility="public")
    last: cython.int = cython.declare(cython.int, visibility="public")

    def __init__(
        self,
        num_cams: int = 0,
        img_base_name: list[str] | None = None,
        first: int = 0,
        last: int = 0,
        # optv-compatible aliases (used via the tests' `optv` -> openptv2 shim)
        image_base: list[str] | None = None,
        frame_range: tuple[int, int] | None = None,
    ) -> None:
        if image_base is not None:
            img_base_name = image_base
        if frame_range is not None:
            first, last = frame_range
        self.num_cams = num_cams
        self.img_base_name = img_base_name if img_base_name is not None else []
        if not self.img_base_name and num_cams > 0:
            self.img_base_name = [""] * num_cams
        self.first = first
        self.last = last

    @staticmethod
    def from_file(filename: str | Path, num_cams: int) -> SequencePar:
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        if len(lines) < num_cams + 2:
            raise ValueError(
                f"Expected at least {num_cams + 2} lines, got {len(lines)}"
            )
        img_base_name = [lines[i].strip() for i in range(num_cams)]
        first = int(lines[num_cams].strip())
        last = int(lines[num_cams + 1].strip())
        return SequencePar(num_cams, img_base_name, first, last)

    @staticmethod
    def from_yaml(filename: str | Path, num_cams: int | None = None) -> SequencePar:
        """Build a SequencePar from an experiment YAML (YAML-only)."""
        data = _load_yaml_params(filename)
        seq = data.get("sequence", {})
        nc = int(num_cams if num_cams is not None else data.get("num_cams", 0))
        return SequencePar(
            num_cams=nc,
            img_base_name=_clean_name_list(seq.get("base_name"), nc),
            first=int(seq.get("first", 0)),
            last=int(seq.get("last", 0)),
        )

    # --- Backward Compatibility OOP Methods ---
    def get_first(self) -> int:
        return self.first

    def set_first(self, first: int) -> None:
        self.first = int(first)

    def get_last(self) -> int:
        return self.last

    def set_last(self, last: int) -> None:
        self.last = int(last)

    def get_img_base_name(self, cam: int) -> str:
        try:
            return self.img_base_name[cam]
        except IndexError:
            return ""

    def set_img_base_name(self, cam: int, name: str) -> None:
        while len(self.img_base_name) <= cam:
            self.img_base_name.append("")
        self.img_base_name[cam] = str(name)

    def read_sequence_par(self, filename: str | Path, num_cams: int) -> None:
        """Read sequence parameters from file in-place."""
        new_par = SequencePar.from_file(filename, num_cams)
        self.num_cams = new_par.num_cams
        self.img_base_name = new_par.img_base_name
        self.first = new_par.first
        self.last = new_par.last


@cython.cclass
class TrackPar:
    dvxmin: cython.double = cython.declare(cython.double, visibility="public")
    dvxmax: cython.double = cython.declare(cython.double, visibility="public")
    dvymin: cython.double = cython.declare(cython.double, visibility="public")
    dvymax: cython.double = cython.declare(cython.double, visibility="public")
    dvzmin: cython.double = cython.declare(cython.double, visibility="public")
    dvzmax: cython.double = cython.declare(cython.double, visibility="public")
    dangle: cython.double = cython.declare(cython.double, visibility="public")
    dacc: cython.double = cython.declare(cython.double, visibility="public")
    add: cython.int = cython.declare(cython.int, visibility="public")
    track_mode: cython.int = cython.declare(cython.int, visibility="public")
    dsumg: cython.double = cython.declare(cython.double, visibility="public")
    dn: cython.double = cython.declare(cython.double, visibility="public")
    dnx: cython.double = cython.declare(cython.double, visibility="public")
    dny: cython.double = cython.declare(cython.double, visibility="public")

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
        track_mode: int = 0,
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
        self.dsumg = 0.0
        self.dn = 0.0
        self.dnx = 0.0
        self.dny = 0.0

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
            float(lines[0]),
            float(lines[1]),
            float(lines[2]),
            float(lines[3]),
            float(lines[4]),
            float(lines[5]),
            float(lines[6]),
            float(lines[7]),
            int(lines[8]),
            track_mode,
        )

    @staticmethod
    def from_yaml(filename: str | Path) -> TrackPar:
        """Build a TrackPar from an experiment YAML (YAML-only)."""
        t = _load_yaml_params(filename).get("track", {})
        # 'angle' is the YAML name for dangle; 'add'/'flagNewParticles' -> add.
        add = t.get("add", t.get("flagNewParticles", 0))
        return TrackPar(
            dvxmin=float(t.get("dvxmin", 0.0)),
            dvxmax=float(t.get("dvxmax", 0.0)),
            dvymin=float(t.get("dvymin", 0.0)),
            dvymax=float(t.get("dvymax", 0.0)),
            dvzmin=float(t.get("dvzmin", 0.0)),
            dvzmax=float(t.get("dvzmax", 0.0)),
            dangle=float(t.get("angle", t.get("dangle", 0.0))),
            dacc=float(t.get("dacc", 0.0)),
            add=int(bool(add)),
            track_mode=int(t.get("track_mode", 0)),
        )

    # --- Backward Compatibility OOP Methods ---
    def get_dvxmin(self) -> float:
        return self.dvxmin

    def set_dvxmin(self, val: float) -> None:
        self.dvxmin = float(val)

    def get_dvxmax(self) -> float:
        return self.dvxmax

    def set_dvxmax(self, val: float) -> None:
        self.dvxmax = float(val)

    def get_dvymin(self) -> float:
        return self.dvymin

    def set_dvymin(self, val: float) -> None:
        self.dvymin = float(val)

    def get_dvymax(self) -> float:
        return self.dvymax

    def set_dvymax(self, val: float) -> None:
        self.dvymax = float(val)

    def get_dvzmin(self) -> float:
        return self.dvzmin

    def set_dvzmin(self, val: float) -> None:
        self.dvzmin = float(val)

    def get_dvzmax(self) -> float:
        return self.dvzmax

    def set_dvzmax(self, val: float) -> None:
        self.dvzmax = float(val)

    def get_dangle(self) -> float:
        return self.dangle

    def set_dangle(self, val: float) -> None:
        self.dangle = float(val)

    def get_dacc(self) -> float:
        return self.dacc

    def set_dacc(self, val: float) -> None:
        self.dacc = float(val)

    def get_add(self) -> int:
        return self.add

    def set_add(self, val: int) -> None:
        self.add = int(val)

    def get_track_mode(self) -> int:
        return self.track_mode

    def set_track_mode(self, val: int) -> None:
        self.track_mode = int(val)

    def get_dsumg(self) -> float:
        return self.dsumg

    def set_dsumg(self, val: float) -> None:
        self.dsumg = float(val)

    def get_dn(self) -> float:
        return self.dn

    def set_dn(self, val: float) -> None:
        self.dn = float(val)

    def get_dnx(self) -> float:
        return self.dnx

    def set_dnx(self, val: float) -> None:
        self.dnx = float(val)

    def get_dny(self) -> float:
        return self.dny

    def set_dny(self, val: float) -> None:
        self.dny = float(val)

    def read_track_par(self, filename: str | Path) -> None:
        """Read tracking parameters from file in-place."""
        new_par = TrackPar.from_file(filename)
        self.dvxmin = new_par.dvxmin
        self.dvxmax = new_par.dvxmax
        self.dvymin = new_par.dvymin
        self.dvymax = new_par.dvymax
        self.dvzmin = new_par.dvzmin
        self.dvzmax = new_par.dvzmax
        self.dangle = new_par.dangle
        self.dacc = new_par.dacc
        self.add = new_par.add
        self.track_mode = new_par.track_mode
        self.dsumg = new_par.dsumg
        self.dn = new_par.dn
        self.dnx = new_par.dnx
        self.dny = new_par.dny


@cython.cclass
class VolumePar:
    X_lay: object = cython.declare(object, visibility="public")
    Zmin_lay: object = cython.declare(object, visibility="public")
    Zmax_lay: object = cython.declare(object, visibility="public")
    cnx: cython.double = cython.declare(cython.double, visibility="public")
    cny: cython.double = cython.declare(cython.double, visibility="public")
    cn: cython.double = cython.declare(cython.double, visibility="public")
    csumg: cython.double = cython.declare(cython.double, visibility="public")
    corrmin: cython.double = cython.declare(cython.double, visibility="public")
    eps0: cython.double = cython.declare(cython.double, visibility="public")

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
        eps0: float = 0.0,
    ) -> None:
        self.X_lay = np.zeros(2) if X_lay is None else np.array(X_lay, dtype=np.float64)
        self.Zmin_lay = (
            np.zeros(2) if Zmin_lay is None else np.array(Zmin_lay, dtype=np.float64)
        )
        self.Zmax_lay = (
            np.zeros(2) if Zmax_lay is None else np.array(Zmax_lay, dtype=np.float64)
        )
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

    @staticmethod
    def from_yaml(filename: str | Path) -> VolumePar:
        """Build a VolumePar from an experiment YAML (YAML-only)."""
        c = _load_yaml_params(filename).get("criteria", {})
        return VolumePar(
            X_lay=list(c.get("X_lay", [0.0, 0.0])),
            Zmin_lay=list(c.get("Zmin_lay", [0.0, 0.0])),
            Zmax_lay=list(c.get("Zmax_lay", [0.0, 0.0])),
            cnx=float(c.get("cnx", 0.0)),
            cny=float(c.get("cny", 0.0)),
            cn=float(c.get("cn", 0.0)),
            csumg=float(c.get("csumg", 0.0)),
            corrmin=float(c.get("corrmin", 0.0)),
            eps0=float(c.get("eps0", 0.0)),
        )

    # --- Backward Compatibility OOP Methods ---
    def get_X_lay(self, copy: bool = True) -> np.ndarray:
        if copy:
            return self.X_lay.copy()
        return self.X_lay

    def set_X_lay(self, X_lay) -> None:
        self.X_lay = np.asarray(X_lay, dtype=np.float64)

    def get_Zmin_lay(self, copy: bool = True) -> np.ndarray:
        if copy:
            return self.Zmin_lay.copy()
        return self.Zmin_lay

    def set_Zmin_lay(self, Zmin_lay) -> None:
        self.Zmin_lay = np.asarray(Zmin_lay, dtype=np.float64)

    def get_Zmax_lay(self, copy: bool = True) -> np.ndarray:
        if copy:
            return self.Zmax_lay.copy()
        return self.Zmax_lay

    def set_Zmax_lay(self, Zmax_lay) -> None:
        self.Zmax_lay = np.asarray(Zmax_lay, dtype=np.float64)

    def get_cn(self) -> float:
        return self.cn

    def set_cn(self, cn: float) -> None:
        self.cn = float(cn)

    def get_cnx(self) -> float:
        return self.cnx

    def set_cnx(self, cnx: float) -> None:
        self.cnx = float(cnx)

    def get_cny(self) -> float:
        return self.cny

    def set_cny(self, cny: float) -> None:
        self.cny = float(cny)

    def get_csumg(self) -> float:
        return self.csumg

    def set_csumg(self, csumg: float) -> None:
        self.csumg = float(csumg)

    def get_eps0(self) -> float:
        return self.eps0

    def set_eps0(self, eps0: float) -> None:
        self.eps0 = float(eps0)

    def get_corrmin(self) -> float:
        return self.corrmin

    def set_corrmin(self, corrmin: float) -> None:
        self.corrmin = float(corrmin)

    def read_volume_par(self, filename: str | Path) -> None:
        """Read volume parameters from file in-place."""
        new_par = VolumePar.from_file(filename)
        self.X_lay = new_par.X_lay
        self.Zmin_lay = new_par.Zmin_lay
        self.Zmax_lay = new_par.Zmax_lay
        self.cnx = new_par.cnx
        self.cny = new_par.cny
        self.cn = new_par.cn
        self.csumg = new_par.csumg
        self.corrmin = new_par.corrmin
        self.eps0 = new_par.eps0


@cython.cclass
class MmNp:
    nlay: cython.int = cython.declare(cython.int, visibility="public")
    n1: cython.double = cython.declare(cython.double, visibility="public")
    n2: object = cython.declare(object, visibility="public")
    d: object = cython.declare(object, visibility="public")
    n3: cython.double = cython.declare(cython.double, visibility="public")

    def __init__(
        self,
        nlay: int = 1,
        n1: float = 1.0,
        n2: list[float] | np.ndarray | None = None,
        d: list[float] | np.ndarray | None = None,
        n3: float = 1.0,
        _mm: MmNp | None = None,
    ) -> None:
        if _mm is not None:
            self.nlay = _mm.nlay
            self.n1 = _mm.n1
            self.n2 = _mm.n2
            self.d = _mm.d
            self.n3 = _mm.n3
            return

        self.nlay = nlay
        self.n1 = n1
        self.n2 = np.ones(3) if n2 is None else np.array(n2, dtype=np.float64)
        self.d = np.zeros(3) if d is None else np.array(d, dtype=np.float64)
        self.n3 = n3

    # --- Backward Compatibility OOP Methods ---
    def get_nlay(self) -> int:
        return self.nlay

    def get_n1(self) -> float:
        return self.n1

    def set_n1(self, n1: float) -> None:
        self.n1 = float(n1)

    def get_n2(self, copy: bool = True) -> np.ndarray:
        if copy:
            return self.n2.copy()
        return self.n2

    def get_d(self, copy: bool = True) -> np.ndarray:
        if copy:
            return self.d.copy()
        return self.d

    def set_layers(self, n2, d) -> None:
        self.n2 = np.asarray(n2, dtype=np.float64)
        self.d = np.asarray(d, dtype=np.float64)
        self.nlay = len(n2)

    def get_n3(self) -> float:
        return self.n3

    def set_n3(self, n3: float) -> None:
        self.n3 = float(n3)


@cython.cclass
class ControlPar:
    num_cams: cython.int = cython.declare(cython.int, visibility="public")
    img_base_name: object = cython.declare(object, visibility="public")
    cal_img_base_name: object = cython.declare(object, visibility="public")
    hp_flag: cython.int = cython.declare(cython.int, visibility="public")
    allCam_flag: cython.int = cython.declare(cython.int, visibility="public")
    tiff_flag: cython.int = cython.declare(cython.int, visibility="public")
    imx: cython.int = cython.declare(cython.int, visibility="public")
    imy: cython.int = cython.declare(cython.int, visibility="public")
    pix_x: cython.double = cython.declare(cython.double, visibility="public")
    pix_y: cython.double = cython.declare(cython.double, visibility="public")
    chfield: cython.int = cython.declare(cython.int, visibility="public")
    mm: object = cython.declare(object, visibility="public")

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
        mm: MmNp | None = None,
    ) -> None:
        self.num_cams = num_cams
        self.img_base_name = img_base_name if img_base_name is not None else []
        self.cal_img_base_name = (
            cal_img_base_name if cal_img_base_name is not None else []
        )
        if not self.img_base_name and num_cams > 0:
            self.img_base_name = [""] * num_cams
        if not self.cal_img_base_name and num_cams > 0:
            self.cal_img_base_name = [""] * num_cams
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

    @staticmethod
    def from_yaml(filename: str | Path, num_cams: int | None = None) -> ControlPar:
        """Build a ControlPar from an experiment YAML (YAML-only)."""
        data = _load_yaml_params(filename)
        ptv = data.get("ptv", {})
        seq = data.get("sequence", {})
        nc = int(num_cams if num_cams is not None else data.get("num_cams", 0))
        mm = MmNp(
            nlay=1,
            n1=float(ptv.get("mmp_n1", 1.0)),
            n2=[float(ptv.get("mmp_n2", 1.0)), 1.0, 1.0],
            d=[float(ptv.get("mmp_d", 0.0)), 0.0, 0.0],
            n3=float(ptv.get("mmp_n3", 1.0)),
        )
        return ControlPar(
            num_cams=nc,
            img_base_name=_clean_name_list(seq.get("base_name"), nc),
            cal_img_base_name=_clean_name_list(ptv.get("img_cal"), nc),
            hp_flag=int(ptv.get("hp_flag", 0)),
            allCam_flag=int(ptv.get("allcam_flag", 0)),
            tiff_flag=int(ptv.get("tiff_flag", 0)),
            imx=int(ptv["imx"]),
            imy=int(ptv["imy"]),
            pix_x=float(ptv["pix_x"]),
            pix_y=float(ptv["pix_y"]),
            chfield=int(ptv.get("chfield", 0)),
            mm=mm,
        )

    # --- Backward Compatibility OOP Methods ---
    def get_num_cams(self) -> int:
        return self.num_cams

    def get_image_size(self, copy: bool = True) -> tuple[int, int]:
        return (self.imx, self.imy)

    def set_image_size(self, size: tuple[int, int]) -> None:
        self.imx = int(size[0])
        self.imy = int(size[1])

    def get_pixel_size(self, copy: bool = True) -> tuple[float, float]:
        return (self.pix_x, self.pix_y)

    def set_pixel_size(self, size: tuple[float, float]) -> None:
        pix_x, pix_y = size
        self.pix_x = float(pix_x)
        self.pix_y = float(pix_y)

    def get_hp_flag(self) -> int:
        return self.hp_flag

    def set_hp_flag(self, flag: int) -> None:
        self.hp_flag = int(flag)

    def get_allCam_flag(self) -> int:
        return self.allCam_flag

    def set_allCam_flag(self, flag: int) -> None:
        self.allCam_flag = int(flag)

    def get_tiff_flag(self) -> int:
        return self.tiff_flag

    def set_tiff_flag(self, flag: int) -> None:
        self.tiff_flag = int(flag)

    def get_chfield(self) -> int:
        return self.chfield

    def set_chfield(self, chfield: int) -> None:
        self.chfield = int(chfield)

    def get_multimedia_params(self) -> MmNp:
        return self.mm

    def get_img_base_name(self, cam: int) -> str:
        try:
            return self.img_base_name[cam]
        except IndexError:
            return ""

    def set_img_base_name(self, cam: int, name: str) -> None:
        while len(self.img_base_name) <= cam:
            self.img_base_name.append("")
        self.img_base_name[cam] = str(name)

    def get_cal_img_base_name(self, cam: int) -> str:
        try:
            return self.cal_img_base_name[cam]
        except IndexError:
            return ""

    def set_cal_img_base_name(self, cam: int, name: str) -> None:
        while len(self.cal_img_base_name) <= cam:
            self.cal_img_base_name.append("")
        self.cal_img_base_name[cam] = str(name)

    def read_control_par(self, filename: str | Path) -> None:
        """Read control parameters from file in-place."""
        new_par = ControlPar.from_file(filename)
        self.num_cams = new_par.num_cams
        self.img_base_name = new_par.img_base_name
        self.cal_img_base_name = new_par.cal_img_base_name
        self.hp_flag = new_par.hp_flag
        self.allCam_flag = new_par.allCam_flag
        self.tiff_flag = new_par.tiff_flag
        self.imx = new_par.imx
        self.imy = new_par.imy
        self.pix_x = new_par.pix_x
        self.pix_y = new_par.pix_y
        self.chfield = new_par.chfield
        self.mm = new_par.mm


@cython.cclass
class TargetPar:
    gvthres: object = cython.declare(object, visibility="public")
    discont: cython.int = cython.declare(cython.int, visibility="public")
    nnmin: cython.int = cython.declare(cython.int, visibility="public")
    nnmax: cython.int = cython.declare(cython.int, visibility="public")
    nxmin: cython.int = cython.declare(cython.int, visibility="public")
    nxmax: cython.int = cython.declare(cython.int, visibility="public")
    nymin: cython.int = cython.declare(cython.int, visibility="public")
    nymax: cython.int = cython.declare(cython.int, visibility="public")
    sumg_min: cython.int = cython.declare(cython.int, visibility="public")
    cr_sz: cython.int = cython.declare(cython.int, visibility="public")

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
        cr_sz: int = 0,
        # support TargetParams constructor arguments:
        gvthresh: list[int] | np.ndarray | None = None,
        pixel_count_bounds: tuple[int, int] | None = None,
        xsize_bounds: tuple[int, int] | None = None,
        ysize_bounds: tuple[int, int] | None = None,
        min_sum_grey: int | None = None,
        cross_size: int | None = None,
    ) -> None:
        self.discont = discont

        if gvthresh is not None:
            self.gvthres = np.array(gvthresh, dtype=int)
        else:
            self.gvthres = (
                np.zeros(4, dtype=int)
                if gvthres is None
                else np.array(gvthres, dtype=int)
            )

        if pixel_count_bounds is not None:
            self.nnmin, self.nnmax = pixel_count_bounds
        else:
            self.nnmin, self.nnmax = nnmin, nnmax

        if xsize_bounds is not None:
            self.nxmin, self.nxmax = xsize_bounds
        else:
            self.nxmin, self.nxmax = nxmin, nxmax

        if ysize_bounds is not None:
            self.nymin, self.nymax = ysize_bounds
        else:
            self.nymin, self.nymax = nymin, nymax

        if min_sum_grey is not None:
            self.sumg_min = min_sum_grey
        else:
            self.sumg_min = sumg_min

        if cross_size is not None:
            self.cr_sz = cross_size
        else:
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
        return TargetPar(
            gvthres, discont, nnmin, nnmax, nxmin, nxmax, nymin, nymax, sumg_min, cr_sz
        )

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

    # --- Backward Compatibility OOP Methods ---
    def get_grey_thresholds(self, num_cams=4, copy: bool = True) -> np.ndarray:
        if copy:
            return self.gvthres.copy()
        return self.gvthres

    def set_grey_thresholds(self, gvthres) -> None:
        self.gvthres = np.asarray(gvthres, dtype=np.int32)

    def get_max_discontinuity(self) -> int:
        return self.discont

    def set_max_discontinuity(self, discont: int) -> None:
        self.discont = int(discont)

    def get_pixel_count_bounds(self) -> tuple[int, int]:
        return (self.nnmin, self.nnmax)

    def set_pixel_count_bounds(self, bounds: tuple[int, int]) -> None:
        self.nnmin = int(bounds[0])
        self.nnmax = int(bounds[1])

    def get_xsize_bounds(self) -> tuple[int, int]:
        return (self.nxmin, self.nxmax)

    def set_xsize_bounds(self, bounds: tuple[int, int]) -> None:
        self.nxmin = int(bounds[0])
        self.nxmax = int(bounds[1])

    def get_ysize_bounds(self) -> tuple[int, int]:
        return (self.nymin, self.nymax)

    def set_ysize_bounds(self, bounds: tuple[int, int]) -> None:
        self.nymin = int(bounds[0])
        self.nymax = int(bounds[1])

    def get_min_sum_grey(self) -> int:
        return self.sumg_min

    def set_min_sum_grey(self, sumg_min: int) -> None:
        self.sumg_min = int(sumg_min)

    def get_cross_size(self) -> int:
        return self.cr_sz

    def set_cross_size(self, cr_sz: int) -> None:
        self.cr_sz = int(cr_sz)

    def read(self, filename: str | Path) -> None:
        """Read target parameters from file in-place."""
        new_par = TargetPar.from_file(filename)
        self.gvthres = new_par.gvthres
        self.discont = new_par.discont
        self.nnmin = new_par.nnmin
        self.nnmax = new_par.nnmax
        self.nxmin = new_par.nxmin
        self.nxmax = new_par.nxmax
        self.nymin = new_par.nymin
        self.nymax = new_par.nymax
        self.sumg_min = new_par.sumg_min
        self.cr_sz = new_par.cr_sz


@cython.cclass
class OrientPar:
    useflag: cython.int = cython.declare(cython.int, visibility="public")
    ccflag: cython.int = cython.declare(cython.int, visibility="public")
    xhflag: cython.int = cython.declare(cython.int, visibility="public")
    yhflag: cython.int = cython.declare(cython.int, visibility="public")
    k1flag: cython.int = cython.declare(cython.int, visibility="public")
    k2flag: cython.int = cython.declare(cython.int, visibility="public")
    k3flag: cython.int = cython.declare(cython.int, visibility="public")
    p1flag: cython.int = cython.declare(cython.int, visibility="public")
    p2flag: cython.int = cython.declare(cython.int, visibility="public")
    scxflag: cython.int = cython.declare(cython.int, visibility="public")
    sheflag: cython.int = cython.declare(cython.int, visibility="public")
    interfflag: cython.int = cython.declare(cython.int, visibility="public")

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
        interfflag: int = 0,
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


@cython.cclass
class MultimediaPar:
    n1: cython.double = cython.declare(cython.double, visibility="public")
    n2: object = cython.declare(object, visibility="public")
    d: object = cython.declare(object, visibility="public")
    n3: cython.double = cython.declare(cython.double, visibility="public")
    nlay: cython.int = cython.declare(cython.int, visibility="public")

    def __init__(
        self,
        n1: float = 1.0,
        n2: list[float] | np.ndarray | None = None,
        d: list[float] | np.ndarray | None = None,
        n3: float = 1.0,
        nlay: int = 1,
    ) -> None:
        self.n1 = n1
        self.n2 = np.ones(3) if n2 is None else np.array(n2, dtype=np.float64)
        self.d = np.zeros(3) if d is None else np.array(d, dtype=np.float64)
        self.n3 = n3
        self.nlay = nlay


@cython.cclass
class CalibrationPar:
    """Calibration parameters for calibration workflow."""

    fixp_name: object = cython.declare(object, visibility="public")
    img_cal_name: object = cython.declare(object, visibility="public")
    img_ori: object = cython.declare(object, visibility="public")
    tiff_flag: cython.int = cython.declare(cython.int, visibility="public")
    pair_flag: cython.int = cython.declare(cython.int, visibility="public")
    chfield: cython.int = cython.declare(cython.int, visibility="public")

    def __init__(
        self,
        fixp_name: str = "",
        img_cal_name: list[str] | None = None,
        img_ori: list[str] | None = None,
        tiff_flag: int = 0,
        pair_flag: int = 0,
        chfield: int = 0,
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

        return CalibrationPar(
            fixp_name, img_cal_name, img_ori, tiff_flag, pair_flag, chfield
        )


@cython.cclass
class MultiPlanesPar:
    """Multiplanes parameters."""

    num_planes: cython.int = cython.declare(cython.int, visibility="public")
    filename: object = cython.declare(object, visibility="public")

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


@cython.cclass
class ExaminePar:
    """Examine parameters."""

    examine_flag: cython.bint = cython.declare(cython.bint, visibility="public")
    combine_flag: cython.bint = cython.declare(cython.bint, visibility="public")

    def __init__(self, examine_flag: bool = False, combine_flag: bool = False) -> None:
        self.examine_flag = examine_flag
        self.combine_flag = combine_flag

    @staticmethod
    def from_file(file_path: str | Path) -> ExaminePar:
        """Read from examine.par file."""
        examine_flag = False
        combine_flag = False
        with open(file_path, "r", encoding="utf-8") as file:
            examine_flag = bool(int(file.readline().strip()))
            combine_flag = bool(int(file.readline().strip()))
        return ExaminePar(examine_flag, combine_flag)


@cython.cclass
class PftVersionPar:
    """PFT version parameters."""

    existing_target_flag: cython.bint = cython.declare(cython.bint, visibility="public")

    def __init__(self, existing_target_flag: bool = False) -> None:
        self.existing_target_flag = existing_target_flag

    @staticmethod
    def from_file(file_path: str | Path) -> PftVersionPar:
        """Read from pft_version.par file."""
        existing_target_flag = False
        with open(file_path, "r", encoding="utf-8") as file:
            existing_target_flag = bool(int(file.readline().strip()))
        return PftVersionPar(existing_target_flag)


# Aliases for compatibility with legacy test code (must be after all class definitions)
read_control_par = ControlPar.from_file
read_volume_par = VolumePar.from_file
read_sequence_par = SequencePar.from_file
read_track_par = TrackPar.from_file

ControlParams = ControlPar
VolumeParams = VolumePar
TargetParams = TargetPar
TrackingParams = TrackPar
SequenceParams = SequencePar
MultimediaParams = MmNp


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
