import cython

# Convert TrackPar to TrackParTuple for test compatibility
@cython.ccall
def convert_track_par_to_tuple(track_par):
    return TrackParTuple(
        track_par.dvxmin, track_par.dvxmax, track_par.dvymin, track_par.dvymax,
        track_par.dvzmin, track_par.dvzmax, track_par.dangle, track_par.dacc,
        track_par.add, getattr(track_par, 'dsumg', 0.0), getattr(track_par, 'dn', 0.0),
        getattr(track_par, 'dnx', 0.0), getattr(track_par, 'dny', 0.0)
    )
from collections import namedtuple

# TrackParTuple for test compatibility
TrackParTuple = namedtuple('TrackParTuple', [
    'dvxmin', 'dvxmax', 'dvymin', 'dvymax', 'dvzmin', 'dvzmax',
    'dangle', 'dacc', 'add', 'dsumg', 'dn', 'dnx', 'dny'
])
import numpy as np
from pathlib import Path

class SequencePar:
    def __init__(self, num_cams=0, img_base_name=None, first=0, last=0):
        self.num_cams = num_cams
        self.img_base_name = img_base_name if img_base_name is not None else []
        self.first = first
        self.last = last
    @staticmethod
    def from_file(filename, num_cams):
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        if len(lines) < num_cams + 2:
            raise ValueError(f"Expected at least {num_cams + 2} lines, got {len(lines)}")
        img_base_name = [lines[i].strip() for i in range(num_cams)]
        first = int(lines[num_cams].strip())
        last = int(lines[num_cams + 1].strip())
        return SequencePar(num_cams, img_base_name, first, last)

class TrackPar:
    def __init__(self, dvxmin=0.0, dvxmax=0.0, dvymin=0.0, dvymax=0.0, dvzmin=0.0, dvzmax=0.0, dangle=0.0, dacc=0.0, add=0, track_mode=0):
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
    def from_file(filename):
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
    def __init__(self, X_lay=None, Zmin_lay=None, Zmax_lay=None, cnx=0.0, cny=0.0, cn=0.0, csumg=0.0, corrmin=0.0, eps0=0.0):
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
    def from_file(filename):
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
    def __init__(self, nlay=1, n1=1.0, n2=None, d=None, n3=1.0):
        self.nlay = nlay
        self.n1 = n1
        self.n2 = np.ones(3) if n2 is None else np.array(n2, dtype=np.float64)
        self.d = np.zeros(3) if d is None else np.array(d, dtype=np.float64)
        self.n3 = n3

class ControlPar:
    def __init__(self, num_cams=0, img_base_name=None, cal_img_base_name=None, hp_flag=0, allCam_flag=0, all_cam_flag=None, tiff_flag=0, imx=0, imy=0, pix_x=0.0, pix_y=0.0, chfield=0, mm=None):
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
    def from_file(filename):
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
    def __init__(self, gvthres=None, discont=0, nnmin=0, nnmax=0, nxmin=0, nxmax=0, nymin=0, nymax=0, sumg_min=0, cr_sz=0):
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
    def from_file(filename):
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
    def to_file(self, filename):
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
    def __init__(self, useflag=0, ccflag=0, xhflag=0, yhflag=0,
                 k1flag=0, k2flag=0, k3flag=0, p1flag=0, p2flag=0,
                 scxflag=0, sheflag=0, interfflag=0):
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
    def from_file(filename):
        path = Path(filename)
        lines = path.read_text().strip().splitlines()
        flags = [int(line.strip()) for line in lines[:12]]
        while len(flags) < 12:
            flags.append(0)
        return OrientPar(*flags)

class MultimediaPar:
    def __init__(self, n1=1.0, n2=None, d=None, n3=1.0, nlay=1):
        self.n1 = n1
        self.n2 = np.ones(3) if n2 is None else np.array(n2, dtype=np.float64)
        self.d = np.zeros(3) if d is None else np.array(d, dtype=np.float64)
        self.n3 = n3
        self.nlay = nlay

class CalibrationPar:
    """Calibration parameters for calibration workflow."""


    def __init__(self, fixp_name="", img_cal_name=None, img_ori=None, tiff_flag=0, pair_flag=0, chfield=0):
        self.fixp_name = fixp_name
        self.img_cal_name = img_cal_name if img_cal_name is not None else []
        self.img_ori = img_ori if img_ori is not None else []
        self.tiff_flag = tiff_flag
        self.pair_flag = pair_flag
        self.chfield = chfield

    @staticmethod
    def from_file(file_path: str, num_cams: int):
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

    def __init__(self, num_planes=0, filename=None):
        self.num_planes = num_planes
        self.filename = filename if filename is not None else []

    @staticmethod
    def from_file(file_path: str):
        """Read from multiplanes.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            num_planes = int(file.readline().strip())
            filename = [file.readline().strip() for _ in range(num_planes)]
        return MultiPlanesPar(num_planes, filename)


class ExaminePar:
    """Examine parameters."""

    def __init__(self, examine_flag=False, combine_flag=False):
        self.examine_flag = examine_flag
        self.combine_flag = combine_flag

    @staticmethod
    def from_file(file_path: str):
        """Read from examine.par file."""
        with open(file_path, "r", encoding="utf-8") as file:
            examine_flag = bool(int(file.readline().strip()))
            combine_flag = bool(int(file.readline().strip()))
        return ExaminePar(examine_flag, combine_flag)


class PftVersionPar:
    """PFT version parameters."""

    def __init__(self, existing_target_flag=False):
        self.existing_target_flag = existing_target_flag

    @staticmethod
    def from_file(file_path: str):
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
