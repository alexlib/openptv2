"""Compatibility and direct parameter API with optv-like wrappers."""

import numpy as np
from openptv2.algorithms.parameters import (
    ControlPar as AlgoControlPar,
    VolumePar as AlgoVolumePar,
    TrackPar as AlgoTrackPar,
    SequencePar as AlgoSequencePar,
    TargetPar as AlgoTargetPar,
    MmNp as AlgoMmNp,
    TrackParTuple,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.parameter_converters import (
    convert_optv_calibrations,
    get_all_params,
    get_calibration_par,
    get_control_par,
    get_examine_par,
    get_multimedia_par,
    get_multiplanes_par,
    get_orient_par,
    get_pft_version_par,
    get_sequence_par,
    get_target_par,
    get_track_par_tuple,
    get_volume_par,
)


class MultimediaParams:
    """Wrapper for algorithms.parameters.MmNp with optv-compatible API."""

    def __init__(self, n1=1.0, n2=None, n3=1.0, d=None, _mm=None):
        """Initialize multimedia parameters."""
        if _mm is not None:
            self._mm = _mm
            return

        if n2 is None:
            n2 = np.ones(3, dtype=np.float64)
        if d is None:
            d = np.zeros(3, dtype=np.float64)

        nlay = 1
        if isinstance(n2, (list, np.ndarray)):
            n2_arr = np.asarray(n2, dtype=np.float64)
            if n2_arr.size == 3:
                d_arr = (
                    np.asarray(d, dtype=np.float64) if d is not None else np.zeros(3)
                )
                nlay = max(1, int(np.sum(d_arr > 0)))
            else:
                nlay = n2_arr.size

        self._mm = AlgoMmNp(nlay=nlay, n1=n1, n2=n2, d=d, n3=n3)

    def get_nlay(self):
        """Get number of layers."""
        return self._mm.nlay

    def get_n1(self):
        """Get refractive index of first layer (air)."""
        return self._mm.n1

    def set_n1(self, n1):
        """Set refractive index of first layer."""
        self._mm.n1 = float(n1)

    def get_n2(self, copy=True):
        """Get refractive indices of middle layers."""
        if copy:
            return self._mm.n2.copy()
        return self._mm.n2

    def get_d(self, copy=True):
        """Get thicknesses of middle layers."""
        if copy:
            return self._mm.d.copy()
        return self._mm.d

    def set_layers(self, n2, d):
        """Set middle layer parameters."""
        self._mm.n2 = np.asarray(n2, dtype=np.float64)
        self._mm.d = np.asarray(d, dtype=np.float64)
        # Update nlay to match Cython/C API: nlay is the number of middle layers
        self._mm.nlay = len(n2)

    def get_n3(self):
        """Get refractive index of last layer (glass)."""
        return self._mm.n3

    def set_n3(self, n3):
        """Set refractive index of last layer."""
        self._mm.n3 = float(n3)


class ControlParams:
    """Wrapper for algorithms.parameters.ControlPar with optv-compatible API."""

    def __init__(self, num_cams=0, **kwargs):
        """Initialize control parameters."""
        self._cpar = AlgoControlPar(num_cams=num_cams, **kwargs)
        if not self._cpar.img_base_name:
            self._cpar.img_base_name = [""] * num_cams
        if not self._cpar.cal_img_base_name:
            self._cpar.cal_img_base_name = [""] * num_cams

    def get_num_cams(self):
        """Get number of cameras."""
        return self._cpar.num_cams

    def get_image_size(self, copy=True):
        """Get image size as tuple (imx, imy)."""
        return (self._cpar.imx, self._cpar.imy)

    def set_image_size(self, size):
        """Set image size from tuple (imx, imy)."""
        self._cpar.imx = int(size[0])
        self._cpar.imy = int(size[1])

    def get_pixel_size(self, copy=True):
        """Get pixel size as tuple (pix_x, pix_y)."""
        return (self._cpar.pix_x, self._cpar.pix_y)

    def set_pixel_size(self, size):
        """Set pixel size from tuple (pix_x, pix_y)."""
        self._cpar.pix_x = float(size[0])
        self._cpar.pix_y = float(size[1])

    def get_hp_flag(self):
        """Get high-pass filter flag."""
        return self._cpar.hp_flag

    def set_hp_flag(self, flag):
        """Set high-pass filter flag."""
        self._cpar.hp_flag = int(flag)

    def get_allCam_flag(self):
        """Get all-camera flag."""
        return self._cpar.allCam_flag

    def set_allCam_flag(self, flag):
        """Set all-camera flag."""
        self._cpar.allCam_flag = int(flag)

    def get_tiff_flag(self):
        """Get TIFF flag."""
        return self._cpar.tiff_flag

    def set_tiff_flag(self, flag):
        """Set TIFF flag."""
        self._cpar.tiff_flag = int(flag)

    def get_chfield(self):
        """Get chfield parameter."""
        return self._cpar.chfield

    def set_chfield(self, chfield):
        """Set chfield parameter."""
        self._cpar.chfield = int(chfield)

    def get_multimedia_params(self):
        """Get multimedia parameters (returns wrapper around same underlying object)."""
        return MultimediaParams(_mm=self._cpar.mm)

    def get_img_base_name(self, cam):
        """Get image base name for camera."""
        try:
            return self._cpar.img_base_name[cam]
        except IndexError:
            return ""

    def set_img_base_name(self, cam, name):
        """Set image base name for camera."""
        self._cpar.img_base_name[cam] = str(name)

    def get_cal_img_base_name(self, cam):
        """Get calibration image base name for camera."""
        try:
            return self._cpar.cal_img_base_name[cam]
        except IndexError:
            return ""

    def set_cal_img_base_name(self, cam, name):
        """Set calibration image base name for camera."""
        self._cpar.cal_img_base_name[cam] = str(name)

    def read_control_par(self, filename):
        """Read control parameters from file (in-place, like optv)."""
        self._cpar = AlgoControlPar.from_file(filename)


class VolumeParams:
    """Wrapper for algorithms.parameters.VolumePar with optv-compatible API."""

    def __init__(self, **kwargs):
        """Initialize volume parameters."""
        self._vpar = AlgoVolumePar(**kwargs)

    def get_X_lay(self, copy=True):
        """Get X layer bounds."""
        if copy:
            return self._vpar.X_lay.copy()
        return self._vpar.X_lay

    def set_X_lay(self, X_lay):
        """Set X layer bounds."""
        self._vpar.X_lay = np.asarray(X_lay, dtype=np.float64)

    def get_Zmin_lay(self, copy=True):
        """Get Zmin layer bounds."""
        if copy:
            return self._vpar.Zmin_lay.copy()
        return self._vpar.Zmin_lay

    def set_Zmin_lay(self, Zmin_lay):
        """Set Zmin layer bounds."""
        self._vpar.Zmin_lay = np.asarray(Zmin_lay, dtype=np.float64)

    def get_Zmax_lay(self, copy=True):
        """Get Zmax layer bounds."""
        if copy:
            return self._vpar.Zmax_lay.copy()
        return self._vpar.Zmax_lay

    def set_Zmax_lay(self, Zmax_lay):
        """Set Zmax layer bounds."""
        self._vpar.Zmax_lay = np.asarray(Zmax_lay, dtype=np.float64)

    def get_cn(self):
        """Get cn parameter."""
        return self._vpar.cn

    def set_cn(self, cn):
        """Set cn parameter."""
        self._vpar.cn = float(cn)

    def get_cnx(self):
        """Get cnx parameter."""
        return self._vpar.cnx

    def set_cnx(self, cnx):
        """Set cnx parameter."""
        self._vpar.cnx = float(cnx)

    def get_cny(self):
        """Get cny parameter."""
        return self._vpar.cny

    def set_cny(self, cny):
        """Set cny parameter."""
        self._vpar.cny = float(cny)

    def get_csumg(self):
        """Get csumg parameter."""
        return self._vpar.csumg

    def set_csumg(self, csumg):
        """Set csumg parameter."""
        self._vpar.csumg = float(csumg)

    def get_eps0(self):
        """Get eps0 parameter."""
        return self._vpar.eps0

    def set_eps0(self, eps0):
        """Set eps0 parameter."""
        self._vpar.eps0 = float(eps0)

    def get_corrmin(self):
        """Get corrmin parameter."""
        return self._vpar.corrmin

    def set_corrmin(self, corrmin):
        """Set corrmin parameter."""
        self._vpar.corrmin = float(corrmin)

    def read_volume_par(self, filename):
        """Read volume parameters from file (in-place, like optv)."""
        self._vpar = AlgoVolumePar.from_file(filename)


class TrackingParams:
    """Wrapper for algorithms.parameters.TrackPar with optv-compatible API."""

    def __init__(self, **kwargs):
        """Initialize tracking parameters."""
        self._tpar = AlgoTrackPar(**kwargs)

    def get_dvxmin(self):
        return self._tpar.dvxmin

    def set_dvxmin(self, val):
        self._tpar.dvxmin = float(val)

    def get_dvxmax(self):
        return self._tpar.dvxmax

    def set_dvxmax(self, val):
        self._tpar.dvxmax = float(val)

    def get_dvymin(self):
        return self._tpar.dvymin

    def set_dvymin(self, val):
        self._tpar.dvymin = float(val)

    def get_dvymax(self):
        return self._tpar.dvymax

    def set_dvymax(self, val):
        self._tpar.dvymax = float(val)

    def get_dvzmin(self):
        return self._tpar.dvzmin

    def set_dvzmin(self, val):
        self._tpar.dvzmin = float(val)

    def get_dvzmax(self):
        return self._tpar.dvzmax

    def set_dvzmax(self, val):
        self._tpar.dvzmax = float(val)

    def get_dangle(self):
        return self._tpar.dangle

    def set_dangle(self, val):
        self._tpar.dangle = float(val)

    def get_dacc(self):
        return self._tpar.dacc

    def set_dacc(self, val):
        self._tpar.dacc = float(val)

    def get_add(self):
        return self._tpar.add

    def set_add(self, val):
        self._tpar.add = int(val)

    def get_track_mode(self):
        return getattr(self._tpar, "track_mode", 0)

    def set_track_mode(self, val):
        self._tpar.track_mode = int(val)

    def get_dsumg(self):
        return self._tpar.dsumg

    def set_dsumg(self, val):
        self._tpar.dsumg = float(val)

    def get_dn(self):
        return self._tpar.dn

    def set_dn(self, val):
        self._tpar.dn = float(val)

    def get_dnx(self):
        return self._tpar.dnx

    def set_dnx(self, val):
        self._tpar.dnx = float(val)

    def get_dny(self):
        return self._tpar.dny

    def set_dny(self, val):
        self._tpar.dny = float(val)

    def read_track_par(self, filename):
        """Read tracking parameters from file (in-place, like optv)."""
        self._tpar = AlgoTrackPar.from_file(filename)


class SequenceParams:
    """Wrapper for algorithms.parameters.SequencePar with optv-compatible API."""

    def __init__(self, num_cams=0, **kwargs):
        """Initialize sequence parameters."""
        # Translate legacy kwarg names to match AlgoSequencePar.__init__
        if "image_base" in kwargs:
            kwargs["img_base_name"] = kwargs.pop("image_base")
        if "frame_range" in kwargs:
            fr = kwargs.pop("frame_range")
            kwargs["first"] = fr[0]
            kwargs["last"] = fr[1]
        self._spar = AlgoSequencePar(num_cams=num_cams, **kwargs)
        if not self._spar.img_base_name:
            self._spar.img_base_name = [""] * num_cams

    def get_first(self):
        """Get first frame number."""
        return self._spar.first

    def set_first(self, first):
        """Set first frame number."""
        self._spar.first = int(first)

    def get_last(self):
        """Get last frame number."""
        return self._spar.last

    def set_last(self, last):
        """Set last frame number."""
        self._spar.last = int(last)

    def get_img_base_name(self, cam):
        """Get image base name for camera."""
        return self._spar.img_base_name[cam]

    def set_img_base_name(self, cam, name):
        """Set image base name for camera."""
        self._spar.img_base_name[cam] = str(name)

    def read_sequence_par(self, filename, num_cams):
        """Read sequence parameters from file (in-place, like optv)."""
        self._spar = AlgoSequencePar.from_file(filename, num_cams)


class TargetParams:
    """Wrapper for algorithms.parameters.TargetPar with optv-compatible API."""

    def __init__(
        self,
        discont=0,
        gvthresh=None,
        pixel_count_bounds=(0, 1000),
        xsize_bounds=(0, 100),
        ysize_bounds=(0, 100),
        min_sum_grey=0,
        cross_size=2,
        **kwargs,
    ):
        """Initialize target parameters (matching optv signature)."""
        self._tpar = AlgoTargetPar(**kwargs)
        self.set_max_discontinuity(discont)
        if gvthresh is not None:
            self.set_grey_thresholds(gvthresh)
        self.set_pixel_count_bounds(pixel_count_bounds)
        self.set_xsize_bounds(xsize_bounds)
        self.set_ysize_bounds(ysize_bounds)
        self.set_min_sum_grey(min_sum_grey)
        self.set_cross_size(cross_size)

    def get_grey_thresholds(self, num_cams=4, copy=True):
        """Get grey value thresholds."""
        if copy:
            return self._tpar.gvthres.copy()
        return self._tpar.gvthres

    def set_grey_thresholds(self, gvthres):
        """Set grey value thresholds."""
        self._tpar.gvthres = np.asarray(gvthres, dtype=np.int32)

    def get_max_discontinuity(self):
        """Get maximum discontinuity."""
        return self._tpar.discont

    def set_max_discontinuity(self, discont):
        """Set maximum discontinuity."""
        self._tpar.discont = int(discont)

    def get_pixel_count_bounds(self):
        """Get pixel count bounds as tuple (min, max)."""
        return (self._tpar.nnmin, self._tpar.nnmax)

    def set_pixel_count_bounds(self, bounds):
        """Set pixel count bounds from tuple (min, max)."""
        self._tpar.nnmin = int(bounds[0])
        self._tpar.nnmax = int(bounds[1])

    def get_xsize_bounds(self):
        """Get x size bounds as tuple (min, max)."""
        return (self._tpar.nxmin, self._tpar.nxmax)

    def set_xsize_bounds(self, bounds):
        """Set x size bounds from tuple (min, max)."""
        self._tpar.nxmin = int(bounds[0])
        self._tpar.nxmax = int(bounds[1])

    def get_ysize_bounds(self):
        """Get y size bounds as tuple (min, max)."""
        return (self._tpar.nymin, self._tpar.nymax)

    def set_ysize_bounds(self, bounds):
        """Set y size bounds from tuple (min, max)."""
        self._tpar.nymin = int(bounds[0])
        self._tpar.nymax = int(bounds[1])

    def get_min_sum_grey(self):
        """Get minimum sum grey value."""
        return self._tpar.sumg_min

    def set_min_sum_grey(self, sumg_min):
        """Set minimum sum grey value."""
        self._tpar.sumg_min = int(sumg_min)

    def get_cross_size(self):
        """Get cross size."""
        return self._tpar.cr_sz

    def set_cross_size(self, cr_sz):
        """Set cross size."""
        self._tpar.cr_sz = int(cr_sz)

    def read(self, filename):
        """Read target parameters from file (in-place, like optv)."""
        self._tpar = AlgoTargetPar.from_file(filename)


__all__ = [
    "ControlParams",
    "MultimediaParams",
    "SequenceParams",
    "TargetParams",
    "TrackingParams",
    "VolumeParams",
    "convert_optv_calibrations",
    "get_all_params",
    "get_calibration_par",
    "get_control_par",
    "get_examine_par",
    "get_multimedia_par",
    "get_multiplanes_par",
    "get_orient_par",
    "get_pft_version_par",
    "get_sequence_par",
    "get_target_par",
    "get_track_par_tuple",
    "get_volume_par",
    "TrackParTuple",
    "convert_track_par_to_tuple",
]
