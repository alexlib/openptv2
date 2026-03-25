"""Adapter layer to provide Cython-compatible Parameters API for Python engine."""

from typing import Optional, List, Union


class ControlParams:
    """
    Cython-compatible ControlParams wrapper for Python engine.

    Provides the same API as bindings/optv/parameters.pyx ControlParams class:
    - __init__(num_cams, flags=None, image_size=None, pixel_size=None, ...)
    - get_num_cams(), get_hp_flag(), set_hp_flag(), etc.
    """

    def __init__(
        self,
        num_cams: int,
        flags: Optional[List[str]] = None,
        image_size: Optional[tuple] = None,
        pixel_size: Optional[tuple] = None,
        cam_side_n: float = 1.0,
        wall_ns: Optional[List[float]] = None,
        wall_thicks: Optional[List[float]] = None,
        object_side_n: float = 1.0,
    ):
        """
        Initialize ControlParams.

        Arguments:
        num_cams - number of cameras used in the scene.
        flags - list containing name of set flags, select from 'hp', 'allcam', 'headers'.
        image_size - sequence, w,h image size in pixels.
        pixel_size - sequence, w,h pixel size in mm.
        cam_side_n, wall_ns, wall_thicks, object_side_n - see MultimediaParams
        """
        from algorithms.parameters import ControlPar, MultimediaPar

        self._impl = ControlPar(num_cams=num_cams)

        # Set flags
        if flags:
            if "hp" in flags:
                self.set_hp_flag(True)
            if "allcam" in flags:
                self.set_allCam_flag(True)
            if "headers" in flags:
                self.set_tiff_flag(True)

        # Set image size
        if image_size:
            self.set_image_size(image_size)

        # Set pixel size
        if pixel_size:
            self.set_pixel_size(pixel_size)

        # Set multimedia params
        self._mm = MultimediaPar(
            n1=cam_side_n,
            n2=wall_ns if wall_ns else [1.0],
            d=wall_thicks if wall_thicks else [0.0],
            n3=object_side_n,
        )

    def __getattr__(self, name):
        """Delegate any unknown attribute access to the underlying implementation."""
        if hasattr(self, "_impl") and self._impl is not None:
            return getattr(self._impl, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def img_base_name(self):
        """Return list of image base names using getter methods."""
        return [
            self._impl.get_img_base_name(i) for i in range(self._impl.get_num_cams())
        ]

    @property
    def num_cams(self):
        # First check if we have a cached value, otherwise get from impl
        if hasattr(self, "_num_cams"):
            return self._num_cams
        return self.get_num_cams()

    @property
    def imx(self):
        if hasattr(self, "_imx"):
            return self._imx
        return getattr(self._impl, "imx", 0)

    @property
    def imy(self):
        if hasattr(self, "_imy"):
            return self._imy
        return getattr(self._impl, "imy", 0)

    @property
    def mm(self):
        if hasattr(self, "_mm"):
            return self._mm
        return getattr(self._impl, "mm", None)

    @property
    def pix_x(self):
        if hasattr(self, "_pix_x"):
            return self._pix_x
        return getattr(self._impl, "pix_x", 0.0)

    @property
    def pix_y(self):
        if hasattr(self, "_pix_y"):
            return self._pix_y
        return getattr(self._impl, "pix_y", 0.0)

    def get_num_cams(self):
        # Use the actual _impl if it has num_cams attribute, otherwise use cached
        if hasattr(self._impl, "num_cams"):
            return self._impl.num_cams
        return self._impl.get_num_cams()
        return self._impl.num_cams

    def get_hp_flag(self):
        return getattr(self._impl, "hp_flag", False)

    def set_hp_flag(self, value):
        self._impl.hp_flag = 1 if value else 0

    def set_allCam_flag(self, value):
        self._impl.allcam = 1 if value else 0

    def set_tiff_flag(self, value):
        self._impl.tif = 1 if value else 0

    def set_chfield(self, value):
        self._impl.chfield = value

    def set_image_size(self, size):
        if hasattr(self._impl, "imx"):
            self._impl.imx = size[0]
        if hasattr(self._impl, "imy"):
            self._impl.imy = size[1]

    def set_pixel_size(self, size):
        if hasattr(self._impl, "pix_x"):
            self._impl.pix_x = size[0]
        if hasattr(self._impl, "pix_y"):
            self._impl.pix_y = size[1]


class SequenceParams:
    """
    Cython-compatible SequenceParams wrapper for Python engine.

    Provides the same API as bindings/optv/parameters.pyx SequenceParams class:
    - __init__(num_cams=..., image_base=..., frame_range=...)
    - get_first(), set_first(), get_last(), set_last(), etc.
    """

    def __init__(
        self,
        num_cams: Optional[int] = None,
        image_base: Optional[List[str]] = None,
        frame_range: Optional[tuple] = None,
    ):
        """
        Initialize SequenceParams.

        Arguments (either num_cams or image_base required):
        num_cams - number of cameras used in the scene.
        image_base - a list of image base names.
        frame_range - (first, last) frame numbers.
        """
        from algorithms.parameters import SequencePar

        if num_cams is None and image_base is None:
            raise ValueError("SequenceParams requires either num_cams or image_base")

        # Use image_base if provided, otherwise create empty list
        img_base = image_base if image_base else []

        first = frame_range[0] if frame_range else 0
        last = frame_range[1] if frame_range else 0

        self._impl = SequencePar(img_base_name=img_base, first=first, last=last)

    def __getattr__(self, name):
        """Delegate any unknown attribute access to the underlying implementation."""
        if hasattr(self, "_impl") and self._impl is not None:
            return getattr(self._impl, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    @property
    def img_base_name(self):
        """Return list of image base names using getter methods."""
        return [
            self._impl.get_img_base_name(i) for i in range(self._impl.get_num_cams())
        ]

    def get_first(self):
        return self._impl.first

    def set_first(self, first):
        self._impl.first = first

    def get_last(self):
        return self._impl.last

    def set_last(self, last):
        self._impl.last = last

    def set_img_base_name(self, cam, name):
        if cam < len(self._impl.img_base_name):
            self._impl.img_base_name[cam] = name


class VolumeParams:
    """
    Cython-compatible VolumeParams wrapper for Python engine.

    Provides the same API as bindings/optv/parameters.pyx VolumeParams class.
    """

    def __init__(self, xmin=0.0, xmax=0.0, ymin=0.0, ymax=0.0, zmin=0.0, zmax=0.0):
        """Initialize VolumeParams with min/max coordinates."""
        from algorithms.parameters import VolumePar

        self._impl = VolumePar()

        # Store the bounds (these may be used differently in Cython vs Python)
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

        # Additional attributes used by epipolar code
        self.z_min_lay = [zmin]
        self.z_max_lay = [zmax]
        self.x_lay = [xmin, xmax]

    def get_x_min(self):
        return self._xmin

    def set_x_min(self, value):
        self._xmin = value

    def get_x_max(self):
        return self._xmax

    def set_x_max(self, value):
        self._xmax = value

    def get_y_min(self):
        return self._ymin

    def set_y_min(self, value):
        self._ymin = value

    def get_y_max(self):
        return self._ymax

    def set_y_max(self, value):
        self._ymax = value

    def get_z_min(self):
        return self._zmin

    def set_z_min(self, value):
        self._zmin = value

    def get_z_max(self):
        return self._zmax

    def set_z_max(self, value):
        self._zmax = value


class TrackingParams:
    """
    Cython-compatible TrackingParams wrapper for Python engine.

    Provides the same API as bindings/optv/parameters.pyx TrackingParams class.
    """

    def __init__(
        self,
        pft_par=None,
        chk_par=None,
        cal_tol=None,
        exb_par=None,
        lm_cutoff=None,
        cand=None,
        glob=False,
    ):
        """Initialize TrackingParams."""
        from algorithms.parameters import TrackPar

        self._impl = TrackPar()

        # Set attributes if provided
        if pft_par is not None:
            self._impl.pft_par = pft_par
        if chk_par is not None:
            self._impl.chk_par = chk_par
        if cal_tol is not None:
            self._impl.cal_tol = cal_tol
        if exb_par is not None:
            self._impl.exb_par = exb_par
        if lm_cutoff is not None:
            self._impl.lm_cutoff = lm_cutoff
        if cand is not None:
            self._impl.cand = cand

    def get_pft_par(self):
        return getattr(self._impl, "pft_par", None)

    def set_pft_par(self, value):
        self._impl.pft_par = value

    def get_chk_par(self):
        return getattr(self._impl, "chk_par", None)

    def set_chk_par(self, value):
        self._impl.chk_par = value

    def get_cal_tol(self):
        return getattr(self._impl, "cal_tol", None)

    def set_cal_tol(self, value):
        self._impl.cal_tol = value

    def get_exb_par(self):
        return getattr(self._impl, "exb_par", None)

    def set_exb_par(self, value):
        self._impl.exb_par = value


class TargetParams:
    """
    Cython-compatible TargetParams wrapper for Python engine.

    Provides the same API as bindings/optv/parameters.pyx TargetParams class.
    Maps to Python's TargetPar.
    """

    def __init__(
        self,
        nx=None,
        ny=None,
        corr=None,
        sbt=None,
        sz=None,
        grey=None,
        sumg=None,
        replicates=None,
        criteria=None,
    ):
        """Initialize TargetParams."""
        from algorithms.parameters import TargetPar

        self._impl = TargetPar()

        if nx is not None:
            self._impl.nx = nx
        if ny is not None:
            self._impl.ny = ny
        if corr is not None:
            self._impl.corr = corr
        if sbt is not None:
            self._impl.sbt = sbt
        if sz is not None:
            self._impl.sz = sz
        if grey is not None:
            self._impl.grey = grey

    def get_nx(self):
        return getattr(self._impl, "nx", 0)

    def set_nx(self, value):
        self._impl.nx = value

    def get_ny(self):
        return getattr(self._impl, "ny", 0)

    def set_ny(self, value):
        self._impl.ny = value


class MultimediaParams:
    """
    Cython-compatible MultimediaParams wrapper for Python engine.

    Provides the same API as bindings/optv/parameters.pyx MultimediaParams class.
    Maps to Python's MultimediaPar.
    """

    def __init__(self, n1=1.0, n2=None, n3=1.0, d=None):
        """
        Initialize MultimediaParams.

        Arguments:
        n1 - index of refraction of the first medium (air = 1.0)
        n2 - sequence of indices of refraction of intermediate layers
        n3 - index of refraction of the last medium
        d - sequence of thickness values for intermediate layers
        """
        from algorithms.parameters import MultimediaPar

        self._impl = MultimediaPar(
            n1=n1,
            n2=n2 if n2 is not None else [1.0],
            d=d if d is not None else [0.0],
            n3=n3,
        )

    def get_nlay(self):
        return self._impl.nlay

    def get_n1(self):
        return self._impl.n1

    def set_n1(self, n1):
        self._impl.n1 = n1

    def get_n3(self):
        return self._impl.n3

    def set_n3(self, n3):
        self._impl.n3 = n3

    def get_n2(self):
        return self._impl.n2

    def set_layers(self, refr_index, thickness):
        self._impl.n2 = refr_index
        self._impl.d = thickness
        self._impl.nlay = len(refr_index)
