from dataclasses import dataclass, field

import cython
import numpy as np

from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.tracking_frame_buf import FrameBuf


@dataclass
class TrackingRun:
    seq_par: SequencePar
    tpar: TrackPar
    vpar: VolumePar
    cpar: ControlPar
    buf_len: int
    max_targets: int
    corres_file_base: str
    linkage_file_base: str
    prio_file_base: str
    cal: list
    flatten_tol: float
    fb: FrameBuf = field(init=False)
    lmax: float = field(init=False)
    ymin: float = field(init=False)
    ymax: float = field(init=False)
    npart: int = 0
    nlinks: int = 0

    def __post_init__(self):
        self.fb = FrameBuf(
            self.buf_len,
            self.cpar.num_cams,
            self.max_targets,
            self.corres_file_base,
            self.linkage_file_base,
            self.prio_file_base,
            self.seq_par.img_base_name,
        )

        self.lmax = np.linalg.norm(
            [
                self.tpar.dvxmin - self.tpar.dvxmax,
                self.tpar.dvymin - self.tpar.dvymax,
                self.tpar.dvzmin - self.tpar.dvzmax,
            ]
        )

        from openptv2.algorithms.multimed import init_mmlut, volumedimension

        xmax, xmin, self.ymax, self.ymin, zmax, zmin = volumedimension(
            self.vpar, self.cpar, self.cal
        )
        self.vpar.X_lay[1] = xmax
        self.vpar.X_lay[0] = xmin
        self.vpar.Zmax_lay[1] = zmax
        self.vpar.Zmin_lay[0] = zmin

        for c in self.cal:
            if not c.mmlut.is_initialized:
                init_mmlut(self.vpar, self.cpar, c)


@cython.ccall
def tr_new(
    seq_par,
    tpar,
    vpar,
    cpar,
    buf_len,
    max_targets,
    corres_file_base,
    linkage_file_base,
    prio_file_base,
    cal,
    flatten_tol,
):
    """Python translation of C tr_new/tr_new_legacy.

    Accepts either parameter objects or file paths. If file paths are
    passed, reads and constructs parameter objects automatically.
    """
    from openptv2.algorithms.parameters import (
        ControlPar,
        SequencePar,
        TrackPar,
        VolumePar,
        convert_track_par_to_tuple,
    )

    if isinstance(cpar, str):
        cpar_obj = ControlPar.from_file(cpar)
    else:
        cpar_obj = cpar

    if isinstance(seq_par, str):
        seq_par = SequencePar.from_file(seq_par, cpar_obj.num_cams)

    if isinstance(tpar, str):
        tpar = TrackPar.from_file(tpar)
    # Convert a TrackPar (from .par or .from_yaml) into the tuple the tracker
    # uses; an already-converted tuple is passed through unchanged.
    if isinstance(tpar, TrackPar):
        tpar = convert_track_par_to_tuple(tpar)

    if isinstance(vpar, str):
        vpar = VolumePar.from_file(vpar)

    return TrackingRun(
        seq_par,
        tpar,
        vpar,
        cpar_obj,
        buf_len,
        max_targets,
        corres_file_base,
        linkage_file_base,
        prio_file_base,
        cal,
        flatten_tol,
    )


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
