import numpy as np
from dataclasses import dataclass, field
from algorithms.parameters import SequencePar, TrackPar, VolumePar, ControlPar
from algorithms.tracking_frame_buf import Frame

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
    fb: Frame = field(init=False)
    lmax: float = field(init=False)
    npart: int = 0
    nlinks: int = 0

    def __post_init__(self):
        self.fb = Frame(self.cpar.num_cams, self.max_targets)
        # lmax: Euclidean norm of the tracking volume diagonal
        self.lmax = np.linalg.norm([
            self.tpar.dvxmax - self.tpar.dvxmin,
            self.tpar.dvymax - self.tpar.dvymin,
            self.tpar.dvzmax - self.tpar.dvzmin
        ])

def tr_new(seq_par, tpar, vpar, cpar, buf_len, max_targets,
           corres_file_base, linkage_file_base, prio_file_base, cal, flatten_tol):
    """Python translation of C tr_new/tr_new_legacy.

    Accepts either parameter objects or file paths. If file paths are
    passed, reads and constructs parameter objects automatically.
    """
    from algorithms.parameters import (
        SequencePar, TrackPar, VolumePar, ControlPar,
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
        tpar = convert_track_par_to_tuple(tpar)

    if isinstance(vpar, str):
        vpar = VolumePar.from_file(vpar)

    return TrackingRun(seq_par, tpar, vpar, cpar_obj, buf_len, max_targets,
                       corres_file_base, linkage_file_base, prio_file_base, cal, flatten_tol)
