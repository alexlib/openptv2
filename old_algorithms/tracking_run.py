"""Tracking run module."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .calibration import Calibration
from .tracking_frame_buf import FrameBuf

from .multimed import volumedimension, CalibRawArrays, init_mmlut
from .parameters import (
    ControlPar,
    SequencePar,
    TrackParTuple,
    VolumePar,
    convert_track_par_to_tuple,
    read_control_par,
    read_sequence_par,
    read_track_par,
    read_volume_par,
)


@dataclass
class TrackingRun:
    """A tracking run."""

    fb: FrameBuf
    seq_par: SequencePar
    tpar: TrackParTuple
    vpar: VolumePar
    cpar: ControlPar
    cal: List[Calibration]
    flatten_tol: float = 0.0
    ymin: float = 0.0
    ymax: float = 0.0
    lmax: float = 0.0
    npart: int = 0
    nlinks: int = 0

    def __init__(
        self,
        seq_par: SequencePar,
        tpar: TrackParTuple,
        vpar: VolumePar,
        cpar: ControlPar,
        buf_len: int,
        max_targets: int,
        corres_file_base: str,
        linkage_file_base: str,
        prio_file_base: str,
        cal: List[Calibration],
        flatten_tol: float,
    ):
        self.tpar = tpar
        self.vpar = vpar
        self.cpar = cpar
        self.seq_par = seq_par
        self.cal = cal
        self.flatten_tol = flatten_tol

        self.fb = FrameBuf(
            buf_len,
            cpar.num_cams,
            max_targets,
            corres_file_base,
            linkage_file_base,
            prio_file_base,
            seq_par.img_base_name,
        )

        self.lmax = math.sqrt(
            (tpar.dvxmin - tpar.dvxmax) ** 2
            + (tpar.dvymin - tpar.dvymax) ** 2
            + (tpar.dvzmin - tpar.dvzmax) ** 2
        )

        (
            vpar.x_lay[1],
            vpar.x_lay[0],
            self.ymax,
            self.ymin,
            vpar.z_max_lay[1],
            vpar.z_min_lay[0],
        ) = volumedimension(
            vpar.x_lay[1],
            vpar.x_lay[0],
            self.ymax,
            self.ymin,
            vpar.z_max_lay[1],
            vpar.z_min_lay[0],
            vpar,
            cpar,
            cal,
        )

        self.npart = 0
        self.nlinks = 0

        # Ensure MMLUT is initialized for each camera
        for c in cal:
            if c.mmlut.nz == 0:
                init_mmlut(vpar, cpar, c)

        # Pre-extract raw calibration arrays for fast numba projections
        self.raw_cal = [CalibRawArrays(c, cpar) for c in cal]

        # Pack calibration data for all cameras into single arrays for Numba vectorization
        self.cal_ex_pos = np.ascontiguousarray(np.stack([c.ex_pos for c in self.raw_cal]))
        self.cal_ex_dm = np.ascontiguousarray(np.stack([c.ex_dm for c in self.raw_cal]))
        self.cal_int_cc = np.ascontiguousarray(np.array([c.int_cc for c in self.raw_cal], dtype=np.float64))
        self.cal_int_xh = np.ascontiguousarray(np.array([c.int_xh for c in self.raw_cal], dtype=np.float64))
        self.cal_int_yh = np.ascontiguousarray(np.array([c.int_yh for c in self.raw_cal], dtype=np.float64))
        self.cal_added_par = np.ascontiguousarray(np.stack([c.added_par for c in self.raw_cal]))
        self.cal_glass_par = np.ascontiguousarray(np.stack([c.glass_par for c in self.raw_cal]))
        
        self.cal_mm_d = np.ascontiguousarray(np.stack([c.mm_d for c in self.raw_cal]))
        self.cal_mm_n1 = np.ascontiguousarray(np.array([c.mm_n1 for c in self.raw_cal], dtype=np.float64))
        self.cal_mm_n2 = np.ascontiguousarray(np.stack([c.mm_n2 for c in self.raw_cal]))
        self.cal_mm_n3 = np.ascontiguousarray(np.array([c.mm_n3 for c in self.raw_cal], dtype=np.float64))
        self.cal_mm_nlay = np.ascontiguousarray(np.array([c.mm_nlay for c in self.raw_cal], dtype=np.int32))
        
        self.cal_mmlut_origin = np.ascontiguousarray(np.stack([c.mmlut_origin for c in self.raw_cal]))
        self.cal_mmlut_nz = np.ascontiguousarray(np.array([c.mmlut_nz for c in self.raw_cal], dtype=np.int32))
        self.cal_mmlut_nr = np.ascontiguousarray(np.array([c.mmlut_nr for c in self.raw_cal], dtype=np.int32))
        self.cal_mmlut_rw = np.ascontiguousarray(np.array([c.mmlut_rw for c in self.raw_cal], dtype=np.int32))
        self.cal_imx = np.ascontiguousarray(np.array([c.imx for c in self.raw_cal], dtype=np.int32))
        self.cal_imy = np.ascontiguousarray(np.array([c.imy for c in self.raw_cal], dtype=np.int32))
        self.cal_pix_x = np.ascontiguousarray(np.array([c.pix_x for c in self.raw_cal], dtype=np.float64))
        self.cal_pix_y = np.ascontiguousarray(np.array([c.pix_y for c in self.raw_cal], dtype=np.float64))

        # List of arrays for varying-size data (LUTs can have different sizes)
        self.cal_mmlut_data = [c.mmlut_data for c in self.raw_cal]


def tr_new(
    seq_par_fname: Path,
    tpar_fname: Path,
    vpar_fname: Path,
    cpar_fname: Path,
    buf_len: int,
    max_targets: int,
    corres_file_base: str,
    linkage_file_base: str,
    prio_file_base: str,
    cal: List[Calibration],
    flatten_tol: float,
) -> TrackingRun:
    """Create a new tracking run from legacy files."""
    cpar = read_control_par(cpar_fname)
    seq_par = read_sequence_par(seq_par_fname, cpar.num_cams)
    tpar = convert_track_par_to_tuple(read_track_par(tpar_fname))
    vpar = read_volume_par(vpar_fname)

    tr = TrackingRun(
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
    )

    return tr
