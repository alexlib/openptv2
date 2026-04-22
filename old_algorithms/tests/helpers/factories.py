"""Factories for deterministic synthetic test scenes."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from algorithms.calibration import Calibration
from algorithms.correspondences import MatchedCoords
from algorithms.imgcoord import image_coordinates
from algorithms.parameters import ControlPar, MultimediaPar, VolumePar
from algorithms.tracking_frame_buf import Frame, Target
from algorithms.trafo import metric_to_pixel

CAL_DIR = "test_data/calibration"


def build_corresp_control_par() -> ControlPar:
    """Build deterministic control parameters for synthetic correspondences tests."""
    mm = MultimediaPar(n1=1.0, n2=[1.0001], n3=1.0001, d=[1.0])
    return ControlPar(
        num_cams=4,
        imx=1280,
        imy=1024,
        pix_x=0.017,
        pix_y=0.017,
        chfield=0,
        mm=mm,
        all_cam_flag=0,
    )


def build_corresp_volume_par() -> VolumePar:
    """Build volume parameters matching the correspondences synthetic test scene."""
    return VolumePar(
        x_lay=[-250.0, 250.0],
        z_min_lay=[-100.0, -100.0],
        z_max_lay=[100.0, 100.0],
        cnx=0.3,
        cny=0.3,
        cn=0.01,
        csumg=0.01,
        corrmin=33.0,
        eps0=1.0,
    )


def load_sym_calibrations(num_cams: int = 4) -> List[Calibration]:
    """Load symmetric camera calibrations used by synthetic correspondences tests."""
    cals: List[Calibration] = []
    for cam in range(num_cams):
        cal = Calibration()
        cal.from_file(
            f"{CAL_DIR}/sym_cam{cam + 1}.tif.ori",
            f"{CAL_DIR}/cam1.tif.addpar",
        )
        cals.append(cal)
    return cals


def generate_grid_frame(
    cals: List[Calibration],
    cpar: ControlPar,
    *,
    num_cams: int = 4,
    rows: int = 4,
    cols: int = 4,
    spacing: float = 10.0,
) -> Tuple[Frame, list]:
    """Generate deterministic 4-camera synthetic frame and corrected coords."""
    num_pts = rows * cols
    frm = Frame(num_cams)
    corrected = [None] * num_cams
    mm = cpar.mm

    for cam in range(num_cams):
        frm.num_targets[cam] = num_pts
        targs = [None] * num_pts

        for row in range(rows):
            for col in range(cols):
                targ_ix = row * cols + col
                # Keep this consistent with historical C/Cython synthetic scene.
                if cam % 2:
                    targ_ix = (num_pts - 1) - targ_ix

                pos3d = spacing * np.array([[col, row, 0]], dtype=np.float64)
                pos2d = image_coordinates(pos3d, cals[cam], mm)
                px, py = metric_to_pixel(pos2d[0, 0], pos2d[0, 1], cpar)

                t = Target()
                t.pnr = targ_ix
                t.x = px
                t.y = py
                t.n = 25
                t.nx = 5
                t.ny = 5
                t.sumg = 10
                t.tnr = -1
                targs[targ_ix] = t

        frm.targets[cam] = targs
        mc = MatchedCoords(targs, cpar, cals[cam])
        corrected[cam] = mc.buf

    return frm, corrected
