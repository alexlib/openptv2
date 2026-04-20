import numpy as np
import pytest
from algorithms.sortgrid import sortgrid, nearest_neighbour_pix, read_sortgrid_par, read_calblock
from algorithms.calibration import Calibration
from algorithms.parameters import ControlPar
from algorithms.tracking_frame_buf import Target, read_targets
from pathlib import Path

EPS = 1e-6

def test_nearest_neighbour_pix():
    class SimpleTarget:
        def __init__(self, x, y): self.x = x; self.y = y; self.pnr = 0
    t1 = SimpleTarget(1127.0, 796.0)
    targets = [t1]

    pnr = nearest_neighbour_pix(targets, 1128.0, 795.0, 0.0)
    assert pnr == -999

    pnr = nearest_neighbour_pix(targets, 1128.0, 795.0, -1.0)
    assert pnr == -999

    pnr = nearest_neighbour_pix(targets, -1127.0, -796.0, 1e3)
    assert pnr == -999

    pnr = nearest_neighbour_pix(targets, 1127.0, 796.0, 1e-5)
    assert pnr == 0

def test_read_sortgrid_par():
    correct_eps = 25
    eps = read_sortgrid_par("test_data/parameters/sortgrid.par")
    assert eps == correct_eps

def test_read_calblock():
    calblock_file = Path("test_data/calibration/calblock.txt")
    assert calblock_file.exists()

    fix, num_points = read_calblock(calblock_file)
    assert num_points == 5

def test_sortgrid():
    eps = read_sortgrid_par("test_data/parameters/sortgrid.par")
    assert eps == 25

    pix = read_targets("test_data/sample_", 42)
    assert len(pix) == 2

    cal = Calibration.from_file("test_data/calibration/cam1.tif.ori",
                                "test_data/calibration/cam1.tif.addpar")
    cpar = ControlPar.from_file("test_data/parameters/ptv.par")
    fix, nfix = read_calblock("test_data/calibration/calblock.txt")
    assert nfix == 5

    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    assert sorted_pix[0] is None or sorted_pix[0].pnr == -999
    assert sorted_pix[1] is None or sorted_pix[1].pnr == -999

    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), 120, pix)
    assert sorted_pix[1] is not None
    assert sorted_pix[1].pnr == 1
    assert sorted_pix[1].x == 796
