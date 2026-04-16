import numpy as np
import pytest
from algorithms.sortgrid import sortgrid, nearest_neighbour_pix, read_sortgrid_par, read_calblock
from algorithms.calibration import Calibration
from algorithms.parameters import ControlPar
from openptv2.tracking_framebuf import Target, read_targets
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
    correct_eps = 20 # Value from sortgrid.par in burgers/parameters
    eps = read_sortgrid_par("test_data/burgers/parameters/sortgrid.par")
    assert eps == correct_eps

def test_read_calblock():
    calblock_file = Path("test_data/burgers/cal/target_file.txt")
    assert calblock_file.exists()
    
    fix, num_points = read_calblock(calblock_file)
    assert num_points == 25

def test_sortgrid():
    eps = 20
def test_sortgrid():
    eps = 20
    # For file at test_data/burgers/img_orig/cam1.10001_targets
    # file_base = "test_data/burgers/img_orig/cam1."
    # The read_targets implementation expects `file_base` + `frame_num` + `_targets`
    # Let's verify the `read_targets` implementation
    pix = read_targets("test_data/burgers/img_orig/cam1.", 10001)
    assert len(pix) > 0

    ori_file = "test_data/burgers/cal/cam1.tif.ori"
    add_file = "test_data/burgers/cal/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)
    cpar = ControlPar.from_file("test_data/burgers/parameters/ptv.par")
    fix, nfix = read_calblock("test_data/burgers/cal/target_file.txt")
    assert nfix == 25

    # Sort grid
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    
    # Verify that at least some were assigned (pnr != -999)
    assigned = [p for p in sorted_pix if p is not None and hasattr(p, 'pnr') and p.pnr != -999]
    assert len(assigned) > 0
