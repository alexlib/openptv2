import numpy as np
import pytest
import os
import shutil
from pathlib import Path

from algorithms.track3d import find_candidates_in_3d, track3d_loop
from algorithms.track import track_forward_start
from algorithms.tracking_frame_buf import Frame
from algorithms.tracking_run import tr_new
from algorithms.parameters import read_control_par
from algorithms.calibration import Calibration

EPS = 1e-5

def read_all_calibration(num_cams, base_path="test_data/track"):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals

def test_find_candidates_in_3d_empty_frame():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 0
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 0

def test_find_candidates_in_3d_single_match():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 1
    frm.path_info = [type('Pathinfo', (), {'x': np.array([5.0, 5.0, 5.0])})()]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 1
    assert indices[0] == 0

def test_find_candidates_in_3d_no_match_outside_box():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 1
    frm.path_info = [type('Pathinfo', (), {'x': np.array([5.0, 5.0, 5.0])})()]
    pos = np.array([10.0, 10.0, 10.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 0

def test_find_candidates_in_3d_multiple_matches():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 5
    frm.path_info = [type('Pathinfo', (), {'x': np.array(p, dtype=np.float64)})() for p in [[0,0,0],[1,1,1],[5,5,5],[6,6,6],[10,10,10]]]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 2.0, 2.0, 2.0, 4)
    assert len(indices) == 2

def test_find_candidates_in_3d_max_cands_limit():
    frm = Frame(num_cams=1, max_targets=20)
    frm.num_parts = 10
    frm.path_info = [type('Pathinfo', (), {'x': np.array([5.0 + i * 0.01, 5.0, 5.0])})() for i in range(10)]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 3)
    assert len(indices) == 3

def test_find_candidates_in_3d_boundary():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 1
    frm.path_info = [type('Pathinfo', (), {'x': np.array([6.0, 5.0, 5.0])})()]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 0

def test_track3d_no_add():
    import os
    original = os.getcwd()
    try:
        test_dir = os.path.join(os.path.dirname(__file__), '../../test_data/track')
        os.chdir(test_dir)
        if os.path.exists("res"): shutil.rmtree("res")
        if os.path.exists("img"): shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")
        run = tr_new(
            "parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
            "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added", calib, 0.0001
        )
        run.tpar = run.tpar._replace(add=0)
        track_forward_start(run)
        track3d_loop(run, run.seq_par.first)
        for step in range(run.seq_par.first + 1, run.seq_par.last):
            track3d_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)
        range_val = run.seq_par.last - run.seq_par.first
        npart = run.npart / range_val
        nlinks = run.nlinks / range_val
        assert abs(npart - 0.8) < EPS
        assert abs(nlinks - 0.8) < EPS
    finally:
        os.chdir(original)

def track3d_test_cavity():
    import os
    original = os.getcwd()
    try:
        os.chdir("test_data/test_cavity")
        if os.path.exists("res"): shutil.rmtree("res")
        if os.path.exists("img"): shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        run = tr_new(
            "parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
            "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added", calib, 0.0001
        )

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            track3d_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 672 + 699 + 711
        assert run.nlinks >= 132 + 176 + 144

    finally:
        os.chdir(original)

def track3d_test_burgers():
    import os
    original = os.getcwd()
    try:
        os.chdir("test_data/burgers")
        if os.path.exists("res"): shutil.rmtree("res")
        if os.path.exists("img"): shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        run = tr_new(
            "parameters/sequence.par", "parameters/track.par", "parameters/criteria.par",
            "parameters/ptv.par", 4, 20000, "res/rt_is", "res/ptv_is", "res/added", calib, 0.0001
        )

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            track3d_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 19
        assert run.nlinks == 18

    finally:
        os.chdir(original)

def test_track3d_test_cavity():
    track3d_test_cavity()
    
def test_track3d_test_burgers():
    track3d_test_burgers()
