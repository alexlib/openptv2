import numpy as np
import pytest
import os
from algorithms.tracking_frame_buf import (
    Target, read_targets, write_targets, compare_targets,
    read_path_frame, write_path_frame, Frame, compare_path_info, Pathinfo, Corres_dtype
)
from algorithms.constants import POSI

EPS = 1e-6

def test_read_targets():
    file_base = "testing_fodder/sample_"
    frame_num = 42
    
    t1 = Target(pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1)
    t2 = Target(pnr=1, x=796.0000, y=809.0000, n=13108, nx=113, ny=116, sumg=658928, tnr=0)
    
    targets = read_targets(file_base, frame_num)
    assert len(targets) == 2
    assert compare_targets(targets[0], t1)
    assert compare_targets(targets[1], t2)

def test_zero_targets():
    file_base = "testing_fodder/sample_"
    frame_num = 1
    
    targets = read_targets(file_base, frame_num)
    assert len(targets) == 0

def test_write_targets():
    t1 = Target(pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1)
    t2 = Target(pnr=1, x=796.0000, y=809.0000, n=13108, nx=113, ny=116, sumg=658928, tnr=0)
    tbuf = [t1, t2]
    
    file_base = "testing_fodder/test_"
    frame_num = 42
    num_targets = 2
    
    assert write_targets(tbuf, num_targets, file_base, frame_num)
    
    targets_read = read_targets(file_base, frame_num)
    assert len(targets_read) == 2
    assert compare_targets(targets_read[0], t1)
    assert compare_targets(targets_read[1], t2)
    
    try:
        os.remove("testing_fodder/test_0042_targets")
    except OSError:
        pass

def test_read_path_frame():
    path_correct = Pathinfo(
        x=np.array([45.219, -20.269, 25.946]),
        prev_frame=-1,
        next_frame=-2,
        prio=4,
        finaldecis=1000000.0,
        inlist=0,
        decis=[0.0] * POSI,
        linkdecis=[-999] * POSI
    )
    
    c_correct_nr = 3
    c_correct_p = np.array([96, 66, 26, 26], dtype=np.int32)
    
    file_base = "testing_fodder/rt_is"
    frame_num = 818
    
    # Test unlinked frame
    cor_buf, path_buf = read_path_frame(file_base, "", "", frame_num)
    assert len(cor_buf) == 80
    assert len(path_buf) == 80
    
    assert cor_buf[2].nr == c_correct_nr
    assert np.array_equal(cor_buf[2].p, c_correct_p)
    assert compare_path_info(path_buf[2], path_correct)
    
    # Test frame with links
    path_correct.prev_frame = 0
    path_correct.next_frame = 0
    path_correct.prio = 0
    
    linkage_base = "testing_fodder/ptv_is"
    prio_base = "testing_fodder/added"
    
    cor_buf, path_buf = read_path_frame(file_base, linkage_base, prio_base, frame_num)
    
    assert len(cor_buf) == 80
    assert len(path_buf) == 80
    assert cor_buf[2].nr == c_correct_nr
    assert np.array_equal(cor_buf[2].p, c_correct_p)
    assert compare_path_info(path_buf[2], path_correct)

def test_write_path_frame():
    corres_nr = np.array([1, 2], dtype=np.int32)
    corres_p = np.array([
        [96, 66, 26, 26],
        [30, 31, 32, 33]
    ], dtype=np.int32)
    
    path_buf = [
        Pathinfo(
            x=np.array([45.219, -20.269, 25.946]),
            prev_frame=-1, next_frame=-2, prio=4, finaldecis=1000000.0, inlist=0
        ),
        Pathinfo(
            x=np.array([45.219, -20.269, 25.946]),
            prev_frame=-1, next_frame=-2, prio=0, finaldecis=2000000.0, inlist=1
        )
    ]
    
    corres_file_base = "testing_fodder/rt_is"
    linkage_file_base = "testing_fodder/ptv_is"
    frame_num = 42
    
    assert write_path_frame(
        corres_nr, corres_p, path_buf, 2,
        corres_file_base, linkage_file_base, "", frame_num
    )
    
    try:
        os.remove("testing_fodder/rt_is.42")
        os.remove("testing_fodder/ptv_is.42")
    except OSError:
        pass

def test_init_frame():
    cams = 4
    max_targets = 100
    
    frm = Frame(num_cams=cams, max_targets=max_targets)
    
    t_target = Target()
    t_path = Pathinfo()
    
    frm.corres_nr[42] = 1
    frm.path_info[42] = t_path
    
    for cam_ix in range(cams):
        frm.targets[cam_ix][42] = t_target
        
    assert frm.num_cams == cams
    assert frm.max_targets == max_targets

def test_read_write_frame():
    target_files = [
        "testing_fodder/target_test_cam0",
        "testing_fodder/target_test_cam1"
    ]
    corres_base = "testing_fodder/corres_test"
    linkage_base = "testing_fodder/ptv_test"
    prio_base = "testing_fodder/added_test"
    frame_num = 7
    cams = 2
    max_targets = 100
    
    t_target = Target(pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1)
    
    t_path = Pathinfo(
        x=np.array([45.219, -20.269, 25.946]),
        prev_frame=-1, next_frame=-2, prio=4, finaldecis=1000000.0, inlist=0,
        decis=[0.0] * POSI, linkdecis=[-999] * POSI
    )
    
    frm = Frame(num_cams=cams, max_targets=max_targets)
    frm.corres_nr[2] = 3
    frm.corres_p[2] = np.array([96, 66, 26, 26])
    frm.path_info[2] = t_path
    frm.num_parts = 3
    
    for cam_ix in range(cams):
        frm.targets[cam_ix][42] = t_target
        frm.num_targets[cam_ix] = 43
        
    frm.num_targets[cams - 1] = 0
    
    assert frm.write(corres_base, linkage_base, "", target_files, frame_num)
    
    readback = Frame(num_cams=cams, max_targets=max_targets)
    assert readback.read(corres_base, "", "", target_files, frame_num)
    
    assert readback.corres_nr[2] == 3
    assert np.array_equal(readback.corres_p[2], np.array([96, 66, 26, 26]))
    assert compare_path_info(t_path, readback.path_info[2])
    assert compare_targets(t_target, readback.targets[0][42])
    
    t_path.prev_frame = 0
    t_path.next_frame = 0
    t_path.prio = 0
    frm.path_info[2] = t_path
    
    assert frm.write(corres_base, linkage_base, prio_base, target_files, frame_num)
    
    readback = Frame(num_cams=cams, max_targets=max_targets)
    assert readback.read(corres_base, linkage_base, prio_base, target_files, frame_num)
    
    assert readback.corres_nr[2] == 3
    assert np.array_equal(readback.corres_p[2], np.array([96, 66, 26, 26]))
    assert compare_path_info(t_path, readback.path_info[2])
    assert compare_targets(t_target, readback.targets[0][42])
    
    try:
        os.remove(f"{corres_base}.{frame_num}")
        os.remove(f"{linkage_base}.{frame_num}")
        os.remove(f"{prio_base}.{frame_num}")
        os.remove("testing_fodder/target_test_cam100007_targets")
        os.remove("testing_fodder/target_test_cam000007_targets")
    except OSError:
        pass
