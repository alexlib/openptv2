import numpy as np
import pytest
import os
from openptv2.algorithms.tracking_frame_buf import (
    Target, read_targets, write_targets, compare_targets,
    read_path_frame, write_path_frame, Frame, compare_path_info,
    Pathinfo, Corres, compare_corres, register_link_candidate,
    reset_links, Corres_dtype
)
from openptv2.algorithms.constants import POSI, PT_UNUSED, PREV_NONE, NEXT_NONE

TEST_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')

EPS = 1e-6


def test_read_targets():
    file_base = os.path.join(TEST_DATA, "sample_")
    frame_num = 42

    t1 = Target(pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1)
    t2 = Target(pnr=1, x=796.0000, y=809.0000, n=13108, nx=113, ny=116, sumg=658928, tnr=0)

    targets = read_targets(file_base, frame_num)
    assert len(targets) == 2
    assert compare_targets(targets[0], t1)
    assert compare_targets(targets[1], t2)


def test_zero_targets():
    file_base = os.path.join(TEST_DATA, "sample_")
    frame_num = 1

    targets = read_targets(file_base, frame_num)
    assert len(targets) == 0


def test_write_targets(tmp_path):
    t1 = Target(pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1)
    t2 = Target(pnr=1, x=796.0000, y=809.0000, n=13108, nx=113, ny=116, sumg=658928, tnr=0)
    tbuf = [t1, t2]

    file_base = os.path.join(str(tmp_path), "test_")
    frame_num = 42
    num_targets = 2

    assert write_targets(tbuf, num_targets, file_base, frame_num)

    targets_read = read_targets(file_base, frame_num)
    assert len(targets_read) == 2
    assert compare_targets(targets_read[0], t1)
    assert compare_targets(targets_read[1], t2)


def test_read_path_frame():
    path_correct = Pathinfo(
        x=np.array([45.219, -20.269, 25.946]),
        prev=PREV_NONE,
        next=NEXT_NONE,
        prio=4,
        finaldecis=1000000.0,
        inlist=0,
    )
    c_correct = Corres(nr=3, p=np.array([96, 66, 26, 26], dtype=np.int32))

    file_base = os.path.join(TEST_DATA, "rt_is")
    frame_num = 818

    cor_buf, path_buf = read_path_frame(file_base, "", "", frame_num)
    assert len(cor_buf) == 80
    assert len(path_buf) == 80

    assert cor_buf[2].nr == c_correct.nr
    assert np.array_equal(cor_buf[2].p, c_correct.p)
    assert compare_path_info(path_buf[2], path_correct)

    path_correct.prev = 0
    path_correct.next = 0
    path_correct.prio = 0

    linkage_base = os.path.join(TEST_DATA, "ptv_is")
    prio_base = os.path.join(TEST_DATA, "added")

    cor_buf, path_buf = read_path_frame(file_base, linkage_base, prio_base, frame_num)
    assert len(cor_buf) == 80
    assert len(path_buf) == 80
    assert cor_buf[2].nr == c_correct.nr
    assert np.array_equal(cor_buf[2].p, c_correct.p)
    assert compare_path_info(path_buf[2], path_correct)


def test_write_path_frame(tmp_path):
    cor_buf = [
        Corres(nr=1, p=np.array([96, 66, 26, 26], dtype=np.int32)),
        Corres(nr=2, p=np.array([30, 31, 32, 33], dtype=np.int32)),
    ]
    path_buf = [
        Pathinfo(
            x=np.array([45.219, -20.269, 25.946]),
            prev=-1, next=-2, prio=4, finaldecis=1000000.0, inlist=0
        ),
        Pathinfo(
            x=np.array([45.219, -20.269, 25.946]),
            prev=-1, next=-2, prio=0, finaldecis=2000000.0, inlist=1
        ),
    ]

    corres_file_base = os.path.join(str(tmp_path), "rt_is")
    linkage_file_base = os.path.join(str(tmp_path), "ptv_is")
    frame_num = 42

    assert write_path_frame(
        cor_buf, path_buf, 2,
        corres_file_base, linkage_file_base, None, frame_num
    )


def test_init_frame():
    cams = 4
    max_targets = 100

    frm = Frame(num_cams=cams, max_targets=max_targets)

    t_target = Target()
    t_path = Pathinfo()

    frm.correspond[42] = Corres(nr=1)
    frm.path_info[42] = t_path

    for cam_ix in range(cams):
        frm.targets[cam_ix][42] = t_target

    assert frm.num_cams == cams
    assert frm.max_targets == max_targets


def test_read_write_frame(tmp_path):
    target_files = [
        os.path.join(str(tmp_path), "target_test_cam0"),
        os.path.join(str(tmp_path), "target_test_cam1"),
    ]
    corres_base = os.path.join(str(tmp_path), "corres_test")
    linkage_base = os.path.join(str(tmp_path), "ptv_test")
    prio_base = os.path.join(str(tmp_path), "added_test")
    frame_num = 7
    cams = 2
    max_targets = 100

    t_target = Target(pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1)

    t_path = Pathinfo(
        x=np.array([45.219, -20.269, 25.946]),
        prev=-1, next=-2, prio=4, finaldecis=1000000.0, inlist=0,
    )

    frm = Frame(num_cams=cams, max_targets=max_targets)
    frm.correspond[2] = Corres(nr=3, p=np.array([96, 66, 26, 26], dtype=np.int32))
    frm.path_info[2] = t_path
    frm.num_parts = 3

    for cam_ix in range(cams):
        frm.targets[cam_ix][42] = t_target
        frm.num_targets[cam_ix] = 43

    frm.num_targets[cams - 1] = 0

    assert frm.write(corres_base, linkage_base, "", target_files, frame_num)

    readback = Frame(num_cams=cams, max_targets=max_targets)
    assert readback.read(corres_base, "", "", target_files, frame_num)

    assert readback.correspond[2].nr == 3
    assert np.array_equal(readback.correspond[2].p, np.array([96, 66, 26, 26]))
    assert compare_path_info(t_path, readback.path_info[2])
    assert compare_targets(t_target, readback.targets[0][42])

    t_path.prev = 0
    t_path.next = 0
    t_path.prio = 0
    frm.path_info[2] = t_path

    assert frm.write(corres_base, linkage_base, prio_base, target_files, frame_num)

    readback = Frame(num_cams=cams, max_targets=max_targets)
    assert readback.read(corres_base, linkage_base, prio_base, target_files, frame_num)

    assert readback.correspond[2].nr == 3
    assert np.array_equal(readback.correspond[2].p, np.array([96, 66, 26, 26]))
    assert compare_path_info(t_path, readback.path_info[2])
    assert compare_targets(t_target, readback.targets[0][42])


def test_compare_corres():
    c1 = Corres(nr=3, p=np.array([96, 66, 26, 26], dtype=np.int32))
    c2 = Corres(nr=3, p=np.array([96, 66, 26, 26], dtype=np.int32))
    c3 = Corres(nr=4, p=np.array([96, 66, 26, 26], dtype=np.int32))
    assert compare_corres(c1, c2)
    assert not compare_corres(c1, c3)


def test_register_link_candidate():
    p = Pathinfo()
    assert p.inlist == 0
    register_link_candidate(p, 0.5, 10)
    assert p.inlist == 1
    assert p.decis[0] == 0.5
    assert p.linkdecis[0] == 10
    register_link_candidate(p, 0.3, 20)
    assert p.inlist == 2
    assert p.decis[1] == 0.3
    assert p.linkdecis[1] == 20


def test_reset_links():
    p = Pathinfo(prev=5, next=10, prio=3)
    reset_links(p)
    assert p.prev == PREV_NONE
    assert p.next == NEXT_NONE
    assert p.prio == 2  # PRIO_DEFAULT
