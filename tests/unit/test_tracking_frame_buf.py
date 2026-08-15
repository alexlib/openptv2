import os

import numpy as np
import pytest

from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
from openptv2.algorithms.tracking_frame_buf import (
    Corres,
    Frame,
    Pathinfo,
    Target,
    compare_corres,
    compare_path_info,
    compare_targets,
    read_path_frame,
    read_targets,
    register_link_candidate,
    reset_links,
    write_path_frame,
    write_targets,
)

TEST_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "test_data")

EPS = 1e-6


def test_read_targets():
    file_base = os.path.join(TEST_DATA, "sample_")
    frame_num = 42

    t1 = Target(
        pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1
    )
    t2 = Target(
        pnr=1, x=796.0000, y=809.0000, n=13108, nx=113, ny=116, sumg=658928, tnr=0
    )

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
    t1 = Target(
        pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1
    )
    t2 = Target(
        pnr=1, x=796.0000, y=809.0000, n=13108, nx=113, ny=116, sumg=658928, tnr=0
    )
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
        next_idx=NEXT_NONE,
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
    path_correct.next_idx = 0
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
            prev=-1,
            next_idx=-2,
            prio=4,
            finaldecis=1000000.0,
            inlist=0,
        ),
        Pathinfo(
            x=np.array([45.219, -20.269, 25.946]),
            prev=-1,
            next_idx=-2,
            prio=0,
            finaldecis=2000000.0,
            inlist=1,
        ),
    ]

    corres_file_base = os.path.join(str(tmp_path), "rt_is")
    linkage_file_base = os.path.join(str(tmp_path), "ptv_is")
    frame_num = 42

    assert write_path_frame(
        cor_buf, path_buf, 2, corres_file_base, linkage_file_base, None, frame_num
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

    t_target = Target(
        pnr=0, x=1127.0000, y=796.0000, n=13320, nx=111, ny=120, sumg=828903, tnr=1
    )

    t_path = Pathinfo(
        x=np.array([45.219, -20.269, 25.946]),
        prev=-1,
        next_idx=-2,
        prio=4,
        finaldecis=1000000.0,
        inlist=0,
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
    t_path.next_idx = 0
    t_path.prio = 0
    frm.path_info[2] = t_path

    assert frm.write(corres_base, linkage_base, prio_base, target_files, frame_num)

    readback = Frame(num_cams=cams, max_targets=max_targets)
    assert readback.read(corres_base, linkage_base, prio_base, target_files, frame_num)

    assert readback.correspond[2].nr == 3
    assert np.array_equal(readback.correspond[2].p, np.array([96, 66, 26, 26]))
    assert compare_path_info(t_path, readback.path_info[2])
    assert compare_targets(t_target, readback.targets[0][42])


def test_read_raises_when_particle_count_exceeds_max_targets(tmp_path):
    """Regression for a segfault: reading more particles than a Frame's
    fixed-size buffers were allocated for used to silently corrupt memory
    (boundscheck is off in the compiled hot path) instead of failing. A
    frame with more particles than max_targets must now raise a clear
    ValueError from read(), before any buffer is written past its capacity.
    """
    target_files = [os.path.join(str(tmp_path), "target_test_cam0")]
    corres_base = os.path.join(str(tmp_path), "corres_test")
    linkage_base = os.path.join(str(tmp_path), "ptv_test")
    frame_num = 1
    n_particles = 5

    writer = Frame(num_cams=1, max_targets=n_particles)
    for i in range(n_particles):
        writer.correspond[i] = Corres(nr=1, p=np.array([i, -1, -1, -1], dtype=np.int32))
        writer.path_info[i] = Pathinfo(
            x=np.array([float(i), 0.0, 0.0]), prev=-1, next_idx=-2, prio=4,
            finaldecis=1000000.0, inlist=0,
        )
        writer.targets[0][i] = Target(
            pnr=i, x=float(i), y=0.0, n=1, nx=1, ny=1, sumg=1, tnr=i,
        )
    writer.num_parts = n_particles
    writer.num_targets[0] = n_particles
    assert writer.write(corres_base, linkage_base, "", target_files, frame_num)

    too_small = Frame(num_cams=1, max_targets=n_particles - 1)
    with pytest.raises(ValueError, match="exceeds max_targets"):
        too_small.read(corres_base, linkage_base, "", target_files, frame_num)


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
    p = Pathinfo(prev=5, next_idx=10, prio=3)
    reset_links(p)
    assert p.prev == PREV_NONE
    assert p.next_idx == NEXT_NONE
    assert p.prio == 2  # PRIO_DEFAULT


def test_read_path_frame_prefers_store_linkage_over_unrelated_ascii_correspondence_file(tmp_path):
    """Regression test for a real bug found while benchmarking Stage 1/2
    (docs/plans/2026-08-15-tracking-quality-overhaul.md): read_path_frame
    used to decide store-vs-ascii by checking whether the CORRESPONDENCE
    ascii file existed, not whether the STORE had data. A correspondence
    ascii file can exist for reasons unrelated to whether the store holds
    this frame's (already-tracked) linkage -- e.g. it was fixture input, or
    written by an earlier, non-store-backed step. Store-backed tracking
    writes linkage ONLY to the store, never ascii, so once the correspondence
    file's mere existence wrongly skipped the store branch, the linkage
    ascii open failed and the frame silently read back as fully unlinked --
    which trackback_c's buffer-priming re-write then persisted as real data,
    wiping out an already-tracked frame's links (full_forward() correctly
    links a store-backed run; full_backward() immediately afterward read
    every frame back as unlinked and overwrote the store with zeros).
    """
    from openptv2.storage import RunStore

    corres_base = str(tmp_path / "rt_is")
    linkage_base = str(tmp_path / "ptv_is")
    frame = 1

    # An ascii correspondence file exists for this frame (fixture input /
    # legacy sequence-step output) -- but no ascii linkage file was ever
    # written, matching a store-backed tracking run.
    with open(f"{corres_base}.{frame}", "w") as fh:
        fh.write("2\n")
        fh.write("1    0.000    0.000    0.000    0   0   0   0\n")
        fh.write("2    1.000    0.000    0.000    1   1   1   1\n")

    store = RunStore(str(tmp_path / "run.zarr"), mode="w")
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cam_ids = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    store.write_correspondences(frame, pos, cam_ids)
    # Real, already-tracked linkage: particle 0 -> particle 1.
    store.write_linkage(
        frame,
        prev_ids=np.array([-1, 0], dtype=np.int32),
        next_ids=np.array([1, -2], dtype=np.int32),
        pos_3d=pos,
        name="ptv_is",
    )

    cor_buf, path_buf = read_path_frame(corres_base, linkage_base, "", frame, store=store)

    assert len(path_buf) == 2
    assert path_buf[0].prev == -1
    assert path_buf[0].next_idx == 1, (
        "read_path_frame ignored the store's real linkage and fell back to "
        "the (nonexistent) ascii linkage file, reading the frame as unlinked"
    )
    assert path_buf[1].prev == 0
    assert path_buf[1].next_idx == -2
