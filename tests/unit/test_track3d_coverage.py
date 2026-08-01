import types

import numpy as np

from openptv2.algorithms.track3d import (
    _sync_soa_to_aos,
    find_candidates_in_3d,
    is_compiled,
    sort,
    track3d_loop,
)
from openptv2.algorithms.tracking_frame_buf import Frame, FrameBuf


def test_is_compiled_returns_bool():
    assert isinstance(is_compiled(), bool)


def test_sort_ascending_by_a():
    a = [3.0, 1.0, 2.0]
    b = [30, 10, 20]
    sorted_a, sorted_b = sort(len(a), a, b)
    assert sorted_a == [1.0, 2.0, 3.0]
    assert sorted_b == [10, 20, 30]


def test_sort_empty():
    sorted_a, sorted_b = sort(0, [], [])
    assert sorted_a == []
    assert sorted_b == []


def test_sync_soa_to_aos_roundtrip():
    frm = Frame(num_cams=1, max_targets=8)
    frm.num_parts = 2

    # Populate SoA path arrays
    frm.path_x[0] = [1.0, 2.0, 3.0]
    frm.path_x[1] = [4.0, 5.0, 6.0]
    frm.path_prev[:2] = [7, 8]
    frm.path_next[:2] = [9, 10]
    frm.path_prio[:2] = [1, 2]

    # Populate SoA correspond arrays
    frm.corres_nr[:2] = [11, 12]
    frm.corres_p[0] = [1, 2, 3, 4]
    frm.corres_p[1] = [5, 6, 7, 8]

    # Populate SoA target-number array for the one camera
    frm.num_targets[0] = 2
    frm.targ_tnr[0][0] = 100
    frm.targ_tnr[0][1] = 101

    _sync_soa_to_aos(frm)

    # AoS path_info mirrors SoA
    assert list(frm.path_info[0].x) == [1.0, 2.0, 3.0]
    assert list(frm.path_info[1].x) == [4.0, 5.0, 6.0]
    assert frm.path_info[0].prev == 7
    assert frm.path_info[1].prev == 8
    assert frm.path_info[0].next_idx == 9
    assert frm.path_info[1].next_idx == 10
    assert frm.path_info[0].prio == 1
    assert frm.path_info[1].prio == 2

    # AoS correspond mirrors SoA
    assert frm.correspond[0].nr == 11
    assert frm.correspond[1].nr == 12
    assert list(frm.correspond[0].p) == [1, 2, 3, 4]
    assert list(frm.correspond[1].p) == [5, 6, 7, 8]

    # AoS targets mirror SoA tnr
    assert frm.targets[0][0].tnr == 100
    assert frm.targets[0][1].tnr == 101


def _pathinfo_frame(positions):
    frm = Frame(num_cams=1, max_targets=16)
    frm.num_parts = len(positions)
    frm.path_info = [
        type("Pathinfo", (), {"x": np.asarray(p, dtype=np.float64)})()
        for p in positions
    ]
    return frm


def test_find_candidates_returns_closest_sorted():
    frm = _pathinfo_frame([[5.3, 5.0, 5.0], [5.1, 5.0, 5.0], [5.2, 5.0, 5.0]])
    pos = np.array([5.0, 5.0, 5.0])
    idx = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    # Closest first: particle 1 (0.1) < particle 2 (0.2) < particle 0 (0.3)
    assert idx == [1, 2, 0]


def test_find_candidates_more_than_max_cands():
    # 5 in-box candidates, max_cands=4: the farthest never beats a filled slot,
    # exercising the slot loop exhausting without a break.
    frm = _pathinfo_frame([[5.0 + 0.1 * k, 5.0, 5.0] for k in range(1, 6)])
    pos = np.array([5.0, 5.0, 5.0])
    idx = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert idx == [0, 1, 2, 3]


def test_find_candidates_empty_and_out_of_box():
    frm = _pathinfo_frame([[10.0, 10.0, 10.0]])
    pos = np.array([0.0, 0.0, 0.0])
    assert find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4) == []


def _make_frame(positions, prev_links):
    """Build a 1-camera Frame with the given particle positions/prev links."""
    frm = Frame(num_cams=1, max_targets=16)
    frm.num_parts = len(positions)
    for i, (pos, prev) in enumerate(zip(positions, prev_links)):
        p = frm.path_info[i]
        p.x[:] = np.asarray(pos, dtype=np.float64)
        p.prev = prev
        p.next_idx = -1
        p.prio = 4
    return frm


def _make_run(tmp_path, prev, curr, nextf, first, last):
    fb = FrameBuf(
        buf_len=3,
        num_cams=1,
        max_targets=16,
        corres_file_base=str(tmp_path / "rt_is"),
        linkage_file_base=str(tmp_path / "ptv_is"),
        prio_file_base=str(tmp_path / "added"),
        target_file_base=[str(tmp_path / "cam1.")],
    )
    fb.buf[0] = prev
    fb.buf[1] = curr
    fb.buf[2] = nextf
    tpar = types.SimpleNamespace(dvxmax=2.0, dvymax=2.0, dvzmax=2.0)
    seq_par = types.SimpleNamespace(first=first, last=last)
    return types.SimpleNamespace(fb=fb, tpar=tpar, npart=0, nlinks=0, seq_par=seq_par)


def test_track3d_loop_no_read_branch(tmp_path):
    """Exercise track3d_loop with the read_frame_at_end branch skipped."""
    prev = _make_frame([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [-1, -1])
    curr = _make_frame([[0.1, 0.0, 0.0], [1.1, 1.0, 1.0]], [-1, -1])
    nextf = _make_frame([[0.2, 0.0, 0.0], [1.2, 1.0, 1.0]], [-1, -1])

    # last chosen so step < last - 2 is False -> read_frame_at_end skipped
    run = _make_run(tmp_path, prev, curr, nextf, first=1, last=2)
    curr_parts = run.fb.buf[1].num_parts

    track3d_loop(run, step=1)

    # Accounting invariants (no magic values)
    assert run.npart == curr_parts
    assert run.nlinks >= 0
    # write_frame_from_start wrote the current-frame linkage file for step 1
    assert (tmp_path / "ptv_is.1").exists()


def test_track3d_loop_empty_frames(tmp_path):
    """track3d_loop over empty frames links nothing and writes an empty frame."""
    prev = _make_frame([], [])
    curr = _make_frame([], [])
    nextf = _make_frame([], [])

    # last high so step < last - 2 is True -> read_frame_at_end branch taken
    # (the file for step+3 does not exist, so read returns False gracefully)
    run = _make_run(tmp_path, prev, curr, nextf, first=1, last=10)
    track3d_loop(run, step=1)

    assert run.npart == 0
    assert run.nlinks == 0
    assert (tmp_path / "ptv_is.1").exists()
