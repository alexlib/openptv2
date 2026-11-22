"""Backward = the same tracker on mirrored frame numbers, merged Forward-First (docs/trackers.md)."""

import numpy as np

from openptv2.reversed_pass import ReversedStore, _links_of, merge_reversed_pass
from openptv2.storage import RunStore


def _store(tmp_path, n_frames=4):
    s = RunStore(tmp_path / "run.zarr", mode="w")
    for f in range(1, n_frames + 1):
        pts = np.array([[float(f), 0.0, 0.0], [float(f), 5.0, 0.0]])
        s.write_correspondences(f, pts, np.array([[0, 0], [1, 1]], dtype=np.int32))
    return s


def test_reversed_store_mirrors_frames_and_keeps_writes_in_memory(tmp_path):
    s = _store(tmp_path)
    rev = ReversedStore(s, 1, 4)
    assert np.allclose(rev.read_correspondences(1)[0][:, 0], 4.0)      # reversed frame 1 = real frame 4
    assert np.allclose(rev.read_correspondences(4)[0][:, 0], 1.0)
    rev.write_correspondences(2, np.zeros((1, 3)), np.zeros((1, 2), dtype=np.int32))
    assert rev.read_correspondences(2)[0].shape == (1, 3)                 # own write is visible
    assert s.read_correspondences(3)[0].shape == (2, 3)                   # the real store is untouched
    rev.write_linkage(1, np.array([-1]), np.array([0]), np.zeros((1, 3)), name="ptv_is")
    assert rev.has_linkage(1) and not s.has_linkage(1, "ptv_is")


def test_merge_adds_backward_links_only_where_forward_left_both_ends_free(tmp_path):
    s = _store(tmp_path)
    pos = [s.read_correspondences(f)[0] for f in range(1, 5)]
    # forward found 1->2 and 3->4 for both particles, missed 2->3
    for f in range(1, 5):
        nxt = np.array([0, 1]) if f in (1, 3) else np.array([-2, -2])
        prv = np.array([0, 1]) if f in (2, 4) else np.array([-1, -1])
        s.write_linkage(f, prv, nxt, pos[f - 1], name="ptv_is")
    assert len(_links_of(s, 1, 4, "ptv_is")) == 4

    def backward(rev):
        # in reversed numbering the pass links every consecutive pair: reversed frame g -> g+1
        for g in range(1, 5):
            nxt = np.array([0, 1]) if g < 4 else np.array([-2, -2])
            rev.write_linkage(g, np.array([-1, -1]), nxt, np.zeros((2, 3)), name="ptv_is")

    stats = merge_reversed_pass(s, 1, 4, backward)
    assert stats["forward"] == 4 and stats["backward"] == 6 and stats["merged"] == 6   # the missing 2->3 added
    links = _links_of(s, 1, 4, "ptv_is")
    assert (1, 0, 2, 0) in links and (1, 1, 2, 1) in links                  # frame 2 -> 3, both particles
    assert len(links) == 6
