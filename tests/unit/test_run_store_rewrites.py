"""Tracking re-saves every frame's targets and points. Identical re-writes are skipped, and a write over an
array of the same shape happens in place; the stored content must be right either way."""

import numpy as np

from openptv2.storage import RunStore


def _targets(n, fill):
    t = np.zeros((n, 8))
    t[:, 0] = np.arange(n)
    t[:, 1:3] = fill
    t[:, 7] = -1
    return t


def test_targets_rewrite_same_shape_different_content_and_new_shape(tmp_path):
    s = RunStore(tmp_path / "run.zarr", mode="w")
    s.write_targets(0, 1, _targets(5, 1.0))
    s.write_targets(0, 1, _targets(5, 2.0))                      # in place
    assert np.all(np.asarray(s.root["targets/cam_0/frame_000001"])[:, 1] == 2.0)
    s.write_targets(0, 1, _targets(7, 3.0))                      # shape changes: recreated
    assert np.asarray(s.root["targets/cam_0/frame_000001"]).shape == (7, 8)
    s.write_targets(0, 2, _targets(0, 0.0))                      # empty frame
    assert np.asarray(s.root["targets/cam_0/frame_000002"]).shape == (0, 8)


def test_identical_rewrite_is_skipped_but_a_changed_one_is_not(tmp_path, monkeypatch):
    s = RunStore(tmp_path / "run.zarr", mode="w")
    pos = np.arange(12.0).reshape(4, 3)
    ids = np.arange(8, dtype=np.int32).reshape(4, 2)
    s.write_correspondences(1, pos, ids)
    calls = []
    import openptv2.storage.run_store as rs

    real = rs._put_array
    monkeypatch.setattr(rs, "_put_array", lambda g, n, d: (calls.append(n), real(g, n, d))[1])
    s.write_correspondences(1, pos, ids)                         # identical: no write
    assert calls == []
    s.write_correspondences(1, pos + 1.0, ids)                   # changed: written
    assert calls == ["frame_000001"]
    got_pos, got_ids = s.read_correspondences(1)
    assert np.allclose(got_pos, pos + 1.0) and np.array_equal(got_ids, ids)
    # read, then write the same content back (what the tracker does every frame): skipped
    calls.clear()
    s.write_correspondences(1, got_pos, got_ids)
    assert calls == []


def test_linkage_rewrite_in_place_keeps_the_right_values(tmp_path):
    s = RunStore(tmp_path / "run.zarr", mode="w")
    pos = np.zeros((3, 3))
    s.write_linkage(1, np.array([-1, -1, -1]), np.array([0, 1, 2]), pos, name="ptv_is")
    s.write_linkage(1, np.array([-1, 0, -1]), np.array([-2, 1, 2]), pos + 1, name="ptv_is")
    prev, nxt, got = s.read_linkage(1, name="ptv_is")[:3]
    assert list(prev) == [-1, 0, -1] and list(nxt) == [-2, 1, 2] and np.allclose(got, 1.0)


def test_deferred_linkage_is_readable_inside_and_written_once_at_the_end(tmp_path):
    s = RunStore(tmp_path / "run.zarr", mode="w")
    pos = np.zeros((2, 3))
    with s.deferred_linkage():
        s.write_linkage(1, np.array([-1, -1]), np.array([0, 1]), pos, name="ptv_is", prio=np.array([3, 4]))
        s.write_linkage(2, np.array([0, 1]), np.array([-2, -2]), pos + 1, name="ptv_is", prio=np.array([5, 6]))
        s.write_linkage(1, np.array([-1, -1]), np.array([1, 0]), pos, name="ptv_is", prio=np.array([3, 4]))  # rewrite
        assert s.has_linkage(2, "ptv_is") and s.linkage_frames("ptv_is") == [1, 2]
        assert list(s.read_linkage(1, "ptv_is")[1]) == [1, 0]               # the newest held version
        assert "linkage/ptv_is" not in s.root                               # nothing on disk yet
    assert s.linkage_frames("ptv_is") == [1, 2]
    prev, nxt, got = s.read_linkage(2, "ptv_is")
    assert list(prev) == [0, 1] and list(nxt) == [-2, -2] and np.allclose(got, 1.0)
    assert list(s.read_prio(2, "ptv_is")) == [5, 6]
    assert list(s.read_linkage(1, "ptv_is")[1]) == [1, 0]


def test_deferred_linkage_flushes_when_the_pass_fails_and_nests(tmp_path):
    s = RunStore(tmp_path / "run.zarr", mode="w")
    pos = np.zeros((1, 3))
    try:
        with s.deferred_linkage():
            with s.deferred_linkage():                                       # joins the outer one
                s.write_linkage(1, np.array([-1]), np.array([-2]), pos, name="ptv_is")
            assert "linkage/ptv_is" not in s.root                            # inner exit did not flush
            raise RuntimeError("pass failed")
    except RuntimeError:
        pass
    assert s.has_linkage(1, "ptv_is")
