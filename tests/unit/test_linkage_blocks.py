"""Block layout for tracker output (storage/linkage_blocks.py)."""

import numpy as np
import pytest
import zarr

from openptv2.storage import RunStore
from openptv2.storage.linkage_blocks import (
    BlockSet,
    committed_gen,
    iter_linkage,
    read_all_quality,
)
from openptv2.storage.seal import seal

pytestmark = pytest.mark.unit


def _chain(n_frames=6, n=5, seed=0):
    """Linked points moving +1 mm/frame in x; row i links to row i."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 10, (n, 3))
    frames = list(range(1, n_frames + 1))
    pos = [base + [k, 0, 0] for k in range(n_frames)]
    link = []
    for k in range(n_frames):
        prv = np.arange(n) if k else np.full(n, -1)
        nxt = np.arange(n) if k < n_frames - 1 else np.full(n, -1)
        link.append((prv, nxt))
    return frames, link, pos


def _same(a, b):
    return all(a_.dtype == b_.dtype and np.array_equal(a_, b_) for a_, b_ in zip(a, b))


def test_block_roundtrip_matches_per_frame(tmp_path):
    frames, link, pos = _chain()
    link[2] = (np.zeros(0, np.int32), np.zeros(0, np.int32))  # an empty frame
    pos[2] = np.zeros((0, 3))
    prio = [np.arange(len(p), dtype=np.int32) for p in pos]
    one = RunStore(tmp_path / "one.zarr", mode="w")
    many = RunStore(tmp_path / "many.zarr", mode="w")
    for f, (p, n), x, pr in zip(frames, link, pos, prio):
        one.write_linkage(f, p, n, x, prio=pr)
    many.write_linkage_many(frames, link, pos, prio=prio)
    assert many.linkage_frames() == one.linkage_frames() == frames
    for f in frames:
        assert _same(one.read_linkage(f), many.read_linkage(f))
        assert np.array_equal(one.read_prio(f), many.read_prio(f))
        assert many.has_linkage(f)
    assert not many.has_linkage(99)
    assert many.frames("linkage/ptv_is") == frames


def test_reads_are_copies(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    st.write_linkage_many(frames, link, pos)
    st.read_linkage(2)[1][0] = 77  # postprocess passes mutate what they read
    assert st.read_linkage(2)[1][0] == 0


def test_small_object_count(tmp_path):
    frames, link, pos = _chain(n_frames=300)
    st = RunStore(tmp_path / "s.zarr", mode="w")
    before = sum(1 for _ in (tmp_path / "s.zarr").rglob("*"))
    st.write_linkage_many(frames, link, pos)
    added = sum(1 for _ in (tmp_path / "s.zarr").rglob("*")) - before
    assert added < 100  # per-frame groups would add ~10 entries per frame


def test_newest_generation_wins_and_old_blocks_are_collected(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    st.write_linkage_many(frames, link, pos)
    new_next = np.full(5, -1, np.int32)
    st.write_linkage_many([3, 4], [(link[2][0], new_next)] * 2, pos[2:4])
    assert (st.read_linkage(3)[1] == -1).all()
    assert (st.read_linkage(4)[1] == -1).all()
    assert (st.read_linkage(5)[1] >= 0).all()
    # a full rewrite shadows everything and the old blocks disappear
    st.write_linkage_many(frames, link, pos)
    blocks = [k for k in st.root["linkage/ptv_is"].keys() if k.startswith("block_")]
    assert len(blocks) == 1
    assert (st.read_linkage(3)[1] >= 0).all()


def test_single_frame_write_in_block_mode_goes_to_a_block(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    st.write_linkage_many(frames, link, pos)
    st.write_linkage(2, link[1][0], np.full(5, -1), pos[1])
    assert (st.read_linkage(2)[1] == -1).all()
    assert not any(k.startswith("frame_") for k in st.root["linkage/ptv_is"].keys())


def test_halted_write_is_invisible_and_cleaned_up(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    # halted before the FIRST commit: nothing is visible
    bs = BlockSet(st.root.require_group("linkage").require_group("ptv_is"))
    cols = {"prev": [l[0] for l in link], "next": [l[1] for l in link], "pos": pos}
    bs.write(frames, cols, commit=False)
    st2 = RunStore(tmp_path / "s.zarr", mode="a")
    assert st2.linkage_frames() == [] and not st2.has_linkage(1)
    # a good write after the halt works and leaves no stray blocks
    st2.write_linkage_many(frames, link, pos)
    keys = [k for k in st2.root["linkage/ptv_is"].keys() if k.startswith("block_")]
    assert len(keys) == 1 and committed_gen(st2.root["linkage/ptv_is"]) == 1
    # halted rewrite: the committed result survives untouched
    other = [(np.full(5, -1, np.int32),) * 2] * len(frames)
    BlockSet(st2.root["linkage/ptv_is"]).write(frames, {
        "prev": [o[0] for o in other], "next": [o[1] for o in other], "pos": pos,
    }, commit=False)
    st3 = RunStore(tmp_path / "s.zarr", mode="r")
    assert (st3.read_linkage(2)[1] >= 0).all()


def test_legacy_per_frame_groups_still_read_and_blocks_win(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    for f, (p, n), x in zip(frames, link, pos):
        st.write_linkage(f, p, n, x)  # legacy layout
    assert st.linkage_frames() == frames
    st.write_linkage_many([3, 4], [(link[2][0], np.full(5, -1, np.int32))] * 2, pos[2:4])
    assert st.linkage_frames() == frames
    assert (st.read_linkage(1)[1] >= 0).all()  # per-frame
    assert (st.read_linkage(3)[1] == -1).all()  # block shadows stale per-frame
    got = dict(iter_linkage(st.root["linkage/ptv_is"]))
    assert sorted(got) == frames
    assert (got[3]["next"] == -1).all() and (got[1]["next"] >= 0).all()


def test_clear_linkage_resets_block_mode(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    st.write_linkage_many(frames, link, pos)
    st.clear_linkage()
    assert st.linkage_frames() == [] and not st.has_linkage(1)
    st.write_linkage(1, link[0][0], link[0][1], pos[0])  # per-frame again
    assert any(k.startswith("frame_") for k in st.root["linkage/ptv_is"].keys())


def test_quality_blocks_and_single_write(tmp_path):
    rng = np.random.default_rng(3)
    frames = [1, 2, 3]
    ghost = [rng.random(n).astype(np.float32) for n in (4, 0, 6)]
    st = RunStore(tmp_path / "s.zarr", mode="w")
    st.write_point_quality_many(frames, ghost)
    for f, g in zip(frames, ghost):
        assert np.array_equal(st.read_point_quality(f), g)
    assert st.read_point_quality(9) is None
    st.write_point_quality(2, np.ones(3))  # block mode: shadows with a new block
    assert np.array_equal(st.read_point_quality(2), np.ones(3, np.float32))
    allq = read_all_quality(st.root["quality"])
    assert sorted(allq) == frames and len(allq[3]) == 6


def test_seal_identical_for_both_layouts(tmp_path):
    frames, link, pos = _chain(n_frames=8, n=7, seed=5)
    a = RunStore(tmp_path / "a.zarr", mode="w")
    b = RunStore(tmp_path / "b.zarr", mode="w")
    for f, (p, n), x in zip(frames, link, pos):
        a.write_linkage(f, p, n, x)
    b.write_linkage_many(frames, link, pos)
    seal(a)
    seal(b)
    ta, tb = a.trajectories(), b.trajectories()
    assert ta.keys() == tb.keys()
    for k in ta:
        assert np.array_equal(ta[k], tb[k])
    # trajid is written back next to the rows (in the block here)
    assert b.sealed and "trajid" in b._blocks("linkage/ptv_is")._load(
        b._blocks("linkage/ptv_is").index[1][0]
    )
    seal(b)  # second seal is a no-op at the same source hash


def test_rewrite_over_legacy_store_drops_shadowed_per_frame_entries(tmp_path):
    frames, link, pos = _chain()
    st = RunStore(tmp_path / "s.zarr", mode="w")
    for f, (p, n), x in zip(frames, link, pos):
        st.write_linkage(f, p, n, x)
        st.write_point_quality(f, np.zeros(5))
    st.write_linkage_many(frames, link, pos)
    st.write_point_quality_many(frames, [np.ones(5)] * len(frames))
    assert not any(k.startswith("frame_") for k in st.root["linkage/ptv_is"].keys())
    assert not any(k.startswith("frame_") for k in st.root["quality"].keys())
    assert st.linkage_frames() == frames
    assert np.array_equal(st.read_point_quality(1), np.ones(5, np.float32))
