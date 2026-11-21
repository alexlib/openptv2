"""Tests for the unified Zarr RunStore (openptv2.storage.{run_store,seal})."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from openptv2.storage import (
    RunStore,
    RunStoreError,
    needs_reseal,
    seal,
)
from openptv2.storage.run_store import resolve_store_path
from openptv2.tracking_framebuf import TargetArray

pytestmark = pytest.mark.ci


@pytest.fixture
def cavity_store(tmp_path):
    """A fresh, unsealed 4-frame, 4-camera RunStore with targets,
    correspondences, linkage ('ptv_is'), and priority ('added')."""
    store = RunStore(tmp_path / "run" / "res" / "run.zarr", mode="w")
    frames = [10001, 10002, 10003, 10004]
    n_particles = 5
    for fi, fnum in enumerate(frames):
        for cam in range(4):
            tarr = TargetArray(n_particles)
            for i in range(n_particles):
                t = tarr[i]
                t.set_pnr(i)
                t.set_pos((100.0 + i * 10.0 + fi, 200.0 + i * 5.0))
                t.set_pixel_counts(25, 5, 5)
                t.set_sum_grey_value(500)
                t.set_tnr(i)
            store.write_targets(cam, fnum, tarr)

        pos = np.array(
            [[10.0 * i + fi * 1.0, 5.0 * i, 0.0] for i in range(n_particles)],
            dtype=np.float64,
        )
        cam_ids = np.tile(np.arange(n_particles, dtype=np.int32)[:, None], (1, 4))
        # Make one triplet and one pair so stats partition has variety
        cam_ids[1, 3] = -1
        cam_ids[2, 2:] = -1
        store.write_correspondences(fnum, pos, cam_ids)

        prev = np.full(n_particles, -1, dtype=np.int32)
        nxt = np.full(n_particles, -2, dtype=np.int32)
        if fi > 0:
            prev[:4] = np.arange(4, dtype=np.int32)
        if fi < len(frames) - 1:
            nxt[:4] = np.arange(4, dtype=np.int32)
        store.write_linkage(fnum, prev, nxt, pos, name="ptv_is")
        store.write_linkage(
            fnum,
            prev,
            nxt,
            pos,
            name="added",
            prio=np.full(n_particles, 4, dtype=np.int32),
        )

    return store


def test_resolve_store_path_variants(tmp_path):
    assert resolve_store_path(tmp_path) == tmp_path / "res" / "run.zarr"
    assert resolve_store_path(tmp_path / "res") == tmp_path / "res" / "run.zarr"
    assert (
        resolve_store_path(tmp_path / "res" / "run.zarr")
        == tmp_path / "res" / "run.zarr"
    )


def test_store_discovers_cams_and_frames(cavity_store):
    assert cavity_store.target_cameras() == [0, 1, 2, 3]
    assert cavity_store.frames() == [10001, 10002, 10003, 10004]


def test_added_stream_is_prio_not_second_linkage(cavity_store):
    """'added' is the tracker's prio output: integer priority per particle."""
    prio = cavity_store.read_prio(10001, "added")
    assert prio is not None
    assert prio.shape[0] > 0
    assert np.issubdtype(prio.dtype, np.integer)

    # plain linkage never carries a prio column
    assert cavity_store.read_prio(10001, "ptv_is") is None


def test_seal_builds_trajectory_index(cavity_store):
    summary = seal(cavity_store)
    assert summary["n_trajectories"] > 0
    assert summary["n_rows"] >= summary["n_trajectories"]

    idx = cavity_store.traj_index()
    assert len(idx["trajid"]) == summary["n_trajectories"]
    assert (idx["last"] >= idx["first"]).all()
    assert (idx["length"] >= 1).all()

    traj = cavity_store.trajectory(int(idx["trajid"][0]))
    assert traj["pos"].shape == (int(idx["length"][0]), 3)
    # trajectories/ is in metres; source linkage data is in mm.
    assert np.abs(traj["pos"]).max() < 1.0


def test_seal_is_idempotent(cavity_store):
    first = seal(cavity_store)
    assert first.get("skipped") is not True

    second = seal(cavity_store)
    assert second["skipped"] is True
    assert needs_reseal(cavity_store) is False


def test_reseal_after_linkage_mutation(cavity_store):
    seal(cavity_store)
    assert needs_reseal(cavity_store) is False

    prev, nxt, pos = cavity_store.read_linkage(10001, "ptv_is")
    nxt = nxt.copy()
    nxt[0] = -2  # sever one link
    cavity_store.write_linkage(10001, prev, nxt, pos, name="ptv_is")

    assert needs_reseal(cavity_store) is True
    reseal_summary = seal(cavity_store)
    assert reseal_summary.get("skipped") is not True
    assert needs_reseal(cavity_store) is False


def test_trajectories_group_requires_seal(cavity_store):
    with pytest.raises(RunStoreError):
        cavity_store.trajectories()


def test_stats_partition_matches_correspondence_count(cavity_store):
    for row in cavity_store.stats():
        assert row["n_quads"] + row["n_trips"] + row["n_pairs"] == row["n_corres"]
        assert row["n_targets"].shape[0] == len(cavity_store.target_cameras())
        assert (row["cam_seen"] <= row["n_targets"]).all()


def test_traj_index_matches_zarr_frame_store_after_singleton_filter(
    cavity_store, tmp_path
):
    """traj/ is a superset of what read_zarr_trajectories returns:
    it labels every particle including length-1 (unlinked) singletons.
    Filtering traj/ by length >= 2 must reproduce read_zarr_trajectories."""
    from openptv2.storage.zarr_store import ZarrFrameStore, read_zarr_trajectories

    seal(cavity_store)
    idx = cavity_store.traj_index()

    legacy_path = tmp_path / "reference.zarr"
    legacy_store = ZarrFrameStore(legacy_path, mode="w")
    for frame in cavity_store.frames():
        prev, nxt, pos = cavity_store.read_linkage(frame, "ptv_is")
        legacy_store.write_linkage(frame, prev, nxt, pos, linkage_name="ptv_is")

    legacy_trajs = read_zarr_trajectories(legacy_path)
    assert legacy_trajs, "pipeline run must produce tracked linkage"

    mine_lengths = sorted(int(x) for x in idx["length"] if x >= 2)
    legacy_lengths = sorted(len(t) for t in legacy_trajs)
    assert mine_lengths == legacy_lengths
    assert sum(mine_lengths) == sum(legacy_lengths)


def test_to_flowtracks_trajectories_matches_zarr_reader(cavity_store, tmp_path):
    """RunStore.to_flowtracks_trajectories must return the same trajectories
    as read_zarr_trajectories, and must seal on demand."""
    from openptv2.storage.zarr_store import read_zarr_trajectories

    assert not cavity_store.sealed

    result = cavity_store.to_flowtracks_trajectories()
    assert cavity_store.sealed  # sealed itself as a side effect

    legacy = read_zarr_trajectories(tmp_path / "run" / "res" / "run.zarr")
    assert sorted(len(t) for t in result) == sorted(len(t) for t in legacy)
    assert sum(len(t) for t in result) == sum(len(t) for t in legacy)
    # metres, not mm
    assert max(abs(t.pos()).max() for t in result) < 1.0


def test_to_flowtracks_trajectories_frame_range_filter(cavity_store):
    all_trajs = cavity_store.to_flowtracks_trajectories()
    narrow = cavity_store.to_flowtracks_trajectories(
        first=10001, last=10002, traj_min_len=2
    )
    assert len(narrow) <= len(all_trajs)
    for t in narrow:
        assert t.time().max() <= 10002


def test_write_targets_rejects_bad_input(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    with pytest.raises(RunStoreError):
        store.write_targets(0, 1, [object()])


def test_linkage_row_count_mismatch_raises(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    with pytest.raises(RunStoreError):
        store.write_linkage(
            1, prev_ids=[0, 1], next_ids=[0], pos_3d=np.zeros((2, 3)), name="ptv_is"
        )


def test_read_correspondences_handles_flat_zero_particle_array(tmp_path):
    """Regression test: a zero-particle frame can be stored as a flat (0,)
    array rather than the usual (0, 3+C) shape -- observed on real data with
    a particle-count ramp-up at the sequence start (a caller elsewhere wrote
    np.empty(0) instead of np.empty((0, 3 + num_cams))). read_correspondences
    used to crash with "too many indices for array" on this shape instead of
    reporting zero particles."""
    from openptv2.storage.run_store import _frame_key

    store = RunStore(tmp_path / "run.zarr", mode="w")
    store.root["correspondences"].create_array(_frame_key(1), data=np.empty(0))

    pos, cam_ids = store.read_correspondences(1)
    assert pos.shape == (0, 3)
    assert cam_ids.shape[0] == 0


def test_clear_linkage_removes_all_frames(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    store.write_linkage(
        1, prev_ids=[-1], next_ids=[-2], pos_3d=np.zeros((1, 3)), name="ptv_is"
    )
    store.write_linkage(
        2, prev_ids=[-1], next_ids=[-2], pos_3d=np.zeros((1, 3)), name="ptv_is"
    )
    assert store.has_linkage(1, "ptv_is")
    assert store.has_linkage(2, "ptv_is")

    store.clear_linkage("ptv_is")

    assert not store.has_linkage(1, "ptv_is")
    assert not store.has_linkage(2, "ptv_is")


def test_clear_linkage_noop_when_absent(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    store.clear_linkage("nonexistent")  # must not raise


def test_seal_carries_trajid_across_a_bridged_gap(tmp_path):
    """relink_trajectory_gaps writes a cross-frame link (next pointing past
    the skipped frame). Seal must recognise it as a continuation instead of
    starting a fresh trajectory id at every bridge."""
    store = RunStore(tmp_path / "run.zarr", mode="w")
    # frame 0 -> frame 1 (step 1), frame 1 -> frame 3 (step 2, bridged gap)
    store.write_linkage(
        0,
        np.array([-1], np.int32),
        np.array([0], np.int32),
        np.array([[0.0, 0.0, 0.0]]),
    )
    store.write_linkage(
        1, np.array([0], np.int32), np.array([0], np.int32), np.array([[2.0, 0.0, 0.0]])
    )
    store.write_linkage(
        2, np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 3))
    )
    store.write_linkage(
        3,
        np.array([0], np.int32),
        np.array([-2], np.int32),
        np.array([[6.0, 0.0, 0.0]]),
    )

    summary = seal(store)
    assert summary["n_trajectories"] == 1  # one trajectory, not two
    assert list(store.traj_index()["length"]) == [3]


def _make_seen_store(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    pos = np.array([[0.0, 0, 0], [1.0, 1, 1], [2.0, 2, 2]])
    cids = np.array([[0, 1, -1, 3], [-1, -1, -1, -1], [5, 6, 7, 8]],
                    dtype=np.int32)
    store.write_correspondences(10000, pos, cids)
    return store


def test_read_seen_single_definition(tmp_path):
    """read_seen is THE definition of seen: agrees with cam_ids >= 0."""
    seen = _make_seen_store(tmp_path).read_seen(10000)
    assert seen.dtype == bool
    assert seen.tolist() == [
        [True, True, False, True],
        [False, False, False, False],
        [True, True, True, True],
    ]


def test_read_seen_missing_frame_raises(tmp_path):
    store = _make_seen_store(tmp_path)
    with pytest.raises(RunStoreError):
        store.read_seen(99999)


def test_read_seen_empty_frame(tmp_path):
    store = RunStore(tmp_path / "run.zarr", mode="w")
    store.write_correspondences(10001, np.empty((0, 3)),
                                np.empty((0, 4), dtype=np.int32))
    seen = store.read_seen(10001)
    assert seen.dtype == bool and seen.shape == (0, 4)


def test_point_quality_roundtrip_and_missing(tmp_path):
    from openptv2.storage import RunStore

    s = RunStore(tmp_path / "run.zarr", mode="a")
    assert s.read_point_quality(3) is None
    s.write_point_quality(3, np.array([0.1, 0.9, 0.5]))
    got = s.read_point_quality(3)
    assert got.shape == (3,) and np.allclose(got, [0.1, 0.9, 0.5], atol=1e-6)
    s.write_point_quality(3, np.array([0.2]))  # overwrite
    assert s.read_point_quality(3).shape == (1,)


def test_batched_writes_match_single_frame_writes(tmp_path):
    rng = np.random.default_rng(0)
    frames = [3, 4, 7]
    link = [
        (rng.integers(-1, 9, n), rng.integers(-1, 9, n)) for n in (9, 5, 12)
    ]
    pos = [rng.normal(size=(n, 3)) for n in (9, 5, 12)]
    ghost = [rng.random(n) for n in (9, 5, 12)]
    one = RunStore(tmp_path / "one.zarr", mode="w")
    many = RunStore(tmp_path / "many.zarr", mode="w")
    for f, (p, n), x, g in zip(frames, link, pos, ghost):
        one.write_linkage(f, p, n, x)
        one.write_point_quality(f, g)
    many.write_linkage_many(frames, link, pos)
    many.write_point_quality_many(frames, ghost)
    assert many.frames("linkage/ptv_is") == one.frames("linkage/ptv_is") == frames
    for f in frames:
        for a, b in zip(one.read_linkage(f), many.read_linkage(f)):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        assert np.array_equal(one.read_point_quality(f), many.read_point_quality(f))
    with pytest.raises(RunStoreError):
        many.write_linkage_many([9], [(np.zeros(2), np.zeros(2))], [np.zeros((3, 3))])


def test_store_records_length_and_time_units(tmp_path):
    """Every group states its units: pixels in, mm out, frames for time."""
    store = RunStore(tmp_path / "run.zarr", mode="w")
    meta = dict(store.root["meta"].attrs)
    assert meta["length_unit"] == "mm"
    assert meta["time_unit"] == "frame"
    assert meta["raw_units"] == "mm"
    assert store.root["targets"].attrs["x_units"] == "pixel"
    assert store.root["correspondences"].attrs["pos_units"] == "mm"
    assert store.root["linkage"].attrs["time_units"] == "frame"


def test_sealed_trajectories_mark_vel_unfilled(tmp_path):
    """Seal rescales mm->m but never differentiates: vel/accel are zeros,
    marked unfilled so no reader mistakes them for measured stillness."""
    store = RunStore(tmp_path / "run.zarr", mode="w")
    n = 3
    store.write_trajectories(
        pos=np.zeros((n, 3)),
        vel=np.zeros((n, 3)),
        accel=np.zeros((n, 3)),
        time=np.arange(1, n + 1),
        trajid=np.zeros(n, dtype=np.int64),
    )
    attrs = dict(store.root["trajectories"].attrs)
    assert attrs["pos_units"] == "m"
    assert "unfilled" in attrs["vel_units"]
    assert attrs["time_units"] == "frame"


def test_observations_round_trip(tmp_path):
    store = RunStore(tmp_path / "run" / "res" / "run.zarr", mode="w")
    assert store.read_observations(7) is None
    assert not store.has_observations(7)
    obs = np.zeros((3, 4, 4), dtype=np.float32)
    obs[:, :, 0:2] = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    obs[:, :, 2] = 0.04
    obs[:, :, 3] = 1
    obs[1, 2] = [np.nan, np.nan, np.nan, 0]  # camera not used
    store.write_observations(7, obs)
    assert store.has_observations(7)
    back = store.read_observations(7)
    assert back.shape == (3, 4, 4)
    np.testing.assert_array_equal(np.isnan(back), np.isnan(obs))
    np.testing.assert_allclose(np.nan_to_num(back), np.nan_to_num(obs))
    # overwrite with a different row count
    store.write_observations(7, obs[:2])
    assert store.read_observations(7).shape == (2, 4, 4)


def test_observations_reject_bad_shape(tmp_path):
    store = RunStore(tmp_path / "run" / "res" / "run.zarr", mode="w")
    with pytest.raises(RunStoreError):
        store.write_observations(1, np.zeros((3, 4)))
    with pytest.raises(RunStoreError):
        store.write_observations(1, np.zeros((3, 4, 3)))


def test_point_sigma_round_trip(tmp_path):
    store = RunStore(tmp_path / "run" / "res" / "run.zarr", mode="w")
    assert store.read_point_sigma(3) is None
    store.write_point_sigma(3, np.array([0.01, 0.05, 0.2]))
    np.testing.assert_allclose(store.read_point_sigma(3), [0.01, 0.05, 0.2], rtol=1e-6)
