"""Tests for the Phase A unified run store (openptv2.storage.{run_store,seal,legacy}).

See docs/plans/2026-08-14-storage-formats-as-built.md for the legacy formats
and the plan file this implements (approved plan "Unified Zarr run store for
openptv2 -- Phase A"). These tests are the backward-compatibility gate: a
change to the store must keep import_run -> seal -> export_run byte-identical
to the checked-in ASCII fixtures.
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.storage import RunStore, RunStoreError, export_run, import_run, needs_reseal, seal
from openptv2.storage.run_store import resolve_store_path
from tests._support import find_test_data_root

TEST_DATA_ROOT = find_test_data_root()
CAVITY_DIR = TEST_DATA_ROOT / "test_cavity"


@pytest.fixture
def cavity_ascii_only(tmp_path):
    """A private, freshly-generated run: raw calibration/images copied from
    test_cavity into tmp_path, the real detection/correspondence/tracking
    pipeline run against that copy (store-only output, per
    docs/plans/2026-08-15-zarr-only-transition-plan.md), then ``export_run``
    regenerates the legacy ASCII files from that store -- this is now the
    only way to get ASCII output, on demand, since the pipeline itself no
    longer writes it.

    ``test_data/**/res/`` (and every ``*_targets`` file) is gitignored --
    regenerated locally, not checked in -- and other tests in this suite
    write pipeline output directly into the shared test_cavity directory.
    Depending on suite execution order that shared res/ can be mid-rewrite or
    absent, so this fixture does not read it; it produces its own private,
    deterministic copy via the same ``pyptv_batch.main`` entry point the
    batch tests use."""
    from openptv2.batch import pyptv_batch

    dst = tmp_path / "run"
    shutil.copytree(
        CAVITY_DIR,
        dst,
        ignore=shutil.ignore_patterns("*_targets", "res", "run.zarr", "tmp*.yaml", "tmp*.txt"),
    )
    pyptv_batch.main(dst / "parameters_Run1.yaml", 10001, 10004)
    store = RunStore(resolve_store_path(dst), mode="r")
    export_run(store, dst)
    # Tests reuse this fixture's directory with their own fresh import_run()
    # call; leaving the pipeline's own store behind would make that reopen
    # (not recreate) this store, appending on top of it (e.g. stale prio left
    # on the "ptv_is" linkage group from the live-pipeline write).
    shutil.rmtree(dst / "res" / "run.zarr", ignore_errors=True)
    return dst


def test_resolve_store_path_variants(tmp_path):
    assert resolve_store_path(tmp_path) == tmp_path / "res" / "run.zarr"
    assert resolve_store_path(tmp_path / "res") == tmp_path / "res" / "run.zarr"
    assert resolve_store_path(tmp_path / "res" / "run.zarr") == tmp_path / "res" / "run.zarr"


def test_import_discovers_cams_and_frames(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    assert store.target_cameras() == [0, 1, 2, 3]
    assert store.frames() == [10001, 10002, 10003, 10004]


def test_round_trip_byte_identical(cavity_ascii_only, tmp_path):
    store = import_run(cavity_ascii_only)
    out = tmp_path / "exported"
    export_run(store, out)

    for pattern in ("rt_is.*", "ptv_is.*", "added.*"):
        originals = sorted((cavity_ascii_only / "res").glob(pattern))
        assert originals, f"fixture missing {pattern}"
        for orig in originals:
            exported = out / "res" / orig.name
            assert exported.exists(), f"export missing {orig.name}"
            assert filecmp.cmp(orig, exported, shallow=False), f"{orig.name} differs"

    for orig in sorted((cavity_ascii_only / "img").glob("*_targets")):
        exported = out / "img" / orig.name
        assert exported.exists(), f"export missing {orig.name}"
        assert filecmp.cmp(orig, exported, shallow=False), f"{orig.name} differs"


def test_added_stream_is_prio_not_second_linkage(cavity_ascii_only):
    """res/added.* is the tracker's prio output (default_naming['prio'] =
    'res/added', tracker.py:14-18): 6 columns (prev next x y z prio), not a
    second 5-column linkage pass. Caught by the round-trip byte-diff."""
    store = import_run(cavity_ascii_only)
    prio = store.read_prio(10001, "added")
    assert prio is not None
    assert prio.shape[0] > 0
    assert np.issubdtype(prio.dtype, np.integer)

    # plain linkage never carries a prio column
    assert store.read_prio(10001, "ptv_is") is None


def test_seal_builds_trajectory_index(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    summary = seal(store)
    assert summary["n_trajectories"] > 0
    assert summary["n_rows"] >= summary["n_trajectories"]

    idx = store.traj_index()
    assert len(idx["trajid"]) == summary["n_trajectories"]
    assert (idx["last"] >= idx["first"]).all()
    assert (idx["length"] >= 1).all()

    traj = store.trajectory(int(idx["trajid"][0]))
    assert traj["pos"].shape == (int(idx["length"][0]), 3)
    # trajectories/ is in metres; rt_is/ptv_is source data is in mm.
    assert np.abs(traj["pos"]).max() < 1.0


def test_seal_is_idempotent(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    first = seal(store)
    assert first.get("skipped") is not True

    second = seal(store)
    assert second["skipped"] is True
    assert needs_reseal(store) is False


def test_reseal_after_linkage_mutation(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    seal(store)
    assert needs_reseal(store) is False

    prev, nxt, pos = store.read_linkage(10001, "ptv_is")
    nxt = nxt.copy()
    nxt[0] = -2  # sever one link
    store.write_linkage(10001, prev, nxt, pos, name="ptv_is")

    assert needs_reseal(store) is True
    reseal_summary = seal(store)
    assert reseal_summary.get("skipped") is not True
    assert needs_reseal(store) is False


def test_trajectories_group_requires_seal(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    with pytest.raises(RunStoreError):
        store.trajectories()


def test_stats_partition_matches_correspondence_count(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    for row in store.stats():
        assert row["n_quads"] + row["n_trips"] + row["n_pairs"] == row["n_corres"]
        assert row["n_targets"].shape[0] == len(store.target_cameras())
        assert (row["cam_seen"] <= row["n_targets"]).all()


def test_traj_index_matches_legacy_reader_after_singleton_filter(cavity_ascii_only, tmp_path):
    """traj/ is a superset of what the legacy read_zarr_trajectories returns:
    it labels every particle including length-1 (unlinked) singletons, since
    nothing should be silently dropped from the store. The legacy reader
    (openptv2/storage/zarr_store.py:684, and flowtracks itself) only ever
    returns trajectories of length >= 2. Filtering traj/ the same way must
    reproduce the legacy result exactly.

    Cross-checked against the OLD, trusted ZarrFrameStore/read_zarr_trajectories
    path, fed from the same ptv_is.* ASCII this run produced -- in a store
    path of its own, so it can't collide with the RunStore under test (see
    the OPENPTV_STORAGE note on ``cavity_ascii_only``: the two stores use
    different frame-key conventions and must not share one run.zarr)."""
    from openptv2.storage.legacy import _load_linkage
    from openptv2.storage.zarr_store import ZarrFrameStore, read_zarr_trajectories

    store = import_run(cavity_ascii_only)
    seal(store)
    idx = store.traj_index()

    legacy_path = tmp_path / "legacy_reference.zarr"
    legacy_store = ZarrFrameStore(legacy_path, mode="w")
    for frame in store.frames():
        loaded = _load_linkage(cavity_ascii_only / "res" / f"ptv_is.{frame}")
        assert loaded is not None
        prev, nxt, pos, _prio = loaded
        legacy_store.write_linkage(frame, prev, nxt, pos, linkage_name="ptv_is")

    legacy_trajs = read_zarr_trajectories(legacy_path)
    assert legacy_trajs, "pipeline run must produce tracked linkage"

    mine_lengths = sorted(int(x) for x in idx["length"] if x >= 2)
    legacy_lengths = sorted(len(t) for t in legacy_trajs)
    assert mine_lengths == legacy_lengths
    assert sum(mine_lengths) == sum(legacy_lengths)


def test_to_flowtracks_trajectories_matches_legacy_reader(cavity_ascii_only):
    """RunStore.to_flowtracks_trajectories (Phase D's GUI display read path)
    must return the same trajectories as the legacy read_zarr_trajectories
    linkage-walk, and must seal on demand rather than requiring the caller
    to remember to."""
    from openptv2.storage.zarr_store import read_zarr_trajectories

    store = import_run(cavity_ascii_only)
    assert not store.sealed

    result = store.to_flowtracks_trajectories()
    assert store.sealed  # sealed itself as a side effect

    legacy = read_zarr_trajectories(cavity_ascii_only / "res" / "run.zarr")
    assert sorted(len(t) for t in result) == sorted(len(t) for t in legacy)
    assert sum(len(t) for t in result) == sum(len(t) for t in legacy)
    # metres, not mm
    assert max(abs(t.pos()).max() for t in result) < 1.0


def test_to_flowtracks_trajectories_frame_range_filter(cavity_ascii_only):
    store = import_run(cavity_ascii_only)
    all_trajs = store.to_flowtracks_trajectories()
    narrow = store.to_flowtracks_trajectories(first=10001, last=10002, traj_min_len=2)
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
