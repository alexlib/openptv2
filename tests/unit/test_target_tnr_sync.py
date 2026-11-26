"""Re-matching must leave the stored targets pointing at their new 3D points (the legacy tracker follows those pointers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.correspondences import (
    match_correspondences_batch_parallel,
    sync_target_tnr,
)
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore
from openptv2.synthetic_rig import TruthRig

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"


def _targets_store(path, cals, cpar, n=120):
    rng = np.random.default_rng(0)
    rig = TruthRig.from_cals(cals, cpar)
    pts = rng.uniform(-3.0, 3.0, (n * 4, 3))
    px = rig.project(pts)
    ok = np.isfinite(px).all(axis=(1, 2))
    px = px[ok][:n]
    st = RunStore(str(path), mode="a")
    for c in range(4):
        tg = np.zeros((len(px), 8))
        tg[:, 0] = np.arange(len(px))
        tg[:, 1:3] = px[:, c]
        tg[:, 3] = 9
        tg[:, 7] = 5.0                      # stale pointers of some earlier matching
        st.write_targets(c, 1, tg)
    return st, len(px)


def test_sync_points_every_target_at_its_point_and_unused_targets_at_minus_one(tmp_path):
    cpar, _vpar, cals, _ = load_parameters(YAML)
    st, n = _targets_store(tmp_path / "a.zarr", cals, cpar)
    ids = np.full((3, 4), -1, np.int64)
    ids[0] = [10, 11, 12, 13]
    ids[1] = [20, 21, -1, 23]
    ids[2] = [30, -1, -1, -1]
    sync_target_tnr(st, 1, ids, 4)
    t0 = np.asarray(st.root["targets/cam_0/frame_000001"])[:, 7]
    t2 = np.asarray(st.root["targets/cam_2/frame_000001"])[:, 7]
    assert (t0[[10, 20, 30]] == [0, 1, 2]).all() and (t2[[12]] == [0]).all()
    assert (t0 == -1).sum() == n - 3 and (t2 == -1).sum() == n - 1


def test_batch_matching_writes_consistent_pointers(tmp_path):
    cpar, vpar, cals, _ = load_parameters(YAML)
    st, n = _targets_store(tmp_path / "b.zarr", cals, cpar)
    res = match_correspondences_batch_parallel(
        frames=[1], cpar=cpar, cals=list(cals), vpar=vpar, zarr_store_path=str(tmp_path / "b.zarr"), n_workers=1, write_to_store=True
    )
    pos, ids = RunStore(str(tmp_path / "b.zarr"), mode="r").read_correspondences(1)
    ids = np.asarray(ids)[:, :4]
    assert len(pos) > 20 and 1 in res
    for c in range(4):
        t = np.asarray(RunStore(str(tmp_path / "b.zarr"), mode="r").root[f"targets/cam_{c}/frame_000001"])[:, 7]
        rows = np.flatnonzero(ids[:, c] >= 0)
        assert (t[ids[rows, c]] == rows).all()
        assert (t == -1).sum() == len(t) - len(rows)
