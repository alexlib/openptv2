"""Re-correspondence of stored detections, the linkage reader and track_store,
on the committed test_cavity store (copied, never written)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.correspondences import match_frame_correspondences
from openptv2.plugins.two_phase_tracking import track_store
from openptv2.recorrespond import (
    correspond_frame,
    load_parameters,
    stored_frames,
    write_store,
)
from openptv2.storage import RunStore
from openptv2.storage.linkage_arrays import read_linkage_arrays

CAVITY = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"

pytestmark = pytest.mark.skipif(
    not (CAVITY / "run.zarr").exists(), reason="test_cavity run.zarr fixture missing"
)


@pytest.fixture(scope="module")
def params():
    return load_parameters(CAVITY / "parameters_Run1.yaml")


@pytest.fixture
def src(tmp_path):
    shutil.copytree(CAVITY / "run.zarr", tmp_path / "run.zarr")
    return RunStore(tmp_path / "run.zarr", mode="r")


def test_load_parameters_reads_the_yaml_from_any_folder(params):
    cpar, vpar, cals, parameters = params
    assert len(cals) == 4 == parameters["num_cams"]
    assert (cpar.imx, cpar.pix_x) == (parameters["ptv"]["imx"], parameters["ptv"]["pix_x"])
    assert vpar.eps0 == parameters["criteria"]["eps0"]


def test_stored_frames_need_targets_in_every_camera(src):
    frames = stored_frames(src, 4)
    assert frames and frames == sorted(frames)
    assert all(src.has_targets(c, f) for f in frames for c in range(4))
    assert stored_frames(src, 4, frames[1], frames[1]) == [frames[1]]


def test_correspond_frame_is_the_pipeline_matcher(src, params):
    cpar, vpar, cals, _ = params
    f = stored_frames(src, 4)[0]
    pos, ids = correspond_frame(src, f, cpar, cals, vpar)
    det = [src.read_targets(c, f) for c in range(4)]
    ref_pos, ref_ids = match_frame_correspondences(det, cpar, cals, vpar)
    assert np.array_equal(ids, ref_ids) and np.allclose(pos, ref_pos)
    four, _ = correspond_frame(src, f, cpar, cals, vpar, min_cams=4)
    assert 0 < len(four) < len(pos) and len(four) == int(((ids >= 0).sum(1) == 4).sum())


def test_write_store_then_track_store(src, params, tmp_path):
    cpar, vpar, cals, parameters = params
    frames = stored_frames(src, 4)
    results = {f: correspond_frame(src, f, cpar, cals, vpar) for f in frames}
    dst = RunStore(tmp_path / "new.zarr", mode="w")
    write_store(src, dst, results)
    for f in frames:
        pos, ids = dst.read_correspondences(f)
        assert np.array_equal(ids, results[f][1]) and np.allclose(pos, results[f][0])
        assert np.array_equal(dst.root[f"targets/cam_0/frame_{f:06d}"],
                              src.root[f"targets/cam_0/frame_{f:06d}"])

    track_cfg = dict(parameters.get("track", {}), v_max=5.0)
    links = track_store(dst, frames, track_cfg, cals, cpar)
    assert links
    trajid, time, pos = read_linkage_arrays(tmp_path / "new.zarr")
    assert sorted(set(time.tolist())) == frames
    assert len(trajid) == sum(len(results[f][0]) for f in frames)
    assert np.allclose(pos * 1000.0, np.vstack([results[f][0] for f in frames]))
    # every frame-to-frame link joins two points of one trajectory
    # (links across a gap, max_gap, are not on the frame-to-frame path)
    consecutive = sum(1 for t0, _, t1, _ in links if t1 == t0 + 1)
    assert len(trajid) - len(np.unique(trajid)) == consecutive
    # the same store and settings track the same way
    again = RunStore(tmp_path / "again.zarr", mode="w")
    write_store(src, again, results)
    assert track_store(again, frames, track_cfg, cals, cpar) == links


def test_read_linkage_arrays_follows_reciprocal_links(tmp_path):
    st = RunStore(tmp_path / "l.zarr", mode="w")
    pos = [np.array([[0.0, 0, 0], [10, 0, 0]]), np.array([[1.0, 0, 0], [11, 0, 0]]),
           np.array([[2.0, 0, 0], [50, 0, 0]])]
    # frame 0 -> 1: both linked; frame 1 -> 2: only the first; point 1 of
    # frame 2 claims a prev that does not point back, so it starts anew
    link = [(np.array([-1, -1]), np.array([0, 1])),
            (np.array([0, 1]), np.array([0, -1])),
            (np.array([0, 1]), np.array([-1, -1]))]
    st.write_linkage_many([5, 6, 7], link, pos)
    trajid, time, p = read_linkage_arrays(tmp_path / "l.zarr")
    assert time.tolist() == [5, 5, 6, 6, 7, 7]
    assert trajid[0] == trajid[2] == trajid[4]
    assert trajid[1] == trajid[3] != trajid[5]
    assert np.allclose(p[:, 0], [0.0, 0.01, 0.001, 0.011, 0.002, 0.05])  # mm -> m
    sub = read_linkage_arrays(tmp_path / "l.zarr", first=6, last=6)
    assert sub[1].tolist() == [6, 6]
    RunStore(tmp_path / "empty.zarr", mode="w")
    assert read_linkage_arrays(tmp_path / "empty.zarr") is None
