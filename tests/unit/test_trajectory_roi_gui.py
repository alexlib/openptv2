"""load_trajectory_positions must return the same units filter_by_trajectory_roi
filters against (mm), and must work before tracking has run -- it prefers
per-frame correspondences (already mm, available right after triangulation)
over the sealed trajectories/ cache (metres, and only exists after tracking
+ seal()), falling back to that cache -- scaled to mm -- only when no
correspondences exist yet either."""

from unittest.mock import Mock

import numpy as np

from openptv2.gui import trajectory_roi_gui as m
from openptv2.storage import RunStore, find_existing_store

CAVITY_DIR = "test_data/test_cavity"


def test_real_store_uses_correspondences_directly_in_mm():
    """test_cavity's store has correspondences but wasn't necessarily just
    sealed -- this is the path a user hits drawing a ROI before tracking."""
    store_path = find_existing_store(CAVITY_DIR)
    store = RunStore(store_path, mode="r")
    frames = store.frames(source="correspondences")
    assert frames, "sanity: test_cavity should have correspondence frames"
    expected = np.concatenate(
        [store.read_correspondences(f)[0] for f in frames], axis=0
    )

    positions = m.load_trajectory_positions(CAVITY_DIR)

    assert np.array_equal(positions, expected)


def test_falls_back_to_ptv_is_linkage_when_no_correspondences(monkeypatch, tmp_path):
    """Some runs (splitter/zarr_only paths) have tracking output but an
    empty correspondences group -- this is the traceback that motivated the
    fallback: 'trajectories/ is stale or absent' even after a real tracking
    run finished with real links, because frames(source='correspondences')
    came back empty for that run."""
    fake_store = Mock()
    fake_store.frames.side_effect = lambda source: {
        "correspondences": [],
        "linkage/ptv_is": [1, 2],
    }[source]
    fake_store.read_linkage.side_effect = [
        (None, None, np.array([[1.0, 2.0, 3.0]])),
        (None, None, np.array([[4.0, 5.0, 6.0]])),
    ]
    monkeypatch.setattr(m, "find_existing_store", lambda wf: tmp_path)
    monkeypatch.setattr(m, "RunStore", lambda path, mode: fake_store)

    positions = m.load_trajectory_positions(tmp_path)

    fake_store.read_linkage.assert_any_call(1, "ptv_is")
    np.testing.assert_allclose(positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_falls_back_to_scaled_trajectories_when_nothing_else_exists(
    monkeypatch, tmp_path
):
    fake_store = Mock()
    fake_store.frames.return_value = []
    fake_store.trajectories.return_value = {
        "pos": np.array([[0.001, 0.002, 0.003]])
    }
    monkeypatch.setattr(m, "find_existing_store", lambda wf: tmp_path)
    monkeypatch.setattr(m, "RunStore", lambda path, mode: fake_store)

    positions = m.load_trajectory_positions(tmp_path)

    fake_store.frames.assert_any_call(source="correspondences")
    fake_store.frames.assert_any_call(source="linkage/ptv_is")
    np.testing.assert_allclose(positions, [[1.0, 2.0, 3.0]])
