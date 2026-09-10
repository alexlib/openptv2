import os
import time

import numpy as np

from openptv2.trajectory_roi import (
    ROI_FILENAME,
    TrajectoryROI,
    filter_by_trajectory_roi,
    load_trajectory_roi,
)


def test_empty_roi_contains_everything():
    roi = TrajectoryROI()
    assert roi.is_empty()
    pts = np.array([[0, 0, 0], [1e6, 1e6, 1e6]], dtype=float)
    assert roi.contains(pts).all()


def test_single_plane_restricts_only_that_projection():
    roi = TrajectoryROI(xy_polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float))
    pts = np.array([[1, 1, 999], [10, 10, -999]], dtype=float)
    assert roi.contains(pts).tolist() == [True, False]


def test_planes_combine_with_and():
    roi = TrajectoryROI(
        xy_polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float),
        xz_polygon=np.array([[0, 0], [4, 0], [4, 1], [0, 1]], dtype=float),
    )
    inside_both = np.array([1, 1, 0.5])
    outside_xz = np.array([1, 1, 5.0])
    assert roi.contains(inside_both)[0]
    assert not roi.contains(outside_xz)[0]


def test_json_roundtrip(tmp_path):
    roi = TrajectoryROI(
        xy_polygon=np.array([[0, 0], [1, 0], [1, 1]], dtype=float),
        yz_polygon=np.array([[2, 2], [3, 2], [3, 3]], dtype=float),
    )
    path = tmp_path / "roi.json"
    roi.to_json(path)

    loaded = TrajectoryROI.from_json(path)
    assert np.array_equal(loaded.xy_polygon, roi.xy_polygon)
    assert np.array_equal(loaded.yz_polygon, roi.yz_polygon)
    assert loaded.xz_polygon is None


def test_missing_roi_file_means_default_unrestricted(tmp_path):
    """No trajectory_roi.json in the working folder -> nothing is filtered."""
    assert not (tmp_path / ROI_FILENAME).exists()
    roi = load_trajectory_roi(tmp_path)
    assert roi.is_empty()

    pos = np.array([[0, 0, 0], [1e6, 1e6, 1e6]], dtype=float)
    corresp = np.array([[1, 2], [3, 4]])
    kept_pos, kept_corresp = filter_by_trajectory_roi(pos, corresp, tmp_path)
    assert np.array_equal(kept_pos, pos)
    assert np.array_equal(kept_corresp, corresp)


def test_filter_drops_rows_and_matching_columns(tmp_path):
    roi = TrajectoryROI(xy_polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float))
    roi.to_json(tmp_path / ROI_FILENAME)

    pos = np.array([[1, 1, 0], [10, 10, 0], [2, 2, 0]], dtype=float)
    corresp = np.array([[0, 1, 2], [10, 11, 12]])  # (C, N) -- one column per row of pos

    kept_pos, kept_corresp = filter_by_trajectory_roi(pos, corresp, tmp_path)

    assert kept_pos.shape == (2, 3)
    assert np.array_equal(kept_pos, pos[[0, 2]])
    assert kept_corresp.shape == (2, 2)
    assert np.array_equal(kept_corresp, corresp[:, [0, 2]])


def test_empty_pos_is_a_noop(tmp_path):
    roi = TrajectoryROI(xy_polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float))
    roi.to_json(tmp_path / ROI_FILENAME)

    pos = np.zeros((0, 3))
    corresp = np.zeros((2, 0))
    kept_pos, kept_corresp = filter_by_trajectory_roi(pos, corresp, tmp_path)
    assert kept_pos.shape == (0, 3)
    assert kept_corresp.shape == (2, 0)


def test_edited_roi_invalidates_the_cache(tmp_path):
    roi_path = tmp_path / ROI_FILENAME
    TrajectoryROI(
        xy_polygon=np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
    ).to_json(roi_path)

    pos = np.array([[10, 10, 0]], dtype=float)
    assert load_trajectory_roi(tmp_path).contains(pos)[0] == False  # noqa: E712

    # Widen the polygon to include the point, then bump mtime forward --
    # to_json() itself sets mtime to "now", so the bump must come after.
    TrajectoryROI(
        xy_polygon=np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=float)
    ).to_json(roi_path)
    os.utime(roi_path, (time.time() + 1, time.time() + 1))

    assert load_trajectory_roi(tmp_path).contains(pos)[0] == True  # noqa: E712
