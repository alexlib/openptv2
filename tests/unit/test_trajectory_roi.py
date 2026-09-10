import numpy as np

from openptv2.trajectory_roi import TrajectoryROI


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
