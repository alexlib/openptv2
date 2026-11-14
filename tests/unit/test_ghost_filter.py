import numpy as np

from openptv2.ghost_filter import camera_support, filter_low_support


class _Root(dict):
    def create_group(self, name):
        self[name] = {}
        return self[name]


def test_camera_support():
    ids = np.array([[0, 1, 2], [0, -1, -1], [-1, -1, -1]], dtype=np.int32)
    sup = camera_support(ids)
    assert list(sup) == [3, 1, 0]


class _Store:
    def __init__(self):
        self.root = _Root()
        self._corr = {}

    def has_correspondences(self, f):
        return f in self._corr

    def read_correspondences(self, f):
        return self._corr[f]

    def write_correspondences(self, f, pos, cam_ids):
        self._corr[f] = (np.asarray(pos), np.asarray(cam_ids, dtype=np.int32))


def test_filter_low_support():
    store = _Store()
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    cam = np.array([[0, 1, 2], [0, -1, -1], [0, 1, 2]], dtype=np.int32)
    store._corr[1] = (pos, cam)
    dropped = filter_low_support(store, [1], min_cameras=3)
    assert dropped[1] == 1
    p2, c2 = store.read_correspondences(1)
    assert len(p2) == 2
    # backup created
    assert "correspondences_all" in store.root