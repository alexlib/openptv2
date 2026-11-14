import numpy as np

from openptv2.layered_tracking import assign_layers, peel_layers, quality_scores


def test_quality_scores():
    cam = np.array([[0, 1, 2], [0, -1, -1]], dtype=np.int32)
    q = quality_scores(cam)
    assert list(q) == [3.0, 1.0]


def test_assign_layers_basic():
    scores = np.array([5, 4, 3, 2, 1], dtype=float)
    lyr = assign_layers(scores, n_layers=2, quantiles=[0.4, 1.0])
    # top 2 (scores 5,4) -> layer 0; rest -> layer 1
    assert lyr[0] == 0 and lyr[1] == 0
    assert all(lyr[2:] == 1)


def test_peel_layers():
    class _Store:
        def __init__(self):
            self._c = {}

        def has_correspondences(self, f):
            return f in self._c

        def read_correspondences(self, f):
            return self._c[f]

    store = _Store()
    pos = np.arange(15, dtype=float).reshape(5, 3)
    cam = np.array([[0, 1, 2]] * 5, dtype=np.int32)
    store._c[1] = (pos, cam)
    layers = peel_layers(store, [1], n_layers=3, quantiles=[0.34, 0.67, 1.0])
    assert len(layers) == 3
    total = sum(len(layers[i][1][0]) for i in range(3))
    assert total == 5