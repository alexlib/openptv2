"""Tests for frame_quality module."""

import numpy as np

from openptv2.frame_quality import frame_quality_scores, low_quality_frames


def test_frame_quality_scores_basic():
    class FakeStore:
        def read_correspondences(self, f):
            return (np.zeros((f * 10 + 1, 3)), np.zeros((f * 10 + 1, 2), dtype=int))
    s = FakeStore()
    scores = frame_quality_scores(s, 1, 3)
    assert set(scores.keys()) == {1, 2, 3}
    assert all(0 <= v <= 1 for v in scores.values())


def test_low_quality_frames():
    class FakeStore:
        def read_correspondences(self, f):
            n = 100 if f != 2 else 10
            return (np.zeros((n, 3)), np.zeros((n, 2), dtype=int))
    s = FakeStore()
    lq = low_quality_frames(s, 1, 3, threshold=0.5)
    assert 2 in lq
