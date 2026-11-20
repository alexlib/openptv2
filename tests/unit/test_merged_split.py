import numpy as np
import pytest

from openptv2 import merged_split


def test_split_returns_constituent_count(monkeypatch):
    # two constituents near known predictions
    preds = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    intensities = np.array([1.0, 1.0])
    merged_px = np.array([[5.0, 5.0], [5.0, 5.0]])
    seen = np.array([True, True])
    # identity-ish projection: project(x) = x[:2]*5 (so merged centre of two points)
    def fake_project(pt, cals, cpar):
        return np.tile(np.asarray(pt, float)[:2] * 5.0, (len(cals), 1)).reshape(len(cals), 2)
    monkeypatch.setattr(merged_split, "_project", fake_project)
    pos, flags = merged_split.split_merged_blob(
        merged_px, seen, intensities, preds, cals=[None, None], cpar=None
    )
    assert pos.shape == (2, 3)
    assert flags.shape == (2,)
    assert flags.all()


def test_split_moves_toward_predictions(monkeypatch):
    preds = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    intensities = np.array([1.0, 1.0])
    # merged centre is midpoint projection
    def fake_project(pt, cals, cpar):
        return np.array([[pt[0] * 2.0, 0.0]] * 2)  # two cams identical
    monkeypatch.setattr(merged_split, "_project", fake_project)
    merged_px = np.array([[1.0, 0.0], [1.0, 0.0]])  # midpoint of 0 and 2 -> proj 1
    seen = np.array([True, True])
    pos, _ = merged_split.split_merged_blob(merged_px, seen, intensities, preds, cals=[None, None], cpar=None, weight_pred=0.1)
    # constituents should separate
    assert pos[1, 0] > pos[0, 0]
