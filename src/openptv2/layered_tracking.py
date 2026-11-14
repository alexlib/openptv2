"""Layered / peeling tracking (backbone improvement, Milestone K).

Strategy: instead of tracking the full 3D correspondence cloud at once (which
inflates ghost count and ambiguity), track top-down in quality layers:

  1. Score each correspondence by quality (camera support count, or ghost
     probability when calibration is available).
  2. Layer 0 tracks the top quality tier; those particles are removed.
  3. Layer 1 tracks the next tier on the cleaned, sparser data.
  4. Repeat; final postprocess on the accumulated set.

This module provides the layering/filtering primitives (testable without a full
experiment). The orchestration that calls into ``Tracker`` lives in
``plugins/unified_tracking.py`` / ``track_assisted`` and is wired later.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openptv2.ghost_filter import camera_support


def quality_scores(cam_target_ids: np.ndarray) -> np.ndarray:
    """Per-correspondence quality (higher = more trustworthy).

    Default: camera support count. When ghost probabilities are available,
    pass them in via ``layer_by_quality``'s ``scores`` override.
    """
    return camera_support(cam_target_ids).astype(float)


def assign_layers(
    scores: np.ndarray, n_layers: int = 3, quantiles: list[float] | None = None
) -> np.ndarray:
    """Assign each correspondence to a layer index (0..n_layers-1).

    Layer 0 = highest quality. Quantiles define cumulative cutoffs.
    """
    if quantiles is None:
        quantiles = [i / n_layers for i in range(1, n_layers)] + [1.0]
    if len(quantiles) != n_layers:
        raise ValueError("quantiles length must equal n_layers")
    order = np.argsort(scores)[::-1]  # highest first
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(len(scores))
    cutoffs = (np.array(quantiles) * len(scores)).astype(int)
    layer = np.zeros(len(scores), dtype=np.int32)
    prev = 0
    for i, c in enumerate(cutoffs):
        layer[(ranks >= prev) & (ranks < c)] = i
        prev = c
    # tail
    layer[ranks >= cutoffs[-1]] = n_layers - 1
    return layer


def peel_layers(
    store: Any,
    frames: list[int],
    n_layers: int = 3,
    quantiles: list[float] | None = None,
    score_fn=None,
) -> list[dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Split each frame's correspondences into ``n_layers`` quality tiers.

    Returns a list (one per layer) of dicts: frame -> (pos, cam_ids) for that
    layer's subset. Layers are ordered best-quality first.
    """
    layers = [{} for _ in range(n_layers)]
    for f in frames:
        if not store.has_correspondences(f):
            for L in layers:
                L[f] = (np.empty((0, 3)), np.empty((0, 0), dtype=np.int32))
            continue
        pos, cam = store.read_correspondences(f)
        if len(pos) == 0:
            for L in layers:
                L[f] = (pos, cam)
            continue
        scores = score_fn(cam) if score_fn is not None else quality_scores(cam)
        lyr = assign_layers(scores, n_layers, quantiles)
        for i in range(n_layers):
            mask = lyr == i
            layers[i][f] = (pos[mask], cam[mask])
    return layers