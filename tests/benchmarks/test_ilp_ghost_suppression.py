"""Benchmarks for ILP-based ghost suppression.

Synthetic scenarios with known ground-truth 3D particles and injected ghosts.
Measures precision/recall of the ILP uniqueness pass and its runtime.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from openptv2.ghost_ilp import ilp_uniqueness_frame


def _make_scenario(n_real: int, n_ghost: int, n_targets_per_cam: int, seed: int = 0):
    """Build a frame's candidate 3D points: real particles (unique targets) +
    ghost particles (reusing real targets) + noise. Returns (pos, cam_ids, is_real)."""
    rng = np.random.default_rng(seed)
    n_cams = 4
    # real particles: each uses a distinct target in every camera
    real_targets = rng.integers(0, n_targets_per_cam, size=(n_real, n_cams))
    # ensure uniqueness per camera by shifting
    for c in range(n_cams):
        real_targets[:, c] = np.arange(n_real) % n_targets_per_cam
    pos_real = rng.normal(size=(n_real, 3))
    cam_real = real_targets.copy()
    is_real = np.ones(n_real, dtype=bool)

    # ghosts: reuse targets from real particles (conflict)
    ghost_idx = rng.integers(0, n_real, size=n_ghost)
    pos_ghost = rng.normal(size=(n_ghost, 3))
    cam_ghost = real_targets[ghost_idx].copy()
    # perturb one camera target to keep them "distinct" but conflicting
    cam_ghost[:, 0] = cam_ghost[:, 0]  # same target -> conflict
    is_ghost = np.zeros(n_ghost, dtype=bool)

    pos = np.vstack([pos_real, pos_ghost])
    cam_ids = np.vstack([cam_real, cam_ghost])
    truth = np.concatenate([is_real, is_ghost])
    # shuffle
    perm = rng.permutation(len(pos))
    return pos[perm], cam_ids[perm], truth[perm]


@pytest.mark.parametrize("density", [50, 200, 800])
@pytest.mark.parametrize("ghost_ratio", [0.1, 0.3, 0.6])
def test_ilp_ghost_suppression_precision_recall(density, ghost_ratio):
    n_real = density
    n_ghost = int(density * ghost_ratio)
    pos, cam, truth = _make_scenario(n_real, n_ghost, n_targets_per_cam=max(64, density), seed=42)
    conf = np.ones(len(pos))
    # ghosts get slightly lower conf (realistic: rcm higher -> lower conf)
    conf[~truth] = 0.3
    keep = ilp_uniqueness_frame(pos, cam, conf=conf)
    # precision: fraction kept that are real
    kept_real = int((keep & truth).sum())
    kept_total = int(keep.sum())
    precision = kept_real / max(kept_total, 1)
    recall = kept_real / max(int(truth.sum()), 1)
    # ILP should keep most real particles and drop most ghosts
    assert precision >= 0.5, f"precision {precision:.3f} < 0.5 (density={density}, ghost_ratio={ghost_ratio})"
    assert recall >= 0.5, f"recall {recall:.3f} < 0.5"


def test_ilp_runtime_scaling():
    for n in [100, 500, 2000]:
        pos, cam, truth = _make_scenario(n, n // 2, n_targets_per_cam=max(64, n), seed=7)
        conf = np.where(truth, 1.0, 0.3)
        t0 = time.time()
        keep = ilp_uniqueness_frame(pos, cam, conf=conf)
        dt = time.time() - t0
        # Should scale sub-quadratically and stay fast.
        assert dt < 5.0, f"n={n} took {dt:.2f}s (>5s)"


def test_ilp_no_ghosts_keeps_all():
    # When there are no conflicts, ILP should keep everything.
    pos = np.arange(50)[:, None] * np.ones((1, 3))
    cam = np.column_stack([np.arange(50), np.arange(50) + 100])
    keep = ilp_uniqueness_frame(pos, cam)
    assert keep.all()


def test_ilp_all_conflict_keeps_best():
    # All points share one target -> keep only highest conf.
    n = 10
    pos = np.arange(n)[:, None] * np.ones((1, 3))
    cam = np.zeros((n, 2), dtype=int)
    conf = np.arange(n, dtype=float)
    keep = ilp_uniqueness_frame(pos, cam, conf=conf)
    assert int(keep.sum()) == 1
    assert keep[int(np.argmax(conf))]