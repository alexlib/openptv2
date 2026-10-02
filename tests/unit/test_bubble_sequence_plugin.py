"""Unit tests for bubble_detection algorithm and bubble_sequence plugin."""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.algorithms.bubble_detection import detect_bubbles_fast
from openptv2.plugins.loader import BUILTIN_SEQUENCE_PLUGINS, resolve_plugin_module


def test_detect_bubbles_fast_synthetic_glare_merge():
    """Verify that multiple glare points of a single bubble merge into one centroid."""
    imy, imx = 100, 100
    # Uneven background with gradient and noise
    y_grad, x_grad = np.mgrid[:imy, :imx]
    bg = 50.0 + 0.2 * x_grad + 0.1 * y_grad + np.random.RandomState(42).normal(0, 1.0, (imy, imx))

    # Bubble 1 at center (50, 50): 2 glare points separated by 3 pixels
    img = bg.copy()
    img[50, 49] += 40.0
    img[50, 52] += 40.0

    # Bubble 2 far away at (20, 80): single glare point
    img[20, 80] += 50.0

    xs, ys, npix = detect_bubbles_fast(
        img,
        win=15,
        z_thresh=6.0,
        merge_radius=4,
        nnmin=1,
        nnmax=5000,
    )

    assert len(xs) == 2, f"Expected 2 merged bubbles, got {len(xs)}"

    # One detection should be near (50.5, 50), the other near (80, 20)
    dists_b1 = np.hypot(xs - 50.5, ys - 50.0)
    dists_b2 = np.hypot(xs - 80.0, ys - 20.0)

    assert np.min(dists_b1) < 1.0, "Bubble 1 centroid mismatch"
    assert np.min(dists_b2) < 1.0, "Bubble 2 centroid mismatch"


def test_detect_bubbles_fast_separate_when_beyond_merge_radius():
    """Verify that points further than merge_radius remain separate detections."""
    imy, imx = 80, 80
    img = np.ones((imy, imx), dtype=np.float64) * 20.0

    # Two separate bubbles 12 pixels apart (beyond merge_radius=4)
    img[40, 30] += 50.0
    img[40, 42] += 50.0

    xs, ys, npix = detect_bubbles_fast(
        img,
        win=15,
        z_thresh=5.0,
        merge_radius=4,
    )

    assert len(xs) == 2


def test_bubble_sequence_plugin_registration():
    """Verify bubble_sequence plugin is registered and resolvable."""
    assert "bubble_sequence" in BUILTIN_SEQUENCE_PLUGINS
    assert "bubble" in BUILTIN_SEQUENCE_PLUGINS
    assert "bubble_detection" in BUILTIN_SEQUENCE_PLUGINS

    mod = resolve_plugin_module("bubble_sequence", BUILTIN_SEQUENCE_PLUGINS)
    assert hasattr(mod, "Sequence")
    assert hasattr(mod.Sequence, "do_sequence")


def test_bubble_sequence_batch_end_to_end(tmp_path):
    """Verify bubble_sequence runs end-to-end via pyptv_batch and populates RunStore."""
    import shutil
    from pathlib import Path

    from openptv2.batch.pyptv_batch import run_batch
    from openptv2.storage import RunStore

    src = Path(__file__).parents[2] / "test_data" / "test_cavity_small"
    work = tmp_path / "exp"
    shutil.copytree(src, work)

    run_batch(
        yaml_file=work / "parameters_Run1_1.yaml",
        seq_first=10000,
        seq_last=10001,
        mode="sequence",
        sequence_plugin="bubble_sequence",
    )

    store = RunStore(work / "res" / "run.zarr", mode="r")
    for cam in range(4):
        targets = store.read_targets(cam, 10000)
        assert len(targets) > 0, f"Camera {cam} should have detected targets"

    pos, cam_ids = store.read_correspondences(10000)
    assert len(pos) > 0, "Frame 10000 should have 3D correspondences"
    assert cam_ids.shape[0] == len(pos)

