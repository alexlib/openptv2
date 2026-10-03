"""Unit tests for bubble_detection algorithm and bubble_sequence plugin."""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.algorithms.bubble_detection import detect_bubbles_fast
from openptv2.algorithms.bubble_background import (
    estimate_background,
    subtract_background,
)
from openptv2.plugins.loader import BUILTIN_SEQUENCE_PLUGINS, resolve_plugin_module


def test_estimate_background_recovers_constant_pixels():
    """Temporal median must recover the static background under moving bubbles."""
    rng = np.random.RandomState(7)
    imy, imx, K = 60, 60, 12
    yy, xx = np.mgrid[:imy, :imx]
    true_bg = 40.0 + 0.3 * xx + 0.1 * yy  # static wall shading gradient
    stack = np.empty((K, imy, imx), dtype=np.uint8)
    for k in range(K):
        img = true_bg + rng.normal(0, 1.0, (imy, imx))
        # one bright bubble at a different spot each frame (minority occupant)
        img[10 + 3 * k, 30] += 60.0
        stack[k] = np.clip(img, 0, 255).astype(np.uint8)

    bg, info = estimate_background(stack)
    assert info["K"] == K
    assert info["frac_unstable"] < 0.05
    # Pixels that saw a bubble use the temporal-minimum fallback, which
    # carries noise-extreme error; the static majority must match closely.
    err = np.abs(bg - true_bg)
    assert np.median(err) < 1.0
    assert np.percentile(err, 99) < 5.0


def test_estimate_background_dark_polarity_uses_maximum():
    """Dark-on-bright bubbles: unstable pixels fall back to temporal max."""
    rng = np.random.RandomState(3)
    imy, imx, K = 40, 40, 10
    stack = np.full((K, imy, imx), 200, dtype=np.uint8)
    for k in range(K):
        img = stack[k].astype(float) + rng.normal(0, 1.0, (imy, imx))
        img[5 + 3 * k, 20] -= 120.0  # dark bubble, moving
        stack[k] = np.clip(img, 0, 255).astype(np.uint8)

    bg, _ = estimate_background(stack, polarity="dark")
    np.testing.assert_allclose(bg, 200.0, atol=3.0)


def test_subtract_background_keeps_bubbles_positive():
    """Bright: img-bg clipped at 0; dark: bg-img so bubbles stay positive."""
    bg = np.full((8, 8), 50.0)
    img = bg.copy()
    img[3, 3] = 120.0
    sub = subtract_background(img, bg, polarity="bright")
    assert sub[3, 3] == 70.0
    assert sub.min() >= 0.0
    assert np.count_nonzero(sub) == 1  # only the bubble survives

    dark_bg = np.full((8, 8), 200.0)
    dark_img = dark_bg.copy()
    dark_img[5, 5] = 80.0  # dark bubble on bright background
    dark = subtract_background(dark_img, dark_bg, polarity="dark")
    assert dark[5, 5] == 120.0
    assert dark.min() >= 0.0
    assert np.count_nonzero(dark) == 1  # only the bubble survives
    with pytest.raises(ValueError):
        subtract_background(img, bg, polarity="sideways")
    with pytest.raises(ValueError):
        estimate_background(np.ones((4, 4)))  # not a stack


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

