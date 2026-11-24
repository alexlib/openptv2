"""Pixel-level de-merging of a shared blob (openptv2.pixel_fit)."""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.pixel_fit import PixelFitter, fit_window

pytestmark = pytest.mark.unit


def render(spots, size=40, sigma=0.8, noise=1.0, seed=0, bg=10.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] + 0.5            # pixel (i, j) is centred at i + 0.5, as in targ_rec
    img = np.full((size, size), bg)
    for x, y, a in spots:
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255)


def test_two_overlapping_spots_are_separated_better_than_their_seeds():
    truth = np.array([[19.6, 20.3], [21.5, 20.9]])                    # 2 px apart: one merged blob
    img = render([(*truth[0], 120.0), (*truth[1], 70.0)])
    seeds = truth + np.array([[0.25, -0.2], [-0.3, 0.2]])             # the track's prediction, a few tenths off
    centre = truth.mean(0)
    pos, var, noise = fit_window(img, seeds, np.full(2, 0.3), centre, 7)
    err_fit = np.linalg.norm(pos - truth, axis=1)
    err_seed = np.linalg.norm(seeds - truth, axis=1)
    assert (err_fit < 0.5 * err_seed).all() and err_fit.max() < 0.15
    assert (var > 0).all() and np.sqrt(var).max() < 0.3 and noise > 0


def test_window_outside_the_image_returns_none():
    img = render([(3.0, 3.0, 100.0)])
    assert fit_window(img, np.array([[3.0, 3.0]]), np.array([0.3]), np.array([3.0, 3.0]), 7) is None


def test_a_spot_below_the_noise_gets_no_variance():
    img = render([(20.0, 20.0, 100.0)], noise=2.0)
    pos, var, _ = fit_window(img, np.array([[20.0, 20.0], [24.0, 20.0]]), np.array([0.3, 0.3]), np.array([22.0, 20.0]), 7)
    assert np.isfinite(var[0]) and np.isnan(var[1])


def test_fitter_reads_a_frame_once_and_maps_its_result():
    truth = np.array([[19.6, 20.3], [21.5, 20.9]])
    img = render([(*truth[0], 120.0), (*truth[1], 70.0)])
    reads = []

    def read(frame):
        reads.append(frame)
        return [img, img]

    fit = PixelFitter(read)
    for cam in (0, 1):
        out = fit(5, cam, truth.mean(0), truth + 0.2, 0.3, np.array([[5.0, 5.0], [30.0, 30.0]]))
        assert out is not None and out[0].shape == (2, 2) and np.isfinite(out[1]).all()
    fit(5, 0, truth.mean(0), truth + 0.2, 0.3, None)
    assert reads == [5] and fit.stats["fitted"] == 3
    assert fit(5, 0, np.array([2.0, 2.0]), np.array([[2.0, 2.0], [3.0, 2.0]]), 0.3, None) is None   # at the border
