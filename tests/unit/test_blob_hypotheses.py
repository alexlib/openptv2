import numpy as np
import pytest

from openptv2.blob_hypotheses import Logistic, blob_features, fit_one, fit_two
from openptv2.segmentation import _vectorized_targ_rec

S2 = 0.4


def _image(centres, amps, size=40):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    img = np.zeros((size, size))
    for (x, y), a in zip(centres, amps):
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * S2))
    return np.clip(img, 0, 255).astype(np.uint8)


def _blobs(img, thres=10):
    pix: list = []
    ts = _vectorized_targ_rec(img, thres, 80, 1, 100, 1, 20, 1, 20, 0, 1, 38, 1, 38, pixels_out=pix)
    return ts, pix


@pytest.mark.unit
def test_pixel_capture_matches_targets():
    img = _image([(12.3, 14.1)], [120])
    ts, pix = _blobs(img)
    assert len(ts) == len(pix) == 1
    assert len(pix[0]) == ts[0].count_pixels()[0]
    assert pix[0][:, 2].sum() == ts[0].sum_grey_value()


@pytest.mark.unit
def test_single_blob_is_round_and_two_blob_is_elongated():
    one = _blobs(_image([(15.2, 15.4)], [120]))[1][0]
    two = _blobs(_image([(14.0, 15.0), (15.9, 15.0)], [100, 100]))[1]
    assert len(two) == 1
    two = two[0]
    f1, f2 = blob_features(one, 10.0), blob_features(two, 10.0)
    names = ("n", "mean_int", "sxx", "sxy", "syy", "lam1", "lam2", "ecc", "n_max", "m3")
    a, b = dict(zip(names, f1)), dict(zip(names, f2))
    assert b["n"] > a["n"] and b["lam1"] > 2 * a["lam1"] and b["ecc"] > a["ecc"]


@pytest.mark.unit
def test_fit_two_recovers_both_centres_where_the_centroid_cannot():
    centres = [(14.0, 15.0), (16.0, 15.3)]
    img = _image(centres, [110, 80])
    px = _blobs(img)[1][0]
    r = fit_two(img, px, S2)
    got = sorted([(r[0], r[1]), (r[3], r[4])])
    for g, t in zip(got, sorted(centres)):
        assert np.hypot(g[0] - t[0], g[1] - t[1]) < 0.15
    centroid = (px[:, 0] * px[:, 2]).sum() / px[:, 2].sum()
    assert min(abs(centroid - c[0]) for c in centres) > 0.8   # the centroid sits between the two


@pytest.mark.unit
def test_fit_one_recovers_a_single_centre_and_costs_less_than_two_on_a_single():
    img = _image([(15.3, 14.6)], [130])
    px = _blobs(img)[1][0]
    x, y, a, cost1 = fit_one(img, px, S2)
    assert abs(x - 15.3) < 0.1 and abs(y - 14.6) < 0.1
    assert cost1 < 50


@pytest.mark.unit
def test_logistic_separates_and_is_calibrated_on_overlapping_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (2000, 2)), rng.normal(1.5, 1, (2000, 2))])
    y = np.r_[np.zeros(2000), np.ones(2000)]
    p = Logistic.fit(X, y).predict(X)
    assert ((p > 0.5) == y).mean() > 0.8
    assert abs(p[p > 0.7].mean() - y[p > 0.7].mean()) < 0.08
