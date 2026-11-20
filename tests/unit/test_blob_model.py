"""Blob content model (openptv2.blob_model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.blob_model import (
    brightness_shares,
    signature_from_history,
    split_blob,
    triangulate_weighted,
)
from openptv2.loco import project_pixels
from openptv2.recorrespond import load_parameters

pytestmark = pytest.mark.unit
YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"


def test_single_particle_blob_returns_the_centre():
    s, v = split_blob(np.array([10.0, 20.0]), np.array([[10.2, 19.9]]), np.array([5.0]), 1e-4, 1.0)
    assert s[0] == pytest.approx([10.0, 20.0], abs=1e-3)       # prediction variance >> centre variance
    assert v[0] == pytest.approx(1e-4, rel=0.01)


def test_noise_free_split_preserves_the_weighted_mean():
    pred = np.array([[0.0, 0.0], [4.0, 0.0]])
    bright = np.array([1.0, 3.0])
    w = brightness_shares(bright)
    centre = np.array([1.5, 0.0])                    # weighted mean of the *true* spots (1.0 and 4.0 ... shifted)
    s, v = split_blob(centre, pred, bright, 0.0, 0.1)
    assert w @ s == pytest.approx(centre)            # the measurement is honoured exactly
    # minimum-norm solution: displacement proportional to the share
    r = centre - w @ pred
    expected = pred + (w / np.sum(w**2))[:, None] * r[None, :]
    assert s == pytest.approx(expected)


def test_correct_prediction_is_left_alone():
    pred = np.array([[1.0, 2.0], [3.0, 5.0]])
    w = brightness_shares(np.array([2.0, 2.0]))
    centre = w @ pred
    s, _ = split_blob(centre, pred, np.array([2.0, 2.0]), 0.01, 0.2)
    assert s == pytest.approx(pred)


def test_merged_particles_are_less_certain_than_clean_ones():
    pred = np.array([[0.0, 0.0], [3.0, 0.0]])
    _, v_merged = split_blob(np.array([1.5, 0.0]), pred, np.array([1.0, 1.0]), 0.01, 0.1)
    _, v_clean = split_blob(np.array([0.0, 0.0]), pred[:1], np.array([1.0]), 0.01, 0.1)
    assert np.all(v_merged > v_clean[0])
    assert np.all(v_merged < 0.1 + 1e-12)            # but the blob still tells something about each


def test_shift_is_shared_in_proportion_to_brightness_share():
    # equal prior uncertainty: to explain a shifted blob centre with the least total movement, each
    # particle moves in proportion to its brightness share (the bright one has the longer lever arm)
    pred = np.array([[0.0, 0.0], [3.0, 0.0]])
    bright = np.array([1.0, 9.0])
    s, _ = split_blob(np.array([1.0, 0.0]), pred, bright, 0.0, 0.1)
    d = s - pred
    w = brightness_shares(bright)
    assert abs(d[1, 0]) > abs(d[0, 0])
    assert d[1, 0] / d[0, 0] == pytest.approx(w[1] / w[0])
    assert w @ s == pytest.approx([1.0, 0.0])


def test_brightness_shares_handle_missing_signatures():
    w = brightness_shares(np.array([np.nan, np.nan]))
    assert w == pytest.approx([0.5, 0.5])
    w = brightness_shares(np.array([2.0, np.nan]))
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)


def test_signature_uses_only_alone_frames():
    sumg = np.array([100.0, 205.0, 98.0, 102.0, 310.0])
    alone = np.array([True, False, True, True, False])
    assert signature_from_history(sumg, alone) == pytest.approx(100.0)
    assert np.isnan(signature_from_history(sumg, np.zeros(5, bool)))


@pytest.fixture(scope="module")
def rig():
    cpar, vpar, cals, _ = load_parameters(YAML)
    rng = np.random.default_rng(3)
    pts = rng.uniform(-10, 10, (2000, 3))
    px = np.stack([project_pixels(pts, c, cpar) for c in cals], axis=1)
    ok = np.all((px[:, :, 0] > 5) & (px[:, :, 0] < cpar.imx - 5) & (px[:, :, 1] > 5) & (px[:, :, 1] < cpar.imy - 5), axis=1)
    return cpar, cals, pts[ok][:100], px[ok][:100]


def test_weighted_triangulation_recovers_exact_points(rig):
    cpar, cals, pts, px = rig
    x0 = pts + np.random.default_rng(0).normal(0, 0.3, pts.shape)
    x = triangulate_weighted(px, np.full(px.shape[:2], 0.01), cpar, cals, x0, n_iter=4)
    assert np.abs(x - pts).max() < 1e-2


def test_a_noisy_camera_with_large_variance_is_downweighted(rig):
    cpar, cals, pts, px = rig
    bad = px.copy()
    bad[:, 1, 0] += 3.0                                           # camera 2 off by 3 px
    var = np.full(px.shape[:2], 0.01)
    x_eq = triangulate_weighted(bad, var, cpar, cals, pts, n_iter=4)
    var_w = var.copy()
    var_w[:, 1] = 100.0                                           # tell the model camera 2 is poor
    x_w = triangulate_weighted(bad, var_w, cpar, cals, pts, n_iter=4)
    assert np.linalg.norm(x_w - pts, axis=1).mean() < 0.1 * np.linalg.norm(x_eq - pts, axis=1).mean()


def test_points_with_fewer_than_two_cameras_keep_the_start(rig):
    cpar, cals, pts, px = rig
    p = px.copy()
    p[:5, 1:] = np.nan
    x0 = pts + 1.0
    x = triangulate_weighted(p, np.full(p.shape[:2], 0.01), cpar, cals, x0)
    assert np.all(x[:5] == x0[:5])
    assert np.abs(x[5:] - pts[5:]).max() < 1e-2


def test_effective_observation_of_a_single_particle_is_the_centre():
    from openptv2.blob_model import split_blob_effective

    s, v = split_blob_effective(np.array([4.0, 5.0]), np.array([[3.0, 3.0]]), np.array([1.0]), 0.01, 0.1)
    assert s[0] == pytest.approx([4.0, 5.0]) and v[0] == pytest.approx(0.01)


def test_effective_observation_recovers_the_true_spot_when_the_other_prediction_is_exact():
    from openptv2.blob_model import split_blob_effective

    true = np.array([[0.0, 0.0], [3.0, 1.0]])
    bright = np.array([2.0, 3.0])
    w = brightness_shares(bright)
    centre = w @ true                                   # noise-free blob centre
    pred = true + np.array([[0.3, -0.2], [0.0, 0.0]])  # particle 0's own prediction is off; particle 1's exact
    s, _ = split_blob_effective(centre, pred, bright, 0.01, 0.05)
    assert s[0] == pytest.approx(true[0])               # particle 0 is recovered from the blob and 1's prediction
    # the particle's own (wrong) prediction is not used at all
    pred2 = pred.copy()
    pred2[0] = [50.0, 50.0]
    s2, _ = split_blob_effective(centre, pred2, bright, 0.01, 0.05)
    assert s2[0] == pytest.approx(s[0])


def test_effective_variance_grows_as_the_particle_gets_dimmer():
    from openptv2.blob_model import split_blob_effective

    pred = np.array([[0.0, 0.0], [3.0, 0.0]])
    _, v_eq = split_blob_effective(np.array([1.5, 0.0]), pred, np.array([1.0, 1.0]), 0.01, 0.05)
    _, v_dim = split_blob_effective(np.array([1.5, 0.0]), pred, np.array([1.0, 9.0]), 0.01, 0.05)
    assert v_eq[0] == pytest.approx(v_eq[1])            # equal particles, equal certainty
    assert v_eq[0] == pytest.approx(4 * 0.01 + 0.05)    # (var_c + 0.25 var_p) / 0.25
    assert v_dim[0] > 10 * v_dim[1]                     # the dim one is nearly unobserved by this blob


def test_local_poly_predict_is_exact_for_a_quadratic_and_ignores_the_point_itself():
    from openptv2.blob_model import local_poly_predict

    t = np.arange(20)
    traj = np.c_[0.1 * t, 0.02 * t**2, 1.0 - 0.3 * t]
    tid = np.zeros(20, int)
    pred = local_poly_predict(tid, t, traj, window=9, order=2)
    assert pred == pytest.approx(traj, abs=1e-6)
    bad = traj.copy()
    bad[10] += [5.0, -5.0, 5.0]                         # a wild observation
    pred = local_poly_predict(tid, t, bad, window=9, order=2)
    assert pred[10] == pytest.approx(traj[10], abs=1e-6)   # its own prediction is not pulled by it
    assert np.abs(pred[9] - traj[9]).max() > 0.05          # but its neighbours' predictions are


def test_local_poly_predict_handles_gaps_short_tracks_and_several_tracks():
    from openptv2.blob_model import local_poly_predict

    f = np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 0, 1, 2])           # track 0 has a gap; track 1 has 3 points
    tid = np.array([0] * 9 + [1] * 3)
    pos = np.c_[f * 1.0, f * 2.0, np.zeros(12)]
    pred = local_poly_predict(tid, f, pos, window=7, order=1)
    assert pred[:9] == pytest.approx(pos[:9], abs=1e-6)            # a line stays a line across the gap
    assert np.all(np.isnan(pred[9:]))                              # fewer than min_points


def test_local_poly_predict_reports_support_and_downweights_doubtful_points():
    from openptv2.blob_model import local_poly_predict

    t = np.arange(15)
    traj = np.c_[0.1 * t, 0.02 * t, 0.0 * t]
    tid = np.zeros(15, int)
    w = np.ones(15)
    w[5:10] = 0.01                                       # five doubtful frames
    bad = traj.copy()
    bad[5:10] += 3.0                                     # ... and they are wrong
    pred, sup = local_poly_predict(tid, t, bad, w, window=9, order=1, return_support=True)
    assert sup[2] > 3.0                                  # well supported by good neighbours
    _, sup5 = local_poly_predict(tid, t, bad, w, window=5, order=1, return_support=True)
    assert sup5[7] < 0.1                                 # window entirely inside the doubtful stretch
    assert np.abs(pred[2] - traj[2]).max() < 0.1         # the doubtful frames hardly pull a good one
    plain = local_poly_predict(tid, t, bad, None, window=9, order=1)
    assert np.abs(pred[3] - traj[3]).max() < np.abs(plain[3] - traj[3]).max()


def test_prior_slots_pull_a_poorly_measured_point_to_the_path(rig):
    cpar, cals, pts, px = rig
    bad = px.copy()
    bad[:, :, :] += np.random.default_rng(5).normal(0, 2.0, bad.shape)       # every camera badly off
    var = np.full(px.shape[:2], 4.0)                                           # ... and said to be
    x_noprior = triangulate_weighted(bad, var, cpar, cals, pts, n_iter=4)
    both = np.concatenate([bad, px], axis=1)                                   # slots 4..7: exact path prediction
    var_both = np.concatenate([var, np.full(px.shape[:2], 0.01)], axis=1)
    x_prior = triangulate_weighted(both, var_both, cpar, cals, pts, n_iter=4, cam_index=[0, 1, 2, 3, 0, 1, 2, 3])
    assert np.linalg.norm(x_prior - pts, axis=1).mean() < 0.1 * np.linalg.norm(x_noprior - pts, axis=1).mean()


def test_duplicate_camera_slots_add_their_weights(rig):
    cpar, cals, pts, px = rig
    noisy = px + np.random.default_rng(6).normal(0, 0.5, px.shape)
    var = np.full(px.shape[:2], 0.2)
    a = triangulate_weighted(noisy, var, cpar, cals, pts, n_iter=4)
    b = triangulate_weighted(np.concatenate([noisy, noisy], axis=1), np.concatenate([var, var], axis=1),
                             cpar, cals, pts, n_iter=4, cam_index=[0, 1, 2, 3, 0, 1, 2, 3])
    assert np.abs(a - b).max() < 1e-6                                          # same estimate, just more certain
    with pytest.raises(ValueError):
        triangulate_weighted(px, var, cpar, cals, pts, cam_index=[0, 1])


def test_covariance_is_larger_with_fewer_or_noisier_cameras(rig):
    cpar, cals, pts, px = rig
    var = np.full(px.shape[:2], 0.01)
    x, cov = triangulate_weighted(px, var, cpar, cals, pts, return_cov=True)
    tr4 = np.trace(cov, axis1=1, axis2=2)
    assert np.all(tr4 > 0)
    p3 = px.copy()
    p3[:, 3] = np.nan
    _, cov3 = triangulate_weighted(p3, var, cpar, cals, pts, return_cov=True)
    assert np.all(np.trace(cov3, axis1=1, axis2=2) > tr4)                  # one camera fewer: less certain
    _, cov_n = triangulate_weighted(px, var * 4, cpar, cals, pts, return_cov=True)
    assert np.trace(cov_n, axis1=1, axis2=2) == pytest.approx(4 * tr4, rel=1e-3)   # variance scales with pixel variance
    p1 = px.copy()
    p1[:3, 1:] = np.nan
    _, c1 = triangulate_weighted(p1, var, cpar, cals, pts, return_cov=True)
    assert np.all(np.isnan(c1[:3]))


def test_joint_cluster_solve_recovers_two_particles_sharing_a_blob(rig):
    from openptv2.blob_model import solve_cluster

    cpar, cals, pts, px = rig
    a, b = pts[0], pts[0] + np.array([0.15, 0.05, 0.0])
    pa = np.stack([project_pixels(a[None], c, cpar)[0] for c in cals])
    pb = np.stack([project_pixels(b[None], c, cpar)[0] for c in cals])
    wa, wb = 2 / 3, 1 / 3                                     # a is twice as bright
    clean = [(0, 1, pa[1], 0.01), (0, 2, pa[2], 0.01), (1, 2, pb[2], 0.01), (1, 3, pb[3], 0.01)]
    shared = [((0, 1), (wa, wb), 0, wa * pa[0] + wb * pb[0], 0.01),
              ((0, 1), (wa, wb), 3, None, 0.01)]
    shared = [shared[0]]                                       # merged in camera 0 only
    clean.append((0, 3, pa[3], 0.01))
    clean.append((1, 1, pb[1], 0.01))
    x0 = np.vstack([a, b]) + np.random.default_rng(0).normal(0, 0.05, (2, 3))
    x = solve_cluster(x0, clean, shared, [], cpar, cals)
    assert np.linalg.norm(x[0] - a) < 1e-3 and np.linalg.norm(x[1] - b) < 1e-3


def test_joint_cluster_solve_places_a_particle_seen_only_inside_shared_blobs(rig):
    from openptv2.blob_model import solve_cluster

    cpar, cals, pts, px = rig
    a, b = pts[1], pts[1] + np.array([0.1, -0.08, 0.05])
    pa = np.stack([project_pixels(a[None], c, cpar)[0] for c in cals])
    pb = np.stack([project_pixels(b[None], c, cpar)[0] for c in cals])
    w = (0.5, 0.5)
    clean = [(0, c, pa[c], 0.01) for c in range(4)]           # a is alone in a blob in every camera ...
    clean = clean[:2]                                          # ... only in cameras 0 and 1
    shared = [((0, 1), w, c, 0.5 * pa[c] + 0.5 * pb[c], 0.01) for c in (2, 3)]
    clean += [(1, 0, pb[0], 0.01)]                             # b alone only in camera 0
    x0 = np.vstack([a, b]) + 0.05
    x = solve_cluster(x0, clean, shared, [], cpar, cals)
    assert np.linalg.norm(x[1] - b) < 1e-3                     # b is placed from one clean view + two shared blobs
