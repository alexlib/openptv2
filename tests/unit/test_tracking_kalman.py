"""Unit tests for Constant-Acceleration 3D Kalman Filter predictor (tracking_kalman.py)."""

import numpy as np
import pytest

from openptv2.tracking_kalman import ConstantAccelerationKF3D, KalmanTrackState


def test_init_state():
    kf = ConstantAccelerationKF3D(measurement_noise=0.05, v_max=10.0, a_max=5.0)
    pos = np.array([10.0, 20.0, 30.0])
    ts = kf.init_state(track_id=1, pos=pos, frame_idx=0)

    assert ts.track_id == 1
    assert ts.history_len == 1
    assert ts.last_frame == 0
    np.testing.assert_allclose(ts.state[0:3], pos)
    np.testing.assert_allclose(ts.state[3:9], 0.0)
    assert ts.cov[0, 0] == pytest.approx(0.05**2)
    assert ts.cov[3, 3] == pytest.approx(10.0**2)
    assert ts.cov[6, 6] == pytest.approx(5.0**2)


def test_linear_motion_prediction_and_update():
    kf = ConstantAccelerationKF3D(process_noise_acc=0.1, measurement_noise=0.02)
    true_v = np.array([2.0, -1.0, 0.5])
    pos0 = np.array([0.0, 0.0, 0.0])

    ts = kf.init_state(track_id=42, pos=pos0, frame_idx=0)

    # Simulate 5 frames of constant velocity motion
    curr_pos = pos0.copy()
    for frame in range(1, 6):
        curr_pos = pos0 + true_v * frame + np.random.normal(0, 0.01, 3)
        ts = kf.update(ts, curr_pos, dt=1.0, frame_idx=frame)

    # Velocity estimate should converge near true velocity
    np.testing.assert_allclose(ts.state[3:6], true_v, atol=0.2)
    assert ts.history_len == 6


def test_innovation_covariance_and_mahalanobis_gating():
    kf = ConstantAccelerationKF3D(measurement_noise=0.1, gate_chi2=11.34)
    ts = kf.init_state(track_id=1, pos=np.array([0.0, 0.0, 0.0]))

    pred_state, pred_cov, pred_pos, S = kf.predict(ts, dt=1.0)

    # S should be symmetric 3x3 positive definite
    assert S.shape == (3, 3)
    np.testing.assert_allclose(S, S.T)
    assert np.all(np.linalg.eigvals(S) > 0)

    # Candidates: one near prediction, one far outside gate
    cands = np.array([
        [0.05, 0.05, -0.05],  # Near prediction
        [100.0, 0.0, 0.0],    # Well outside gate (6.6+ sigma away)
    ])

    mask = kf.in_gate(cands, pred_pos, S)
    assert mask[0] == True
    assert mask[1] == False


def test_gap_expands_innovation_covariance():
    kf = ConstantAccelerationKF3D(process_noise_acc=1.0, measurement_noise=0.05)
    ts = kf.init_state(track_id=10, pos=np.array([10.0, 10.0, 10.0]))

    _, _, _, S_dt1 = kf.predict(ts, dt=1.0)
    _, _, _, S_dt3 = kf.predict(ts, dt=3.0)

    # Innovation matrix diagonal for 3-frame gap should be strictly larger than 1-frame gap
    assert np.all(np.diag(S_dt3) > np.diag(S_dt1))


def test_batch_predict():
    kf = ConstantAccelerationKF3D()
    ts1 = kf.init_state(track_id=1, pos=np.array([0.0, 0.0, 0.0]))
    ts2 = kf.init_state(track_id=2, pos=np.array([10.0, 20.0, 30.0]))

    states, covs, positions, S_mats = kf.batch_predict([ts1, ts2], dt=1.0)

    assert states.shape == (2, 9)
    assert covs.shape == (2, 9, 9)
    assert positions.shape == (2, 3)
    assert S_mats.shape == (2, 3, 3)

    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[1], [10.0, 20.0, 30.0])


def test_empty_candidates_and_batch():
    kf = ConstantAccelerationKF3D()
    empty_cands = np.empty((0, 3))
    S = np.eye(3)
    pred_pos = np.array([0.0, 0.0, 0.0])

    dists = kf.mahalanobis_distance_sq(empty_cands, pred_pos, S)
    assert len(dists) == 0

    states, covs, positions, S_mats = kf.batch_predict([])
    assert states.shape == (0, 9)
