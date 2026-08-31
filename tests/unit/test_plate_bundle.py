"""Joint plate bundle: does it recover a rig it did not start from?

Built on synthetic data with known truth, so "better" is not a matter of taste.
The initial guess is deliberately the single-plane answer -- camera poses exact
on the reference frame, every other plate pose merely propagated -- which is the
situation the bundle exists to improve on.
"""
import numpy as np
import pytest

from openptv2.plate_bundle import (
    BundleResult,
    PlateObservations,
    agreeing_views,
    bundle_plate_poses,
    project,
    rodrigues,
    rotvec,
    tilt_off_vertical_deg,
)

NX, NY, PITCH = 6, 7, 120.0
K = np.array([[1717.0, 0.0, 1280.0], [0.0, 1717.0, 1024.0], [0.0, 0.0, 1.0]])


def plate_points():
    ix, iy = np.meshgrid(np.arange(NX), np.arange(NY))
    return np.stack([(ix.ravel() - 2) * PITCH, (iy.ravel() - 3) * PITCH,
                     np.zeros(NX * NY)], 1).astype(float)


def look_at(C, target=(0.0, 0.0, 0.0)):
    """World->camera (R, t) for a camera at C looking at target, y roughly up."""
    C = np.asarray(C, float)
    f = np.asarray(target, float) - C
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])          # rows: camera x, y(down), z(forward)
    return R, -R @ C


TRUE_CAMS = [look_at(c) for c in [(1470, 137, 3060), (1482, 2253, 3000),
                                  (-1472, 137, 2968), (-1500, 2254, 3007)]]


def make_scene(n_frames=9, seed=0):
    """Reference frame plus n_frames-1 vertical plates at assorted yaw/depth."""
    rng = np.random.default_rng(seed)
    obj = plate_points()
    plate_R, plate_t = [], []
    for _ in range(n_frames - 1):
        yaw = np.radians(rng.uniform(-25, 30))
        # a hand-held plate: mostly a yaw about +Y, with a degree of slop
        R = rodrigues([0.0, yaw, 0.0]) @ rodrigues(np.radians(rng.normal(0, 0.4, 3)))
        plate_R.append(R)
        plate_t.append(np.array([rng.uniform(-600, 600), rng.uniform(-400, 400),
                                 rng.uniform(-1200, 400)]))
    cam, frame, o, pix = [], [], [], []
    for fi in range(n_frames):
        Rf = np.eye(3) if fi == 0 else plate_R[fi - 1]
        tf = np.zeros(3) if fi == 0 else plate_t[fi - 1]
        Xw = obj @ Rf.T + tf
        for ci, (R, t) in enumerate(TRUE_CAMS):
            Xc = Xw @ R.T + t
            p = np.stack([K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2],
                          K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]], 1)
            p += rng.normal(0, 0.3, p.shape)          # detector noise
            cam.append(np.full(len(obj), ci))
            frame.append(np.full(len(obj), fi - 1))   # -1 = reference
            o.append(obj)
            pix.append(p)
    obs = PlateObservations(np.concatenate(cam), np.concatenate(frame),
                            np.concatenate(o), np.concatenate(pix))
    return obs, plate_R, plate_t


def perturbed_start(plate_R, plate_t, seed=1):
    """The single-plane starting point: camera poses slightly off, plates rough."""
    rng = np.random.default_rng(seed)
    cam_rvec = np.array([rotvec(R) + rng.normal(0, 0.004, 3) for R, _ in TRUE_CAMS])
    cam_tvec = np.array([t + rng.normal(0, 12.0, 3) for _, t in TRUE_CAMS])
    prv = np.array([rotvec(R) + rng.normal(0, 0.01, 3) for R in plate_R])
    ptv = np.array([t + rng.normal(0, 25.0, 3) for t in plate_t])
    return cam_rvec, cam_tvec, prv, ptv


def test_bundle_recovers_the_true_rig():
    obs, plate_R, plate_t = make_scene()
    start = perturbed_start(plate_R, plate_t)
    before = np.linalg.norm(
        project(np.concatenate([start[0].ravel(), start[1].ravel(),
                                np.column_stack([start[2], start[3]]).ravel()]),
                obs, K, 4, len(plate_R)) - obs.pix, axis=1)
    res = bundle_plate_poses(obs, *start, K)

    assert np.sqrt(np.mean(res.residual_px ** 2)) < 0.5 * np.sqrt(np.mean(before ** 2))
    assert np.median(res.residual_px) < 0.6          # detector noise is 0.3 px

    # Every camera centre must end up closer to truth than it started.  An
    # absolute bound would be arbitrary here: 9 synthetic frames at 3 m with
    # 0.3 px noise pin a centre to a few mm, so a fixed threshold would be
    # testing the scene's geometry rather than the solver.
    for ci, (R, t) in enumerate(TRUE_CAMS):
        truth = -R.T @ t
        started = np.linalg.norm(-rodrigues(start[0][ci]).T @ start[1][ci] - truth)
        ended = np.linalg.norm(res.camera_centre(ci) - truth)
        assert ended < started, f"cam{ci}: {started:.1f} mm -> {ended:.1f} mm"
        assert ended < 10.0


def test_reference_frame_gauge_is_not_moved():
    """The world must stay pinned to the reference plate, or the datum drifts."""
    obs, plate_R, plate_t = make_scene()
    res = bundle_plate_poses(obs, *perturbed_start(plate_R, plate_t), K)
    # the reference frame owns no parameters at all
    assert len(res.plate_rvec) == len(plate_R)
    # and its dots reproject through the identity gauge
    ref = obs.frame < 0
    assert ref.sum() > 0
    assert np.median(res.residual_px[ref]) < 0.6


def test_trimming_removes_an_injected_mislabelled_view():
    obs, plate_R, plate_t = make_scene()
    bad = (obs.cam == 2) & (obs.frame == 3)
    assert bad.sum() > 0
    obs.pix[bad] = obs.pix[bad][::-1]                # scramble that view's labels
    res = bundle_plate_poses(obs, *perturbed_start(plate_R, plate_t), K)
    # most of the scrambled dots are dropped, and the good ones are not
    assert res.keep[bad].mean() < 0.5
    assert res.keep[~bad].mean() > 0.9


def test_vertical_prior_does_not_break_a_vertical_scene():
    obs, plate_R, plate_t = make_scene()
    start = perturbed_start(plate_R, plate_t)
    plain = bundle_plate_poses(obs, *start, K)
    prior = bundle_plate_poses(obs, *start, K, vertical_px=10.0, vertical_sigma_deg=1.0)
    assert np.median(prior.residual_px) < 1.5 * np.median(plain.residual_px)
    tilts = [tilt_off_vertical_deg(rodrigues(r)) for r in prior.plate_rvec]
    assert max(tilts) < 5.0


def test_rodrigues_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        r = rng.normal(0, 1, 3)
        np.testing.assert_allclose(rodrigues(rotvec(rodrigues(r))), rodrigues(r),
                                   atol=1e-9)
    np.testing.assert_allclose(rodrigues(np.zeros(3)), np.eye(3))
    # the near-180 degree branch
    R = rodrigues([0.0, np.pi - 1e-9, 0.0])
    np.testing.assert_allclose(rodrigues(rotvec(R)), R, atol=1e-6)


def test_tilt_off_vertical():
    assert tilt_off_vertical_deg(np.eye(3)) == pytest.approx(0.0)
    # a pure yaw about +Y is still vertical, however large
    assert tilt_off_vertical_deg(rodrigues([0.0, 0.9, 0.0])) == pytest.approx(0.0, abs=1e-9)
    # a roll about +Z is not
    assert tilt_off_vertical_deg(rodrigues([0.0, 0.0, np.radians(7.0)])) == \
        pytest.approx(7.0, abs=1e-6)


def test_agreeing_views_is_per_dot_not_per_centroid():
    base = plate_points()
    # a scramble that preserves the centroid exactly but not the pattern
    scrambled = base[::-1].copy()
    np.testing.assert_allclose(scrambled.mean(0), base.mean(0), atol=1e-9)
    views = {0: base, 1: base + 2.0, 2: scrambled}
    assert sorted(agreeing_views(views, tol_mm=50.0)) == [0, 1]


def test_bundle_result_camera_centre():
    r = BundleResult(np.zeros((1, 3)), np.array([[0.0, 0.0, -5.0]]),
                     np.zeros((0, 3)), np.zeros((0, 3)),
                     np.zeros(0, bool), np.zeros(0))
    np.testing.assert_allclose(r.camera_centre(0), [0.0, 0.0, 5.0])
