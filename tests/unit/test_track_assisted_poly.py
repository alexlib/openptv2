"""Unit tests for Step 2c upgrades to openptv2.track_assisted:
- 3-4 point polynomial predictor ("poly" vs "linear")
- 2-of-4 / 3-of-4 camera-subset triangulation
- Optional non-exclusive (shared) 2D target fallback (allow_shared_targets=True)
"""

from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.track import point_to_pixel
from openptv2.storage import RunStore
from openptv2.track_assisted import _predict_position, run_corrective_pass

FIX = Path(__file__).resolve().parents[2] / "test_data" / "tracking_synthetic"


def _load_params():
    yaml_path = FIX / "parameters_Run1.yaml"
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    tpar = TrackPar.from_yaml(str(yaml_path))
    spar = SequencePar.from_yaml(str(yaml_path), cpar.num_cams)
    cals = [
        Calibration.from_file(
            str(FIX / f"cal/cam{c + 1}.tif.ori"),
            str(FIX / f"cal/cam{c + 1}.tif.addpar"),
        )
        for c in range(cpar.num_cams)
    ]
    return cpar, vpar, tpar, spar, cals


def _write_frame(store, frame, num_cams, cals, cpar, positions_for_targets, corr_indices):
    """Write 2D targets for ``positions_for_targets`` and 3D correspondences
    for ``corr_indices`` (indices into ``positions_for_targets``)."""
    for cam in range(num_cams):
        rows = []
        for pnr, pos in enumerate(positions_for_targets):
            if pos is None:
                continue
            px, py = point_to_pixel(pos, cals[cam], cpar)
            rows.append([pnr, px, py, 25, 5, 5, 100, -1])
        store.write_targets(
            cam,
            frame,
            np.asarray(rows, dtype=np.float64).reshape(-1, 8),
        )
    if corr_indices:
        pos_rows = np.asarray([positions_for_targets[i] for i in corr_indices], dtype=np.float64)
        id_rows = np.asarray([[i] * num_cams for i in corr_indices], dtype=np.int32)
    else:
        pos_rows = np.empty((0, 3), dtype=np.float64)
        id_rows = np.empty((0, num_cams), dtype=np.int32)
    store.write_correspondences(frame, pos_rows, id_rows)


def _curved_scene(tmp_path):
    """5-frame scene (10001..10005) with a strongly accelerating particle
    where 2-point linear extrapolation misses radius_px=3.0 at frame 10001,
    while 3-4 point quadratic prediction hits it."""
    cpar, vpar, tpar, spar, cals = _load_params()
    num_cams = cpar.num_cams
    frames = list(range(spar.first, spar.last + 1))
    assert len(frames) == 5

    # Quadratic trajectory: x(s) = 0.0, y(s) = 1.5 * s^2, z(s) = 0.0
    # where s = frame - 10003 in [-2, -1, 0, 1, 2].
    # At frame 10001 (s = -2): true y = 6.0.
    # Linear from 10002 (s=-1, y=1.5) and 10003 (s=0, y=0.0): predicts y = 3.0 (3.0 mm off -> >3 px!).
    # Quadratic from 10002..10005: predicts y = 6.0 (exact!).
    traj = {
        f: np.array([0.0, 1.5 * (f - 10003) ** 2, 0.0], dtype=np.float64)
        for f in frames
    }

    store = RunStore(str(tmp_path / "curved.zarr"), mode="w")
    dropped_frame = frames[0]  # 10001
    for f in frames:
        corr = [] if f == dropped_frame else [0]
        _write_frame(store, f, num_cams, cals, cpar, [traj[f]], corr)

    # Linkage: particle 0 is linked across frames[1:] (10002..10005), absent at frames[0]
    for fi, f in enumerate(frames):
        if f == dropped_frame:
            store.write_linkage(
                f,
                np.empty(0, np.int32),
                np.empty(0, np.int32),
                np.empty((0, 3), np.float64),
                name="ptv_is",
            )
        else:
            prev = np.array([-1 if fi == 1 else 0], dtype=np.int32)
            nxt = np.array([-2 if fi == len(frames) - 1 else 0], dtype=np.int32)
            store.write_linkage(f, prev, nxt, traj[f][None, :], name="ptv_is")

    return {
        "cpar": cpar,
        "vpar": vpar,
        "tpar": tpar,
        "spar": spar,
        "cals": cals,
        "store": store,
        "traj": traj,
        "dropped_frame": dropped_frame,
    }


def test_predict_position_polynomial_exactness():
    # Quadratic y(k) = 2 - 3k + 1.5 k^2, evaluated at k = -1 -> 2 + 3 + 1.5 = 6.5
    pts = [np.array([0.0, 2.0 - 3.0 * k + 1.5 * k * k, 1.0]) for k in range(4)]
    pred_3 = _predict_position(pts[:3], predictor="poly")
    pred_4 = _predict_position(pts[:4], predictor="poly")
    expected = np.array([0.0, 6.5, 1.0])
    np.testing.assert_allclose(pred_3, expected, atol=1e-12)
    np.testing.assert_allclose(pred_4, expected, atol=1e-12)

    pred_lin = _predict_position(pts[:4], predictor="linear")
    np.testing.assert_allclose(pred_lin, 2.0 * pts[0] - pts[1], atol=1e-12)


def test_poly_predictor_recovers_curved_trajectory_where_linear_misses(tmp_path):
    # First verify linear misses at radius_px=3.0
    lin_scene = _curved_scene(tmp_path / "lin")
    stats_lin = run_corrective_pass(
        lin_scene["cpar"],
        lin_scene["vpar"],
        lin_scene["tpar"],
        lin_scene["spar"],
        lin_scene["cals"],
        lin_scene["store"],
        linkage_name="ptv_is",
        max_passes=1,
        predictor="linear",
        radius_px=3.0,
    )
    assert stats_lin.claimed_total == 0

    # Now verify poly (default) hits and recovers the curved particle
    poly_scene = _curved_scene(tmp_path / "poly")
    stats_poly = run_corrective_pass(
        poly_scene["cpar"],
        poly_scene["vpar"],
        poly_scene["tpar"],
        poly_scene["spar"],
        poly_scene["cals"],
        poly_scene["store"],
        linkage_name="ptv_is",
        max_passes=1,
        predictor="poly",
        radius_px=3.0,
    )
    assert stats_poly.claimed_total == 1
    pos, _ids = poly_scene["store"].read_correspondences(poly_scene["dropped_frame"])
    assert pos.shape[0] == 1
    true_pos = poly_scene["traj"][poly_scene["dropped_frame"]]
    assert np.linalg.norm(pos[0] - true_pos) < 1e-3


@pytest.mark.parametrize("visible_cams", [(0, 1), (0, 1, 2)])
def test_camera_subset_triangulation_2of4_and_3of4(tmp_path, visible_cams):
    cpar, vpar, tpar, spar, cals = _load_params()
    num_cams = cpar.num_cams
    frames = list(range(spar.first, spar.last + 1))
    dropped_frame = frames[0]
    traj = {f: np.array([0.5 * (f - 10003), 0.2, 0.0], dtype=np.float64) for f in frames}

    store = RunStore(str(tmp_path / "subset.zarr"), mode="w")
    for fi, f in enumerate(frames):
        if f == dropped_frame:
            for cam in range(num_cams):
                if cam in visible_cams:
                    px, py = point_to_pixel(traj[f], cals[cam], cpar)
                    rows = np.array([[0, px, py, 25, 5, 5, 100, -1]], dtype=np.float64)
                else:
                    rows = np.empty((0, 8), dtype=np.float64)
                store.write_targets(cam, f, rows)
            store.write_correspondences(
                f, np.empty((0, 3), np.float64), np.empty((0, num_cams), np.int32)
            )
            store.write_linkage(
                f,
                np.empty(0, np.int32),
                np.empty(0, np.int32),
                np.empty((0, 3), np.float64),
                name="ptv_is",
            )
        else:
            _write_frame(store, f, num_cams, cals, cpar, [traj[f]], [0])
            prev = np.array([-1 if fi == 1 else 0], dtype=np.int32)
            nxt = np.array([-2 if fi == len(frames) - 1 else 0], dtype=np.int32)
            store.write_linkage(f, prev, nxt, traj[f][None, :], name="ptv_is")

    stats = run_corrective_pass(
        cpar,
        vpar,
        tpar,
        spar,
        cals,
        store,
        linkage_name="ptv_is",
        max_passes=1,
    )
    assert stats.claimed_total == 1
    if len(visible_cams) == 2:
        assert stats.claimed_2cam == 1
        assert stats.claimed_3cam == 0
    else:
        assert stats.claimed_3cam == 1
        assert stats.claimed_2cam == 0

    pos, ids = store.read_correspondences(dropped_frame)
    assert pos.shape[0] == 1
    assert np.linalg.norm(pos[0] - traj[dropped_frame]) < 1e-3
    for cam in range(num_cams):
        if cam in visible_cams:
            assert ids[0, cam] == 0
        else:
            assert ids[0, cam] == -1


def test_shared_target_fallback_recovers_occluded_particle_without_deleting_owner(tmp_path):
    """Particle A is linked across all 5 frames and owns target 0 in all 4 cams.
    Particle B is missing at frame 10001; at frame 10001 its projections in
    cams 0 and 1 coincide with A's claimed targets (merged blob), cam 2 has a
    free target for B, and cam 3 is occluded.

    - allow_shared_targets=False: only 1 free cam (cam 2) -> cannot recover B.
    - allow_shared_targets=True: 1 free cam + 2 shared cams -> recovers B,
      records claimed_shared == 1, and preserves Particle A's track intact.
    """
    cpar, vpar, tpar, spar, cals = _load_params()
    num_cams = cpar.num_cams
    frames = list(range(spar.first, spar.last + 1))
    dropped_frame = frames[0]

    def build_store(sub_dir):
        store = RunStore(str(sub_dir / "shared.zarr"), mode="w")
        traj_a = {f: np.array([0.0, 0.0, 0.0], dtype=np.float64) for f in frames}
        # B moves linearly toward A's position at dropped_frame, except offset slightly
        # so at frames 10002..10005 it is distinct, and at dropped_frame its true
        # position is at (0.0, 0.0, 0.0) in cams 0, 1 (merged) while cam 2 sees
        # its true target at (0.0, 0.0, 0.0) -> let's place B slightly offset in 3D
        # (0.05 mm) so its 2D projection in cams 0, 1 is < 1 px from A's target!
        pos_b_drop = np.array([0.05, 0.0, 0.0], dtype=np.float64)
        traj_b = {
            f: pos_b_drop + np.array([0.0, 1.0 * (f - dropped_frame), 0.0])
            for f in frames
        }

        for fi, f in enumerate(frames):
            if f == dropped_frame:
                # Cam 0, 1: only A's target (claimed by A)
                # Cam 2: A's target (0, claimed by A) and B's free target (1)
                # Cam 3: occluded (no targets at all)
                for cam in range(num_cams):
                    rows = []
                    if cam != 3:
                        px_a, py_a = point_to_pixel(traj_a[f], cals[cam], cpar)
                        rows.append([0, px_a, py_a, 25, 5, 5, 100, -1])
                    if cam == 2:
                        px_b, py_b = point_to_pixel(traj_b[f], cals[cam], cpar)
                        rows.append([1, px_b, py_b, 25, 5, 5, 100, -1])
                    store.write_targets(
                        cam, f, np.asarray(rows, dtype=np.float64).reshape(-1, 8)
                    )
                a_ids = np.array([[0, 0, 0, -1]], dtype=np.int32)
                store.write_correspondences(f, traj_a[f][None, :], a_ids)
                store.write_linkage(
                    f,
                    np.array([-1], dtype=np.int32),
                    np.array([0], dtype=np.int32),
                    traj_a[f][None, :],
                    name="ptv_is",
                )
            else:
                _write_frame(store, f, num_cams, cals, cpar, [traj_a[f], traj_b[f]], [0, 1])
                prev = np.array([0, -1 if fi == 1 else 1], dtype=np.int32)
                nxt = np.array([0, -2 if fi == len(frames) - 1 else 1], dtype=np.int32)
                xyz = np.vstack([traj_a[f], traj_b[f]])
                store.write_linkage(f, prev, nxt, xyz, name="ptv_is")
        return store, traj_a, traj_b

    store_off, _ta, _tb = build_store(tmp_path / "off")
    stats_off = run_corrective_pass(
        cpar,
        vpar,
        tpar,
        spar,
        cals,
        store_off,
        linkage_name="ptv_is",
        max_passes=1,
        allow_shared_targets=False,
    )
    assert stats_off.claimed_total == 0

    store_on, traj_a, traj_b = build_store(tmp_path / "on")
    stats_on = run_corrective_pass(
        cpar,
        vpar,
        tpar,
        spar,
        cals,
        store_on,
        linkage_name="ptv_is",
        max_passes=1,
        allow_shared_targets=True,
    )
    assert stats_on.claimed_total == 1
    assert stats_on.claimed_shared == 1

    pos, ids = store_on.read_correspondences(dropped_frame)
    assert pos.shape[0] == 2
    # Particle A (row 0) is untouched
    np.testing.assert_allclose(pos[0], traj_a[dropped_frame], atol=1e-6)
    # Particle B (row 1) shares target 0 in cams 0, 1 and uses free target 1 in cam 2
    assert ids[1, 0] == 0
    assert ids[1, 1] == 0
    assert ids[1, 2] == 1
    assert ids[1, 3] == -1
    assert np.linalg.norm(pos[1] - traj_b[dropped_frame]) < 0.2
