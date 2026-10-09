"""Multi-plane plate calibration steps on a synthetic rig with known truth.

Four pinhole cameras watch a 6x7 plate (120 mm pitch, datum at grid (2,3)) held
vertical at several yaws and depths -- the Illmenau geometry, without its noise.
Every step must recover the truth exactly, and every gate must catch the
labelling failure it exists for.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.plate_bundle import bundle_plate_poses
from openptv2.plate_multiplane import (
    PlateGrid,
    datum_index_from_complete_view,
    detect_coded_plate,
    epipolar_curve_misses,
    epipolar_horizon_z,
    label_plate_view,
    load_plate_views,
    plate_frame_metrics,
    rigid_fit,
    save_plate_views,
    sight_ray,
    triangulate_plate,
)

GRID = PlateGrid(nx=6, ny=7, pitch_x=120.0, pitch_y=120.0, datum_ix=2, datum_iy=3)
IMX, IMY, PIX, CC = 2560, 2048, 0.005, 8.5
K = np.array([[CC / PIX, 0, IMX / 2], [0, CC / PIX, IMY / 2], [0, 0, 1.0]])
CENTRES = [
    (1470, 137, 3060),
    (1482, 2253, 3000),
    (-1472, 137, 2968),
    (-1500, 2254, 3007),
]
# (yaw deg, translation) of the plate per frame; frame "00" is the reference
POSES = {
    "00": (0.0, (0.0, 0.0, 0.0)),
    "01": (20.0, (300.0, 200.0, -900.0)),
    "02": (-25.0, (-400.0, 600.0, -1500.0)),
    "03": (10.0, (100.0, -300.0, 600.0)),
    "04": (-10.0, (500.0, 900.0, -400.0)),
    "05": (30.0, (-200.0, 300.0, 300.0)),
}
CODED = [(2, 3), (2, 4), (4, 3)]  # L corner, +1 pitch along Y, +2 pitch along X


def look_at(C):
    """World -> OpenCV camera (R, t) for a camera at C looking at the origin."""
    C = np.asarray(C, float)
    f = -C / np.linalg.norm(C)
    r = np.cross(f, [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])
    return R, -R @ C


CAMS = [look_at(c) for c in CENTRES]


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def world_points(frame, ids):
    yaw, t = POSES[frame]
    return GRID.object_points(ids) @ rot_y(yaw).T + np.asarray(t)


def project(ci, X):
    R, t = CAMS[ci]
    Xc = X @ R.T + t
    return (Xc[:, :2] / Xc[:, 2:]) * K[0, 0] + K[:2, 2]


def make_views():
    ids = np.arange(1, GRID.n_points + 1)
    return {
        (ci, fr): (ids, project(ci, world_points(fr, ids)))
        for ci in range(len(CAMS))
        for fr in POSES
    }


def control_par(ncam=4):
    return ControlPar(
        num_cams=ncam,
        imx=IMX,
        imy=IMY,
        pix_x=PIX,
        pix_y=PIX,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""] * ncam,
        cal_img_base_name=[""] * ncam,
    )


def true_cals():
    pytest.importorskip("cv2")
    import cv2

    from openptv2.plate_multiplane import pinhole_calibration

    return [
        pinhole_calibration(K, cv2.Rodrigues(R)[0], t, imx=IMX, imy=IMY, pix_mm=PIX)
        for R, t in CAMS
    ]


# ------------------------------------------------------------------ the plate


def test_object_points_put_the_datum_at_the_origin():
    assert np.allclose(GRID.object_points([GRID.datum_id]), 0.0)
    assert GRID.datum_id == 3 * 6 + 2 + 1
    assert np.allclose(GRID.object_points([1]), [[-240.0, -360.0, 0.0]])
    assert np.allclose(GRID.object_points([42]), [[360.0, 360.0, 0.0]])
    assert GRID.ids_from_index(5, 6) == 42


def test_calibration_block_lines():
    lines = GRID.calibration_block_lines()
    assert len(lines) == 42
    assert lines[GRID.datum_id - 1] == f"{GRID.datum_id} 0.0 0.0 0.0"
    assert lines[0] == "1 -240.0 -360.0 0.0"


def test_from_yaml(tmp_path):
    p = tmp_path / "plate.yaml"
    p.write_text(
        "plate:\n  nx: 6\n  ny: 7\n  pitch_x: 120.0\n  pitch_y: 110.0\n"
        "  datum:\n    ix: 2\n    iy: 3\n"
    )
    assert PlateGrid.from_yaml(p) == PlateGrid(6, 7, 120.0, 110.0, 2, 3)


def test_neighbour_pitches_do_not_wrap_rows():
    ids = np.array([5, 6, 7, 12])  # 6 ends row 0, 7 starts row 1
    pos = GRID.object_points(ids)
    dx, dy = GRID.neighbour_pitches(pos, ids)
    assert np.allclose(dx, [120.0])  # 5-6 only, never 6-7
    assert np.allclose(dy, [120.0])  # 6-12


def test_plate_views_cache_round_trip(tmp_path):
    views = {(0, "00000000"): (np.arange(1, 4), np.random.rand(3, 2))}
    save_plate_views(tmp_path / "v.npz", views)
    back = load_plate_views(tmp_path / "v.npz")
    assert back.keys() == views.keys()
    assert np.array_equal(back[(0, "00000000")][0], views[(0, "00000000")][0])
    assert np.allclose(back[(0, "00000000")][1], views[(0, "00000000")][1])


# ------------------------------------------------- detection and labelling


def _detection(ci=0, frame="00"):
    ids = np.arange(1, GRID.n_points + 1)
    px = project(ci, world_points(frame, ids))
    coded = np.isin(ids, [GRID.ids_from_index(ix, iy) for ix, iy in CODED])
    return SimpleNamespace(centroids=px, coded_mask=coded)


def test_label_plate_view_recovers_ids():
    det = _detection(1, "01")
    order = np.random.default_rng(0).permutation(len(det.centroids))
    det = SimpleNamespace(
        centroids=det.centroids[order], coded_mask=det.coded_mask[order]
    )
    ids, px = label_plate_view(det, GRID)
    truth = np.arange(1, GRID.n_points + 1)[order]
    lookup = dict(zip(map(tuple, det.centroids), truth))
    assert len(ids) == GRID.n_points
    assert all(lookup[tuple(p)] == i for i, p in zip(ids, px))


def test_label_plate_view_keeps_the_datum_when_a_column_is_missing():
    """Without the datum, a view missing column 0 would be labelled one pitch off."""
    det = _detection(0, "00")
    keep = (np.arange(GRID.n_points) % GRID.nx) != 0
    det = SimpleNamespace(
        centroids=det.centroids[keep], coded_mask=det.coded_mask[keep]
    )
    ids, _ = label_plate_view(det, GRID)
    assert sorted(ids) == sorted(np.arange(1, GRID.n_points + 1)[keep])


def test_datum_index_from_complete_view():
    assert datum_index_from_complete_view(_detection(2, "03"), GRID) == (2, 3)
    det = _detection(2, "03")
    partial = SimpleNamespace(
        centroids=det.centroids[1:], coded_mask=det.coded_mask[1:]
    )
    assert datum_index_from_complete_view(partial, GRID) is None


def test_detect_coded_plate_searches_for_the_coded_threshold(monkeypatch):
    import openptv2.detect_plate as dp

    tried = []

    def fake(image, tpar, cpar, cam=0, *, coded_thr=40.0, **kw):
        tried.append(coded_thr)
        n = {30.0: 1, 25.0: 2, 20.0: 3}.get(coded_thr, 5)
        return SimpleNamespace(coded_mask=np.arange(10) < n)

    monkeypatch.setattr(dp, "detect_plate_targets", fake)
    r = detect_coded_plate(None, None, None, 0)
    assert int(r.coded_mask.sum()) == 3 and tried == [30.0, 25.0, 20.0]
    assert detect_coded_plate(None, None, None, 0, coded_thresholds=(30, 10)) is None


# ------------------------------------------------- PnP, cc and the pinhole


def test_pnp_pose_recovers_the_reference_pose():
    pytest.importorskip("cv2")
    from openptv2.plate_bundle import rodrigues
    from openptv2.plate_multiplane import pnp_pose

    ids, px = make_views()[(1, "00")]
    rv, tv, rms = pnp_pose(GRID.object_points(ids), px, K)
    assert rms < 1e-6
    assert np.allclose(rodrigues(rv), CAMS[1][0], atol=1e-8)
    assert np.allclose(tv, CAMS[1][1], atol=1e-5)
    assert pnp_pose(GRID.object_points(ids[:5]), px[:5], K) is None


def test_fit_shared_cc_recovers_the_focal_length():
    pytest.importorskip("cv2")
    from openptv2.plate_multiplane import fit_shared_cc

    fit = fit_shared_cc(
        make_views(),
        GRID,
        "00",
        4,
        np.arange(7.0, 10.01, 0.25),
        imx=IMX,
        imy=IMY,
        pix_mm=PIX,
    )
    assert abs(fit.cc - CC) < 1e-3
    assert fit.spread_mm < 0.01
    assert len(fit.rows) == len(POSES) - 1
    # the curve has its minimum at the truth, not a flat valley
    scan = {round(c, 2): s for c, _, s in fit.scan}
    assert scan[8.5] < scan[8.0] and scan[8.5] < scan[9.0]


def test_pinhole_calibration_reprojects_like_opencv():
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.trafo import metric_to_pixel

    cals, cpar = true_cals(), control_par()
    X = world_points("02", np.arange(1, 43))
    for ci, cal in enumerate(cals):
        got = np.array([metric_to_pixel(*img_coord(p, cal, cpar.mm), cpar) for p in X])
        assert np.abs(got - project(ci, X)).max() < 1e-6


# ------------------------------------------------------------ the bundle


def test_prepare_bundle_gates_and_the_bundle_recovers_the_rig():
    pytest.importorskip("cv2")
    from openptv2.plate_multiplane import prepare_bundle

    views = make_views()
    # Gate 2: the L resolved to the wrong dot, grid turned 90 deg in its plane
    # (on the 6x6 block that such a turn maps onto itself).  Fits perfectly.
    ids, px = views[(1, "05")]
    ix, iy = (ids - 1) % GRID.nx, (ids - 1) // GRID.nx
    block = iy < 6
    views[(1, "05")] = (GRID.ids_from_index(5 - iy[block], ix[block]), px[block])
    # Gate 2 too: a 180 deg relabelling (numbered from the wrong corner) implies
    # an upside-down plate.  Still exactly vertical, but a plate is never upside
    # down.  Fits perfectly.
    ids, px = views[(2, "03")]
    views[(2, "03")] = (GRID.n_points + 1 - ids, px)
    # Gate 3: a one-pitch shift along X puts an upright plate in the wrong
    # place.  Fits perfectly too.
    ids, px = views[(3, "04")]
    inner = (ids - 1) % GRID.nx < GRID.nx - 1
    views[(3, "04")] = (ids[inner] + 1, px[inner])

    setup = prepare_bundle(views, GRID, K, 4, "00")
    assert setup.n_pnp == 4 * len(POSES)  # gate 1 sees nothing wrong
    rejects = {(fr, ci): t for fr, ci, t in setup.tilt_rejects}
    assert set(rejects) == {("03", 2), ("05", 1)}
    assert abs(rejects[("03", 2)] - 180.0) < 1e-6
    assert [(fr, cams) for fr, cams, _ in setup.dropped] == [("04", [3])]
    assert all(k not in setup.poses for k in [(1, "05"), (2, "03"), (3, "04")])
    assert setup.free == ["01", "02", "03", "04", "05"]

    res = bundle_plate_poses(
        setup.obs,
        setup.cam_rvec0,
        setup.cam_tvec0,
        setup.plate_rvec0,
        setup.plate_tvec0,
        K,
    )
    for ci, C in enumerate(CENTRES):
        assert np.linalg.norm(res.camera_centre(ci) - C) < 1e-3


def test_prepare_bundle_needs_a_clean_reference():
    pytest.importorskip("cv2")
    from openptv2.plate_multiplane import prepare_bundle

    views = make_views()
    del views[(1, "00")]
    with pytest.raises(ValueError, match="reference frame 00"):
        prepare_bundle(views, GRID, K, 4, "00")


# ------------------------------------------------------------ the checks


def test_triangulation_metrics_on_an_exact_plate():
    cals, cpar = true_cals(), control_par()
    views = make_views()
    tri = triangulate_plate({ci: views[(ci, "02")] for ci in range(4)}, cals, cpar)
    assert len(tri.ids) == 42 and (tri.ncam == 4).all()
    assert np.abs(tri.pos - world_points("02", tri.ids)).max() < 1e-6
    assert tri.rcm.max() < 1e-6
    m = plate_frame_metrics(tri.pos, tri.ids, GRID)
    assert m.planarity_rms < 1e-6
    assert np.allclose(m.pitch_x, 120.0) and np.allclose(m.pitch_y, 120.0)
    assert m.grid_dev.max() < 1e-6
    assert abs(abs(m.normal @ (rot_y(-25.0) @ [0, 0, 1.0])) - 1) < 1e-9


def test_reference_frame_matches_the_block_without_alignment():
    cals, cpar = true_cals(), control_par()
    views = make_views()
    tri = triangulate_plate({ci: views[(ci, "00")] for ci in range(4)}, cals, cpar)
    m = plate_frame_metrics(tri.pos, tri.ids, GRID)
    assert np.abs(m.nominal_error).max() < 1e-6


def test_a_swapped_label_shows_in_rcm_and_grid_deviation():
    cals, cpar = true_cals(), control_par()
    views = make_views()
    per = {ci: views[(ci, "01")] for ci in range(4)}
    ids, px = per[0]
    ids = ids.copy()
    ids[[0, 7]] = ids[[7, 0]]  # camera 0 swaps two dots
    per[0] = (ids, px)
    tri = triangulate_plate(per, cals, cpar)
    bad = np.isin(tri.ids, [1, 8])
    assert tri.rcm[bad].min() > 10 * max(tri.rcm[~bad].max(), 1e-9)
    m = plate_frame_metrics(tri.pos, tri.ids, GRID)
    assert m.grid_dev[bad].min() > m.grid_dev[~bad].max()


def test_rigid_fit_does_not_scale():
    A = GRID.object_points(np.arange(1, 43))
    B = 1.01 * A @ rot_y(30).T + [5.0, 6.0, 7.0]
    fit = rigid_fit(A, B)
    assert np.allclose(np.linalg.norm(fit[1] - fit[0]), 120.0)


def test_epipolar_curve_misses_are_zero_for_an_exact_rig():
    cals, cpar = true_cals(), control_par()
    views = make_views()
    det = [dict(zip(*map(lambda a: a.tolist(), views[(ci, "00")]))) for ci in range(4)]
    misses, monotone = epipolar_curve_misses(
        det[0], det[3], cals[0], cals[3], cpar, np.linspace(-1500, 1500, 301)
    )
    assert len(misses) == 42 and monotone == 42
    assert max(misses) < 0.5  # sampling step, not model error


def test_epipolar_horizon_is_on_the_principal_plane():
    cals, cpar = true_cals(), control_par()
    ids, px = make_views()[(0, "00")]
    pos, v = sight_ray(px[0], cals[0], cpar)
    z = epipolar_horizon_z(pos, v, cals[2])
    P = pos + ((z - pos[2]) / v[2]) * v
    C = np.array(CENTRES[2], float)
    axis = -np.asarray(cals[2].ext_par.dm)[:, 2]
    assert abs((P - C) @ axis) < 1e-6
