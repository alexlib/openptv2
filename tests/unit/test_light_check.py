"""Light check of 3-camera points (openptv2.light_check)."""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.light_check import (
    DEFAULT_Z_MIN,
    R_IN,
    R_RING,
    R_WIN,
    apply_light_check,
    light_check_from_params,
    light_keep,
    light_z,
    missing_view_light,
)

pytestmark = pytest.mark.unit


def render(spots, size=60, sigma=0.9, noise=2.0, seed=0, bg=12.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] + 0.5            # pixel (i, j) is centred at i + 0.5
    img = np.full((size, size), bg)
    for x, y, a in spots:
        img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255)


def light_z_scalar(img, p):
    """The definition, written as a plain loop (one point)."""
    x, y = p
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    ring, inner = [], []
    for j in range(y0 - R_WIN, y0 + R_WIN + 1):
        for i in range(x0 - R_WIN, x0 + R_WIN + 1):
            d = np.hypot(i + 0.5 - x, j + 0.5 - y)
            if d > R_RING:
                ring.append(img[j, i])
            if d <= R_IN:
                inner.append(img[j, i])
    bg = np.median(ring)
    sd = 1.4826 * np.median(np.abs(np.array(ring) - bg)) + 1.0
    return (max(inner) - bg) / sd


def test_params():
    assert light_check_from_params(None) is None
    assert light_check_from_params({}) is None
    assert light_check_from_params({"light_check": False}) is None
    assert light_check_from_params({"light_check": 0}) is None
    assert light_check_from_params({"light_check": True}) == DEFAULT_Z_MIN
    assert light_check_from_params({"light_check": 7}) == 7.0


def test_spot_has_light_and_blank_has_none():
    img = render([(30.3, 25.6, 90.0)])
    z = light_z(img, np.array([[30.3, 25.6], [10.0, 45.0]]))
    assert z[0] > 20 and abs(z[1]) < 6


def test_a_spot_one_pixel_off_still_counts_but_far_off_does_not():
    img = render([(30.3, 25.6, 90.0)])
    z = light_z(img, np.array([[31.2, 25.6], [36.0, 25.6]]))
    assert z[0] > 20 and z[1] < 6


def test_vectorised_matches_the_definition():
    rng = np.random.default_rng(1)
    img = render([(20.2, 20.7, 80.0), (40.6, 33.1, 40.0)], seed=3)
    pts = np.column_stack([rng.uniform(10, 50, 25), rng.uniform(10, 50, 25)])
    z = light_z(img, pts)
    ref = np.array([light_z_scalar(img, p) for p in pts])
    np.testing.assert_allclose(z, ref, rtol=1e-9, atol=1e-9)


def test_outside_or_near_the_edge_or_not_finite_is_nan():
    img = render([])
    z = light_z(img, np.array([[3.0, 30.0], [30.0, 57.0], [-5.0, -5.0], [np.nan, 10.0], [30.0, 30.0]]))
    assert np.isnan(z[:4]).all() and np.isfinite(z[4])


class Cal:                                           # a stand-in camera: the pixel is x, y of the point
    pass


def project(pts, cal, cpar):
    return np.asarray(pts)[:, :2] + getattr(cal, "shift", 0.0)


def scene():
    """4 cameras; camera 3 shows a spot at (30, 30) and (20, 40) but nothing at (45, 15)."""
    spot = render([(30.0, 30.0, 90.0), (20.0, 40.0, 90.0)], seed=5)
    blank = render([], seed=6)
    images = [blank, blank, blank, spot]
    pos = np.array([
        [30.0, 30.0, 0.0],      # 3 cameras, light in camera 3: keep
        [45.0, 15.0, 0.0],      # 3 cameras, no light in camera 3: drop
        [20.0, 40.0, 0.0],      # 4 cameras: keep, never judged
        [30.0, 30.0, 0.0],      # 3 cameras, missing camera 0 (blank) -> no light there: drop
        [2.0, 30.0, 0.0],       # 3 cameras, projection near the edge: keep (nothing known)
        [30.0, 30.0, 0.0],      # 2 cameras: not judged: keep
    ])
    ids = np.array([[3, 4, 5, -1], [3, 4, 5, -1], [1, 2, 3, 4], [-1, 4, 5, 6], [3, 4, 5, -1], [3, 4, -1, -1]])
    return pos, ids, images, [Cal()] * 4


def test_keep_mask_on_a_scene():
    pos, ids, images, cals = scene()
    keep = light_keep(pos, ids, images, cals, None, 10.0, project)
    assert keep.tolist() == [True, False, True, False, True, True]
    z = missing_view_light(pos, ids, images, cals, None, project)
    assert np.isnan(z[[2, 4, 5]]).all() and z[0] > 12 and z[1] < 6


def test_fewer_than_four_cameras_never_filters():
    pos, ids, images, cals = scene()
    keep = light_keep(pos, ids[:, :3], images[:3], cals[:3], None, 10.0, project)
    assert keep.all()


class FakeStore:
    def __init__(self, data, rcm):
        self.data, self.rcm = dict(data), dict(rcm)

    def has_correspondences(self, f):
        return f in self.data

    def read_correspondences(self, f):
        return self.data[f]

    def write_correspondences(self, f, pos, ids):
        self.data[f] = (pos, ids)

    def read_rcm(self, f):
        return self.rcm.get(f)

    def write_rcm(self, f, rcm):
        self.rcm[f] = rcm


def test_store_step_filters_rows_and_the_row_aligned_rcm():
    pos, ids, images, cals = scene()
    store = FakeStore({1: (pos, ids), 2: (pos[[0, 2]], ids[[0, 2]])}, {1: np.arange(6, dtype=np.float32)})
    stats = apply_light_check(store, [1, 2, 3], lambda f: images, cals, None, 10.0, project)
    assert stats == {1: (4, 6), 2: (2, 2)}                       # frame 3 has no correspondences: skipped
    p1, i1 = store.data[1]
    assert len(p1) == 4 and (i1 == ids[[0, 2, 4, 5]]).all()
    assert store.rcm[1].tolist() == [0.0, 2.0, 4.0, 5.0]
    assert len(store.data[2][0]) == 2


def test_empty_frame_is_fine():
    store = FakeStore({1: (np.empty((0, 3)), np.empty((0, 4), int))}, {})
    assert apply_light_check(store, [1], lambda f: [render([])] * 4, [Cal()] * 4, None, 10.0, project) == {1: (0, 0)}


def _run_sequence(tmp_path, name, light_check):
    """py_sequence_loop on two frames of test_cavity with ``ptv.light_check`` set; returns {frame: (pos, ids, rcm)}."""
    import os
    import shutil
    from pathlib import Path

    from openptv2.gui import ptv
    from openptv2.gui.experiment import Experiment
    from openptv2.storage import RunStore

    src = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"
    if not src.exists():
        pytest.skip("test_cavity directory not found")
    work = tmp_path / name
    shutil.copytree(src, work)
    old = os.getcwd()
    os.chdir(work)
    try:
        exp = Experiment()
        exp.exp_path = str(work)
        exp.pm.from_yaml(work / "parameters_Run1.yaml")
        exp.pm.parameters.setdefault("ptv", {})["light_check"] = light_check
        exp.target_filenames = exp.pm.get_target_filenames()
        cpar, spar, vpar, track_par, tpar, cals, epar = ptv.py_start_proc_c(exp.pm)
        exp.cpar, exp.spar, exp.vpar, exp.track_par, exp.tpar, exp.cals = cpar, spar, vpar, track_par, tpar, cals
        spar.set_first(10000)
        spar.set_last(10001)
        ptv.py_sequence_loop(exp)
        store = RunStore.open(work, mode="r")
        out = {}
        for f in (10000, 10001):
            pos, ids = store.read_correspondences(f)
            out[f] = (np.asarray(pos), np.asarray(ids)[:, :4], store.read_rcm(f))
        return out
    finally:
        os.chdir(old)


def test_pipeline_option_drops_only_one_camera_missing_points(tmp_path):
    base = _run_sequence(tmp_path, "base", False)
    drop_all = _run_sequence(tmp_path, "all", 1e9)           # threshold no light can reach: every judged point goes
    for f in base:
        pos, ids, _ = base[f]
        pos2, ids2, rcm2 = drop_all[f]
        judged = ((ids < 0).sum(1) == 1)
        assert judged.any(), "the test dataset has no 3-camera point"
        kept_judged = int(((ids2 < 0).sum(1) == 1).sum())
        assert int((~judged).sum()) == len(pos2) - kept_judged      # every point that is not judged is kept
        assert kept_judged < 0.06 * judged.sum()                    # only the points too near the image edge to judge stay
        assert len(pos2) > 0
        if rcm2 is not None:
            assert len(rcm2) == len(pos2)                          # rcm stays row-aligned
    default = _run_sequence(tmp_path, "default", True)
    for f in base:
        assert len(drop_all[f][0]) <= len(default[f][0]) <= len(base[f][0])


def test_unreadable_images_skip_the_frame_with_one_warning_and_keep_every_point():
    pos, ids, images, cals = scene()
    store = FakeStore({1: (pos, ids), 2: (pos, ids), 3: (pos, ids)}, {})

    def read(f):
        if f == 2:
            raise FileNotFoundError("img/frame_000002.tif does not exist")
        return images

    with pytest.warns(RuntimeWarning, match=r"skipped 1 of 3 frames.*frame 2"):
        stats = apply_light_check(store, [1, 2, 3], read, cals, None, 10.0, project)
    assert stats == {1: (4, 6), 2: (6, 6), 3: (4, 6)}
    assert len(store.data[2][0]) == 6 and len(store.data[1][0]) == 4


def test_no_warning_when_every_image_is_readable():
    import warnings

    pos, ids, images, cals = scene()
    store = FakeStore({1: (pos, ids)}, {})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_light_check(store, [1], lambda f: images, cals, None, 10.0, project)
