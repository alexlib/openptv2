"""Merged-blob refinement (openptv2.merge_refine) on a synthetic crossing in the small cavity rig."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.loco import project_pixels
from openptv2.merge_refine import refine_merged, shared_blobs
from openptv2.recorrespond import load_parameters

pytestmark = pytest.mark.unit
YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"
N_FR = 21
MERGED = range(7, 14)             # frames in which the two particles share one blob in camera 0


@pytest.fixture(scope="module")
def scene():
    cpar, vpar, cals, _ = load_parameters(YAML)
    t = np.arange(N_FR, dtype=float)
    base = np.array([0.0, 0.0, 0.0])
    sep = np.array([0.0, 0.0, 0.0])
    # two particles, a few px apart in camera 0 (offset chosen from the camera's pixel scale)
    probe = project_pixels(np.array([base, base + [1.0, 0, 0]]), cals[0], cpar)
    scale = np.linalg.norm(probe[1] - probe[0])
    sep[0] = 3.0 / scale                                           # 3 px in camera 0
    vel1, vel2 = np.array([0.05, 0.02, 0.0]), np.array([0.04, 0.025, 0.01])
    p1 = base + t[:, None] * vel1
    p2 = base + sep + t[:, None] * vel2
    rng = np.random.default_rng(7)
    rows = []                                                      # (frame, pid, blob[4], centre[4,2], sumg[4])
    bright = {1: np.array([300.0, 310, 290, 305]), 2: np.array([150.0, 160, 140, 155])}
    for f in range(N_FR):
        px1 = np.stack([project_pixels(p1[f:f + 1], c, cpar)[0] for c in cals])
        px2 = np.stack([project_pixels(p2[f:f + 1], c, cpar)[0] for c in cals])
        for pid, px, other in ((1, px1, px2), (2, px2, px1)):
            blob = np.array([2 * f + (pid - 1) + 0 * c + 100 * c for c in range(4)])
            centre = px + rng.normal(0, 0.1, px.shape)
            sumg = bright[pid].copy()
            if f in MERGED:                                        # camera 0: one blob, brightness-weighted centre
                blob[0] = 2 * f
                w1, w2 = bright[1][0], bright[2][0]
                mean = (w1 * px1[0] + w2 * px2[0]) / (w1 + w2) + rng.normal(0, 0.1, 2)
                centre[0] = mean
                sumg[0] = w1 + w2
            rows.append((f, pid, blob, centre, sumg))
    frame = np.array([r[0] for r in rows])
    pid = np.array([r[1] for r in rows])
    blob = np.array([r[2] for r in rows])
    centre = np.array([r[3] for r in rows])
    sumg = np.array([r[4] for r in rows])
    truth = np.where((pid == 1)[:, None], p1[frame], p2[frame])
    return cpar, vpar, cals, frame, pid, blob, centre, sumg, truth


def test_shared_blobs_counts_rows_in_the_same_blob():
    frame = np.array([0, 0, 0, 1])
    blob = np.array([[5, 1], [5, 2], [6, 3], [5, 4]])
    s = shared_blobs(frame, blob)
    assert s[:, 0].tolist() == [2, 2, 1, 1]       # frame 1 uses blob 5 alone: a different frame
    assert s[:, 1].tolist() == [1, 1, 1, 1]
    assert shared_blobs(frame, np.full((4, 2), -1)).sum() == 0


def test_refinement_improves_merged_frames_and_leaves_clean_ones(scene):
    cpar, vpar, cals, frame, pid, blob, centre, sumg, truth = scene
    r = refine_merged(frame, pid, blob, centre, sumg, cpar, cals, vpar, rounds=3, window=9, order=2)
    merged_rows = np.isin(frame, list(MERGED))
    raw_err = np.linalg.norm(r.x_rounds[0] - truth, axis=1)
    new_err = np.linalg.norm(r.x - truth, axis=1)
    assert np.all(np.isfinite(r.x))
    assert np.sqrt(np.mean(new_err[merged_rows] ** 2)) < 0.6 * np.sqrt(np.mean(raw_err[merged_rows] ** 2))
    clean_rows = ~merged_rows
    assert np.sqrt(np.mean(new_err[clean_rows] ** 2)) < 1.5 * np.sqrt(np.mean(raw_err[clean_rows] ** 2)) + 1e-3
    # the output says which observations are shared and how certain they are
    assert (r.n_share[merged_rows, 0] == 2).all() and (r.n_share[clean_rows, 0] == 1).all()
    assert np.nanmax(r.obs_var[clean_rows, 1]) <= 0.05           # clean blobs keep the small variance
    bright = merged_rows & (pid == 1)
    dim = merged_rows & (pid == 2)
    assert np.nanmax(r.obs_var[clean_rows, 1]) < np.nanmin(r.obs_var[bright, 0])   # shared: less certain than clean
    assert np.nanmin(r.obs_var[dim, 0]) > 3 * np.nanmax(r.obs_var[bright, 0])      # the dim member most of all


def test_prediction_is_only_trusted_where_clean_neighbours_exist(scene):
    cpar, vpar, cals, frame, pid, blob, centre, sumg, truth = scene
    r = refine_merged(frame, pid, blob, centre, sumg, cpar, cals, vpar, rounds=1)
    assert np.isfinite(r.pred_px[:, 1, 0]).mean() > 0.9           # clean cameras: paths exist
    # no clean data at all -> nothing is trusted
    blob_all = blob.copy()
    blob_all[:, :] = np.where(pid[:, None] == 1, blob[:, :], blob[:, :])
    one_blob = np.zeros_like(blob)                                # every row in every camera in one blob
    one_blob += frame[:, None] * 0
    r2 = refine_merged(frame, pid, one_blob, centre, sumg, cpar, cals, vpar, rounds=1)
    assert not np.isfinite(r2.pred_px).any()


def test_a_lone_blob_much_brighter_than_its_particle_is_down_weighted(scene):
    cpar, vpar, cals, frame, pid, blob, centre, sumg, truth = scene
    # particle 2 only, and in frames 7-13 its camera-0 blob hides an unseen particle: twice as bright, shifted
    keep = pid == 2
    f2, p2, b2, c2, s2, t2 = frame[keep], pid[keep], blob[keep].copy(), centre[keep].copy(), sumg[keep].copy(), truth[keep]
    hid = np.isin(f2, list(MERGED))
    c2[hid, 0] += [1.5, 0.0]
    s2[hid, 0] *= 2.2
    r_on = refine_merged(f2, p2, b2, c2, s2, cpar, cals, vpar, rounds=2, hidden_mass=1.4)
    r_off = refine_merged(f2, p2, b2, c2, s2, cpar, cals, vpar, rounds=2, hidden_mass=None)
    assert r_on.info[-1]["hidden_partner_obs"] == hid.sum()
    assert r_off.info[-1]["hidden_partner_obs"] == 0
    assert np.nanmin(r_on.obs_var[hid, 0]) > 10 * np.nanmax(r_on.obs_var[~hid, 0])
    e_on = np.linalg.norm(r_on.x[hid] - t2[hid], axis=1).mean()
    e_off = np.linalg.norm(r_off.x[hid] - t2[hid], axis=1).mean()
    assert e_on < 0.5 * e_off


def test_joint_mode_solves_the_merged_frames_at_least_as_well(scene):
    cpar, vpar, cals, frame, pid, blob, centre, sumg, truth = scene
    merged_rows = np.isin(frame, list(MERGED))
    eff = refine_merged(frame, pid, blob, centre, sumg, cpar, cals, vpar, rounds=3)
    joint = refine_merged(frame, pid, blob, centre, sumg, cpar, cals, vpar, rounds=3, mode="joint")
    raw = np.linalg.norm(joint.x_rounds[0] - truth, axis=1)[merged_rows].mean()
    e_eff = np.linalg.norm(eff.x - truth, axis=1)[merged_rows].mean()
    e_joint = np.linalg.norm(joint.x - truth, axis=1)[merged_rows].mean()
    assert e_joint < 0.6 * raw
    assert e_joint <= 1.1 * e_eff


@pytest.mark.parametrize("m", [2, 3, 4])
def test_batched_split_equals_the_one_blob_function(m):
    from openptv2.blob_model import split_blob_effective, split_blobs_effective

    rng = np.random.default_rng(m)
    g = 40
    ctr = rng.normal(size=(g, 2)) * 100
    pred = ctr[:, None, :] + rng.normal(size=(g, m, 2))
    bright = rng.uniform(50, 500, size=(g, m))
    bright[0, 0] = np.nan            # unknown signature -> median of the others
    bright[1] = np.nan               # all unknown -> equal shares
    bright[2, 1] = -3.0              # non-positive counts as unknown
    var_p = rng.uniform(0.05, 1.0, size=(g, m))
    s, v = split_blobs_effective(ctr, pred, bright, 0.01, var_p)
    for k in range(g):
        s1, v1 = split_blob_effective(ctr[k], pred[k], bright[k], 0.01, var_p[k])
        np.testing.assert_allclose(s[k], s1, rtol=1e-12)
        np.testing.assert_allclose(v[k], v1, rtol=1e-12)


@pytest.mark.parametrize("q", [0.5, 0.25])
def test_signatures_equal_the_per_particle_quantile(q):
    from openptv2.blob_model import signature_from_history
    from openptv2.merge_refine import _signatures

    rng = np.random.default_rng(5)
    n, n_cams = 400, 3
    pid = rng.integers(0, 40, n)
    sumg = rng.uniform(20, 300, (n, n_cams))
    sumg[rng.random((n, n_cams)) < 0.1] = np.nan
    sumg[rng.random((n, n_cams)) < 0.05] = -1.0
    clean = rng.random((n, n_cams)) < 0.7
    got = _signatures(pid, clean, sumg, q)
    want = {}
    for c in range(n_cams):
        for p_ in np.unique(pid[clean[:, c]]):
            sel = (pid == p_) & clean[:, c]
            v = signature_from_history(sumg[sel, c], np.ones(sel.sum(), bool), q)
            if np.isfinite(v):
                want[(int(p_), c)] = v
    assert got.keys() == want.keys()
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-12)
