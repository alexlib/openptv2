"""Track extraction and membership inference for merge-aware refinement (openptv2.merge_post)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.loco import project_pixels
from openptv2.merge_post import chains_from_linkage, extend_membership
from openptv2.recorrespond import load_parameters

pytestmark = pytest.mark.unit
YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"


def test_chains_follow_links_and_number_every_row():
    prev = {1: np.array([-1, -1]), 2: np.array([0, -1, -1]), 3: np.array([0, 1])}
    nxt = {1: np.array([0, -2]), 2: np.array([0, -2, -2]), 3: np.array([-2, -2])}
    tid = chains_from_linkage(prev, nxt)
    assert tid[1][0] == tid[2][0] == tid[3][0]               # one chain across three frames
    singles = {tid[1][1], tid[2][1], tid[2][2], tid[3][1]}                     # one-sided links: own chains
    assert len(singles) == 4 and tid[1][0] not in singles
    assert all((tid[f] >= 0).all() for f in tid)


def test_chains_bridge_a_gap_where_the_neighbour_reciprocates():
    prev = {1: np.array([-1]), 2: np.array([-1]), 3: np.array([0])}       # row 0 of frame 3 comes from frame 1
    nxt = {1: np.array([0]), 2: np.array([-2]), 3: np.array([-2])}
    tid = chains_from_linkage(prev, nxt)
    assert tid[1][0] == tid[3][0]
    assert tid[2][0] != tid[1][0]


def test_chains_do_not_loop_on_broken_links():
    prev = {1: np.array([-1]), 2: np.array([-1])}
    nxt = {1: np.array([0]), 2: np.array([0])}                           # link to a row that points elsewhere
    tid = chains_from_linkage(prev, nxt)
    assert (tid[1] >= 0).all() and (tid[2] >= 0).all()


@pytest.fixture(scope="module")
def rig():
    cpar, vpar, cals, _ = load_parameters(YAML)
    pts = np.random.default_rng(2).uniform(-8, 8, (40, 3))
    px = np.stack([project_pixels(pts, c, cpar) for c in cals], axis=1)
    return cpar, cals, pts, px


def test_a_point_without_a_view_attaches_to_the_blob_it_falls_on(rig):
    cpar, cals, pts, px = rig
    n = len(pts)
    frame = np.ones(n, int)
    blob = np.tile(np.arange(n)[:, None], (1, 4))
    centre = px.copy()
    sumg = np.full((n, 4), 100.0)
    blob[0, 2] = -1                                           # point 0 has no blob in camera 3 ...
    centre[0, 2] = np.nan
    sumg[0, 2] = np.nan
    cent3 = px[:, 2, :].copy()
    cent3[0] = px[0, 2] + [0.0, 0.0]                          # ... its blob exists but is listed under point 5's number
    cent3 = np.vstack([cent3[1:], px[0, 2] + [0.7, 0.2]])      # a free blob 0.73 px from point 0's projection
    blobs = {(1, c): (px[:, c, :], np.full(n, 100.0)) for c in range(4)}
    blobs[(1, 2)] = (cent3, np.full(n, 100.0))
    b2, c2, s2, n_att = extend_membership(frame, pts, blob, centre, sumg, blobs, cpar, cals, radius=2.0)
    assert n_att >= 1
    assert b2[0, 2] == len(cent3) - 1                         # the free blob next to it
    assert c2[0, 2] == pytest.approx(cent3[-1])
    assert (b2[1:, :] == blob[1:, :]).all() or n_att > 1      # other rows only change if they also lacked a view


def test_no_attachment_beyond_the_radius_or_for_points_with_too_few_cameras(rig):
    cpar, cals, pts, px = rig
    n = len(pts)
    frame = np.ones(n, int)
    blob = np.tile(np.arange(n)[:, None], (1, 4))
    blob[0, 2] = -1
    blob[1, :] = [0, -1, -1, 3]                               # point 1 has only two cameras: not extended
    centre = np.where((blob >= 0)[:, :, None], px, np.nan)
    sumg = np.where(blob >= 0, 100.0, np.nan)
    far = {(1, c): (px[:, c, :] + 50.0, np.full(n, 100.0)) for c in range(4)}    # all blobs 50 px away
    b2, _, _, n_att = extend_membership(frame, pts, blob, centre, sumg, far, cpar, cals, radius=2.0)
    assert n_att == 0 and (b2 == blob).all()
    near = {(1, c): (px[:, c, :], np.full(n, 100.0)) for c in range(4)}
    b3, _, _, n_att3 = extend_membership(frame, pts, blob, centre, sumg, near, cpar, cals, radius=2.0)
    assert b3[1, 1] == -1 and b3[1, 2] == -1                  # two-camera point untouched
    assert b3[0, 2] == 0                                      # three-camera point 0 took the blob under its projection


def _track_with_a_gap(rig, gap_frames=(4,)):
    from openptv2.merge_post import fill_gaps  # noqa: F401

    cpar, cals, pts, px = rig
    p0 = pts[0]
    vel = np.array([0.05, 0.03, 0.0])
    frames = [f for f in range(1, 9) if f not in gap_frames]
    pos = np.array([p0 + vel * f for f in frames])
    blobs = {}
    for f in range(1, 9):
        truth = p0 + vel * f
        for c in range(4):
            blobs[(f, c)] = (project_pixels(truth[None], cals[c], cpar), np.array([100.0]))
    n = len(frames)
    a = {"frame": np.array(frames), "tid": np.zeros(n, int), "pos": pos,
         "blob": np.zeros((n, 4), int), "centre": np.zeros((n, 4, 2)), "sumg": np.full((n, 4), 100.0),
         "blobs": blobs}
    return cpar, cals, a, p0 + vel * gap_frames[0]


def test_a_track_gap_is_filled_from_the_blobs_where_the_path_says(rig):
    from openptv2.merge_post import fill_gaps

    cpar, cals, a, truth_gap = _track_with_a_gap(rig)
    new = fill_gaps(a, a["pos"], cpar, cals, radius=1.5)
    assert len(new["frame"]) == 1 and new["frame"][0] == 4 and new["tid"][0] == 0
    assert (new["blob"][0] == 0).all()                        # a blob in every camera
    assert np.linalg.norm(new["pos"][0] - truth_gap) < 1e-3


def test_gap_filling_needs_blobs_short_gaps_and_enough_cameras(rig):
    from openptv2.merge_post import fill_gaps

    cpar, cals, a, _ = _track_with_a_gap(rig)
    far = dict(a, blobs={k: (v[0] + 30.0, v[1]) for k, v in a["blobs"].items()})
    assert len(fill_gaps(far, a["pos"], cpar, cals, radius=1.5)["frame"]) == 0          # no blob near the path
    cpar, cals, a2, _ = _track_with_a_gap(rig, gap_frames=(3, 4, 5, 6))
    assert len(fill_gaps(a2, a2["pos"], cpar, cals, max_gap=3)["frame"]) == 0           # gap of four frames
    assert len(fill_gaps(a2, a2["pos"], cpar, cals, max_gap=4)["frame"]) == 4
    two = dict(a, blobs={k: v for k, v in a["blobs"].items() if k[1] < 2})                # only two cameras have blobs
    assert len(fill_gaps(two, a["pos"], cpar, cals, min_cams=3)["frame"]) == 0
    assert len(fill_gaps(two, a["pos"], cpar, cals, min_cams=2)["frame"]) == 1
    short = dict(a, frame=a["frame"][:3], tid=a["tid"][:3], pos=a["pos"][:3], blob=a["blob"][:3],
                 centre=a["centre"][:3], sumg=a["sumg"][:3])
    assert len(fill_gaps(short, short["pos"], cpar, cals, min_rows=4)["frame"]) == 0     # too short a track


def test_hermite_interpolation_follows_a_curved_path_better_than_linear(rig):
    from openptv2.merge_post import _hermite

    p = lambda t: np.array([0.1 * t, 0.02 * t**2, 0.0])          # noqa: E731
    v = lambda t: np.array([0.1, 0.04 * t, 0.0])                 # noqa: E731
    got = _hermite(p(2), p(8), v(2), v(8), 3, 6)                  # frame 5 of a gap 2..8
    lin = p(2) + (p(8) - p(2)) * 3 / 6
    assert np.linalg.norm(got - p(5)) < 1e-9                      # exact for a quadratic path
    assert np.linalg.norm(lin - p(5)) > 0.1


def test_a_long_gap_is_bridged_with_growing_radius_and_hermite(rig):
    from openptv2.merge_post import fill_gaps

    cpar, cals, a, truth_gap = _track_with_a_gap(rig, gap_frames=(3, 4, 5, 6, 7))
    # blobs sit on the true (straight) path; ask for the long gap with the Hermite path and a mass check
    new = fill_gaps(a, a["pos"], cpar, cals, radius=1.0, max_gap=6, interp="hermite", min_rows=2,
                    radius_growth=0.2, mass_range=(0.7, 1.3))
    assert sorted(new["frame"].tolist()) == [3, 4, 5, 6, 7]


def test_mass_check_rejects_a_blob_that_is_too_bright_or_too_dim(rig):
    from openptv2.merge_post import fill_gaps

    cpar, cals, a, _ = _track_with_a_gap(rig)
    bright = dict(a, blobs={k: (v[0], np.array([300.0])) for k, v in a["blobs"].items()})   # 3x the signature
    assert len(fill_gaps(bright, a["pos"], cpar, cals, mass_range=(0.7, 1.3))["frame"]) == 0
    assert len(fill_gaps(bright, a["pos"], cpar, cals)["frame"]) == 1                          # no check: accepted
    dim = dict(a, blobs={k: (v[0], np.array([20.0])) for k, v in a["blobs"].items()})
    assert len(fill_gaps(dim, a["pos"], cpar, cals, mass_range=(0.7, 1.3))["frame"]) == 0
    ok = fill_gaps(a, a["pos"], cpar, cals, mass_range=(0.7, 1.3))
    assert len(ok["frame"]) == 1


def test_residual_target_recovers_a_hidden_particle_position():
    from openptv2.merge_post import birth_targets

    # camera 0, one blob holding a tracked particle (signature 100, predicted at (10, 10)) and an unseen one of
    # brightness 50 at (13, 10): the blob centre is the brightness-weighted mean (11, 10), grey sum 150
    frame = np.array([1])
    blob = np.array([[0]])
    pred = np.array([[[10.0, 10.0]]])
    sig = np.array([[100.0]])
    blobs = {(1, 0): (np.array([[11.0, 10.0], [40.0, 40.0]]), np.array([150.0, 90.0]))}
    counts = {(1, 0): np.array([[9, 3, 3], [9, 3, 3]])}
    t = birth_targets(frame, blob, pred, sig, blobs, counts, min_resid=0.3)[(1, 0)]
    assert t["kind"].tolist() == [0, 1]                         # the free blob and one residual target
    assert t["src"].tolist() == [1, 0]
    assert t["xy"][1] == pytest.approx([13.0, 10.0])
    assert t["sumg"][1] == pytest.approx(50.0)


def test_no_residual_target_when_the_mass_is_explained_or_unknown():
    from openptv2.merge_post import birth_targets

    frame = np.array([1])
    blob = np.array([[0]])
    blobs = {(1, 0): (np.array([[11.0, 10.0]]), np.array([105.0]))}
    counts = {(1, 0): np.array([[9, 3, 3]])}
    t = birth_targets(frame, blob, np.array([[[10.0, 10.0]]]), np.array([[100.0]]), blobs, counts, min_resid=0.3)
    assert len(t[(1, 0)]["xy"]) == 0                            # 105 vs 100: nothing hidden
    t2 = birth_targets(frame, blob, np.array([[[np.nan, np.nan]]]), np.array([[100.0]]),
                       {(1, 0): (np.array([[11.0, 10.0]]), np.array([200.0]))}, counts, min_resid=0.3)
    assert len(t2[(1, 0)]["xy"]) == 0                           # no trusted prediction: cannot place it


def test_fallback_position_places_a_hidden_particle_without_a_trusted_prediction():
    from openptv2.merge_post import birth_targets

    frame, blob = np.array([1]), np.array([[0]])
    blobs = {(1, 0): (np.array([[11.0, 10.0]]), np.array([150.0]))}
    counts = {(1, 0): np.array([[9, 3, 3]])}
    no_pred = np.array([[[np.nan, np.nan]]])
    t = birth_targets(frame, blob, no_pred, np.array([[100.0]]), blobs, counts, min_resid=0.3,
                      fallback_px=np.array([[[10.0, 10.0]]]))[(1, 0)]
    assert t["kind"].tolist() == [1] and t["xy"][0] == pytest.approx([13.0, 10.0])
