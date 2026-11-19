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
