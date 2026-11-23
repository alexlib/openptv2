"""The shared-blob second pass of the correspondence stage (docs/plans/2026-11-21-tracker-setup.md, section 14)."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from openptv2.loco import project_pixels
from openptv2.recorrespond import load_parameters
from openptv2.tracking_framebuf import TargetArray

oc = importlib.import_module("openptv2.correspondences")      # the package re-exports a function of that name

pytestmark = pytest.mark.unit
YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"
GRAIN = 300.0                                                  # grey sum of a particle's blob


def _scene(sep_mm: float = 1.0, n_distract: int = 25):
    """Particles A and B merged into one blob in cameras 0 and 1, separate in cameras 2 and 3, among clean ones.

    B is displaced from A along the direction that moves least in cameras 0 and 1 (depth), so the merged blob's
    centre stays within the matcher's 1 px epipolar band of B."""
    cpar, vpar, cals, _ = load_parameters(YAML)
    rng = np.random.default_rng(11)
    pts = rng.uniform(-8, 8, (n_distract, 3))
    a = np.zeros(3)
    jac = np.vstack([
        np.array([project_pixels((a + 0.01 * e)[None], cals[c], cpar)[0] - project_pixels((a - 0.01 * e)[None], cals[c], cpar)[0]
                  for e in np.eye(3)]).T / 0.02
        for c in (0, 1)
    ])
    direction = np.linalg.svd(jac)[2][-1]
    b = a + sep_mm * direction
    pts = np.vstack([a, b, pts])
    amp = np.full(len(pts), GRAIN)
    amp[0] = 1.3 * GRAIN                                       # A is the brighter one
    dets = []
    for c, cal in enumerate(cals):
        px = project_pixels(pts, cal, cpar)
        rows = []                                              # (x, y, sumg)
        for i in range(len(pts)):
            if c in (0, 1) and i in (0, 1):
                continue
            rows.append((px[i, 0], px[i, 1], amp[i]))
        if c in (0, 1):                                        # the merged blob: grey-weighted centre, summed grey
            w = amp[:2] / amp[:2].sum()
            ctr = w @ px[:2]
            rows.append((ctr[0], ctr[1], amp[0] + amp[1]))
        rows.sort(key=lambda r: r[1])                          # the matcher needs y order
        ta = TargetArray(len(rows))
        for j, (x, y, g) in enumerate(rows):
            ta[j].set_pnr(j)
            ta[j].set_pos((float(x), float(y)))
            ta[j].set_pixel_counts(int(4 + g / 100), 3, 3)
            ta[j].set_sum_grey_value(int(g))
            ta[j].set_tnr(-1)
        dets.append(ta)
    return cpar, vpar, cals, dets, pts


def _has(pos, p, tol=0.3):
    return len(pos) > 0 and bool((np.linalg.norm(np.asarray(pos) - p, axis=1) < tol).any())


def test_the_standard_matcher_loses_one_of_two_particles_in_a_shared_blob():
    cpar, vpar, cals, dets, pts = _scene()
    pos, _ids = oc.match_frame_correspondences(dets, cpar, cals, vpar)
    assert sum(_has(pos, pts[i]) for i in (0, 1)) <= 1          # the premise of the pass


def test_the_second_pass_gives_the_partner_its_point():
    cpar, vpar, cals, dets, pts = _scene()
    pos0, ids0 = oc.match_frame_correspondences(dets, cpar, cals, vpar)
    pos1, ids1 = oc.match_frame_correspondences(dets, cpar, cals, vpar, shared_pass=1.7)
    assert _has(pos1, pts[0]) and _has(pos1, pts[1])
    assert len(pos1) == len(pos0) + 1 and ids1.shape[1] == ids0.shape[1]
    new = ids1[len(ids0):][0]
    assert (new >= 0).sum() >= 3
    # the partner uses the shared blob again: some blob id appears in two points
    both = np.vstack([ids0, ids1[len(ids0):]])
    assert any(len(np.unique(col[col >= 0])) < (col >= 0).sum() for col in both.T)


def test_the_pass_changes_nothing_without_a_shared_blob():
    cpar, vpar, cals, dets, pts = _scene()
    clean = [TargetArray([t for t in d if t.sumg < 2 * GRAIN * 0.9]) for d in dets]     # drop the merged blobs
    pos0, ids0 = oc.match_frame_correspondences(clean, cpar, cals, vpar)
    pos1, ids1 = oc.match_frame_correspondences(clean, cpar, cals, vpar, shared_pass=1.7)
    assert len(pos1) == len(pos0)
    np.testing.assert_array_equal(ids1, ids0)


def test_shared_pass_from_params():
    f = oc.shared_pass_from_params
    assert f(None) is None and f({}) is None and f({"shared_pass": False}) is None and f({"shared_pass": 0}) is None
    assert f({"shared_pass": True}) == oc.DEFAULT_SHARED_PASS
    assert f({"shared_pass": 1.5}) == 1.5
