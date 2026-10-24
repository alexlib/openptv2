"""The sequence step's triangulation method (ptv.triangulation) reaches point_positions.

test_cavity has cameras on opposite sides of the tank; there "pairs" (mean of pairwise
midpoints) misplaces points and "lsq" (point nearest all rays) reprojects better.
"""

import os

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameters import ControlPar, VolumePar
from openptv2.algorithms.trafo import metric_to_pixel_batch
from openptv2.correspondences import match_frame_correspondences
from openptv2.tracking_framebuf import read_targets
from tests._support import find_test_data_root

CAVITY = find_test_data_root() / "test_cavity"


def _worst_reprojection_px(pos, ids, targets, cals, cpar):
    err = np.zeros(len(pos))
    for cam, cal in enumerate(cals):
        seen = ids[:, cam] >= 0
        det = np.array([[t.x, t.y] for t in targets[cam]])[ids[seen, cam]]
        px = metric_to_pixel_batch(img_coord_batch(pos[seen], cal, cpar.mm), cpar)
        err[seen] = np.maximum(err[seen], np.hypot(*(px - det).T))
    return err


def test_lsq_triangulation_reprojects_better_than_pairs(monkeypatch):
    monkeypatch.chdir(CAVITY)
    yaml_path = "parameters_Run1.yaml"
    cpar, vpar = ControlPar.from_yaml(yaml_path), VolumePar.from_yaml(yaml_path)
    cals = [
        Calibration.from_file(f"cal/cam{c}.tif.ori", f"cal/cam{c}.tif.addpar")
        for c in range(1, 5)
    ]
    assert os.path.exists("img/cam1.10001_targets")

    def run(method):
        targets = [read_targets(f"img/cam{c}.%d", 10001) for c in range(1, 5)]
        pos, ids = match_frame_correspondences(targets, cpar, cals, vpar, method=method)
        # match_frame_correspondences sorts the targets in place, as ids refer to
        return pos, ids, _worst_reprojection_px(pos, ids, targets, cals, cpar)

    pos_p, ids_p, err_p = run("pairs")
    pos_l, ids_l, err_l = run("lsq")

    np.testing.assert_array_equal(ids_p, ids_l)  # same matches, other positions
    assert not np.allclose(pos_p, pos_l)
    assert np.median(err_l) < np.median(err_p)
    assert np.percentile(err_l, 90) < np.percentile(err_p, 90)
