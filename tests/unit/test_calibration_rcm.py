"""Cross-camera ray-convergence miss distance (RCM) over the calblock.

Per-camera reprojection RMS cannot see cross-camera geometric inconsistency;
cross_camera_rcm can. Built directly from the committed per-camera .ori/.addpar
(no detection/_targets needed): each calblock point is reprojected through its
camera's own calibration, so the four rays are as consistent as the committed
bundle. Test B is the real regression guard: perturbing one camera's exterior
must strictly increase the median RCM.
"""
import copy
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.sortgrid import read_calblock
from openptv2.autocalibration import (
    CamResult,
    _load_dataset_params,
    _reproject_px,
    cam_files,
    cross_camera_rcm,
    resolve_calblock,
)

DATASET = Path("test_data/test_cavity")


@pytest.fixture(scope="module")
def results_and_cpar():
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    base = DATASET.resolve()
    cpar = _load_dataset_params(base, resolve_calblock(base)).cpar
    fix, _ = read_calblock(str(resolve_calblock(base)))
    fix = np.asarray(fix, float)

    results = []
    for cam in range(cpar.num_cams):
        _, ori, addpar = cam_files(base, cam)
        cal = Calibration.from_file(str(ori), str(addpar))
        det = np.array([_reproject_px(cal, cpar.mm, p, cpar) for p in fix])
        results.append(CamResult(
            cam=cam, matched=len(fix), nfix=len(fix), rms=0.0,
            flags=["cc", "xh", "yh"], cal=cal, ref=fix.copy(),
            det=det, rep=det.copy(),
        ))
    return results, cpar


@pytest.mark.unit
def test_rcm_basic(results_and_cpar):
    results, cpar = results_and_cpar
    rcm = cross_camera_rcm(results, cpar)
    assert rcm is not None
    assert rcm["n_common"] >= 2
    assert rcm["median"] >= 0
    assert np.isfinite(rcm["max"])


@pytest.mark.unit
def test_rcm_monotonic_under_perturbation(results_and_cpar):
    results, cpar = results_and_cpar
    base_rcm = cross_camera_rcm(results, cpar)

    perturbed = copy.deepcopy(results)
    pos = perturbed[0].cal.get_pos()
    pos[0] += 2.0
    perturbed[0].cal.set_pos(pos)

    pert_rcm = cross_camera_rcm(perturbed, cpar)
    assert pert_rcm["median"] > base_rcm["median"]


@pytest.mark.unit
def test_rcm_single_camera_returns_none(results_and_cpar):
    results, cpar = results_and_cpar
    assert cross_camera_rcm(results[:1], cpar) is None
