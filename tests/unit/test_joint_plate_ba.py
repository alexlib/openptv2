"""Joint plate bundle adjustment lowers cross-camera RCM.

Built like test_calibration_rcm: CamResults are made directly from committed
.ori/.addpar by reprojecting the calblock (no _targets needed). Detections
encode the TRUE geometry, so RCM starts ~0. We then perturb two cameras' seed
cals (positions nudged ~1-2mm) while KEEPING the detections fixed -- so the
per-camera fit is inconsistent and RCM > 0. A well-anchored joint BA should
pull the cameras back toward the consistent geometry and lower RCM.
"""
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
    joint_plate_bundle_adjust,
    resolve_calblock,
)

DATASET = Path("test_data/test_cavity")


def _build_results():
    base = DATASET.resolve()
    cpar = _load_dataset_params(base, resolve_calblock(base)).cpar
    fix, _ = read_calblock(str(resolve_calblock(base)))
    fix = np.asarray(fix, float)
    results = []
    for cam in range(cpar.num_cams):
        _, ori, addpar = cam_files(base, cam)
        cal = Calibration.from_file(str(ori), str(addpar))
        # detections from the TRUE cal -> consistent geometry
        det = np.array([_reproject_px(cal, cpar.mm, p, cpar) for p in fix])
        results.append(CamResult(
            cam=cam, matched=len(fix), nfix=len(fix), rms=0.0,
            flags=["cc", "xh", "yh"], cal=cal, ref=fix.copy(),
            det=det, rep=det.copy(),
        ))
    return results, cpar


@pytest.mark.unit
def test_joint_ba_lowers_rcm():
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    results, cpar = _build_results()

    # Perturb two cameras' cals (detections stay fixed -> RCM becomes nonzero).
    for cam in (0, 1):
        pos = results[cam].cal.get_pos()
        pos[0] += 1.5
        pos[1] -= 1.0
        results[cam].cal.set_pos(pos)

    before = cross_camera_rcm(results, cpar)
    assert before is not None and before["median"] > 0

    new_results, info = joint_plate_bundle_adjust(results, cpar)
    assert info["success"]
    assert np.isfinite(info["cost_after"])
    assert info["rcm_before"] is not None and info["rcm_after"] is not None
    # BA must not make cross-camera convergence worse; it should improve here.
    assert info["rcm_after"] <= info["rcm_before"] * 1.01
    assert info["rcm_after"] < info["rcm_before"]


@pytest.mark.unit
def test_joint_ba_with_detection_noise():
    """Honest counterpart to test_joint_ba_lowers_rcm: real detections carry
    noise, so a perfect solution does NOT exist. BA should still lower RCM but
    it must NOT collapse to ~0 -- that would signal a tautological fixture."""
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    results, cpar = _build_results()

    rng = np.random.default_rng(0)
    for cam in range(cpar.num_cams):
        # 0.3 px Gaussian detection noise (typical sub-pixel centroid scatter)
        results[cam].det = results[cam].det + rng.normal(0, 0.3, results[cam].det.shape)
    for cam in (0, 1):
        pos = results[cam].cal.get_pos()
        pos[0] += 1.5
        pos[1] -= 1.0
        results[cam].cal.set_pos(pos)

    new_results, info = joint_plate_bundle_adjust(results, cpar)
    assert info["success"]
    assert info["rcm_after"] < info["rcm_before"]  # BA helps
    assert info["rcm_after"] > 1e-3  # but noise floors it above machine-zero


@pytest.mark.unit
def test_joint_ba_rejects_nonpositive_reg_weight():
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    results, cpar = _build_results()
    with pytest.raises(ValueError):
        joint_plate_bundle_adjust(results, cpar, reg_weight=0)


@pytest.mark.unit
def test_shake_distortion_helps_or_neutral():
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    results, cpar = _build_results()
    for cam in (0, 1):
        pos = results[cam].cal.get_pos()
        pos[0] += 1.5
        pos[1] -= 1.0
        results[cam].cal.set_pos(pos)

    _, info = joint_plate_bundle_adjust(results, cpar, shake_distortion=True)
    # Groups are only accepted on improvement, so the final can never be worse
    # than the exterior-only baseline.
    assert info["rcm_after"] <= info["rcm_exterior_only"] + 1e-9
    assert isinstance(info["shaken_groups"], list)
    # Each accepted group in the trace was strictly better than the running best.
    running = info["rcm_exterior_only"]
    for name, trial_rcm, accepted in info["rcm_trace"]:
        if accepted:
            assert trial_rcm < running
            running = trial_rcm
    print("shaken_groups:", info["shaken_groups"])
    print("rcm_trace:", info["rcm_trace"])
    print("exterior_only -> after:", info["rcm_exterior_only"], info["rcm_after"])

    # No-shake path is unchanged: no groups shaken.
    _, info0 = joint_plate_bundle_adjust(results, cpar, shake_distortion=False)
    assert info0["shaken_groups"] == []


@pytest.mark.unit
def test_shake_distortion_gate_rejects_unhelpful():
    """Detections come from a ZERO-distortion cal, only exterior is perturbed.
    Distortion can't explain the residual, so the gate must reject groups (or at
    least never accept one that didn't strictly reduce RCM)."""
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    results, cpar = _build_results()
    for cam in (0, 1):
        pos = results[cam].cal.get_pos()
        pos[0] += 1.0
        pos[1] -= 0.8
        results[cam].cal.set_pos(pos)

    _, info = joint_plate_bundle_adjust(results, cpar, shake_distortion=True)
    # Gate correctness: every accepted group strictly reduced the running RCM.
    running = info["rcm_exterior_only"]
    for name, trial_rcm, accepted in info["rcm_trace"]:
        if accepted:
            assert trial_rcm < running
            running = trial_rcm
    assert info["rcm_after"] <= info["rcm_exterior_only"] + 1e-9


@pytest.mark.unit
def test_joint_ba_single_camera_skips():
    if not DATASET.exists():
        pytest.skip("test_cavity dataset not present")
    results, cpar = _build_results()
    one = results[:1]
    out, info = joint_plate_bundle_adjust(one, cpar)
    assert "skipped" in info
    assert out is one  # returned unchanged
