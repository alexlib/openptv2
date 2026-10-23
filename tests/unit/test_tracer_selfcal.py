"""tracer_self_calibrate runs end-to-end on a Zarr RunStore tracking fixture,
diagnoses which camera causes the most ghosts, filters out ghost particles, and
progressively increases the RCM separation between real (shaken) and ghost
particles."""

import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.autocalibration import (
    _cpar_from_ptv,
    _find_yaml,
    cam_files,
    retriangulate_store,
    tracer_self_calibrate,
)
from openptv2.storage import RunStore, find_existing_store

SYNTH = Path("test_data/synthetic")


def _load():
    import yaml

    base = SYNTH.resolve()
    y = yaml.safe_load(_find_yaml(base).read_text())
    cpar = _cpar_from_ptv(y["ptv"], int(y.get("num_cams") or y["ptv"]["num_cams"]))
    cals = [
        Calibration.from_file(*[str(p) for p in cam_files(base, c)[1:]])
        for c in range(cpar.num_cams)
    ]
    return base, cpar, cals


@pytest.mark.unit
def test_tracer_selfcal_runs_and_does_not_diverge():
    if find_existing_store(SYNTH) is None:
        pytest.skip("synthetic Zarr RunStore not present")
    cwd = os.getcwd()
    try:
        base, cpar, cals = _load()
        os.chdir(base)
        new_cals, info = tracer_self_calibrate(
            base, cpar, cals, tol_px=3.0, max_particles=150
        )
    finally:
        os.chdir(cwd)
    assert "skipped" not in info, info
    assert info["n_particles"] > 0 and info["n_obs"] > 0
    assert info["rcm_before"] is not None and info["rcm_after"] is not None
    assert np.isfinite(info["rcm_after"])
    # Core contract: a gauge-fixed joint fit must never worsen convergence.
    assert info["rcm_after"] <= info["rcm_before"] * 1.02
    assert len(new_cals) == cpar.num_cams
    assert "camera_diagnostics_before" in info and "camera_diagnostics_after" in info


@pytest.mark.unit
def test_tracer_selfcal_iterated_never_worsens():
    """Iterated shaking (refine -> re-match -> repeat) accepts a pass only if RCM
    improves, so the result never worsens and reports its accepted-iteration
    count + trace."""
    if find_existing_store(SYNTH) is None:
        pytest.skip("synthetic Zarr RunStore not present")
    cwd = os.getcwd()
    try:
        base, cpar, cals = _load()
        os.chdir(base)
        _, info = tracer_self_calibrate(
            base, cpar, cals, tol_px=3.0, max_particles=150, iters=4
        )
    finally:
        os.chdir(cwd)
    assert "skipped" not in info, info
    assert info["rcm_after"] <= info["rcm_before"] * 1.02
    assert "iterations" in info and info["iterations"] >= 0
    assert isinstance(info["rcm_trace"], list)
    # every accepted pass in the trace strictly improved on the running best
    accepted = [t for t in info["rcm_trace"] if t["accepted"]]
    for a, b in zip(accepted, accepted[1:]):
        assert b["rcm"] < a["rcm"]


@pytest.mark.unit
def test_tracer_selfcal_reads_store_backed_run(tmp_path):
    """A normal sequence+tracking run through the GUI/batch pipeline is
    store-backed (writes only to res/run.zarr, no legacy ASCII files). Confirm
    tracer_self_calibrate reads directly from RunStore."""
    src_store_path = find_existing_store(SYNTH)
    if src_store_path is None:
        pytest.skip("synthetic Zarr RunStore not present")

    base = tmp_path / "synth_store"
    shutil.copytree(
        SYNTH,
        base,
        ignore=shutil.ignore_patterns("res", "res_orig", "img", "*_targets", "*.zarr"),
    )
    shutil.copytree(src_store_path, base / "res" / "run.zarr")

    cpar_src, cals_src = _load()[1:]
    new_cals, info = tracer_self_calibrate(
        base, cpar_src, cals_src, tol_px=3.0, max_particles=150
    )
    assert "skipped" not in info, info
    assert info["n_particles"] > 0 and info["n_obs"] > 0
    assert len(new_cals) == cpar_src.num_cams


@pytest.mark.unit
def test_tracer_selfcal_focuses_on_perturbed_camera_and_separates_ghosts(tmp_path):
    """When one camera has a calibration offset and ghost particles are present
    in the Zarr store, tracer_self_calibrate identifies the worst camera, avoids
    holding it fixed when hold_cam='auto', lowers real-particle RCM, and
    increases the ghost-to-real RCM separation ratio."""
    src_store_path = find_existing_store(SYNTH)
    if src_store_path is None:
        pytest.skip("synthetic Zarr RunStore not present")

    base = tmp_path / "synth_ghost_shake"
    shutil.copytree(
        SYNTH,
        base,
        ignore=shutil.ignore_patterns("res", "res_orig", "img", "*_targets", "*.zarr"),
    )
    shutil.copytree(src_store_path, base / "res" / "run.zarr")

    _, cpar, cals = _load()
    # Inject synthetic ghost correspondences into the Zarr store (mismatched target IDs)
    store = RunStore(base / "res" / "run.zarr", mode="a")
    rng = np.random.default_rng(42)
    for f in store.frames():
        if not store.has_correspondences(f):
            continue
        pos, cid = store.read_correspondences(f)
        if len(pos) < 6:
            continue
        n_g = max(2, len(pos) // 4)
        g_pos = pos[:n_g] + rng.uniform(-1.5, 1.5, size=(n_g, 3))
        g_cid = cid[:n_g].copy()
        # Swap target indices in camera 2 and drop camera 3 on half of them (3-cam ghosts)
        g_cid[:, 2] = np.roll(g_cid[:, 2], 1)
        g_cid[::2, 3] = -1
        new_pos = np.vstack([pos, g_pos])
        new_cid = np.vstack([cid, g_cid])
        store.write_correspondences(f, new_pos, new_cid)
        if store.has_linkage(f, "ptv_is"):
            prv, nxt, lpos = store.read_linkage(f, "ptv_is")
            store.write_linkage(
                f,
                np.concatenate([prv, np.full(n_g, -1, dtype=np.int32)]),
                np.concatenate([nxt, np.full(n_g, -1, dtype=np.int32)]),
                np.vstack([lpos, g_pos]),
                name="ptv_is",
            )

    # Perturb camera 2's exterior position so it is the dominant source of mismatch
    bad_cals = [copy.deepcopy(c) for c in cals]
    p2 = np.asarray(bad_cals[2].get_pos(), dtype=float)
    bad_cals[2].set_pos(p2 + np.array([0.25, -0.20, 0.15]))

    shaken_cals, info = tracer_self_calibrate(
        base,
        cpar,
        bad_cals,
        tol_px=4.0,
        hold_cam="auto",
        max_particles=200,
        iters=3,
    )
    assert "skipped" not in info, info
    assert info["worst_cam"] == 2
    assert info["hold_cam"] != 2
    assert info["rcm_after"] < info["rcm_before"]
    assert info["real_rcm_after"] < info["real_rcm_before"]
    if (
        info["separation_ratio_before"] is not None
        and info["separation_ratio_after"] is not None
    ):
        assert info["separation_ratio_after"] > info["separation_ratio_before"]

    # Also verify retriangulate_store updates the store's 3D correspondences cleanly
    retri = retriangulate_store(base, shaken_cals, cpar)
    assert retri["n_frames"] > 0 and retri["n_points"] > 0


@pytest.mark.unit
def test_tracer_selfcal_no_tracking_skips(tmp_path):
    class _Cpar:
        num_cams = 4
        mm = None

    out, info = tracer_self_calibrate(tmp_path, _Cpar(), [None] * 4)
    assert "skipped" in info
    assert out is not None
