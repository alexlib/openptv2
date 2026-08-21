"""tracer_self_calibrate runs end-to-end on a tracking fixture and never
diverges (the coupled joint fit over free tracer particles, gauge-fixed by
holding one camera). Real improvement is demonstrated on flow data with depth
coverage the plate lacks; here we guard the plumbing + the no-worsen contract."""

import os
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.autocalibration import (
    _cpar_from_ptv,
    _find_yaml,
    cam_files,
    tracer_self_calibrate,
)

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
    if not (SYNTH / "res").exists() or not list((SYNTH / "res").glob("ptv_is.*")):
        pytest.skip("synthetic tracking results not present")
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


@pytest.mark.unit
def test_tracer_selfcal_iterated_never_worsens():
    """Iterated shaking (refine -> re-match -> repeat) accepts a pass only if RCM
    improves, so the result never worsens and reports its accepted-iteration
    count + trace."""
    if not (SYNTH / "res").exists() or not list((SYNTH / "res").glob("ptv_is.*")):
        pytest.skip("synthetic tracking results not present")
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
    store-backed (writes only to res/run.zarr, no ASCII ptv_is.*/targets --
    see tracking_frame_buf.write_path_frame/write_targets). Reproduce that
    here: same known-consistent geometry as the ASCII fixture above (copied
    from its checked-in ptv_is.*/targets files into a fresh RunStore, no
    ASCII alongside), and confirm tracer_self_calibrate actually reads it
    instead of reporting "no res/ptv_is.* frames"."""
    if not (SYNTH / "res").exists() or not list((SYNTH / "res").glob("ptv_is.*")):
        pytest.skip("synthetic tracking results not present")
    import shutil

    from openptv2.storage import RunStore

    base = tmp_path / "synth_store"
    shutil.copytree(SYNTH, base, ignore=shutil.ignore_patterns("res", "img", "*_targets"))
    (base / "img").mkdir()
    for f in (SYNTH / "img").glob("*.tif"):
        shutil.copy2(f, base / "img" / f.name)

    cpar_src, cals_src = _load()[1:]
    n_cams = cpar_src.num_cams

    store = RunStore.open(base, mode="a")
    for pf in sorted(SYNTH.glob("res/ptv_is.*"), key=lambda p: int(p.suffix.lstrip("."))):
        frame = int(pf.suffix.lstrip("."))
        lines = pf.read_text().splitlines()
        nn = int(lines[0])
        prev, nxt, pos = [], [], []
        for line in lines[1 : nn + 1]:
            p = line.split()
            prev.append(int(p[0]))
            nxt.append(int(p[1]))
            pos.append([float(p[2]), float(p[3]), float(p[4])])
        store.write_linkage(frame, prev, nxt, pos, name="ptv_is")

        for cam in range(n_cams):
            tf = SYNTH / "img" / f"cam{cam + 1}.{frame}_targets"
            if not tf.exists():
                continue
            tlines = tf.read_text().splitlines()
            tn = int(tlines[0])
            rows = [[float(x) for x in tlines[i + 1].split()] for i in range(tn)]
            store.write_targets(cam, frame, np.asarray(rows, float) if rows else np.empty((0, 8)))

    cwd = os.getcwd()
    try:
        os.chdir(base)
        new_cals, info = tracer_self_calibrate(
            base, cpar_src, cals_src, tol_px=3.0, max_particles=150
        )
    finally:
        os.chdir(cwd)
    assert "skipped" not in info, info
    assert info["n_particles"] > 0 and info["n_obs"] > 0
    assert len(new_cals) == cpar_src.num_cams


@pytest.mark.unit
def test_tracer_selfcal_no_tracking_skips(tmp_path):
    class _Cpar:
        num_cams = 4
        mm = None

    out, info = tracer_self_calibrate(tmp_path, _Cpar(), [None] * 4)
    assert "skipped" in info
    assert out is not None
