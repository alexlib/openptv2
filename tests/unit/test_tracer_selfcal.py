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
            base, cpar, cals, tol_px=3.0, max_particles=150)
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
def test_tracer_selfcal_no_tracking_skips(tmp_path):
    class _Cpar:
        num_cams = 4
        mm = None

    out, info = tracer_self_calibrate(tmp_path, _Cpar(), [None] * 4)
    assert "skipped" in info
    assert out is not None
