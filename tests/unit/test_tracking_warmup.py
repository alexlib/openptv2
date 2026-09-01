"""Stage 1 (docs/plans/2026-08-15-tracking-quality-overhaul.md): standalone
warmup auto-calibration.

Uses test_data/tracking_synthetic (12 well-separated particles, 8 frames) --
fast, and its res_orig/rt_is.# ASCII files are populated straight into a
RunStore, matching what a prior `openptv sequence` run would have produced.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.storage import RunStore
from openptv2.tracking_warmup import run_warmup, write_result_to_yaml

FIX = Path(__file__).resolve().parents[2] / "test_data" / "tracking_synthetic"


def _populate_store_from_rt_is(store, res_dir, first, last):
    for f in range(first, last + 1):
        p = res_dir / f"rt_is.{f}"
        if not p.exists():
            continue
        data = np.loadtxt(p, skiprows=1, ndmin=2)
        if data.size == 0:
            continue
        pos = data[:, 1:4]
        cam_ids = data[:, 4:].astype(np.int32)
        store.write_correspondences(f, pos, cam_ids)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    dst = tmp_path / "run"
    shutil.copytree(FIX / "res_orig", dst / "res")
    shutil.copytree(FIX / "img_orig", dst / "img")
    shutil.copytree(FIX / "cal", dst / "cal")
    shutil.copy(FIX / "parameters_Run1.yaml", dst / "parameters_Run1.yaml")
    monkeypatch.chdir(dst)  # spar.img_base_name/naming paths are relative

    yaml_path = dst / "parameters_Run1.yaml"
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    tpar = TrackPar.from_yaml(str(yaml_path))
    spar = SequencePar.from_yaml(str(yaml_path), cpar.num_cams)
    cals = [
        Calibration.from_file(
            str(dst / f"cal/cam{c + 1}.tif.ori"),
            str(dst / f"cal/cam{c + 1}.tif.addpar"),
        )
        for c in range(cpar.num_cams)
    ]

    store = RunStore(str(dst / "res" / "run.zarr"), mode="w")
    _populate_store_from_rt_is(store, dst / "res", spar.first, spar.last)

    return {
        "dst": dst,
        "yaml_path": yaml_path,
        "cpar": cpar,
        "vpar": vpar,
        "tpar": tpar,
        "spar": spar,
        "cals": cals,
        "store": store,
    }


def test_run_warmup_produces_sane_result(scene):
    result = run_warmup(
        scene["cpar"],
        scene["vpar"],
        scene["tpar"],
        scene["spar"],
        scene["cals"],
        scene["store"],
        frames=5,
        max_cycles=2,
    )

    assert result.tracker in ("priority_segment_3d", "full_multipass")
    assert 0.0 <= result.agreement_rate <= 1.0
    assert result.noise_estimate_mm >= 0.0
    assert result.cycles >= 1
    assert result.frames[0] == scene["spar"].first
    assert result.frames[1] <= scene["spar"].first + 4

    tp = result.track_par
    for key in ("dvxmin", "dvxmax", "dvymin", "dvymax", "dvzmin", "dvzmax", "dacc"):
        assert np.isfinite(tp[key])
    assert tp["dvxmax"] > 0
    assert tp["dvxmin"] < 0

    # persisted into the store for later inspection, per the plan (stats/warmup)
    persisted = scene["store"].root["meta"].attrs["warmup"]
    assert persisted["tracker"] == result.tracker
    assert persisted["frames"] == list(result.frames)


def test_write_result_to_yaml_round_trips(scene):
    import yaml

    result = run_warmup(
        scene["cpar"],
        scene["vpar"],
        scene["tpar"],
        scene["spar"],
        scene["cals"],
        scene["store"],
        frames=5,
        max_cycles=1,
    )
    write_result_to_yaml(result, scene["yaml_path"])

    with open(scene["yaml_path"], encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert data["plugins"]["selected_tracking"] == result.tracker
    assert data["track"]["dvxmax"] == result.track_par["dvxmax"]
    assert data["track"]["dacc"] == result.track_par["dacc"]

    # a plain TrackPar.from_yaml (what production tracking loads) picks up
    # the written-back values with no warmup-awareness of its own.
    reloaded = TrackPar.from_yaml(str(scene["yaml_path"]))
    assert reloaded.dvxmax == result.track_par["dvxmax"]
    assert reloaded.dacc == result.track_par["dacc"]
