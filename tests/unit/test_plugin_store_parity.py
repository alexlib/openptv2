"""Plugin store-vs-ASCII parity regression (plan Phase 3.3).

Running the same tracking plugin twice on the same inputs -- once fed from
ASCII rt_is files with no store present ("legacy run"), once fed from the
committed Zarr store -- must produce identical linkage output. This pins the
store-native migration of the tracking plugins: any divergence between the
two ingestion paths fails here.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar
from openptv2.gui.parameter_manager import ParameterManager
from openptv2.plugins.myptv_3d_tracking import Tracking as MyPTV3DTracking
from openptv2.storage import RunStore

pytestmark = pytest.mark.ci

REPO = Path(__file__).parent.parent.parent
SRC = REPO / "test_data" / "test_cavity"


class _StubExp:
    """Minimal experiment surface the plugins read (mirrors gui.Experiment)."""


def _make_exp(work: Path) -> _StubExp:
    old = os.getcwd()
    os.chdir(work)
    try:
        exp = _StubExp()
        exp.exp_path = str(work)
        exp.res_dir = "res"
        exp.cpar = ControlPar.from_yaml("parameters.yaml")
        exp.spar = SequencePar.from_yaml("parameters.yaml")
        pm = ParameterManager()
        pm.from_yaml(Path("parameters.yaml").resolve())
        exp.pm = pm
        exp.cals = [
            Calibration.from_file(
                f"cal/cam{c + 1}.tif.ori", f"cal/cam{c + 1}.tif.addpar"
            )
            for c in range(exp.cpar.num_cams)
        ]
        return exp
    finally:
        os.chdir(old)


def _run_plugin(work: Path, *, ascii_fed: bool) -> RunStore:
    if work.exists():
        shutil.rmtree(work)
    # Exclude res* AND run.zarr: neither the checkout's res/ nor its
    # (mutable, other tests write through it) fixture store may leak in.
    shutil.copytree(
        SRC,
        work,
        ignore=shutil.ignore_patterns("res*", "run.zarr"),
    )
    if ascii_fed:
        # Legacy mode: no store at all -- inputs come from res/rt_is ASCII;
        # a fresh (empty) store is created for outputs.
        shutil.copytree(SRC / "res_orig", work / "res")
    else:
        # Store-fed mode: build a pristine store from res_orig + img targets
        # (never the repo's mutable fixture store).
        from openptv2.algorithms.tracking_frame_buf import (
            read_targets as read_targets_ascii,
        )
        from openptv2.storage.legacy import _load_rt_is

        store = RunStore(work / "run.zarr", mode="a")
        res_dir = SRC / "res_orig"
        frames = sorted(
            int(p.name.split(".")[-1])
            for p in res_dir.glob("rt_is.*")
            if p.name.split(".")[-1].isdigit()
        )
        for frame in frames:
            pos, ids = _load_rt_is(res_dir / f"rt_is.{frame}")
            store.write_correspondences(frame, pos, ids)
            for cam in range(4):
                targs = read_targets_ascii(str(SRC / "img" / f"cam{cam}."), frame, cam_idx=cam)
                store.write_targets(cam, frame, targs)

        shutil.rmtree(work / "res", ignore_errors=True)

    exp = _make_exp(work)
    old = os.getcwd()
    os.chdir(work)
    try:
        # the plugin resolves its relative res/ bases against the process
        # cwd, exactly like the GUI/batch entry points do
        MyPTV3DTracking(exp=exp).do_tracking()
    finally:
        os.chdir(old)
    return RunStore(find_store(work), mode="r")


def find_store(root: Path) -> Path:
    from openptv2.storage import find_existing_store

    return find_existing_store(root)


def test_myptv_3d_plugin_ascii_and_store_runs_agree(tmp_path):
    ref = _run_plugin(tmp_path / "ascii_run", ascii_fed=True)
    store_fed = _run_plugin(tmp_path / "store_run", ascii_fed=False)

    frames_ref = [f for f in ref.frames() if ref.has_linkage(f, "ptv_is")]
    frames_new = [f for f in store_fed.frames() if store_fed.has_linkage(f, "ptv_is")]
    assert frames_ref == frames_new and frames_ref, "plugin tracked different frames"

    for frame in frames_ref:
        rp, rn_, _ = ref.read_linkage(frame, "ptv_is")
        sp, sn_, _ = store_fed.read_linkage(frame, "ptv_is")
        assert len(rp) == len(sp), f"particle count mismatch frame {frame}"
        assert np.array_equal(rp, sp), f"prev mismatch frame {frame}"
        assert np.array_equal(rn_, sn_), f"next mismatch frame {frame}"

    # sanity: the tracker actually linked something
    total_links = sum(int((ref.read_linkage(f, "ptv_is")[1] >= 0).sum()) for f in frames_ref)
    assert total_links > 0
