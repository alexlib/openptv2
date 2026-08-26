"""Regression test for the tnr-default bug (fixed 2026-08-18): a caller that
writes 2D targets into the RunStore for trackcorr WITHOUT setting
Target.tnr (the reverse link from a 2D target back to its 3D-particle row)
gets every candidate match silently resolved to particle 0 regardless of
which target was actually found -- deterministic near-total link failure
that looks nothing like real search-window/density problems (see
docs/plans/2026-08-17-lagrangian-accuracy-program.md, "the tnr bug").

This mirrors exactly what scripts/adapt_proptv_dataset.py does: build a
dataset with 2D targets and 3D correspondences written straight into the
store (no ascii _targets/rt_is at all), reusing an existing scaffold's own
working calibration, then run trackcorr and assert it actually links.
Doesn't depend on external proPTV data -- self-contained synthetic points.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.algorithms.tracking_frame_buf import Target
from openptv2.algorithms.trafo import metric_to_pixel
from openptv2.storage import RunStore

SCAFFOLD = Path(__file__).resolve().parents[2] / "test_data" / "synthetic_turbulent"
FIRST = 10001
NUM_CAMS = 4
N_PARTICLES = 5


def _build_dataset(out: Path, set_tnr: bool) -> None:
    if not SCAFFOLD.exists():
        pytest.skip("test_data/synthetic_turbulent scaffold not present")
    shutil.copytree(SCAFFOLD, out)
    res, img = out / "res", out / "img"
    # The scaffold ships without generated res/img (gitignored) -- tolerate
    # both the fresh-checkout case and a previously-populated copy.
    if res.exists():
        shutil.rmtree(res)
    res.mkdir()
    if img.exists():
        shutil.rmtree(img)
    img.mkdir()

    yaml_path = out / "parameters_Run1.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    ptv = data["ptv"]
    ptv["mmp_n1"] = ptv["mmp_n2"] = ptv["mmp_n3"] = 1.0
    ptv["mmp_d"] = 0.0
    cpar = ControlPar(
        num_cams=NUM_CAMS, imx=ptv["imx"], imy=ptv["imy"],
        pix_x=ptv["pix_x"], pix_y=ptv["pix_y"],
        mm=MmNp(n1=1.0, n2=[1.0], n3=1.0, d=[0.0]),
    )
    cals = [
        Calibration.from_file(
            str(out / "cal" / f"cam{c + 1}.tif.ori"), str(out / "cal" / f"cam{c + 1}.tif.addpar")
        )
        for c in range(NUM_CAMS)
    ]

    rng = np.random.default_rng(0)
    pos0 = rng.uniform(-15, 15, size=(N_PARTICLES, 3))
    vel = rng.uniform(-0.5, 0.5, size=(N_PARTICLES, 3))

    store = RunStore.open(out, mode="a")
    for fi in range(3):
        fn = FIRST + fi
        pos = pos0 + vel * fi
        cam_ids = np.full((N_PARTICLES, NUM_CAMS), -1, dtype=np.int32)
        for c in range(NUM_CAMS):
            projected = []
            for i in range(N_PARTICLES):
                mx, my = img_coord(pos[i], cals[c], cpar.mm)
                px, py = metric_to_pixel(mx, my, cpar)
                projected.append((i, px, py))
            projected.sort(key=lambda t: t[2])  # sort by y, required by the search
            targets = []
            for slot, (i, px, py) in enumerate(projected):
                cam_ids[i, c] = slot
                kwargs = dict(pnr=i, x=px, y=py)
                if set_tnr:
                    kwargs["tnr"] = i
                targets.append(Target(**kwargs))
            store.write_targets(c, fn, targets)
        store.write_correspondences(frame=fn, pos_3d=pos, cam_target_ids=cam_ids)

    data["sequence"]["first"] = FIRST
    data["sequence"]["last"] = FIRST + 2
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))


def _run_trackcorr(out: Path) -> int:
    """Returns the number of particles with an established next-link after
    the first forward step."""
    import os as _os

    from openptv2.batch.pyptv_batch import build_processing_experiment
    from openptv2.gui.ptv import _open_run_store
    from openptv2.tracker import Tracker

    prev_cwd = _os.getcwd()
    _os.chdir(out)
    try:
        exp = build_processing_experiment(out / "parameters_Run1.yaml", FIRST, FIRST + 2)
    finally:
        _os.chdir(prev_cwd)

    class _FakeExp:
        pass

    fe = _FakeExp()
    fe.pm = exp.pm
    fe.exp_dir = str(out)
    store = _open_run_store(fe)

    tp = exp.track_par
    tp.set_dvxmax(6.0)
    tp.set_dvxmin(-6.0)
    tp.set_dvymax(6.0)
    tp.set_dvymin(-6.0)
    tp.set_dvzmax(6.0)
    tp.set_dvzmin(-6.0)
    tp.set_dacc(6.0)

    tracker = Tracker(exp.cpar, exp.vpar, tp, exp.spar, exp.cals, store=store)
    tracker.restart()
    tracker.step_forward()
    fb = tracker._run.fb
    # step_forward() rotates the ring buffer; the just-computed step's links
    # land in buf[0] afterward (see docs/plans -- ring-buffer semantics).
    pn = fb.buf[0].path_next
    return int(sum(1 for v in pn[: fb.buf[0].num_parts] if int(v) >= 0))


def _write_fake_proptv_origin(case_dir: Path) -> None:
    """A minimal proPTV-style case: origin/origin_NNNNN.txt with just
    ID,X,Y,Z (adapt_proptv_dataset.convert() ignores every column after Z --
    proPTV's own xc/yc columns are not used, see that module's docstring),
    5 well-separated particles drifting slightly over 3 frames."""
    origin_dir = case_dir / "origin"
    origin_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    pos0 = rng.uniform(0.1, 0.9, size=(N_PARTICLES, 3))
    vel = rng.uniform(-0.02, 0.02, size=(N_PARTICLES, 3))
    for fi in range(3):
        pos = pos0 + vel * fi
        lines = [f"{i} {pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f}" for i in range(N_PARTICLES)]
        (origin_dir / f"origin_{fi:05d}.txt").write_text("\n".join(lines) + "\n")


@pytest.mark.unit
def test_adapt_proptv_dataset_convert_links_on_a_fresh_copy(tmp_path):
    """Exercises the actual shipped scripts/adapt_proptv_dataset.py convert()
    end to end (not a reimplementation) against a fresh synthetic proPTV-style
    case, then runs trackcorr and requires it to actually link -- the
    regression this whole file exists to catch."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import adapt_proptv_dataset as apd

    if not SCAFFOLD.exists():
        pytest.skip("test_data/synthetic_turbulent scaffold not present")

    case_dir = tmp_path / "fake_proptv_case"
    _write_fake_proptv_origin(case_dir)
    out = tmp_path / "converted"
    apd.convert(case_dir, SCAFFOLD, out)

    n_linked = _run_trackcorr(out)
    assert n_linked == N_PARTICLES, (
        f"expected all {N_PARTICLES} particles linked via the real adapter output, "
        f"got {n_linked}"
    )


@pytest.mark.unit
def test_adapt_proptv_dataset_convert_realistic_runs_end_to_end(tmp_path):
    """Smoke test for convert_realistic(): the real detection/correspondence
    pipeline (not ground-truth injection) must run to completion, produce a
    plausible (neither ~0% nor ~100%) match rate, and its output must still
    be trackable -- catches wiring breaks (wrong calibration list, wrong
    vpar fields, MatchedCoords/Frame index mismatches) that a pure "does it
    link perfectly" test on noise-free data wouldn't exercise."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import adapt_proptv_dataset as apd

    if not SCAFFOLD.exists():
        pytest.skip("test_data/synthetic_turbulent scaffold not present")

    case_dir = tmp_path / "fake_proptv_case"
    _write_fake_proptv_origin(case_dir)
    out = tmp_path / "converted_realistic"
    apd.convert_realistic(case_dir, SCAFFOLD, out, seed=0)

    store = RunStore.open(out, mode="r")
    pos, cam_ids = store.read_correspondences(FIRST)
    assert 0 < len(pos) <= N_PARTICLES, (
        f"expected a plausible partial match count (0, {N_PARTICLES}], got {len(pos)}"
    )

    n_linked = _run_trackcorr(out)
    assert n_linked >= 1, "expected at least one real link on the realistic-pipeline output"


@pytest.mark.unit
def test_trackcorr_links_when_tnr_is_set(tmp_path):
    out = tmp_path / "with_tnr"
    _build_dataset(out, set_tnr=True)
    n_linked = _run_trackcorr(out)
    assert n_linked == N_PARTICLES, (
        f"expected all {N_PARTICLES} well-separated particles to link, got {n_linked} "
        "-- store-only 2D targets/correspondences with tnr set should track cleanly"
    )


@pytest.mark.unit
def test_trackcorr_fails_when_tnr_is_unset(tmp_path):
    """Documents the bug this file guards against: omitting Target.tnr makes
    every candidate resolve to particle 0, so at most 1 particle can ever
    link regardless of how many are actually present and well-separated."""
    out = tmp_path / "without_tnr"
    _build_dataset(out, set_tnr=False)
    n_linked = _run_trackcorr(out)
    assert n_linked == 1, f"expected exactly 1 (the tnr-collapse signature), got {n_linked}"
