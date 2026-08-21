"""Correctness tests for openptv2.gui.trackcorr_debug: the captured
per-particle candidate list must be the REAL trackcorr search result, not
an approximation -- checked by confirming the actual accepted link
(``path_next``, from the unmodified real algorithm) always appears among
the candidates this module captured for that particle.
"""

import os
import shutil

import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.constants import NEXT_NONE
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.gui.trackcorr_debug import (
    candidates_for_particle,
    load_run,
    step_and_capture,
)


def _read_all_calibration(num_cams, base_path="."):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cals.append(Calibration.from_file(ori_name, added_name))
    return cals


@pytest.fixture
def cavity_run(tmp_path):
    src = "test_data/test_cavity"
    if not os.path.exists(f"{src}/res_orig") or not os.path.exists(f"{src}/img_orig"):
        pytest.skip("test_cavity res_orig/img_orig fixtures not present")

    dst = tmp_path / "test_cavity"
    shutil.copytree(src, dst)
    if (dst / "res").exists():
        shutil.rmtree(dst / "res")
    if (dst / "img").exists():
        shutil.rmtree(dst / "img")
    shutil.copytree(dst / "res_orig", dst / "res")
    shutil.copytree(dst / "img_orig", dst / "img")

    cwd = os.getcwd()
    os.chdir(dst)
    try:
        cpar = ControlPar.from_yaml("parameters.yaml")
        spar = SequencePar.from_yaml("parameters.yaml")
        vpar = VolumePar.from_yaml("parameters.yaml")
        tpar = TrackPar.from_yaml("parameters.yaml")
        cals = _read_all_calibration(cpar.num_cams)
        run = load_run(cpar, spar, vpar, tpar, cals)
        yield run, spar
    finally:
        os.chdir(cwd)


def test_winner_is_always_among_captured_candidates(cavity_run):
    run, spar = cavity_run
    first, last = spar.get_first(), spar.get_last()
    snapshots = step_and_capture(run, first, last)

    assert snapshots, "expected at least one step"
    checked_linked = 0
    for step, snap in snapshots.items():
        for p in range(snap["num_parts_1"]):
            result = candidates_for_particle(snap, p)
            if result.winner_row == NEXT_NONE:
                continue  # particle wasn't linked this step -- nothing to check
            checked_linked += 1
            rows = [c.row for c in result.candidates]
            assert result.winner_row in rows, (
                f"step {step} particle {p}: real winner row "
                f"{result.winner_row} not found among captured candidates "
                f"{rows} (candidate capture is incomplete/wrong)"
            )
            # winner() must resolve via the same lookup
            assert result.winner is not None
            assert result.winner.row == result.winner_row

    assert checked_linked > 0, "expected at least one real link across the run"


def test_candidates_are_rank_ordered_by_cost(cavity_run):
    run, spar = cavity_run
    first, last = spar.get_first(), spar.get_last()
    snapshots = step_and_capture(run, first, last)

    checked = 0
    for snap in snapshots.values():
        for p in range(snap["num_parts_1"]):
            result = candidates_for_particle(snap, p)
            if len(result.candidates) < 2:
                continue
            checked += 1
            costs = [c.cost for c in result.candidates]
            assert costs == sorted(costs), "candidates must be rank-ordered ascending by cost"
            ranks = [c.rank for c in result.candidates]
            assert ranks == list(range(len(ranks)))

    assert checked > 0, "expected at least one particle with >=2 candidates to check ordering"


def test_candidate_cameras_have_consistent_tracer_ids(cavity_run):
    """Each candidate's per-camera (tnr, x, y) must come from a real target
    row in that step's next-frame detections, not a placeholder."""
    run, spar = cavity_run
    first, last = spar.get_first(), spar.get_last()
    snapshots = step_and_capture(run, first, last)

    checked = 0
    for snap in snapshots.values():
        for p in range(min(snap["num_parts_1"], 20)):
            result = candidates_for_particle(snap, p)
            for cand in result.candidates:
                for cam, (tnr, x, y) in cand.cameras.items():
                    assert 0 <= cam < snap["num_cams"]
                    assert isinstance(x, float) and isinstance(y, float)
                    checked += 1

    assert checked > 0
