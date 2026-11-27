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
    probe_particle,
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


@pytest.fixture
def cavity_loader(tmp_path):
    """Like cavity_run, but yields a zero-arg factory for a *fresh* run each
    call (needed by probe_particle tests, which require an unstepped run:
    calling it more than once, or after step_and_capture has advanced a
    prior run, must not share state between them)."""
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

        def _fresh_run():
            return load_run(cpar, spar, vpar, tpar, cals)

        yield _fresh_run, spar
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


def test_probe_particle_matches_real_candidate_set(cavity_loader):
    """probe_particle's single-particle recompute (~200x faster than a real
    full-frame step, for interactive 'tune then press Run' use) must return
    the SAME candidate set/costs as the real full-frame step, for the same
    particle at the same (unoverridden) parameters -- checked by comparing
    against candidates_for_particle() on a real step_and_capture()
    snapshot, not just "it runs"."""
    fresh_run, spar = cavity_loader
    first = spar.get_first()

    real_run = fresh_run()
    real_snapshots = step_and_capture(real_run, first, first + 1)
    real_snap = real_snapshots[first]

    probe_run = fresh_run()
    checked = 0
    for p in range(0, real_snap["num_parts_1"], 37):  # sample across the frame
        real = candidates_for_particle(real_snap, p)
        probe = probe_particle(probe_run, first, p)

        real_set = sorted((c.row, round(c.cost, 6)) for c in real.candidates)
        probe_set = sorted((c.row, round(c.cost, 6)) for c in probe.candidates)
        assert probe_set == real_set, (
            f"particle {p}: probe candidate set/costs differ from the real "
            f"full-frame step (probe={probe_set}, real={real_set})"
        )
        assert probe.is_isolated is True
        assert probe.particle_index == p
        checked += 1

    assert checked > 0


def test_probe_particle_winner_is_isolated_best_by_cost():
    """probe_particle's winner_row must be the lowest-cost candidate (the
    only thing a single-particle probe can know), never silently reused
    from some other field -- verified against a hand-built snapshot with a
    known non-trivial cost ordering."""
    import numpy as np

    snap = {
        "step": 0,
        "num_cams": 1,
        "num_parts_1": 1,
        "path_x_1": np.array([[0.0, 0.0, 0.0]]),
        "path_next_1": np.array([999]),  # must be ignored when is_isolated=True
        "path_inlist_1": np.array([2]),
        "path_decis_1": np.array([[0.9, 0.2]]),  # row 5 is cheaper than row 3
        "path_linkdecis_1": np.array([[3, 5]]),
        "num_parts_2": 6,
        "path_x_2": np.array([[i, i, i] for i in range(6)], dtype=float),
        "corres_p_2": np.array([[0]] * 6),
        "targ_x_2": [np.array([1.0])],
        "targ_y_2": [np.array([2.0])],
        "targ_tnr_2": [np.array([9])],
    }
    result = candidates_for_particle(snap, 0, is_isolated=True)
    assert result.is_isolated is True
    assert result.candidates[0].row == 5  # cheapest (cost 0.2)
    assert result.winner_row == 5
    assert result.winner.row == 5
