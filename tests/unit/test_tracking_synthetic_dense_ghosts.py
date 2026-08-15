"""Ghost-inclusive primary benchmark (Stage 0.5,
docs/plans/2026-08-15-tracking-quality-overhaul.md).

test_tracking_synthetic_dense.py's build_fixture writes rt_is directly from
known particle identity -- ghost-free by construction, so it cannot show the
dominant real-data failure mode this whole plan is about (see
docs/plans/two-subrig-calibration.md: test_cavity's 2-cam pairs are 64%
ghost, quads 16%). build_fixture_with_correspondence runs the REAL
combinatorial correspondence matcher on noisy 2D targets instead, so ghosts
(mismatched-identity quads/triplets/pairs) arise from genuine epipolar
ambiguity as density increases -- with NO injected false detections needed.

Because rows are no longer guaranteed to equal particle id once real
correspondence runs (a ghost row mixes identities; a missed particle skips a
row), correctness here is checked against row_gt (row -> true pid, -1 for a
ghost), not the row-index-equals-particle-id shortcut build_fixture's own
tests use.
"""

import shutil
import sys
from pathlib import Path

import pytest

FIX = Path(__file__).resolve().parents[2] / "test_data" / "tracking_synthetic_dense"
sys.path.insert(0, str(FIX))
from generate import build_fixture_with_correspondence  # noqa: E402

from openptv2.algorithms.calibration import Calibration  # noqa: E402
from openptv2.algorithms.parameters import (  # noqa: E402
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)
from openptv2.tracker import Tracker  # noqa: E402


def _make_scene(tmp_path, n_particles, spacing_mm, motion_mm, noise_px, n_frames, seed):
    scene_dir = tmp_path / "scene"
    frames, row_gt = build_fixture_with_correspondence(
        scene_dir,
        n_particles=n_particles,
        spacing_mm=spacing_mm,
        motion_mm=motion_mm,
        noise_px=noise_px,
        n_frames=n_frames,
        seed=seed,
    )
    return scene_dir, row_gt


def _load(scene_dir):
    y = str(scene_dir / "parameters_Run1.yaml")
    cpar = ControlPar.from_yaml(y)
    vpar = VolumePar.from_yaml(y)
    spar = SequencePar.from_yaml(y, cpar.num_cams)
    cals = []
    for c in range(cpar.num_cams):
        cal = Calibration()
        cal.from_file(
            str(scene_dir / f"cal/cam{c + 1}.tif.ori"),
            str(scene_dir / f"cal/cam{c + 1}.tif.addpar"),
        )
        cals.append(cal)
    for c in range(cpar.num_cams):
        spar.set_img_base_name(c, str(scene_dir / f"img_orig/cam{c + 1}.%d"))
    return cpar, vpar, spar, cals


def _run_track3d(tmp_path, scene_dir, first, last, **tpar_overrides):
    cpar, vpar, spar, cals = _load(scene_dir)
    tpar = TrackPar.from_yaml(str(scene_dir / "parameters_Run1.yaml"))
    for k, v in tpar_overrides.items():
        setattr(tpar, k, v)

    res = tmp_path / "res_track3d"
    res.mkdir(parents=True, exist_ok=True)
    for f in range(first, last + 1):
        shutil.copy(scene_dir / "res_orig" / f"rt_is.{f}", res / f"rt_is.{f}")
    naming = {
        "corres": str(res / "rt_is"),
        "linkage": str(res / "ptv_is"),
        "prio": str(res / "added"),
    }
    tr = Tracker(cpar, vpar, tpar, spar, cals, naming)
    tr.full_forward_3d()
    return naming["linkage"]


def _validate_by_identity(linkage_base, row_gt, first, last):
    """correct: link matches ground truth (same true pid both ends, not a
    ghost); wrong: link exists but the two ends disagree (includes any link
    touching a ghost row -- a ghost has no consistent identity to match)."""
    correct = wrong = 0
    lost = 0
    for f in range(first, last):
        with open(f"{linkage_base}.{f}") as fh:
            lines = fh.readlines()
        n = int(lines[0])
        gt_here = row_gt[f]
        gt_next = row_gt[f + 1]
        for row in range(n):
            parts = lines[row + 1].split()
            nxt = int(parts[1])
            pid_here = gt_here[row] if row < len(gt_here) else -1
            if nxt < 0:
                lost += 1
                continue
            pid_next = gt_next[nxt] if nxt < len(gt_next) else -1
            if pid_here >= 0 and pid_here == pid_next:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, lost


def test_ghost_fixture_actually_contains_ghosts():
    """Sanity check on the generator itself: at a density comparable to
    test_cavity, real correspondence matching must produce some ghost rows,
    or this benchmark is not exercising anything the ghost-free fixture
    doesn't already cover."""
    import tempfile

    scene_dir = Path(tempfile.mkdtemp()) / "scene"
    _frames, row_gt = build_fixture_with_correspondence(
        scene_dir, n_particles=200, spacing_mm=4.0, motion_mm=0.3,
        noise_px=1.0, n_frames=5, seed=2,
    )
    total_ghosts = sum(1 for pids in row_gt.values() for p in pids if p < 0)
    assert total_ghosts > 0, (
        "ghost-inclusive generator produced zero ghosts at test_cavity-like "
        "density -- the primary benchmark isn't reproducing the real-data "
        "failure mode"
    )


def test_tracking_shows_wrong_links_from_ghosts(tmp_path):
    """The actual Stage 0.5 success criterion: a tracker run on this fixture
    must show wrong links (or lost links caused by ghost contamination),
    which the ghost-free build_fixture-based benchmark structurally cannot
    show -- every row there is correct by construction."""
    n_frames = 6
    scene_dir, row_gt = _make_scene(
        tmp_path, n_particles=200, spacing_mm=4.0, motion_mm=0.3,
        noise_px=1.0, n_frames=n_frames, seed=2,
    )
    first, last = 10001, 10001 + n_frames - 1

    gate = 0.3 * 4  # matches test_tracking_synthetic_dense.py's convention
    linkage_base = _run_track3d(
        tmp_path, scene_dir, first, last,
        dvxmax=gate, dvxmin=-gate, dvymax=gate, dvymin=-gate,
        dvzmax=gate, dvzmin=-gate, dacc=gate * 0.4, dangle=90.0,
    )
    correct, wrong, lost = _validate_by_identity(linkage_base, row_gt, first, last)
    print(f"\nghost-inclusive test_cavity-like: correct={correct} wrong={wrong} lost={lost}")

    total_ghost_rows = sum(1 for pids in row_gt.values() for p in pids if p < 0)
    assert total_ghost_rows > 0, "fixture produced no ghosts -- test is not exercising anything"
    assert wrong > 0, (
        "tracker produced zero wrong links despite ghost-contaminated "
        "correspondences -- either the fixture's ghosts aren't reaching the "
        "tracker, or this regime is too easy to show the real-data failure "
        "mode; raise density or lower dacc/dv"
    )
