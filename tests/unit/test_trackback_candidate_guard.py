"""Regression test for the trackback_loop_fast acceptance-guard bug.

track_kernels_corr.py's backward pass (trackback_loop_fast) previously
de-dented the acceptance test

    if (acc < dacc and angle < dangle) or acc < dacc * 0.1:

so only the ``d13`` distance assignment was gated by it; the rest of the
body (``d01``, ``dl``, ``rr``, and the append to ``path_decis_1`` /
``path_linkdecis_1``) executed unconditionally for every candidate that
merely fell inside the velocity box, using a stale (function-scope, often
zero) ``d13``. That means a spatially-nearby but kinematically-wrong
candidate got registered as a link candidate regardless of the
acceleration/angle test -- and if it was the *only* candidate found for a
track head, the corrupted list caused a spurious link where none should
exist. The forward pass (L825-844) always kept the whole body inside the
``if``, matching the original C ``track.c``.

This test builds a minimal two-particle, three-frame scene, tuned so a
late-entry particle's true backward continuation does not exist (nothing
should link), but a "bait" particle sits within the (generously wide,
default) velocity search box while clearly failing the acc/angle test.
Pre-fix, the bait gets linked anyway (the only candidate in an otherwise
non-empty list always "wins"); post-fix, it is correctly excluded and the
particle stays unlinked.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.track import (
    point_to_pixel,
    track_forward_start,
    trackback_c,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.tracking_run import tr_new

SYNTHETIC_DIR = (
    Path(__file__).resolve().parent.parent.parent / "test_data" / "synthetic"
)
NUM_CAMS = 4
F0, F1, F2, F3 = 10001, 10002, 10003, 10004

# Particle "A": a late entry. Present only at F1, F2, F3 with constant
# velocity (3, 0, 0) -- three points so the forward 4-point criterion has
# enough lookahead to link it cleanly. Forward tracking gives it
# next(F1) = F2, prev(F1) = -1 (a genuine track head for the backward
# pass), and its true backward continuation at F0 does not exist --
# nothing should link there.
A_F1 = np.array([0.0, 0.0, 0.0])
A_F2 = np.array([3.0, 0.0, 0.0])
A_F3 = np.array([6.0, 0.0, 0.0])
# Backward-predicted point: 2*A_F1 - A_F2 = (-3, 0, 0).

# Particle "B": the bait. Present only at F0, offset from the predicted
# point so it falls inside the default dv box (+-10 in every axis) but far
# enough from the prediction to fail the default dacc=5.0 test on both
# angle and magnitude (acc = |B - predicted| = 9, angle ~ 80 gon).
B_F0 = np.array([-3.0, 9.0, 0.0])


def _write_frame(res_dir, img_dir, frame, particles, cals, cpar):
    """Write rt_is/ptv_is/added/target files for one frame.

    ``particles`` is a list of (x, y, z) triples. Targets are written
    y-sorted per camera (candsearch_in_pix relies on that ordering).
    """
    n = len(particles)

    cam_slot_to_targ = {}
    cam_targ_entries = {}
    for cam in range(NUM_CAMS):
        entries = []
        for slot, (x, y, z) in enumerate(particles):
            pos = np.array([x, y, z], dtype=np.float64)
            px, py = point_to_pixel(pos, cals[cam], cpar)
            entries.append((slot, px, py))
        entries.sort(key=lambda t: t[2])
        cam_targ_entries[cam] = entries
        cam_slot_to_targ[cam] = {
            slot: targ_idx for targ_idx, (slot, px, py) in enumerate(entries)
        }

    with open(res_dir / f"rt_is.{frame}", "w") as fh:
        fh.write(f"{n}\n")
        for slot, (x, y, z) in enumerate(particles):
            cam_indices = " ".join(
                f"{cam_slot_to_targ[cam][slot]:4d}" for cam in range(NUM_CAMS)
            )
            fh.write(f"{slot + 1:4d} {x:9.3f} {y:9.3f} {z:9.3f} {cam_indices}\n")

    with open(res_dir / f"ptv_is.{frame}", "w") as fh:
        fh.write(f"{n}\n")
        for x, y, z in particles:
            fh.write(f"  -1   -2 {x:10.3f} {y:10.3f} {z:10.3f}\n")

    with open(res_dir / f"added.{frame}", "w") as fh:
        fh.write(f"{n}\n")
        for x, y, z in particles:
            fh.write(f"  -1   -2 {x:10.3f} {y:10.3f} {z:10.3f} 4\n")

    for cam in range(NUM_CAMS):
        entries = cam_targ_entries[cam]
        with open(img_dir / f"cam{cam + 1}.{frame}_targets", "w") as fh:
            fh.write(f"{n}\n")
            for targ_pnr, (orig_slot, px, py) in enumerate(entries):
                fh.write(
                    f"{targ_pnr:4d} {px:9.4f} {py:9.4f} "
                    f"  100    10    10  1000 {orig_slot:5d}\n"
                )


@pytest.fixture
def bait_scene(tmp_path):
    """Build the two-particle bait scene in a throwaway working directory."""
    res_dir = tmp_path / "res"
    img_dir = tmp_path / "img"
    res_dir.mkdir()
    img_dir.mkdir()

    cals = [
        Calibration.from_file(
            str(SYNTHETIC_DIR / f"cal/cam{i + 1}.tif.ori"),
            str(SYNTHETIC_DIR / f"cal/cam{i + 1}.tif.addpar"),
        )
        for i in range(NUM_CAMS)
    ]
    yaml_path = SYNTHETIC_DIR / "parameters.yaml"
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    tpar = TrackPar.from_yaml(str(yaml_path))  # default dv=+-10, dacc=5.0, dangle=120

    _write_frame(res_dir, img_dir, F0, [tuple(B_F0)], cals, cpar)
    _write_frame(res_dir, img_dir, F1, [tuple(A_F1)], cals, cpar)
    _write_frame(res_dir, img_dir, F2, [tuple(A_F2)], cals, cpar)
    _write_frame(res_dir, img_dir, F3, [tuple(A_F3)], cals, cpar)

    seq_par = SequencePar(
        num_cams=NUM_CAMS,
        img_base_name=[f"img/cam{i + 1}." for i in range(NUM_CAMS)],
        first=F0,
        last=F3,
    )

    return {
        "tmp_path": tmp_path,
        "seq_par": seq_par,
        "tpar": tpar,
        "vpar": vpar,
        "cpar": cpar,
        "cals": cals,
    }


def test_backward_pass_rejects_out_of_criteria_candidate(bait_scene):
    """The bait particle must NOT become particle A's prev link.

    A's true backward continuation at F0 does not exist. The bait sits
    inside the velocity box but fails acc/angle -- pre-fix, the buggy
    guard registered it anyway and it was the only (hence winning)
    candidate; post-fix, A stays unlinked at F0.
    """
    original = os.getcwd()
    try:
        os.chdir(bait_scene["tmp_path"])

        run = tr_new(
            bait_scene["seq_par"],
            bait_scene["tpar"],
            bait_scene["vpar"],
            bait_scene["cpar"],
            4,
            20,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            bait_scene["cals"],
            0.0001,
        )

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        trackback_c(run)

        with open("res/ptv_is." + str(F1)) as fh:
            lines = fh.readlines()
        n = int(lines[0])
        assert n == 1, "expected exactly one particle (A) at F1"
        prev, next_ = (int(v) for v in lines[1].split()[:2])

        assert next_ == 0, "forward link A(F1)->A(F2) must still be present"
        assert prev == -1, (
            "backward pass linked particle A to the bait candidate despite "
            "it failing the acc/angle acceptance test"
        )

    finally:
        os.chdir(original)
