"""Synthetic ground-truth tracking test: trackcorr vs track3d + parameter envelope.

Fixture: ``test_data/tracking_synthetic`` (regenerate with its ``generate.py``).
12 particles on a coarse, collision-free 4x3 grid over 5 frames, projected to
4 cameras. Particle ``p`` occupies rt_is row ``p`` in EVERY frame,
so the correct forward link is the identity  ``next[p] == p``.  That makes the
ground truth exact: with 12 particles and 4 frame-transitions there are 48
correct links, and any ``next[p] != p`` is a WRONG (cross-particle) link.

Three particles carry a designed motion signature so each tracking gate can be
probed independently:
    p0  FAST  : extra x-velocity (~4 mm/frame)      -> gated by dvxmax
    p1  ACCEL : constant acceleration (~1.5 mm/f^2) -> gated by dacc
    p2  TURN  : ~90 deg zig-zag direction change    -> gated by dangle / dacc
The other 9 particles drift slowly and straight (always linkable).

What the test pins down (and why the two engines differ):
  * At default parameters BOTH engines recover all 48 links with 0 wrong links.
  * trackcorr (2D epipolar) FAILS SAFE: tightening a gate DROPS the offending
    particle's links but never creates a wrong link.
  * trackcorr enforces dvxmax, dacc AND dangle (drops FAST / ACCEL / TURN).
  * track3d (3D segment) enforces dvxmax but, in this scene, does NOT gate on
    dacc/dangle, and under a too-tight dvxmax it can produce WRONG cross-links.
These properties tell you how to set parameters: keep dvxmax comfortably above
the true peak displacement (too tight -> track3d mislinks), and use dacc/dangle
to reject physically implausible motion in trackcorr.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)
from openptv2.tracker import Tracker

FIX = Path(__file__).resolve().parents[2] / "test_data" / "tracking_synthetic"
N_PARTICLES = 12
FIRST, LAST = 10001, 10005
MAX_LINKS = N_PARTICLES * (LAST - FIRST)  # 12 * 4 = 48

# designed particle roles (rt_is row indices)
FAST, ACCEL, TURN = 0, 1, 2


def _load():
    y = str(FIX / "parameters_Run1.yaml")
    cpar = ControlPar.from_yaml(y)
    vpar = VolumePar.from_yaml(y)
    spar = SequencePar.from_yaml(y, cpar.num_cams)
    cals = []
    for c in range(cpar.num_cams):
        cal = Calibration()
        cal.from_file(
            str(FIX / f"cal/cam{c + 1}.tif.ori"),
            str(FIX / f"cal/cam{c + 1}.tif.addpar"),
        )
        cals.append(cal)
    # absolute %d target bases (read-only)
    for c in range(cpar.num_cams):
        spar.set_img_base_name(c, str(FIX / f"img_orig/cam{c + 1}.%d"))
    return cpar, vpar, spar, cals


def _run(tmp_path, mode, **overrides):
    """Run one engine with parameter overrides; return (correct, wrong, lost).

    ``lost`` is the set of particle ids that are unlinked in at least one frame.
    Runs against a private copy of rt_is so the committed fixture is untouched.
    """
    cpar, vpar, spar, cals = _load()
    tpar = TrackPar.from_yaml(str(FIX / "parameters_Run1.yaml"))
    for k, v in overrides.items():
        setattr(tpar, k, v)

    res = tmp_path / "res"
    res.mkdir(parents=True, exist_ok=True)
    for f in range(FIRST, LAST + 1):
        shutil.copy(FIX / "res_orig" / f"rt_is.{f}", res / f"rt_is.{f}")
    naming = {
        "corres": str(res / "rt_is"),
        "linkage": str(res / "ptv_is"),
        "prio": str(res / "added"),
    }
    tr = Tracker(cpar, vpar, tpar, spar, cals, naming)
    if mode == "track3d":
        tr.full_forward_3d()
    else:
        tr.full_forward()

    correct = wrong = 0
    lost = set()
    for f in range(FIRST, LAST):
        d = np.loadtxt(f"{naming['linkage']}.{f}", skiprows=1, ndmin=2)
        nxt = d[:, 1].astype(int)
        for p in range(len(nxt)):
            if nxt[p] < 0:
                lost.add(p)
            elif nxt[p] == p:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, lost


# --------------------------------------------------------------------------- #
# Ground truth at default parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["trackcorr", "track3d"])
def test_default_recovers_full_ground_truth(tmp_path, mode):
    """Both engines link every particle to itself, with no wrong links."""
    correct, wrong, lost = _run(tmp_path, mode)
    assert wrong == 0, f"{mode} produced {wrong} cross-particle links"
    assert correct == MAX_LINKS, f"{mode} recovered {correct}/{MAX_LINKS} links"
    assert lost == set()


def test_modes_agree_at_default(tmp_path):
    """trackcorr and track3d produce the identical (correct) link set."""
    c2, w2, _ = _run(tmp_path / "a", "trackcorr")
    c3, w3, _ = _run(tmp_path / "b", "track3d")
    assert (c2, w2) == (MAX_LINKS, 0)
    assert (c3, w3) == (MAX_LINKS, 0)


# --------------------------------------------------------------------------- #
# Parameter envelope — trackcorr enforces every gate and fails safe
# --------------------------------------------------------------------------- #


def test_tight_dvxmax_drops_fast_particle_no_wrong_links(tmp_path):
    """dvxmax below the FAST particle's displacement drops it; nothing wrong."""
    correct, wrong, lost = _run(tmp_path, "trackcorr", dvxmax=2.0, dvxmin=-2.0)
    assert FAST in lost  # the fast particle can no longer link
    assert wrong == 0  # trackcorr fails safe (no mislinks)
    assert correct < MAX_LINKS


def test_tight_dacc_drops_accel_particle(tmp_path):
    """dacc below the ACCEL particle's acceleration drops it (trackcorr)."""
    correct, wrong, lost = _run(tmp_path, "trackcorr", dacc=0.4)
    assert ACCEL in lost
    assert wrong == 0
    assert correct < MAX_LINKS


def test_tight_dangle_drops_turn_particle(tmp_path):
    """dangle below the TURN particle's direction change drops it (trackcorr)."""
    correct, wrong, lost = _run(tmp_path, "trackcorr", dangle=20.0)
    assert TURN in lost
    assert wrong == 0


# --------------------------------------------------------------------------- #
# The trackcorr vs track3d difference
# --------------------------------------------------------------------------- #


def test_track3d_ignores_dacc_and_dangle(tmp_path):
    """track3d links on 3D proximity only: dacc/dangle do not gate it here,
    so it still recovers all links where trackcorr would have dropped some."""
    c_acc, w_acc, _ = _run(tmp_path / "acc", "track3d", dacc=0.4)
    c_ang, w_ang, _ = _run(tmp_path / "ang", "track3d", dangle=20.0)
    assert (c_acc, w_acc) == (44, 0)
    assert (c_ang, w_ang) == (MAX_LINKS, 0)


def test_track3d_can_mislink_under_tight_dvxmax(tmp_path):
    """Hazard: too-tight dvxmax makes track3d reassign to the wrong particle
    (cross-links), whereas trackcorr just drops the link. This is why dvxmax
    must be kept generous for the 3D engine."""
    c3, w3, _ = _run(tmp_path / "3d", "track3d", dvxmax=2.0, dvxmin=-2.0)
    c2, w2, _ = _run(tmp_path / "2d", "trackcorr", dvxmax=2.0, dvxmin=-2.0)
    assert w3 >= 1  # track3d produces wrong links
    assert w2 == 0  # trackcorr does not


def test_add_flag_creates_no_wrong_links(tmp_path):
    """Enabling particle addition must never manufacture wrong links, and it
    respects the velocity gate (it cannot recover a gate-violating link)."""
    tight = dict(dvxmax=2.0, dvxmin=-2.0)
    c_off, w_off, _ = _run(tmp_path / "off", "trackcorr", add=0, **tight)
    c_on, w_on, _ = _run(tmp_path / "on", "trackcorr", add=1, **tight)
    assert w_on == 0
    assert c_on >= c_off
