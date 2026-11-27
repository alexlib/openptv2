"""The synthetic dumbbell case (scripts/synthetic_dumbbell.py): cameras with a known truth, a bar of known length."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def _module():
    spec = importlib.util.spec_from_file_location("synthetic_dumbbell", ROOT / "scripts" / "synthetic_dumbbell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dumbbell_recovers_the_cameras(tmp_path):
    o = _module().run(tmp_path / "ok", frames=40, noise=0.1, length=46.0, seed=1, quiet=True)
    assert max(o["pos_before"]) > 2.0                     # the start is clearly off
    assert max(o["pos_after"]) < 0.3                      # ... and the fit finds the truth (mm)
    assert max(o["ang_after"]) < 0.05                     # degrees
    assert abs(o["len_after"] - 46.0) < 0.05              # the bar measured with the fitted cameras


def test_a_wrong_bar_length_gives_wrong_cameras(tmp_path):
    """Tell the program 47 mm for a 46 mm bar: the cameras bend to make the bar 47 mm."""
    o = _module().run(tmp_path / "bad", frames=40, noise=0.1, length=46.0, claimed_length=47.0, seed=1, quiet=True)
    assert abs(o["len_after"] - 47.0) < 0.05
    assert max(o["pos_after"]) > 3.0                     # mm: far worse than with the true length
