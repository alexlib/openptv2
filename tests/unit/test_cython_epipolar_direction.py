"""Regression test: CythonEpipolarTracker.do_tracking() (the plugin every
epipolar-family preset resolves to -- trackcorr, full_multipass,
standard_forward, two_directional, cython_epipolar_tracking,
openptv_epipolar; see plugins/loader.py's BUILTIN_TRACKING_PLUGINS) used to
call Tracker.full_forward() unconditionally, ignoring the preset's resolved
direction. "full_multipass"/"two_directional" are meant to run
forward+backward (tracking_presets._DIRECTION_BACKWARD_PRESETS) but
silently ran forward-only too, identically to every other preset here --
confirmed directly by benchmarking full_multipass against trackcorr and
finding byte-identical metrics and no extra wall time.

Fixed by delegating to plugins.default_tracking.Tracking.do_tracking(),
which already has the direction-aware dispatch (and is exercised by
test_default_tracking_postprocess.py for the fast_3d/postprocess side of
that same dispatch).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import openptv2.benchmarking as bm
from openptv2.tracker import Tracker


def _tiny_yaml(tmp_path: Path) -> Path:
    spec = bm.ScenarioSpec(num_particles=10, num_frames=6, velocity=1.0, seed=7)
    _, fg = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    return bm.write_experiment(rig, fg, tmp_path, first_frame=10001)


def test_full_multipass_runs_backward_pass(monkeypatch):
    calls = []
    original = Tracker.full_backward

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    monkeypatch.setattr(Tracker, "full_backward", spy)

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "full_multipass")
    assert calls == [1]


def test_trackcorr_forward_only_does_not_run_backward_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Tracker, "full_backward", lambda self, *a, **kw: calls.append(1)
    )

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "trackcorr")
    assert calls == []


def test_cython_epipolar_tracking_runs_trackcorr_not_track3d(monkeypatch):
    """Regression test: runner.py's _CORE_PRESETS allow-list (which controls
    whether run_tracker actually sets selected_tracking in the experiment
    config) was missing "cython_epipolar_tracking" -- harmless before the
    do_tracking() fix above, since the plugin ignored selected_tracking
    entirely, but once it started honoring it, infer_tracker fell back to
    the untouched "default" -> "priority_segment_3d" and silently ran the
    Fast 3D engine instead of trackcorr. Confirmed directly: benchmarking
    "cython_epipolar_tracking" this way produced Fast-3D-identical metrics
    instead of trackcorr's."""
    calls = []
    monkeypatch.setattr(
        Tracker, "full_forward_3d", lambda self, *a, **kw: calls.append(1)
    )

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "cython_epipolar_tracking")
    assert calls == []
