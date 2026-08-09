"""fast_3d's postprocess wiring is opt-in, gated by track.postprocess.

Regression for Stage 1c of the tracker consolidation roadmap
(docs/plans/master-plan.md): plugins/
default_tracking.py's fast_3d branch can now run Tracker.postprocess()
(seed_cold_start -> relink_trajectory_gaps -> enforce_reciprocity) after
full_forward_3d(), same as the full_multipass path already does. It
defaults OFF (matching tracking_presets.PRESET_CONFIGS["fast_3d"]
["postprocess"] == False) since it measurably is NOT cost-neutral on the
synthetic_turbulent benchmark: postprocess added a net 0-1 links while
costing 5-13x the tracking time (see the roadmap doc / bench_trackers.py
--dacc-sweep-style runs). This test only pins the wiring (the flag
actually gates the call); it does not assert a quality effect.
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


def test_postprocess_runs_when_flag_is_true(monkeypatch):
    calls = []
    original = Tracker.postprocess

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    monkeypatch.setattr(Tracker, "postprocess", spy)

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "fast_3d", track_overrides={"postprocess": True})
    assert calls == [1]


def test_postprocess_does_not_run_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(Tracker, "postprocess", lambda self, *a, **kw: calls.append(1))

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "fast_3d")
    assert calls == []
