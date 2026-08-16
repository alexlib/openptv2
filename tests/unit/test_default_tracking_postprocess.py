"""fast_3d's postprocess wiring is gated by track.postprocess, and now
defaults ON.

This test only pins the wiring (the flag actually gates the call); it does
not assert a quality effect.

History, because the default flipped: it used to default OFF, and the
recorded reason was that postprocess "added a net 0-1 links while costing
5-13x the tracking time". That was not a property of post-processing -- it
was the bug in it. ``relink_trajectory_gaps`` wrote a cross-frame link to
bridge a gap, and ``enforce_reciprocity``, running immediately after,
compared frame k only against k+1, so it severed every bridge right back out
(286 bridged, 286 severed, net zero). On top of that the bridging tolerance
was handed ``dvxmax`` -- a velocity gate -- for what is an
acceleration-scale residual.

Both are fixed (docs/plans/2026-08-16-tracking-next-steps.md §3.1, §3.2), and
the pass is now a measured win on both ground-truth synthetic sets: mean
trajectory length 7.13 -> 10.61 at 220 particles/frame and 8.18 -> 11.04 at
970, for under a point of precision. It is still not free (~20-40% wall);
set ``track.postprocess: false`` to opt out. See the table in
``tracking_presets.PRESET_CONFIGS``.
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


def test_postprocess_runs_by_default(monkeypatch):
    calls = []
    original = Tracker.postprocess

    def spy(self, *a, **kw):
        calls.append(1)
        return original(self, *a, **kw)

    monkeypatch.setattr(Tracker, "postprocess", spy)

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "fast_3d")
    assert calls == [1]


def test_postprocess_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(Tracker, "postprocess", lambda self, *a, **kw: calls.append(1))

    yaml_path = _tiny_yaml(Path(tempfile.mkdtemp()))
    bm.run_tracker(yaml_path, "fast_3d", track_overrides={"postprocess": False})
    assert calls == []
