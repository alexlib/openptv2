"""Per-step ("dynamic") tracking parameters, for flows too transient/periodic
for one constant search window across the whole run.

Static tracking (the default, unchanged) uses one TrackPar for every frame
transition. Opt in per-experiment with `track.dynamic_tracking: true` in the
YAML -- absent by default, so existing YAMLs are unaffected. When enabled,
`Tracker.step_forward()`/`step_forward_3d()` swap in a per-step TrackPar
loaded from a small sidecar YAML (default: `dynamic_track.yaml` next to the
experiment YAML; override with `track.dynamic_params_file`), falling back to
the base (static) TrackPar for any step with no override:

    steps:
      117: {dvxmin: -25.0, dvxmax: 25.0, dacc: 12.0}
      118: {dvxmin: -25.0, dvxmax: 25.0, dacc: 12.0}

`openptv tune-dynamic` generates that file automatically: it runs one static
tracking pass, flags frames with an anomalously low link rate (a real particle
that failed to link forward is exactly what widening dv/dacc for that step
fixes -- no ground truth needed), and derives each flagged step's bounds from
a local window of its own rt_is data (reusing tracking_recommender's
percentile sizing, just windowed around the step instead of averaged over the
whole, possibly periodic, run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from openptv2.algorithms.parameters import TrackPar

_OVERRIDABLE_FIELDS = (
    "dvxmin",
    "dvxmax",
    "dvymin",
    "dvymax",
    "dvzmin",
    "dvzmax",
    "dangle",
    "dacc",
    "add",
)


def _clone_trackpar(base: TrackPar) -> TrackPar:
    """Explicit-field copy: TrackPar is a Cython cclass, not copy.copy-safe."""
    return TrackPar(
        dvxmin=base.dvxmin,
        dvxmax=base.dvxmax,
        dvymin=base.dvymin,
        dvymax=base.dvymax,
        dvzmin=base.dvzmin,
        dvzmax=base.dvzmax,
        dangle=base.dangle,
        dacc=base.dacc,
        add=base.add,
        track_mode=base.track_mode,
        w_vel=base.w_vel,
        w_acc=base.w_acc,
        w_intensity=base.w_intensity,
    )


class DynamicTrackParams:
    """Per-step TrackPar overrides, falling back to a base (static) TrackPar."""

    def __init__(self, base: TrackPar, steps: dict[int, dict[str, float]]):
        self.base = base
        self.steps = steps

    def get(self, step: int) -> TrackPar:
        overrides = self.steps.get(step)
        if not overrides:
            return self.base
        tp = _clone_trackpar(self.base)
        for key, value in overrides.items():
            if key in _OVERRIDABLE_FIELDS:
                setattr(tp, key, value)
        return tp

    @staticmethod
    def from_yaml(path: str | Path, base: TrackPar) -> "DynamicTrackParams":
        data = yaml.safe_load(Path(path).read_text()) or {}
        raw_steps = data.get("steps", {}) or {}
        steps = {int(k): dict(v) for k, v in raw_steps.items()}
        return DynamicTrackParams(base, steps)


def resolve_dynamic_params_path(track_cfg: dict[str, Any], yaml_dir: Path) -> Path:
    """Sidecar file path: `track.dynamic_params_file` if set, else
    `dynamic_track.yaml` next to the experiment YAML."""
    filename = track_cfg.get("dynamic_params_file", "dynamic_track.yaml")
    p = Path(filename)
    return p if p.is_absolute() else yaml_dir / p


# ---------------------------------------------------------------------------
# `openptv tune-dynamic`: static run -> per-step loss -> suggested overrides.
# ---------------------------------------------------------------------------


def per_frame_link_rate(linkage_base: str | Path, first: int, last: int) -> dict[int, float]:
    """Fraction of particles in frame k with a forward link, for k in
    [first, last). A ground-truth-free tracking-quality proxy: a real
    particle that failed to link is exactly what widening dv/dacc fixes.
    """
    from openptv2.tracking_postprocess import read_linkage

    rates: dict[int, float] = {}
    for k in range(first, last):
        r = read_linkage(str(linkage_base), k)
        if r is None:
            continue
        _prev, nxt, _xyz = r
        n = len(nxt)
        rates[k] = float((nxt >= 0).sum()) / n if n > 0 else 1.0
    return rates


def flag_low_quality_steps(
    rates: dict[int, float], threshold: float | None = None, z: float = 1.0
) -> list[int]:
    """Steps whose link rate is anomalously low vs. this run's own
    distribution (mean - z*std), or below an explicit threshold."""
    if not rates:
        return []
    vals = np.array(list(rates.values()))
    cut = threshold if threshold is not None else max(0.0, float(vals.mean() - z * vals.std()))
    return sorted(k for k, v in rates.items() if v < cut)


def suggest_step_overrides(
    rt_is_dir: str | Path,
    flagged_steps: list[int],
    first: int,
    last: int,
    num_cams: int,
    tracker_name: str = "priority_segment_3d",
    window: int = 5,
) -> dict[int, dict[str, float]]:
    """Local-window kinematic bounds for each flagged step: the same
    percentile-based sizing tracking_recommender uses for a whole sequence,
    windowed around the step instead of averaged over the whole run."""
    from openptv2.algorithms.tracking_frame_buf import Frame
    from openptv2.tracking_recommender import _suggest_params, compute_dataset_stats
    from openptv2.tracking_registry import get_tracker_info

    info = get_tracker_info(tracker_name)
    rt_is_dir = Path(rt_is_dir)
    corres_base = str(rt_is_dir / "rt_is")

    overrides: dict[int, dict[str, float]] = {}
    for step in flagged_steps:
        lo, hi = max(first, step - window), min(last, step + window)
        frame_particles = []
        for fn in range(lo, hi + 1):
            f = rt_is_dir / f"rt_is.{fn}"
            if not f.exists():
                frame_particles.append(np.empty((0, 3)))
                continue
            frame = Frame(num_cams=num_cams, max_targets=10000)
            frame.read(corres_base, "", target_file_base="", frame_num=fn)
            frame_particles.append(frame.positions())

        stats = compute_dataset_stats(frame_particles)
        params = _suggest_params(info, stats)
        overrides[step] = {
            k: v
            for k, v in params.items()
            if k in ("dvxmin", "dvxmax", "dvymin", "dvymax", "dvzmin", "dvzmax", "dacc")
        }
    return overrides


def write_dynamic_params_yaml(path: str | Path, steps: dict[int, dict[str, float]]) -> None:
    data = {
        "steps": {
            int(k): {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in steps.items()
        }
    }
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


__all__ = [
    "DynamicTrackParams",
    "resolve_dynamic_params_path",
    "per_frame_link_rate",
    "flag_low_quality_steps",
    "suggest_step_overrides",
    "write_dynamic_params_yaml",
]
