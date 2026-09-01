"""Coordinate-wise saturating search for tracking dv/dacc/angle.

Not gradient descent: ``run_tracker`` is a discrete, non-differentiable
black box (candidate search box + greedy cost-ordered assignment), so there
is no usable gradient and small parameter perturbations can produce
discontinuous jumps in the output. Instead this automates the manual
heuristic of starting a parameter small and growing it until the tracked
trajectories' "fluidity" stops improving, then backing off a step or two --
robust to that noise/discreteness without needing derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from openptv2.benchmarking.metrics import PhysicsMetrics, compute_physics_metrics
from openptv2.benchmarking.runner import run_tracker


def fluidity_score(
    pm: PhysicsMetrics, kurtosis_weight: float = 2.0, kurtosis_target: float = 3.0
) -> float:
    """Higher = longer, smoother trajectories.

    Rewards mean track length; penalizes acceleration kurtosis above the
    Gaussian target of ~3 -- heavy tails mean some fraction of the
    trajectory jumped implausibly relative to the rest of it (mid-track
    identity swaps), not genuine physical acceleration.
    """
    if pm.n_tracks == 0 or pm.mean_track_length <= 0:
        return float("-inf")
    penalty = 0.0
    if pm.acceleration_kurtosis == pm.acceleration_kurtosis:  # not NaN
        penalty = kurtosis_weight * max(0.0, pm.acceleration_kurtosis - kurtosis_target)
    return pm.mean_track_length - penalty


@dataclass
class SearchStep:
    param: str
    value: float
    score: float
    pm: PhysicsMetrics


@dataclass
class SearchResult:
    dv: float
    dacc: float
    angle: float
    score: float
    history: list[SearchStep] = field(default_factory=list)


def _saturating_1d(
    eval_fn: Callable[[float], float],
    start: float,
    growth: float,
    max_steps: int,
    patience: int,
    backoff: int,
) -> float:
    """Grow a value geometrically from ``start``; stop ``patience`` steps
    after the score last improved; return the value ``backoff`` steps
    before the stopping point (never below ``start``)."""
    value = start
    history: list[tuple[float, float]] = []
    best_i = 0
    best_score = float("-inf")
    for i in range(max_steps):
        score = eval_fn(value)
        history.append((value, score))
        if score > best_score:
            best_score = score
            best_i = i
        elif i - best_i >= patience:
            break
        value *= growth
    chosen_i = max(0, best_i - backoff)
    return history[chosen_i][0]


def find_smooth_params(
    yaml_path: str | Path,
    tracker: str,
    dv_start: float = 0.1,
    dacc_start: float = 0.1,
    angle_start: float = 10.0,
    growth: float = 1.6,
    max_steps: int = 12,
    patience: int = 2,
    backoff: int = 1,
) -> SearchResult:
    """Coordinate-wise saturating search over dv, then dacc, then angle.

    Each parameter is grown geometrically from a small start value while
    the other two are held at their already-chosen (or start) value,
    stopping once :func:`fluidity_score` stops improving for ``patience``
    steps, then backing off ``backoff`` steps -- "increase until results
    saturate, then step back", automated.
    """
    history: list[SearchStep] = []
    dv, dacc, angle = dv_start, dacc_start, angle_start

    def run(dv_v: float, dacc_v: float, angle_v: float) -> PhysicsMetrics:
        ov = {
            "dvxmax": dv_v,
            "dvxmin": -dv_v,
            "dvymax": dv_v,
            "dvymin": -dv_v,
            "dvzmax": dv_v,
            "dvzmin": -dv_v,
            "dacc": dacc_v,
            "angle": angle_v,
        }
        pred = run_tracker(yaml_path, tracker, track_overrides=ov)
        return compute_physics_metrics(pred)

    def make_eval(
        param_name: str, get_triple: Callable[[float], tuple[float, float, float]]
    ):
        def eval_fn(value: float) -> float:
            pm = run(*get_triple(value))
            score = fluidity_score(pm)
            history.append(SearchStep(param_name, value, score, pm))
            return score

        return eval_fn

    dv = _saturating_1d(
        make_eval("dv", lambda v: (v, dacc, angle)),
        dv_start,
        growth,
        max_steps,
        patience,
        backoff,
    )
    dacc = _saturating_1d(
        make_eval("dacc", lambda v: (dv, v, angle)),
        dacc_start,
        growth,
        max_steps,
        patience,
        backoff,
    )
    angle = _saturating_1d(
        make_eval("angle", lambda v: (dv, dacc, v)),
        angle_start,
        growth,
        max_steps,
        patience,
        backoff,
    )

    final_pm = run(dv, dacc, angle)
    return SearchResult(
        dv=dv, dacc=dacc, angle=angle, score=fluidity_score(final_pm), history=history
    )


__all__ = ["fluidity_score", "find_smooth_params", "SearchResult", "SearchStep"]
