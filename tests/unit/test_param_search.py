"""Tests for the coordinate-wise saturating parameter search.

``_saturating_1d`` is tested directly against a synthetic score function
(fast, deterministic) since its logic -- grow until saturated, then back
off -- doesn't depend on the tracker at all. ``find_smooth_params`` gets one
integration smoke test against a real (synthetic-scenario) tracker run.
"""

import tempfile
from pathlib import Path

import openptv2.benchmarking as bm
from openptv2.benchmarking.param_search import _saturating_1d, find_smooth_params


def test_saturating_1d_grows_then_backs_off_at_the_peak():
    """Score rises to a peak at value=4 then flattens; the search should
    stop shortly after the peak and back off to just before it."""
    # start=0.5, growth=2 -> 0.5, 1, 2, 4, 8, 16, ...
    scores = {0.5: 1.0, 1.0: 2.0, 2.0: 3.0, 4.0: 4.0, 8.0: 4.0, 16.0: 4.0}

    chosen = _saturating_1d(
        eval_fn=lambda v: scores[v],
        start=0.5,
        growth=2.0,
        max_steps=10,
        patience=2,
        backoff=1,
    )

    assert chosen == 2.0  # one step before the peak (4.0)


def test_saturating_1d_never_grows_when_already_at_the_peak():
    """If the very first value tried is already the best, stay there."""
    chosen = _saturating_1d(
        eval_fn=lambda v: -v,  # strictly decreasing: smaller is always better
        start=0.5,
        growth=2.0,
        max_steps=10,
        patience=2,
        backoff=1,
    )
    assert chosen == 0.5


def test_saturating_1d_backoff_never_goes_below_start():
    """backoff=5 with the peak at step 0 must clamp to the start value."""
    chosen = _saturating_1d(
        eval_fn=lambda v: -v,
        start=1.0,
        growth=2.0,
        max_steps=5,
        patience=1,
        backoff=5,
    )
    assert chosen == 1.0


def test_find_smooth_params_runs_end_to_end_on_synthetic_scenario():
    """Integration smoke test: runs against a real tracker, returns a
    finite score and a full per-step history for every parameter."""
    spec = bm.ScenarioSpec(
        num_particles=12,
        num_frames=6,
        velocity=1.0,
        gap_probability=0.0,
        noise_mm=0.0,
        seed=4,
    )
    _tt, fg = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    d = Path(tempfile.mkdtemp())
    yaml_path = bm.write_experiment(rig, fg, d, first_frame=10001)

    result = find_smooth_params(
        yaml_path,
        "fast_3d",
        dv_start=0.2,
        dacc_start=0.2,
        angle_start=10.0,
        growth=1.6,
        max_steps=6,
        patience=2,
        backoff=1,
    )

    assert result.dv >= 0.2 and result.dacc >= 0.2 and result.angle >= 10.0
    assert result.score > float("-inf")
    params_seen = {step.param for step in result.history}
    assert params_seen == {"dv", "dacc", "angle"}


if __name__ == "__main__":
    test_saturating_1d_grows_then_backs_off_at_the_peak()
    test_saturating_1d_never_grows_when_already_at_the_peak()
    test_saturating_1d_backoff_never_goes_below_start()
    test_find_smooth_params_runs_end_to_end_on_synthetic_scenario()
    print("ok")
