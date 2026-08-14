"""assess_tracking_conditioning must classify a fast/well-separated flow as
well-conditioned and a slow flow with poor camera z-sensitivity as
poorly-conditioned -- the actual failure mode measured on test_cavity this
session (true motion ~0.3mm/frame vs ~0.47mm z-noise floor -> ratio 1.38,
poorly-conditioned)."""

import numpy as np
import pytest

from openptv2.tracking_feasibility import (
    assess_tracking_conditioning,
    measure_motion_scale,
)


def test_measure_motion_scale_ignores_ghost_noise_via_low_percentile():
    rng = np.random.default_rng(0)
    # Genuine matches: small, tight displacement.
    a_real = rng.uniform(-50, 50, (200, 3))
    b_real = a_real + rng.normal(0, 0.1, (200, 3))
    # Ghosts: unrelated points scattered independently in frame b.
    b_ghost = rng.uniform(-50, 50, (300, 3))
    b = np.concatenate([b_real, b_ghost])

    displacement, spacing = measure_motion_scale(a_real, b)
    assert displacement < 1.0  # not dragged up by the ghost population
    assert spacing > 0


def test_too_few_points_returns_none():
    assert measure_motion_scale(np.zeros((2, 3)), np.zeros((2, 3))) is None


def test_assess_tracking_conditioning_flags_slow_flow_weak_z_as_poor(monkeypatch):
    rng = np.random.default_rng(1)
    a = rng.uniform(-50, 50, (500, 3))
    b = a + rng.normal(0, 0.1, (500, 3))  # slow flow

    monkeypatch.setattr(
        "openptv2.tracking_feasibility.z_noise_floor_mm", lambda *a, **k: 2.0
    )
    report = assess_tracking_conditioning(a, b, cals=[], cpar=None)
    assert report.verdict == "poorly-conditioned"
    assert report.ratio > 1.0


def test_assess_tracking_conditioning_flags_fast_flow_as_well_conditioned(monkeypatch):
    rng = np.random.default_rng(2)
    a = rng.uniform(-50, 50, (500, 3))
    b = a + rng.normal([3.0, 0, 0], 0.1, (500, 3))  # fast flow

    monkeypatch.setattr(
        "openptv2.tracking_feasibility.z_noise_floor_mm", lambda *a, **k: 0.1
    )
    report = assess_tracking_conditioning(a, b, cals=[], cpar=None)
    assert report.verdict == "well-conditioned"
    assert report.ratio < 0.3


def test_assess_tracking_conditioning_returns_none_with_too_little_data():
    assert (
        assess_tracking_conditioning(np.zeros((1, 3)), np.zeros((1, 3)), [], None)
        is None
    )
