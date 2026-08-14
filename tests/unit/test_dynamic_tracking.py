import pytest

from openptv2.algorithms.parameters import TrackPar
from openptv2.dynamic_tracking import (
    DynamicTrackParams,
    flag_low_quality_steps,
    resolve_dynamic_params_path,
)


@pytest.mark.unit
def test_dynamic_track_params_falls_back_to_base_when_no_override():
    base = TrackPar(dvxmin=-5.0, dvxmax=5.0, dacc=2.0)
    dyn = DynamicTrackParams(base, steps={})

    tp = dyn.get(42)

    assert tp is base


@pytest.mark.unit
def test_dynamic_track_params_overrides_only_given_fields():
    base = TrackPar(dvxmin=-5.0, dvxmax=5.0, dvymin=-5.0, dvymax=5.0, dacc=2.0)
    dyn = DynamicTrackParams(base, steps={7: {"dvxmax": 25.0, "dacc": 12.0}})

    tp = dyn.get(7)

    assert tp.dvxmax == 25.0
    assert tp.dacc == 12.0
    # Untouched fields carry over from the base:
    assert tp.dvxmin == -5.0
    assert tp.dvymax == 5.0
    # Base itself is unmodified:
    assert base.dvxmax == 5.0


@pytest.mark.unit
def test_flag_low_quality_steps_auto_threshold():
    rates = {1: 0.95, 2: 0.96, 3: 0.30, 4: 0.94}

    flagged = flag_low_quality_steps(rates)

    assert flagged == [3]


@pytest.mark.unit
def test_flag_low_quality_steps_explicit_threshold():
    rates = {1: 0.5, 2: 0.9}

    assert flag_low_quality_steps(rates, threshold=0.6) == [1]
    assert flag_low_quality_steps(rates, threshold=0.1) == []


@pytest.mark.unit
def test_resolve_dynamic_params_path_default_and_override(tmp_path):
    default_path = resolve_dynamic_params_path({}, tmp_path)
    assert default_path == tmp_path / "dynamic_track.yaml"

    custom_path = resolve_dynamic_params_path(
        {"dynamic_params_file": "custom.yaml"}, tmp_path
    )
    assert custom_path == tmp_path / "custom.yaml"
