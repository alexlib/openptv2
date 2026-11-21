"""Unit tests for openptv2.tracking_registry."""

import pytest

from openptv2.tracking_registry import (
    TRACKER_REGISTRY,
    ParameterGuide,
    TrackerInfo,
    get_tracker_info,
    list_trackers,
    print_tracker_detail,
    print_tracker_table,
)


def test_registry_has_exactly_the_three_trackers():
    """Three trackers, default first; old names are translated, not listed."""
    assert list(TRACKER_REGISTRY) == ["two_phase", "trackcorr", "fast_3d"]


def test_registry_does_not_contain_hybrid():
    """Verify hybrid_3d_corr was removed from the registry."""
    assert "hybrid_3d_corr" not in TRACKER_REGISTRY


def test_every_tracker_has_required_fields():
    """Verify every TrackerInfo has non-empty identity fields."""
    for name, info in TRACKER_REGISTRY.items():
        assert info.display_name, f"{name}: missing display_name"
        assert info.short_description, f"{name}: missing short_description"
        assert info.algorithm_summary, f"{name}: missing algorithm_summary"
        assert info.speed_ranking in ("fastest", "fast", "moderate", "slow")
        assert info.accuracy_ranking in ("draft", "standard", "high", "highest")
        assert info.density_ranking in ("low", "low_to_moderate", "moderate", "high")


def test_get_tracker_info():
    """Lookup works for the three names and translates the old ones."""
    assert get_tracker_info("fast_3d").name == "fast_3d"
    for old, new in (
        ("priority_segment_3d", "fast_3d"),
        ("fast", "fast_3d"),
        ("full_multipass", "trackcorr"),
        ("standard_forward", "trackcorr"),
        ("holistic", "two_phase"),
        ("unified", "two_phase"),
    ):
        assert get_tracker_info(old).name == new, old


def test_get_tracker_info_unknown():
    """Verify unknown name returns None."""
    assert get_tracker_info("nonexistent") is None


def test_list_trackers():
    """Verify list_trackers returns all entries."""
    trackers = list_trackers()
    assert len(trackers) == len(TRACKER_REGISTRY)


def test_parameter_guide_defaults():
    """Verify ParameterGuide dataclass works."""
    pg = ParameterGuide(
        name="dvxmax",
        type="float",
        default="15.5",
        description="Max velocity in X.",
    )
    assert pg.name == "dvxmax"
    assert pg.how_to_choose == ""


def test_print_tracker_table():
    """Verify table printing returns a non-empty string."""
    table = print_tracker_table()
    assert len(table) > 100
    assert "Name" in table
    assert "Speed" in table


def test_print_tracker_detail():
    """Verify detail printing returns a non-empty string."""
    detail = print_tracker_detail("fast_3d")
    assert detail
    assert "Algorithm" in detail
    assert "Capabilities" in detail
    assert "Parameters" in detail


def test_print_tracker_detail_unknown(capsys):
    """Verify unknown tracker name returns error message."""
    result = print_tracker_detail("unknown_tracker")
    assert "Unknown tracker" in result


def test_every_tracker_supports_both_directions():
    for info in list_trackers():
        assert info.supports_backward is True, info.name


def test_capabilities_that_differ():
    assert get_tracker_info("fast_3d").supports_2d is False
    assert get_tracker_info("two_phase").supports_2d is True
    assert get_tracker_info("trackcorr").supports_postprocessing is True
    assert get_tracker_info("fast_3d").speed_ranking == "fastest"
    assert get_tracker_info("two_phase").accuracy_ranking == "highest"


@pytest.mark.parametrize(
    "name",
    [
        "myptv_3d_tracking",
        "myptv_2d_tracking",
        "proptv_tracking",
        "proptv",
        "nearest_hungarian_3d",
        "predictive_gmm_3d",
    ],
)
def test_removed_tracker_names_are_gone(name):
    assert name not in TRACKER_REGISTRY
    assert get_tracker_info(name) is None
