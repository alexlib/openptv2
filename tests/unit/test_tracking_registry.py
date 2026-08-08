"""Unit tests for openptv2.tracking_registry."""

import pytest

from openptv2.tracking_registry import (
    TRACKER_REGISTRY,
    ParameterGuide,
    TrackerInfo,
    get_tracker_info,
    list_trackers,
    print_tracker_table,
    print_tracker_detail,
)


def test_registry_contains_all_trackers():
    """Verify all expected tracker names are in the registry."""
    expected = {
        "hybrid_3d_corr",
        "full_multipass",
        "fast_3d",
        "standard_forward",
        "two_directional",
        "myptv_3d_tracking",
        "myptv_2d_tracking",
        "splitter_tracking",
        "proptv_tracking",
    }
    registered = set(TRACKER_REGISTRY)
    for name in expected:
        assert name in registered, f"Missing tracker: {name}"


def test_every_tracker_has_required_fields():
    """Verify every TrackerInfo has non-empty identity fields."""
    for name, info in TRACKER_REGISTRY.items():
        assert info.display_name, f"{name}: missing display_name"
        assert info.short_description, f"{name}: missing short_description"
        assert info.algorithm_summary, f"{name}: missing algorithm_summary"
        assert info.speed_ranking in ("fastest", "fast", "moderate", "slow")
        assert info.accuracy_ranking in ("draft", "standard", "high", "highest")
        assert info.density_ranking in (
            "low", "low_to_moderate", "moderate", "high"
        )


def test_get_tracker_info():
    """Verify lookup by name works."""
    info = get_tracker_info("hybrid_3d_corr")
    assert info is not None
    assert info.name == "hybrid_3d_corr"
    assert info.display_name == "Hybrid 3D + 2D Correlation (Recommended Default)"


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
    detail = print_tracker_detail("hybrid_3d_corr")
    assert detail
    assert "Hybrid" in detail
    assert "Algorithm" in detail
    assert "Capabilities" in detail
    assert "Parameters" in detail


def test_print_tracker_detail_unknown(capsys):
    """Verify unknown tracker name returns error message."""
    result = print_tracker_detail("unknown_tracker")
    assert "Unknown tracker" in result


def test_hybrid_3d_corr_capabilities():
    """Verify hybrid tracker capabilities."""
    info = get_tracker_info("hybrid_3d_corr")
    assert info.supports_new_particles is True
    assert info.supports_2d is True
    assert info.supports_backward is False
    assert info.supports_gap_relinking is False
    assert info.supports_multimedia is False


def test_full_multipass_capabilities():
    """Verify full_multipass has all post-processing."""
    info = get_tracker_info("full_multipass")
    assert info.supports_backward is True
    assert info.supports_postprocessing is True
    assert info.supports_gap_relinking is True
    assert info.accuracy_ranking == "highest"


def test_myptv_3d_supports_cost_weights():
    """Verify myptv_3d_tracking advertises cost weight support."""
    info = get_tracker_info("myptv_3d_tracking")
    assert info.supports_cost_weights is True
    assert info.speed_ranking == "slow"


def test_proptv_fields():
    """Verify proPTV tracker info is present and has citation."""
    info = get_tracker_info("proptv_tracking")
    assert info is not None
    assert info.citation
    assert "Barta" in info.citation
    assert info.supports_backward is True
    assert info.supports_gap_relinking is True
    assert info.accuracy_ranking == "highest"
