# ruff: noqa: E501
"""Unit tests for multi-density synthetic tracking and tracker selection defaults."""

import pytest

from openptv2.plugins.loader import BUILTIN_TRACKING_PLUGINS, LEGACY_ALIASES
from openptv2.tracking_presets import PRESET_CHOICES, TrackingPreset, infer_preset


def test_default_tracker_is_hybrid_3d_corr():
    """Verify hybrid_3d_corr is the primary default tracking plugin."""
    assert BUILTIN_TRACKING_PLUGINS["default"] == "openptv2.plugins.hybrid_tracker"
    assert (
        BUILTIN_TRACKING_PLUGINS["hybrid_3d_corr"] == "openptv2.plugins.hybrid_tracker"
    )
    assert BUILTIN_TRACKING_PLUGINS["hybrid"] == "openptv2.plugins.hybrid_tracker"


def test_fast_alias_mapping():
    """Verify fast is registered as an alias to fast_3d."""
    assert LEGACY_ALIASES["fast"] == "fast_3d"
    assert BUILTIN_TRACKING_PLUGINS["fast"] == "openptv2.plugins.default_tracking"
    assert BUILTIN_TRACKING_PLUGINS["fast_3d"] == "openptv2.plugins.default_tracking"


def test_preset_order():
    """Verify preset choice list starts with hybrid_3d_corr followed by fast and fast_3d."""
    preset_keys = [key for key, _ in PRESET_CHOICES]
    assert preset_keys[0] == "hybrid_3d_corr"
    assert preset_keys[1] == "fast"
    assert preset_keys[2] == "fast_3d"


def test_infer_preset_default_fallback():
    """Verify infer_preset defaults to hybrid_3d_corr."""
    res = infer_preset({}, {})
    assert res == TrackingPreset.HYBRID_3D_CORR.value
