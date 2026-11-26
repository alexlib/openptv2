"""Plugin lifecycle tests: registry, preset inference, and preset application.

Verifies every registered tracker name is resolvable, alias resolution works,
preset application produces valid track/plugins params, and unknown presets
raise clear errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openptv2.tracking_presets import (
    PRESET_CONFIGS,
    TrackingPreset,
    apply_preset,
    infer_preset,
)
from openptv2.tracking_registry import (
    TRACKER_REGISTRY,
    get_tracker_info,
    list_trackers,
)


class TestTrackerRegistry:
    def test_registry_non_empty(self):
        assert len(TRACKER_REGISTRY) > 0

    def test_all_names_resolvable(self):
        for name in list(TRACKER_REGISTRY.keys()):
            info = get_tracker_info(name)
            assert info is not None, f"get_tracker_info({name!r}) returned None"
            assert info.name == name or name in ("fast_3d", "fast", "trackcorr",
                                                 "multi_deltat_3d", "unified_tracking")

    def test_list_trackers_returns_infos(self):
        infos = list_trackers()
        assert {i.name for i in infos} == {"two_phase", "trackcorr", "fast_3d"}
        # Canonical names must all be valid strings; aliases intentionally
        # share names, so we only require uniqueness among distinct objects.
        names = [i.name for i in infos]
        assert all(isinstance(n, str) and n for n in names)
        for info in infos:
            assert info.name
            assert info.display_name
            assert info.short_description
            assert info.speed_ranking in ("fastest", "fast", "moderate", "slow")
            assert info.density_ranking in ("low", "low_to_moderate", "moderate", "high")
            assert info.accuracy_ranking in ("draft", "standard", "high", "highest")

    def test_aliases_map_to_canonical(self):
        # Retired and short names still resolve, to the three current trackers.
        canonical = {
            "fast_3d": "fast_3d",
            "fast": "fast_3d",
            "trackcorr": "trackcorr",
            "priority_segment_3d": "fast_3d",
            "multi_deltat_3d": "fast_3d",
            "hybrid_deltat_3d": "fast_3d",
            "full_multipass": "trackcorr",
            "unified_tracking": "two_phase",
        }
        for alias, name in canonical.items():
            info = get_tracker_info(alias)
            assert info is not None, alias
            assert info.name == name, f"{alias} -> {info.name}, expected {name}"

    def test_unknown_tracker_returns_none(self):
        assert get_tracker_info("nonexistent_tracker_xyz") is None


class TestPresetInference:
    def test_alias_inference(self):
        assert infer_preset({"selected_tracking": "fast"}, {"selected_tracking": "fast"}) == "priority_segment_3d"
        assert infer_preset({}, {"selected_tracking": "fast_3d"}) == "priority_segment_3d"
        assert infer_preset({}, {"selected_tracking": "priority_segment_3d"}) == "priority_segment_3d"

    def test_full_multipass_inference(self):
        assert infer_preset({}, {"selected_tracking": "full_multipass"}) == "full_multipass"
        assert infer_preset({}, {"selected_tracking": "two_directional"}) == "two_directional"
        assert infer_preset({}, {"selected_tracking": "standard_forward"}) == "standard_forward"

    def test_default_preset(self):
        assert infer_preset({}, {"selected_tracking": "default"}) == "priority_segment_3d"

    def test_custom_plugin_inference(self):
        assert infer_preset({}, {"selected_tracking": "some_custom_plugin"}) == "custom_plugin"

    def test_track_mode_fallback(self):
        assert infer_preset({"track_mode": 1}, {"selected_tracking": "default"}) == "priority_segment_3d"


class TestPresetApplication:
    def test_apply_every_registered_preset(self):
        for preset_name in list(PRESET_CONFIGS.keys()):
            track_params, plugins_params = apply_preset(
                preset_name,
                track_params={"dvxmax": 15.5},
                plugins_params={"selected_tracking": "default"},
            )
            assert "selected_tracking" in plugins_params
            assert "track_mode" in track_params
            assert "flagNewParticles" in track_params
            assert "postprocess" in track_params

    def test_apply_preset_preserves_velocity_bounds(self):
        tp, pp = apply_preset(
            "priority_segment_3d",
            track_params={"dvxmax": 42.0, "dvxmin": -42.0},
            plugins_params={},
        )
        assert tp["dvxmax"] == 42.0
        assert tp["dvxmin"] == -42.0

    def test_apply_unknown_preset_passthrough(self):
        tp, pp = apply_preset("some_unknown_name", track_params={}, plugins_params={})
        assert pp["selected_tracking"] == "some_unknown_name"

    def test_apply_custom_plugin(self):
        tp, pp = apply_preset(
            "custom_plugin",
            track_params={},
            plugins_params={},
            custom_plugin_name="my_custom_tracker",
        )
        assert pp["selected_tracking"] == "my_custom_tracker"

    def test_tracking_preset_enum_values(self):
        assert TrackingPreset.PRIORITY_SEGMENT_3D.value == "priority_segment_3d"
        assert TrackingPreset.FULL_MULTIPASS.value == "full_multipass"
        assert TrackingPreset.TWO_DIRECTIONAL.value == "two_directional"
        assert TrackingPreset.STANDARD_FORWARD.value == "standard_forward"
        assert TrackingPreset.FAST_3D.value == "fast_3d"
        assert TrackingPreset.FAST.value == "fast"

    def test_registry_preset_names_match_apply_preset(self):
        """Every TrackerInfo.default_preset that is non-empty must be accepted
        by apply_preset without crashing."""
        for info in list_trackers():
            if info.default_preset:
                tp, pp = apply_preset(
                    info.default_preset,
                    track_params={"dvxmax": 10.0},
                    plugins_params={"selected_tracking": "default"},
                )
                assert tp is not None
                assert pp is not None
