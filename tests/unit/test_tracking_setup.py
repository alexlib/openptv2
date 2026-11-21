"""The tracker setup: three trackers, two switches, one add-on (docs/trackers.md)."""

from types import SimpleNamespace

import pytest

from openptv2.tracking_setup import (
    DEFAULT_DIRECTION,
    DEFAULT_TRACKER,
    TRACKERS,
    apply_tracking_setup,
    resolve_tracking_setup,
)


def test_defaults_are_two_phase_both_directions_no_holistic():
    s = resolve_tracking_setup({}, {})
    assert (s.tracker, s.direction, s.postprocess, s.holistic_rounds) == ("two_phase", "both", True, 0)
    assert DEFAULT_TRACKER == "two_phase" and DEFAULT_DIRECTION == "both"
    assert resolve_tracking_setup(None, None) == s
    # the saved-by-the-GUI "default" plugin selection means "whatever track.tracker says"
    assert resolve_tracking_setup({}, {"selected_tracking": "default"}) == s


@pytest.mark.parametrize("tracker", TRACKERS)
def test_new_keys_are_read_for_every_tracker(tracker):
    s = resolve_tracking_setup({"tracker": tracker, "direction": "forward", "postprocess": False})
    assert (s.tracker, s.direction, s.postprocess) == (tracker, "forward", False)


@pytest.mark.parametrize(
    ("old", "tracker", "direction"),
    [
        ("priority_segment_3d", "fast_3d", "forward"),
        ("fast", "fast_3d", "forward"),
        ("cython_3d_tracking", "fast_3d", "forward"),
        ("openptv_fast_3d", "fast_3d", "forward"),
        ("trackcorr", "trackcorr", "both"),
        ("full_multipass", "trackcorr", "both"),
        ("two_directional", "trackcorr", "both"),
        ("standard_forward", "trackcorr", "forward"),
        ("cython_epipolar", "trackcorr", "forward"),
        ("holistic", "two_phase", "forward"),
        ("unified", "two_phase", "forward"),
        ("4be", "fast_3d", "forward"),
        ("hybrid_deltat_3d", "fast_3d", "forward"),
    ],
)
def test_old_names_are_translated(old, tracker, direction):
    s = resolve_tracking_setup({}, {"selected_tracking": old})
    assert (s.tracker, s.direction) == (tracker, direction)
    assert s.notes, "a translated name must say what it became"


def test_old_name_given_as_an_explicit_argument_wins_over_the_yaml():
    s = resolve_tracking_setup({"tracker": "two_phase"}, {}, name="full_multipass")
    assert s.tracker == "trackcorr"
    assert resolve_tracking_setup({"tracker": "fast_3d"}, {}, name="default").tracker == "fast_3d"


def test_bare_trackcorr_is_only_the_legacy_algorithm():
    """Decision A: a YAML that says trackcorr means the legacy port, never the 3D hybrid."""
    s = resolve_tracking_setup({}, {"selected_tracking": "trackcorr"})
    assert s.tracker == "trackcorr" and s.epi_weight == 0.0
    assert any("legacy algorithm" in n for n in s.notes)


def test_the_old_hybrid_engine_is_fast_3d_with_the_epipolar_penalty():
    s = resolve_tracking_setup({"trackcorr_engine": "3d_hybrid"}, {"selected_tracking": "trackcorr"})
    assert (s.tracker, s.epi_weight, s.direction) == ("fast_3d", 1.0, "both")
    s = resolve_tracking_setup({"trackcorr_engine": "3d_hybrid", "epi_weight": 0.5}, {"selected_tracking": "trackcorr"})
    assert s.epi_weight == 0.5
    assert resolve_tracking_setup({"trackcorr_engine": "epipolar"}, {"selected_tracking": "trackcorr"}).tracker == "trackcorr"


@pytest.mark.parametrize(
    ("track", "expected"),
    [
        ({"direction": "forward_backward"}, "both"),
        ({"direction": "both"}, "both"),
        ({"direction": "forward"}, "forward"),
        ({"run_backward": True}, "both"),
        ({"run_backward": False}, "forward"),
        ({"backward": True}, "both"),
        ({"bidirectional": True}, "both"),
        ({"bidirectional": False}, "forward"),
    ],
)
def test_every_old_direction_spelling_means_forward_or_both(track, expected):
    assert resolve_tracking_setup(track, {"selected_tracking": "two_phase"}).direction == expected


def test_holistic_is_an_add_on_for_two_phase_only():
    assert resolve_tracking_setup({"holistic_rounds": 2}).holistic_rounds == 2
    s = resolve_tracking_setup({"tracker": "fast_3d", "holistic_rounds": 2})
    assert s.holistic_rounds == 0 and any("holistic_rounds" in n for n in s.notes)
    assert resolve_tracking_setup({}, {"selected_tracking": "holistic"}).holistic_rounds == 2
    assert resolve_tracking_setup({"holistic_rounds": 0}, {"selected_tracking": "holistic"}).holistic_rounds == 0


def test_a_custom_plugin_name_passes_through():
    s = resolve_tracking_setup({}, {"selected_tracking": "my_local_tracker"})
    assert s.custom and s.tracker == "my_local_tracker"


def test_apply_writes_one_spelling_and_drops_the_old_keys():
    setup = resolve_tracking_setup({"run_backward": True, "trackcorr_engine": "3d_hybrid", "dvxmax": 7.0},
                                   {"selected_tracking": "trackcorr"})
    track, plugins = apply_tracking_setup(setup, {"run_backward": True, "preset": "x", "dvxmax": 7.0}, {})
    assert track["tracker"] == "fast_3d" and track["direction"] == "both" and track["epi_weight"] == 1.0
    assert track["dvxmax"] == 7.0
    assert not {"run_backward", "backward", "bidirectional", "preset", "trackcorr_engine"} & set(track)
    assert plugins["selected_tracking"] == "fast_3d"
    # and the result reads back as itself
    again = resolve_tracking_setup(track, plugins)
    assert (again.tracker, again.direction, again.epi_weight) == ("fast_3d", "both", 1.0)


def test_plugin_loader_runs_the_resolved_tracker(monkeypatch):
    """run_tracking_plugin: 'default' -> the YAML's tracker, and the plugin sees the setup."""
    import openptv2.plugins.loader as loader

    seen = []

    class Fake:
        def __init__(self, ptv=None, exp=None):
            self.exp = exp

        def do_tracking(self):
            seen.append(self.exp.tracking_setup)

    fake_module = SimpleNamespace(Tracking=Fake)
    monkeypatch.setattr(loader.importlib, "import_module", lambda path: fake_module)

    def make(track, plugins=None):
        return SimpleNamespace(pm=SimpleNamespace(parameters={"track": track, "plugins": plugins or {}}))

    loader.run_tracking_plugin("default", make({}))
    loader.run_tracking_plugin("default", make({"tracker": "fast_3d", "direction": "forward"}))
    loader.run_tracking_plugin("full_multipass", make({}))
    assert [(s.tracker, s.direction) for s in seen] == [
        ("two_phase", "both"), ("fast_3d", "forward"), ("trackcorr", "both"),
    ]


def test_picker_offers_default_plus_the_three_trackers():
    from openptv2.plugins.loader import discover_available_plugins

    assert set(discover_available_plugins(None)["available_tracking"]) == {"default", *TRACKERS}
