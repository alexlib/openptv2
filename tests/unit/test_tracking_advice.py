"""tracking_advice: measurements -> recommendations, with reasons."""

import numpy as np

from openptv2.tracking_advice import advise, report


def _m(**kw):
    base = {
        "points_per_frame": 1100.0,
        "neighbours": 5.5,
        "three_cam_share": 0.38,
        "flagged_02": 0.10,
        "flagged_05": 0.04,
        "median_step": 0.11,
        "median_kink": 0.07,
        "kink_step_ratio": 0.64,
    }
    base.update(kw)
    return base


def _get(advice, name):
    return next(a for a in advice if a.parameter == name)


def test_typical_sparse_noisy_data_gets_defaults_and_automatic_tolerance():
    a = advise(_m(), fps=5000)
    assert _get(a, "q_seed").value == 0.2 and _get(a, "q_young").value == 3
    tol = _get(a, "confirm_tol")
    assert "automatic" in str(tol.value) and "0.56" in tol.reason  # 8 x 0.07
    assert _get(a, "trajectories.smoothing_window").value == 21
    assert not any(x.parameter == "blob_gate" for x in a)


def test_dense_data_keeps_the_fixed_tolerance_and_says_why():
    tol = _get(advise(_m(neighbours=10.5)), "confirm_tol")
    assert tol.value == 0.3 and "dense" in tol.reason and "NOT safe" in tol.reason


def test_frame_skipped_data_keeps_fixed_tolerance_and_gets_the_brightness_gate():
    a = advise(_m(median_step=0.33, median_kink=0.10, kink_step_ratio=0.30))
    assert _get(a, "confirm_tol").value == 0.3
    assert "motion dominates" in _get(a, "confirm_tol").reason
    assert _get(a, "blob_gate").value == 0.5


def test_many_doubtful_points_select_the_accuracy_mode_and_clean_data_the_light_rule():
    a = advise(_m(three_cam_share=0.55, flagged_02=0.18))
    assert _get(a, "q_seed").value == 0.15 and _get(a, "q_young").value == 6
    b = advise(_m(three_cam_share=0.10, flagged_02=0.02))
    assert _get(b, "q_seed").value == 0.3


def test_missing_ghost_measurement_falls_back_to_defaults():
    a = advise(_m(flagged_02=np.nan, three_cam_share=0.38))
    assert _get(a, "q_seed").value == 0.2 and "not measured" in _get(a, "q_seed").reason


def test_report_lists_measurements_reasons_and_yaml():
    m = _m()
    txt = report(m, advise(m, fps=5000))
    assert "MEASURED ON YOUR DATA" in txt and "why:" in txt and "track:" in txt
    assert "q_seed: 0.2" in txt and "smoothing_window: 21" in txt
    assert "# confirm_tol: leave unset (automatic)" in txt


def test_post_steps_follow_the_noise_dominance_guard():
    a = advise(_m())
    assert _get(a, "trajectories.reconnect_gap").value == 6
    assert _get(a, "trajectories.smooth_filter_k").value == 6
    b = advise(_m(median_step=0.5, median_kink=0.13, kink_step_ratio=0.26))
    assert _get(b, "trajectories.smooth_filter_k").value == "off"
    txt = report(_m(), a)
    assert "reconnect_gap: 6" in txt and "smooth_filter_k: 6" in txt
