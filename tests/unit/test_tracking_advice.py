"""The data-driven tracker advice (src/openptv2/tracking_advice.py)."""

from __future__ import annotations

import numpy as np
import pytest

from openptv2.tracking_advice import (
    RunDescription,
    advise,
    describe_store,
    format_advice,
)

pytestmark = pytest.mark.unit


def _d(**kw):
    base = dict(n_frames=100, ppf=1000.0, nn_mm=2.3, step_mm=0.1, spacing_over_step=23.0, mutual_rate=0.95,
                kink_over_step=0.65, frac_partial=0.3, merge_flag_rate=0.0)
    base.update(kw)
    return RunDescription(**base)


def test_exact_data_gets_fast_3d():
    a = advise(_d(kink_over_step=0.01))
    assert a.tracker == "fast_3d" and not a.shared_correction and not a.shared_pass


def test_noisy_data_gets_two_phase_without_extras_by_default():
    a = advise(_d(merge_flag_rate=0.03))
    assert a.tracker == "two_phase" and not a.shared_correction and not a.shared_pass


@pytest.mark.parametrize(("flag", "correction"), [(0.05, False), (0.09, False), (0.10, True), (0.12, True), (0.40, True)])
def test_correction_follows_the_merge_flag_rate(flag, correction):
    assert advise(_d(merge_flag_rate=flag)).shared_correction is correction


def test_pass_alone_only_on_sparse_data():
    assert advise(_d(merge_flag_rate=0.095, mutual_rate=0.95)).shared_pass is False        # below the flag limit
    assert advise(_d(merge_flag_rate=0.15, mutual_rate=0.95)).shared_pass
    assert advise(_d(merge_flag_rate=0.15, mutual_rate=0.7)).shared_pass      # with the correction it is fine


def test_unmeasurable_noise_is_not_treated_as_exact():
    assert advise(_d(kink_over_step=float("nan"))).tracker == "two_phase"


def test_settings_and_text():
    a = advise(_d(merge_flag_rate=0.2))
    assert a.settings() == {"track": {"tracker": "two_phase", "shared_correction": True},
                            "ptv": {"shared_pass": True}}
    text = format_advice(a)
    assert "shared_correction: true" in text and "two_phase" in text


class _Store:
    """A store with moving, noisy 3D points seen by 4 cameras; blob 3 of every point is twice as bright in 20 %."""

    def __init__(self, noise, n=300, n_fr=12, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.uniform(0, 30, (n, 3))
        vel = np.array([0.1, 0.02, 0.0])
        self.frames_ = list(range(1, n_fr + 1))
        self.pos = {f: base + vel * f + rng.normal(0, noise, (n, 3)) for f in self.frames_}
        bright = np.where(rng.random(n) < 0.2, 2.5, 1.0)
        self.targets = {}
        for f in self.frames_:
            for c in range(4):
                t = np.zeros((n, 8))
                t[:, 6] = 100.0 * (bright if c == 3 else 1.0) * rng.uniform(0.95, 1.05, n)
                self.targets[(c, f)] = t
        self.root = {"targets": {f"cam_{c}": None for c in range(4)}}
        for (c, f), t in self.targets.items():
            self.root[f"targets/cam_{c}/frame_{f:06d}"] = t

    def frames(self):
        return self.frames_

    def read_correspondences(self, f):
        return self.pos[f], np.tile(np.arange(len(self.pos[f]))[:, None], (1, 4))


def test_describe_store_measures_noise_and_the_shared_flag():
    quiet = describe_store(_Store(noise=0.0))
    noisy = describe_store(_Store(noise=0.05))
    assert quiet.kink_over_step < 0.05 < 0.3 < noisy.kink_over_step
    assert 0.1 < noisy.merge_flag_rate < 0.3                   # about 20 % of the points have the bright view
    assert noisy.frac_partial == 0.0 and noisy.mutual_rate > 0.9
    assert advise(quiet).tracker == "fast_3d" and advise(noisy).tracker == "two_phase"
