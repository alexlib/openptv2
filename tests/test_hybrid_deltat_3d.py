"""Tests for openptv2.plugins.hybrid_deltat_3d.

Synthetic ground truth: particles moving at constant velocity with Gaussian
position noise whose per-frame sigma exceeds the per-frame displacement --
the poorly-conditioned regime where fine-rate linking fragments, but a
stride-N coarse pass has displacement >> noise and the refine pass can
re-attach intermediate detections.
"""

import numpy as np
import pytest

from openptv2.plugins.hybrid_deltat_3d import hybrid_track
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

N_PARTICLES = 25
N_FRAMES = 30
VEL = np.array([0.40, 0.06, -0.12])  # mm/frame; |v| ~ 0.42
NOISE_SIGMA = 0.35  # mm per coordinate per frame -> |noise| ~ 0.6 > |v|
STRIDE = 5
V_MAX = 1.5  # per-frame bound, generous vs true speed
A_MAX = 2.0


def _synthetic(seed=7):
    rng = np.random.default_rng(seed)
    truth = []
    clouds = []
    p0 = rng.uniform(0.0, 20.0, size=(N_PARTICLES, 3))
    for f in range(N_FRAMES):
        exact = p0 + VEL[None, :] * f
        truth.append(exact)
        clouds.append(exact + rng.normal(0.0, NOISE_SIGMA, size=exact.shape))
    return truth, clouds


def _mean_track_length(tracks):
    lens = [len(t["time"]) for t in tracks]
    return float(np.mean(lens)) if lens else 0.0


def test_fine_tracker_fragments_but_hybrid_recovers():
    """In the noisy-slow-flow regime hybrid chains are far longer."""
    _, clouds = _synthetic()

    fine = MyPTV3DTracker(v_max=V_MAX, a_max=A_MAX, max_gap=0, dt=1.0)
    fine_tracks = fine.track_frames(clouds)

    chains = hybrid_track(
        clouds, stride=STRIDE, v_max=V_MAX, a_max=A_MAX, refine_gate=1.6
    )

    assert len(fine_tracks) > 0
    # Fine-rate linking cannot maintain long tracks: mean length well under
    # the full sequence.
    assert _mean_track_length(fine_tracks) < N_FRAMES * 0.4

    # Hybrid: most particles yield one near-full-length chain.
    assert len(chains) <= N_PARTICLES * 2
    lengths = [len(c["frame"]) for c in chains]
    # median chain is essentially the full sequence; total coverage high
    assert np.median(lengths) >= N_FRAMES * 0.9
    covered = sum(lengths)
    assert covered >= 0.85 * N_FRAMES * N_PARTICLES


def test_chains_are_consecutive_and_disjoint():
    """Chains use strictly consecutive frames; each detection used once."""
    _, clouds = _synthetic(seed=11)
    chains = hybrid_track(
        clouds, stride=STRIDE, v_max=V_MAX, a_max=A_MAX, refine_gate=1.0
    )

    seen = set()
    for ch in chains:
        frames = ch["frame"]
        assert all(b == a + 1 for a, b in zip(frames, frames[1:]))
        for f, k in zip(frames, ch["idx"]):
            assert 0 <= k < len(clouds[f])
            key = (f, k)
            assert key not in seen, f"detection {key} reused across chains"
            seen.add(key)
        # attached positions must be REAL cloud positions
        for f, k, p in zip(frames, ch["idx"], ch["pos"]):
            np.testing.assert_allclose(p, clouds[f][k], atol=1e-9)


def test_refine_recovers_intermediate_detections_in_clean_case():
    """With no noise every intermediate frame is attached: full coverage."""
    truth, _ = _synthetic(seed=3)
    clean = [t.copy() for t in truth]
    chains = hybrid_track(clean, stride=STRIDE, v_max=0.5, a_max=0.5, refine_gate=1e-6)
    # noise-free: gate can be tiny; prediction is exact
    assert len(chains) == N_PARTICLES
    assert all(len(c["frame"]) == N_FRAMES for c in chains)
    for c in chains:
        np.testing.assert_allclose(c["pos"][0], truth[0][c["idx"][0]], atol=1e-9)


def test_stride_one_matches_plain_hungarian_scale():
    """stride=1 degenerates to the plain predictive Hungarian behaviour."""
    _, clouds = _synthetic(seed=5)
    # hybrid's effective seeded radius at stride 1: (N-1)*v + a*N^2/2 -> a/2,
    # floored at a_max -> exactly the plain tracker's a_max.
    plain = MyPTV3DTracker(v_max=V_MAX, a_max=A_MAX, max_gap=0, dt=1.0)
    ref = plain.track_frames(clouds)
    out = hybrid_track(clouds, stride=1, v_max=V_MAX, a_max=A_MAX, refine_gate=1e9)
    # same seeding/matching dynamics -> identical track count scale
    assert len(out) == pytest.approx(len(ref), abs=2)


def test_too_short_sequence_returns_empty():
    assert hybrid_track([np.zeros((3, 3))], stride=5) == []
