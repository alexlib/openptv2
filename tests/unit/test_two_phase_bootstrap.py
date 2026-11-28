"""Unit tests for Phase 2A of the two_phase tracker: cold-start velocity bootstrapping.

Phase 2A removes the "newborn tracks start with zero velocity" blind spot of the
two_phase tracker.  Instead of v = 0 at frame 0, the tracker estimates a coarse,
*local* velocity field from the displacement mode of the frame 0 -> 1 candidate
pairs (``compute_bootstrap_velocity_field``) and uses it to initialise the
frame-0 tracks.

Plan: docs/plans/2026-10-12-ideal-tracker-bootstrap-and-affine-consensus.md
"""

from __future__ import annotations

import numpy as np
import pytest

import openptv2.plugins.two_phase_tracking as tp
from openptv2.plugins.two_phase_tracking import (
    TwoPhaseTracker,
    TwoPhaseTrackerConfig,
    compute_bootstrap_velocity_field,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _uniform_cloud(
    n: int,
    v: np.ndarray,
    extent: float = 20.0,
    seed: int = 0,
    n_ghosts: int = 0,
    ghost_extent: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Random cloud moving with a constant velocity, plus optional random ghosts.

    Returns (pts0, pts1, n_real) where the first n_real rows are the real
    particles and the rest are ghosts (random at both frames).
    """
    rng = np.random.default_rng(seed)
    p_real = rng.uniform(-extent, extent, size=(n, 3))
    p1 = [p_real + v]
    p0 = [p_real]
    if n_ghosts:
        g_extent = ghost_extent if ghost_extent is not None else extent
        p0.append(rng.uniform(-g_extent, g_extent, size=(n_ghosts, 3)))
        p1.append(rng.uniform(-g_extent, g_extent, size=(n_ghosts, 3)))
    return np.vstack(p0), np.vstack(p1), n


def _spy_bootstrap(monkeypatch, velocity: np.ndarray) -> list[dict]:
    """Replace the bootstrap helper by a recorder that returns a fixed velocity."""
    calls: list[dict] = []

    def fake(pts0, pts1, v_max, dt=1.0, grid_size=20.0, min_pairs=4):
        calls.append(
            {
                "pts0": np.asarray(pts0, dtype=np.float64).copy(),
                "pts1": np.asarray(pts1, dtype=np.float64).copy(),
                "v_max": v_max,
                "dt": dt,
                "grid_size": grid_size,
            }
        )
        return np.tile(np.asarray(velocity, dtype=np.float64), (len(pts0), 1))

    monkeypatch.setattr(tp, "compute_bootstrap_velocity_field", fake)
    return calls


def _shear_frames(
    n: int = 96,
    extent: float = 12.0,
    a: float = 0.12,
    num_frames: int = 4,
    seed: int = 1,
) -> tuple[list[np.ndarray], np.ndarray]:
    """A linear shear flow vx = a*y, and the true velocity of frame 0."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-extent, extent, size=(n, 3))
    frames = []
    for _ in range(num_frames):
        v = np.column_stack([a * p[:, 1], np.zeros(len(p)), np.zeros(len(p))])
        frames.append(p.copy())
        p = p + v
    v0 = np.column_stack([a * frames[0][:, 1], np.zeros(len(p)), np.zeros(len(p))])
    return frames, v0


# --------------------------------------------------------------------------
# compute_bootstrap_velocity_field
# --------------------------------------------------------------------------


def test_bootstrap_returns_zeros_for_empty_frames():
    """No particles on either side -> empty / zero field, no exception."""
    out = compute_bootstrap_velocity_field(
        np.zeros((0, 3)), np.zeros((0, 3)), v_max=3.0, dt=1.0
    )
    assert out.shape == (0, 3)

    p0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = compute_bootstrap_velocity_field(p0, np.zeros((0, 3)), v_max=3.0, dt=1.0)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, 0.0)


def test_bootstrap_recovers_uniform_velocity():
    """A spatially uniform flow is recovered exactly (the mode is the true displacement)."""
    v = np.array([1.5, -0.8, 0.3])
    p0, p1, n = _uniform_cloud(120, v, extent=20.0, seed=0)
    boot_v = compute_bootstrap_velocity_field(p0, p1, v_max=4.0, dt=1.0, grid_size=30.0)
    assert boot_v.shape == (n, 3)
    np.testing.assert_allclose(boot_v, np.tile(v, (n, 1)), atol=1e-6)


def test_bootstrap_is_robust_against_ghost_particles():
    """Ghosts at both frames scatter the candidate displacements but not the mode."""
    v = np.array([1.5, -0.8, 0.3])
    for n_ghosts in (0, 60, 120, 240):
        p0, p1, n = _uniform_cloud(
            120, v, extent=20.0, seed=3, n_ghosts=n_ghosts, ghost_extent=25.0
        )
        boot_v = compute_bootstrap_velocity_field(
            p0, p1, v_max=4.0, dt=1.0, grid_size=30.0
        )
        err = np.linalg.norm(boot_v[:n] - v, axis=1)
        assert np.median(err) < 0.1, (
            f"ghost ratio {n_ghosts / 120:.0%}: {np.median(err)}"
        )


def test_bootstrap_recovers_a_locally_uniform_field():
    """The bootstrap is a *local* field: two regions with different velocities.

    This is the property Phase 2A needs: a track must be initialised with the
    velocity of its own neighbourhood, not with a global average.

    The median is used because a cell that holds only a handful of candidates
    can still pick a spurious mode; the bulk of the field must be exact.
    """
    rng = np.random.default_rng(11)
    v_a = np.array([1.0, 0.0, 0.0])
    v_b = np.array([-1.0, 0.5, 0.0])
    p_a = rng.uniform(-3, 3, size=(60, 3)) + np.array([-15.0, 0.0, 0.0])
    p_b = rng.uniform(-3, 3, size=(60, 3)) + np.array([15.0, 0.0, 0.0])
    p0 = np.vstack([p_a, p_b])
    p1 = np.vstack([p_a + v_a, p_b + v_b])

    boot_v = compute_bootstrap_velocity_field(p0, p1, v_max=3.0, dt=1.0, grid_size=6.0)
    # Each region gets its own velocity, not a mixture of the two.
    np.testing.assert_allclose(np.median(boot_v[:60], axis=0), v_a, atol=1e-6)
    np.testing.assert_allclose(np.median(boot_v[60:], axis=0), v_b, atol=1e-6)
    # ... and the bulk of the particles is exact.
    err_a = np.linalg.norm(boot_v[:60] - v_a, axis=1)
    err_b = np.linalg.norm(boot_v[60:] - v_b, axis=1)
    assert np.mean(err_a < 1e-6) > 0.9
    assert np.mean(err_b < 1e-6) > 0.9


def test_bootstrap_divides_displacement_by_dt():
    """The field is a velocity: displacement / dt, not the raw displacement."""
    v = np.array([2.0, 0.0, 0.0])
    dt = 2.0
    p0, p1, n = _uniform_cloud(120, v * dt, extent=20.0, seed=1)
    boot_v = compute_bootstrap_velocity_field(p0, p1, v_max=8.0, dt=dt, grid_size=30.0)
    np.testing.assert_allclose(np.median(boot_v, axis=0), v, atol=1e-6)
    err = np.linalg.norm(boot_v - v, axis=1)
    assert np.mean(err < 1e-6) > 0.9


def test_bootstrap_requires_min_pairs_to_form_a_mode():
    """A cell with fewer than `min_pairs` candidates does not produce a mode."""
    rng = np.random.default_rng(5)
    v = np.array([1.0, 0.0, 0.0])
    p0 = rng.uniform(-10, 10, size=(60, 3))
    p1 = p0 + v

    # Far more pairs than available -> no cell can form a mode.
    boot_v = compute_bootstrap_velocity_field(
        p0, p1, v_max=4.0, dt=1.0, grid_size=20.0, min_pairs=10_000
    )
    np.testing.assert_allclose(boot_v, 0.0)

    # With enough pairs the mode is found.
    boot_v = compute_bootstrap_velocity_field(
        p0, p1, v_max=4.0, dt=1.0, grid_size=20.0, min_pairs=4
    )
    np.testing.assert_allclose(boot_v, np.tile(v, (60, 1)), atol=1e-6)


def test_bootstrap_falls_back_to_the_nearest_assigned_cell():
    """A particle in a cell without a mode inherits the nearest cell's velocity."""
    rng = np.random.default_rng(9)
    v = np.array([1.0, 0.0, 0.0])
    # Cluster A: many particles -> a mode is found.
    p_a = rng.uniform(-10, 10, size=(60, 3))
    # Cluster B: a single particle far away -> its cell has only one candidate pair.
    p_b = np.array([[60.0, 0.0, 0.0]])
    p0 = np.vstack([p_a, p_b])
    p1 = np.vstack([p_a + v, p_b + v])

    boot_v = compute_bootstrap_velocity_field(
        p0, p1, v_max=4.0, dt=1.0, grid_size=20.0, min_pairs=4
    )
    np.testing.assert_allclose(boot_v[:60], np.tile(v, (60, 1)), atol=1e-6)
    # The lonely particle is not left at zero: it inherits the nearest mode.
    np.testing.assert_allclose(boot_v[60], v, atol=1e-6)


def test_bootstrap_handles_a_single_particle_pair():
    """One particle per frame: too few pairs for a mode, no crash."""
    p0 = np.array([[0.0, 0.0, 0.0]])
    p1 = np.array([[1.0, 0.0, 0.0]])
    boot_v = compute_bootstrap_velocity_field(
        p0, p1, v_max=3.0, dt=1.0, grid_size=10.0, min_pairs=1
    )
    assert boot_v.shape == (1, 3)
    np.testing.assert_allclose(boot_v[0], [1.0, 0.0, 0.0], atol=1e-6)


def test_bootstrap_tracks_a_shear_profile():
    """Characterisation of the shear limit of the mode extraction.

    The mode extraction assumes the flow is *locally uniform* over the binning
    cell: the true displacements of the cell's particles pile up in one
    histogram bin.  In a strong shear the true pile-up is spread over several
    bins while the spurious candidate displacements fill the bins more or less
    uniformly, so the recovered field follows the shear only roughly.
    """
    rng = np.random.default_rng(7)
    a = 0.2
    p0 = rng.uniform(-20, 20, size=(300, 3))
    v = np.column_stack([a * p0[:, 1], np.zeros(len(p0)), np.zeros(len(p0))])
    p1 = p0 + v

    boot_v = compute_bootstrap_velocity_field(p0, p1, v_max=5.0, dt=1.0, grid_size=8.0)

    # The recovered field still follows the shear (same sign, right order of
    # magnitude) even though it is biased low.
    A = np.column_stack([p0[:, 1], np.ones(len(p0))])
    (slope, _intercept), *_ = np.linalg.lstsq(A, boot_v[:, 0], rcond=None)
    assert slope > 0.5 * a
    assert slope < 1.5 * a

    # ... and it is clearly worse than the locally uniform case, which is
    # recovered exactly.
    v_const = np.array([1.0, 0.0, 0.0])
    q0, q1, _ = _uniform_cloud(120, v_const, extent=20.0, seed=0)
    exact = compute_bootstrap_velocity_field(q0, q1, v_max=4.0, dt=1.0, grid_size=30.0)
    err_uniform = np.median(np.linalg.norm(exact - v_const, axis=1))
    err_shear = np.median(np.linalg.norm(boot_v - v, axis=1))
    assert err_shear > 10 * max(err_uniform, 1e-3)


# --------------------------------------------------------------------------
# tracker integration
# --------------------------------------------------------------------------


def test_tracker_bootstraps_frame_zero_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    """The single-pass bootstrap uses the displacement mode with the config values."""
    calls = _spy_bootstrap(monkeypatch, np.array([1.0, 0.0, 0.0]))
    frames = [
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
    ]
    cfg = TwoPhaseTrackerConfig(
        v_max=4.0,
        cost_mode="3d",
        vfc_bootstrap=True,
        vfc_bootstrap_passes=1,
        vfc_bootstrap_grid_size=12.0,
        dt=1.0,
    )
    TwoPhaseTracker(cfg).track_frames(frames)

    assert len(calls) == 1
    call = calls[0]
    np.testing.assert_allclose(call["pts0"], frames[0])
    np.testing.assert_allclose(call["pts1"], frames[1])
    assert call["v_max"] == cfg.v_max
    assert call["dt"] == cfg.dt
    assert call["grid_size"] == cfg.vfc_bootstrap_grid_size


def test_tracker_does_not_bootstrap_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """vfc_bootstrap=False -> the helper is never called and v stays 0 at frame 0."""
    calls = _spy_bootstrap(monkeypatch, np.array([1.0, 0.0, 0.0]))
    frames = [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
    ]
    cfg = TwoPhaseTrackerConfig(v_max=4.0, cost_mode="3d", vfc_bootstrap=False)
    TwoPhaseTracker(cfg).track_frames(frames)
    assert calls == []


def test_tracker_does_not_bootstrap_for_a_single_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    """A one-frame sequence has nothing to bootstrap from."""
    calls = _spy_bootstrap(monkeypatch, np.array([1.0, 0.0, 0.0]))
    frames = [np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])]
    cfg = TwoPhaseTrackerConfig(v_max=4.0, cost_mode="3d", vfc_bootstrap=True)
    TwoPhaseTracker(cfg).track_frames(frames)
    assert calls == []


def test_tracker_bootstraps_frame_zero_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tracks born later in the sequence are not bootstrapped.

    Phase 2A bootstraps the frame-0 tracks only.  A newborn track has no
    history and starts at rest, as before Phase 2A: its first link is protected
    by the consensus of its *neighbours* (which do have an established
    velocity), and its own velocity only matters for its next prediction, by
    which time it has one.
    """
    calls = _spy_bootstrap(monkeypatch, np.array([1.0, 0.0, 0.0]))
    frames = [
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        # The third detection matches nothing -> a newborn track is born here.
        np.array([[1.0, 0.0, 0.0], [6.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0], [21.0, 0.0, 0.0]]),
    ]
    cfg = TwoPhaseTrackerConfig(v_max=4.0, cost_mode="3d", vfc_bootstrap=True)
    TwoPhaseTracker(cfg).track_frames(frames)
    assert len(calls) == 1


def test_tracker_uses_the_bootstrapped_velocity_for_the_frame_zero_prediction(
    monkeypatch: pytest.MonkeyPatch,
):
    """The bootstrapped field is the initial track velocity, not only a field.

    The candidate at 0.2 is nearer to the start position, the true one at 1.0
    is nearer to the *predicted* position -> the link depends on the bootstrap.
    """
    frames = [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ]
    cfg = TwoPhaseTrackerConfig(
        v_max=2.0, cost_mode="3d", use_velocity=True, vfc_bootstrap=True
    )

    _spy_bootstrap(monkeypatch, np.array([0.0, 0.0, 0.0]))
    links_zero = TwoPhaseTracker(cfg).track_frames(frames)
    assert [l[3] for l in links_zero if l[1] == 0] == [0]

    _spy_bootstrap(monkeypatch, np.array([1.0, 0.0, 0.0]))
    links_boot = TwoPhaseTracker(cfg).track_frames(frames)
    assert [l[3] for l in links_boot if l[1] == 0] == [1]


def test_tracker_bootstrap_makes_the_frame_zero_consensus_available(
    monkeypatch: pytest.MonkeyPatch,
):
    """With a bootstrap, frame-0 tracks already have a velocity for the consensus.

    Without it every track has v = 0 at frame 0, so the spatial consensus
    cannot be formed at all and the gate cannot reject a ghost there.
    """
    frames = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
            ]
        ),
    ]
    seen: list[tuple[int, bool]] = []
    original = tp.compute_local_consensus_field

    def spy(positions, velocities, query_positions, radius, min_neighbors=3, **kw):
        seen.append((len(positions), bool(np.any(velocities != 0.0))))
        return original(
            positions,
            velocities,
            query_positions,
            radius,
            min_neighbors=min_neighbors,
            **kw,
        )

    monkeypatch.setattr(tp, "compute_local_consensus_field", spy)

    cfg = TwoPhaseTrackerConfig(
        v_max=4.0,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="pre",
        vfc_gate=0.5,
        vfc_bootstrap=False,
    )
    TwoPhaseTracker(cfg).track_frames(frames)
    # No track has a velocity at frame 0 -> no consensus can be computed.
    assert seen == []

    seen.clear()
    cfg_bootstrap = TwoPhaseTrackerConfig(
        v_max=4.0,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="pre",
        vfc_gate=0.5,
        vfc_bootstrap=True,
    )
    TwoPhaseTracker(cfg_bootstrap).track_frames(frames)
    # With the bootstrap the frame-0 tracks are usable as consensus references
    # and they carry the bootstrapped velocity.
    assert seen and seen[0][0] == 4
    assert seen[0][1]


def test_tracker_bootstrap_rejects_a_ghost_at_frame_zero():
    """The point of Phase 2A: a ghost cannot hijack a newborn track at frame 0.

    The ghost sits right next to the start position of particle 0, so a
    zero-velocity prediction picks it; the bootstrapped prediction points at
    the true detection and the consensus gate removes the ghost.
    """
    frames = [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0, 0.0],  # true match of particle 0
                [1.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
                [0.1, 0.1, 0.0],  # ghost next to the start of particle 0
            ]
        ),
    ]
    common = dict(
        v_max=4.0,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="pre",
        vfc_gate=0.5,
    )
    links_no_boot = TwoPhaseTracker(
        TwoPhaseTrackerConfig(vfc_bootstrap=False, **common)
    ).track_frames(frames)
    assert [l[3] for l in links_no_boot if l[1] == 0] == [4]  # hijacked

    links_boot = TwoPhaseTracker(
        TwoPhaseTrackerConfig(vfc_bootstrap=True, **common)
    ).track_frames(frames)
    assert [l[3] for l in links_boot if l[1] == 0] == [0]  # true match


# --------------------------------------------------------------------------
# two-pass bootstrap: the field comes from the tracks of a first pass
# --------------------------------------------------------------------------


def test_two_pass_field_recovers_a_shear_exactly():
    """The field of the first pass reproduces a linear shear exactly.

    The displacement mode of the raw candidates cannot do that: the true
    displacements are spread over a ridge of histogram bins in a shear.
    """
    frames, v_true = _shear_frames()
    tracker = TwoPhaseTracker(
        TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_bootstrap=True,
            vfc_affine=True,
        )
    )
    links = tracker.track_frames(frames)
    field = tracker._bootstrap_field_from_links(links, frames)
    assert field is not None
    np.testing.assert_allclose(field, v_true, atol=1e-6)

    # The displacement mode of the same data does not recover the shear.
    mode = compute_bootstrap_velocity_field(
        frames[0], frames[1], v_max=4.0, dt=1.0, grid_size=20.0
    )
    assert np.median(np.linalg.norm(mode - v_true, axis=1)) > 0.1


def test_two_pass_bootstrap_runs_the_tracker_twice(
    monkeypatch: pytest.MonkeyPatch,
):
    """The second pass starts from the field of the first one."""
    frames, v_true = _shear_frames()
    # The displacement mode is not used at all when the two-pass field is
    # trusted.
    mode_calls = _spy_bootstrap(monkeypatch, np.array([9.0, 9.0, 9.0]))
    tracker = TwoPhaseTracker(
        TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_bootstrap=True,
            vfc_affine=True,
        )
    )
    seen: list[np.ndarray | None] = []
    original = tracker._run_pass

    def spy(*args, **kwargs):
        seen.append(kwargs.get("init_vels"))
        return original(*args, **kwargs)

    tracker._run_pass = spy  # type: ignore[method-assign]
    tracker.track_frames(frames)

    assert len(seen) == 2
    assert seen[0] is None  # the first pass starts from rest
    assert seen[1] is not None
    np.testing.assert_allclose(seen[1], v_true, atol=1e-6)
    assert mode_calls == []


def test_two_pass_falls_back_when_the_displacement_exceeds_the_spacing(
    monkeypatch: pytest.MonkeyPatch,
):
    """A flow that moves the particles as far as their spacing is not trusted.

    The first link of a track is then a random neighbour, the field of the
    first pass is scrambled, and the single-pass displacement mode is used
    instead.
    """
    frames, _ = _shear_frames(a=0.9, extent=12.0)
    mode_calls = _spy_bootstrap(monkeypatch, np.array([1.0, 0.0, 0.0]))
    tracker = TwoPhaseTracker(
        TwoPhaseTrackerConfig(
            v_max=8.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_bootstrap=True,
            vfc_bootstrap_passes=2,
        )
    )
    seen: list[np.ndarray | None] = []
    original = tracker._run_pass

    def spy(*args, **kwargs):
        seen.append(kwargs.get("init_vels"))
        return original(*args, **kwargs)

    tracker._run_pass = spy  # type: ignore[method-assign]
    tracker.track_frames(frames)

    # The second pass uses the displacement mode, not the field of the
    # first pass.
    assert len(mode_calls) == 1
    assert len(seen) == 2
    assert seen[1] is not None
    np.testing.assert_allclose(seen[1], np.tile([1.0, 0.0, 0.0], (len(frames[0]), 1)))


def test_two_pass_is_skipped_when_the_bootstrap_is_off():
    """Without the bootstrap the tracker runs a single pass from rest."""
    frames, _ = _shear_frames()
    tracker = TwoPhaseTracker(
        TwoPhaseTrackerConfig(v_max=4.0, cost_mode="3d", use_velocity=True)
    )
    seen: list[np.ndarray | None] = []
    original = tracker._run_pass

    def spy(*args, **kwargs):
        seen.append(kwargs.get("init_vels"))
        return original(*args, **kwargs)

    tracker._run_pass = spy  # type: ignore[method-assign]
    tracker.track_frames(frames)

    assert len(seen) == 1
    assert seen[0] is None


def test_two_pass_keeps_the_true_links_in_a_shear():
    """The biased field of the single pass makes the gate cut true links.

    The gate compares the candidate displacement with the local consensus.
    With the displacement mode of a sheared flow the consensus is off by
    ~0.4 mm, the gate then rejects the *true* candidates and the trajectories
    break.  The field of the first pass is exact, the gate keeps them.
    """
    frames, _ = _shear_frames(a=0.12)
    n = len(frames[0])
    common = dict(
        v_max=4.0,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="pre",
        vfc_gate=0.5,
        vfc_radius=12.0,
        vfc_min_neighbors=3,
        vfc_bootstrap=True,
        vfc_affine=True,
        vfc_affine_min_neighbors=4,
    )

    links_one = TwoPhaseTracker(
        TwoPhaseTrackerConfig(vfc_bootstrap_passes=1, **common)
    ).track_frames(frames)
    links_two = TwoPhaseTracker(
        TwoPhaseTrackerConfig(vfc_bootstrap_passes=2, **common)
    ).track_frames(frames)

    # The truth: 3 transitions between 4 frames.
    assert len(links_two) == 3 * n
    # The single pass loses most of them.
    assert len(links_one) < 0.75 * len(links_two)

    # ... and the frame-0 links of the two-pass are the true ones.
    first = {(r0, r1) for (t0, r0, t1, r1) in links_two if t0 == 0}
    assert all((i, i) in first for i in range(n))
