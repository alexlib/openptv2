"""Configurable ground-truth trajectory scenarios.

Defines :class:`ScenarioSpec`, a declarative description of a synthetic PTV
dataset, and :func:`generate_scenario` which turns it into ground-truth 3D
trajectories plus a per-frame identity map.

Supported realism knobs:
  * particle count / frames / domain
  * constant or per-particle velocity, acceleration, curvature
  * engineered trajectory crossings (classic mislink stressor)
  * particles entering / leaving the volume mid-sequence
  * per-frame dropout (gaps) and ghost / false-positive particles
  * a sweep list of velocities (produce one dataset per velocity)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CrossingSpec:
    """Engineer a pair of trajectories that pass close to each other.

    The two tracks are drawn as straight lines that meet (or come within
    ``min_distance``) at ``at_frame``, then continue.  This stresses the
    tracker with ambiguous nearby links.
    """

    at_frame: int = 0
    min_distance: float = 0.0
    speed: float = 2.0
    seed: int = 0


@dataclass
class ScenarioSpec:
    """Description of a synthetic ground-truth dataset."""

    # ── Size ──────────────────────────────────────────────────────────
    num_particles: int = 50
    num_frames: int = 40
    domain: tuple[float, float, float] = (80.0, 80.0, 80.0)

    # ── Motion ────────────────────────────────────────────────────────
    # Constant per-particle velocity spread.  If ``velocity_sweep`` is given,
    # one dataset is produced per sweep value.
    velocity: float = 1.0
    velocity_jitter: float = 0.3
    velocity_sweep: tuple[float, ...] = ()
    # Constant acceleration and curvature (angular deflection per frame).
    acceleration: float = 0.0
    curvature: float = 0.0

    # ── Birth / death / crossing ──────────────────────────────────────
    birth_fraction: float = 0.0   # fraction of tracks that start late mid-sequence
    death_fraction: float = 0.0   # fraction of tracks that end early
    entering_particles: int = 0   # particles that cross INTO the volume and appear
    leaving_particles: int = 0    # particles that cross OUT of the volume and vanish
    crossings: list[CrossingSpec] = field(default_factory=list)
    crossing_particle_ids: list[int] = field(default_factory=list)

    # ── Detection realism ─────────────────────────────────────────────
    gap_probability: float = 0.0   # per-frame dropout probability per track
    noise_mm: float = 0.0          # additive gaussian detection noise
    ghost_ratio: float = 0.0       # fraction of spurious extra particles per frame
    seed: int = 42

    # ── Helpers present purely for ergonomics ─────────────────────────
    flow_type: str = "linear"  # "linear" | "turbulent" (proPTV-style chaotic flow)


def generate_scenario(
    spec: ScenarioSpec,
) -> tuple[dict[int, list[tuple[int, float, float, float]]], dict[int, list[tuple[int, float, float, float]]]]:
    """Generate ground-truth trajectories for a :class:`ScenarioSpec`.

    Returns
    -------
    true_tracks : dict[int, list[(frame, x, y, z)]]
        One entry per real particle, the full intended trajectory.
    frame_gt : dict[int, list[(pid, x, y, z)]]
        Per-frame ground truth that includes detection dropouts (a particle
        missing from a frame it was dropped from) and ghost particles.
    """
    rng = np.random.default_rng(spec.seed)
    half = np.array(spec.domain) / 2.0

    true_tracks: dict[int, list[tuple[int, float, float, float]]] = {}
    next_pid = 0

    def add_track(points: list[tuple[int, float, float, float]]) -> int:
        nonlocal next_pid
        pid = next_pid
        next_pid += 1
        true_tracks[pid] = points
        return pid

    # ── 1. Random smooth trajectories ────────────────────────────────
    for _ in range(spec.num_particles):
        x0 = rng.uniform(-half[0] * 0.7, half[0] * 0.7)
        y0 = rng.uniform(-half[1] * 0.7, half[1] * 0.7)
        z0 = rng.uniform(-half[2] * 0.7, half[2] * 0.7)
        v = rng.normal(spec.velocity, spec.velocity_jitter, 3)
        acc = rng.normal(0.0, spec.acceleration, 3)
        # random heading
        th = rng.uniform(0, 2 * np.pi, 3)
        v = v * np.cos(th)

        vel = np.zeros(3)
        pos = np.array([x0, y0, z0])
        pts = []
        for f in range(spec.num_frames):
            if spec.flow_type == "turbulent":
                # proPTV-style smooth chaotic flow: velocity randomly walks
                # with inertia (Ornstein-Uhlenbeck style) so tracks curve
                # smoothly but unpredictably, like DNS turbulent convection.
                vel = 0.9 * vel + rng.normal(0.0, spec.velocity_jitter, 3)
                pos = pos + vel
                pts.append((f, float(pos[0]), float(pos[1]), float(pos[2])))
                continue

            vel = vel + acc
            # gentle curvature (rotate velocity slightly each frame)
            if spec.curvature > 0:
                a = spec.curvature
                rotz = np.array(
                    [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
                )
                # rotate around global z
                vel = rotz @ vel
            pos = pos + vel
            pts.append((f, float(pos[0]), float(pos[1]), float(pos[2])))
        add_track(pts)

    # ── 2. Engineered crossings ──────────────────────────────────────
    for cr in spec.crossings:
        # two tracks approaching perpendicularly, meeting at at_frame
        p1 = rng.uniform(-half[0], half[0])
        p2 = rng.uniform(-half[1], half[1])
        z = rng.uniform(-half[2] * 0.5, half[2] * 0.5)
        vx, vy = cr.speed, cr.speed
        # track A: moves +x at fixed y
        ptsA = []
        for f in range(spec.num_frames):
            xa = p1 + vx * (f - cr.at_frame)
            ya = p2
            ptsA.append((f, xa, ya, z))
        add_track(ptsA)
        # track B: moves +y at fixed x (crosses A)
        ptsB = []
        for f in range(spec.num_frames):
            xb = p1
            yb = p2 + vy * (f - cr.at_frame)
            ptsB.append((f, xb, yb, z))
        add_track(ptsB)

    # ── 3. Entering / leaving particles ──────────────────────────────
    # Entering particles start just OUTSIDE the +x boundary and drift in,
    # so they are absent (skipped) for the first several frames.
    for _ in range(spec.entering_particles):
        x0 = half[0] + abs(rng.normal(2.0, 1.0))
        y0 = rng.uniform(-half[1] * 0.5, half[1] * 0.5)
        z0 = rng.uniform(-half[2] * 0.5, half[2] * 0.5)
        vx = -abs(rng.normal(spec.velocity, spec.velocity_jitter))
        pts = []
        for f in range(spec.num_frames):
            xa = x0 + vx * f
            if xa > half[0] - 0.5:  # still outside the volume; skip
                continue
            pts.append((f, xa, y0, z0))
        if pts:
            add_track(pts)

    # Leaving particles start just INSIDE the -x boundary and drift out,
    # so they disappear for the last several frames.
    for _ in range(spec.leaving_particles):
        x0 = -half[0] + abs(rng.normal(2.0, 1.0))
        y0 = rng.uniform(-half[1] * 0.5, half[1] * 0.5)
        z0 = rng.uniform(-half[2] * 0.5, half[2] * 0.5)
        vx = abs(rng.normal(spec.velocity, spec.velocity_jitter))
        pts = []
        for f in range(spec.num_frames):
            xa = x0 + vx * f
            if xa < -half[0] + 0.5:  # exited the volume; skip
                continue
            pts.append((f, xa, y0, z0))
        if pts:
            add_track(pts)

    # ── 4. Birth / death offsets (fraction of tracks start/end late) ──
    if spec.birth_fraction > 0:
        all_ids = list(true_tracks.keys())
        n = max(1, int(len(all_ids) * spec.birth_fraction))
        chosen = rng.choice(all_ids, size=n, replace=False)
        for pid in chosen:
            delayed = min(spec.num_frames // 4, 3)
            true_tracks[pid] = true_tracks[pid][delayed:]
    if spec.death_fraction > 0:
        all_ids = list(true_tracks.keys())
        n = max(1, int(len(all_ids) * spec.death_fraction))
        chosen = rng.choice(all_ids, size=n, replace=False)
        for pid in chosen:
            shortened = max(1, spec.num_frames // 4)
            true_tracks[pid] = true_tracks[pid][: shortened]

    # ── 5. Build per-frame ground truth with noise / gaps / ghosts ──
    frame_gt: dict[int, list[tuple[int, float, float, float]]] = {
        f: [] for f in range(spec.num_frames)
    }
    for pid, points in true_tracks.items():
        for f, x, y, z in points:
            # dropout
            if spec.gap_probability > 0 and rng.uniform() < spec.gap_probability:
                continue
            if spec.noise_mm > 0:
                n = rng.normal(0, spec.noise_mm, 3)
                frame_gt[f].append((pid, x + n[0], y + n[1], z + n[2]))
            else:
                frame_gt[f].append((pid, x, y, z))

    # ghosts
    if spec.ghost_ratio > 0:
        for f in range(spec.num_frames):
            n_ghost = int(len(frame_gt[f]) * spec.ghost_ratio)
            for _ in range(n_ghost):
                gx = rng.uniform(-half[0], half[0])
                gy = rng.uniform(-half[1], half[1])
                gz = rng.uniform(-half[2], half[2])
                frame_gt[f].append((-1, gx, gy, gz))  # pid -1 == ghost

    # sort each frame by pid (stable) so we can assign correspondence indices
    for f in frame_gt:
        frame_gt[f].sort(key=lambda t: t[0])

    return true_tracks, frame_gt


__all__ = ["ScenarioSpec", "CrossingSpec", "generate_scenario"]
