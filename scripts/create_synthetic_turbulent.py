#!/usr/bin/env python3
"""Generate test_data/synthetic_turbulent — a proPTV-style turbulent benchmark case.

This is the "Synthetic case (proPTV-style turbulent / DNS-RB-like flow)" used to
compare trackers (proPTV vs MyPTV vs fast) on the SAME synthetic data
with the SAME parameters.

The dataset is produced deterministically by the openptv2 benchmarking
subpackage:

  * flow_type='turbulent' — velocity walks with inertia (Ornstein-Uhlenbeck),
    producing smooth chaotic trajectories like DNS Rayleigh-Benard convection.
  * gaps, noise, ghosts, entering/leaving particles and crossings for realism.

Files written (matching the openptv2 on-disk layout):
  cal/camN.tif.ori, camN.tif.addpar   — synthetic 4-camera calibration
  img/camN.<frame>_targets            — per-camera 2D targets
  res/rt_is.<frame>                   — 3D correspondences (identity p[] = pid)
  res/ptv_is.<frame>, res/added.<frame>
  res/origin_<frame>.txt              — proPTV-style ground truth
  parameters_Run1.yaml                — runnable openptv2 experiment config

Usage (from repo root):
    uv run python scripts/create_synthetic_turbulent.py

The dataset is deterministic given the seed, so results are reproducible.
``make_dataset()`` also backs the density-sweep variants used by
scripts/bench_trackers.py (density=1000/5000/20000 in the same 100mm volume).
"""

from pathlib import Path

import openptv2.benchmarking as bm
from openptv2.benchmarking.scenario import CrossingSpec

OUT_DIR = Path("test_data/synthetic_turbulent")
FIRST_FRAME = 10001
SEED = 2026
VOLUME = (100.0, 100.0, 100.0)

# Base turbulent-scenario config; num_particles is the density knob (see
# make_dataset). Domain stays fixed so num_particles alone sets the density.
BASE_SCENARIO_KWARGS = dict(
    num_particles=220,
    num_frames=30,
    velocity=2.0,
    velocity_jitter=1.0,     # strong turbulence (high acceleration flips)
    gap_probability=0.08,
    noise_mm=0.08,
    ghost_ratio=0.04,
    seed=SEED,
    entering_particles=6,
    leaving_particles=6,
    flow_type="turbulent",
    crossings=[
        CrossingSpec(at_frame=15, min_distance=0.0, speed=2.0),
        CrossingSpec(at_frame=18, min_distance=0.0, speed=1.5),
    ],
)

# Backward-compat alias (used to be the only config).
SCENARIO_KWARGS = BASE_SCENARIO_KWARGS


def make_dataset(
    out_dir: Path,
    num_particles: int = 220,
    num_frames: int = 30,
    seed: int = SEED,
    first_frame: int = FIRST_FRAME,
    volume: tuple[float, float, float] = VOLUME,
    preserve_trackability: bool = True,
) -> Path:
    """Generate one turbulent-flow dataset at a given particle density.

    When ``preserve_trackability=True`` (default), scales particle displacement
    and velocity jitter by (220 / num_particles)**(1/3) so that the PTV
    Trackability Number M = (v * dt) / d_nn remains constant (M <= 0.2) as
    seeding density increases. Returns the written ``parameters_Run1.yaml`` path.
    """
    scale = (220.0 / num_particles) ** (1.0 / 3.0) if (preserve_trackability and num_particles != 220) else 1.0
    scaled_crossings = [
        CrossingSpec(at_frame=cr.at_frame, min_distance=cr.min_distance, speed=cr.speed * scale, seed=cr.seed)
        for cr in BASE_SCENARIO_KWARGS["crossings"]
    ]
    kwargs = {
        **BASE_SCENARIO_KWARGS,
        "num_particles": num_particles,
        "num_frames": num_frames,
        "seed": seed,
        "velocity": BASE_SCENARIO_KWARGS["velocity"] * scale,
        "velocity_jitter": BASE_SCENARIO_KWARGS["velocity_jitter"] * scale,
        "crossings": scaled_crossings,
    }
    spec = bm.ScenarioSpec(**kwargs)
    tt, fg = bm.generate_scenario(spec)
    rig = bm.make_standard_rig()
    yaml_path = bm.write_experiment(rig, fg, out_dir, first_frame=first_frame, volume=volume)

    n_per_frame = [len(fg[f]) for f in fg]
    mean_d_nn = 0.554 * ((volume[0] * volume[1] * volume[2]) / num_particles) ** (1.0 / 3.0)
    eff_vel = BASE_SCENARIO_KWARGS["velocity"] * scale
    trackability_M = eff_vel / mean_d_nn
    print(f"Wrote turbulent experiment -> {out_dir}/")
    print(f"  parameters: {yaml_path.name}")
    print(f"  true trajectories: {len(tt)}")
    print(f"  frames: {spec.num_frames} ({first_frame}..{first_frame + spec.num_frames - 1})")
    print(
        f"  particles/frame: {min(n_per_frame)}-{max(n_per_frame)} "
        f"(mean {sum(n_per_frame) / len(n_per_frame):.0f})"
    )
    print(f"  trackability: d_nn = {mean_d_nn:.2f} mm | v = {eff_vel:.2f} mm/frame | M = {trackability_M:.3f}")
    return yaml_path


def main() -> None:
    make_dataset(OUT_DIR, num_particles=220, num_frames=30, seed=SEED)


if __name__ == "__main__":
    main()
