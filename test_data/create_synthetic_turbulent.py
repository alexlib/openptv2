#!/usr/bin/env python3
"""Generate test_data/synthetic_turbulent — a proPTV-style turbulent benchmark case.

This is the "Synthetic case (proPTV-style turbulent / DNS-RB-like flow)" used to
compare trackers (proPTV vs MyPTV vs hybrid/fast) on the SAME synthetic data
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
    uv run python test_data/create_synthetic_turbulent.py

The dataset is deterministic given the seed, so results are reproducible.
"""

from pathlib import Path

import openptv2.benchmarking as bm
from openptv2.benchmarking.scenario import CrossingSpec

OUT_DIR = Path("test_data/synthetic_turbulent")
FIRST_FRAME = 10001
SEED = 2026
VOLUME = (100.0, 100.0, 100.0)

SCENARIO_KWARGS = dict(
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


def main() -> None:
    spec = bm.ScenarioSpec(**SCENARIO_KWARGS)
    tt, fg = bm.generate_scenario(spec)
    rig = bm.make_standard_rig()

    yaml_path = bm.write_experiment(rig, fg, OUT_DIR, first_frame=FIRST_FRAME, volume=VOLUME)

    n_true = len(tt)
    n_per_frame = [len(fg[f]) for f in fg]
    print(f"Wrote synthetic turbulent experiment -> {OUT_DIR}/")
    print(f"  parameters: {yaml_path.name}")
    print(f"  true trajectories: {n_true}")
    print(f"  frames: {spec.num_frames} ({FIRST_FRAME}..{FIRST_FRAME + spec.num_frames - 1})")
    print(
        f"  particles/frame: {min(n_per_frame)}-{max(n_per_frame)} "
        f"(mean {sum(n_per_frame) / len(n_per_frame):.0f})"
    )


if __name__ == "__main__":
    main()
