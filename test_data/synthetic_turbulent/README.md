# synthetic_turbulent — proPTV-style turbulent benchmark case

A synthetic, deterministic PTV dataset used to compare trackers (proPTV vs
MyPTV vs OpenPTV fast/hybrid) on the **same** ground-truth particles with the
**same** tracking parameters.

## Contents

```
synthetic_turbulent/
├── create_synthetic_turbulent.py   # generator (reproduces the dataset)
├── benchmark_synthetic_turbulent.py # cross-tracker benchmark runner
├── parameters_Run1.yaml            # runnable openptv2 experiment config
├── cal/                            # synthetic 4-camera calibration (.ori/.addpar)
├── img/                            # per-camera 2D targets (camN.<frame>_targets)
├── res/
│   ├── rt_is.<frame>               # 3D correspondences (identity p[] = pid)
│   ├── ptv_is.<frame>, added.<frame>
│   └── origin_<frame>.txt          # proPTV-style ground truth (ID, XYZ, cams)
```

## Case physics

* **230+ trajectories** (`num_particles=220`, plus entering/leaving/crossings)
* **30 frames** (10001..10030), 4 cameras
* **flow_type = "turbulent"** — particle velocities walk with inertia
  (Ornstein-Uhlenbeck style), producing smooth chaotic trajectories that mirror
  the DNS Rayleigh-Benard convective flow proPTV uses as its synthetic case.
* Realistic realism knobs: per-frame gaps, detection noise, ghost particles,
  particles entering/leaving the volume, and engineered trajectory crossings.

## Regenerate

```bash
uv run python test_data/create_synthetic_turbulent.py
```

Deterministic given the fixed seed (2026) — identical outputs every run.

## Benchmark

```bash
uv run python test_data/benchmark_synthetic_turbulent.py
```

Runs `fast_3d`, `hybrid_3d_corr`, `myptv_3d_tracking` and `proptv_tracking` on
isolated copies with identical `track` parameters and reports proPTV-style
identity metrics: F (fragmentation), C (completeness), Cr (purity), pmt
(% correct tracks).
