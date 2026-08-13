# synthetic_turbulent — proPTV-style turbulent benchmark case

A synthetic, deterministic PTV dataset used to compare trackers (proPTV vs
MyPTV vs OpenPTV fast) on the **same** ground-truth particles with the
**same** tracking parameters.

## Contents

```
synthetic_turbulent/
├── parameters_Run1.yaml            # runnable openptv2 experiment config
├── cal/                            # synthetic 4-camera calibration (.ori/.addpar)
├── img/                            # per-camera 2D targets (camN.<frame>_targets)
├── res/
│   ├── rt_is.<frame>               # 3D correspondences (identity p[] = pid)
│   ├── ptv_is.<frame>, added.<frame>
│   └── origin_<frame>.txt          # proPTV-style ground truth (ID, XYZ, cams)
```

The scripts live in `scripts/`:

| Script | Purpose |
| --- | --- |
| `scripts/create_synthetic_turbulent.py` | regenerates this dataset (also backs the density variants below) |
| `scripts/bench_trackers.py` | cross-tracker benchmark runner (single entry point; also drives the 1k/5k/20k density sweep) |

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
uv run python scripts/create_synthetic_turbulent.py
```

Deterministic given the fixed seed (2026) — identical outputs every run.

## Benchmark

```bash
uv run python scripts/bench_trackers.py
uv run python scripts/bench_trackers.py --density 1000,5000,20000  # see test_data/synthetic_turbulent_1k/
```

Runs `priority_segment_3d`, `fast_3d_smooth`, `nearest_hungarian_3d` and `predictive_gmm_3d` on
isolated copies with identical `track` parameters and reports one table
merging both metric systems computed from the same run: proPTV-style
identity metrics (F, C, Cr/purity, pmt, ghost-capture rate) and link-level
metrics (precision, recall/yield, false-connection rate, gap-recovery),
plus ms/frame.
