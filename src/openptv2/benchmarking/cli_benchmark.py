"""CLI helpers for tracking benchmarks (sweep / compare / dataset).

Backs the ``openptv benchmark`` subcommand.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from openptv2.benchmarking.camera_rig import make_standard_rig
from openptv2.benchmarking.experiment import write_experiment
from openptv2.benchmarking.metrics import compute_identity_metrics
from openptv2.benchmarking.runner import run_tracker
from openptv2.benchmarking.scenario import ScenarioSpec, generate_scenario

# Recognised tracker names available for comparison.
BENCHMARK_TRACKERS = [
    "fast_3d",
    "fast_3d_smooth",
    "full_multipass",
    "standard_forward",
    "two_directional",
    "myptv_3d_tracking",
    "proptv_tracking",
]


def _default_spec(**overrides) -> ScenarioSpec:
    base = dict(
        num_particles=60,
        num_frames=40,
        velocity=1.0,
        velocity_jitter=0.2,
        gap_probability=0.05,
        noise_mm=0.02,
        ghost_ratio=0.02,
    )
    base.update(overrides)
    return ScenarioSpec(**base)


def cmd_dataset(
    out_dir: str | Path,
    num_particles: int = 60,
    num_frames: int = 40,
    velocity: float = 1.0,
    crossings: int = 0,
    entering: int = 0,
    leaving: int = 0,
    gap: float = 0.05,
    noise: float = 0.02,
    ghost: float = 0.02,
    refract: bool = False,
    seed: int = 42,
) -> Path:
    """Generate a ground-truth dataset folder and report its size."""
    spec = ScenarioSpec(
        num_particles=num_particles,
        num_frames=num_frames,
        velocity=velocity,
        velocity_jitter=0.2,
        gap_probability=gap,
        noise_mm=noise,
        ghost_ratio=ghost,
        entering_particles=entering,
        leaving_particles=leaving,
        seed=seed,
    )
    if crossings > 0:
        spec.crossings = [
            __import__("openptv2.benchmarking.scenario", fromlist=["CrossingSpec"]).CrossingSpec(
                at_frame=num_frames // 2, min_distance=0.0, speed=velocity * 2
            )
            for _ in range(crossings)
        ]

    tt, fg = generate_scenario(spec)
    rig = make_standard_rig(refract=refract)
    yaml_path = write_experiment(rig, fg, out_dir, first_frame=10001)
    n_true = len(tt)
    n_per_frame = [len(fg[f]) for f in fg]
    print(f"Wrote experiment to {out_dir}")
    print(f"  true trajectories: {n_true}")
    print(f"  frames: {spec.num_frames} ({spec.num_frames} files in res/)")
    print(
        f"  particles/frame: {min(n_per_frame)}–{max(n_per_frame)} "
        f"(mean {np.mean(n_per_frame):.0f})"
    )
    print(f"  parameters: {yaml_path}")
    return yaml_path


def _remap_to_zero(pred: dict[int, list], first: int) -> dict:
    """Remap absolute frame numbers back to 0-based for metric evaluation."""
    return {k: [(f - first, x, y, z) for (f, x, y, z) in v] for k, v in pred.items()}


def cmd_sweep(
    out_dir: str | Path,
    tracker: str = "fast_3d",
    param: str = "dvxmax",
    values: list[float] | tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    num_particles: int = 60,
    num_frames: int = 30,
    refract: bool = False,
    seed: int = 42,
) -> dict:
    """Sweep one tracking parameter for a fixed tracker and dataset.

    Returns a dict mapping param value -> metrics dict.
    """
    spec = _default_spec(num_particles=num_particles, num_frames=num_frames, seed=seed)
    tt, fg = generate_scenario(spec)
    rig = make_standard_rig(refract=refract)
    yaml_path = write_experiment(rig, fg, out_dir, first_frame=10001)

    results = {}
    _base = {"dacc": 3.0, "angle": 120.0}
    for val in values:
        overrides = dict(_base)
        # dvxmax..dvzmax sweep symmetric bounds
        if param.startswith("dv"):
            overrides.update(
                {
                    "dvxmax": val, "dvxmin": -val,
                    "dvymax": val, "dvymin": -val,
                    "dvzmax": val, "dvzmin": -val,
                }
            )
        elif param in ("dacc", "angle"):
            overrides[param] = val

        t0 = time.perf_counter()
        pred = run_tracker(yaml_path, tracker, track_overrides=overrides)
        dt = time.perf_counter() - t0
        pred0 = _remap_to_zero(pred, 10001)
        m = compute_identity_metrics(tt, pred0, eps=1.0)
        results[val] = {**m.to_dict(), "time_s": round(dt, 3)}

    # Print table
    header = f"{param:<10} | {'#tr':>4} | {'F':>5} | {'C':>5} | {'Cr':>5} | {'pmt':>6} | {'time':>6}"
    print(header)
    print("-" * len(header))
    for val in values:
        r = results[val]
        print(
            f"{val:<10.3g} | {r['n_reconstructed']:>4} | {r['fragmentation']:>5.2f} | "
            f"{r['completeness']:>5.2f} | {r['purity']:>5.2f} | {r['pmt']:>5.1f}% | {r['time_s']:>5.2f}"
        )
    return results


def cmd_compare(
    out_dir: str | Path,
    trackers: list[str] | None = None,
    num_particles: int = 60,
    num_frames: int = 30,
    refract: bool = False,
    seed: int = 42,
) -> dict:
    """Run multiple trackers on the same data with the same parameters."""
    trackers = trackers or BENCHMARK_TRACKERS
    spec = _default_spec(num_particles=num_particles, num_frames=num_frames, seed=seed)
    tt, fg = generate_scenario(spec)
    rig = make_standard_rig(refract=refract)
    yaml_path = write_experiment(rig, fg, out_dir, first_frame=10001)

    overrides = {"dvxmax": 3.0, "dvxmin": -3.0, "dvymax": 3.0, "dvymin": -3.0,
                 "dvzmax": 3.0, "dvzmin": -3.0, "dacc": 3.0}

    results = {}
    for tr in trackers:
        try:
            t0 = time.perf_counter()
            pred = run_tracker(yaml_path, tr, track_overrides=overrides)
            dt = time.perf_counter() - t0
            pred0 = _remap_to_zero(pred, 10001)
            m = compute_identity_metrics(tt, pred0, eps=1.0)
            results[tr] = {**m.to_dict(), "time_s": round(dt, 3)}
        except Exception as e:
            results[tr] = {"error": str(e)}

    header = f"{'tracker':<22} | {'#tr':>4} | {'F':>5} | {'C':>5} | {'Cr':>5} | {'pmt':>6} | {'time':>6}"
    print(header)
    print("-" * len(header))
    for tr in trackers:
        r = results.get(tr, {})
        if "error" in r:
            print(f"{tr:<22} | ERROR: {r['error']}")
        else:
            print(
                f"{tr:<22} | {r['n_reconstructed']:>4} | {r['fragmentation']:>5.2f} | "
                f"{r['completeness']:>5.2f} | {r['purity']:>5.2f} | {r['pmt']:>5.1f}% | {r['time_s']:>5.2f}"
            )
    return results


__all__ = ["BENCHMARK_TRACKERS", "cmd_dataset", "cmd_sweep", "cmd_compare"]
