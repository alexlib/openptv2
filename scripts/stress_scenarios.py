"""On-demand, one-knob-at-a-time stress scenarios for the tracker benchmark.

See docs/plans/2026-09-01-tracker-stress-benchmark-plan.md. Every generator
in this module writes a RunStore (zarr) unconditionally, alongside the ASCII
experiment files -- not just when the scenario will run against `two_phase`.
The tracking plugins already prefer store input over ASCII when both exist
(docs/plans/archive/2026-09-01-zarr-only-final-cutover-plan.md), so a
scenario without a store benchmarks a code path production never takes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import openptv2.benchmarking as bm

# Below this fraction of ground-truth points landing on-sensor for any
# camera, a scenario is considered a rig/parameter mismatch rather than a
# real stress case -- fail loudly instead of silently benchmarking a
# thinned-out dataset.
MIN_ON_SENSOR_COVERAGE = 0.8


def _assert_on_sensor_coverage(
    rig: "bm.CameraRig",
    frame_gt: dict[int, list[tuple[int, float, float, float]]],
    min_frac: float = MIN_ON_SENSOR_COVERAGE,
) -> None:
    """Raise if any camera doesn't see at least `min_frac` of the ground
    truth on-sensor. Catches a rig/scenario mismatch (e.g. a perturbed
    calibration pointing off-volume) before it silently shrinks a scenario.
    """
    pts = np.array(
        [(x, y, z) for frame in frame_gt.values() for (_pid, x, y, z) in frame]
    )
    if len(pts) == 0:
        return
    imx, imy = rig.cpar.imx, rig.cpar.imy
    for cam, cam_px in enumerate(bm.project_to_pixels(rig, pts)):
        frac = (
            (cam_px[:, 0] > 0)
            & (cam_px[:, 0] < imx)
            & (cam_px[:, 1] > 0)
            & (cam_px[:, 1] < imy)
        ).mean()
        if frac < min_frac:
            raise RuntimeError(
                f"cam{cam + 1}: on-sensor coverage {frac:.2f} < {min_frac} "
                "-- rig/scenario mismatch, not a real stress case"
            )


def write_experiment_with_store(
    rig: "bm.CameraRig",
    frame_gt: dict[int, list[tuple[int, float, float, float]]],
    out_dir: str | Path,
    first_frame: int = 10001,
    volume: tuple[float, float, float] = (100.0, 100.0, 100.0),
) -> Path:
    """Write a runnable experiment (cal/, ASCII res/img, parameters yaml) via
    `bm.write_experiment`, then write the same ground truth into the
    canonical `<out_dir>/res/run.zarr` RunStore via `bm.write_dataset_store`.

    `write_dataset_store` writes to `spec.dir / "run.zarr"` directly (ignores
    `res_sub`), so `spec.dir` must be `out_dir / "res"` -- not `out_dir` --
    to land at the canonical `<experiment_root>/res/run.zarr` path that
    `per_tracker_overrides`/`find_existing_store` look for. Every tracker
    (not just `two_phase`) prefers this store over ASCII when both exist.

    Returns the `parameters_Run1.yaml` path (same contract as
    `bm.write_experiment`).
    """
    out_dir = Path(out_dir)
    _assert_on_sensor_coverage(rig, frame_gt)
    yaml_path = bm.write_experiment(
        rig, frame_gt, out_dir, first_frame=first_frame, volume=volume
    )
    bm.write_dataset_store(
        rig,
        frame_gt,
        bm.DatasetSpec(
            dir=out_dir / "res", first_frame=first_frame, num_cams=len(rig.cals)
        ),
    )
    return yaml_path
