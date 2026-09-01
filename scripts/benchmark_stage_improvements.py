"""Stage 1+2 improvement benchmark (docs/plans/2026-08-15-tracking-quality-overhaul.md).

Answers the concrete question: did warmup auto-calibration (Stage 1) and the
corrective backward pass (Stage 2) actually improve tracking, measured
against ground truth, not just "the code runs"?

For each tracker (priority_segment_3d = track3d, full_multipass = trackcorr
forward+backward -- the two engines Stage 1/2 support), on each synthetic
density, three conditions are run and scored against ground truth (identity
+ link metrics from scripts/benchmark_utils.py, plus the new physics metrics
from openptv2.benchmarking.metrics -- track lifetime, acceleration
kurtosis):

  BASELINE   -- the dataset's own default track params (loose, hand-set),
                no warmup, no corrective pass. What you get today with no
                tuning effort.
  WARMUP     -- same engine, params tuned by openptv2.tracking_warmup on a
                window of the SAME run (the "iterative test" the plan calls
                warmup: forward+backward agreement measured, dv/dacc
                adjusted, repeated for a few cycles).
  +CORRECTIVE-- warmup params, then Stage 2's track-assisted
                re-correspondence backward pass on top ("cheap STB").

Real data (test_cavity) has no particle-identity ground truth, so it's
reported directionally only: mean track length and acceleration kurtosis
before/after the corrective pass (no BASELINE/WARMUP split -- warmup's own
value there is already demonstrated in tests/unit/test_tracking_warmup.py
and docs/plans/2026-08-15-tracking-quality-overhaul.md's Stage 1 section).

proPTV's 500_25/500_30 datasets exist locally (C:/Users/alex/Github/proPTV/
data/) but use proPTV's own native format, not openptv2's rt_is/parameters.
yaml convention -- no adapter for that exists in this repo yet, so this
script does not attempt the comparison; noted as follow-up instead of
guessing at a conversion.

Usage:
    uv run python scripts/benchmark_stage_improvements.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu  # noqa: E402
from create_synthetic_turbulent import make_dataset  # noqa: E402

from openptv2.algorithms.calibration import Calibration  # noqa: E402
from openptv2.algorithms.parameters import (  # noqa: E402
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)
from openptv2.benchmarking.metrics import compute_physics_metrics  # noqa: E402
from openptv2.storage import RunStore  # noqa: E402
from openptv2.track_assisted import run_corrective_pass  # noqa: E402
from openptv2.tracker import Tracker  # noqa: E402
from openptv2.tracking_metrics import calculate_tracking_metrics  # noqa: E402
from openptv2.tracking_postprocess import count_links  # noqa: E402
from openptv2.tracking_warmup import run_warmup  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tracks_from_store(store, linkage_name: str, first: int, last: int) -> dict:
    """Walk prev/next chains in the store's linkage into
    {track_id: [(frame, x, y, z)]}, 0-based frames -- matches
    benchmark_utils.build_true_tracks' convention so the two can be
    compared directly."""
    frames = {}
    for f in range(first, last + 1):
        if store.has_linkage(f, name=linkage_name):
            frames[f] = store.read_linkage(f, name=linkage_name)

    tracks: dict[int, list] = {}
    visited = set()
    next_id = 0
    for f in sorted(frames):
        prev, _nxt, xyz = frames[f]
        for i in range(len(prev)):
            if (f, i) in visited or prev[i] >= 0:
                continue  # only start a new track at a head
            tid = next_id
            next_id += 1
            cf, ci = f, i
            while True:
                visited.add((cf, ci))
                _p, _n, cxyz = frames[cf]
                tracks.setdefault(tid, []).append(
                    (
                        cf - first,
                        float(cxyz[ci, 0]),
                        float(cxyz[ci, 1]),
                        float(cxyz[ci, 2]),
                    )
                )
                if cf not in frames:
                    break
                _pp, nxt, _xx = frames[cf]
                if ci >= len(nxt) or nxt[ci] < 0 or (cf + 1) not in frames:
                    break
                cf, ci = cf + 1, int(nxt[ci])
    return tracks


def _populate_store(
    store, scene_dir: Path, first: int, last: int, num_cams: int
) -> None:
    for f in range(first, last + 1):
        rt = scene_dir / "res" / f"rt_is.{f}"
        if rt.exists():
            data = np.loadtxt(rt, skiprows=1, ndmin=2)
            if data.size:
                store.write_correspondences(
                    f, data[:, 1:4], data[:, 4:].astype(np.int32)
                )
        for c in range(num_cams):
            tp = scene_dir / "img" / f"cam{c + 1}.{f}_targets"
            if tp.exists():
                tdata = np.loadtxt(tp, skiprows=1, ndmin=2)
                if tdata.size:
                    store.write_targets(c, f, tdata)


def _run_condition(
    cpar,
    vpar,
    tpar,
    spar,
    cals,
    store,
    engine: str,
    linkage_name: str,
    corrective: bool,
) -> dict:
    naming = {
        "corres": "res/rt_is",
        "linkage": f"res/{linkage_name}",
        "prio": "res/added",
    }
    tracker = Tracker(cpar, vpar, tpar, spar, cals, naming=naming, store=store)
    t0 = time.perf_counter()
    if engine == "priority_segment_3d":
        tracker.full_forward_3d()
    else:
        tracker.full_forward()
        tracker.full_backward()
    if corrective:
        run_corrective_pass(
            cpar, vpar, tpar, spar, cals, store, linkage_name=linkage_name, max_passes=2
        )
    dt = time.perf_counter() - t0

    tracks = _tracks_from_store(store, linkage_name, spar.first, spar.last)
    return {
        "tracks": tracks,
        "time_s": dt,
        "links": count_links(linkage_name, spar.first, spar.last, store=store),
    }


def _score(tracks: dict, tt: dict, ghosts: dict) -> dict:
    identity = bu.bm.compute_identity_metrics(
        tt, tracks, eps=1.0, ghost_pos_by_frame=ghosts
    )
    link = calculate_tracking_metrics(tt, tracks, distance_tolerance=1.0)
    phys = compute_physics_metrics(tracks)
    return {**identity.to_dict(), **link.to_dict(), **phys.to_dict()}


def _print_row(label: str, m: dict, dt: float | None = None):
    t = f"{dt:6.2f}s" if dt is not None else "   -  "
    print(
        f"  {label:22s} | prec={m['precision']:.3f} yield={m['yield_recall']:.3f} "
        f"ghost={m['ghost_capture_rate']:.3f} pmt={m['pmt']:5.1f}% "
        f"trklen={m['mean_track_length']:5.1f} K_a={m['acceleration_kurtosis']:6.2f} "
        f"| {t}"
    )


def run_density(out_dir: Path, num_particles: int, num_frames: int, seed: int):
    print(
        f"\n=== synthetic turbulent, {num_particles} particles/frame, {num_frames} frames ==="
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    yaml_path = make_dataset(
        out_dir, num_particles=num_particles, num_frames=num_frames, seed=seed
    )

    frames_gt = bu.read_gt_frames(out_dir, first=10001, n_frames=num_frames)
    tt = bu.build_true_tracks(frames_gt, first=10001)
    ghosts = bu.build_ghost_frames(frames_gt, first=10001)

    os.chdir(out_dir)
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    tpar_default = TrackPar.from_yaml(str(yaml_path))
    spar = SequencePar.from_yaml(str(yaml_path), cpar.num_cams)
    cals = [
        Calibration.from_file(f"cal/cam{c + 1}.tif.ori", f"cal/cam{c + 1}.tif.addpar")
        for c in range(cpar.num_cams)
    ]

    store = RunStore(str(out_dir / "res" / "run.zarr"), mode="w")
    _populate_store(store, out_dir, spar.first, spar.last, cpar.num_cams)

    for engine in ("priority_segment_3d", "full_multipass"):
        print(f"\n-- engine: {engine} --")
        print(
            f"  {'condition':22s} | {'precision':9s} {'yield':6s} {'ghost':6s} "
            f"{'pmt':6s} {'trklen':7s} {'K_a':6s} | time"
        )

        r = _run_condition(
            cpar,
            vpar,
            tpar_default,
            spar,
            cals,
            store,
            engine,
            f"base_{engine}",
            corrective=False,
        )
        _print_row("BASELINE (default)", _score(r["tracks"], tt, ghosts), r["time_s"])

        warm = run_warmup(
            cpar,
            vpar,
            tpar_default,
            spar,
            cals,
            store,
            frames=min(num_frames, 15),
            max_cycles=3,
        )
        tpar_tuned = TrackPar(
            dvxmin=warm.track_par["dvxmin"],
            dvxmax=warm.track_par["dvxmax"],
            dvymin=warm.track_par["dvymin"],
            dvymax=warm.track_par["dvymax"],
            dvzmin=warm.track_par["dvzmin"],
            dvzmax=warm.track_par["dvzmax"],
            dangle=warm.track_par["dangle"],
            dacc=warm.track_par["dacc"],
            add=tpar_default.add,
            track_mode=tpar_default.track_mode,
        )
        print(
            f"  (warmup picked engine={warm.tracker}, agreement={warm.agreement_rate:.1%}, "
            f"noise~{warm.noise_estimate_mm:.3f}mm, dvxmax {tpar_default.dvxmax:.1f}->{tpar_tuned.dvxmax:.2f})"
        )
        print(
            f"  PARAMS|{num_particles}|{engine}|default|"
            f"dvxmin={tpar_default.dvxmin:.3f}|dvxmax={tpar_default.dvxmax:.3f}|"
            f"dvymin={tpar_default.dvymin:.3f}|dvymax={tpar_default.dvymax:.3f}|"
            f"dvzmin={tpar_default.dvzmin:.3f}|dvzmax={tpar_default.dvzmax:.3f}|"
            f"dangle={tpar_default.dangle:.3f}|dacc={tpar_default.dacc:.3f}"
        )
        print(
            f"  PARAMS|{num_particles}|{engine}|warmup|"
            f"dvxmin={tpar_tuned.dvxmin:.3f}|dvxmax={tpar_tuned.dvxmax:.3f}|"
            f"dvymin={tpar_tuned.dvymin:.3f}|dvymax={tpar_tuned.dvymax:.3f}|"
            f"dvzmin={tpar_tuned.dvzmin:.3f}|dvzmax={tpar_tuned.dvzmax:.3f}|"
            f"dangle={tpar_tuned.dangle:.3f}|dacc={tpar_tuned.dacc:.3f}"
        )

        r = _run_condition(
            cpar,
            vpar,
            tpar_tuned,
            spar,
            cals,
            store,
            engine,
            f"warm_{engine}",
            corrective=False,
        )
        _print_row("WARMUP-tuned", _score(r["tracks"], tt, ghosts), r["time_s"])

        r = _run_condition(
            cpar,
            vpar,
            tpar_tuned,
            spar,
            cals,
            store,
            engine,
            f"corr_{engine}",
            corrective=True,
        )
        _print_row("WARMUP + CORRECTIVE", _score(r["tracks"], tt, ghosts), r["time_s"])


def run_test_cavity():
    cavity = REPO_ROOT / "test_data" / "test_cavity"
    if not (cavity / "res_orig").exists():
        print("\n=== test_cavity: skipped (res_orig fixture not present) ===")
        return
    print("\n=== test_cavity (real data, directional -- no identity ground truth) ===")

    tmp = Path(os.environ.get("TEMP", "/tmp")) / "openptv2_bench_cavity"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(cavity / "res_orig", tmp / "res")
    shutil.copytree(cavity / "img_orig", tmp / "img")
    shutil.copytree(cavity / "cal", tmp / "cal")
    shutil.copy(cavity / "parameters.yaml", tmp / "parameters.yaml")
    os.chdir(tmp)

    yaml_path = tmp / "parameters.yaml"
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    tpar = TrackPar.from_yaml(str(yaml_path))
    spar = SequencePar.from_yaml(str(yaml_path), cpar.num_cams)
    cals = [
        Calibration.from_file(f"cal/cam{c + 1}.tif.ori", f"cal/cam{c + 1}.tif.addpar")
        for c in range(cpar.num_cams)
    ]
    store = RunStore(str(tmp / "res" / "run.zarr"), mode="w")
    _populate_store(store, tmp, spar.first, spar.last, cpar.num_cams)

    r_before = _run_condition(
        cpar,
        vpar,
        tpar,
        spar,
        cals,
        store,
        "priority_segment_3d",
        "cavity_base",
        corrective=False,
    )
    phys_before = compute_physics_metrics(r_before["tracks"])
    print(
        f"  before corrective: links={r_before['links']} mean_track_len={phys_before.mean_track_length:.2f} "
        f"K_a={phys_before.acceleration_kurtosis:.2f} (n={phys_before.n_acceleration_samples})"
    )

    r_after = _run_condition(
        cpar,
        vpar,
        tpar,
        spar,
        cals,
        store,
        "priority_segment_3d",
        "cavity_corr",
        corrective=True,
    )
    phys_after = compute_physics_metrics(r_after["tracks"])
    print(
        f"  after corrective:  links={r_after['links']} mean_track_len={phys_after.mean_track_length:.2f} "
        f"K_a={phys_after.acceleration_kurtosis:.2f} (n={phys_after.n_acceleration_samples})"
    )


def main():
    scratch = Path(os.environ.get("TEMP", "/tmp")) / "openptv2_bench_synth"
    original_cwd = os.getcwd()
    try:
        run_density(scratch / "d220", num_particles=220, num_frames=15, seed=7)
        os.chdir(original_cwd)
        run_density(scratch / "d1000", num_particles=1000, num_frames=15, seed=7)
        os.chdir(original_cwd)
        run_test_cavity()
    finally:
        os.chdir(original_cwd)

    print(
        "\nproPTV 500_25/500_30 (C:/Users/alex/Github/proPTV/data/): present locally but in "
        "proPTV's own native format, not openptv2's rt_is/parameters.yaml convention -- no "
        "converter exists in this repo yet. Not run; flagged as follow-up, not guessed at."
    )


if __name__ == "__main__":
    main()
