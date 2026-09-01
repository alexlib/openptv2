"""Per-tracker parameter sweep: find kinematic-bound values that actually
work for each of the 5 trackers separately, instead of trusting one shared
heuristic (tracking_recommender._suggest_params) across structurally
different algorithms.

Motivation: the p95-displacement-based recommended parameters left
trackcorr and predictive_gmm_3d with mean trajectory length ~1 frame on
test_data/synthetic_turbulent, while priority_segment_3d and
kalman_hungarian_3d got mean length 4-8 frames from the SAME parameter
values -- a tracker-specific problem, not a one-size-fits-all tuning bug.

Every run is subprocess-isolated (see scripts/_tracker_run_worker.py):
running the same openptv2 Cython tracker repeatedly in one process was
ALSO observed to eventually crash a multi-point sweep (not just mixing with
optv), so this reuses the same isolation mechanism defensively for every
tracker, not just liboptv comparisons.

Usage:
    uv run python scripts/tune_tracker_params.py trackcorr --dv 2 4 6 10 20 --dacc 3 5.5 10 20 --angle 60 120 200
    uv run python scripts/tune_tracker_params.py --all-best   # print the best config found so far for every tracker
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_utils as bu  # noqa: E402

_WORKER = Path(__file__).parent / "_tracker_run_worker.py"


def _run_via_subprocess(
    tracker: str, src: Path, first: int, n_frames: int, overrides: dict
) -> tuple[dict, float]:
    spec = {
        "tracker": tracker,
        "src": str(src),
        "first": first,
        "n_frames": n_frames,
        "overrides": overrides,
    }
    work = Path(tempfile.mkdtemp())
    spec_path, out_path = work / "spec.json", work / "result.json"
    spec_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, str(_WORKER), str(spec_path), str(out_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"worker failed (exit {proc.returncode}): {proc.stderr[-1500:]}"
        )
    payload = json.loads(out_path.read_text())
    tracks = {int(k): [tuple(pt) for pt in v] for k, v in payload["tracks"].items()}
    return tracks, payload["time_s"]


def sweep(
    tracker: str,
    dv_values: list[float],
    dacc_values: list[float],
    angle_values: list[float],
    src: Path = bu.SRC,
    first: int = bu.FIRST,
    n_frames: int = bu.N_FRAMES,
) -> list[dict]:
    frames = bu.read_gt_frames(src, first, n_frames)
    tt = bu.build_true_tracks(frames, first)
    ghosts = bu.build_ghost_frames(frames, first)

    rows = []
    for dv in dv_values:
        for dacc in dacc_values:
            for angle in angle_values:
                overrides = dict(
                    dvxmin=-dv,
                    dvxmax=dv,
                    dvymin=-dv,
                    dvymax=dv,
                    dvzmin=-dv,
                    dvzmax=dv,
                    dacc=dacc,
                    angle=angle,
                )
                row = {"tracker": tracker, "dv": dv, "dacc": dacc, "angle": angle}
                try:
                    pred, dt = _run_via_subprocess(
                        tracker, src, first, n_frames, overrides
                    )
                    row["time_s"] = dt
                    m = bu.combined_metrics(tt, pred, eps=1.0, ghosts=ghosts)
                    shape = bu.trajectory_shape_stats(pred)
                    row.update({f"gt_{k}": v for k, v in m.items()})
                    row.update({f"shape_{k}": v for k, v in shape.items()})
                    # Composite score: recall matters most (that's the gap
                    # we're chasing), purity must stay high (don't trade
                    # correctness for length), longer mean trajectories are
                    # good. Not a formal optimum -- a ranking heuristic.
                    row["score"] = (
                        row["gt_yield_recall"] * 0.5
                        + row["gt_purity"] * 0.3
                        + min(row["shape_mean_length"] / 10.0, 1.0) * 0.2
                    )
                except Exception as e:
                    row["error"] = str(e)
                    row["score"] = -1.0
                rows.append(row)
    return rows


def adaptive_sweep(
    tracker: str,
    src: Path = bu.SRC,
    first: int = bu.FIRST,
    n_frames: int = bu.N_FRAMES,
    start_dv: float = 1.0,
    step_factor: float = 1.5,
    max_steps: int = 12,
    plateau_patience: int = 2,
    plateau_tol: float = 0.01,
    angle: float = 120.0,
) -> list[dict]:
    """Start low, step dv (and dacc = dv, the pairing the grid sweep found
    works best -- see tracking_recommender._suggest_params's docstring) up
    by step_factor each round, and stop once the score stops improving for
    `plateau_patience` consecutive steps -- instead of grid-searching a wide
    range and crashing into unexplored territory (e.g. dv=20/dacc=20, which
    turned out to trigger a real out-of-bounds bug rather than just being a
    bad parameter choice). Also stops immediately on the first error/crash
    at any step, rather than continuing past it into a worse regime.
    """
    frames = bu.read_gt_frames(src, first, n_frames)
    tt = bu.build_true_tracks(frames, first)
    ghosts = bu.build_ghost_frames(frames, first)

    rows: list[dict] = []
    best_score = -1.0
    plateau_count = 0
    dv = start_dv

    for _ in range(max_steps):
        overrides = dict(
            dvxmin=-dv,
            dvxmax=dv,
            dvymin=-dv,
            dvymax=dv,
            dvzmin=-dv,
            dvzmax=dv,
            dacc=dv,
            angle=angle,
        )
        row = {"tracker": tracker, "dv": dv, "dacc": dv, "angle": angle}
        try:
            pred, dt = _run_via_subprocess(tracker, src, first, n_frames, overrides)
            row["time_s"] = dt
            m = bu.combined_metrics(tt, pred, eps=1.0, ghosts=ghosts)
            shape = bu.trajectory_shape_stats(pred)
            row.update({f"gt_{k}": v for k, v in m.items()})
            row.update({f"shape_{k}": v for k, v in shape.items()})
            row["score"] = (
                row["gt_yield_recall"] * 0.5
                + row["gt_purity"] * 0.3
                + min(row["shape_mean_length"] / 10.0, 1.0) * 0.2
            )
        except Exception as e:
            row["error"] = str(e)
            row["score"] = -1.0
            rows.append(row)
            print(
                f"dv={dv:.2f}: ERROR ({str(e)[:100]}) -- stopping sweep here, not exploring further"
            )
            break

        rows.append(row)
        improvement = row["score"] - best_score
        print(
            f"dv={dv:.2f} dacc={dv:.2f}: score={row['score']:.3f} "
            f"(recall={row['gt_yield_recall']:.3f}, mean_len={row['shape_mean_length']:.2f}, "
            f"{'+' if improvement >= 0 else ''}{improvement:.3f})"
        )

        if improvement > plateau_tol:
            best_score = row["score"]
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= plateau_patience:
                print(
                    f"Plateaued ({plateau_patience} steps with no improvement > {plateau_tol}) -- stopping."
                )
                break

        dv *= step_factor

    return rows


def print_table(rows: list[dict]) -> None:
    print(
        f"{'dv':>6} {'dacc':>6} {'angle':>6} | {'pmt%':>6} {'purity':>7} {'yield':>6} {'prec':>6} {'ghost':>6} "
        f"| {'mean_len':>8} {'frac_short':>10} | {'time_s':>7} {'score':>6}"
    )
    for row in sorted(rows, key=lambda r: -r["score"]):
        if "error" in row:
            print(
                f"{row['dv']:6.1f} {row['dacc']:6.1f} {row['angle']:6.1f} | ERROR: {row['error'][:80]}"
            )
            continue
        print(
            f"{row['dv']:6.1f} {row['dacc']:6.1f} {row['angle']:6.1f} | "
            f"{row['gt_pmt']:6.1f} {row['gt_purity']:7.3f} {row['gt_yield_recall']:6.3f} "
            f"{row['gt_precision']:6.3f} {row['gt_ghost_capture_rate']:6.3f} | "
            f"{row['shape_mean_length']:8.2f} {row['shape_frac_short_lived']:10.2f} | "
            f"{row['time_s']:7.2f} {row['score']:6.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "tracker",
        choices=[
            "priority_segment_3d",
            "trackcorr",
            "nearest_hungarian_3d",
            "predictive_gmm_3d",
        ],
    )
    ap.add_argument(
        "--dv",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 10.0, 15.0],
        help="ignored with --adaptive",
    )
    ap.add_argument(
        "--dacc",
        type=float,
        nargs="+",
        default=[3.0, 6.0, 10.0, 14.0],
        help="ignored with --adaptive",
    )
    ap.add_argument(
        "--angle",
        type=float,
        nargs="+",
        default=[60.0, 120.0, 200.0],
        help="ignored with --adaptive (fixed value, use --start-angle)",
    )
    ap.add_argument(
        "--adaptive",
        action="store_true",
        help="start low and step dv/dacc up together, stopping at the first "
        "plateau or error instead of grid-searching the full range",
    )
    ap.add_argument("--start-dv", type=float, default=1.0)
    ap.add_argument("--step-factor", type=float, default=1.5)
    ap.add_argument("--start-angle", type=float, default=120.0)
    ap.add_argument("--src", type=Path, default=bu.SRC)
    ap.add_argument("--first", type=int, default=bu.FIRST)
    ap.add_argument("--n-frames", type=int, default=bu.N_FRAMES)
    args = ap.parse_args()

    if args.adaptive:
        rows = adaptive_sweep(
            args.tracker,
            args.src,
            args.first,
            args.n_frames,
            start_dv=args.start_dv,
            step_factor=args.step_factor,
            angle=args.start_angle,
        )
    else:
        rows = sweep(
            args.tracker,
            args.dv,
            args.dacc,
            args.angle,
            args.src,
            args.first,
            args.n_frames,
        )
    print_table(rows)


if __name__ == "__main__":
    main()
