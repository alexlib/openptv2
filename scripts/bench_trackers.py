#!/usr/bin/env python3
"""Single entry point for cross-tracker benchmarking.

Replaces benchmark_synthetic_turbulent.py, benchmark_head_to_head.py and
benchmark_all_trackers_fair.py with one script that emits both metric
systems -- proPTV-style identity (F/C/purity/pmt/ghost-capture) and
link-level (yield/precision/FCR/gap-recovery) -- as one table, computed from
one run (see benchmark_utils.combined_metrics).

Usage:
    uv run python scripts/bench_trackers.py
    uv run python scripts/bench_trackers.py --density 1000,5000,20000
    uv run python scripts/bench_trackers.py --trackers priority_segment_3d,predictive_gmm_3d
    uv run python scripts/bench_trackers.py --density 1000 --dacc-sweep

The default (no --density) uses the checked-in test_data/synthetic_turbulent
case (220 particles/frame). Each requested density either reuses a checked-in
variant (test_data/synthetic_turbulent_<N>) if present, or generates one on
demand into a temp dir (fewer frames, to keep the sweep fast) and discards it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_utils as bu  # noqa: E402
from create_synthetic_turbulent import make_dataset  # noqa: E402

# density -> frame count used when generating on demand (not checked in).
# Large densities use fewer frames so a sweep stays fast; the checked-in
# default (220/frame) and any checked-in variant use their own frame count.
ON_DEMAND_FRAMES = 10


def _dataset_for_density(density: int | None) -> tuple[Path, int, int, bool]:
    """Return (src, first_frame, n_frames, is_temp) for one density.

    ``density=None`` means the checked-in default synthetic_turbulent case.
    """
    if density is None:
        return bu.SRC, bu.FIRST, bu.N_FRAMES, False

    checked_in = Path(f"test_data/synthetic_turbulent_{density}")
    if checked_in.exists():
        return checked_in, bu.FIRST, bu.N_FRAMES, False

    tmp = Path(tempfile.mkdtemp(prefix=f"bench_density_{density}_"))
    make_dataset(tmp, num_particles=density, num_frames=ON_DEMAND_FRAMES, seed=2026)
    return tmp, bu.FIRST, ON_DEMAND_FRAMES, True


def run_sweep(
    densities: list[int | None],
    trackers: list[str],
    track_overrides: dict | None = None,
) -> list[dict]:
    """Run every (density, tracker) combination; return a flat list of rows."""
    rows = []
    for density in densities:
        src, first, n_frames, is_temp = _dataset_for_density(density)
        try:
            results = bu.run_all_trackers(
                trackers, track_overrides=track_overrides, silent=True,
                src=src, first=first, n_frames=n_frames,
            )
            for tr in trackers:
                r = results[tr]
                label = "default" if density is None else str(density)
                if r.get("row") is None:
                    rows.append({"tracker": tr, "density": label, "error": r.get("error")})
                    continue
                n_steps = max(1, n_frames - 1)
                ms_per_frame = 1000.0 * r["time_s"] / n_steps
                rows.append({
                    "tracker": tr, "density": label, "ms_per_frame": ms_per_frame,
                    **r["row"],
                })
        finally:
            if is_temp:
                shutil.rmtree(src, ignore_errors=True)
    return rows


def print_table(rows: list[dict]) -> None:
    header = (f"{'tracker':<22} | {'density':>8} | {'precision':>9} | {'recall':>7} | "
              f"{'ghost%':>7} | {'F':>5} | {'C':>5} | {'purity':>6} | {'pmt%':>6} | "
              f"{'ms/frame':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['tracker']:<22} | {r['density']:>8} | ERROR: {r['error']}")
            continue
        print(
            f"{r['tracker']:<22} | {r['density']:>8} | {r['precision']:>9.3f} | "
            f"{r['yield_recall']:>7.3f} | {100 * r['ghost_capture_rate']:>6.2f}% | "
            f"{r['fragmentation']:>5.2f} | {r['completeness']:>5.2f} | "
            f"{r['purity']:>6.3f} | {r['pmt']:>5.1f}% | {r['ms_per_frame']:>9.1f}"
        )


def dacc_sweep(trackers: tuple[str, ...] = ("priority_segment_3d", "nearest_hungarian_3d", "predictive_gmm_3d")) -> None:
    """Hypothesis check: does priority_segment_3d only lose because dacc is a tight
    search window (not an acceleration bound), vs. myptv/proptv's generous
    radius + cost-based assignment? (folded in from benchmark_head_to_head.py)
    """
    rows = []
    for tr in trackers:
        ov = dict(bu.BASE_OVERRIDES)
        results = bu.run_all_trackers([tr], track_overrides=ov, silent=True)
        rows.append({"tracker": f"{tr} dv6/da6", "density": "default",
                      "ms_per_frame": 1000.0 * results[tr]["time_s"] / max(1, bu.N_FRAMES - 1),
                      **results[tr]["row"]})

    for dacc in (12, 24, 50):
        ov = dict(bu.BASE_OVERRIDES)
        ov["dacc"] = dacc
        results = bu.run_all_trackers(["priority_segment_3d"], track_overrides=ov, silent=True)
        rows.append({"tracker": f"priority_segment_3d dacc={dacc}", "density": "default",
                      "ms_per_frame": 1000.0 * results["priority_segment_3d"]["time_s"] / max(1, bu.N_FRAMES - 1),
                      **results["priority_segment_3d"]["row"]})

    for tr, ov in (
        ("nearest_hungarian_3d", dict(dvxmax=10, dvxmin=-10, dvymax=10, dvymin=-10,
                                   dvzmax=10, dvzmin=-10, dacc=50)),
        ("predictive_gmm_3d", dict(dvxmax=15.5, dvxmin=-15.5, dvymax=15.5, dvymin=-15.5,
                                 dvzmax=15.5, dvzmin=-15.5, dacc=50)),
    ):
        results = bu.run_all_trackers([tr], track_overrides=ov, silent=True)
        rows.append({"tracker": f"{tr} generous", "density": "default",
                      "ms_per_frame": 1000.0 * results[tr]["time_s"] / max(1, bu.N_FRAMES - 1),
                      **results[tr]["row"]})

    print("\n=== dacc-sweep (does priority_segment_3d only lose to a tight search window?) ===")
    print_table(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--density", default="", help="comma-separated particle densities, "
                     "e.g. 1000,5000,20000 (default: the checked-in 220/frame case)")
    ap.add_argument("--trackers", default=",".join(bu.TRACKERS),
                     help=f"comma-separated tracker names (default: {','.join(bu.TRACKERS)})")
    ap.add_argument("--dacc-sweep", action="store_true",
                     help="also run the priority_segment_3d-vs-myptv/proptv search-window hypothesis check")
    args = ap.parse_args()

    trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
    densities: list[int | None] = (
        [None] if not args.density else [int(d) for d in args.density.split(",")]
    )

    rows = run_sweep(densities, trackers)
    print_table(rows)

    if args.dacc_sweep:
        dacc_sweep()


if __name__ == "__main__":
    main()
