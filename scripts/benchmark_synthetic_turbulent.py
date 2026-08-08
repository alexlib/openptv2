"""Benchmark the saved synthetic_turbulent test case across trackers.

Runs each tracker on an isolated copy of test_data/synthetic_turbulent with
the same tracking parameters, then reports proPTV-style identity metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_utils import TRACKERS, build_true_tracks, read_gt_frames, run_all_trackers


def main():
    tt = build_true_tracks(read_gt_frames())

    header = (f"{'tracker':<22} | {'#tr':>5} | {'F':>5} | {'C':>5} | "
              f"{'Cr':>5} | {'pmt':>6} | {'time_s':>7}")
    print(f"test_case: synthetic_turbulent (30 frames, {len(tt)} true trajs)")
    print(header)
    print("-" * len(header))

    for tr in TRACKERS:
        results = run_all_trackers([tr], silent=True)
        r = results[tr]
        m = r.get("metrics")
        if m is None:
            print(f"{tr:<22} | ERROR {r.get('error')}")
            continue
        print(f"{tr:<22} | {m.n_reconstructed:>5} | {m.fragmentation:>5.2f} | "
              f"{m.completeness:>5.2f} | {m.purity:>5.2f} | {m.pmt:>5.1f}% | {r['time_s']:>6.2f}")


if __name__ == "__main__":
    main()
