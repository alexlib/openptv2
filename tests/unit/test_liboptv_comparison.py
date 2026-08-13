"""Regression coverage for scripts/compare_trackers_vs_liboptv.py: all 5
canonical trackers run (forward mode, per-tracker recommended parameters)
without raising, and their identity metrics vs ground truth stay above a
loose sanity floor.

Safe to run in-process: run_comparison() never imports `optv` itself -- each
actual tracker/liboptv run happens in its own subprocess (see
scripts/_tracker_run_worker.py's docstring) because openptv2's own compiled
Cython extensions and the optv C bindings corrupt each other's memory when
run back-to-back in one process. A crash there is therefore isolated to a
child process and shows up as `row["error"]`, not a segfault of the pytest
worker itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_utils as bu  # noqa: E402
import compare_trackers_vs_liboptv as cmp_mod  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not bu.has_liboptv(), reason="optv (Cython bindings) not available")
def test_all_trackers_run_and_clear_a_quality_floor():
    # Short frame range: enough to exercise every tracker + both liboptv
    # reference modes without a multi-minute CI run.
    rows = cmp_mod.run_comparison(bu.SRC, bu.FIRST, n_frames=8)

    # rows also includes synthesized "optv (liboptv, ...)" rows (liboptv
    # itself, run with our trackcorr/fast3d parameters) alongside the 5
    # canonical trackers -- check the 5 are all present as a subset rather
    # than exact set equality.
    assert set(cmp_mod.TRACKERS) <= {r["tracker"] for r in rows}

    for row in rows:
        assert "error" not in row, f"{row['tracker']} failed: {row.get('error')}"
        # Loose floor: catches gross breakage (e.g. a tracker producing no
        # tracks at all), not fine-tuned per-tracker quality targets --
        # those belong in test_tracker_quality.py once a tracker's
        # per-recommended-params baseline is actually measured.
        assert row["gt_pmt"] >= 0, row
        assert row["n_tracks"] > 0, row

    # priority_segment_3d is a near-literal translation of liboptv's
    # full_forward_3d(): some nonzero link agreement is expected. Loose on
    # purpose -- this is a "didn't break entirely" floor, not a quality
    # target; see test_tracker_quality.py for measured-baseline floors.
    fast_row = next(r for r in rows if r["tracker"] == "priority_segment_3d")
    if "vs_liboptv_precision" in fast_row:
        assert fast_row["vs_liboptv_precision"] > 0.05, fast_row
