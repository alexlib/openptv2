#!/usr/bin/env python3
"""Run the openptv2 test set targeted by the 2026-08-25 bug-fixing session.

Context: v0.5.5 CI is red. The failing set clusters around PARALLEL ZARR
STORE WRITING / tracking:

  Known CI failures (the bugs):
    - tests/unit/test_parallel_tracking.py            (links: 0, lost: ALL)
    - tests/unit/test_parallel_correspondences.py::test_batch_parallel_in_memory
    - tests/unit/test_trackcorr_store_only.py         (3 tests, rmtree + links)
    - tests/gui/test_standalone_dumbbell_calibration_cycle.py

  Regression net around the same machinery (must stay green):
    - tests/unit/test_parallel_tracking_chunked.py    (chunk partition/stitch)
    - tests/unit/test_detect_targets_batch_parallel.py
    - tests/unit/test_run_store.py
    - tests/unit/test_tracker_run_store.py
    - tests/unit/test_zarr_store.py
    - tests/unit/test_convert_legacy_to_zarr.py

NOTE: test_parallel_tracking PASSED locally on Windows/py3.12 but FAILED on
CI for py3.11 AND py3.13 on both ubuntu and windows -- an interpreter-version-
dependent bug. Use --py 3.13 (uv-managed) to reproduce CI conditions before
trusting a local pass.

Usage:
    uv run python scripts/run_regression_tests.py            # everything
    uv run python scripts/run_regression_tests.py --tier1    # known failures only
    uv run python scripts/run_regression_tests.py --py 3.13  # repro CI python
    uv run python scripts/run_regression_tests.py -- -k determinism -x
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

TIER1 = [
    "tests/unit/test_parallel_tracking.py",
    "tests/unit/test_parallel_correspondences.py::test_batch_parallel_in_memory",
    "tests/unit/test_trackcorr_store_only.py",
    "tests/gui/test_standalone_dumbbell_calibration_cycle.py",
]

TIER2 = [
    "tests/unit/test_parallel_tracking_chunked.py",
    "tests/unit/test_detect_targets_batch_parallel.py",
    "tests/unit/test_run_store.py",
    "tests/unit/test_tracker_run_store.py",
    "tests/unit/test_zarr_store.py",
    "tests/unit/test_convert_legacy_to_zarr.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tier1", action="store_true", help="only the known-failing tests"
    )
    parser.add_argument(
        "--py",
        default=None,
        metavar="VER",
        help="run under this Python via 'uv run --python VER' "
        "(e.g. 3.13) to reproduce CI interpreters",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="extra args passed through to pytest (use -- before them)",
    )
    args = parser.parse_args()

    targets = TIER1 if args.tier1 else TIER1 + TIER2

    if args.py:
        cmd_prefix = ["uv", "run", "--python", args.py]
    else:
        cmd_prefix = [sys.executable]

    overall = 0
    failures: list[str] = []
    print(
        f"[regression] {len(targets)} target(s), python={args.py or sys.version.split()[0]}"
    )
    for t in targets:
        cmd = [*cmd_prefix, "-m", "pytest", t, "-q", "--tb=short", *args.pytest_args]
        print(f"\n[regression] >>> {' '.join(cmd)}")
        t0 = time.monotonic()
        proc = subprocess.run(cmd)
        dt = time.monotonic() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        if proc.returncode != 0:
            overall |= 1
            failures.append(t)
        print(f"[regression] <<< {status} {t} ({dt:.0f}s)")

    print("\n[regression] ===== SUMMARY =====")
    if failures:
        print("[regression] FAILED:")
        for t in failures:
            print(f"[regression]   - {t}")
    else:
        print("[regression] all targets passed")
    return 1 if overall else 0


if __name__ == "__main__":
    sys.exit(main())
