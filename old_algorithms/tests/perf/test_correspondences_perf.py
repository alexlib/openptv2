"""Performance tests for correspondences hot paths."""

from __future__ import annotations

import pytest

from algorithms.correspondences import correspondences, correspondences_soa
from algorithms.tests.helpers.factories import (
    build_corresp_control_par,
    build_corresp_volume_par,
    generate_grid_frame,
    load_sym_calibrations,
)
from algorithms.tests.helpers.perf import measure_seconds


@pytest.mark.perf
@pytest.mark.slow
def test_correspondences_soa_faster_than_baseline() -> None:
    cpar = build_corresp_control_par()
    vpar = build_corresp_volume_par()
    cals = load_sym_calibrations()
    frm, corrected = generate_grid_frame(cals, cpar)

    def run_orig() -> object:
        return correspondences(frm, corrected, vpar, cpar, cals, [0, 0, 0, 0])

    def run_soa() -> object:
        return correspondences_soa(frm, corrected, vpar, cpar, cals, [0, 0, 0, 0])

    t_orig = measure_seconds(run_orig, repeat=5, warmup=1)
    t_soa = measure_seconds(run_soa, repeat=5, warmup=1)

    # Keep threshold loose enough for CI variance while still catching regressions.
    assert t_soa <= t_orig * 0.9, (
        f"Expected SoA to be at least 10% faster. baseline={t_orig:.6f}s soa={t_soa:.6f}s"
    )
