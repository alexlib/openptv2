"""Common-set scoring of the truth benchmark (scripts/synth_bench.py::common_set_errors)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb  # noqa: E402

pytestmark = pytest.mark.unit


def test_common_and_extra_split():
    keys = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    ev = np.array([1.0, 1.0, 3.0, 3.0, 3.0])
    ref = np.array([1, 2, 99], dtype=np.int64)
    r = sb.common_set_errors(keys, ev, ref, u_rms=1.0)
    assert r["n_common"] == 2 and r["n_extra"] == 3
    assert r["vel_err_common"] == pytest.approx(1.0)
    assert r["vel_err_extra"] == pytest.approx(3.0)
    assert r["extra_share"] == pytest.approx(0.6)


def test_normalised_by_u_rms():
    r = sb.common_set_errors(np.array([1, 2]), np.array([2.0, 2.0]), np.array([1, 2]), u_rms=4.0)
    assert r["vel_err_common"] == pytest.approx(0.5)
    assert np.isnan(r["vel_err_extra"]) and r["n_extra"] == 0


def test_no_overlap_and_empty():
    r = sb.common_set_errors(np.array([7, 8]), np.array([1.0, 3.0]), np.array([1]), u_rms=1.0)
    assert r["n_common"] == 0 and np.isnan(r["vel_err_common"])
    assert r["vel_err_extra"] == pytest.approx(np.sqrt(5.0))
    e = sb.common_set_errors(np.array([], dtype=np.int64), np.array([]), np.array([1]), u_rms=1.0)
    assert e["n_common"] == 0 and e["n_extra"] == 0 and e["extra_share"] == 0.0
