"""Multimedia LUT benchmark: init cost, per-projection speedup, break-even,
and LUT-vs-iterative accuracy.

Opt-in (marked perf + slow, so the default ``-m 'not slow'`` run skips it):

    uv run pytest tests/perf/test_mmlut_benchmark.py -m perf -v -s

Numbers are printed as a table (use ``-s`` to see them); the test also asserts
the accuracy bound that justifies enabling the LUT across the pipeline.
"""

import statistics
import time

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.multimed import (
    get_mmf_from_mmlut,
    init_mmlut,
    multimed_r_nlay_iterative,
)
from openptv2.algorithms.parameters import ControlPar, VolumePar

CAVITY = "test_data/test_cavity"
PARAMS = f"{CAVITY}/parameters_Run1.yaml"


def _load(num_cams=4):
    cals = [
        Calibration.from_file(
            f"{CAVITY}/cal/cam{i + 1}.tif.ori",
            f"{CAVITY}/cal/cam{i + 1}.tif.addpar",
        )
        for i in range(num_cams)
    ]
    vpar = VolumePar.from_yaml(PARAMS)
    cpar = ControlPar.from_yaml(PARAMS)
    cpar.num_cams = num_cams
    return cals, vpar, cpar


def _median(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _in_volume_points(vpar, n, seed=0):
    rng = np.random.default_rng(seed)
    xmin, xmax = vpar.X_lay[0], vpar.X_lay[1]
    zmin, zmax = min(vpar.Zmin_lay), max(vpar.Zmax_lay)
    pts = np.empty((n, 3), dtype=np.float64)
    pts[:, 0] = rng.uniform(xmin, xmax, n)
    pts[:, 1] = rng.uniform(xmin, xmax, n)
    pts[:, 2] = rng.uniform(zmin, zmax, n)
    return pts


@pytest.mark.perf
@pytest.mark.slow
def test_mmlut_benchmark():
    cals, vpar, cpar = _load()
    mm = cpar.mm
    n_proj = 4000
    pts = _in_volume_points(vpar, n_proj)

    print("\n=== mmlut benchmark (test_cavity, water/glass) ===")
    print(f"mm: n1={mm.n1} n2={mm.n2[0]} n3={mm.n3} d={mm.d[0]} nlay={mm.nlay}")

    # --- 1. init cost per camera ---
    init_times = []
    for i in range(cpar.num_cams):

        def _do_init(i=i):
            cal = Calibration.from_file(
                f"{CAVITY}/cal/cam{i + 1}.tif.ori",
                f"{CAVITY}/cal/cam{i + 1}.tif.addpar",
            )
            init_mmlut(vpar, cpar, cal)

        init_times.append(_median(_do_init))
    for i, cal in enumerate(cals):
        init_mmlut(vpar, cpar, cal)
    init_t = statistics.median(init_times)
    print(
        f"init_mmlut: median {init_t * 1e3:.2f} ms/cam; "
        f"LUT dims nr={cals[0].mmlut.nr} nz={cals[0].mmlut.nz} "
        f"rw={cals[0].mmlut.rw}"
    )

    # --- 2. mmf sub-computation cost: LUT lookup vs iterative Snell solve ---
    # This isolates exactly what the LUT replaces (the img_coord wrapper adds
    # constant Python overhead that would otherwise swamp the signal).
    cal = cals[0]
    saved = cal.mmlut.data
    origin = cal.mmlut.origin
    nr, nz, rw = cal.mmlut.nr, cal.mmlut.nz, cal.mmlut.rw
    ex0, ey0, ez0 = cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0

    def _mmf_lut_all():
        for k in range(n_proj):
            get_mmf_from_mmlut(pts[k], origin, nr, nz, rw, saved)

    def _mmf_iter_all():
        for k in range(n_proj):
            multimed_r_nlay_iterative(
                pts[k][0],
                pts[k][1],
                pts[k][2],
                ex0,
                ey0,
                ez0,
                mm.n1,
                mm.n2[0],
                mm.n3,
                mm.d[0],
                mm.nlay,
                mm_n2=mm.n2,
                mm_d=mm.d,
            )

    t_lut = _median(_mmf_lut_all, repeats=9) / n_proj
    t_dir = _median(_mmf_iter_all, repeats=9) / n_proj
    speedup = t_dir / t_lut if t_lut > 0 else float("inf")
    print(
        f"mmf compute: LUT lookup {t_lut * 1e9:.0f} ns, "
        f"iterative solve {t_dir * 1e9:.0f} ns, speedup {speedup:.1f}x"
    )

    # Full img_coord projection cost (LUT vs iterative), for context.
    def _project_all():
        for k in range(n_proj):
            img_coord(pts[k], cal, mm)

    cal.mmlut.data = saved
    t_proj_lut = _median(_project_all, repeats=9) / n_proj
    cal.mmlut.data = None
    t_proj_dir = _median(_project_all, repeats=9) / n_proj
    cal.mmlut.data = saved
    print(
        f"img_coord: LUT {t_proj_lut * 1e9:.0f} ns, "
        f"iterative {t_proj_dir * 1e9:.0f} ns, "
        f"speedup {t_proj_dir / t_proj_lut:.2f}x"
    )

    # --- 3. break-even ---
    if t_dir > t_lut:
        n_be = init_t / (t_dir - t_lut)
        print(
            f"break-even: {n_be:.0f} projections/cam "
            f"(~{n_be / cpar.num_cams:.0f} tracer-projections in a "
            f"{cpar.num_cams}-cam frame)"
        )

    # --- 4. accuracy: LUT vs iterative ---
    diffs_mmf = []
    diffs_px = []
    pix_x = cpar.pix_x
    for k in range(n_proj):
        p = pts[k]
        dx = p[0] - cal.ext_par.x0
        dy = p[1] - cal.ext_par.y0
        # mmf via LUT
        mmf_lut = get_mmf_from_mmlut(
            p,
            cal.mmlut.origin,
            cal.mmlut.nr,
            cal.mmlut.nz,
            cal.mmlut.rw,
            cal.mmlut.data,
        )
        if mmf_lut == 0.0:
            continue  # outside LUT → falls back to iterative → exact
        mmf_dir = multimed_r_nlay_iterative(
            p[0],
            p[1],
            p[2],
            cal.ext_par.x0,
            cal.ext_par.y0,
            cal.ext_par.z0,
            mm.n1,
            mm.n2[0],
            mm.n3,
            mm.d[0],
            mm.nlay,
            mm_n2=mm.n2,
            mm_d=mm.d,
        )
        diffs_mmf.append(abs(mmf_lut - mmf_dir))
        del dx, dy

    # pixel-space error: img_coord LUT vs iterative
    cal.mmlut.data = saved
    xy_lut = np.array([img_coord(pts[k], cal, mm) for k in range(n_proj)])
    cal.mmlut.data = None
    xy_dir = np.array([img_coord(pts[k], cal, mm) for k in range(n_proj)])
    cal.mmlut.data = saved
    diffs_px = np.abs(xy_lut - xy_dir).max(axis=1) / pix_x

    max_mmf = max(diffs_mmf) if diffs_mmf else 0.0
    rms_mmf = float(np.sqrt(np.mean(np.square(diffs_mmf)))) if diffs_mmf else 0.0
    print(
        f"accuracy: |mmf_lut-mmf_dir| max {max_mmf:.2e} rms {rms_mmf:.2e}; "
        f"projection error max {diffs_px.max():.4f} px "
        f"rms {np.sqrt(np.mean(diffs_px**2)):.4f} px"
    )

    # The LUT must not move projections enough to change correspondences.
    assert diffs_px.max() < 0.05, f"LUT projection error {diffs_px.max():.4f} px"


@pytest.mark.perf
@pytest.mark.slow
def test_nlay_lut_fill_speedup():
    """Compiled multi-layer LUT fill (init_mmlut_data_nlay_fast) vs the old
    per-cell Python loop over multimed_r_nlay_iterative."""
    from openptv2.algorithms.track_kernels import init_mmlut_data_nlay_fast

    nr, nz, rw = 81, 85, 2.0  # same grid size as the cavity LUT
    cal_t_x0 = cal_t_y0 = 0.0
    cal_t_z0 = 120.0
    Zmin_t = -20.0
    n1, n3 = 1.0, 1.33
    n2 = np.array([1.49, 1.37], dtype=np.float64)
    d = np.array([5.0, 2.0], dtype=np.float64)
    nlay = 2

    def _compiled():
        init_mmlut_data_nlay_fast(
            nr, nz, rw, cal_t_x0, cal_t_y0, cal_t_z0, Zmin_t, n1, n3, n2, d, nlay
        )

    n2_list, d_list = list(n2), list(d)

    def _python_loop():
        data = np.zeros(nr * nz, dtype=np.float64)
        for i in range(nr):
            for j in range(nz):
                R = i * rw + cal_t_x0
                Z = Zmin_t + j * rw
                data[i * nz + j] = multimed_r_nlay_iterative(
                    R,
                    cal_t_y0,
                    Z,
                    cal_t_x0,
                    cal_t_y0,
                    cal_t_z0,
                    n1,
                    n2[0],
                    n3,
                    d[0],
                    nlay,
                    mm_n2=n2_list,
                    mm_d=d_list,
                )
        return data

    t_fast = _median(_compiled, repeats=9)
    t_slow = _median(_python_loop, repeats=5)
    print("\n=== nlay>1 LUT fill (grid 81x85, nlay=2) ===")
    print(
        f"compiled fill {t_fast * 1e3:.2f} ms, "
        f"python loop {t_slow * 1e3:.2f} ms, "
        f"speedup {t_slow / t_fast:.1f}x"
    )
    assert t_fast < t_slow
