import os
import shutil

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)
from openptv2.algorithms.track import track_forward_start, trackcorr_c_finish
from openptv2.algorithms.track3d import find_candidates_in_3d, track3d_loop
from openptv2.algorithms.track_kernels_track3d import track3d_loop_fast
from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.algorithms.tracking_run import tr_new

EPS = 1e-5


def _px(positions):
    return np.array(positions, dtype=np.float64, order="C")


def _has_optv():
    try:
        import optv.tracker  # noqa: F401

        return True
    except ImportError:
        return False


def read_all_calibration(num_cams, base_path="test_data/track"):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


def _require_fixtures(*names):
    """Skip when required fixture directories are absent from the current
    directory. Some datasets (e.g. test_data/track) ship no res_orig/img_orig
    on this checkout, so a missing fixture must be a skip, not a hard error.
    Call after chdir into the dataset directory."""
    missing = [n for n in names if not os.path.exists(n)]
    if missing:
        pytest.skip(f"missing fixture(s): {', '.join(missing)}")


def test_find_candidates_in_3d_empty_frame():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 0
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 0


def test_find_candidates_in_3d_single_match():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 1
    frm.path_info = [type("Pathinfo", (), {"x": np.array([5.0, 5.0, 5.0])})()]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 1
    assert indices[0] == 0


def test_find_candidates_in_3d_no_match_outside_box():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 1
    frm.path_info = [type("Pathinfo", (), {"x": np.array([5.0, 5.0, 5.0])})()]
    pos = np.array([10.0, 10.0, 10.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 0


def test_find_candidates_in_3d_multiple_matches():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 5
    frm.path_info = [
        type("Pathinfo", (), {"x": np.array(p, dtype=np.float64)})()
        for p in [[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 6, 6], [10, 10, 10]]
    ]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 2.0, 2.0, 2.0, 4)
    assert len(indices) == 2


def test_find_candidates_in_3d_max_cands_limit():
    frm = Frame(num_cams=1, max_targets=20)
    frm.num_parts = 10
    frm.path_info = [
        type("Pathinfo", (), {"x": np.array([5.0 + i * 0.01, 5.0, 5.0])})()
        for i in range(10)
    ]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 3)
    assert len(indices) == 3


def test_find_candidates_in_3d_boundary():
    frm = Frame(num_cams=1, max_targets=10)
    frm.num_parts = 1
    frm.path_info = [type("Pathinfo", (), {"x": np.array([6.0, 5.0, 5.0])})()]
    pos = np.array([5.0, 5.0, 5.0])
    indices = find_candidates_in_3d(frm, pos, 1.0, 1.0, 1.0, 4)
    assert len(indices) == 0


def test_track3d_level1_ranks_by_forward_acceleration_not_decoy_behind():
    """Regression for the sign bug in the Level 1 acceleration residual.

    Particle 0 moves at constant velocity +0.1/frame along x: 0.0 -> 0.1 in
    frames 0->1, so the correct frame-2 continuation is 0.2 (zero
    acceleration). A decoy candidate is placed at 0.05 -- behind the
    particle, near midpoint(prev, curr) -- which the old buggy expression
    (curr - 2*cand + prev, instead of cand - 2*curr + prev) would rank as
    closer. Both candidates sit inside the search box, so this only passes
    if the acceleration residual is computed with the correct sign.
    """
    px0 = _px([[0.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0]])
    # index 0 = decoy (behind the particle), index 1 = true continuation.
    px2 = _px([[0.05, 0.0, 0.0], [0.2, 0.0, 0.0]])
    prev0 = np.array([-1], dtype=np.int32)
    prev1 = np.array([0], dtype=np.int32)
    next1 = np.full(1, -2, dtype=np.int32)
    prev2 = np.full(2, -1, dtype=np.int32)
    next2 = np.full(2, -2, dtype=np.int32)

    count = track3d_loop_fast(
        1,
        px0,
        prev0,
        1,
        px1,
        prev1,
        next1,
        1,
        px2,
        prev2,
        next2,
        2,
        0.5,
        0.5,
        0.5,
        4,
    )
    assert count == 1
    assert next1[0] == 1, (
        "linked to the decoy behind the particle, not the true continuation"
    )


def _seed_store_from_orig(store, base_dir, num_cams):
    from pathlib import Path

    base_dir = Path(base_dir)
    for rt_file in sorted((base_dir / "res_orig").glob("rt_is.*")):
        frame = int(rt_file.suffix.lstrip("."))
        lines = [ln.strip() for ln in rt_file.read_text().splitlines() if ln.strip()]
        n = int(lines[0]) if lines else 0
        if n > 0:
            rows = np.loadtxt(lines[1 : n + 1], dtype=np.float64).reshape(n, 8)
            pos_3d = rows[:, 1:4]
            cam_ids = rows[:, 4:8].astype(np.int32)
        else:
            pos_3d = np.empty((0, 3), dtype=np.float64)
            cam_ids = np.empty((0, 4), dtype=np.int32)
        store.write_correspondences(frame=frame, pos_3d=pos_3d, cam_target_ids=cam_ids)

    for cam in range(num_cams):
        for tfile in sorted((base_dir / "img_orig").glob(f"cam{cam + 1}.*_targets")):
            frame = int(tfile.name.split(".")[1].split("_")[0])
            lines = [ln.strip() for ln in tfile.read_text().splitlines() if ln.strip()]
            n = int(lines[0]) if lines else 0
            if n > 0:
                arr = np.loadtxt(lines[1 : n + 1], dtype=np.float64).reshape(n, 8)
            else:
                arr = np.empty((0, 8), dtype=np.float64)
            store.write_targets(cam, frame, arr)


def test_track3d_no_add(tmp_path, monkeypatch):
    from pathlib import Path

    from openptv2.storage import RunStore

    src_dir = Path(__file__).resolve().parent / "../../test_data/track"
    if not (src_dir / "res_orig").exists() or not (src_dir / "img_orig").exists():
        pytest.skip("test_data/track fixtures missing")
    work_dir = tmp_path / "track"
    shutil.copytree(src_dir, work_dir)
    monkeypatch.chdir(work_dir)

    cpar = ControlPar.from_yaml("parameters.yaml")
    calib = read_all_calibration(cpar.num_cams, base_path=".")
    store = RunStore(work_dir / "res" / "run.zarr", mode="w")
    _seed_store_from_orig(store, work_dir, cpar.num_cams)

    run = tr_new(
        SequencePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        VolumePar.from_yaml("parameters.yaml"),
        ControlPar.from_yaml("parameters.yaml"),
        4,
        20000,
        "res/rt_is",
        "res/ptv_is",
        "res/added",
        calib,
        0.0001,
    )
    run.fb.store = store
    run.tpar = run.tpar._replace(add=0)
    track_forward_start(run)
    track3d_loop(run, run.seq_par.first)
    for step in range(run.seq_par.first + 1, run.seq_par.last):
        track3d_loop(run, step)
    trackcorr_c_finish(run, run.seq_par.last)
    range_val = run.seq_par.last - run.seq_par.first
    npart = run.npart / range_val
    nlinks = run.nlinks / range_val
    assert abs(npart - 2.0) < EPS
    assert abs(nlinks - 2.0) < EPS


def test_track3d_test_cavity(tmp_path, monkeypatch):
    from pathlib import Path

    from openptv2.storage import RunStore

    src_dir = Path(__file__).resolve().parent / "../../test_data/test_cavity"
    if not (src_dir / "res_orig").exists() or not (src_dir / "img_orig").exists():
        pytest.skip("test_cavity fixtures missing")
    work_dir = tmp_path / "test_cavity"
    shutil.copytree(src_dir, work_dir)
    monkeypatch.chdir(work_dir)

    cpar = ControlPar.from_yaml("parameters.yaml")
    calib = read_all_calibration(cpar.num_cams, base_path=".")
    store = RunStore(work_dir / "res" / "run.zarr", mode="w")
    _seed_store_from_orig(store, work_dir, cpar.num_cams)

    run = tr_new(
        SequencePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        VolumePar.from_yaml("parameters.yaml"),
        ControlPar.from_yaml("parameters.yaml"),
        4,
        20000,
        "res/rt_is",
        "res/ptv_is",
        "res/added",
        calib,
        0.0001,
    )
    run.fb.store = store

    track_forward_start(run)
    for step in range(run.seq_par.first, run.seq_par.last):
        track3d_loop(run, step)
    trackcorr_c_finish(run, run.seq_par.last)

    assert run.npart == 2082
    assert run.nlinks == 1518


def test_tracker_full_forward_3d_test_cavity(tmp_path, monkeypatch):
    """Tracker.full_forward_3d() must produce the same result as the direct loop."""
    from pathlib import Path

    from openptv2.storage import RunStore
    from openptv2.tracker import Tracker

    src_dir = Path(__file__).resolve().parent / "../../test_data/test_cavity"
    if not (src_dir / "res_orig").exists() or not (src_dir / "img_orig").exists():
        pytest.skip("test_cavity fixtures missing")
    work_dir = tmp_path / "test_cavity"
    shutil.copytree(src_dir, work_dir)
    monkeypatch.chdir(work_dir)

    cpar = ControlPar.from_yaml("parameters.yaml")
    cals = read_all_calibration(cpar.num_cams, base_path=".")
    store = RunStore(work_dir / "res" / "run.zarr", mode="w")
    _seed_store_from_orig(store, work_dir, cpar.num_cams)

    tracker = Tracker(
        cpar,
        VolumePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        SequencePar.from_yaml("parameters.yaml"),
        cals,
        store=store,
    )
    tracker.full_forward_3d()

    assert tracker.npart == 2082
    assert tracker.nlinks == 1518


def _parse_linkage_file(path):
    """Parse a ptv_is linkage file into structured data.

    Returns list of dicts with keys: prev, next, x, y, z.
    """
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    particles = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        particles.append(
            {
                "prev": int(parts[0]),
                "next": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
            }
        )
    return particles


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_track3d_burgers_parity_with_cython():
    """Run track3d on burgers data with both Python and C/Cython, compare
    per-step linkage: prev/next pointers and x/y/z positions."""
    original = os.getcwd()
    try:
        test_dir = os.path.join(os.path.dirname(__file__), "../../test_data/burgers")
        os.chdir(test_dir)
        first, last = 10001, 10005

        # --- C / Cython run ---
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        from optv.calibration import Calibration as CCalib
        from optv.parameters import (
            ControlParams,
            SequenceParams,
            TrackingParams,
            VolumeParams,
        )
        from optv.tracker import Tracker

        cpar_c = ControlParams(4)
        cpar_c.read_control_par("parameters/ptv.par")
        vpar_c = VolumeParams()
        vpar_c.read_volume_par("parameters/criteria.par")
        tpar_c = TrackingParams()
        tpar_c.read_track_par("parameters/track.par")
        img_base = [f"img/cam{i + 1}." for i in range(4)]
        spar_c = SequenceParams(
            image_base=img_base,
            frame_range=(first, last),
        )
        cal_c = []
        for i in range(4):
            c = CCalib()
            c.from_file(
                f"cal/cam{i + 1}.tif.ori",
                f"cal/cam{i + 1}.tif.addpar",
            )
            cal_c.append(c)

        naming = {
            "corres": "res/rt_is",
            "linkage": "res/ptv_is",
            "prio": "res/added",
        }
        import inspect

        if "store" in inspect.signature(Tracker).parameters:
            from openptv2.storage import RunStore

            c_store = RunStore(f"{test_dir}/res/c_run3d.zarr", mode="w")
            _seed_store_from_orig(c_store, test_dir, 4)
            tracker = Tracker(cpar_c, vpar_c, tpar_c, spar_c, cal_c, naming, store=c_store)
        else:
            c_store = None
            tracker = Tracker(cpar_c, vpar_c, tpar_c, spar_c, cal_c, naming)
        tracker.full_forward_3d()

        c_data = {}
        for s in range(first, last):
            if c_store is not None:
                prev_ids, next_ids, pos = c_store.read_linkage(s, "ptv_is")
                c_data[s] = [
                    {
                        "prev": int(pr),
                        "next": int(nx),
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "z": float(pt[2]),
                    }
                    for pr, nx, pt in zip(prev_ids, next_ids, pos)
                ]
            else:
                c_data[s] = _parse_linkage_file(f"res/ptv_is.{s}")

        # --- Python run ---
        from openptv2.storage import RunStore

        store = RunStore(f"{test_dir}/res/py_run3d.zarr", mode="w")
        _seed_store_from_orig(store, test_dir, 4)

        cpar_py = ControlPar.from_yaml("parameters.yaml")
        cal_py = read_all_calibration(cpar_py.num_cams, base_path=".")
        run = tr_new(
            SequencePar.from_yaml("parameters.yaml"),
            TrackPar.from_yaml("parameters.yaml"),
            VolumePar.from_yaml("parameters.yaml"),
            ControlPar.from_yaml("parameters.yaml"),
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            cal_py,
            0.0001,
        )
        run.fb.store = store
        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            track3d_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        # --- Compare every field ---
        max_pos_diff = 0.0

        for s in range(first, last):
            prev_ids, next_ids, pos = store.read_linkage(s, "ptv_is")
            py_data = [
                {
                    "prev": int(pr),
                    "next": int(nx),
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "z": float(pt[2]),
                }
                for pr, nx, pt in zip(prev_ids, next_ids, pos)
            ]

            assert len(c_data[s]) == len(py_data), (
                f"Step {s}: particle count C={len(c_data[s])} vs Py={len(py_data)}"
            )

            for i, (c_p, py_p) in enumerate(zip(c_data[s], py_data)):
                assert c_p["prev"] == py_p["prev"], (
                    f"Step {s} particle {i}: prev C={c_p['prev']} Py={py_p['prev']}"
                )
                assert c_p["next"] == py_p["next"], (
                    f"Step {s} particle {i}: next C={c_p['next']} Py={py_p['next']}"
                )

                dx = abs(c_p["x"] - py_p["x"])
                dy = abs(c_p["y"] - py_p["y"])
                dz = abs(c_p["z"] - py_p["z"])
                max_pos_diff = max(max_pos_diff, dx, dy, dz)

                print(
                    f"step {s} particle {i}: "
                    f"prev=({c_p['prev']:2d},{py_p['prev']:2d})  "
                    f"next=({c_p['next']:2d},{py_p['next']:2d})  "
                    f"dx={dx:.6f}  dy={dy:.6f}  dz={dz:.6f}"
                )

                np.testing.assert_allclose(
                    [c_p["x"], c_p["y"], c_p["z"]],
                    [py_p["x"], py_p["y"], py_p["z"]],
                    atol=1e-4,
                    err_msg=f"Step {s} particle {i} position mismatch",
                )

        print(f"\nMax position difference across all steps: {max_pos_diff:.9f}")

    finally:
        os.chdir(original)


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_trackcorr_burgers_parity_with_cython():
    """Run the trackcorr engine (multi-camera 2D+3D epipolar search,
    Tracker.full_forward()) on burgers data with both openptv2.tracker.Tracker
    and liboptv's optv.tracker.Tracker, compare per-step linkage.

    Companion to test_track3d_burgers_parity_with_cython above, which covers
    the other liboptv-derived engine (full_forward_3d / priority_segment_3d).
    Only burgers-scale data is used here: full_forward() on the denser
    synthetic_turbulent fixture crashes liboptv outright (see
    scripts/compare_trackers_vs_liboptv.py's docstring) -- a real
    density-scaling limitation in the original C code, not something this
    test should paper over by avoiding it silently.
    """
    original = os.getcwd()
    try:
        test_dir = os.path.join(os.path.dirname(__file__), "../../test_data/burgers")
        os.chdir(test_dir)
        first, last = 10001, 10005
        naming = {"corres": "res/rt_is", "linkage": "res/ptv_is", "prio": "res/added"}

        # --- C / Cython run ---
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        from optv.calibration import Calibration as CCalib
        from optv.parameters import (
            ControlParams,
            SequenceParams,
            TrackingParams,
            VolumeParams,
        )
        from optv.tracker import Tracker as CTracker

        cpar_c = ControlParams(4)
        cpar_c.read_control_par("parameters/ptv.par")
        vpar_c = VolumeParams()
        vpar_c.read_volume_par("parameters/criteria.par")
        tpar_c = TrackingParams()
        tpar_c.read_track_par("parameters/track.par")
        img_base = [f"img/cam{i + 1}." for i in range(4)]
        spar_c = SequenceParams(
            image_base=img_base,
            frame_range=(first, last),
        )
        cal_c = []
        for i in range(4):
            c = CCalib()
            c.from_file(
                f"cal/cam{i + 1}.tif.ori",
                f"cal/cam{i + 1}.tif.addpar",
            )
            cal_c.append(c)

        import inspect

        if "store" in inspect.signature(CTracker).parameters:
            from openptv2.storage import RunStore

            c_store = RunStore(f"{test_dir}/res/c_runcorr.zarr", mode="w")
            _seed_store_from_orig(c_store, test_dir, 4)
            tracker_c = CTracker(
                cpar_c, vpar_c, tpar_c, spar_c, cal_c, naming, store=c_store
            )
        else:
            c_store = None
            tracker_c = CTracker(cpar_c, vpar_c, tpar_c, spar_c, cal_c, naming)
        tracker_c.full_forward()

        c_data = {}
        for s in range(first, last):
            if c_store is not None:
                prev_ids, next_ids, pos = c_store.read_linkage(s, "ptv_is")
                c_data[s] = [
                    {
                        "prev": int(pr),
                        "next": int(nx),
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "z": float(pt[2]),
                    }
                    for pr, nx, pt in zip(prev_ids, next_ids, pos)
                ]
            else:
                c_data[s] = _parse_linkage_file(f"res/ptv_is.{s}")

        # --- Python run ---
        from openptv2.storage import RunStore
        from openptv2.tracker import Tracker as PyTracker

        store = RunStore(f"{test_dir}/res/py_runcorr.zarr", mode="w")
        _seed_store_from_orig(store, test_dir, 4)

        cpar_py = ControlPar.from_yaml("parameters.yaml")
        cal_py = read_all_calibration(cpar_py.num_cams, base_path=".")
        tracker_py = PyTracker(
            cpar_py,
            VolumePar.from_yaml("parameters.yaml"),
            TrackPar.from_yaml("parameters.yaml"),
            SequencePar.from_yaml("parameters.yaml"),
            cal_py,
            naming,
            store=store,
        )
        tracker_py.full_forward()

        # --- Compare every field ---
        for s in range(first, last):
            prev_ids, next_ids, pos = store.read_linkage(s, "ptv_is")
            py_data = [
                {
                    "prev": int(pr),
                    "next": int(nx),
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "z": float(pt[2]),
                }
                for pr, nx, pt in zip(prev_ids, next_ids, pos)
            ]

            assert len(c_data[s]) == len(py_data), (
                f"Step {s}: particle count C={len(c_data[s])} vs Py={len(py_data)}"
            )
            for i, (c_p, py_p) in enumerate(zip(c_data[s], py_data)):
                assert c_p["prev"] == py_p["prev"], (
                    f"Step {s} particle {i}: prev C={c_p['prev']} Py={py_p['prev']}"
                )
                assert c_p["next"] == py_p["next"], (
                    f"Step {s} particle {i}: next C={c_p['next']} Py={py_p['next']}"
                )
                np.testing.assert_allclose(
                    [c_p["x"], c_p["y"], c_p["z"]],
                    [py_p["x"], py_p["y"], py_p["z"]],
                    atol=1e-4,
                    err_msg=f"Step {s} particle {i} position mismatch",
                )
    finally:
        os.chdir(original)


def test_track3d_unlinked_particle_keeps_the_next_none_sentinel():
    """An unlinked particle must be left at NEXT_NONE (-2), not -1.

    -1 is PREV_NONE; the ptv_is contract (and flowtracks'
    trajectories_ptvis, which detects the end of a trajectory by next ==
    -2) treats -1 in a `next` column as "not a terminated trajectory", so
    writing it there silently drops those trajectories on read.
    """
    from openptv2.algorithms.constants import NEXT_NONE

    px0 = _px([[0.0, 0.0, 0.0]])
    px1 = _px([[0.1, 0.0, 0.0]])
    px2 = _px([[9.0, 9.0, 9.0]])  # far outside the search box: no link possible
    prev0 = np.array([-1], dtype=np.int32)
    prev1 = np.array([0], dtype=np.int32)
    next1 = np.full(1, NEXT_NONE, dtype=np.int32)
    prev2 = np.full(1, -1, dtype=np.int32)
    next2 = np.full(1, NEXT_NONE, dtype=np.int32)

    count = track3d_loop_fast(
        1,
        px0,
        prev0,
        1,
        px1,
        prev1,
        next1,
        1,
        px2,
        prev2,
        next2,
        1,
        0.5,
        0.5,
        0.5,
        4,
    )
    assert count == 0
    assert next1[0] == NEXT_NONE
