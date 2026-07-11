import numpy as np
import pytest
import os
import shutil
from pathlib import Path

from openptv2.algorithms.track3d import find_candidates_in_3d, track3d_loop
from openptv2.algorithms.track import track_forward_start, trackcorr_c_finish
from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.algorithms.tracking_run import tr_new
from openptv2.algorithms.parameters import (
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)
from openptv2.algorithms.calibration import Calibration

EPS = 1e-5


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


def test_track3d_no_add():
    import os

    original = os.getcwd()
    try:
        test_dir = os.path.join(os.path.dirname(__file__), "../../test_data/track")
        os.chdir(test_dir)
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = ControlPar.from_yaml("parameters.yaml")
        calib = read_all_calibration(cpar.num_cams, base_path=".")
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
    finally:
        os.chdir(original)


def track3d_test_cavity():
    import os

    original = os.getcwd()
    try:
        test_dir = os.path.join(
            os.path.dirname(__file__), "../../test_data/test_cavity"
        )
        os.chdir(test_dir)
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = ControlPar.from_yaml("parameters.yaml")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

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

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            track3d_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 2082
        assert run.nlinks == 1451

    finally:
        os.chdir(original)


def test_track3d_test_cavity():
    track3d_test_cavity()


def test_tracker_full_forward_3d_test_cavity():
    """Tracker.full_forward_3d() must produce the same result as the direct loop.

    This exercises the actual GUI code path (Tracker class, track_mode dispatch,
    step range, and finalize). The direct-loop test above would NOT catch bugs in
    step_forward_3d's frame range or a missing trackcorr_c_finish call.
    """
    original = os.getcwd()
    try:
        test_dir = os.path.join(
            os.path.dirname(__file__), "../../test_data/test_cavity"
        )
        os.chdir(test_dir)
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        from openptv2.tracker import Tracker

        cpar = ControlPar.from_yaml("parameters.yaml")
        cals = read_all_calibration(cpar.num_cams, base_path=".")
        tracker = Tracker(
            cpar,
            VolumePar.from_yaml("parameters.yaml"),
            TrackPar.from_yaml("parameters.yaml"),
            SequencePar.from_yaml("parameters.yaml"),
            cals,
        )
        tracker.full_forward_3d()

        assert tracker.npart == 2082
        assert tracker.nlinks == 1451
    finally:
        os.chdir(original)


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

        from optv.tracker import Tracker
        from optv.calibration import Calibration as CCalib
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )

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
        tracker = Tracker(cpar_c, vpar_c, tpar_c, spar_c, cal_c, naming)
        tracker.full_forward_3d()

        c_data = {}
        for s in range(first, last):
            c_data[s] = _parse_linkage_file(f"res/ptv_is.{s}")

        # --- Python run ---
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

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
        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            track3d_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        # --- Compare every field ---
        max_pos_diff = 0.0

        for s in range(first, last):
            py_data = _parse_linkage_file(f"res/ptv_is.{s}")

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
