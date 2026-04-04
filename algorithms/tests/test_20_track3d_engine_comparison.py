"""
Engine comparison tests for track3d: Cython vs Python.

Runs both engines on the same data and compares outputs frame-by-frame.

Tolerance: 1e-5 (matches C test EPS)
"""

import os
import shutil
import yaml
import numpy as np
import pytest

TOLERANCE = 1e-5

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "track"
)
CAVITY_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "test_cavity"
)


def _read_particles_file(filepath):
    """Read a particles or rt_is file and return (count, positions array)."""
    with open(filepath) as f:
        lines = f.readlines()
    count = int(lines[0].strip())
    # Normalize negative counts to 0 (matching C reader behavior)
    if count < 0:
        count = 0
    positions = []
    for line in lines[1 : count + 1]:
        parts = list(map(float, line.split()))
        positions.append(parts[:3])  # x, y, z
    return count, np.array(positions) if positions else np.empty((0, 3))


@pytest.fixture(params=["track"])
def dataset(request, tmp_path):
    """Parametrized fixture for track dataset."""
    src = TRACK_DATA_DIR
    yaml_file = "conf.yaml"
    ref_prefix = "particles"
    frame_range = range(10001, 10003)

    res_orig = os.path.join(src, "res_orig")
    res_dst = os.path.join(src, "res")
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    shutil.copytree(res_orig, res_dst)

    newpart_dir = os.path.join(src, "newpart")
    backup_dir = str(tmp_path / "newpart_backup")
    shutil.copytree(newpart_dir, backup_dir)

    yield src, yaml_file, ref_prefix, frame_range

    # Cleanup res and res_orig copies
    if os.path.exists(res_dst):
        shutil.rmtree(res_dst)
    if os.path.exists(newpart_dir):
        shutil.rmtree(newpart_dir)
    shutil.copytree(backup_dir, newpart_dir)
    for extra in ("res_cython", "res_python"):
        extra_path = os.path.join(src, extra)
        if os.path.exists(extra_path):
            shutil.rmtree(extra_path)


class TestTrack3DEngineComparison:
    """Compare Cython and Python track3d implementations."""

    def test_python_track3d_matches_reference(self, dataset):
        """Python track3d output matches reference data."""
        src, yaml_file, ref_prefix, frame_range = dataset

        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        naming = {
            "corres": f"{src}/res/particles",
            "linkage": f"{src}/res/linkage",
            "prio": f"{src}/res/whatever",
        }

        from algorithms.parameters import (
            ControlPar,
            VolumePar,
            TrackParTuple,
            SequencePar,
        )
        from algorithms.track import Tracker
        from algorithms.calibration import Calibration

        cals = []
        for cam_spec in yaml_conf["cameras"]:
            cal = Calibration()
            cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
            cals.append(cal)

        scene = yaml_conf["scene"]
        seq_cfg = yaml_conf["sequence"]
        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]

        cpar = ControlPar(num_cams=len(yaml_conf["cameras"]))
        cpar.imx = scene["image_size"][0]
        cpar.imy = scene["image_size"][1]
        cpar.pix_x = scene["pixel_size"][0]
        cpar.pix_y = scene["pixel_size"][1]

        vpar = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[
                corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))
            ],
            z_max_lay=[
                corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))
            ],
        )

        vel = tracking["velocity_lims"]
        tpar = TrackParTuple(
            dvxmin=vel[0][0],
            dvxmax=vel[0][1],
            dvymin=vel[1][0],
            dvymax=vel[1][1],
            dvzmin=vel[2][0],
            dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking["add_particle"],
            dsumg=0.0,
            dn=0.0,
            dnx=0.0,
            dny=0.0,
        )

        img_base = [
            seq_cfg["targets_template"].format(cam=cix + 1)
            for cix in range(len(yaml_conf["cameras"]))
        ]
        spar = SequencePar(
            img_base_name=img_base,
            first=seq_cfg["first"],
            last=seq_cfg["last"],
        )

        tracker = Tracker(cpar, vpar, tpar, spar, cals, naming)
        tracker.full_forward_3d()

        for step in frame_range:
            out_file = os.path.join(src, "res", f"{ref_prefix}.{step}")
            ref_file = os.path.join(src, "res_orig", f"{ref_prefix}.{step}")

            assert os.path.exists(out_file), f"Missing output: {out_file}"

            out_count, out_pos = _read_particles_file(out_file)
            ref_count, ref_pos = _read_particles_file(ref_file)

            assert out_count == ref_count, (
                f"Step {step}: particle count {out_count} != {ref_count}"
            )

            if ref_count > 0:
                np.testing.assert_allclose(
                    out_pos,
                    ref_pos,
                    atol=TOLERANCE,
                    err_msg=f"Step {step}: positions differ",
                )

    def test_cython_track3d_matches_reference(self, dataset):
        """Cython track3d output matches reference data."""
        src, yaml_file, ref_prefix, frame_range = dataset

        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        naming = {
            "corres": f"{src}/res/particles",
            "linkage": f"{src}/res/linkage",
            "prio": f"{src}/res/whatever",
        }

        from optv.tracker import Tracker as CythonTracker
        from optv.calibration import Calibration as CythonCal
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )

        cals = []
        for cam_spec in yaml_conf["cameras"]:
            cal = CythonCal()
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")
            if addpar:
                cal.from_file(ori.encode(), addpar.encode())
            else:
                cal.from_file(ori.encode(), b"")
            cals.append(cal)

        scene = yaml_conf["scene"]
        cpar = ControlParams(len(yaml_conf["cameras"]), **scene)
        vpar = VolumeParams(**yaml_conf["correspondences"])
        tpar = TrackingParams(**yaml_conf["tracking"])

        seq_cfg = yaml_conf["sequence"]
        img_base = []
        for cix in range(len(yaml_conf["cameras"])):
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))
        spar = SequenceParams(
            image_base=img_base,
            frame_range=(seq_cfg["first"], seq_cfg["last"]),
        )

        tracker = CythonTracker(cpar, vpar, tpar, spar, cals, naming)
        tracker.full_forward_3d()

        for step in frame_range:
            out_file = os.path.join(src, "res", f"{ref_prefix}.{step}")
            ref_file = os.path.join(src, "res_orig", f"{ref_prefix}.{step}")

            assert os.path.exists(out_file), f"Missing output: {out_file}"

            out_count, out_pos = _read_particles_file(out_file)
            ref_count, ref_pos = _read_particles_file(ref_file)

            assert out_count == ref_count, (
                f"Step {step}: particle count {out_count} != {ref_count}"
            )

            if ref_count > 0:
                np.testing.assert_allclose(
                    out_pos,
                    ref_pos,
                    atol=TOLERANCE,
                    err_msg=f"Step {step}: positions differ",
                )

    def test_cython_vs_python_track3d_identical(self, dataset):
        """Cython and Python track3d produce identical results."""
        src, yaml_file, ref_prefix, frame_range = dataset

        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        # Run Cython
        cython_naming = {
            "corres": f"{src}/res_cython/particles",
            "linkage": f"{src}/res_cython/linkage",
            "prio": f"{src}/res_cython/whatever",
        }
        os.makedirs(os.path.join(src, "res_cython"), exist_ok=True)
        shutil.copytree(
            os.path.join(src, "res_orig"),
            os.path.join(src, "res_cython"),
            dirs_exist_ok=True,
        )

        from optv.tracker import Tracker as CythonTracker
        from optv.calibration import Calibration as CythonCal
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )

        cals_cython = []
        for cam_spec in yaml_conf["cameras"]:
            cal = CythonCal()
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")
            if addpar:
                cal.from_file(ori.encode(), addpar.encode())
            else:
                cal.from_file(ori.encode(), b"")
            cals_cython.append(cal)

        scene = yaml_conf["scene"]
        cpar_cython = ControlParams(len(yaml_conf["cameras"]), **scene)
        vpar_cython = VolumeParams(**yaml_conf["correspondences"])
        tpar_cython = TrackingParams(**yaml_conf["tracking"])

        seq_cfg = yaml_conf["sequence"]
        img_base = []
        for cix in range(len(yaml_conf["cameras"])):
            img_base.append(seq_cfg["targets_template"].format(cam=cix + 1))
        spar_cython = SequenceParams(
            image_base=img_base,
            frame_range=(seq_cfg["first"], seq_cfg["last"]),
        )

        cython_tracker = CythonTracker(
            cpar_cython,
            vpar_cython,
            tpar_cython,
            spar_cython,
            cals_cython,
            cython_naming,
        )
        cython_tracker.full_forward_3d()

        # Run Python
        python_naming = {
            "corres": f"{src}/res_python/particles",
            "linkage": f"{src}/res_python/linkage",
            "prio": f"{src}/res_python/whatever",
        }
        os.makedirs(os.path.join(src, "res_python"), exist_ok=True)
        shutil.copytree(
            os.path.join(src, "res_orig"),
            os.path.join(src, "res_python"),
            dirs_exist_ok=True,
        )

        from algorithms.parameters import (
            ControlPar,
            VolumePar,
            TrackParTuple,
            SequencePar,
        )
        from algorithms.track import Tracker
        from algorithms.calibration import Calibration

        cals_python = []
        for cam_spec in yaml_conf["cameras"]:
            cal = Calibration()
            cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
            cals_python.append(cal)

        cpar_python = ControlPar(num_cams=len(yaml_conf["cameras"]))
        cpar_python.imx = scene["image_size"][0]
        cpar_python.imy = scene["image_size"][1]
        cpar_python.pix_x = scene["pixel_size"][0]
        cpar_python.pix_y = scene["pixel_size"][1]

        corresp = yaml_conf["correspondences"]
        tracking = yaml_conf["tracking"]
        vpar_python = VolumePar(
            x_lay=corresp["x_span"],
            z_min_lay=[
                corresp["z_spans"][i][0] for i in range(len(corresp["z_spans"]))
            ],
            z_max_lay=[
                corresp["z_spans"][i][1] for i in range(len(corresp["z_spans"]))
            ],
        )

        vel = tracking["velocity_lims"]
        tpar_python = TrackParTuple(
            dvxmin=vel[0][0],
            dvxmax=vel[0][1],
            dvymin=vel[1][0],
            dvymax=vel[1][1],
            dvzmin=vel[2][0],
            dvzmax=vel[2][1],
            dangle=tracking["angle_lim"],
            dacc=tracking["accel_lim"],
            add=tracking["add_particle"],
            dsumg=0.0,
            dn=0.0,
            dnx=0.0,
            dny=0.0,
        )

        img_base = [
            seq_cfg["targets_template"].format(cam=cix + 1)
            for cix in range(len(yaml_conf["cameras"]))
        ]
        spar_python = SequencePar(
            img_base_name=img_base,
            first=seq_cfg["first"],
            last=seq_cfg["last"],
        )

        python_tracker = Tracker(
            cpar_python,
            vpar_python,
            tpar_python,
            spar_python,
            cals_python,
            python_naming,
        )
        python_tracker.full_forward_3d()

        # Compare
        for step in frame_range:
            cython_file = os.path.join(src, "res_cython", f"{ref_prefix}.{step}")
            python_file = os.path.join(src, "res_python", f"{ref_prefix}.{step}")

            cython_count, cython_pos = _read_particles_file(cython_file)
            python_count, python_pos = _read_particles_file(python_file)

            assert cython_count == python_count, (
                f"Step {step}: Cython count {cython_count} != Python count {python_count}"
            )

            if python_count > 0:
                np.testing.assert_allclose(
                    cython_pos,
                    python_pos,
                    atol=TOLERANCE,
                    err_msg=f"Step {step}: Cython vs Python positions differ",
                )

        # Cleanup
        shutil.rmtree(os.path.join(src, "res_cython"))
        shutil.rmtree(os.path.join(src, "res_python"))
