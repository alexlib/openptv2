"""
Engine comparison tests for track3d: Cython vs Python.

Each engine writes temporary .par files from the YAML config and then
reads them through its own native reader. This tests:
1. Reader parity — same files parsed identically by both engines
2. Algorithm parity — same inputs produce same outputs

Tolerance: 1e-5 (matches C test EPS)
"""

import os
import shutil
import yaml
import numpy as np
import pytest
from pathlib import Path

TOLERANCE = 1e-5

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "test_data", "track"
)
CAVITY_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "test_data", "test_cavity"
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


def _write_temp_par_files(src_dir, yaml_conf):
    """Write temporary .par files from YAML config for both engines to read.

    Returns paths to the temp files.
    """
    scene = yaml_conf["scene"]
    corresp = yaml_conf["correspondences"]
    tracking = yaml_conf["tracking"]
    seq_cfg = yaml_conf["sequence"]
    num_cams = len(yaml_conf["cameras"])

    # --- control.par ---
    control_par_path = os.path.join(src_dir, "temp_control.par")
    img_base = [
        seq_cfg["targets_template"].format(cam=cix + 1) for cix in range(num_cams)
    ]
    cal_base = [cam_spec["ori_file"] for cam_spec in yaml_conf["cameras"]]
    with open(control_par_path, "w") as f:
        f.write(f"{num_cams}\n")
        for i in range(num_cams):
            f.write(f"{img_base[i]}\n")
            f.write(f"{cal_base[i]}\n")
        f.write(f"{scene.get('hp_flag', 1)}\n")
        f.write(f"{scene.get('allcam_flag', 0)}\n")
        f.write(f"{scene.get('tiff_flag', 1)}\n")
        f.write(f"{scene['image_size'][0]}\n")
        f.write(f"{scene['image_size'][1]}\n")
        f.write(f"{scene['pixel_size'][0]}\n")
        f.write(f"{scene['pixel_size'][1]}\n")
        f.write(f"{scene.get('chfield', 0)}\n")
        f.write(f"{scene.get('cam_side_n', 1.0)}\n")
        f.write(f"{scene.get('wall_ns', [1.0])[0]}\n")
        f.write(f"{scene.get('object_side_n', 1.0)}\n")
        f.write(f"{scene.get('wall_thicks', [0.0])[0]}\n")

    # --- volume.par ---
    volume_par_path = os.path.join(src_dir, "temp_volume.par")
    z_spans = corresp["z_spans"]
    with open(volume_par_path, "w") as f:
        f.write(f"{corresp['x_span'][0]}\n")
        f.write(f"{z_spans[0][0]}\n")
        f.write(f"{z_spans[0][1]}\n")
        f.write(f"{corresp['x_span'][1]}\n")
        f.write(f"{z_spans[1][0]}\n")
        f.write(f"{z_spans[1][1]}\n")
        f.write(f"{corresp.get('pixels_x', 0.3)}\n")
        f.write(f"{corresp.get('pixels_y', 0.3)}\n")
        f.write(f"{corresp.get('pixels_tot', 0.01)}\n")
        f.write(f"{corresp.get('ref_gray', 0.0)}\n")
        f.write(f"{corresp.get('min_correlation', 33)}\n")
        f.write(f"{corresp.get('epipolar_band', 0.15)}\n")

    # --- tracking.par ---
    tracking_par_path = os.path.join(src_dir, "temp_tracking.par")
    vel = tracking["velocity_lims"]
    with open(tracking_par_path, "w") as f:
        f.write(f"{vel[0][0]}\n")
        f.write(f"{vel[0][1]}\n")
        f.write(f"{vel[1][0]}\n")
        f.write(f"{vel[1][1]}\n")
        f.write(f"{vel[2][0]}\n")
        f.write(f"{vel[2][1]}\n")
        f.write(f"{tracking['angle_lim']}\n")
        f.write(f"{tracking['accel_lim']}\n")
        f.write(f"{tracking['add_particle']}\n")

    # --- sequence.par ---
    sequence_par_path = os.path.join(src_dir, "temp_sequence.par")
    with open(sequence_par_path, "w") as f:
        for name in img_base:
            f.write(f"{name}\n")
        f.write(f"{seq_cfg['first']}\n")
        f.write(f"{seq_cfg['last']}\n")

    return control_par_path, volume_par_path, tracking_par_path, sequence_par_path


def _cleanup_temp_par_files(src_dir):
    """Remove temporary .par files."""
    for fname in (
        "temp_control.par",
        "temp_volume.par",
        "temp_tracking.par",
        "temp_sequence.par",
    ):
        path = os.path.join(src_dir, fname)
        if os.path.exists(path):
            os.remove(path)


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
    _cleanup_temp_par_files(src)


class TestTrack3DEngineComparison:
    """Compare Cython and Python track3d implementations.

    Both engines read the SAME temporary .par files through their own
    native readers, testing reader parity AND algorithm parity.
    """

    def test_python_track3d_matches_reference(self, dataset):
        """Python track3d output matches reference data."""
        src, yaml_file, ref_prefix, frame_range = dataset

        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        # Write temp .par files from YAML config
        ctrl_path, vol_path, track_path, seq_path = _write_temp_par_files(
            src, yaml_conf
        )

        naming = {
            "corres": f"{src}/res/particles",
            "linkage": f"{src}/res/linkage",
            "prio": f"{src}/res/whatever",
        }

        from algorithms.parameters import read_control_par, read_volume_par
        from algorithms.parameters import read_track_par, read_sequence_par
        from algorithms.track import Tracker
        from algorithms.calibration import Calibration

        # Python engine reads from temp files via its own reader
        cpar_python = read_control_par(Path(ctrl_path))
        vpar_python = read_volume_par(Path(vol_path))
        tpar_python = read_track_par(Path(track_path))
        spar_python = read_sequence_par(Path(seq_path))

        cals_python = []
        for cam_spec in yaml_conf["cameras"]:
            cal = Calibration()
            cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
            cals_python.append(cal)

        tracker = Tracker(
            cpar_python, vpar_python, tpar_python, spar_python, cals_python, naming
        )
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

        # Write temp .par files from YAML config
        ctrl_path, vol_path, track_path, seq_path = _write_temp_par_files(
            src, yaml_conf
        )

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

        # Cython engine reads from the SAME temp files via its own reader
        cpar_cython = ControlParams(num_cams=4)
        cpar_cython.read_control_par(ctrl_path)

        vpar_cython = VolumeParams()
        vpar_cython.read_volume_par(vol_path)

        tpar_cython = TrackingParams()
        tpar_cython.read_track_par(track_path)

        spar_cython = SequenceParams(num_cams=4)
        spar_cython.read_sequence_par(seq_path, 4)

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

        tracker = CythonTracker(
            cpar_cython, vpar_cython, tpar_cython, spar_cython, cals_cython, naming
        )
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
        """Cython and Python track3d produce identical results.

        Both engines read the SAME temp .par files through their own readers.
        """
        src, yaml_file, ref_prefix, frame_range = dataset

        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        # Write temp .par files from YAML config
        ctrl_path, vol_path, track_path, seq_path = _write_temp_par_files(
            src, yaml_conf
        )

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

        # Cython engine reads from temp files
        cpar_cython = ControlParams(num_cams=4)
        cpar_cython.read_control_par(ctrl_path)
        vpar_cython = VolumeParams()
        vpar_cython.read_volume_par(vol_path)
        tpar_cython = TrackingParams()
        tpar_cython.read_track_par(track_path)
        spar_cython = SequenceParams(num_cams=4)
        spar_cython.read_sequence_par(seq_path, 4)

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

        from algorithms.parameters import read_control_par, read_volume_par
        from algorithms.parameters import read_track_par, read_sequence_par
        from algorithms.track import Tracker
        from algorithms.calibration import Calibration

        # Python engine reads from the SAME temp files
        cpar_python = read_control_par(Path(ctrl_path))
        vpar_python = read_volume_par(Path(vol_path))
        tpar_python = read_track_par(Path(track_path))
        spar_python = read_sequence_par(Path(seq_path))

        cals_python = []
        for cam_spec in yaml_conf["cameras"]:
            cal = Calibration()
            cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
            cals_python.append(cal)

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
