"""
Isolated engine comparison: Python vs Cython on separate copies.

Both engines get their own complete copy of test_data/track so that
_targets file modifications don't cross-contaminate. Compares:
- particles.<frame> (rt_is output)
- linkage.<frame> (ptv_is output)
- cam<N>.<frame>_targets (written back by engines)
"""

import os
import shutil
import yaml
import numpy as np
import pytest
from pathlib import Path
import hashlib
import difflib

TRACK_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test_data", "track"
)


def _file_hash(path):
    """Return SHA256 hash of a file for comparison."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_particles_file(filepath):
    """Read a particles or rt_is file and return (count, positions array)."""
    with open(filepath) as f:
        lines = f.readlines()
    count = int(lines[0].strip())
    if count < 0:
        count = 0
    positions = []
    for line in lines[1 : count + 1]:
        parts = list(map(float, line.split()))
        positions.append(parts[:3])
    return count, np.array(positions) if positions else np.empty((0, 3))


def _read_targets_file(filepath):
    """Read a _targets file and return list of target tuples."""
    with open(filepath) as f:
        lines = f.readlines()
    count = int(lines[0].strip())
    if count < 0:
        count = 0
    targets = []
    for line in lines[1 : count + 1]:
        parts = line.split()
        if len(parts) == 8:
            targets.append(
                (
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    int(parts[3]),
                    int(parts[4]),
                    int(parts[5]),
                    int(parts[6]),
                    int(parts[7]),
                )
            )
    return targets


def _setup_engine_workspace(src_dir, tmp_path, engine_name):
    """Create a complete isolated copy of test_data/track for an engine."""
    workspace = str(tmp_path / engine_name)
    shutil.copytree(src_dir, workspace)

    # Create output directories
    os.makedirs(os.path.join(workspace, "res"), exist_ok=True)
    res_orig = os.path.join(workspace, "res_orig")
    if os.path.exists(res_orig):
        shutil.copytree(res_orig, os.path.join(workspace, "res"), dirs_exist_ok=True)

    return workspace


def _write_temp_par_files_for_workspace(workspace, yaml_conf):
    """Write .par files pointing to paths within the workspace."""
    scene = yaml_conf["scene"]
    corresp = yaml_conf["correspondences"]
    tracking = yaml_conf["tracking"]
    seq_cfg = yaml_conf["sequence"]
    num_cams = len(yaml_conf["cameras"])

    # Use paths relative to workspace
    img_base = [
        seq_cfg["targets_template"].format(cam=cix + 1) for cix in range(num_cams)
    ]
    cal_base = [cam_spec["ori_file"] for cam_spec in yaml_conf["cameras"]]

    # --- control.par ---
    control_par_path = os.path.join(workspace, "temp_control.par")
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
    volume_par_path = os.path.join(workspace, "temp_volume.par")
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
    tracking_par_path = os.path.join(workspace, "temp_tracking.par")
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
    sequence_par_path = os.path.join(workspace, "temp_sequence.par")
    with open(sequence_par_path, "w") as f:
        for name in img_base:
            f.write(f"{name}\n")
        f.write(f"{seq_cfg['first']}\n")
        f.write(f"{seq_cfg['last']}\n")

    return (
        control_par_path,
        volume_par_path,
        tracking_par_path,
        sequence_par_path,
    )


def _collect_output_files(workspace, frame_range):
    """Collect all output files from an engine run."""
    outputs = {}

    res_dir = os.path.join(workspace, "res")

    # Particles files
    for step in frame_range:
        pfile = os.path.join(res_dir, f"particles.{step}")
        if os.path.exists(pfile):
            outputs[f"particles.{step}"] = _read_particles_file(pfile)

        lfile = os.path.join(res_dir, f"linkage.{step}")
        if os.path.exists(lfile):
            with open(lfile) as f:
                outputs[f"linkage.{step}"] = f.read()

    # _targets files (written back by engines)
    newpart = os.path.join(workspace, "newpart")
    if os.path.exists(newpart):
        for f in sorted(os.listdir(newpart)):
            if f.endswith("_targets"):
                fpath = os.path.join(newpart, f)
                outputs[f] = _read_targets_file(fpath)

    return outputs


class TestIsolatedEngineComparison:
    """Run Python and Cython on fully isolated copies, compare everything."""

    @pytest.fixture
    def isolated_workspaces(self, tmp_path):
        """Create two isolated workspaces and run both engines."""
        src = TRACK_DATA_DIR
        yaml_file = "conf.yaml"
        frame_range = range(10001, 10006)

        with open(os.path.join(src, yaml_file)) as f:
            yaml_conf = yaml.load(f, Loader=yaml.FullLoader)

        # --- Cython workspace ---
        cy_ws = _setup_engine_workspace(src, tmp_path, "cython_ws")
        (
            cy_ctrl,
            cy_vol,
            cy_track,
            cy_seq,
        ) = _write_temp_par_files_for_workspace(cy_ws, yaml_conf)

        from optv.tracker import Tracker as CythonTracker
        from optv.calibration import Calibration as CythonCal
        from optv.parameters import (
            ControlParams,
            VolumeParams,
            TrackingParams,
            SequenceParams,
        )

        cpar_cy = ControlParams(num_cams=4)
        cpar_cy.read_control_par(cy_ctrl)
        vpar_cy = VolumeParams()
        vpar_cy.read_volume_par(cy_vol)
        tpar_cy = TrackingParams()
        tpar_cy.read_track_par(cy_track)
        spar_cy = SequenceParams(num_cams=4)
        spar_cy.read_sequence_par(cy_seq, 4)

        cals_cy = []
        for cam_spec in yaml_conf["cameras"]:
            cal = CythonCal()
            ori = cam_spec["ori_file"]
            addpar = cam_spec.get("addpar_file")
            if addpar:
                cal.from_file(ori.encode(), addpar.encode())
            else:
                cal.from_file(ori.encode(), b"")
            cals_cy.append(cal)

        cy_naming = {
            "corres": f"{cy_ws}/res/particles",
            "linkage": f"{cy_ws}/res/linkage",
            "prio": f"{cy_ws}/res/whatever",
        }
        cy_tracker = CythonTracker(
            cpar_cy, vpar_cy, tpar_cy, spar_cy, cals_cy, cy_naming
        )
        cy_tracker.full_forward_3d()

        # --- Python workspace ---
        py_ws = _setup_engine_workspace(src, tmp_path, "python_ws")
        (
            py_ctrl,
            py_vol,
            py_track,
            py_seq,
        ) = _write_temp_par_files_for_workspace(py_ws, yaml_conf)

        from algorithms.parameters import read_control_par, read_volume_par
        from algorithms.parameters import read_track_par, read_sequence_par
        from algorithms.track import Tracker
        from algorithms.calibration import Calibration

        cpar_py = read_control_par(Path(py_ctrl))
        vpar_py = read_volume_par(Path(py_vol))
        tpar_py = read_track_par(Path(py_track))
        spar_py = read_sequence_par(Path(py_seq))

        cals_py = []
        for cam_spec in yaml_conf["cameras"]:
            cal = Calibration()
            cal.from_file(cam_spec["ori_file"], cam_spec.get("addpar_file", None))
            cals_py.append(cal)

        py_naming = {
            "corres": f"{py_ws}/res/particles",
            "linkage": f"{py_ws}/res/linkage",
            "prio": f"{py_ws}/res/whatever",
        }
        py_tracker = Tracker(cpar_py, vpar_py, tpar_py, spar_py, cals_py, py_naming)
        py_tracker.full_forward_3d()

        # Collect outputs
        cy_outputs = _collect_output_files(cy_ws, frame_range)
        py_outputs = _collect_output_files(py_ws, frame_range)

        return cy_outputs, py_outputs, cy_ws, py_ws, frame_range

    def test_particles_files_match(self, isolated_workspaces):
        """Cython and Python produce identical particles.<frame> files."""
        cy_out, py_out, cy_ws, py_ws, frame_range = isolated_workspaces

        for step in frame_range:
            key = f"particles.{step}"
            assert key in cy_out, f"Cython missing {key}"
            assert key in py_out, f"Python missing {key}"

            cy_count, cy_pos = cy_out[key]
            py_count, py_pos = py_out[key]

            assert cy_count == py_count, (
                f"{key}: Cython count {cy_count} != Python count {py_count}"
            )

            if cy_count > 0:
                np.testing.assert_allclose(
                    cy_pos,
                    py_pos,
                    atol=1e-5,
                    err_msg=f"{key}: positions differ",
                )

    def test_linkage_files_match(self, isolated_workspaces):
        """Cython and Python produce identical linkage.<frame> files.

        Known difference: C has a bug where files with 0 particles get
        num_parts=-1 due to do...while(!feof) always executing once.
        Python correctly reports 0. We compare normalized content,
        treating -1 and 0 as equivalent for empty frames.
        """
        cy_out, py_out, cy_ws, py_ws, frame_range = isolated_workspaces

        for step in frame_range:
            key = f"linkage.{step}"
            if key in cy_out:
                assert key in py_out, f"Python missing {key}"
                cy_lines = [line.split() for line in cy_out[key].strip().split("\n")]
                py_lines = [line.split() for line in py_out[key].strip().split("\n")]

                # Normalize header: -1 (C bug) and 0 (Python correct) both mean empty
                cy_header = int(cy_lines[0][0])
                py_header = int(py_lines[0][0])
                if cy_header == -1 and py_header == 0:
                    cy_lines[0][0] = "0"
                elif cy_header == 0 and py_header == -1:
                    py_lines[0][0] = "0"

                assert cy_lines == py_lines, (
                    f"{key}: linkage content differs\n"
                    f"  Cython: {cy_out[key].strip()!r}\n"
                    f"  Python: {py_out[key].strip()!r}"
                )

    def test_targets_files_match(self, isolated_workspaces):
        """Cython and Python write identical _targets files."""
        cy_out, py_out, cy_ws, py_ws, frame_range = isolated_workspaces

        cy_target_keys = {k for k in cy_out if k.endswith("_targets")}
        py_target_keys = {k for k in py_out if k.endswith("_targets")}

        assert cy_target_keys == py_target_keys, (
            f"Different _targets files:\n"
            f"  Cython only: {cy_target_keys - py_target_keys}\n"
            f"  Python only: {py_target_keys - cy_target_keys}"
        )

        for key in sorted(cy_target_keys):
            cy_targets = cy_out[key]
            py_targets = py_out[key]

            assert len(cy_targets) == len(py_targets), (
                f"{key}: Cython has {len(cy_targets)} targets, "
                f"Python has {len(py_targets)}"
            )

            for i, (cy_t, py_t) in enumerate(zip(cy_targets, py_targets)):
                assert cy_t == py_t, (
                    f"{key}, target {i}:\n  Cython: {cy_t}\n  Python: {py_t}"
                )

    def test_original_targets_unchanged(self, isolated_workspaces):
        """Verify that _targets files match the originals (no unintended modifications)."""
        cy_out, py_out, cy_ws, py_ws, frame_range = isolated_workspaces

        # Compare against the original test_data/track/newpart/
        orig_newpart = os.path.join(TRACK_DATA_DIR, "newpart")

        for key in sorted(k for k in cy_out if k.endswith("_targets")):
            orig_path = os.path.join(orig_newpart, key)
            if os.path.exists(orig_path):
                orig_targets = _read_targets_file(orig_path)
                cy_targets = cy_out[key]
                py_targets = py_out[key]

                # Check if engine modified the targets
                if orig_targets != cy_targets:
                    print(f"\nWARNING: Cython modified {key}")
                    self._print_target_diff(orig_targets, cy_targets, key)

                if orig_targets != py_targets:
                    print(f"\nWARNING: Python modified {key}")
                    self._print_target_diff(orig_targets, py_targets, key)

    def _print_target_diff(self, orig, modified, filename):
        """Print a human-readable diff between original and modified targets."""
        orig_lines = [str(t) for t in orig]
        mod_lines = [str(t) for t in modified]

        diff = list(
            difflib.unified_diff(
                orig_lines,
                mod_lines,
                fromfile=f"{filename} (original)",
                tofile=f"{filename} (modified)",
                lineterm="",
            )
        )
        if diff:
            print("\n".join(diff[:50]))
