"""Synthetic tracking test: known trajectories validate both track3d and trackcorr.

Ground truth particles with smooth trajectories are generated, projected to
2D pixel coordinates via the calibration, and written as test data files.
Both tracking algorithms run on the same data, and every resulting link is
checked against the known correspondence.
"""

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.track import (
    point_to_pixel,
    track_forward_start,
    trackback_c,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv2.algorithms.track3d import track3d_loop
from openptv2.algorithms.tracking_run import tr_new

TEST_DIR = Path(__file__).resolve().parent.parent.parent / "test_data" / "synthetic"
FIRST, LAST = 10001, 10008
NUM_CAMS = 4


# ---------------------------------------------------------------------------
# Ground-truth trajectories
# ---------------------------------------------------------------------------

def _make_trajectories():
    """Return dict mapping particle_id -> list of (frame, x, y, z).

    Trajectory types:
      0-4: constant velocity (straight lines)
      5-7: constant acceleration (curved)
      8-9: crossing paths (meet near frame 10005)
      10:  late entry (appears at frame 10003)
    """
    trajs = {}

    def _const_vel(pid, x0, y0, z0, vx, vy, vz, f_start=FIRST, f_end=LAST):
        pts = []
        for f in range(f_start, f_end + 1):
            t = f - f_start
            pts.append((f, x0 + vx * t, y0 + vy * t, z0 + vz * t))
        trajs[pid] = pts

    def _const_acc(pid, x0, y0, z0, vx, vy, vz, ax, ay, az):
        pts = []
        for f in range(FIRST, LAST + 1):
            t = f - FIRST
            pts.append((
                f,
                x0 + vx * t + 0.5 * ax * t * t,
                y0 + vy * t + 0.5 * ay * t * t,
                z0 + vz * t + 0.5 * az * t * t,
            ))
        trajs[pid] = pts

    # Straight lines — spread across the volume
    _const_vel(0,   0,   0,   0,   1.0,  0.5,  0.2)
    _const_vel(1, -30, -20,   5,   2.0,  1.0,  0.3)
    _const_vel(2,  20, -10,  -5,  -1.0,  0.8,  0.1)
    _const_vel(3, -10,  20,  10,   0.5, -1.5,  0.3)
    _const_vel(4,  30,  30, -10,  -2.0, -1.0,  0.5)

    # Curved trajectories (constant acceleration)
    _const_acc(5, -20,   0,   5,   1.0,  0.0,  0.0,   0.3,  0.15, 0.05)
    _const_acc(6,  10, -20,   0,   0.0,  2.0,  0.5,   0.15, -0.1, 0.03)
    _const_acc(7,  -5,  10,  -5,   2.0, -1.0,  0.3,  -0.15,  0.2, 0.02)

    # Near-miss paths: pass close but don't actually cross (y offset = 4)
    _const_vel(8, -10,   2,   3,   2.0,  0.0,  0.0)
    _const_vel(9,  10,   6,   3,  -2.0,  0.0,  0.0)

    # Late entry — appears at frame 10003
    _const_vel(10, 25,  15,   5,  -1.0, -0.5,  0.1, f_start=FIRST + 2)

    # Close neighbors moving in parallel (separation = 3 units)
    _const_vel(11,   5,  -5,   0,   1.5,  0.5,  0.1)
    _const_vel(12,   5,  -2,   0,   1.5,  0.5,  0.1)

    # Actual crossing in x-y plane (same z, paths cross at t=3.5)
    _const_vel(13, -14,  -8,  -3,   3.0,  1.0,  0.0)
    _const_vel(14,   7,  -1,  -3,  -3.0,  1.0,  0.0)

    return trajs


def _build_frame_data(trajs):
    """Convert trajectories to per-frame particle lists.

    Returns dict: frame_num -> list of (particle_id, x, y, z) sorted by id.
    """
    frames = {}
    for pid, pts in trajs.items():
        for f, x, y, z in pts:
            frames.setdefault(f, []).append((pid, x, y, z))
    for f in frames:
        frames[f].sort(key=lambda t: t[0])
    return frames


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def _load_calibrations():
    cals = []
    for cam in range(NUM_CAMS):
        cal = Calibration.from_file(
            str(TEST_DIR / f"cal/cam{cam+1}.tif.ori"),
            str(TEST_DIR / f"cal/cam{cam+1}.tif.addpar"),
        )
        cals.append(cal)
    return cals


def _generate_test_files(frames, cals, cpar):
    """Write rt_is, ptv_is, added, and target files for all frames.

    Targets must be sorted by pixel-y because candsearch_in_pix does a binary
    search + early termination on y. The rt_is correspondence indices must
    point to the target's position in the y-sorted array, not the particle
    slot.
    """
    res_dir = TEST_DIR / "res_orig"
    img_dir = TEST_DIR / "img_orig"

    for f_num, particles in frames.items():
        n = len(particles)

        # First, project all particles to all cameras and sort by pixel-y.
        # cam_target_order[cam] = list of (target_pnr_in_sorted_file, particle_slot)
        # cam_slot_to_targ[cam][particle_slot] = target index in sorted file
        cam_slot_to_targ = {}
        cam_targ_entries = {}
        for cam in range(NUM_CAMS):
            entries = []
            for slot, (pid, x, y, z) in enumerate(particles):
                pos = np.array([x, y, z], dtype=np.float64)
                px, py = point_to_pixel(pos, cals[cam], cpar)
                entries.append((slot, px, py))
            entries.sort(key=lambda t: t[2])  # sort by pixel-y
            cam_targ_entries[cam] = entries
            s2t = {}
            for targ_idx, (slot, px, py) in enumerate(entries):
                s2t[slot] = targ_idx
            cam_slot_to_targ[cam] = s2t

        # --- rt_is (correspondence) — cam indices point to sorted target positions ---
        with open(res_dir / f"rt_is.{f_num}", 'w') as fh:
            fh.write(f"{n}\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                cam_indices = " ".join(
                    f"{cam_slot_to_targ[cam][slot]:4d}" for cam in range(NUM_CAMS)
                )
                fh.write(f"{slot+1:4d} {x:9.3f} {y:9.3f} {z:9.3f} {cam_indices}\n")

        # --- ptv_is (linkage — initially unlinked) ---
        with open(res_dir / f"ptv_is.{f_num}", 'w') as fh:
            fh.write(f"{n}\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                fh.write(f"  -1   -2 {x:10.3f} {y:10.3f} {z:10.3f}\n")

        # --- added (prio file — initially unlinked) ---
        with open(res_dir / f"added.{f_num}", 'w') as fh:
            fh.write(f"{n}\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                fh.write(f"  -1   -2 {x:10.3f} {y:10.3f} {z:10.3f} 4\n")

        # --- target files per camera (y-sorted) ---
        for cam in range(NUM_CAMS):
            entries = cam_targ_entries[cam]
            with open(img_dir / f"cam{cam+1}.{f_num}_targets", 'w') as fh:
                fh.write(f"{n}\n")
                for targ_pnr, (orig_slot, px, py) in enumerate(entries):
                    fh.write(
                        f"{targ_pnr:4d} {px:9.4f} {py:9.4f} "
                        f"  100    10    10  1000 {orig_slot:5d}\n"
                    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_linkage(path):
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0])
    result = []
    for i in range(1, n + 1):
        parts = lines[i].split()
        result.append({
            "prev": int(parts[0]),
            "next": int(parts[1]),
            "x": float(parts[2]),
            "y": float(parts[3]),
            "z": float(parts[4]),
        })
    return result


def _build_ground_truth_links(frames):
    """Build expected prev/next links from ground truth.

    For each frame, maps (frame, slot_index) -> particle_id,
    and for each consecutive pair of frames, the expected next[slot_i]
    is the slot in the next frame with the same particle_id.

    Returns:
        gt_next: dict (frame, slot) -> slot_in_next_frame or -1
        gt_prev: dict (frame, slot) -> slot_in_prev_frame or -1
        slot_to_pid: dict (frame, slot) -> particle_id
    """
    slot_to_pid = {}
    pid_to_slot = {}  # (frame, pid) -> slot

    for f_num, particles in frames.items():
        for slot, (pid, x, y, z) in enumerate(particles):
            slot_to_pid[(f_num, slot)] = pid
            pid_to_slot[(f_num, pid)] = slot

    gt_next = {}
    gt_prev = {}

    sorted_frames = sorted(frames.keys())
    for fi in range(len(sorted_frames)):
        f = sorted_frames[fi]
        for slot, (pid, x, y, z) in enumerate(frames[f]):
            # next link
            if fi + 1 < len(sorted_frames):
                f_next = sorted_frames[fi + 1]
                if (f_next, pid) in pid_to_slot:
                    gt_next[(f, slot)] = pid_to_slot[(f_next, pid)]
                else:
                    gt_next[(f, slot)] = -1  # particle leaves
            else:
                gt_next[(f, slot)] = -1

            # prev link
            if fi - 1 >= 0:
                f_prev = sorted_frames[fi - 1]
                if (f_prev, pid) in pid_to_slot:
                    gt_prev[(f, slot)] = pid_to_slot[(f_prev, pid)]
                else:
                    gt_prev[(f, slot)] = -1  # particle just entered
            else:
                gt_prev[(f, slot)] = -1

    return gt_next, gt_prev, slot_to_pid


def _validate_tracking_result(frames, gt_next, slot_to_pid, label):
    """Check that every link in the output matches ground truth.

    Returns (n_correct, n_wrong, n_missed, errors).
    - correct: link matches ground truth
    - wrong: link exists but points to wrong particle
    - missed: ground truth says link should exist but tracker didn't find it
    """
    sorted_frames = sorted(frames.keys())
    n_correct = 0
    n_wrong = 0
    n_missed = 0
    errors = []

    for fi, f_num in enumerate(sorted_frames):
        if fi == len(sorted_frames) - 1:
            continue  # last frame has no "next"

        path = TEST_DIR / f"res/ptv_is.{f_num}"
        if not path.exists():
            continue

        linkage = _parse_linkage(path)

        for slot, entry in enumerate(linkage):
            expected_next = gt_next.get((f_num, slot), -1)
            actual_next = entry["next"]

            if actual_next >= 0 and expected_next >= 0:
                # Both have a link — check it's the right one
                next_frame = sorted_frames[fi + 1]
                actual_pid = slot_to_pid.get((next_frame, actual_next), "?")
                expected_pid = slot_to_pid.get((f_num, slot), "?")
                if actual_next == expected_next:
                    n_correct += 1
                else:
                    n_wrong += 1
                    errors.append(
                        f"  {label} frame {f_num} slot {slot} (pid={expected_pid}): "
                        f"next={actual_next} (pid={actual_pid}) "
                        f"expected={expected_next}"
                    )
            elif actual_next >= 0 and expected_next < 0:
                # Tracker created a link where none should exist
                # (particle leaves the scene) — count as wrong
                n_wrong += 1
                next_frame = sorted_frames[fi + 1]
                actual_pid = slot_to_pid.get((next_frame, actual_next), "?")
                errors.append(
                    f"  {label} frame {f_num} slot {slot}: "
                    f"spurious link to slot {actual_next} (pid={actual_pid})"
                )
            elif actual_next < 0 and expected_next >= 0:
                n_missed += 1
            # both < 0: correctly unlinked, nothing to count

    return n_correct, n_wrong, n_missed, errors


def _check_trajectory_distances(frames, label):
    """Check that no trajectory link jumps an unreasonable distance.

    Reads ptv_is files and verifies that linked particles have
    positions consistent with smooth motion.
    """
    sorted_frames = sorted(frames.keys())
    max_jump = 0.0
    jumps = []

    for fi in range(len(sorted_frames) - 1):
        f_curr = sorted_frames[fi]
        f_next = sorted_frames[fi + 1]

        curr_path = TEST_DIR / f"res/ptv_is.{f_curr}"
        next_path = TEST_DIR / f"res/ptv_is.{f_next}"
        if not curr_path.exists() or not next_path.exists():
            continue

        curr_data = _parse_linkage(curr_path)
        next_data = _parse_linkage(next_path)

        for slot, entry in enumerate(curr_data):
            if entry["next"] >= 0:
                nxt = entry["next"]
                if nxt < len(next_data):
                    dx = entry["x"] - next_data[nxt]["x"]
                    dy = entry["y"] - next_data[nxt]["y"]
                    dz = entry["z"] - next_data[nxt]["z"]
                    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                    max_jump = max(max_jump, dist)
                    jumps.append((f_curr, slot, nxt, dist))

    return max_jump, jumps


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """Generate synthetic test data once for all tests in this module."""
    trajs = _make_trajectories()
    frames = _build_frame_data(trajs)
    cals = _load_calibrations()
    cpar = ControlPar.from_yaml(str(TEST_DIR / "parameters.yaml"))
    _generate_test_files(frames, cals, cpar)
    gt_next, gt_prev, slot_to_pid = _build_ground_truth_links(frames)
    return {
        "trajs": trajs,
        "frames": frames,
        "cals": cals,
        "cpar": cpar,
        "gt_next": gt_next,
        "gt_prev": gt_prev,
        "slot_to_pid": slot_to_pid,
    }


def _setup_working_copy():
    """Copy res_orig/img_orig to res/img for a fresh run."""
    res = TEST_DIR / "res"
    img = TEST_DIR / "img"
    if res.exists():
        shutil.rmtree(res)
    if img.exists():
        shutil.rmtree(img)
    shutil.copytree(TEST_DIR / "res_orig", res)
    shutil.copytree(TEST_DIR / "img_orig", img)


def _setup_working_copy_res_only():
    """Reset only res/ (keeps img/ from prior run for second forward pass)."""
    res = TEST_DIR / "res"
    if res.exists():
        shutil.rmtree(res)
    shutil.copytree(TEST_DIR / "res_orig", res)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSyntheticTrack3d:
    """Test track3d (3-frame, 3D distance only) on synthetic data."""

    def test_track3d_links_correct(self, synthetic_data):
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals, 0.0001,
            )
            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                track3d_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            frames = synthetic_data["frames"]
            gt_next = synthetic_data["gt_next"]
            slot_to_pid = synthetic_data["slot_to_pid"]

            n_correct, n_wrong, n_missed, errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "track3d"
            )

            print(f"\ntrack3d: correct={n_correct}, wrong={n_wrong}, missed={n_missed}")
            print(f"  npart={run.npart}, nlinks={run.nlinks}")
            if errors:
                for e in errors:
                    print(e)

            assert n_wrong == 0, "track3d produced wrong links:\n" + "\n".join(errors)

            max_jump, _ = _check_trajectory_distances(frames, "track3d")
            print(f"  max trajectory jump: {max_jump:.3f}")
            assert max_jump < 10.0, f"track3d: trajectory jump {max_jump:.3f} too large"

        finally:
            os.chdir(original)

    def test_track3d_finds_all_links(self, synthetic_data):
        """track3d should find most ground-truth links (allow some misses)."""
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals, 0.0001,
            )
            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                track3d_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            frames = synthetic_data["frames"]
            gt_next = synthetic_data["gt_next"]
            slot_to_pid = synthetic_data["slot_to_pid"]

            n_correct, n_wrong, n_missed, _ = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "track3d"
            )
            total_expected = sum(
                1 for v in gt_next.values() if v >= 0
            )
            # Subtract links from last processed frame (track3d processes first..last-1)
            # and links for the last frame can't be established
            recovery_rate = n_correct / total_expected if total_expected > 0 else 0
            print(f"\ntrack3d recovery: {n_correct}/{total_expected} = {recovery_rate:.1%}")
            assert recovery_rate > 0.7, f"track3d recovery too low: {recovery_rate:.1%}"

        finally:
            os.chdir(original)


class TestSyntheticTrackcorr:
    """Test trackcorr (4-frame, angle/acceleration validated) on synthetic data."""

    def test_trackcorr_links_correct(self, synthetic_data):
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals, 0.0001,
            )
            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                trackcorr_c_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            frames = synthetic_data["frames"]
            gt_next = synthetic_data["gt_next"]
            slot_to_pid = synthetic_data["slot_to_pid"]

            n_correct, n_wrong, n_missed, errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "trackcorr"
            )

            print(f"\ntrackcorr: correct={n_correct}, wrong={n_wrong}, missed={n_missed}")
            print(f"  npart={run.npart}, nlinks={run.nlinks}")
            if errors:
                for e in errors:
                    print(e)

            assert n_wrong == 0, "trackcorr produced wrong links:\n" + "\n".join(errors)

            max_jump, _ = _check_trajectory_distances(frames, "trackcorr")
            print(f"  max trajectory jump: {max_jump:.3f}")
            assert max_jump < 10.0, f"trackcorr: trajectory jump {max_jump:.3f} too large"

        finally:
            os.chdir(original)

    def test_trackcorr_finds_all_links(self, synthetic_data):
        """trackcorr should find most ground-truth links."""
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals, 0.0001,
            )
            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                trackcorr_c_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            frames = synthetic_data["frames"]
            gt_next = synthetic_data["gt_next"]
            slot_to_pid = synthetic_data["slot_to_pid"]

            n_correct, n_wrong, n_missed, _ = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "trackcorr"
            )
            total_expected = sum(1 for v in gt_next.values() if v >= 0)
            recovery_rate = n_correct / total_expected if total_expected > 0 else 0
            print(f"\ntrackcorr recovery: {n_correct}/{total_expected} = {recovery_rate:.1%}")
            assert recovery_rate > 0.7, f"trackcorr recovery too low: {recovery_rate:.1%}"

        finally:
            os.chdir(original)


class TestSyntheticComparison:
    """Compare track3d and trackcorr on the same synthetic data."""

    def test_trackcorr_at_least_as_good_as_track3d(self, synthetic_data):
        """trackcorr should find at least as many correct links as track3d."""
        original = os.getcwd()
        frames = synthetic_data["frames"]
        gt_next = synthetic_data["gt_next"]
        slot_to_pid = synthetic_data["slot_to_pid"]

        try:
            os.chdir(TEST_DIR)

            # --- track3d ---
            _setup_working_copy()
            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]
            run_t3 = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals, 0.0001,
            )
            track_forward_start(run_t3)
            for step in range(run_t3.seq_par.first, run_t3.seq_par.last):
                track3d_loop(run_t3, step)
            trackcorr_c_finish(run_t3, run_t3.seq_par.last)

            t3_correct, t3_wrong, t3_missed, t3_errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "track3d"
            )

            # --- trackcorr ---
            _setup_working_copy()
            cals2 = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]
            run_tc = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals2, 0.0001,
            )
            track_forward_start(run_tc)
            for step in range(run_tc.seq_par.first, run_tc.seq_par.last):
                trackcorr_c_loop(run_tc, step)
            trackcorr_c_finish(run_tc, run_tc.seq_par.last)

            tc_correct, tc_wrong, tc_missed, tc_errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "trackcorr"
            )

            print(f"\n{'Algorithm':<12} {'Correct':>8} {'Wrong':>6} {'Missed':>7} {'Links':>6}")
            print(f"{'track3d':<12} {t3_correct:>8d} {t3_wrong:>6d} {t3_missed:>7d} {run_t3.nlinks:>6}")
            print(f"{'trackcorr':<12} {tc_correct:>8d} {tc_wrong:>6d} {tc_missed:>7d} {run_tc.nlinks:>6}")

            if t3_errors:
                print("\ntrack3d errors:")
                for e in t3_errors:
                    print(e)
            if tc_errors:
                print("\ntrackcorr errors:")
                for e in tc_errors:
                    print(e)

            assert tc_wrong == 0, "trackcorr produced wrong links"
            assert t3_wrong == 0, "track3d produced wrong links"
            assert tc_correct >= t3_correct, (
                f"trackcorr ({tc_correct}) found fewer correct links than "
                f"track3d ({t3_correct})"
            )

        finally:
            os.chdir(original)


class TestSyntheticForwardBackwardForward:
    """Test forward-backward-forward workflow on synthetic data."""

    def _run_fbf(self, add):
        """Run forward-backward-forward and return per-frame link counts."""
        _setup_working_copy()
        cpar = ControlPar.from_yaml("parameters.yaml")
        cals = [
            Calibration.from_file(
                f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
            )
            for i in range(cpar.num_cams)
        ]

        run = tr_new(
            SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
            VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
            4, 20000, "res/rt_is", "res/ptv_is", "res/added",
            cals, 0.0001,
        )
        run.tpar = run.tpar._replace(add=add)

        # Forward
        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)
        fwd_nlinks = run.nlinks

        # Backward
        trackback_c(run)

        # Forward again
        _setup_working_copy_res_only()
        run2 = tr_new(
            SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
            VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
            4, 20000, "res/rt_is", "res/ptv_is", "res/added",
            cals, 0.0001,
        )
        run2.tpar = run2.tpar._replace(add=add)
        track_forward_start(run2)
        for step in range(run2.seq_par.first, run2.seq_par.last):
            trackcorr_c_loop(run2, step)
        trackcorr_c_finish(run2, run2.seq_par.last)

        return run2, fwd_nlinks

    def test_fbf_preserves_links(self, synthetic_data):
        """Forward-backward-forward should not lose correct links."""
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            run, fwd_nlinks = self._run_fbf(add=0)

            frames = synthetic_data["frames"]
            gt_next = synthetic_data["gt_next"]
            slot_to_pid = synthetic_data["slot_to_pid"]

            n_correct, n_wrong, n_missed, errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "fbf"
            )

            print(f"\nFBF: correct={n_correct}, wrong={n_wrong}, missed={n_missed}")
            print(f"  fwd_nlinks={fwd_nlinks}, fbf_nlinks={run.nlinks}")

            assert n_wrong == 0, "FBF produced wrong links:\n" + "\n".join(errors)
            assert run.nlinks >= fwd_nlinks, (
                f"FBF lost links: {run.nlinks} < forward-only {fwd_nlinks}"
            )

            max_jump, _ = _check_trajectory_distances(frames, "fbf")
            assert max_jump < 10.0, f"FBF trajectory jump {max_jump:.3f} too large"

        finally:
            os.chdir(original)

    def test_fbf_backward_does_not_corrupt(self, synthetic_data):
        """Backward pass must not overwrite correct prev/next links."""
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            _setup_working_copy()
            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i+1}.tif.ori", f"cal/cam{i+1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

            run = tr_new(
                SequencePar.from_yaml("parameters.yaml"), TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"), ControlPar.from_yaml("parameters.yaml"),
                4, 20000, "res/rt_is", "res/ptv_is", "res/added",
                cals, 0.0001,
            )

            # Forward
            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                trackcorr_c_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            # Snapshot forward results
            fwd_links = {}
            for frame in range(FIRST, LAST + 1):
                path = TEST_DIR / f"res/ptv_is.{frame}"
                if path.exists():
                    fwd_links[frame] = _parse_linkage(path)

            # Backward
            trackback_c(run)

            # Check backward didn't corrupt any frame
            for frame in range(FIRST, LAST + 1):
                path = TEST_DIR / f"res/ptv_is.{frame}"
                if not path.exists():
                    continue
                back_data = _parse_linkage(path)
                fwd_data = fwd_links[frame]

                assert len(back_data) == len(fwd_data), (
                    f"Frame {frame}: particle count changed "
                    f"({len(fwd_data)} -> {len(back_data)})"
                )

                for slot in range(len(fwd_data)):
                    # next links must be preserved
                    assert back_data[slot]["next"] == fwd_data[slot]["next"], (
                        f"Frame {frame} slot {slot}: next link changed "
                        f"({fwd_data[slot]['next']} -> {back_data[slot]['next']})"
                    )
                    # prev links can only be added, not removed
                    if fwd_data[slot]["prev"] >= 0:
                        assert back_data[slot]["prev"] == fwd_data[slot]["prev"], (
                            f"Frame {frame} slot {slot}: existing prev link changed "
                            f"({fwd_data[slot]['prev']} -> {back_data[slot]['prev']})"
                        )

        finally:
            os.chdir(original)
