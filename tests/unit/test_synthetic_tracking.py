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
            pts.append(
                (
                    f,
                    x0 + vx * t + 0.5 * ax * t * t,
                    y0 + vy * t + 0.5 * ay * t * t,
                    z0 + vz * t + 0.5 * az * t * t,
                )
            )
        trajs[pid] = pts

    # Straight lines — spread across the volume
    _const_vel(0, 0, 0, 0, 1.0, 0.5, 0.2)
    _const_vel(1, -30, -20, 5, 2.0, 1.0, 0.3)
    _const_vel(2, 20, -10, -5, -1.0, 0.8, 0.1)
    _const_vel(3, -10, 20, 10, 0.5, -1.5, 0.3)
    _const_vel(4, 30, 30, -10, -2.0, -1.0, 0.5)

    # Curved trajectories (constant acceleration)
    _const_acc(5, -20, 0, 5, 1.0, 0.0, 0.0, 0.3, 0.15, 0.05)
    _const_acc(6, 10, -20, 0, 0.0, 2.0, 0.5, 0.15, -0.1, 0.03)
    _const_acc(7, -5, 10, -5, 2.0, -1.0, 0.3, -0.15, 0.2, 0.02)

    # Near-miss paths: pass close but don't actually cross (y offset = 4)
    _const_vel(8, -10, 2, 3, 2.0, 0.0, 0.0)
    _const_vel(9, 10, 6, 3, -2.0, 0.0, 0.0)

    # Late entry — appears at frame 10003
    _const_vel(10, 25, 15, 5, -1.0, -0.5, 0.1, f_start=FIRST + 2)

    # Close neighbors moving in parallel (separation = 3 units)
    _const_vel(11, 5, -5, 0, 1.5, 0.5, 0.1)
    _const_vel(12, 5, -2, 0, 1.5, 0.5, 0.1)

    # Actual crossing in x-y plane (same z, paths cross at t=3.5)
    _const_vel(13, -14, -8, -3, 3.0, 1.0, 0.0)
    _const_vel(14, 7, -1, -3, -3.0, 1.0, 0.0)

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
            str(TEST_DIR / f"cal/cam{cam + 1}.tif.ori"),
            str(TEST_DIR / f"cal/cam{cam + 1}.tif.addpar"),
        )
        cals.append(cal)
    return cals


def _generate_test_files(frames, cals, cpar):
    """Write correspondences, initial linkage, and y-sorted targets into res_orig/run.zarr."""
    from openptv2.storage import RunStore

    store = RunStore(TEST_DIR / "res_orig" / "run.zarr", mode="w")

    for f_num, particles in frames.items():
        n = len(particles)

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

        pos_3d = np.array([[x, y, z] for (_pid, x, y, z) in particles], dtype=np.float64).reshape(n, 3)
        cam_ids = np.array(
            [[cam_slot_to_targ[cam][slot] for cam in range(NUM_CAMS)] for slot in range(n)],
            dtype=np.int32,
        ).reshape(n, NUM_CAMS)
        store.write_correspondences(frame=f_num, pos_3d=pos_3d, cam_target_ids=cam_ids)

        prev_ids = np.full(n, -1, dtype=np.int32)
        next_ids = np.full(n, -2, dtype=np.int32)
        prio = np.full(n, 4, dtype=np.int32)
        store.write_linkage(
            frame=f_num, prev_ids=prev_ids, next_ids=next_ids, pos_3d=pos_3d, name="ptv_is", prio=prio
        )

        for cam in range(NUM_CAMS):
            entries = cam_targ_entries[cam]
            arr = np.array(
                [
                    [targ_pnr, px, py, 100.0, 10.0, 10.0, 1000.0, orig_slot]
                    for targ_pnr, (orig_slot, px, py) in enumerate(entries)
                ],
                dtype=np.float64,
            ).reshape(n, 8)
            store.write_targets(cam, f_num, arr)


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
        result.append(
            {
                "prev": int(parts[0]),
                "next": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
            }
        )
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


def _read_linkage_store(store, f_num):
    if not store.has_linkage(f_num, "ptv_is"):
        return None
    prev_ids, next_ids, pos = store.read_linkage(f_num, "ptv_is")
    return [
        {
            "prev": int(pr),
            "next": int(nx),
            "x": float(pt[0]),
            "y": float(pt[1]),
            "z": float(pt[2]),
        }
        for pr, nx, pt in zip(prev_ids, next_ids, pos)
    ]


def _validate_tracking_result(frames, gt_next, slot_to_pid, label):
    """Check that every link in the output matches ground truth.

    Returns (n_correct, n_wrong, n_missed, errors).
    - correct: link matches ground truth
    - wrong: link exists but points to wrong particle
    - missed: ground truth says link should exist but tracker didn't find it
    """
    from openptv2.storage import RunStore

    store = RunStore(TEST_DIR / "res" / "run.zarr", mode="r")
    sorted_frames = sorted(frames.keys())
    n_correct = 0
    n_wrong = 0
    n_missed = 0
    errors = []

    for fi, f_num in enumerate(sorted_frames):
        if fi == len(sorted_frames) - 1:
            continue  # last frame has no "next"

        linkage = _read_linkage_store(store, f_num)
        if linkage is None:
            continue

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

    Reads ptv_is linkage from RunStore and verifies that linked particles have
    positions consistent with smooth motion.
    """
    from openptv2.storage import RunStore

    store = RunStore(TEST_DIR / "res" / "run.zarr", mode="r")
    sorted_frames = sorted(frames.keys())
    max_jump = 0.0
    jumps = []

    for fi in range(len(sorted_frames) - 1):
        f_curr = sorted_frames[fi]
        f_next = sorted_frames[fi + 1]

        curr_data = _read_linkage_store(store, f_curr)
        next_data = _read_linkage_store(store, f_next)
        if curr_data is None or next_data is None:
            continue

        for slot, entry in enumerate(curr_data):
            if entry["next"] >= 0:
                nxt = entry["next"]
                if nxt < len(next_data):
                    dx = entry["x"] - next_data[nxt]["x"]
                    dy = entry["y"] - next_data[nxt]["y"]
                    dz = entry["z"] - next_data[nxt]["z"]
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
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
    """Copy res_orig/run.zarr to res/run.zarr for a fresh run."""
    from openptv2.storage import RunStore

    res = TEST_DIR / "res"
    if res.exists():
        shutil.rmtree(res)
    res.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEST_DIR / "res_orig" / "run.zarr", res / "run.zarr")
    return RunStore(res / "run.zarr", mode="a")


def _setup_working_copy_res_only():
    """Reset res/run.zarr from res_orig/run.zarr."""
    return _setup_working_copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyntheticTrack3d:
    """Test track3d (3-frame, 3D distance only) on synthetic data."""

    def test_track3d_links_correct(self, synthetic_data):
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            store = _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

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
                cals,
                0.0001,
            )
            run.fb.store = store
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
            store = _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

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
                cals,
                0.0001,
            )
            run.fb.store = store
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
            total_expected = sum(1 for v in gt_next.values() if v >= 0)
            # Subtract links from last processed frame (track3d processes first..last-1)
            # and links for the last frame can't be established
            recovery_rate = n_correct / total_expected if total_expected > 0 else 0
            print(
                f"\ntrack3d recovery: {n_correct}/{total_expected} = {recovery_rate:.1%}"
            )
            assert recovery_rate > 0.7, f"track3d recovery too low: {recovery_rate:.1%}"

        finally:
            os.chdir(original)


class TestSyntheticTrackcorr:
    """Test trackcorr (4-frame, angle/acceleration validated) on synthetic data."""

    def test_trackcorr_links_correct(self, synthetic_data):
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            store = _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

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
                cals,
                0.0001,
            )
            run.fb.store = store
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

            print(
                f"\ntrackcorr: correct={n_correct}, wrong={n_wrong}, missed={n_missed}"
            )
            print(f"  npart={run.npart}, nlinks={run.nlinks}")
            if errors:
                for e in errors:
                    print(e)

            assert n_wrong == 0, "trackcorr produced wrong links:\n" + "\n".join(errors)

            max_jump, _ = _check_trajectory_distances(frames, "trackcorr")
            print(f"  max trajectory jump: {max_jump:.3f}")
            assert max_jump < 10.0, (
                f"trackcorr: trajectory jump {max_jump:.3f} too large"
            )

        finally:
            os.chdir(original)

    def test_trackcorr_finds_all_links(self, synthetic_data):
        """trackcorr should find most ground-truth links."""
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            store = _setup_working_copy()

            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

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
                cals,
                0.0001,
            )
            run.fb.store = store
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
            print(
                f"\ntrackcorr recovery: {n_correct}/{total_expected} = {recovery_rate:.1%}"
            )
            assert recovery_rate > 0.7, (
                f"trackcorr recovery too low: {recovery_rate:.1%}"
            )

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
            store = _setup_working_copy()
            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]
            run_t3 = tr_new(
                SequencePar.from_yaml("parameters.yaml"),
                TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"),
                ControlPar.from_yaml("parameters.yaml"),
                4,
                20000,
                "res/rt_is",
                "res/ptv_is",
                "res/added",
                cals,
                0.0001,
            )
            run_t3.fb.store = store
            track_forward_start(run_t3)
            for step in range(run_t3.seq_par.first, run_t3.seq_par.last):
                track3d_loop(run_t3, step)
            trackcorr_c_finish(run_t3, run_t3.seq_par.last)

            t3_correct, t3_wrong, t3_missed, t3_errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "track3d"
            )

            # --- trackcorr ---
            store = _setup_working_copy()
            cals2 = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]
            run_tc = tr_new(
                SequencePar.from_yaml("parameters.yaml"),
                TrackPar.from_yaml("parameters.yaml"),
                VolumePar.from_yaml("parameters.yaml"),
                ControlPar.from_yaml("parameters.yaml"),
                4,
                20000,
                "res/rt_is",
                "res/ptv_is",
                "res/added",
                cals2,
                0.0001,
            )
            run_tc.fb.store = store
            track_forward_start(run_tc)
            for step in range(run_tc.seq_par.first, run_tc.seq_par.last):
                trackcorr_c_loop(run_tc, step)
            trackcorr_c_finish(run_tc, run_tc.seq_par.last)

            tc_correct, tc_wrong, tc_missed, tc_errors = _validate_tracking_result(
                frames, gt_next, slot_to_pid, "trackcorr"
            )

            print(
                f"\n{'Algorithm':<12} {'Correct':>8} {'Wrong':>6} {'Missed':>7} {'Links':>6}"
            )
            print(
                f"{'track3d':<12} {t3_correct:>8d} {t3_wrong:>6d} {t3_missed:>7d} {run_t3.nlinks:>6}"
            )
            print(
                f"{'trackcorr':<12} {tc_correct:>8d} {tc_wrong:>6d} {tc_missed:>7d} {run_tc.nlinks:>6}"
            )

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
        store = _setup_working_copy()
        cpar = ControlPar.from_yaml("parameters.yaml")
        cals = [
            Calibration.from_file(
                f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
            )
            for i in range(cpar.num_cams)
        ]

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
            cals,
            0.0001,
        )
        run.fb.store = store
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
        store2 = _setup_working_copy_res_only()
        run2 = tr_new(
            SequencePar.from_yaml("parameters.yaml"),
            TrackPar.from_yaml("parameters.yaml"),
            VolumePar.from_yaml("parameters.yaml"),
            ControlPar.from_yaml("parameters.yaml"),
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            cals,
            0.0001,
        )
        run2.fb.store = store2
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
            store = _setup_working_copy()
            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

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
                cals,
                0.0001,
            )
            run.fb.store = store

            # Forward
            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                trackcorr_c_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            # Snapshot forward results
            fwd_links = {}
            for frame in range(FIRST, LAST + 1):
                data = _read_linkage_store(store, frame)
                if data is not None:
                    fwd_links[frame] = data

            # Backward
            trackback_c(run)

            # Check backward didn't corrupt any frame
            for frame in range(FIRST, LAST + 1):
                back_data = _read_linkage_store(store, frame)
                if back_data is None:
                    continue
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


class TestTrackbackCandidateGuard:
    """Regression test for the trackback_loop_fast acceptance-guard bug.

    track_kernels_corr.py's backward pass previously de-dented the
    ``(acc < dacc and angle < dangle) or acc < dacc * 0.1`` check so every
    candidate inside the velocity box was registered as a link candidate
    regardless of whether it passed the acceleration/angle test, using a
    stale distance value for ranking. Tightening dacc while keeping the
    velocity box wide forces multiple in-box, kinematically-wrong
    candidates at the crossing trajectories (particles 8-9, near frame
    10005) -- exactly the case the buggy guard would leak through.
    """

    def test_backward_prev_links_respect_acc_angle_guard(self, synthetic_data):
        original = os.getcwd()
        try:
            os.chdir(TEST_DIR)
            store = _setup_working_copy()
            cpar = ControlPar.from_yaml("parameters.yaml")
            cals = [
                Calibration.from_file(
                    f"cal/cam{i + 1}.tif.ori", f"cal/cam{i + 1}.tif.addpar"
                )
                for i in range(cpar.num_cams)
            ]

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
                cals,
                0.0001,
            )
            run.fb.store = store
            # Tight acceleration/angle gate, wide velocity box: candidates
            # near the crossing trajectories fall inside the box but most
            # fail acc/angle. Only the fix keeps them out of the link list.
            run.tpar = run.tpar._replace(dacc=0.02, dangle=15.0)

            track_forward_start(run)
            for step in range(run.seq_par.first, run.seq_par.last):
                trackcorr_c_loop(run, step)
            trackcorr_c_finish(run, run.seq_par.last)

            trackback_c(run)

            frames = synthetic_data["frames"]
            gt_prev = synthetic_data["gt_prev"]
            slot_to_pid = synthetic_data["slot_to_pid"]

            sorted_frames = sorted(frames.keys())
            n_wrong = 0
            errors = []
            for fi, f_num in enumerate(sorted_frames):
                if fi == 0:
                    continue  # first frame has no "prev"
                linkage = _read_linkage_store(store, f_num)
                if linkage is None:
                    continue
                for slot, entry in enumerate(linkage):
                    expected_prev = gt_prev.get((f_num, slot), -1)
                    actual_prev = entry["prev"]
                    if actual_prev >= 0 and actual_prev != expected_prev:
                        n_wrong += 1
                        prev_frame = sorted_frames[fi - 1]
                        actual_pid = slot_to_pid.get((prev_frame, actual_prev), "?")
                        expected_pid = slot_to_pid.get((f_num, slot), "?")
                        errors.append(
                            f"  frame {f_num} slot {slot} (pid={expected_pid}): "
                            f"prev={actual_prev} (pid={actual_pid}) "
                            f"expected={expected_prev}"
                        )

            assert n_wrong == 0, (
                "trackback accepted a candidate that fails the acc/angle "
                "guard:\n" + "\n".join(errors)
            )

        finally:
            os.chdir(original)
