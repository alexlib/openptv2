"""Stage 2 (docs/plans/2026-08-15-tracking-quality-overhaul.md): corrective
backward pass with track-assisted re-correspondence.

Builds a scene where the combinatorial correspondence step "missed" one
particle at one frame (its 2D targets exist in every camera, but no 3D
correspondence row references them at that frame) while the SAME particle's
track resumes the following frame with a known forward velocity -- exactly
the dropout the one-way detect -> correspond -> track pipeline cannot
recover from on its own, and what run_corrective_pass's backward walk is
for. Reuses test_data/tracking_synthetic's 12-particle ground truth and
calibration (already used by test_synthetic_tracking.py, test_tracking_
synthetic_dense.py, and test_tracking_warmup.py this session).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.algorithms.track import point_to_pixel
from openptv2.storage import RunStore
from openptv2.track_assisted import run_corrective_pass

FIX = Path(__file__).resolve().parents[2] / "test_data" / "tracking_synthetic"
DROPPED_PID = 5
DROPPED_FRAME = 10003


def _trajectories():
    """test_data/tracking_synthetic/generate.py, loaded under a unique
    module name -- test_data/tracking_synthetic_dense/generate.py is a
    DIFFERENT module also named "generate"; a bare `sys.path.insert` +
    `import generate` collides with whichever one another test in the same
    session imported first (observed: full-suite runs pick up the wrong
    one)."""
    spec = importlib.util.spec_from_file_location(
        "tracking_synthetic_generate", FIX / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.trajectories()


def _build_scene(store, cpar, cals):
    frames, _n = _trajectories()  # frame -> (12, 3) positions, particle p == row p
    num_cams = cpar.num_cams

    for f, positions in frames.items():
        pids_present = [p for p in range(len(positions)) if not (f == DROPPED_FRAME and p == DROPPED_PID)]

        # Targets: every particle projects into every camera, every frame --
        # the dropout is a correspondence-stage miss, not a detection miss.
        for cam in range(num_cams):
            rows = []
            for pnr, p in enumerate(range(len(positions))):
                px, py = point_to_pixel(positions[p], cals[cam], cpar)
                rows.append([pnr, px, py, 25, 5, 5, 100, -1])
            store.write_targets(cam, f, np.asarray(rows, dtype=np.float64))

        # Correspondences: identity target index == particle id in every
        # camera (matches the ground-truth positions exactly, no noise),
        # EXCEPT DROPPED_PID's row is simply absent at DROPPED_FRAME.
        pos_rows = [positions[p] for p in pids_present]
        id_rows = [[p] * num_cams for p in pids_present]
        store.write_correspondences(
            f, np.asarray(pos_rows), np.asarray(id_rows, dtype=np.int32)
        )

    return frames


def _build_linkage(store, frames):
    """Forward-tracking's own output given the dropout: identity prev/next
    for every particle except DROPPED_PID, whose track is cut in two by the
    missing frame -- ends at DROPPED_FRAME-1 (next=-1), restarts at
    DROPPED_FRAME+1 (prev=-1) with a normal forward link onward."""
    sorted_frames = sorted(frames)
    row_of = {}  # frame -> {pid: row}
    for f in sorted_frames:
        present = [p for p in range(len(frames[f])) if not (f == DROPPED_FRAME and p == DROPPED_PID)]
        row_of[f] = {p: i for i, p in enumerate(present)}

    for fi, f in enumerate(sorted_frames):
        n = len(row_of[f])
        prev = np.full(n, -1, dtype=np.int32)
        nxt = np.full(n, -2, dtype=np.int32)
        xyz = np.asarray([frames[f][p] for p in sorted(row_of[f], key=row_of[f].get)])

        f_prev = sorted_frames[fi - 1] if fi > 0 else None
        f_next = sorted_frames[fi + 1] if fi + 1 < len(sorted_frames) else None
        for p, row in row_of[f].items():
            if f_prev is not None and p in row_of[f_prev]:
                prev[row] = row_of[f_prev][p]
            if f_next is not None and p in row_of[f_next]:
                nxt[row] = row_of[f_next][p]
        store.write_linkage(f, prev, nxt, xyz, name="ptv_is")


@pytest.fixture
def scene(tmp_path):
    yaml_path = FIX / "parameters_Run1.yaml"
    cpar = ControlPar.from_yaml(str(yaml_path))
    vpar = VolumePar.from_yaml(str(yaml_path))
    tpar = TrackPar.from_yaml(str(yaml_path))
    spar = SequencePar.from_yaml(str(yaml_path), cpar.num_cams)
    cals = [
        Calibration.from_file(
            str(FIX / f"cal/cam{c + 1}.tif.ori"), str(FIX / f"cal/cam{c + 1}.tif.addpar")
        )
        for c in range(cpar.num_cams)
    ]

    store = RunStore(str(tmp_path / "run.zarr"), mode="w")
    frames = _build_scene(store, cpar, cals)
    _build_linkage(store, frames)

    return {"cpar": cpar, "vpar": vpar, "tpar": tpar, "spar": spar, "cals": cals, "store": store}


def test_backward_walk_recovers_the_dropped_correspondence(scene):
    store = scene["store"]

    # Confirm the dropout is really there before the pass runs.
    _pos, ids = store.read_correspondences(DROPPED_FRAME)
    assert DROPPED_PID not in ids[:, 0]
    prev, _next, _xyz = store.read_linkage(DROPPED_FRAME + 1, name="ptv_is")
    row_at_next = _next_frame_row(store, DROPPED_FRAME + 1, DROPPED_PID)
    assert prev[row_at_next] == -1, "test setup: dropped particle's track must start fresh"

    stats = run_corrective_pass(
        scene["cpar"], scene["vpar"], scene["tpar"], scene["spar"], scene["cals"], store,
        linkage_name="ptv_is", max_passes=1,
    )

    assert stats.claimed_total >= 1, "corrective pass claimed no particles"

    pos_after, ids_after = store.read_correspondences(DROPPED_FRAME)
    assert pos_after.shape[0] > ids.shape[0], "no row was appended at the dropped frame"

    # The claimed row's 3D position must be close to the true (noise-free)
    # position of the dropped particle.
    frames = _load_trajectories()
    true_pos = frames[DROPPED_FRAME][DROPPED_PID]
    dists = np.linalg.norm(pos_after - true_pos, axis=1)
    assert dists.min() < 1e-3, "claimed particle's position doesn't match the true dropped particle"

    # And frame DROPPED_FRAME+1's prev pointer for that particle must now
    # point at the newly-claimed row (rewired, not left at -1).
    prev_after, _n, _x = store.read_linkage(DROPPED_FRAME + 1, name="ptv_is")
    row_next = _next_frame_row(store, DROPPED_FRAME + 1, DROPPED_PID)
    claimed_row = int(np.argmin(dists))
    assert prev_after[row_next] == claimed_row


def _load_trajectories():
    frames, _n = _trajectories()
    return frames


def _next_frame_row(store, frame, pid):
    """Row index of the given particle id at `frame`, found by exact
    position match against ground truth (identity mapping everywhere
    except the dropped frame, by construction)."""
    frames = _load_trajectories()
    true_pos = frames[frame][pid]
    _prev, _next, xyz = store.read_linkage(frame, name="ptv_is")
    dists = np.linalg.norm(xyz - true_pos, axis=1)
    return int(np.argmin(dists))
