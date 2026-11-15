from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import openptv2.benchmarking as bm
from openptv2.batch.pyptv_batch import build_processing_experiment
from openptv2.benchmarking.metrics import _frames_dict, _match_frame
from openptv2.benchmarking.runner import read_trajectories
from openptv2.layered_tracking import (
    assign_layers,
    peel_and_track,
    peel_layers,
    prune_claimed_correspondences,
    prune_frame_correspondences,
    quality_scores,
    resolve_conflict_components,
)
from openptv2.storage import RunStore
from openptv2.tracker import Tracker, default_naming
from openptv2.tracking_postprocess import enforce_reciprocity


def test_quality_scores():
    cam = np.array([[0, 1, 2], [0, -1, -1]], dtype=np.int32)
    q = quality_scores(cam)
    assert list(q) == [3.0, 1.0]


def test_assign_layers_basic():
    scores = np.array([5, 4, 3, 2, 1], dtype=float)
    lyr = assign_layers(scores, n_layers=2, quantiles=[0.4, 1.0])
    # top 2 (scores 5,4) -> layer 0; rest -> layer 1
    assert lyr[0] == 0 and lyr[1] == 0
    assert all(lyr[2:] == 1)


def test_peel_layers():
    class _Store:
        def __init__(self):
            self._c = {}

        def has_correspondences(self, f):
            return f in self._c

        def read_correspondences(self, f):
            return self._c[f]

    store = _Store()
    pos = np.arange(15, dtype=float).reshape(5, 3)
    cam = np.array([[0, 1, 2]] * 5, dtype=np.int32)
    store._c[1] = (pos, cam)
    layers = peel_layers(store, [1], n_layers=3, quantiles=[0.34, 0.67, 1.0])
    assert len(layers) == 3
    total = sum(len(layers[i][1][0]) for i in range(3))
    assert total == 5


def test_prune_claimed_correspondences_removes_ghosts_keeps_unclaimed(tmp_path: Path):
    store = RunStore(tmp_path / "run.zarr", mode="w")

    # Frame 1:
    # row 0: Pass-1 real particle [10, 20, 30, 40] (claimed)
    # row 1: Ghost sharing cam0=10 with Pass-1 [10, 99, -1, -1] (must be pruned)
    # row 2: Ghost sharing cam3=40 with Pass-1 [-1, 88, 77, 40] (must be pruned)
    # row 3: Unclaimed real particle [11, 21, 31, 41] (must survive)
    # row 4: Unclaimed 3-cam particle [12, 22, -1, 42] (must survive; -1 ignored)
    pos1 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ids1 = np.array(
        [
            [10, 20, 30, 40],
            [10, 99, -1, -1],
            [-1, 88, 77, 40],
            [11, 21, 31, 41],
            [12, 22, -1, 42],
        ],
        dtype=np.int32,
    )
    store.write_correspondences(1, pos1, ids1)

    # Frame 2: no claimed targets -> all rows survive
    pos2 = np.array([[5.0, 6.0, 7.0]], dtype=np.float64)
    ids2 = np.array([[1, 2, -1, -1]], dtype=np.int32)
    store.write_correspondences(2, pos2, ids2)

    claimed = {
        1: [{10}, {20}, {30}, {40}],
        2: [set(), set(), set(), set()],
    }
    dropped = prune_claimed_correspondences(store, [1, 2], claimed)
    assert dropped == {1: 3, 2: 0}

    rem_pos1, rem_ids1 = store.read_correspondences(1)
    assert len(rem_pos1) == 2
    assert np.allclose(rem_pos1[:, 0], [3.0, 4.0])
    assert rem_ids1.tolist() == [[11, 21, 31, 41], [12, 22, -1, 42]]

    rem_pos2, _ = store.read_correspondences(2)
    assert len(rem_pos2) == 1

    # Also verify pure array helper directly
    p_arr, c_arr, mask = prune_frame_correspondences(pos1, ids1, claimed[1])
    assert mask.tolist() == [False, False, False, True, True]
    assert len(p_arr) == 2 and len(c_arr) == 2


def test_resolve_conflict_components_2_and_3_particle_optimal_vs_greedy():
    # Unambiguous 1-to-1 + 2-particle swapped conflict where greedy fails:
    # Greedy picks (0->0, cost 1.0), forcing (1->1, cost 10.0) => total 11.0.
    # Optimal Hungarian picks (0->1, cost 1.5) and (1->0, cost 1.2) => total 2.7.
    links_2 = [
        (0, 0, 1.0),
        (0, 1, 1.5),
        (1, 0, 1.2),
        (1, 1, 10.0),
        (5, 5, 0.3),  # separate unambiguous 1-to-1 component
    ]
    res_2 = resolve_conflict_components(links_2)
    assert res_2 == [(0, 1, 1.5), (1, 0, 1.2), (5, 5, 0.3)]

    # 3-particle conflict where greedy fails:
    # Greedy picks (10->10, 0.5), then (11->11, 0.6), forcing (12->12, 50.0) => 51.1.
    # Optimal picks (10->11, 1.0), (11->12, 1.0), (12->10, 0.8) => 2.8.
    links_3 = [
        (10, 10, 0.5),
        (10, 11, 1.0),
        (11, 11, 0.6),
        (11, 12, 1.0),
        (12, 10, 0.8),
        (12, 11, 5.0),
        (12, 12, 50.0),
    ]
    res_3 = resolve_conflict_components(links_3)
    assert [(s, d) for s, d, _ in res_3] == [(10, 11), (11, 12), (12, 10)]
    assert sum(c for _, _, c in res_3) == 2.8


def _link_metrics(
    true_tracks: dict[int, list[tuple[int, float, float, float]]],
    pred_tracks: dict[int, list[tuple[int, float, float, float]]],
    eps: float,
) -> tuple[float, float]:
    true_by_frame = _frames_dict(true_tracks)
    pred_by_frame = _frames_dict(pred_tracks)
    all_frames = sorted(set(true_by_frame) | set(pred_by_frame))
    frame_matches = {
        f: _match_frame(true_by_frame.get(f, {}), pred_by_frame.get(f, {}), eps)
        for f in all_frames
    }
    total_true_links = sum(
        1
        for pts in true_tracks.values()
        for a, b in zip(sorted(pts), sorted(pts)[1:])
        if b[0] == a[0] + 1
    )
    total_pred_links = 0
    matched_true_links = 0
    for pred_id, pts in pred_tracks.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        for i in range(len(pts_sorted) - 1):
            f0 = pts_sorted[i][0]
            f1 = pts_sorted[i + 1][0]
            total_pred_links += 1
            if f1 == f0 + 1:
                t0 = frame_matches.get(f0, {}).get(pred_id)
                t1 = frame_matches.get(f1, {}).get(pred_id)
                if t0 is not None and t0 == t1:
                    matched_true_links += 1
    link_yield = matched_true_links / total_true_links if total_true_links else 0.0
    precision = matched_true_links / total_pred_links if total_pred_links else 0.0
    return link_yield, precision


def _setup_exp(yaml_path: Path, first: int, last: int, dv: float):
    prev_cwd = os.getcwd()
    try:
        os.chdir(yaml_path.parent)
        exp = build_processing_experiment(yaml_path, first, last)
    finally:
        os.chdir(prev_cwd)
    tp = exp.track_par
    tp.set_dvxmin(-dv)
    tp.set_dvxmax(dv)
    tp.set_dvymin(-dv)
    tp.set_dvymax(dv)
    tp.set_dvzmin(-dv)
    tp.set_dvzmax(dv)
    tp.set_dacc(max(3.0, dv))
    return exp


def test_peel_and_track_dense_ghost_improves_over_single_pass(tmp_path: Path):
    spec = bm.ScenarioSpec(
        num_particles=40,
        num_frames=10,
        velocity=1.2,
        velocity_jitter=0.3,
        noise_mm=0.1,
        ghost_ratio=0.50,
        seed=42,
    )
    first_frame = 10001
    last_frame = first_frame + spec.num_frames - 1
    dv = max(3.0, spec.velocity * 3.0 + 3.0 * spec.noise_mm)
    eps = max(0.5, 2.0 * spec.noise_mm)

    true_tracks, frame_gt = bm.generate_scenario(spec)
    true_tracks_abs = {
        pid: [(f + first_frame, x, y, z) for f, x, y, z in pts]
        for pid, pts in true_tracks.items()
    }
    rig = bm.make_standard_rig(refract=False)

    # 1. Single-pass baseline
    from openptv2.storage.legacy import export_run, import_run

    base_dir = tmp_path / "baseline"
    base_yaml = bm.write_experiment(rig, frame_gt, base_dir, first_frame=first_frame)
    base_exp = _setup_exp(base_yaml, first_frame, last_frame, dv)
    base_store = import_run(base_dir, base_dir / "res" / "run.zarr")
    base_tracker = Tracker(
        base_exp.cpar,
        base_exp.vpar,
        base_exp.track_par,
        base_exp.spar,
        base_exp.cals,
        default_naming,
        store=base_store,
    )
    base_tracker.full_forward()
    base_tracker.full_backward()
    enforce_reciprocity("ptv_is", first_frame, last_frame, store=base_store)
    export_run(base_store, base_dir)
    base_pred = read_trajectories(base_dir / "res", first_frame, last_frame, num_cams=4)
    base_yield, base_prec = _link_metrics(true_tracks_abs, base_pred, eps=eps)

    # 2. 2-Pass Clean-and-Claim (peel_and_track)
    peel_dir = tmp_path / "peel"
    peel_yaml = bm.write_experiment(rig, frame_gt, peel_dir, first_frame=first_frame)
    peel_exp = _setup_exp(peel_yaml, first_frame, last_frame, dv)
    peel_store = import_run(peel_dir, peel_dir / "res" / "run.zarr")
    stats = peel_and_track(
        peel_exp.cpar,
        peel_exp.vpar,
        peel_exp.track_par,
        peel_exp.spar,
        peel_exp.cals,
        peel_store,
        corrective_passes=2,
        allow_shared_targets=False,
        predictor="poly",
    )
    export_run(peel_store, peel_dir)
    peel_pred = read_trajectories(peel_dir / "res", first_frame, last_frame, num_cams=4)
    peel_yield, peel_prec = _link_metrics(true_tracks_abs, peel_pred, eps=eps)

    print(
        f"\n[dense_ghost 50%] baseline yield={base_yield:.3f} prec={base_prec:.3f} | "
        f"peel_and_track yield={peel_yield:.3f} prec={peel_prec:.3f} | "
        f"core_tracks={stats.verified_core_tracks} pruned={stats.pruned_correspondences}"
    )

    assert stats.verified_core_tracks > 0
    assert stats.pruned_correspondences > 0
    assert peel_prec >= 0.95
    assert peel_yield > base_yield


def test_split_kinky_chain_trims_ghost_jump():
    from openptv2.layered_tracking import _split_kinky_chain

    # Smooth 5-frame track at v=(1,0,0) with a leading ghost jump at frame 0
    core_pos = {
        0: np.array([[0.0, 10.0, 0.0]]),  # kinky jump from (0,10,0) -> (1,0,0)
        1: np.array([[1.0, 0.0, 0.0]]),
        2: np.array([[2.0, 0.0, 0.0]]),
        3: np.array([[3.0, 0.0, 0.0]]),
        4: np.array([[4.0, 0.0, 0.0]]),
    }
    chain = [(f, 0) for f in range(5)]
    subs = _split_kinky_chain(chain, core_pos, max_accel_err=2.0)
    assert subs == [[(1, 0), (2, 0), (3, 0), (4, 0)]]


def test_run_store_write_and_read_rcm(tmp_path: Path):
    store = RunStore(tmp_path / "rcm.zarr", mode="w")
    assert store.read_rcm(10001) is None
    rcm = np.array([0.05, 0.42, 0.01], dtype=np.float32)
    store.write_rcm(10001, rcm)
    got = store.read_rcm(10001)
    assert got is not None
    assert np.allclose(got, rcm)


def test_relink_trajectory_gaps_bidir_vel_rejects_crossing_ghost(tmp_path: Path):
    from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
    from openptv2.tracking_postprocess import relink_trajectory_gaps

    store = RunStore(tmp_path / "gap.zarr", mode="w")
    # Track A moves along +x: frame 1 (0,0,0) -> frame 2 (1,0,0), ends at frame 2.
    # Crossing ghost B starts at frame 4 (3,0,0) -> frame 5 (3,2,0) (90-deg turn, v_out=(0,2,0)).
    store.write_linkage(
        1,
        np.array([PREV_NONE], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([[0.0, 0.0, 0.0]]),
    )
    store.write_linkage(
        2,
        np.array([0], dtype=np.int32),
        np.array([NEXT_NONE], dtype=np.int32),
        np.array([[1.0, 0.0, 0.0]]),
    )
    store.write_linkage(
        3,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
    )
    store.write_linkage(
        4,
        np.array([PREV_NONE], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([[3.0, 0.0, 0.0]]),
    )
    store.write_linkage(
        5,
        np.array([0], dtype=np.int32),
        np.array([NEXT_NONE], dtype=np.int32),
        np.array([[3.0, 2.0, 0.0]]),
    )

    res = relink_trajectory_gaps(
        "ptv_is",
        1,
        5,
        max_gap=2,
        max_accel_err=0.5,
        store=store,
        require_bidir_vel=True,
        uniqueness_ratio=1.5,
    )
    assert res["bridged_gaps"] == 0


def test_prune_kinky_links_breaks_step1_and_gap_bridged_kinks(tmp_path: Path):
    from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
    from openptv2.tracking_postprocess import prune_kinky_links, read_linkage

    store = RunStore(tmp_path / "kinks.zarr", mode="w")
    # Smooth track 0->1->2 at v=(1, 0, 0), then a sharp kink at frame 4 (gap-1 bridge 2->4).
    frames = {
        0: ([PREV_NONE], [0], [[0.0, 0.0, 0.0]]),
        1: ([0], [0], [[1.0, 0.0, 0.0]]),
        2: ([0], [0], [[2.0, 0.0, 0.0]]),
        4: ([0], [NEXT_NONE], [[4.0, 5.0, 0.0]]),
    }
    for f, (prv, nxt, xyz) in frames.items():
        store.write_linkage(
            f,
            np.array(prv, dtype=np.int32),
            np.array(nxt, dtype=np.int32),
            np.array(xyz, dtype=np.float64),
            name="ptv_is",
        )

    res = prune_kinky_links(
        "ptv_is",
        0,
        4,
        max_accel_err=0.5,
        max_angle_deg=45.0,
        store=store,
    )
    assert res["broken_kinks"] == 1
    _, nxt1, _ = read_linkage("ptv_is", 1, store=store)
    _, nxt2, _ = read_linkage("ptv_is", 2, store=store)
    prv4, _, _ = read_linkage("ptv_is", 4, store=store)
    assert nxt1[0] == 0  # smooth 0->1->2 preserved
    assert nxt2[0] == NEXT_NONE  # kinky 2->4 broken
    assert prv4[0] == PREV_NONE
