"""Assemble full-diameter correspondences and track one Illmenau rig.

Reads partial npz shards from correspond_fulldiam.py, merges them, rebuilds
leaves/seen/ghost exactly like the production Tracking plugin (projected
2D-leaf costs, RCM ghost model), tracks with the yaml `track:` section
(overridden to the tuned gates), and writes res/run_fulldiam.zarr.

Usage (ONE rig at a time; env selects it):
    ILLMENAU_DIR=.../openptv_illmenau_4cam ILLMENAU_CAMS=1,2,3,4 \
    uv run python scripts/illmenau/assemble_track.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _config as CFG  # noqa: E402

t0 = time.time()


def log(s):
    print(s, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--out", default="run_fulldiam.zarr")
    a = ap.parse_args()
    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.plugins.two_phase_tracking import (
        AUTO_MAX_NEIGHBOURS,
        AUTO_MIN_KINK_RATIO,
        DEFAULT_Q_MODEL,
        DEFAULT_Q_SEED,
        DEFAULT_Q_YOUNG,
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
        _links_to_linkage,
        resolve_confirm,
        resolve_confirm_auto,
    )
    from openptv2.point_quality import (
        brightness_spread,
        fit_scale,
        ghost_probability,
        log_brightness_from_arrays,
        rcm_from_arrays,
    )
    from openptv2.storage import RunStore
    from openptv2.storage.run_store import seen_mask

    y = yaml.safe_load((CFG.DIR / "parameters_Run1.yaml").read_text())
    track_cfg = dict(y.get("track", {}))
    log(f"cams {CFG.CAMS}: track cfg v_max={track_cfg.get('v_max')} "
        f"confirm_tol={track_cfg.get('confirm_tol')} "
        f"bidirectional={track_cfg.get('bidirectional')}")

    v_max = float(track_cfg.get("v_max", track_cfg.get("dvxmax", 15.5)))
    confirm_tol, confirm_ends, note = resolve_confirm(track_cfg, v_max)
    confirm_auto = resolve_confirm_auto(track_cfg)
    log(f"resolved: v_max={v_max} confirm_tol={confirm_tol} "
        f"confirm_ends={confirm_ends} ({note})")
    assert abs(v_max - 40.0) < 1e-9, f"tuned v_max=40 expected, got {v_max}"
    assert confirm_tol is None, f"confirm off expected, got {confirm_tol}"

    cpar = ControlPar(
        num_cams=4, imx=2560, imy=2048, pix_x=0.005, pix_y=0.005,
        mm=MmNp(n1=1.0, n2=[1.0], d=[0.0], n3=1.0),
        chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
        img_base_name=[""] * 4, cal_img_base_name=[""] * 4)
    cals = CFG.load_calibrations()
    old = RunStore(str(CFG.DIR / "res" / "run.zarr"), mode="r")

    # ---- merge shards ----------------------------------------------------
    parts = sorted((CFG.DIR / "cal").glob("fulldiam_part_*.npz"))
    assert parts, "no fulldiam_part_*.npz shards — run correspond_fulldiam first"
    merged = {}
    for p in parts:
        with np.load(str(p), allow_pickle=True) as z:
            keys = list(z.files)
            for k in keys:
                if k.endswith("/pos"):
                    f = int(k.split("/")[0])
                    merged[f] = (np.asarray(z[k], float),
                                 np.asarray(z[f"{f}/ids"], np.int32))
    frames = sorted(merged)
    if a.first is not None:
        frames = [f for f in frames if a.first <= f <= (a.last or a.first)]
    log(f"merged {len(parts)} shards, {len(frames)} frames "
        f"{frames[0]}..{frames[-1]} ({time.time() - t0:.0f}s)")

    # ---- per-frame arrays (mirrors Tracking plugin data prep) -------------
    new_path = CFG.DIR / "res" / a.out
    new = RunStore(str(new_path), mode="w")
    P, LE, SE, GH, LB = [], [], [], [], []
    all_pos, all_rcm, all_nseen = [], [], []
    for fi, f in enumerate(frames):
        pos, cam_ids = merged[f]
        rows = []
        for c in range(4):
            key = f"targets/cam_{c}/frame_{f:06d}"
            rows.append(np.asarray(old.root[key]) if key in old.root else None)
            if rows[-1] is not None:
                new.write_targets(c, f, rows[-1])
        new.write_correspondences(f, pos, cam_ids)
        n = len(pos)
        xy = np.full((n, 4, 2), np.nan)
        for c in range(4):
            if rows[c] is not None:
                valid = cam_ids[:, c] >= 0
                xy[valid, c] = rows[c][cam_ids[valid, c]][:, 1:3]
        P.append(np.asarray(pos))
        LE.append(np.nan_to_num(xy.reshape(n, -1)))
        seen = seen_mask(cam_ids)
        SE.append(seen)
        rcm = rcm_from_arrays(pos, cam_ids, rows, cals, cpar)
        all_pos.append(pos)
        all_rcm.append(rcm)
        all_nseen.append(seen.sum(axis=1))
        LB.append(log_brightness_from_arrays(cam_ids, rows))
        if (fi + 1) % 50 == 0:
            log(f"prep {fi + 1}/{len(frames)} ({time.time() - t0:.0f}s)")
    log(f"arrays ready ({time.time() - t0:.0f}s)")

    # ---- ghost probabilities (production defaults from yaml) --------------
    q_weight = float(track_cfg.get("q_weight", 0.0))
    q_seed_raw = track_cfg.get("q_seed", DEFAULT_Q_SEED)
    q_seed = None if q_seed_raw is None else float(q_seed_raw)
    q_young = int(track_cfg.get("q_young", DEFAULT_Q_YOUNG))
    q_model = str(track_cfg.get("q_model", DEFAULT_Q_MODEL))
    scale = fit_scale(np.vstack(all_pos), np.concatenate(all_rcm),
                      np.concatenate(all_nseen))
    with np.errstate(all="ignore"):
        offsets = np.nanmedian(np.vstack(LB), axis=0)
        varies = float(np.nanstd(np.vstack(LB))) > 1e-6
    for fi in range(len(frames)):
        sp = (brightness_spread(LB[fi], offsets) if varies
              else np.full(len(P[fi]), np.nan))
        GH.append(ghost_probability(P[fi], all_rcm[fi], all_nseen[fi], scale, sp))
    new.write_point_quality_many(frames, [g.astype(np.float32) for g in GH])
    log(f"ghost ready ({time.time() - t0:.0f}s)")

    # ---- project_fn (reprojection through current cals) --------------------
    from openptv2.algorithms.imgcoord import img_coord_batch

    mm = cpar.mm
    imx, imy, pix_x, pix_y = 2560.0, 2048.0, 0.005, 0.005

    def project_fn(pred):
        pred = np.asarray(pred, dtype=np.float64)
        n = len(pred)
        xy = np.full((n, 8), np.nan)
        if n == 0:
            return xy
        for ci, cal in enumerate(cals):
            m = np.asarray(img_coord_batch(pred, cal, mm))
            xy[:, 2 * ci] = m[:, 0] / pix_x + imx / 2
            xy[:, 2 * ci + 1] = imy / 2 - m[:, 1] / pix_y
        return np.nan_to_num(xy)

    project_fn(np.zeros((1, 3)))

    cfg = TwoPhaseTrackerConfig(
        v_max=v_max,
        blob_gate=None,
        q_weight=q_weight,
        q_seed=q_seed,
        q_young=q_young,
        leaf_weight=float(track_cfg.get("leaf_weight", 1.0)),
        use_velocity=bool(track_cfg.get("use_velocity", True)),
        cost_mode=str(track_cfg.get("cost_mode", "projected")),
        max_gap=int(track_cfg.get("max_gap", 2)),
        allow_shared=bool(track_cfg.get("allow_shared", False)),
        max_shared=int(track_cfg.get("max_shared", 2)),
        share_tol=(None if track_cfg.get("share_tol", 1.0) is None
                   else float(track_cfg.get("share_tol", 1.0))),
        max_group_size=int(track_cfg.get("max_group_size", 128)),
        confirm_tol=confirm_tol,
        confirm_ends=confirm_ends,
        confirm_auto=confirm_auto,
        confirm_auto_max_neighbours=float(track_cfg.get(
            "confirm_auto_max_neighbours", AUTO_MAX_NEIGHBOURS)),
        confirm_auto_min_ratio=float(track_cfg.get(
            "confirm_auto_min_ratio", AUTO_MIN_KINK_RATIO)),
        confirm_auto_radius=float(track_cfg.get("confirm_auto_radius", 5.0)),
        bidirectional=bool(track_cfg.get("bidirectional", False)),
        parallel_directions=False,
        bwd_v_max=(None if track_cfg.get("bwd_v_max") is None
                   else float(track_cfg.get("bwd_v_max"))),
    )
    log(f"tracking {len(frames)} frames "
        f"(cost_mode={cfg.cost_mode} q={q_model}/{q_seed}/{q_young}) ...")
    t1 = time.time()
    links = TwoPhaseTracker(cfg).track_frames(
        P, LE, project_fn, frame_seen=SE, frame_ghost=GH, frame_logb=LB)
    log(f"tracked: {len(links)} links ({time.time() - t1:.0f}s)")
    per = _links_to_linkage(
        links, frames, [len(p) for p in P])
    new.write_linkage_many(
        frames, per, [np.asarray(p) for p in P])
    log(f"wrote {new_path} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
