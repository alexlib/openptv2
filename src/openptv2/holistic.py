"""The holistic loop: stereo matching and tracking resolve each other's ambiguities
(docs/plans/2026-11-20-holistic-ptv.md).

Starting from a run store with targets and correspondences, and a linker (by default ``two_phase``):

    link
    repeat ``rounds`` times:
        shared blobs from the tracks (a point without a view in a camera joins the blob its projection falls on)
        brightness signatures per track and camera (joint with blob counts, cross-camera check)
        refine positions (effective observations of shared blobs, trusted path prior, hidden-partner check,
                          young tracks do not lend predictions to partners)
        births: particles hidden in other particles' blobs, from the residual brightness, matched in 3D
        write the refined points + the births back as the point cloud (and the observation table)
        link again
    the output is the last linkage; the final tracks are also written to ``trajectories/`` (see
    :func:`write_trajectories_cache`).

Pure library code; :mod:`openptv2.plugins.holistic_tracking` runs it in the pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np


def run_holistic(
    store: Any,
    frames: list[int],
    cpar: Any,
    cals: list,
    vpar: Any,
    relink: Callable[[], Any],
    *,
    rounds: int = 2,
    linkage_name: str = "ptv_is",
    min_resid: float = 0.25,
    partner_min_len: int = 10,
    radius: float = 2.0,
    initial_link: bool = True,
    birth_sigma_factor: float = 3.0,
    drop_unlinked_births: bool = True,
    use_positions: bool = True,
    use_births: bool = True,
    refine_rounds: int = 3,
    log: Callable[[str], Any] = print,
) -> dict:
    """Run the loop on ``store`` over the contiguous ``frames``; ``relink()`` links the store's current points.

    Returns per-round statistics (attachments, births, refinement info).
    """
    from openptv2.loco import project_pixels
    from openptv2.merge_post import (
        arrays_from_store,
        birth_points,
        birth_targets,
        estimate_signatures,
        extend_membership,
        stamp_tnr,
    )
    from openptv2.merge_refine import refine_merged, shared_blobs

    frames = list(frames)
    first, last = frames[0], frames[-1]
    n_cams = len(cals)
    stats: dict = {"rounds": []}

    def fresh(st):
        # the linker may write through another store object; linkage blocks are cached per object, so read the
        # store again after every relink (a stale linkage next to fresh correspondences breaks row alignment)
        path = getattr(st, "store_path", None)
        if path is None:
            return st
        from openptv2.storage import RunStore

        return RunStore(path, mode="a")

    if initial_link:
        relink()
    store = fresh(store)
    n_orig = {f: len(store.read_correspondences(f)[0]) for f in frames}
    for rnd in range(1, rounds + 1):
        t_round = time.perf_counter()
        arr = arrays_from_store(store, linkage_name, first, last, n_cams)
        blob, centre, sumg, n_att = extend_membership(arr["frame"], arr["pos"], arr["blob"], arr["centre"],
                                                      arr["sumg"], arr["blobs"], cpar, cals, radius=radius)
        sig = estimate_signatures(arr["tid"], arr["frame"], blob, sumg)
        t_refine = time.perf_counter()
        res = refine_merged(arr["frame"], arr["tid"], blob, centre, sumg, cpar, cals, vpar, signatures=sig, rounds=refine_rounds,
                            partner_min_len=partner_min_len)
        t_births = time.perf_counter()
        counts = {(f, c): np.asarray(store.root[f"targets/cam_{c}/frame_{f:06d}"])[:, 3:6].astype(int)
                  for f in frames for c in range(n_cams)}
        fallback = np.full(res.pred_px.shape, np.nan)
        okx = np.isfinite(res.x[:, 0])
        for c in range(n_cams):
            m = okx & (blob[:, c] >= 0)
            if m.any():
                fallback[m, c] = project_pixels(res.x[m], cals[c], cpar)
        targets = birth_targets(arr["frame"], blob, res.pred_px, sig, arr["blobs"], counts, min_resid=min_resid,
                                fallback_px=fallback)
        births = birth_points(targets, frames, cpar, cals, vpar)
        if not use_births:
            births = {k: v[:0] for k, v in births.items()}
        t_write = time.perf_counter()
        new_pos = np.where(np.isfinite(res.x[:, :1]) & use_positions, res.x, arr["pos"])
        sd = np.sqrt(res.pos_var) if res.pos_var is not None else np.full(len(new_pos), np.nan)
        sd_med = float(np.nanmedian(sd)) if np.isfinite(sd).any() else 0.05
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, sd_med)
        off = 0
        for f in frames:
            pos0, ids0 = store.read_correspondences(f)
            n = len(pos0)
            if not (arr["frame"][off: off + n] == f).all():
                raise RuntimeError("row alignment between correspondences and linkage lost")
            seg = new_pos[off: off + n]
            b = births["frame"] == f
            ids0 = np.asarray(ids0, dtype=np.int64)
            ids_b = np.full((int(b.sum()), ids0.shape[1]), -1, dtype=np.int64)
            ids_b[:, :n_cams] = births["blob"][b]
            all_ids = np.vstack([ids0, ids_b])
            store.write_correspondences(f, np.vstack([seg, births["pos"][b]]), all_ids)
            # observation table: what each point used, how certain, how many particles share the blob
            obs = np.zeros((len(all_ids), n_cams, 4), np.float32)
            obs[:, :, :2] = np.nan
            obs[:, :, 2] = np.nan
            obs[:n, :, :2] = res.obs_px[off: off + n]
            obs[:n, :, 2] = res.obs_var[off: off + n]
            cent = [store_blob_centres(arr, f, c) for c in range(n_cams)]
            for c in range(n_cams):
                for k in np.flatnonzero(all_ids[n:, c] >= 0):
                    obs[n + k, c, :2] = cent[c][all_ids[n + k, c]]
                    obs[n + k, c, 2] = 0.01
            share = shared_blobs(np.full(len(all_ids), f), all_ids[:, :n_cams])
            obs[:, :, 3] = share
            store.write_observations(f, obs)
            # position uncertainty per point: refined points from their fit, births 3x the median (their placement
            # error was measured about 4x the established points')
            store.write_point_sigma(f, np.concatenate([sd[off: off + n], np.full(len(all_ids) - n, birth_sigma_factor * sd_med)]))
            off += n
        stamp_tnr(store, frames, n_cams)
        info = {"round": rnd, "rows": int(len(arr["frame"])), "attachments": int(n_att),
                "births": int(len(births["frame"])), "refine": res.info[-1] if res.info else {}}
        stats["rounds"].append(info)
        t_link = time.perf_counter()
        relink()
        store = fresh(store)
        log(f"holistic round {rnd}: {info['births']} births, {info['attachments']} attachments; seconds: "
            f"membership+signatures {t_refine - t_round:.1f}, refine {t_births - t_refine:.1f}, "
            f"births {t_write - t_births:.1f}, write {t_link - t_write:.1f}, relink {time.perf_counter() - t_link:.1f}")
    if drop_unlinked_births and rounds > 0:
        # births that no track took are about half wrong (benchmark: 53 % against 96 % for linked births): drop them
        from openptv2.tracking_postprocess import read_linkage

        n_drop = 0
        for f in frames:
            pos, ids = store.read_correspondences(f)
            lk = read_linkage(linkage_name, f, store=store)
            if lk is None:
                continue
            prev, nxt, _ = lk
            rows = np.arange(len(pos))
            drop = (rows >= n_orig[f]) & (np.asarray(prev) < 0) & (np.asarray(nxt) < 0)
            if not drop.any():
                continue
            keep = ~drop
            n_drop += int(drop.sum())
            store.write_correspondences(f, pos[keep], np.asarray(ids)[keep])
            obs = store.read_observations(f)
            if obs is not None and len(obs) == len(keep):
                store.write_observations(f, obs[keep])
            sd = store.read_point_sigma(f)
            if sd is not None and len(sd) == len(keep):
                store.write_point_sigma(f, sd[keep])
        stats["dropped_unlinked_births"] = n_drop
        if n_drop:
            stamp_tnr(store, frames, n_cams)
            log(f"holistic: dropped {n_drop} unlinked births, linking again")
            relink()
            store = fresh(store)
    write_trajectories_cache(fresh(store), frames, n_cams, linkage_name)
    return stats


def write_trajectories_cache(store: Any, frames: list[int], n_cams: int, linkage_name: str = "ptv_is",
                             min_len: int = 2) -> int:
    """Write the final tracks into ``trajectories/`` (the flat cache that openptv-cloud's ``03_results`` notebook and
    flowtracks read): positions in metres, velocity and acceleration per frame (central differences), sorted by
    ``(trajid, time)``, tracks of at least ``min_len`` points. Returns the number of tracks."""
    from openptv2.merge_post import arrays_from_store

    a = arrays_from_store(store, linkage_name, int(min(frames)), int(max(frames)), n_cams)
    tid, fr, pos = np.asarray(a["tid"]), np.asarray(a["frame"]), np.asarray(a["pos"], float)
    _, inv, cnt = np.unique(tid, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_len
    tid, fr, pos = tid[keep], fr[keep], pos[keep]
    order = np.lexsort((fr, tid))
    tid, fr, pos = tid[order], fr[order], pos[order]
    _, tid = np.unique(tid, return_inverse=True)
    tid = tid + 1
    vel, acc = np.zeros_like(pos), np.zeros_like(pos)
    cuts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True])
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo >= 2:
            vel[lo:hi] = np.gradient(pos[lo:hi], fr[lo:hi], axis=0)
        if hi - lo >= 3:
            acc[lo:hi] = np.gradient(vel[lo:hi], fr[lo:hi], axis=0)
    store.write_trajectories(pos * 1e-3, vel * 1e-3, acc * 1e-3, fr, tid)
    return int(tid.max()) if len(tid) else 0


def store_blob_centres(arr: dict, frame: int, cam: int) -> np.ndarray:
    """Blob centres (px) of ``frame`` and ``cam`` from :func:`openptv2.merge_post.arrays_from_store` output."""
    return arr["blobs"][(int(frame), cam)][0]


def run_holistic_experiment(exp: Any, store: Any, frames: list[int], track_cfg: dict, rounds: int) -> None:
    """Run the holistic loop for an experiment object, linking with ``two_phase``.

    Called by the ``two_phase`` plugin when ``track.holistic_rounds`` > 0 (YAML keys: ``holistic_rounds``,
    ``holistic_min_resid``, ``holistic_partner_min_len``, ``holistic_extend``; experiment switches ``holistic_positions``,
    ``holistic_births``, ``holistic_refine_rounds``: see the plan, section 13).
    """
    from openptv2.plugins.two_phase_tracking import track_store

    cals, cpar, vpar = getattr(exp, "cals", None), getattr(exp, "cpar", None), getattr(exp, "vpar", None)

    def relink():
        return track_store(store, frames, track_cfg, cals, cpar)

    run_holistic(store, frames, cpar, list(cals), vpar, relink, rounds=rounds,
                 min_resid=float(track_cfg.get("holistic_min_resid", 0.25)),
                 partner_min_len=int(track_cfg.get("holistic_partner_min_len", 10)),
                 use_positions=bool(track_cfg.get("holistic_positions", True)),
                 use_births=bool(track_cfg.get("holistic_births", True)),
                 refine_rounds=int(track_cfg.get("holistic_refine_rounds", 3)))
    passes = int(track_cfg.get("holistic_extend", 0))
    if passes > 0:                                        # strict track-assisted extension on the final tracks
        from openptv2.storage import RunStore
        from openptv2.track_assisted import run_corrective_pass

        store = RunStore(store.store_path, mode="a")          # fresh object: no cached linkage
        spar = exp.spar
        spar.first, spar.last = frames[0], frames[-1]
        run_corrective_pass(cpar, vpar, exp.track_par, spar, list(cals), store, max_passes=passes,
                            forward=True, release_ghosts=False, radius_px=1.2, max_ray_miss_mm=0.15,
                            max_pos_err_mm=0.3)
        write_trajectories_cache(RunStore(store.store_path, mode="a"), frames, len(cals))
