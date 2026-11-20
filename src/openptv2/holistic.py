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
    the output is the last linkage.

Pure library code; :mod:`openptv2.plugins.holistic_tracking` runs it in the pipeline.
"""

from __future__ import annotations

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
        arr = arrays_from_store(store, linkage_name, first, last, n_cams)
        blob, centre, sumg, n_att = extend_membership(arr["frame"], arr["pos"], arr["blob"], arr["centre"],
                                                      arr["sumg"], arr["blobs"], cpar, cals, radius=radius)
        sig = estimate_signatures(arr["tid"], arr["frame"], blob, sumg)
        res = refine_merged(arr["frame"], arr["tid"], blob, centre, sumg, cpar, cals, vpar, signatures=sig,
                            partner_min_len=partner_min_len)
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
        new_pos = np.where(np.isfinite(res.x[:, :1]), res.x, arr["pos"])
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
        log(f"holistic round {rnd}: {info['births']} births, {info['attachments']} attachments")
        relink()
        store = fresh(store)
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
    return stats


def store_blob_centres(arr: dict, frame: int, cam: int) -> np.ndarray:
    """Blob centres (px) of ``frame`` and ``cam`` from :func:`openptv2.merge_post.arrays_from_store` output."""
    return arr["blobs"][(int(frame), cam)][0]
