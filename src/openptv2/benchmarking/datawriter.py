"""Write ground-truth tracking datasets in openptv2's on-disk format.

Given a :class:`~openptv2.benchmarking.camera_rig.CameraRig` and per-frame
ground truth (from :func:`~openptv2.benchmarking.scenario.generate_scenario`),
projects the 3D particles to per-camera pixels through the calibration +
multimedia model and writes:

  * ``rt_is.#``       — 3D correspondences (identity ``p[]`` = particle id)
  * ``ptv_is.#``      — linkage output (initially unlinked)
  * ``added.#``       — priority file
  * ``camN.FFFFF_targets`` — per-camera 2D pixel targets (y-sorted, ``tnr``=id)
  * ``origin_FFFFF.txt``  — proPTV-style ground truth (ID, XYZ, per-cam coords)

for downstream stage-by-stage validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openptv2.algorithms.constants import TR_UNUSED
from openptv2.benchmarking.camera_rig import CameraRig, project_to_pixels
from openptv2.tracking_framebuf import TargetArray


@dataclass
class DatasetSpec:
    """Location and naming of a generated dataset."""

    dir: Path
    res_sub: str = "res"
    img_sub: str = "img"
    first_frame: int = 10001
    num_cams: int = 4


def write_dataset(
    rig: CameraRig,
    frame_gt: dict[int, list[tuple[int, float, float, float]]],
    spec: DatasetSpec,
    max_targets: int = 30000,
) -> None:
    """Write a full ground-truth dataset to disk.

    Parameters
    ----------
    rig : CameraRig
        Camera setup (projection + multimedia).
    frame_gt : dict[int, list[(pid, x, y, z)]]
        Per-frame ground truth, one entry per visible particle (ghosts use
        pid == -1 and are written as targets with ``tnr = PT_UNUSED``).
    spec : DatasetSpec
        Output location / naming.
    max_targets : int
        Capacity for target arrays.
    """
    res_dir = spec.dir / spec.res_sub
    img_dir = spec.dir / spec.img_sub
    res_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    for rel_f, particles in sorted(frame_gt.items()):
        fnum = spec.first_frame + rel_f
        n = len(particles)

        # Project all particles to all cameras, then sort each camera's
        # targets by pixel-y (candsearch_in_pix does a binary search on y).
        # Keep slot -> target-index mapping for the rt_is correspondence.
        cam_slot_to_targ: list[dict[int, int]] = []
        cam_targ_entries: list[list[tuple[int, float, float]]] = []
        # per-cam arrays of pixel coords
        pos3d = np.array([(x, y, z) for _, x, y, z in particles])

        px = project_to_pixels(rig, pos3d)

        for cam in range(spec.num_cams):
            entries = []
            for slot in range(n):
                entries.append((slot, float(px[cam][slot, 0]), float(px[cam][slot, 1])))
            entries.sort(key=lambda t: t[2])  # sort by pixel-y
            cam_targ_entries.append(entries)
            s2t = {slot: idx for idx, (slot, _, _) in enumerate(entries)}
            cam_slot_to_targ.append(s2t)

        # --- rt_is (correspondence) ---
        with open(res_dir / f"rt_is.{fnum}", "w") as fh:
            fh.write(f"{n}\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                cam_indices = " ".join(
                    f"{cam_slot_to_targ[cam][slot]:4d}" for cam in range(spec.num_cams)
                )
                fh.write(f"{slot + 1:4d} {x:9.3f} {y:9.3f} {z:9.3f} {cam_indices}\n")

        # --- ptv_is (linkage — initially unlinked) ---
        with open(res_dir / f"ptv_is.{fnum}", "w") as fh:
            fh.write(f"{n}\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                fh.write(f"  -1   -2 {x:10.3f} {y:10.3f} {z:10.3f}\n")

        # --- added (prio — initially unlinked) ---
        with open(res_dir / f"added.{fnum}", "w") as fh:
            fh.write(f"{n}\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                fh.write(f"  -1   -2 {x:10.3f} {y:10.3f} {z:10.3f} 4\n")

        # --- per-camera target files (y-sorted).
        # Naming matches the reader's `_resolve_file_base` when given a base
        # with a `%d` placeholder (e.g. `img/cam1.%d`), which resolves to
        # `img/cam1.<frame>_targets` — the convention used by real datasets.
        for cam in range(spec.num_cams):
            entries = cam_targ_entries[cam]
            with open(img_dir / f"cam{cam + 1}.{fnum}_targets", "w") as fh:
                fh.write(f"{len(entries)}\n")
                for targ_pnr, (slot, pxv, pyv) in enumerate(entries):
                    pid = particles[slot][0]
                    # tnr is consumed directly as an index into that frame's
                    # path_x/corres array (see track_kernels_search.py's
                    # ftnr_out[...] = targ_tnr[cam, idx], then
                    # path_x_2[ftnr_i] in track_kernels_corr.py) -- it must
                    # be the particle's row index within THIS frame's
                    # rt_is/particles list (== slot), not its ground-truth
                    # pid. The two coincide only while every frame holds a
                    # dense 0..n-1 pid range; entering/leaving particles
                    # break that, which silently starved trackcorr/
                    # full_multipass of nearly all links on every dataset
                    # this writer produced with entry/exit turbulence.
                    tnr = slot if pid >= 0 else TR_UNUSED
                    fh.write(
                        f"{targ_pnr:4d} {pxv:9.4f} {pyv:9.4f} "
                        f"  100    10    10  1000 {tnr:5d}\n"
                    )

        # --- origin (proPTV-style ground truth) ---
        with open(res_dir / f"origin_{fnum}.txt", "w") as fh:
            fh.write("ID,X,Y,Z,xc0,yc0,xc1,yc1,xc2,yc2,xc3,yc3\n")
            for slot, (pid, x, y, z) in enumerate(particles):
                coords = []
                for cam in range(spec.num_cams):
                    # cam_slot_to_targ already maps slot -> its index in
                    # cam_targ_entries[cam]; index directly instead of
                    # rescanning the whole (sorted) entries list per slot.
                    _, ex, ey = cam_targ_entries[cam][cam_slot_to_targ[cam][slot]]
                    coords.append(f"{ex:.4f}")
                    coords.append(f"{ey:.4f}")
                fh.write(f"{pid},{x:.6f},{y:.6f},{z:.6f},{','.join(coords)}\n")


def write_dataset_store(
    rig: CameraRig,
    frame_gt: dict[int, list[tuple[int, float, float, float]]],
    spec: DatasetSpec,
    store=None,
):
    """Write the same ground-truth dataset as :func:`write_dataset`, but into
    a RunStore (zarr is the database of record). The per-frame content is
    identical to the ASCII writer's output, parsed:

    * targets: ``pnr`` = y-sorted index, ``tnr`` = particle slot (or
      ``TR_UNUSED`` for ghosts),
    * correspondences: identity ``p[]`` = y-sorted target index per camera,
    * linkage ``ptv_is`` + ``added``: unlinked (prev=-1, next=-2, prio=4).

    Args:
        rig: camera setup.
        frame_gt: per-frame ground truth, same format as :func:`write_dataset`.
        spec: output location/naming (``res_sub``/``img_sub`` ignored).
        store: optional pre-opened RunStore; default opens
            ``<spec.dir>/run.zarr`` in append mode.

    Returns:
        The RunStore written to.
    """
    if store is None:
        from openptv2.storage import RunStore

        store = RunStore(spec.dir / "run.zarr", mode="a")

    for rel_f, particles in sorted(frame_gt.items()):
        fnum = spec.first_frame + rel_f
        n = len(particles)
        pos3d = np.array([(x, y, z) for _, x, y, z in particles])
        px = project_to_pixels(rig, pos3d)

        # y-sorted per-camera target order, mirroring write_dataset exactly
        cam_entries = []
        for cam in range(spec.num_cams):
            entries = [
                (slot, float(px[cam][slot, 0]), float(px[cam][slot, 1]))
                for slot in range(n)
            ]
            entries.sort(key=lambda t: t[2])
            cam_entries.append(entries)

        for cam in range(spec.num_cams):
            tarr = TargetArray(len(cam_entries[cam]))
            for targ_pnr, (slot, pxv, pyv) in enumerate(cam_entries[cam]):
                pid = particles[slot][0]
                t = tarr[targ_pnr]
                t.set_pnr(targ_pnr)
                t.set_pos((pxv, pyv))
                t.set_pixel_counts(100, 10, 10)
                t.set_sum_grey_value(1000)
                t.set_tnr(slot if pid >= 0 else TR_UNUSED)
            store.write_targets(cam, fnum, tarr)

        pos = np.array([(x, y, z) for _, x, y, z in particles], dtype=np.float64)
        cam_ids = np.zeros((n, spec.num_cams), dtype=np.int32)
        for cam in range(spec.num_cams):
            s2t = {slot: idx for idx, (slot, _, _) in enumerate(cam_entries[cam])}
            for slot in range(n):
                cam_ids[slot, cam] = s2t[slot]
        store.write_correspondences(fnum, pos, cam_ids)

        prev = np.full(n, -1, dtype=np.int32)
        nxt = np.full(n, -2, dtype=np.int32)
        store.write_linkage(fnum, prev, nxt, pos, name="ptv_is")
        store.write_linkage(
            fnum, prev, nxt, pos, name="added", prio=np.full(n, 4, dtype=np.int32)
        )

    return store


__all__ = ["DatasetSpec", "write_dataset", "write_dataset_store"]
