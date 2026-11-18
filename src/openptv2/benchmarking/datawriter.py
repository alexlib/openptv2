"""Write ground-truth tracking datasets into a Zarr RunStore.

Given a :class:`~openptv2.benchmarking.camera_rig.CameraRig` and per-frame
ground truth (from :func:`~openptv2.benchmarking.scenario.generate_scenario`),
projects the 3D particles to per-camera pixels through the calibration +
multimedia model and writes targets, correspondences, and initial linkage
into a :class:`~openptv2.storage.RunStore` for downstream stage-by-stage
validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def write_dataset_store(
    rig: CameraRig,
    frame_gt: dict[int, list[tuple[int, float, float, float]]],
    spec: DatasetSpec,
    store: Any = None,
) -> Any:
    """Write a full ground-truth dataset into a RunStore (zarr is the database of record).

    * targets: ``pnr`` = y-sorted index, ``tnr`` = particle slot (or
      ``TR_UNUSED`` for ghosts),
    * correspondences: identity ``p[]`` = y-sorted target index per camera,
    * linkage ``ptv_is`` + ``added``: unlinked (prev=-1, next=-2, prio=4).

    Args:
        rig: camera setup.
        frame_gt: per-frame ground truth.
        spec: output location/naming (``res_sub``/``img_sub`` ignored when
            ``store`` is None).
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


__all__ = ["DatasetSpec", "write_dataset_store"]
