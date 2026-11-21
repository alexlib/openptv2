"""ASCII export of a RunStore (img/*_targets, res/rt_is.*, ptv_is.*, added.*).

Only used to feed the ORIGINAL C liboptv trackcorr (the ``optv`` package) as the legacy comparator in
``synth_bench.py``. The library itself is zarr-only; this was the old ``openptv2.storage.legacy.export_run``
(removed in 3e3809a2) kept here so the comparison with the legacy tracker still runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from openptv2.storage import RunStore


def _write_targets_ascii(path: Path, arr: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"{len(arr)}\n")
        for row in arr:
            f.write(
                f"{int(row[0]):4d} {row[1]:9.4f} {row[2]:9.4f} "
                f"{int(row[3]):5d} {int(row[4]):5d} {int(row[5]):5d} "
                f"{int(row[6]):5d} {int(row[7]):5d}\n"
            )


def _write_rt_is_ascii(path: Path, pos: np.ndarray, cam_ids: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        for i, (p, ids) in enumerate(zip(pos, cam_ids)):
            id_str = " ".join(f"{int(c):4d}" for c in ids)
            f.write(f"{i + 1:4d} {p[0]:9.3f} {p[1]:9.3f} {p[2]:9.3f} {id_str}\n")


def _write_linkage_ascii(
    path: Path,
    prev: np.ndarray,
    nxt: np.ndarray,
    pos: np.ndarray,
    prio: Optional[np.ndarray] = None,
) -> None:
    with open(path, "w") as f:
        f.write(f"{len(pos)}\n")
        if prio is None:
            for p_i, n_i, p in zip(prev, nxt, pos):
                f.write(
                    f"{int(p_i):4d} {int(n_i):4d} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f}\n"
                )
        else:
            for p_i, n_i, p, pr in zip(prev, nxt, pos, prio):
                f.write(
                    f"{int(p_i):4d} {int(n_i):4d} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f} {int(pr):d}\n"
                )


def export_run(store: RunStore, experiment_root: Union[str, Path]) -> None:
    """Regenerate ``img/*_targets``, ``res/rt_is.*``, ``res/ptv_is.*``,
    ``res/added.*`` from a RunStore, byte-compatible with the legacy writers.
    """
    root = Path(experiment_root)
    img_dir = root / "img"
    res_dir = root / "res"
    img_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    cams = store.target_cameras()
    frames = store.frames()

    for frame in frames:
        for cam in cams:
            targets = store.read_targets(cam, frame)
            arr = np.array(
                [
                    [t.pnr(), t.x(), t.y(), t.n, t.nx, t.ny, t.sumg, t.tnr()]
                    for t in targets
                ],
                dtype=np.float64,
            )
            _write_targets_ascii(img_dir / f"cam{cam + 1}.{frame}_targets", arr)

        if store.has_correspondences(frame):
            pos, cam_ids = store.read_correspondences(frame)
            _write_rt_is_ascii(res_dir / f"rt_is.{frame}", pos, cam_ids)

        if store.has_linkage(frame, "ptv_is"):
            prev, nxt, pos = store.read_linkage(frame, "ptv_is")
            # ptv_is.<frame> never carries a prio column in the legacy
            # format, even though the live pipeline's own "ptv_is" store
            # group does (write_path_frame embeds prio there directly,
            # since added.* is just that same data plus a prio column --
            # see write_linkage's docstring).
            _write_linkage_ascii(res_dir / f"ptv_is.{frame}", prev, nxt, pos, prio=None)

            if store.has_linkage(frame, "added"):
                a_prev, a_nxt, a_pos = store.read_linkage(frame, "added")
                a_prio = store.read_prio(frame, "added")
                _write_linkage_ascii(
                    res_dir / f"added.{frame}", a_prev, a_nxt, a_pos, a_prio
                )
            else:
                prio = store.read_prio(frame, "ptv_is")
                if prio is not None:
                    _write_linkage_ascii(
                        res_dir / f"added.{frame}", prev, nxt, pos, prio
                    )



def read_ascii_trajectories(res_dir: Union[str, Path], first: int, last: int):
    """Trajectories from legacy ``ptv_is.<frame>`` files (``prev next x y z`` rows, negative = no link).

    Returns ``{track id: [(frame, x, y, z), ...]}`` like ``openptv2.benchmarking.read_trajectories``."""
    res_dir = Path(res_dir)
    frames = []
    for f in range(first, last + 1):
        p = res_dir / f"ptv_is.{f}"
        if not p.exists():
            frames.append(None)
            continue
        rows = [ln.split() for ln in p.read_text().splitlines()[1:] if ln.strip()]
        frames.append({
            "prev": [int(r[0]) for r in rows],
            "next": [int(r[1]) for r in rows],
            "xyz": [tuple(float(v) for v in r[2:5]) for r in rows],
        })
    tracks: dict[int, list] = {}
    for fi, fr in enumerate(frames):
        if fr is None:
            continue
        for slot in range(len(fr["next"])):
            if fr["prev"][slot] >= 0:
                continue
            pts = [(first + fi, *fr["xyz"][slot])]
            cf, cs = fi, slot
            while True:
                nx = frames[cf]["next"][cs]
                if nx < 0 or cf + 1 >= len(frames) or frames[cf + 1] is None:
                    break
                cf, cs = cf + 1, nx
                pts.append((first + cf, *frames[cf]["xyz"][cs]))
            tracks[len(tracks)] = pts
    return tracks
