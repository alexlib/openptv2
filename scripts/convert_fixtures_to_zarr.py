"""Convert test_data fixtures with ASCII run inputs into Zarr run stores.

For every dataset under ``test_data/`` that has run inputs (``res_orig/rt_is.*``,
``img/*_targets`` or ``img_orig/*_targets``) but no store yet, this ingests
into ``<dataset>/run.zarr`` -- the fixture location that survives test copies
excluding ``res*`` directories.

Inputs are taken from ``res_orig`` ONLY (never a working-copy ``res/``, which
may hold a previous tracking pass output -- see the 2026-08-25 parity
incident in docs/plans/2026-08-25-ci-red-and-zarr-parity-plan.md). Datasets
without res_orig get targets-only stores from their img target files.

Usage:
    uv run python scripts/convert_fixtures_to_zarr.py            # convert all missing
    uv run python scripts/convert_fixtures_to_zarr.py track burgers   # specific ones
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_DATA = REPO / "test_data"

from openptv2.algorithms.tracking_frame_buf import read_targets as read_targets_ascii
from openptv2.storage import RunStore, find_existing_store
from openptv2.storage.legacy import _load_linkage, _load_rt_is


def ingest_dataset(root: Path) -> str:
    """Ingest one dataset's ASCII inputs into <root>/run.zarr. Returns status."""
    store_path = root / "run.zarr"
    if find_existing_store(root) is not None:
        return "already-has-store"

    res_dir = root / "res_orig"
    img_dirs = [d for d in ("img", "img_orig") if (root / d).is_dir()]
    if not res_dir.is_dir() and not img_dirs:
        return "no-inputs"

    store = RunStore(store_path, mode="a")

    # correspondences + linkage (+prio) from res_orig only
    frames = []
    if res_dir.is_dir():
        frames = sorted(
            int(p.name.split(".")[-1])
            for p in res_dir.glob("rt_is.*")
            if p.name.split(".")[-1].isdigit()
        )
        for frame in frames:
            pos, cam_ids = _load_rt_is(res_dir / f"rt_is.{frame}")
            store.write_correspondences(frame, pos, cam_ids)
            for name in ("ptv_is", "added"):
                parsed = _load_linkage(res_dir / f"{name}.{frame}")
                if parsed is not None:
                    prev, nxt, lpos, prio = parsed
                    store.write_linkage(frame, prev, nxt, lpos, name=name, prio=prio)

    # targets: prefer img/, fall back to img_orig/ per camera+frame
    import re

    target_re = re.compile(r"cam(\d+)\.(\d+)_targets$")
    seen = {}
    for d in img_dirs:
        for p in (root / d).glob("cam*_targets"):
            m = target_re.search(p.name)
            if not m:
                continue
            cam, frame = int(m.group(1)) - 1, int(m.group(2))
            seen.setdefault((cam, frame), p)
    for (cam, frame), path in sorted(seen.items()):
        targs = read_targets_ascii(str(path.parent / path.name.split(".")[0]) + ".", frame, cam_idx=cam)
        store.write_targets(cam, frame, targs)
        if frame not in frames:
            frames.append(frame)

    n_frames = len(store.frames())
    print(f"{root.name}: wrote {n_frames} frames -> {store_path}")
    return "converted"


def main(argv):
    names = argv[1:]
    roots = (
        [TEST_DATA / n for n in names]
        if names
        else sorted(d for d in TEST_DATA.iterdir() if d.is_dir())
    )
    counts = {}
    for root in roots:
        if not root.is_dir():
            print(f"skip (missing): {root}")
            continue
        status = ingest_dataset(root)
        counts[status] = counts.get(status, 0) + 1
        if status == "converted":
            continue
        print(f"{root.name}: {status}")
    print("\nsummary:", counts)


if __name__ == "__main__":
    main(sys.argv)
