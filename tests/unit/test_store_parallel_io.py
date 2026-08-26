"""Parallel RunStore I/O regression tests.

Zarr-backed runs are read and written concurrently (parallel tracking
stages, chunked workers, GUI viewers during batch runs). These tests pin
the concurrency contract:

- concurrent writers to distinct frames/cameras must all land bit-exact;
- a reader must never observe a torn frame (a frame is either absent or
  complete).

Runtime budget: a few seconds.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from openptv2.storage import RunStore
from openptv2.tracking_framebuf import TargetArray

pytestmark = pytest.mark.ci

N_FRAMES = 24
N_WORKERS = 8


def _make_targets(cam: int, frame: int, n: int) -> TargetArray:
    tarr = TargetArray(n)
    for i in range(n):
        t = tarr[i]
        t.set_pnr(i)
        t.set_pos((float(frame) + i * 0.25, float(cam) + i * 0.5))
        t.set_pixel_counts(3, 3, 3)
        t.set_sum_grey_value(100 + i)
        t.set_tnr(i)
    return tarr


def _expected_corres(frame: int, n_parts: int):
    pos = np.full((n_parts, 3), float(frame), dtype=np.float64)
    ids = np.full((n_parts, 4), frame % 100, dtype=np.int32)
    return pos, ids


def test_concurrent_writers_distinct_frames(tmp_path):
    """N workers writing disjoint frames must all land, bit-exact."""
    store = RunStore(tmp_path / "run.zarr", mode="a")

    def write_frame(fn):
        n_parts = 5 + (fn % 4)
        for cam in range(4):
            store.write_targets(cam, fn, _make_targets(cam, fn, 2 + fn % 3))
        pos, ids = _expected_corres(fn, n_parts)
        store.write_correspondences(fn, pos, ids)
        prev = np.full(n_parts, -1, dtype=np.int32)
        nxt = np.arange(n_parts, dtype=np.int32) % max(1, n_parts - 1)
        store.write_linkage(fn, prev, nxt, pos, name="ptv_is")
        return fn

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        written = set(ex.map(write_frame, range(N_FRAMES)))

    assert written == set(range(N_FRAMES))

    verify = RunStore(tmp_path / "run.zarr", mode="r")
    assert verify.frames() == list(range(N_FRAMES))
    for fn in range(N_FRAMES):
        n_parts = 5 + (fn % 4)
        for cam in range(4):
            tg = verify.read_targets(cam, fn)
            assert len(tg) == 2 + fn % 3
            assert int(tg[0].sumg) == 100
        pos, ids = verify.read_correspondences(fn)
        epos, eids = _expected_corres(fn, n_parts)
        assert np.array_equal(pos, epos) and np.array_equal(ids, eids)


def test_reader_never_sees_torn_frames(tmp_path):
    """Frames appearing mid-write must be read complete or not at all."""
    store = RunStore(tmp_path / "run.zarr", mode="a")
    n_parts = 50
    errors = []

    def write_chunk(start):
        for fn in range(start, N_FRAMES, N_WORKERS):
            pos, ids = _expected_corres(fn, n_parts)
            store.write_targets(0, fn, _make_targets(0, fn, 4))
            store.write_correspondences(fn, pos, ids)

    def read_until_done():
        seen_complete = 0
        while True:
            frames = store.frames()
            if len(frames) == N_FRAMES and all(
                store.has_correspondences(f) for f in range(N_FRAMES)
            ):
                break
            for fn in frames:
                try:
                    pos, ids = store.read_correspondences(fn)
                except Exception as exc:  # noqa: BLE001 - record any race symptom
                    errors.append(f"frame {fn}: {exc!r}")
                    continue
                if len(pos) != n_parts or len(ids) != n_parts:
                    errors.append(f"torn frame {fn}: {len(pos)} rows")
                else:
                    seen_complete += 1
        return seen_complete

    with ThreadPoolExecutor(max_workers=N_WORKERS + 1) as ex:
        writer_futs = [ex.submit(write_chunk, w) for w in range(N_WORKERS)]
        reader_fut = ex.submit(read_until_done)
        for f in writer_futs:
            f.result()
        reader_fut.result()

    assert not errors, errors[:10]


def test_parallel_reads_of_static_store(tmp_path):
    """Concurrent readers of an immutable store see identical data."""
    store = RunStore(tmp_path / "run.zarr", mode="a")
    for fn in range(10):
        pos, ids = _expected_corres(fn, 7)
        store.write_correspondences(fn, pos, ids)

    verify = RunStore(tmp_path / "run.zarr", mode="r")

    def read_all(_):
        out = []
        for fn in range(10):
            pos, ids = verify.read_correspondences(fn)
            out.append(np.array_equal(pos, _expected_corres(fn, 7)[0]))
        return out

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(read_all, range(N_WORKERS * 4)))

    flat = [ok for chunk in results for ok in chunk]
    assert all(flat) and len(flat) == N_WORKERS * 4 * 10
