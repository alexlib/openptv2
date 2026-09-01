"""Legacy ASCII <-> RunStore round-trip regression tests.

Guarantees that migrating datasets from ASCII (rt_is/ptv_is/added/_targets)
to Zarr stores loses nothing: ingesting a legacy run and exporting it back
reproduces the original file contents numerically, and tracking a store-fed
run reproduces the ASCII-fed result bit-for-bit.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.storage import RunStore
from openptv2.storage.legacy import _load_linkage, _load_rt_is, export_run, import_run

pytestmark = pytest.mark.ci

REPO = Path(__file__).parent.parent.parent
TRACK = REPO / "test_data" / "track"


@pytest.fixture
def staged_legacy_run(tmp_path):
    """Stage track's committed ASCII inputs as an importable legacy run."""
    run = tmp_path / "legacy_run"
    (run / "img").mkdir(parents=True)
    shutil.copytree(TRACK / "res_orig", run / "res")
    for p in (TRACK / "img_orig").glob("*_targets"):
        shutil.copy2(p, run / "img" / p.name)
    return run


def _read_dir(res_dir: Path, img_dir: Path):
    """Parse every ascii frame file of a legacy run dir into comparable arrays."""
    out = {"rt_is": {}, "ptv_is": {}, "added": {}, "targets": {}}
    for p in res_dir.glob("rt_is.*"):
        out["rt_is"][int(p.suffix.lstrip("."))] = _load_rt_is(p)
    for name in ("ptv_is", "added"):
        for p in res_dir.glob(f"{name}.*"):
            parsed = _load_linkage(p)
            if parsed is not None:
                prev, nxt, pos, prio = parsed
                out[name][int(p.suffix.lstrip("."))] = (prev, nxt, pos, prio)
    for p in img_dir.glob("*_targets"):
        m = re.search(r"cam(\d+)\.(\d+)_targets$", p.name)
        frame = int(m.group(2))
        cam = int(m.group(1)) - 1
        data = np.loadtxt(p, skiprows=1, ndmin=2)
        out["targets"][(cam, frame)] = data
    return out


def test_import_export_roundtrip_matches_original(staged_legacy_run, tmp_path):
    """import_run -> export_run must reproduce the original ASCII contents."""
    original = _read_dir(staged_legacy_run / "res", staged_legacy_run / "img")

    store = import_run(staged_legacy_run)
    exported = tmp_path / "exported"
    export_run(store, exported)

    rebuilt = _read_dir(exported / "res", exported / "img")

    assert set(rebuilt["rt_is"]) == set(original["rt_is"])
    for frame, (apos, aids) in original["rt_is"].items():
        bpos, bids = rebuilt["rt_is"][frame]
        assert np.array_equal(apos, bpos), f"rt_is pos mismatch frame {frame}"
        assert np.array_equal(aids, bids), f"rt_is ids mismatch frame {frame}"

    for name in ("ptv_is", "added"):
        assert set(rebuilt[name]) == set(original[name]), name
        for frame, (ap, an, apos, apr) in original[name].items():
            bp, bn, bpos, bpr = rebuilt[name][frame]
            assert np.array_equal(ap, bp), f"{name} prev mismatch frame {frame}"
            assert np.array_equal(an, bn), f"{name} next mismatch frame {frame}"
            assert np.allclose(apos, bpos), f"{name} pos mismatch frame {frame}"
            if apr is None or bpr is None:
                assert apr is None and bpr is None, (
                    f"{name} prio presence frame {frame}"
                )
            else:
                assert np.array_equal(apr, bpr), f"{name} prio mismatch frame {frame}"

    assert set(rebuilt["targets"]) == set(original["targets"])
    for key, adata in original["targets"].items():
        bdata = rebuilt["targets"][key]
        assert np.array_equal(adata, bdata), f"targets mismatch {key}"


def test_store_contents_match_ascii_inputs(staged_legacy_run):
    """Direct store reads must equal the ASCII source files bit-exactly."""
    original = _read_dir(staged_legacy_run / "res", staged_legacy_run / "img")
    store = import_run(staged_legacy_run)

    frames = store.frames()
    assert frames == sorted(original["rt_is"])

    for frame in frames:
        spos, sids = store.read_correspondences(frame)
        apos, aids = original["rt_is"][frame]
        assert np.array_equal(spos, apos) and np.array_equal(sids, aids)

        if store.has_linkage(frame, "ptv_is"):
            sp, sn, _ = store.read_linkage(frame, "ptv_is")
            ap, an, _, _ = original["ptv_is"][frame]
            assert np.array_equal(sp, ap) and np.array_equal(sn, an)


def test_converted_track_fixture_store_matches_res_orig():
    """The committed test_data/track/run.zarr fixture must equal res_orig.

    Guards against regenerating the fixture from a working-copy res/
    directory (the 2026-08-25 parity incident).
    """
    if not (TRACK / "run.zarr").exists():
        pytest.skip("track/run.zarr fixture not present")

    store = RunStore(TRACK / "run.zarr", mode="r")
    res = TRACK / "res_orig"

    for frame in store.frames():
        rt = res / f"rt_is.{frame}"
        if not rt.exists():
            continue
        apos, aids = _load_rt_is(rt)
        spos, sids = store.read_correspondences(frame)
        assert np.array_equal(apos, spos), f"fixture drift frame {frame} (pos)"
        assert np.array_equal(aids, sids), f"fixture drift frame {frame} (ids)"
