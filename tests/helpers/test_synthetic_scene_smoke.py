from pathlib import Path

import pytest

from openptv2.storage import RunStore
from tests.helpers.synthetic_scene import make_cavity_scene

pytestmark = pytest.mark.ci


def test_synthetic_scene_smoke(tmp_path: Path):
    scene = make_cavity_scene(tmp_path, n_frames=4, n_particles=10, seed=0)
    store_path = scene / "res" / "run.zarr"
    assert store_path.exists()
    store = RunStore(store_path, mode="r")
    assert len(store.frames()) == 4
    # each frame should have some targets
    for f in [10001, 10002, 10003, 10004]:
        assert store.has_targets(0, f)
        assert store.has_correspondences(f)


def test_synthetic_scene_gaps_and_noise(tmp_path: Path):
    scene = make_cavity_scene(
        tmp_path, n_frames=12, n_particles=20, gap_prob=0.15, pixel_noise=0.5, seed=1
    )
    store = RunStore(scene / "res" / "run.zarr", mode="r")
    assert len(store.frames()) == 12
    # gaps should cause some frames to have fewer particles than n_particles
    has_gap = False
    for f in range(12):
        arr = store.read_targets(0, 10001 + f)
        try:
            n = len(arr)
        except Exception:
            n = 0
        if n < 20:
            has_gap = True
            break
    assert has_gap
