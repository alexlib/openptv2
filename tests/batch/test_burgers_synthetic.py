"""
Synthetic-image round-trip tests for the burgers dataset.

The test_data/burgers/img_orig folder contains synthetic grayscale TIFFs
(`camN.NNNNN`) rendered from the ground-truth `camN.NNNNN_targets` files (each
tracer drawn as a Gaussian PSF centered at the stored centroid, offset by 0.5px
to match the ``targ_rec`` +0.5 centroid convention).

These tests close the ground-truth -> image -> detection -> tracking loop:

1. Copy the synthetic images from img_orig into img/ (the folder the sequence
   pipeline reads from, per the dataset YAML `img/cam%d`).
2. Run the detection (sequence) step against the images.
3. Compare the **detected** targets (img/*_targets) against the **ground-truth**
   targets (img_orig/*_targets), asserting the recovered sub-pixel centroid
   error stays below a strict threshold.
4. Compare reconstructed 3D positions (res/rt_is.*) against the committed
   ground truth (res_orig/rt_is.*), with the caveat that the match is not
   always perfect (see below).
5. Exercise the image-space "add particle" scenario: the committed res_orig
   ground truth drops the 5th (fast) tracer from 3D in frame 10003 while all
   4 cameras still see it in 2D. Feed that gap as correspondence input and
   check whether each tracker can recover the tracer through image space.

All runs happen in a writable temp copy so the checked-in test_data folders are
never modified.

NOT A PERFECT MATCH
-------------------
The committed ground-truth correspondences (res_orig/rt_is.10003) contain only
4 particles -- the 5th (fast) tracer is missing from 3D space in frame 10003.
When the sequence step runs on the *clean* synthetic images it recovers all 5
in every frame (including 10003). So the reconstructed trajectory set may have
5 particles where res_orig has 4 in that one frame. The trajectory comparison
therefore checks that every position present in res_orig is recovered
(recall-based), rather than asserting exact per-frame particle counts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from openptv2.batch import pyptv_batch
from tests._support import find_test_data_root

TEST_DATA_ROOT = find_test_data_root(Path(__file__))
BURGERS_DIR = TEST_DATA_ROOT / "burgers"

# Frames generated/ground-truthed: 10001 .. 10005
FRAMES = list(range(10001, 10006))
NCAMS = 4

# Detection: recovered sub-pixel centroids must match ground truth within 0.1px
# (measured ~0.04px on the clean synthetic images).
DET_MAX_ERROR_PX = 0.1

# Trackers to exercise.
TRACKERS = ["default", "myptv_2d_tracking", "proptv_tracking"]

# Per-frame 3D position match radius (mm) for the trajectory comparison.
TRAJ_MATCH_RADIUS_MM = 0.1


def _require_burgers() -> Path:
    if not BURGERS_DIR.exists():
        pytest.skip(f"burgers dataset not found at {BURGERS_DIR}")
    img_orig = BURGERS_DIR / "img_orig"
    res_orig = BURGERS_DIR / "res_orig"
    if not img_orig.exists() or not res_orig.exists():
        pytest.skip("burgers img_orig/res_orig fixtures missing")
    return BURGERS_DIR


@pytest.fixture
def burgers_workdir(tmp_path):
    """Writable copy of the burgers dataset with synthetic images staged into
    img/ (ground-truth targets stay only in img_orig/)."""
    src = _require_burgers()
    dst = tmp_path / "burgers"
    # Don't copy the existing res/ (pipeline output) or a previous img/; we
    # re-derive both.
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("res", "img"))
    (dst / "img").mkdir()
    for img in (dst / "img_orig").glob("cam*.1*"):
        if "_targets" not in img.name:
            shutil.copy2(img, dst / "img" / img.name)
    return dst


def _read_targets(path: Path) -> dict[int, tuple[float, float]]:
    """Read a *_targets file -> {pnr: (x, y)} using only the centroid columns."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    out = {}
    for line in lines[1 : n + 1]:
        p = line.split()
        if len(p) < 8:
            continue
        out[int(p[0])] = (float(p[1]), float(p[2]))
    return out


def _read_detected_targets(
    workdir: Path, cam: int, frame: int
) -> dict[int, tuple[float, float]]:
    """Read detected targets for one camera/frame from the run's ASCII
    ``img/*_targets`` file, or -- for a store-backed run (no ``_targets``
    files are written, see ``tracking_frame_buf.write_targets``) -- from the
    ``res/run.zarr`` RunStore. ``cam`` is 1-based to match the ASCII naming.
    """
    ascii_path = workdir / "img" / f"cam{cam}.{frame}_targets"
    if ascii_path.exists():
        return _read_targets(ascii_path)

    from openptv2.storage import RunStore

    store = RunStore(workdir / "res" / "run.zarr", mode="r")
    tarr = store.read_targets(cam=cam - 1, frame=frame)
    return {t.pnr(): (t.x(), t.y()) for t in tarr}


def _read_ptv_is(path: Path) -> list[list[str]]:
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    return [line.split() for line in lines[1 : n + 1]]


def _read_rt_is(path: Path) -> list[np.ndarray]:
    """Read a res/rt_is.* file -> list of (x, y, z) 3D positions."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].strip())
    out = []
    for line in lines[1 : n + 1]:
        p = line.split()
        if len(p) < 4:
            continue
        out.append(np.array([float(p[1]), float(p[2]), float(p[3])]))
    return out


def _read_reconstructed_3d(workdir: Path, frame: int) -> list[np.ndarray]:
    """Read reconstructed 3D positions for one frame from ``res/rt_is.*``, or
    -- for a store-backed run -- from the ``res/run.zarr`` RunStore's
    correspondences (see ``_read_detected_targets``)."""
    ascii_path = workdir / "res" / f"rt_is.{frame}"
    if ascii_path.exists():
        return _read_rt_is(ascii_path)

    from openptv2.storage import RunStore

    store = RunStore(workdir / "res" / "run.zarr", mode="r")
    if not store.has_correspondences(frame):
        return []
    pos_3d, _ = store.read_correspondences(frame)
    return [row for row in pos_3d]


def _find_yaml(workdir: Path) -> Path:
    y = workdir / "parameters_Run1.yaml"
    if not y.exists():
        y = next(workdir.glob("parameters_*.yaml"))
    return y


def _run_batch(
    workdir: Path, *, mode: str, tracking_plugin=None, seq_first=None, seq_last=None
):
    seq_first = seq_first or FRAMES[0]
    seq_last = seq_last or FRAMES[-1]
    kw = dict(
        yaml_file=_find_yaml(workdir),
        seq_first=seq_first,
        seq_last=seq_last,
        mode=mode,
    )
    if tracking_plugin is not None:
        kw["tracking_plugin"] = tracking_plugin
    pyptv_batch.run_batch(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Detection round-trip (synthetic images -> detected targets)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
def test_burgers_detection_roundtrip(burgers_workdir):
    """Detection on the synthetic images must recover the ground-truth target
    sub-pixel centroids within DET_MAX_ERROR_PX."""
    _run_batch(burgers_workdir, mode="sequence")

    all_errors: list[float] = []
    n_matched = 0
    n_gt = 0
    for cam in range(1, NCAMS + 1):
        for frame in FRAMES:
            gt = _read_targets(
                burgers_workdir / "img_orig" / f"cam{cam}.{frame}_targets"
            )
            det = _read_detected_targets(burgers_workdir, cam, frame)

            # Ground truth sets the count; detection must recover every one.
            assert len(det) == len(gt), (
                f"cam{cam}.{frame}: detected {len(det)} != ground truth {len(gt)}"
            )

            gt_centers = np.array(list(gt.values()))
            det_centers = np.array(list(det.values()))
            n_gt += len(gt)
            for g in gt_centers:
                d = np.linalg.norm(det_centers - g, axis=1)
                best = float(np.min(d))
                all_errors.append(best)
                if best <= DET_MAX_ERROR_PX:
                    n_matched += 1

    max_err = float(np.max(all_errors))
    mean_err = float(np.mean(all_errors))

    print(f"\n  target count (gt) : {n_gt}")
    print(f"  matched           : {n_matched}/{n_gt}")
    print(f"  max centroid err  : {max_err:.4f} px (threshold {DET_MAX_ERROR_PX})")
    print(f"  mean centroid err : {mean_err:.4f} px")

    assert n_matched == n_gt, (
        f"detection failed to match {n_gt - n_matched}/{n_gt} "
        f"targets within {DET_MAX_ERROR_PX}px"
    )
    assert max_err <= DET_MAX_ERROR_PX, (
        f"max centroid error {max_err:.4f}px exceeds {DET_MAX_ERROR_PX}px"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: 3D trajectory round-trip vs committed res_orig ground truth
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
def test_burgers_3d_trajectory_vs_res_orig(burgers_workdir):
    """Reconstructed 3D positions must recover every position in the committed
    ground truth res_orig/rt_is.* (recall-based, nearest-neighbour matching).

    The synthetic images are rendered from the same projection that produced
    res_orig, so a correct pipeline reproduces the 3D particle cloud. We do NOT
    assert exact per-frame counts: the clean synthetic images recover all 5 in
    frame 10003 while res_orig has only 4 (see module docstring), so counts may
    legitimately differ there.
    """
    _run_batch(burgers_workdir, mode="sequence")

    all_errors: list[float] = []
    n_gt = 0
    n_matched = 0
    for frame in FRAMES:
        gt = _read_rt_is(burgers_workdir / "res_orig" / f"rt_is.{frame}")
        rec = _read_reconstructed_3d(burgers_workdir, frame)

        rec_arr = np.array(rec) if rec else np.empty((0, 3))
        n_gt += len(gt)
        frame_matched = 0
        for g in gt:
            if len(rec_arr):
                d = np.linalg.norm(rec_arr - g, axis=1)
                best = float(np.min(d))
                all_errors.append(best)
                if best <= TRAJ_MATCH_RADIUS_MM:
                    frame_matched += 1
                    n_matched += 1
            else:
                all_errors.append(float("inf"))

        print(f"  frame {frame}: rec={len(rec)} gt={len(gt)} matched={frame_matched}")

    max_err = float(np.max([e for e in all_errors if np.isfinite(e)]))
    mean_err = float(np.mean([e for e in all_errors if np.isfinite(e)]))

    print(f"\n  3D GT positions   : {n_gt}")
    print(f"  matched           : {n_matched}/{n_gt}")
    print(f"  max 3D err        : {max_err:.4f} mm (radius {TRAJ_MATCH_RADIUS_MM})")
    print(f"  mean 3D err       : {mean_err:.4f} mm")

    assert n_matched == n_gt, (
        f"failed to recover {n_gt - n_matched}/{n_gt} ground-truth "
        f"positions within {TRAJ_MATCH_RADIUS_MM}mm"
    )
    assert max_err <= TRAJ_MATCH_RADIUS_MM


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Image-space "add particle" across trackers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("tracker", TRACKERS)
def test_burgers_image_space_add_particle(burgers_workdir, tracker):
    """Exercise each tracker against the committed frame-10003 3D gap.

    res_orig/rt_is.10003 drops the 5th (fast) tracer from 3D while all 4
    cameras still see it in 2D (img_orig/*_targets has 5 every frame). We feed
    res_orig as the correspondence input and img/*_targets as the 2D input, so
    an image-space tracker that re-triangulates from 2D targets can recover the
    tracer *through* the gap in 3D.

    NOTE: bridging is not yet implemented in any tracker. This test asserts the
    trackers run end-to-end on the gapped input without error and reports how
    many 3D particles each recovers in frame 10003 (4 if the gap is not
    bridged, 5 if an image-space re-triangulation recovers it). That 5-vs-4
    distinction is the future acceptance criterion once bridging lands.
    """
    # Stage res_orig/rt_is.* as the correspondence input (the gapped one).
    dst = burgers_workdir
    (dst / "res").mkdir(exist_ok=True)
    for f in (dst / "res_orig").glob("rt_is.*"):
        shutil.copy2(f, dst / "res" / f.name)

    # Copy the synthetic images AND their 2D detection targets into img/.
    # (The _targets in img_orig are the ground-truth 2D detections.)
    for f in (dst / "img_orig").glob("cam*.1*"):
        shutil.copy2(f, dst / "img" / f.name)

    # Run the tracker (sequence is skipped: correspondences already exist).
    _run_batch(dst, mode="tracking", tracking_plugin=tracker)

    print(f"\n  tracker: {tracker}  (input: res_orig/rt_is gapped at frame 10003)")
    for frame in FRAMES:
        p = dst / "res" / f"rt_is.{frame}"
        n = int(p.read_text().splitlines()[0].strip())
        gt_n = int(
            (dst / "res_orig" / f"rt_is.{frame}").read_text().splitlines()[0].strip()
        )
        print(f"    rt_is.{frame}: rec={n} gt={gt_n}")
        assert n >= gt_n, (
            f"{tracker}: rt_is.{frame} has {n} particles, fewer than ground "
            f"truth {gt_n}"
        )


# TODO(image-space gap-bridging / "add particle"):
# res_orig/rt_is.10003 contains 4 particles while all 4 cameras still see the
# 5th tracer in 2D. A future feature should re-add that particle *through image
# space* (2D targets -> re-triangulation) so a single continuous 5-frame
# trajectory spans the gap. None of the current trackers do this yet: default
# and myptv_2d track 3D only, and proptv (unfinished) runs but loses the last
# step. Add a test here once bridging is implemented: feed res_orig/rt_is as
# correspondence input + img/*_targets (2D) and assert the fast (pnr=2) tracer
# yields a 5-frame trajectory (i.e. rt_is.10003 recovers 5, the fast particle
# bridges the gap).

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
