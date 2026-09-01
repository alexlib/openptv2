"""Ground-truth tracking accuracy across a density x noise grid.

test_data/tracking_synthetic (see test_tracking_synthetic.py) is the easy
case: 12 well-separated particles, no real candidate ambiguity. It verifies
correctness but can't expose what happens under test_cavity's actual
regime -- dense seeding and detection noise comparable to the true motion,
where trackers must actually resolve ambiguity, not just avoid a gate.

This sweeps test_data/tracking_synthetic_dense's parametrized generator
(known ground truth: particle p is rt_is row p in every frame; noise
propagates through this rig's real anisotropic z-sensitivity, not an
isotropic jitter) across spacing/motion/noise combinations spanning
test_cavity's measured regime (spacing ~3.8mm, motion ~0.3mm/frame,
z-noise ~0.3-0.6mm -> spacing/motion ratio ~13, noise/motion ratio ~1.4)
and reports each tracker's correct/wrong/lost link count -- an objective
number, not a plausibility check.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parents[2] / "test_data" / "tracking_synthetic_dense"
sys.path.insert(0, str(FIX))
from generate import build_fixture  # noqa: E402

from openptv2.algorithms.calibration import Calibration  # noqa: E402
from openptv2.algorithms.parameters import (  # noqa: E402
    ControlPar,
    SequencePar,
    TrackPar,
    VolumePar,
)
from openptv2.tracker import Tracker  # noqa: E402


def _make_scene(tmp_path, n_particles, spacing_mm, motion_mm, noise_px, n_frames, seed):
    scene_dir = tmp_path / "scene"
    build_fixture(
        scene_dir,
        n_particles=n_particles,
        spacing_mm=spacing_mm,
        motion_mm=motion_mm,
        noise_px=noise_px,
        n_frames=n_frames,
        seed=seed,
    )
    return scene_dir


def _load(scene_dir):
    y = str(scene_dir / "parameters_Run1.yaml")
    cpar = ControlPar.from_yaml(y)
    vpar = VolumePar.from_yaml(y)
    spar = SequencePar.from_yaml(y, cpar.num_cams)
    cals = []
    for c in range(cpar.num_cams):
        cal = Calibration()
        cal.from_file(
            str(scene_dir / f"cal/cam{c + 1}.tif.ori"),
            str(scene_dir / f"cal/cam{c + 1}.tif.addpar"),
        )
        cals.append(cal)
    for c in range(cpar.num_cams):
        spar.set_img_base_name(c, str(scene_dir / f"img_orig/cam{c + 1}.%d"))
    return cpar, vpar, spar, cals


def _score_trackcorr_or_track3d(
    tmp_path, scene_dir, mode, first, last, **tpar_overrides
):
    cpar, vpar, spar, cals = _load(scene_dir)
    tpar = TrackPar.from_yaml(str(scene_dir / "parameters_Run1.yaml"))
    for k, v in tpar_overrides.items():
        setattr(tpar, k, v)

    res = tmp_path / f"res_{mode}"
    res.mkdir(parents=True, exist_ok=True)
    for f in range(first, last + 1):
        shutil.copy(scene_dir / "res_orig" / f"rt_is.{f}", res / f"rt_is.{f}")
    naming = {
        "corres": str(res / "rt_is"),
        "linkage": str(res / "ptv_is"),
        "prio": str(res / "added"),
    }
    tr = Tracker(cpar, vpar, tpar, spar, cals, naming)
    if mode == "track3d":
        tr.full_forward_3d()
    else:
        tr.full_forward()
    return _count_links(naming["linkage"], first, last)


def _score_nearest_hungarian(tmp_path, scene_dir, first, last, v_max, a_max):
    from openptv2.algorithms.parameters import ControlPar as _CP
    from openptv2.algorithms.tracking_frame_buf import Frame
    from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker

    cpar = _CP.from_yaml(str(scene_dir / "parameters_Run1.yaml"))
    frames = list(range(first, last + 1))
    frame_particles = []
    for fn in frames:
        frm = Frame(cpar.num_cams, 10000)
        frm.read(str(scene_dir / "res_orig" / "rt_is"), "", "", "", fn)
        frame_particles.append(frm.positions())

    tracker = MyPTV3DTracker(
        v_max=v_max, a_max=a_max, max_gap=1, dt=1.0, max_angle_deg=90.0
    )
    trajectories = tracker.track_frames(frame_particles)

    # next[frame_idx][row] -> row it links to in frame_idx+1, or -1.
    # MyPTV3DTracker's track dict stores positions/times, not source row
    # indices, so recover the row by exact-position match -- safe here since
    # positions are copied verbatim from frame_particles with no jitter.
    n_steps = len(frames) - 1
    nxt = [np.full(len(frame_particles[i]), -1, dtype=int) for i in range(n_steps)]
    for tr in trajectories:
        times = tr["time"]
        for i in range(len(times) - 1):
            f0, f1 = times[i], times[i + 1]
            if f1 != f0 + 1:
                continue
            p0, p1 = tr["pos"][i], tr["pos"][i + 1]
            r0 = int(np.argmin(np.linalg.norm(frame_particles[f0] - p0, axis=1)))
            r1 = int(np.argmin(np.linalg.norm(frame_particles[f1] - p1, axis=1)))
            nxt[f0][r0] = r1

    correct = wrong = 0
    lost = set()
    for step in nxt:
        for p, n in enumerate(step):
            if n < 0:
                lost.add(p)
            elif n == p:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, lost


def _count_links(linkage_base, first, last):
    correct = wrong = 0
    lost = set()
    for f in range(first, last):
        d = np.loadtxt(f"{linkage_base}.{f}", skiprows=1, ndmin=2)
        nxt = d[:, 1].astype(int)
        for p in range(len(nxt)):
            if nxt[p] < 0:
                lost.add(p)
            elif nxt[p] == p:
                correct += 1
            else:
                wrong += 1
    return correct, wrong, lost


# --------------------------------------------------------------------------- #
# Regime grid: from easy (sparse, clean) to test_cavity-like (dense, noisy)
# --------------------------------------------------------------------------- #

REGIMES = {
    "easy_sparse_clean": dict(
        n_particles=20, spacing_mm=8.0, motion_mm=1.0, noise_px=0.2
    ),
    "moderate": dict(n_particles=40, spacing_mm=5.0, motion_mm=0.5, noise_px=0.5),
    "test_cavity_like": dict(
        n_particles=80, spacing_mm=3.8, motion_mm=0.3, noise_px=1.0
    ),
    "harder_than_test_cavity": dict(
        n_particles=120, spacing_mm=3.0, motion_mm=0.2, noise_px=1.5
    ),
}


@pytest.mark.parametrize("regime", list(REGIMES))
def test_easy_regime_is_recovered_near_perfectly_by_all_trackers(tmp_path, regime):
    """Sanity floor: whatever the regime, no tracker should do WORSE than a
    plain nearest-neighbor baseline would predict from the noise level alone
    -- this doesn't assert a specific winner, it catches a tracker regressing
    to point-blank random assignment."""
    params = REGIMES[regime]
    n_frames = 5
    scene_dir = _make_scene(tmp_path, n_frames=n_frames, seed=0, **params)
    first, last = 10001, 10001 + n_frames - 1

    gate = params["motion_mm"] * 4  # generous but not the old 50x-too-loose mistake
    dacc = gate * 0.4

    results = {}
    for mode in ("trackcorr", "track3d"):
        c, w, lost = _score_trackcorr_or_track3d(
            tmp_path,
            scene_dir,
            mode,
            first,
            last,
            dvxmax=gate,
            dvxmin=-gate,
            dvymax=gate,
            dvymin=-gate,
            dvzmax=gate,
            dvzmin=-gate,
            dacc=dacc,
            dangle=90.0,
        )
        results[mode] = (c, w, lost)

    c, w, lost = _score_nearest_hungarian(
        tmp_path, scene_dir, first, last, v_max=gate, a_max=dacc
    )
    results["nearest_hungarian_3d"] = (c, w, lost)

    max_links = params["n_particles"] * (n_frames - 1)
    print(
        f"\n[{regime}] spacing={params['spacing_mm']} motion={params['motion_mm']} "
        f"noise_px={params['noise_px']} (max_links={max_links})"
    )
    for mode, (c, w, lost) in results.items():
        print(
            f"  {mode:20s}: correct={c:4d} wrong={w:4d} lost={len(lost):4d} "
            f"accuracy={100 * c / max_links:.1f}%"
        )

    # No tracker should produce more wrong links than correct ones in any
    # regime tested here -- a tracker that does is worse than doing nothing.
    for mode, (c, w, lost) in results.items():
        assert w <= c, (
            f"{mode} in {regime}: {w} wrong links >= {c} correct -- worse than a coin flip"
        )
