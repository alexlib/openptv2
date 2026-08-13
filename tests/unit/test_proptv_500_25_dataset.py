"""Integration unit test for running openptv2 ProPTVTracker on proPTV 500_25 synthetic dataset."""

import os
import numpy as np
import pytest

from openptv2.plugins.proptv import ProPTVConfig
from openptv2.plugins.proptv_tracking import ProPTVTracker

PROPTV_DATA_DIR = r"C:\Users\alex\Github\proPTV\data\500_25"


@pytest.mark.skipif(
    not os.path.exists(PROPTV_DATA_DIR),
    reason="proPTV 500_25 test dataset not found on disk"
)
def test_openptv2_proptv_500_25_accuracy():
    origin_files = [f"{PROPTV_DATA_DIR}/origin/origin_{str(t).zfill(5)}.txt" for t in range(5)]
    frame_particles = []
    for filepath in origin_files:
        assert os.path.exists(filepath), f"Missing {filepath}"
        data = np.loadtxt(filepath)
        frame_particles.append(data[:, 1:4])

    config = ProPTVConfig(
        t_init=3,
        maxvel=0.015,
        angle=60.0,
        dt=1.0,
        Vmin=[0.0, 0.0, 0.0],
        Vmax=[1.0, 1.0, 1.0],
    )
    tracker = ProPTVTracker(config)
    tracks = tracker.track_frames(frame_particles)

    # Reconstructed track count should match ground truth particles (500)
    assert len(tracks) >= 490, f"Expected ~500 tracks, got {len(tracks)}"

    # Measure matching accuracy against ground truth at t=0
    gt_p0 = frame_particles[0]
    rec_p0 = np.array([tr["pos"][0] for tr in tracks])
    matched = 0
    for p in rec_p0:
        if np.min(np.linalg.norm(gt_p0 - p, axis=1)) < 0.02:
            matched += 1

    pmp = (matched / len(gt_p0)) * 100.0
    assert pmp > 99.0, f"Expected PMP > 99%, got {pmp:.2f}%"
